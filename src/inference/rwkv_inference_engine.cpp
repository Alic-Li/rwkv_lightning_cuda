#include "rwkv/inference/rwkv_inference_engine.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>
#include <unordered_set>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <list>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include "rwkv/inference/text_stop_matcher.hpp"
#include "rwkv/inference/prefix_cache.hpp"
#include "rwkv/inference/utf8.hpp"

namespace rwkv7_server {
namespace {

// One JSON record per adapter request. TTFT includes admission/H2D/prefill;
// TPOT measures sample-to-sample wall time. VRAM is sampled device usage,
// including other processes, not an allocator-level per-request peak.
struct AdapterMetrics {
  GenerationState &state;
  int tokens = 0;
  std::chrono::steady_clock::time_point first{}, last{};
  explicit AdapterMetrics(GenerationState &s) : state(s) {}
  void token() {
    if (!state.adapter)
      return;
    last = std::chrono::steady_clock::now();
    if (!tokens)
      first = last;
    ++tokens;
  }
  ~AdapterMetrics() {
    if (!state.adapter || !tokens)
      return;
    Json::Value m;
    m["event"] = "miss_request_metrics";
    m["adapter_version"] = state.adapter->package->version;
    m["gpu_cache"] = state.adapter_cold ? "cold" : "hot";
    m["h2d_ms"] = state.adapter_h2d_ms;
    m["ttft_ms"] =
        std::chrono::duration<double, std::milli>(first - state.request_started)
            .count();
    m["tpot_ms"] =
        tokens > 1
            ? std::chrono::duration<double, std::milli>(last - first).count() /
                  (tokens - 1)
            : 0;
    m["sample_steps"] = tokens;
    m["observed_device_vram_peak_bytes"] =
        Json::UInt64(state.observed_device_vram_peak);
    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::clog << Json::writeString(w, m) << '\n';
  }
};

void mark_interrupted(InferenceEngine::GenerationStats& stats) {
  stats.stopped = true;
  for (auto& reason : stats.finish_reasons) {
    if (reason == "length") reason = "stop";
  }
}

struct ThinkPromptConfig {
  const char* header = "<think></think";
  const char* user_msg_footer = "";
  bool force_reasoning = false;
};

ThinkPromptConfig think_prompt_config(ThinkType think_type) {
  switch (think_type) {
    case ThinkType::None:
      return {"", "", false};
    case ThinkType::Fast:
      return {"<think></think", "", false};
    case ThinkType::Free:
      return {"<think", "", true};
    case ThinkType::PreferChinese:
      return {"<think>\xE5\x97\xAF", "", true};
    case ThinkType::En:
      return {"<think", " (think)", true};
    case ThinkType::EnShort:
      return {"<think", " (think a bit)", true};
    case ThinkType::EnLong:
      return {"<think", " (think a lot)", true};
  }
  return {"<think></think", "", false};
}


}  // namespace

struct HostPromptState {
  // CPU RAM State Cache，Not save to adapter_gpu。
  std::shared_ptr<const rwkv7_miss::AdapterHandle> adapter;
  std::string effective_key;
  std::string adapter_identity;
  std::string initial_state_identity;
  bool advanced = false;
  bool wkv32 = false;

  std::vector<uint16_t> shift;
  std::vector<uint16_t> wkv16;
  std::vector<float> wkv32_data;
  std::vector<int> elapsed;
};

namespace {

template <typename Host, typename Device>
std::vector<Host> download_prompt_buffer(
    const rwkv7_fast_v4::DeviceBuffer<Device>& buffer) {
  static_assert(sizeof(Host) == sizeof(Device), "prompt cache element size mismatch");
  std::vector<Host> host(buffer.n);
  if (!host.empty()) {
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(host.data(), buffer.p, host.size() * sizeof(Host), cudaMemcpyDeviceToHost),
        "download prompt cache state");
  }
  return host;
}

template <typename Host, typename Device>
void upload_prompt_buffer(
    const std::vector<Host>& host,
    rwkv7_fast_v4::DeviceBuffer<Device>& buffer) {
  static_assert(sizeof(Host) == sizeof(Device), "prompt cache element size mismatch");
  buffer.resize(host.size(), "allocate prompt cache staging state");
  if (!host.empty()) {
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(Host), cudaMemcpyHostToDevice),
        "upload prompt cache state");
  }
}

HostPromptState download_prompt_state(const GenerationState& gpu) {
  if (gpu.batch_size != 1) {
    throw std::runtime_error("prompt cache snapshot requires a single-row state");
  }
  HostPromptState host;
  host.adapter = gpu.adapter;
  host.effective_key = gpu.effective_key;
  host.adapter_identity = gpu.adapter_identity;
  host.initial_state_identity = gpu.initial_state_identity;
  host.advanced = gpu.advanced;
  host.wkv32 = gpu.wkv32;
  host.shift = download_prompt_buffer<uint16_t>(gpu.shift);
  if (gpu.wkv32) {
    host.wkv32_data = download_prompt_buffer<float>(gpu.wkv_state32);
  } else {
    host.wkv16 = download_prompt_buffer<uint16_t>(gpu.wkv_state);
  }
  host.elapsed = download_prompt_buffer<int>(gpu.elapsed);
  return host;
}

void upload_prompt_state(const HostPromptState& host, GenerationState& gpu) {
  if (gpu.batch_size != 1 || gpu.wkv32 != host.wkv32) {
    throw std::runtime_error("prompt cache staging state shape/precision mismatch");
  }
  gpu.adapter_gpu.reset();
  gpu.adapter = host.adapter;
  gpu.effective_key = host.effective_key;
  gpu.adapter_identity = host.adapter_identity;
  gpu.initial_state_identity = host.initial_state_identity;
  gpu.advanced = host.advanced;
  upload_prompt_buffer(host.shift, gpu.shift);
  if (host.wkv32) {
    upload_prompt_buffer(host.wkv32_data, gpu.wkv_state32);
  } else {
    upload_prompt_buffer(host.wkv16, gpu.wkv_state);
  }
  upload_prompt_buffer(host.elapsed, gpu.elapsed);
}

}  // namespace

// Shared radix index; existing CPU state/logits snapshots retain their format.
struct InferenceEngine::PromptCache {
  struct Entry { HostPromptState state; std::vector<float> logits; };
  explicit PromptCache(size_t limit) : capacity(limit), index(limit) {}
  size_t capacity;
  std::mutex mutex;
  PrefixCache<Entry> index;
};

void InferenceEngine::prefill_cached(
    std::vector<std::vector<int64_t>>& prompt_ids,
    std::vector<size_t>& sorted_to_original,
    GenerationState& state, DeviceLogits& logits,
    const std::string& state_id, const std::string& adapter_id,
    GenerationStats& stats) const {
  if (state.advanced) {
    throw std::runtime_error("prompt cache requires an initial request state");
  }
  const auto initial_identity = state.initial_state_identity;
  const auto adapter_identity = state.adapter_identity;
  auto& cache = *prompt_cache_;
  const size_t count = prompt_ids.size();
  std::vector<std::shared_ptr<PromptCache::Entry>> hits(count);
  std::vector<size_t> offsets(count, 0), order(count);
  // Length-prefixed fields avoid ambiguous namespace concatenations.
  std::string identity;
  for (const auto &part : {state_id, adapter_id, initial_identity, adapter_identity,
                           std::string(state.wkv32 ? "fp32" : "fp16")})
    identity += std::to_string(part.size()) + ":" + part;
  for (size_t i = 0; i < count; ++i) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto hit = cache.index.find(identity, prompt_ids[i]);
    hits[i] = hit.value;
    offsets[i] = hit.length;
    stats.cached_prompt_tokens += static_cast<int>(hit.length);
  }
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return prompt_ids[a].size() - offsets[a] > prompt_ids[b].size() - offsets[b];
  });
  // Full hits occupy the inactive tail and supply their cached terminal logits.
  logits.rows = static_cast<int>(count);
  logits.vocab_size = model_->vocab_size();
  logits.values.resize(count * logits.vocab_size, "prompt cache batch logits");
  std::vector<std::vector<int64_t>> remaining, reordered;
  std::vector<size_t> original;
  // One temporary GPU row serves all transfers; cached entries own CPU memory only.
  auto staging = model_->create_state(1);
  for (size_t row = 0; row < count; ++row) {
    const auto i = order[row];
    if (hits[i]) {
      upload_prompt_state(hits[i]->state, staging);
      model_->copy_state_slice(staging, 0, state, static_cast<int>(row), 1);
      if (offsets[i] == prompt_ids[i].size()) {
        if (hits[i]->logits.size() != static_cast<size_t>(logits.vocab_size)) {
          throw std::runtime_error("prompt cache logits shape mismatch");
        }
        rwkv7_fast_v4::check_cuda(
            cudaMemcpy(logits.values.p + row * logits.vocab_size,
                       hits[i]->logits.data(), hits[i]->logits.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            "upload prompt cache logits");
      }
    }
    remaining.emplace_back(prompt_ids[i].begin() + offsets[i], prompt_ids[i].end());
    reordered.push_back(std::move(prompt_ids[i]));
    original.push_back(sorted_to_original[i]);
  }
  hits.clear();
  prompt_ids = std::move(reordered);
  sorted_to_original = std::move(original);
  prefill_batch_chunked(remaining, state, logits);

  // Snapshot before any sampling/decode can mutate the prompt state.
  for (size_t row = 0; row < count; ++row) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.index.find(identity, prompt_ids[row]).length == prompt_ids[row].size())
      continue;
    auto entry = std::make_shared<PromptCache::Entry>();
    model_->copy_state_slice(state, static_cast<int>(row), staging, 0, 1);
    entry->state = download_prompt_state(staging);
    entry->logits.resize(logits.vocab_size);
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(entry->logits.data(), logits.values.p + row * logits.vocab_size,
                   entry->logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download prompt cache logits");
    cache.index.put(identity, prompt_ids[row], std::move(entry));
  }
}

// A single GPU owner per engine (one loaded model per server/device). HTTP
// threads only enqueue work and deliver events; no network callback runs here.
struct InferenceEngine::Scheduler {
  struct Event {
    int row;
    std::string text;
    std::optional<GenerationStats> stop_stats;
  };
  struct Request {
    GenerationState* state;
    DeviceLogits* logits;
    GenerateOptions options;
    std::unique_ptr<TextStopMatcher> text_stops;
    ControlCallback should_stop;
    std::unique_ptr<AdapterMetrics> metrics;
    int chunk_size;
    bool notify_text_stop = false;
    GenerationStats stats;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Event> events;
    std::atomic<bool> cancelled{false};
    bool done = false;
    std::exception_ptr error;
    int remaining = 0;
  };
  struct Row {
    std::shared_ptr<Request> request;
    int index = 0, step = 0, token = 0;
    bool finished = false, completed = false;
    std::string pending, stop_tail;
    int buffered_tokens = 0;
    std::unique_ptr<SamplerPenaltyState> saved_sampler;
  };
  struct Group {
    std::vector<std::shared_ptr<Row>> rows, incoming;
    GenerationState state, spare_state;
    size_t shift_per_row = 0, wkv16_per_row = 0, wkv32_per_row = 0, elapsed_per_row = 0;
    DeviceLogits logits;
    SamplerPenaltyState sampler;
    std::vector<GenerateOptions> options;
    std::string adapter;
    bool wkv32 = false;
  };

  std::shared_ptr<IModelBackend> model;
  std::shared_ptr<TrieTokenizer> tokenizer;
  int device = 0;
  int max_rows;
  std::mutex mutex;
  std::condition_variable changed;
  bool stopping = false;
  std::deque<std::function<void()>> prefill;
  std::deque<std::shared_ptr<Request>> incoming;
  std::vector<std::unique_ptr<Group>> groups;
  std::deque<std::shared_ptr<Row>> pending_rows;
  std::shared_ptr<void> prefill_workspace, decode_workspace;
  std::thread worker;

  Scheduler(std::shared_ptr<IModelBackend> m, std::shared_ptr<TrieTokenizer> t, int limit)
      : model(std::move(m)), tokenizer(std::move(t)), max_rows(limit) {
    rwkv7_fast_v4::check_cuda(cudaGetDevice(&device), "get scheduler device");
    worker = std::thread([this] { loop(); });
  }
  ~Scheduler() {
    { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
    changed.notify_all();
    worker.join();
  }

  void execute_prefill(std::function<void()> fn) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
    auto result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopping) throw std::runtime_error("scheduler is shutting down");
      prefill.push_back([task] { (*task)(); });
    }
    changed.notify_one();
    result.get();
  }

  void forward_prefill(const std::vector<std::vector<int64_t>>& tokens,
                       GenerationState& state, DeviceLogits& logits) {
    execute_prefill([&] {
      logits.backend_workspace = std::move(prefill_workspace);
      try { model->forward_prefill(tokens, state, logits); }
      catch (...) { prefill_workspace = std::move(logits.backend_workspace); throw; }
      prefill_workspace = std::move(logits.backend_workspace);
    });
  }

  GenerationStats generate(GenerationState& state, DeviceLogits& logits,
      const GenerateOptions& options, int chunk_size, const StreamCallback& emit,
      const ControlCallback& should_stop, const StatsCallback& on_row_stop) {
    if (state.batch_size != logits.rows || logits.rows <= 0)
      throw std::invalid_argument("continuous decode state/logits shape mismatch");
    // Validate before admitting the request so one bad request cannot fail its peers.
    if (!(options.temperature >= 0.001 && options.temperature <= 1000))
      throw std::invalid_argument("temperature must be in [0.001, 1000]");
    auto req = std::make_shared<Request>();
    req->state = &state;
    req->logits = &logits;
    req->options = options;
    req->text_stops = std::make_unique<TextStopMatcher>(options.stop.value_or(std::vector<std::string>{}));
    req->should_stop = should_stop;
    req->metrics = std::make_unique<AdapterMetrics>(state);
    req->chunk_size = chunk_size;
    req->notify_text_stop = bool(on_row_stop);
    req->remaining = logits.rows;
    req->stats.finish_reasons.assign(logits.rows, "length");
    req->stats.matched_stops.resize(logits.rows);
    std::exception_ptr callback_error;
    bool accepting = true;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopping) throw std::runtime_error("scheduler is shutting down");
      incoming.push_back(req);
    }
    changed.notify_one();
    // Keep state/logits alive even if a callback throws. Cancellation is observed
    // by the owner before the next step, then all resident rows are exported.
    for (;;) {
      std::deque<Event> events;
      bool done;
      {
        std::unique_lock<std::mutex> lock(req->mutex);
        req->changed.wait_for(lock, std::chrono::milliseconds(10), [&] {
          return req->done || !req->events.empty();
        });
        events.swap(req->events);
        done = req->done;
      }
      changed.notify_one();
      try {
        if (!callback_error) {
          for (const auto& event : events) {
            if (event.stop_stats && on_row_stop) on_row_stop(*event.stop_stats);
            if (accepting && emit && !emit(event.row, event.text)) {
              accepting = false; req->cancelled = true;
            }
          }
        }
      } catch (...) { callback_error = std::current_exception(); req->cancelled = true; }
      if (done) break;
    }
    if (callback_error) std::rethrow_exception(callback_error);
    if (req->error) std::rethrow_exception(req->error);
    if (req->cancelled) mark_interrupted(req->stats);
    return req->stats;
  }

  void event(Row& row, std::string text, bool stop = false) {
    auto& req = *row.request;
    std::lock_guard<std::mutex> lock(req.mutex);
    req.events.push_back({row.index, std::move(text),
                         stop ? std::optional<GenerationStats>(req.stats) : std::nullopt});
    req.changed.notify_one();
  }
  void flush(Row& row, bool final) {
    auto text = take_complete_utf8(row.pending, final);
    row.buffered_tokens = 0;
    if (!text.empty()) event(row, std::move(text));
  }

  // A packed execution state may contain rows initialized from different saved
  // states. Identity remains attached to each request, never to a physical slot.
  // Backend slice copying intentionally checks identities for public callers;
  // only this internal packing operation bypasses that whole-batch check.
  void pack(const GenerationState& src, int from, GenerationState& dst, int to) {
    dst.advanced = false;
    model->copy_state_slice(src, from, dst, to, 1);
  }
  void export_row(Group& group, int slot) {
    auto& row = *group.rows[slot];
    auto& dst = *row.request->state;
    const auto key = dst.effective_key;
    const auto initial = dst.initial_state_identity;
    const auto identity = dst.adapter_identity;
    const auto adapter = dst.adapter;
    pack(group.state, slot, dst, row.index);
    dst.effective_key = key;
    dst.initial_state_identity = initial;
    dst.adapter_identity = identity;
    dst.adapter = adapter;
    // Preserve backend diagnostics for adapter metrics.
    dst.adapter_gpu = group.state.adapter_gpu;
    dst.observed_device_vram_peak = std::max(dst.observed_device_vram_peak,
                                            group.state.observed_device_vram_peak);
    model->copy_logits_slice(group.logits, slot, *row.request->logits, row.index, 1);
    if (dst.batch_size == 1) dst.text_stop_tail = row.stop_tail;
  }
  void complete(Row& row) {
    if (row.completed) return;
    row.completed = true;
    auto& req = *row.request;
    if (req.cancelled) mark_interrupted(req.stats);
    flush(row, true);
    if (req.stats.finish_reasons[row.index] == "stop" && !req.cancelled &&
        (req.notify_text_stop || req.stats.matched_stops[row.index].empty()))
      event(row, {}, true);
    if (--req.remaining == 0) {
      req.metrics.reset();
      req.stats.decode_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - req.start).count();
      std::lock_guard<std::mutex> lock(req.mutex);
      req.done = true;
      req.changed.notify_one();
    }
  }

  bool backpressured(Request& req) {
    // The mutex also protects the event queue from the HTTP consumer.
    std::lock_guard<std::mutex> lock(req.mutex);
    return req.events.size() >= 128;
  }

  void rebuild(Group& group, bool allow_pause = false) {
    std::vector<std::shared_ptr<Row>> live;
    std::vector<int> old_slots;
    for (size_t i = 0; i < group.rows.size(); ++i) {
      auto& row = *group.rows[i];
      if (row.completed) continue;
      if (row.request->cancelled) row.finished = true;
      if (row.finished) {
        export_row(group, int(i));
        complete(row);
      } else if (allow_pause && backpressured(*row.request)) {
        export_row(group, int(i));
        row.saved_sampler = std::make_unique<SamplerPenaltyState>(
            make_sampler_penalties(model->vocab_size()));
        copy_sampler_row(group.sampler, int(i), *row.saved_sampler, 0);
        pending_rows.push_back(group.rows[i]);
      } else {
        live.push_back(group.rows[i]);
        old_slots.push_back(int(i));
      }
    }
    for (auto& row : group.incoming) {
      if (row->request->cancelled || row->request->options.max_tokens <= 0) {
        complete(*row);
      } else {
        live.push_back(row);
        old_slots.push_back(-1);
      }
    }
    if (live == group.rows) { group.incoming.clear(); return; }
    if (live.empty()) {
      group.incoming.clear();
      group.rows.clear();
      group.state = {};
      group.logits = {};
      group.sampler = {};
      return;
    }
    GenerationState state;
    if (group.state.batch_size == 0) {
      state = model->create_state(int(live.size()));
      group.shift_per_row = state.shift.n / live.size();
      group.wkv16_per_row = state.wkv_state.n / live.size();
      group.wkv32_per_row = state.wkv_state32.n / live.size();
      group.elapsed_per_row = state.elapsed.n / live.size();
    } else {
      state = std::move(group.spare_state);
      state.batch_size = int(live.size());
      state.wkv32 = group.wkv32;
      state.shift.resize(group.shift_per_row * live.size(), "grow shift slots");
      state.wkv_state.resize(group.wkv16_per_row * live.size(), "grow WKV16 slots");
      state.wkv_state32.resize(group.wkv32_per_row * live.size(), "grow WKV32 slots");
      state.elapsed.resize(group.elapsed_per_row * live.size(), "grow elapsed slots");
    }
    DeviceLogits logits;
    logits.rows = int(live.size());
    logits.vocab_size = model->vocab_size();
    logits.values.resize(live.size() * logits.vocab_size, "allocate decode slot logits");
    auto sampler = make_sampler_penalties(logits.vocab_size, logits.rows);
    for (size_t i = 0; i < live.size(); ++i) {
      auto& row = *live[i];
      if (old_slots[i] >= 0) {
        pack(group.state, old_slots[i], state, int(i));
        model->copy_logits_slice(group.logits, old_slots[i], logits, int(i), 1);
        copy_sampler_row(group.sampler, old_slots[i], sampler, int(i));
      } else {
        pack(*row.request->state, row.index, state, int(i));
        model->copy_logits_slice(*row.request->logits, row.index, logits, int(i), 1);
        if (row.saved_sampler) {
          copy_sampler_row(*row.saved_sampler, 0, sampler, int(i));
          row.saved_sampler.reset();
        }
      }
    }
    // Reuse the backend's forward scratch as membership changes.
    logits.backend_workspace = std::move(group.logits.backend_workspace);
    group.spare_state = std::move(group.state);
    group.state = std::move(state);
    group.logits = std::move(logits);
    group.sampler = std::move(sampler);
    group.rows = std::move(live);
    group.incoming.clear();
    group.options.clear();
    for (const auto& row : group.rows) group.options.push_back(row->request->options);
  }

  void step(Group& group) {
    rebuild(group, true);
    if (group.rows.empty()) return;
    std::vector<int> steps;
    for (const auto& row : group.rows) {
      steps.push_back(row->step);
    }
    const auto tokens = sample_rows(group.logits, group.sampler, group.options, steps);
    std::unordered_set<Request*> sampled;
    for (const auto& row : group.rows)
      if (sampled.insert(row->request.get()).second) row->request->metrics->token();
    for (size_t i = 0; i < group.rows.size(); ++i) {
      auto& row = *group.rows[i];
      auto& req = *row.request;
      const auto& o = req.options;
      row.token = tokens[i];
      if (!o.stop && std::find(o.stop_tokens.begin(), o.stop_tokens.end(), row.token) != o.stop_tokens.end()) {
        req.stats.stop_token = true;
        req.stats.finish_reasons[row.index] = "stop";
        row.finished = true; // EOS is not fed back into the reusable state.
        continue;
      }
      ++row.step;
      ++req.stats.generated_tokens;
      const auto decoded = tokenizer->decode(row.token);
      row.pending += decoded;
      ++row.buffered_tokens;
      if (o.stop) {
        auto matched = req.text_stops->append(row.stop_tail, decoded);
        if (!matched.empty()) {
          req.stats.matched_stops[row.index] = std::move(matched);
          req.stats.finish_reasons[row.index] = "stop";
          row.stop_tail.clear();
        }
      }
      if (req.chunk_size > 0 && row.buffered_tokens >= req.chunk_size) flush(row, false);
    }
    // Remove EOS/cancelled rows before forwarding. Length/text-stop rows must
    // consume their last visible token before exporting a resumable state.
    rebuild(group);
    if (group.rows.empty()) return;
    std::vector<int64_t> decode;
    for (const auto& row : group.rows) decode.push_back(row->token);
    group.logits.backend_workspace = decode_workspace;
    model->forward_decode(decode, group.state, group.logits);
    decode_workspace = group.logits.backend_workspace;
    for (auto& row : group.rows) {
      auto& req = *row->request;
      if (row->step >= req.options.max_tokens ||
          req.stats.finish_reasons[row->index] == "stop") row->finished = true;
    }
    rebuild(group);
  }

  void admit(const std::shared_ptr<Request>& req) {
    for (int i = 0; i < req->state->batch_size; ++i) {
      auto row = std::make_shared<Row>();
      row->request = req;
      row->index = i;
      if (req->state->batch_size == 1) row->stop_tail = req->state->text_stop_tail;
      pending_rows.push_back(std::move(row));
    }
  }
  void admit_pending() {
    size_t occupied = 0;
    for (const auto& group : groups) occupied += group->rows.size() + group->incoming.size();
    for (auto it = pending_rows.begin(); it != pending_rows.end();) {
      auto row = *it;
      const auto& req = row->request;
      if (req->cancelled || req->options.max_tokens <= 0) {
        complete(*row);
        it = pending_rows.erase(it);
        continue;
      }
      if (occupied >= size_t(max_rows) || backpressured(*req)) { ++it; continue; }
      Group* target = nullptr;
      for (auto& group : groups) {
        if (group->adapter == req->state->adapter_identity && group->wkv32 == req->state->wkv32) {
          target = group.get(); break;
        }
      }
      if (!target) {
        groups.push_back(std::make_unique<Group>());
        target = groups.back().get();
        target->adapter = req->state->adapter_identity;
        target->wkv32 = req->state->wkv32;
      }
      target->incoming.push_back(row);
      ++occupied;
      it = pending_rows.erase(it);
    }
  }
  void fail(Group& group, std::exception_ptr error) {
    auto rows = std::move(group.rows);
    rows.insert(rows.end(), group.incoming.begin(), group.incoming.end());
    group.incoming.clear();
    for (auto& row : rows) {
      if (row->completed) continue;
      row->request->error = error;
      row->request->cancelled = true;
      complete(*row);
    }
    group.state = {};
    group.logits = {};
    group.sampler = {};
  }
  void loop() {
    // Device selection is thread-local on CUDA and HIP.
    const auto device_error = cudaSetDevice(device);
    for (;;) {
      std::deque<std::shared_ptr<Request>> arrivals;
      std::function<void()> chunk;
      bool shutdown;
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (groups.empty() && incoming.empty() && prefill.empty() && !stopping) {
          if (pending_rows.empty())
            changed.wait(lock, [&] { return stopping || !incoming.empty() || !prefill.empty(); });
          else
            changed.wait_for(lock, std::chrono::milliseconds(10));
        }
        arrivals.swap(incoming);
        if (!prefill.empty()) { chunk = std::move(prefill.front()); prefill.pop_front(); }
        shutdown = stopping;
      }
      for (auto& req : arrivals) admit(req);
      if (shutdown) for (auto& row : pending_rows) row->request->cancelled = true;
      admit_pending();
      // At most one prefill chunk between decode sweeps. New long prompts
      // therefore cannot monopolize the owner while existing rows decode.
      if (chunk) chunk();
      std::unordered_set<Request*> checked;
      {
        const auto check = [&](const auto& rows) {
          for (auto& row : rows) {
            auto& req = *row->request;
            if (!row->completed && checked.insert(&req).second && req.should_stop && !req.cancelled) {
              try { if (req.should_stop()) req.cancelled = true; }
              catch (...) { req.error = std::current_exception(); req.cancelled = true; }
            }
          }
        };
        for (auto& group : groups) {
          check(group->rows);
          check(group->incoming);
        }
        check(pending_rows);
      }
      for (auto& group : groups) {
        try {
          rwkv7_fast_v4::check_cuda(device_error, "select scheduler device");
          if (shutdown) {
            for (auto& row : group->rows) row->request->cancelled = true;
            for (auto& row : group->incoming) row->request->cancelled = true;
          }
          step(*group);
        } catch (...) { fail(*group, std::current_exception()); }
      }
      groups.erase(std::remove_if(groups.begin(), groups.end(), [](const auto& group) {
        return group->rows.empty() && group->incoming.empty();
      }), groups.end());
      if (shutdown) {
        std::lock_guard<std::mutex> lock(mutex);
        if (groups.empty() && pending_rows.empty() && incoming.empty() && prefill.empty()) break;
      }
    }
  }
};

InferenceEngine::Scheduler& InferenceEngine::scheduler() const {
  std::lock_guard<std::mutex> lock(scheduler_mutex_);
  if (!scheduler_) scheduler_ = std::make_shared<Scheduler>(model_, tokenizer_, max_decode_batch_size_);
  return *scheduler_;
}

InferenceEngine::InferenceEngine(
    std::shared_ptr<IModelBackend> model,
    std::shared_ptr<TrieTokenizer> tokenizer,
    std::string model_name,
    int prefill_chunk_size,
    size_t prompt_cache_capacity,
    int max_decode_batch_size)
    : max_decode_batch_size_(max_decode_batch_size),
      prompt_cache_(std::make_shared<PromptCache>(prompt_cache_capacity)),
      model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      model_name_(std::move(model_name)),
      prefill_chunk_size_(prefill_chunk_size) {
  if (max_decode_batch_size_ <= 0) throw std::invalid_argument("decode batch size must be positive");
  if (prefill_chunk_size_ <= 0) {
    throw std::runtime_error("prefill chunk size must be positive");
  }
}

std::vector<int64_t> InferenceEngine::encode_prompt(const std::string& prompt) const {
  auto ids = tokenizer_->encode(prompt);
  std::vector<int64_t> out(ids.begin(), ids.end());
  if (out.empty()) {
    throw std::runtime_error("prompt is empty after tokenization");
  }
  return out;
}

std::vector<std::vector<int64_t>> InferenceEngine::encode_prompts_sorted(
    const std::vector<std::string>& prompts,
    std::vector<size_t>& sorted_to_original) const {
  std::vector<std::vector<int64_t>> encoded_prompts;
  encoded_prompts.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded_prompts.push_back(encode_prompt(prompt));
  }

  sorted_to_original.resize(prompts.size());
  std::iota(sorted_to_original.begin(), sorted_to_original.end(), 0);
  std::stable_sort(
      sorted_to_original.begin(),
      sorted_to_original.end(),
      [&](size_t lhs, size_t rhs) { return encoded_prompts[lhs].size() > encoded_prompts[rhs].size(); });

  std::vector<std::vector<int64_t>> sorted_prompt_ids;
  sorted_prompt_ids.reserve(prompts.size());
  for (size_t prompt_index : sorted_to_original) {
    sorted_prompt_ids.push_back(std::move(encoded_prompts[prompt_index]));
  }
  return sorted_prompt_ids;
}

void InferenceEngine::prefill_batch_chunked(
    const std::vector<std::vector<int64_t>>& sorted_prompt_ids,
    GenerationState& state,
    DeviceLogits& logits) const {
  const int batch_size = static_cast<int>(sorted_prompt_ids.size());
  if (batch_size <= 0) {
    throw std::runtime_error("prefill batch must not be empty");
  }
  if (state.batch_size != batch_size) {
    throw std::runtime_error("prefill state batch size mismatch");
  }

  std::vector<int> lengths;
  lengths.reserve(sorted_prompt_ids.size());
  std::vector<int> pos(sorted_prompt_ids.size(), 0);
  for (const auto& prompt_ids : sorted_prompt_ids) {
    lengths.push_back(static_cast<int>(prompt_ids.size()));
  }

  bool initialized_logits = logits.rows == batch_size;
  while (true) {
    int active_count = 0;
    while (active_count < batch_size && pos[static_cast<size_t>(active_count)] < lengths[static_cast<size_t>(active_count)]) {
      ++active_count;
    }
    if (active_count == 0) {
      break;
    }

    int step = lengths[0] - pos[0];
    for (int i = 1; i < active_count; ++i) {
      step = std::min(step, lengths[static_cast<size_t>(i)] - pos[static_cast<size_t>(i)]);
    }
    step = std::min(step, prefill_chunk_size_);

    std::vector<std::vector<int64_t>> chunk_tokens;
    chunk_tokens.reserve(static_cast<size_t>(active_count));
    for (int i = 0; i < active_count; ++i) {
      const auto& prompt_ids = sorted_prompt_ids[static_cast<size_t>(i)];
      const int begin = pos[static_cast<size_t>(i)];
      chunk_tokens.emplace_back(
          prompt_ids.begin() + begin,
          prompt_ids.begin() + begin + step);
    }

    if (active_count == batch_size) {
      scheduler().forward_prefill(chunk_tokens, state, logits);
      initialized_logits = true;
    } else {
      auto sub_state = model_->create_state(active_count);
      model_->copy_state_slice(state, 0, sub_state, 0, active_count);
      DeviceLogits sub_logits;
      scheduler().forward_prefill(chunk_tokens, sub_state, sub_logits);
      model_->copy_state_slice(sub_state, 0, state, 0, active_count);
      if (!initialized_logits) {
        throw std::runtime_error("batched prefill logits were not initialized");
      }
      model_->copy_logits_slice(sub_logits, 0, logits, 0, active_count);
    }

    for (int i = 0; i < active_count; ++i) {
      pos[static_cast<size_t>(i)] += step;
    }
  }
}

std::string InferenceEngine::generate_one(
    const std::string& prompt,
    GenerationState& state,
    const GenerateOptions& options,
    const StreamCallback* emit,
    int stream_index,
    int chunk_size,
    const ControlCallback& should_stop,
    GenerationStats* out_stats) const {
  bind_adapter(state, options.adapter);
  GenerationStats stats;
  stats.finish_reasons = {"length"};
  stats.matched_stops.resize(1);
  DeviceLogits logits;
  stats.prompt_tokens = prefill_prompt(prompt, state, logits);

  std::string result;
  const auto decode = scheduler().generate(state, logits, options, chunk_size,
      [&](int, const std::string& text) {
        result += text;
        return !emit || (*emit)(stream_index, text);
      }, should_stop, {});
  const int prompt_tokens = stats.prompt_tokens;
  stats = decode;
  stats.prompt_tokens = prompt_tokens;
  if (out_stats) *out_stats = std::move(stats);
  return result;
}

std::vector<std::string> InferenceEngine::batch_generate(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    GenerationStats* out_stats) const {
  if (prompts.empty()) {
    if (out_stats) *out_stats = {};
    return {};
  }

  auto state = model_->create_state(static_cast<int>(prompts.size()));
  return batch_generate_with_state(prompts, state, options, out_stats);
}

std::vector<std::string> InferenceEngine::batch_generate_with_state(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options,
    GenerationStats* out_stats) const {
  std::vector<std::string> outputs(prompts.size());
  auto stats = batch_generate_stream_with_state(prompts, state, options, 32,
      [&](int row, const std::string& text) { outputs.at(row) += text; return true; });
  if (out_stats) *out_stats = std::move(stats);
  return outputs;
}

std::vector<std::string> InferenceEngine::batch_generate_state(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options,
    GenerationStats* out_stats) const {
  if (prompts.size() != 1 || state.batch_size != 1) {
    throw std::runtime_error("stateful generation only supports single prompt batch");
  }
  return {generate_one(prompts.front(), state, options, nullptr, 0, 0, {}, out_stats)};
}

InferenceEngine::GenerationStats InferenceEngine::batch_generate_stream(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit,
    const ControlCallback& should_stop,
    const StatsCallback& on_prefill_complete) const {
  GenerationStats stats;
  if (prompts.empty()) {
    return stats;
  }

  auto state = model_->create_state(static_cast<int>(prompts.size()));
  return batch_generate_stream_with_state(
      prompts, state, options, chunk_size, emit, should_stop, on_prefill_complete);
}

InferenceEngine::GenerationStats InferenceEngine::batch_generate_stream_with_state(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit,
    const ControlCallback& should_stop,
    const StatsCallback& on_prefill_complete,
    const std::optional<std::string>& prompt_cache_state_id,
    const StatsCallback& on_row_stop) const {
  bind_adapter(state, options.adapter);
  GenerationStats stats;
  if (prompts.empty()) {
    return stats;
  }
  stats.finish_reasons.assign(prompts.size(), "length");
  stats.matched_stops.resize(prompts.size());
  const int batch_size = static_cast<int>(prompts.size());
  if (state.batch_size != batch_size) {
    throw std::runtime_error("streaming generation state batch size mismatch");
  }

  std::vector<size_t> sorted_to_original;
  auto sorted_prompt_ids = encode_prompts_sorted(prompts, sorted_to_original);
  for (const auto& prompt_ids : sorted_prompt_ids) {
    stats.prompt_tokens += static_cast<int>(prompt_ids.size());
  }

  DeviceLogits logits;
  const auto prefill_begin = std::chrono::steady_clock::now();
  if (prompt_cache_state_id && prompt_cache_->capacity > 0) {
    prefill_cached(sorted_prompt_ids, sorted_to_original, state, logits,
                   *prompt_cache_state_id, options.adapter_id, stats);
  } else {
    prefill_batch_chunked(sorted_prompt_ids, state, logits);
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  stats.prefill_seconds = std::chrono::duration<double>(prefill_end - prefill_begin).count();
  if (on_prefill_complete) {
    on_prefill_complete(stats);
  }

  const auto remap = [&](GenerationStats decoded) {
    const auto reasons = decoded.finish_reasons;
    const auto stops = decoded.matched_stops;
    for (size_t i = 0; i < sorted_to_original.size(); ++i) {
      decoded.finish_reasons[sorted_to_original[i]] = reasons[i];
      decoded.matched_stops[sorted_to_original[i]] = stops[i];
    }
    decoded.prompt_tokens = stats.prompt_tokens;
    decoded.cached_prompt_tokens = stats.cached_prompt_tokens;
    decoded.prefill_seconds = stats.prefill_seconds;
    return decoded;
  };
  return remap(scheduler().generate(state, logits, options, chunk_size,
      [&](int row, const std::string& text) { return emit(int(sorted_to_original[row]), text); },
      should_stop, [&](const GenerationStats& decoded) {
        if (on_row_stop) on_row_stop(remap(decoded));
      }));
}

InferenceEngine::GenerationStats InferenceEngine::batch_generate_state_stream(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit,
    const ControlCallback& should_stop) const {
  if (prompts.size() != 1 || state.batch_size != 1) {
    throw std::runtime_error("stateful generation only supports single prompt batch");
  }
  GenerationStats stats;
  generate_one(prompts.front(), state, options, &emit, 0, chunk_size, should_stop, &stats);
  return stats;
}

int InferenceEngine::prefill_prompt(
    const std::string& prompt,
    GenerationState& state,
    DeviceLogits& logits) const {
  state.text_stop_tail.clear();
  const auto prompt_ids = encode_prompt(prompt);
  std::vector<std::vector<int64_t>> prefill_batch{prompt_ids};
  prefill_batch_chunked(prefill_batch, state, logits);
  return static_cast<int>(prompt_ids.size());
}

InferenceEngine::GenerationStats InferenceEngine::generate_from_logits_stream(
    GenerationState& state,
    DeviceLogits& logits,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit,
    const ControlCallback& should_stop) const {
  bind_adapter(state, options.adapter);
  if (state.batch_size != 1 || logits.rows != 1) {
    throw std::runtime_error("logit continuation only supports single prompt batch");
  }

  return scheduler().generate(state, logits, options, chunk_size, emit, should_stop, {});
}

std::string InferenceEngine::format_openai_prompt(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& messages,
    ThinkType think_type) const {
  const auto think_config = think_prompt_config(think_type);
  size_t footer_index = messages.size();
  if (think_config.user_msg_footer[0] != '\0') {
    for (size_t i = messages.size(); i > 0; --i) {
      if (messages[i - 1].first == "User") {
        footer_index = i - 1;
        break;
      }
    }
  }

  std::ostringstream oss;
  bool has_prefix = false;
  if (!system.empty()) {
    oss << "System: " << system;
    has_prefix = true;
  }
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& [role, content] = messages[i];
    if (has_prefix) {
      oss << "\n\n";
    }
    oss << role << ": " << content;
    if (i == footer_index) {
      oss << think_config.user_msg_footer;
    }
    has_prefix = true;
  }
  if (has_prefix) {
    oss << "\n\n";
  }
  oss << "Assistant: ";
  oss << think_config.header;
  return oss.str();
}

std::string InferenceEngine::format_openai_prompt(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool enable_think) const {
  return format_openai_prompt(system, messages, enable_think ? ThinkType::Free : ThinkType::Fast);
}

int InferenceEngine::count_tokens(const std::string& text) const {
  return static_cast<int>(tokenizer_->encode(text).size());
}

}  // namespace rwkv7_server
