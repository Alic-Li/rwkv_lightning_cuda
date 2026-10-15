#include "rwkv/inference/rwkv_inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <list>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

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

constexpr int kForceReasoningMaskStartStep = 1;
constexpr int kForceReasoningMaskEndStep = 2;
constexpr int kForceReasoningDenyTokenA = 111;
constexpr int kForceReasoningDenyTokenB = 754;
constexpr float kMaskedLogit = -1.0e30f;

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

bool should_force_reasoning_mask(const GenerateOptions& options, int generated_step) {
  const int total_generated_step = options.force_reasoning_token_offset + generated_step;
  return options.force_reasoning &&
         total_generated_step >= kForceReasoningMaskStartStep &&
         total_generated_step <= kForceReasoningMaskEndStep;
}

class ScopedLogitMask {
 public:
  ScopedLogitMask(DeviceLogits& logits, const std::vector<int>& tokens) : logits_(logits) {
    if (logits_.rows <= 0 || logits_.vocab_size <= 0 || logits_.values.p == nullptr) {
      return;
    }
    for (int row = 0; row < logits_.rows; ++row) {
      for (int token : tokens) {
        if (token < 0 || token >= logits_.vocab_size) {
          continue;
        }
        const std::size_t index =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(logits_.vocab_size) +
            static_cast<std::size_t>(token);
        Entry entry;
        entry.index = index;
        rwkv7_fast_v4::check_cuda(
            cudaMemcpy(&entry.original, logits_.values.p + index, sizeof(float), cudaMemcpyDeviceToHost),
            "copy original masked logit");
        const float masked = kMaskedLogit;
        rwkv7_fast_v4::check_cuda(
            cudaMemcpy(logits_.values.p + index, &masked, sizeof(float), cudaMemcpyHostToDevice),
            "write masked logit");
        entries_.push_back(entry);
      }
    }
  }

  ScopedLogitMask(const ScopedLogitMask&) = delete;
  ScopedLogitMask& operator=(const ScopedLogitMask&) = delete;

  ~ScopedLogitMask() {
    for (const auto& entry : entries_) {
      rwkv7_fast_v4::check_cuda(
          cudaMemcpy(logits_.values.p + entry.index, &entry.original, sizeof(float), cudaMemcpyHostToDevice),
          "restore masked logit");
    }
  }

 private:
  struct Entry {
    std::size_t index = 0;
    float original = 0.0f;
  };

  DeviceLogits& logits_;
  std::vector<Entry> entries_;
};

std::vector<int> sample_batch_with_options_mask(
    DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options,
    int generated_step) {
  if (should_force_reasoning_mask(options, generated_step)) {
    ScopedLogitMask mask(logits, {kForceReasoningDenyTokenA, kForceReasoningDenyTokenB});
    return sample_batch_repetition_temperature_topk_topp(logits, penalties, options);
  }
  return sample_batch_repetition_temperature_topk_topp(logits, penalties, options);
}

int sample_with_options_mask(
    DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options,
    int generated_step) {
  if (logits.rows != 1) {
    throw std::runtime_error("sampler currently expects a single logits row");
  }
  return sample_batch_with_options_mask(logits, penalties, options, generated_step).front();
}

bool is_utf8_continuation(unsigned char byte) {
  return (byte & 0xC0u) == 0x80u;
}

int utf8_sequence_length(unsigned char lead) {
  if (lead <= 0x7Fu) {
    return 1;
  }
  if (lead >= 0xC2u && lead <= 0xDFu) {
    return 2;
  }
  if (lead >= 0xE0u && lead <= 0xEFu) {
    return 3;
  }
  if (lead >= 0xF0u && lead <= 0xF4u) {
    return 4;
  }
  return 0;
}

bool is_valid_utf8_sequence(const std::string& text, size_t pos, int len) {
  const auto b0 = static_cast<unsigned char>(text[pos]);
  switch (len) {
    case 1:
      return true;
    case 2:
      return is_utf8_continuation(static_cast<unsigned char>(text[pos + 1]));
    case 3: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2)) {
        return false;
      }
      if (b0 == 0xE0u && b1 < 0xA0u) {
        return false;
      }
      if (b0 == 0xEDu && b1 >= 0xA0u) {
        return false;
      }
      return true;
    }
    case 4: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      const auto b3 = static_cast<unsigned char>(text[pos + 3]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2) || !is_utf8_continuation(b3)) {
        return false;
      }
      if (b0 == 0xF0u && b1 < 0x90u) {
        return false;
      }
      if (b0 == 0xF4u && b1 > 0x8Fu) {
        return false;
      }
      return true;
    }
    default:
      return false;
  }
}

std::string take_complete_utf8(std::string& pending, bool flush_all) {
  std::string out;
  size_t i = 0;
  size_t last_copy = 0;
  while (i < pending.size()) {
    const auto lead = static_cast<unsigned char>(pending[i]);
    const int len = utf8_sequence_length(lead);
    if (len == 0) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    if (i + static_cast<size_t>(len) > pending.size()) {
      break;
    }
    if (!is_valid_utf8_sequence(pending, i, len)) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    i += static_cast<size_t>(len);
  }
  out.append(pending, last_copy, i - last_copy);
  pending.erase(0, i);
  if (flush_all && !pending.empty()) {
    pending.clear();
  }
  return out;
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

// Engine-local ownership isolates model lifetimes. Hashes only accelerate lookup;
// a hit must also match every token and both state/adapter identities.
struct InferenceEngine::PromptCache {
  struct Entry {
    std::vector<int64_t> tokens;
    uint64_t hash = 0;
    std::string state_id, adapter_id, initial_identity, adapter_identity;
    HostPromptState state;
    std::vector<float> logits;
  };
  explicit PromptCache(size_t limit) : capacity(limit) {}
  size_t capacity;
  std::mutex mutex;
  std::list<std::shared_ptr<Entry>> lru;

  static std::vector<uint64_t> hashes(const std::vector<int64_t>& tokens) {
    std::vector<uint64_t> out{14695981039346656037ULL};
    for (auto token : tokens) {
      out.push_back((out.back() ^ static_cast<uint64_t>(token)) * 1099511628211ULL);
    }
    return out;
  }
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
  for (size_t i = 0; i < count; ++i) {
    const auto hashes = PromptCache::hashes(prompt_ids[i]);
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto best = cache.lru.end();
    for (auto it = cache.lru.begin(); it != cache.lru.end(); ++it) {
      const auto& e = **it;
      const size_t n = e.tokens.size();
      if (n > offsets[i] && n <= prompt_ids[i].size() &&
          e.state_id == state_id && e.adapter_id == adapter_id && e.initial_identity == initial_identity &&
          e.adapter_identity == adapter_identity && e.state.wkv32 == state.wkv32 &&
          e.hash == hashes[n] &&
          std::equal(e.tokens.begin(), e.tokens.end(), prompt_ids[i].begin())) {
        best = it;
        offsets[i] = n;
      }
    }
    if (best != cache.lru.end()) {
      hits[i] = *best;
      cache.lru.splice(cache.lru.begin(), cache.lru, best);
      stats.cached_prompt_tokens += static_cast<int>(offsets[i]);
    }
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
    const auto hash = PromptCache::hashes(prompt_ids[row]).back();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto existing = std::find_if(cache.lru.begin(), cache.lru.end(), [&](const auto& e) {
      return e->hash == hash && e->tokens == prompt_ids[row] &&
          e->state_id == state_id && e->adapter_id == adapter_id && e->initial_identity == initial_identity &&
          e->adapter_identity == adapter_identity && e->state.wkv32 == state.wkv32;
    });
    if (existing != cache.lru.end()) {
      cache.lru.splice(cache.lru.begin(), cache.lru, existing);
      continue;
    }
    while (cache.lru.size() >= cache.capacity) cache.lru.pop_back();
    auto entry = std::make_shared<PromptCache::Entry>();
    entry->tokens = prompt_ids[row];
    entry->hash = hash;
    entry->state_id = state_id;
    entry->adapter_id = adapter_id;
    entry->initial_identity = initial_identity;
    entry->adapter_identity = adapter_identity;
    model_->copy_state_slice(state, static_cast<int>(row), staging, 0, 1);
    entry->state = download_prompt_state(staging);
    entry->logits.resize(logits.vocab_size);
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(entry->logits.data(), logits.values.p + row * logits.vocab_size,
                   entry->logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download prompt cache logits");
    cache.lru.push_front(std::move(entry));
  }
}

InferenceEngine::InferenceEngine(
    std::shared_ptr<IModelBackend> model,
    std::shared_ptr<TrieTokenizer> tokenizer,
    std::string model_name,
    int prefill_chunk_size,
    size_t prompt_cache_capacity)
    : prompt_cache_(std::make_shared<PromptCache>(prompt_cache_capacity)),
      model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      model_name_(std::move(model_name)),
      prefill_chunk_size_(prefill_chunk_size) {
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
      model_->forward_prefill(chunk_tokens, state, logits);
      initialized_logits = true;
    } else {
      auto sub_state = model_->create_state(active_count);
      model_->copy_state_slice(state, 0, sub_state, 0, active_count);
      DeviceLogits sub_logits;
      model_->forward_prefill(chunk_tokens, sub_state, sub_logits);
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
  AdapterMetrics adapter_metrics(state);
  GenerationStats stats;
  stats.finish_reasons = {"length"};
  DeviceLogits logits;
  stats.prompt_tokens = prefill_prompt(prompt, state, logits);

  std::vector<int> token_buffer;
  std::string utf8_pending;
  std::string result;

  auto penalties = make_sampler_penalties(model_->vocab_size());
  for (int step = 0; step < options.max_tokens; ++step) {
    if (should_stop && should_stop()) {
      mark_interrupted(stats);
      break;
    }
    const int token = sample_with_options_mask(logits, penalties, options, step);
    adapter_metrics.token();
    if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
      stats.stop_token = true;
      stats.finish_reasons[0] = "stop";
      break;
    }

    ++stats.generated_tokens;
    token_buffer.push_back(token);
    result += tokenizer_->decode(token);

    if (emit != nullptr && chunk_size > 0 && token_buffer.size() >= static_cast<size_t>(chunk_size)) {
      utf8_pending += tokenizer_->decode(token_buffer);
      token_buffer.clear();
      const std::string chunk = take_complete_utf8(utf8_pending, false);
      if (!chunk.empty() && !(*emit)(stream_index, chunk)) {
        mark_interrupted(stats);
        if (out_stats) *out_stats = stats;
        return result;
      }
    }

    model_->forward_decode(std::vector<int64_t>{token}, state, logits);
  }

  if (emit != nullptr) {
    if (!token_buffer.empty()) {
      utf8_pending += tokenizer_->decode(token_buffer);
      token_buffer.clear();
    }
    const std::string tail = take_complete_utf8(utf8_pending, true);
    if (!tail.empty()) {
      if (!(*emit)(stream_index, tail)) mark_interrupted(stats);
    }
    // Empty chunks are per-row stop events; length is reported in final stats.
    if (stats.finish_reasons[0] == "stop" && !(*emit)(stream_index, {})) {
      mark_interrupted(stats);
    }
  }
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
  bind_adapter(state, options.adapter);
  AdapterMetrics adapter_metrics(state);
  if (prompts.empty()) {
    if (out_stats) *out_stats = {};
    return {};
  }
  GenerationStats stats;
  stats.finish_reasons.assign(prompts.size(), "length");
  const int batch_size = static_cast<int>(prompts.size());
  if (state.batch_size != batch_size) {
    throw std::runtime_error("generation state batch size mismatch");
  }

  std::vector<size_t> sorted_to_original;
  const auto sorted_prompt_ids = encode_prompts_sorted(prompts, sorted_to_original);

  for (const auto& ids : sorted_prompt_ids) stats.prompt_tokens += static_cast<int>(ids.size());
  DeviceLogits logits;
  prefill_batch_chunked(sorted_prompt_ids, state, logits);

  const int fallback_token = options.stop_tokens.empty() ? 0 : static_cast<int>(options.stop_tokens.front());
  auto penalties = make_sampler_penalties(model_->vocab_size(), batch_size);
  std::vector<std::string> sorted_outputs(prompts.size());
  std::vector<bool> finished(prompts.size(), false);
  std::vector<int64_t> decode_batch(prompts.size(), fallback_token);
  int active = batch_size;

  for (int step = 0; step < options.max_tokens && active > 0; ++step) {
    const auto sampled_tokens = sample_batch_with_options_mask(logits, penalties, options, step);
    adapter_metrics.token();
    for (size_t i = 0; i < sampled_tokens.size(); ++i) {
      if (finished[i]) {
        decode_batch[i] = fallback_token;
        continue;
      }

      const int token = sampled_tokens[i];
      if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
        finished[i] = true;
        stats.stop_token = true;
        stats.finish_reasons[sorted_to_original[i]] = "stop";
        decode_batch[i] = fallback_token;
        --active;
        continue;
      }

      ++stats.generated_tokens;
      decode_batch[i] = token;
      sorted_outputs[i] += tokenizer_->decode(token);
    }

    if (active == 0) {
      break;
    }
    model_->forward_decode(decode_batch, state, logits);
  }

  std::vector<std::string> outputs(prompts.size());
  for (size_t i = 0; i < sorted_outputs.size(); ++i) {
    outputs[sorted_to_original[i]] = std::move(sorted_outputs[i]);
  }
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
    const std::optional<std::string>& prompt_cache_state_id) const {
  bind_adapter(state, options.adapter);
  AdapterMetrics adapter_metrics(state);
  GenerationStats stats;
  if (prompts.empty()) {
    return stats;
  }
  stats.finish_reasons.assign(prompts.size(), "length");
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

  const int fallback_token = options.stop_tokens.empty() ? 0 : static_cast<int>(options.stop_tokens.front());
  auto penalties = make_sampler_penalties(model_->vocab_size(), batch_size);
  std::vector<bool> finished(prompts.size(), false);
  std::vector<int64_t> decode_batch(prompts.size(), fallback_token);
  std::vector<std::vector<int>> token_buffers(prompts.size());
  std::vector<std::string> utf8_pending(prompts.size());
  int active = batch_size;
  const auto decode_begin = std::chrono::steady_clock::now();

  for (int step = 0; step < options.max_tokens && active > 0; ++step) {
    if (should_stop && should_stop()) {
      mark_interrupted(stats);
      break;
    }
    const auto sampled_tokens = sample_batch_with_options_mask(logits, penalties, options, step);
    adapter_metrics.token();
    for (size_t i = 0; i < sampled_tokens.size(); ++i) {
      if (finished[i]) {
        decode_batch[i] = fallback_token;
        continue;
      }

      const int token = sampled_tokens[i];
      if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
        finished[i] = true;
        stats.stop_token = true;
        stats.finish_reasons[sorted_to_original[i]] = "stop";
        decode_batch[i] = fallback_token;
        --active;
        if (!token_buffers[i].empty()) {
          utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
          token_buffers[i].clear();
        }
        const std::string tail = take_complete_utf8(utf8_pending[i], true);
        // Flush text before the empty-chunk stop event, without waiting for other rows.
        if ((!tail.empty() && !emit(static_cast<int>(sorted_to_original[i]), tail)) ||
            !emit(static_cast<int>(sorted_to_original[i]), {})) {
          mark_interrupted(stats);
          stats.decode_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin).count();
          return stats;
        }
        continue;
      }

      ++stats.generated_tokens;
      decode_batch[i] = token;
      token_buffers[i].push_back(token);

      if (chunk_size > 0 && token_buffers[i].size() >= static_cast<size_t>(chunk_size)) {
        utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
        token_buffers[i].clear();
        const std::string chunk = take_complete_utf8(utf8_pending[i], false);
        if (!chunk.empty() && !emit(static_cast<int>(sorted_to_original[i]), chunk)) {
          mark_interrupted(stats);
          stats.decode_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin).count();
          return stats;
        }
      }
    }

    if (active == 0) {
      break;
    }
    model_->forward_decode(decode_batch, state, logits);
  }

  for (size_t i = 0; i < prompts.size(); ++i) {
    if (!token_buffers[i].empty()) {
      utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
      token_buffers[i].clear();
    }
    const std::string tail = take_complete_utf8(utf8_pending[i], true);
    if (!tail.empty() && !emit(static_cast<int>(sorted_to_original[i]), tail)) {
      mark_interrupted(stats);
      stats.decode_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin).count();
      return stats;
    }
  }
  stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin).count();
  return stats;
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
  AdapterMetrics adapter_metrics(state);
  if (state.batch_size != 1 || logits.rows != 1) {
    throw std::runtime_error("logit continuation only supports single prompt batch");
  }

  GenerationStats stats;
  stats.finish_reasons = {"length"};
  std::vector<int> token_buffer;
  std::string utf8_pending;
  auto penalties = make_sampler_penalties(model_->vocab_size());
  const auto decode_begin = std::chrono::steady_clock::now();

  for (int step = 0; step < options.max_tokens; ++step) {
    if (should_stop && should_stop()) {
      mark_interrupted(stats);
      break;
    }

    const int token = sample_with_options_mask(logits, penalties, options, step);
    adapter_metrics.token();
    if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
      stats.stop_token = true;
      stats.finish_reasons[0] = "stop";
      break;
    }

    ++stats.generated_tokens;
    token_buffer.push_back(token);

    if (chunk_size > 0 && token_buffer.size() >= static_cast<size_t>(chunk_size)) {
      utf8_pending += tokenizer_->decode(token_buffer);
      token_buffer.clear();
      const std::string chunk = take_complete_utf8(utf8_pending, false);
      if (!chunk.empty() && !emit(0, chunk)) {
        mark_interrupted(stats);
        break;
      }
    }

    model_->forward_decode(std::vector<int64_t>{token}, state, logits);
  }

  if (!token_buffer.empty()) {
    utf8_pending += tokenizer_->decode(token_buffer);
    token_buffer.clear();
  }
  const std::string tail = take_complete_utf8(utf8_pending, true);
  if (!tail.empty() && !emit(0, tail)) {
    mark_interrupted(stats);
  }
  // Same per-row stop event as the prompt-based streaming paths.
  if (stats.finish_reasons[0] == "stop" && !emit(0, {})) {
    mark_interrupted(stats);
  }

  const auto decode_end = std::chrono::steady_clock::now();
  stats.decode_seconds = std::chrono::duration<double>(decode_end - decode_begin).count();
  return stats;
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
