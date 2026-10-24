#include "server.hpp"
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include "rwkv/inference/prefix_cache.hpp"
#include "rwkv/inference/rwkv_tokenizer.hpp"
#include "rwkv/inference/text_stop_matcher.hpp"
#include "rwkv/inference/utf8.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/version.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <drogon/drogon.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace rwkv_online_rl {
namespace {
using namespace rwkv_bf16_training;
using Clock = std::chrono::steady_clock;
struct HttpError : std::runtime_error {
  int status;
  HttpError(int code, const std::string &s)
      : std::runtime_error(s), status(code) {}
};
void require(bool value, const std::string &s) {
  if (!value)
    throw HttpError(400, s);
}
int number(const Json::Value &v, const char *key, int fallback, int low,
           int high) {
  if (!v.isMember(key))
    return fallback;
  require(v[key].isInt(), std::string(key) + " must be an integer");
  int n = v[key].asInt();
  require(n >= low && n <= high, std::string(key) + " out of range");
  return n;
}
float real(const Json::Value &v, const char *key, float fallback, float low,
           float high) {
  if (!v.isMember(key))
    return fallback;
  require(v[key].isNumeric(), std::string(key) + " must be numeric");
  float f = v[key].asFloat();
  require(std::isfinite(f) && f >= low && f <= high,
          std::string(key) + " out of range");
  return f;
}
bool flag(const Json::Value &v, const char *key, bool fallback) {
  if (!v.isMember(key))
    return fallback;
  require(v[key].isBool(), std::string(key) + " must be boolean");
  return v[key].asBool();
}
std::vector<std::string> text_stops(const Json::Value &value) {
  Json::Value list = value;
  if (value.isString()) {
    list = Json::Value(Json::arrayValue);
    list.append(value);
  }
  require(list.isArray() && list.size() <= 16,
          "stop must be a string or an array of at most 16 strings");
  std::vector<std::string> result;
  for (const auto &s : list) {
    require(s.isString() && !s.asString().empty() &&
                s.asString().size() <= 256,
            "each stop string must be 1..256 bytes");
    result.push_back(s.asString());
  }
  return result;
}
struct Options {
  std::string model, adapter, state, resume, host = "127.0.0.1",
                                             output = "online-rl-output";
  std::string vocab = RWKV_STATE_TUNE_DEFAULT_VOCAB, targets = "all";
  std::string gemm_mode = "bf16";
  std::vector<std::string> stop = {"</tool_call>", "</answer>", "\n\nUser:"};
  int port = 8848, chunk = 64, ctx = 4096, batch = 32, cache = 16, versions = 2,
      rank = 16, max_tokens = 1024;
  float alpha = 16, lr = 2e-5f;
  bool rank_set = false, alpha_set = false, targets_set = false;
};
Options parse(int argc, char **argv, bool miss) {
  Options o;
  if (!miss) {
    o.port = 8849;
    o.lr = 0.01f;
  }
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      std::cout
          << "Usage: " << argv[0]
          << " --model MODEL.pth [--adapter ADAPTER.pth] [--state STATE.pth]\n"
             "  --host 127.0.0.1 --port "
          << o.port
          << " --chunk 64 --ctx 4096 --max-batch 32\n"
             "  --cache-capacity 16 --keep-versions 2 --lr "
          << o.lr
          << " --output online-rl-output\n"
             "  --gemm-mode bf16|strict (default: bf16 Tensor Cores)\n"
             "  --max-tokens 1024 (generation safety cap, not a target length)\n"
             "  --stop JSON (default: [\"</tool_call>\",\"</answer>\",\"\\n\\nUser:\"])\n"
             "  --vocab PATH --rank 16 --alpha 16 --targets all (inferred from "
             "cold adapter)\n"
             "  --version, -v prints the version and exits\n";
      std::exit(0);
    }
    if (rwkv::is_version_flag(key))
      std::exit(rwkv::print_version(miss ? "rwkv_miss_tune_server"
                                         : "rwkv_state_tune_server"));
    if (++i >= argc)
      throw std::invalid_argument("missing value for " + key);
    std::string v = argv[i];
    auto integer = [&] {
      size_t used = 0;
      int n = std::stoi(v, &used);
      if (used != v.size())
        throw std::invalid_argument(key);
      return n;
    };
    auto floating = [&] {
      size_t used = 0;
      float n = std::stof(v, &used);
      if (used != v.size() || !std::isfinite(n))
        throw std::invalid_argument(key);
      return n;
    };
    if (key == "--model")
      o.model = v;
    else if (key == "--adapter")
      o.adapter = v;
    else if (key == "--state")
      o.state = v;
    else if (key == "--resume")
      o.resume = v;
    else if (key == "--vocab")
      o.vocab = v;
    else if (key == "--host")
      o.host = v;
    else if (key == "--output")
      o.output = v;
    else if (key == "--gemm-mode")
      o.gemm_mode = v;
    else if (key == "--max-tokens")
      o.max_tokens = integer();
    else if (key == "--stop") {
      Json::Value stops;
      std::istringstream input(v);
      if (!(input >> stops))
        throw std::invalid_argument("--stop requires JSON strings");
      o.stop = text_stops(stops);
    }
    else if (key == "--port")
      o.port = integer();
    else if (key == "--chunk")
      o.chunk = integer();
    else if (key == "--ctx")
      o.ctx = integer();
    else if (key == "--max-batch")
      o.batch = integer();
    else if (key == "--cache-capacity")
      o.cache = integer();
    else if (key == "--keep-versions")
      o.versions = integer();
    else if (key == "--lr")
      o.lr = floating();
    else if (key == "--rank") {
      o.rank = integer();
      o.rank_set = true;
    } else if (key == "--alpha") {
      o.alpha = floating();
      o.alpha_set = true;
    } else if (key == "--targets") {
      o.targets = v;
      o.targets_set = true;
    } else
      throw std::invalid_argument("unknown option: " + key);
  }
  if (o.model.empty() || o.port < 1 || o.port > 65535 || o.chunk < 1 ||
      o.ctx < 2 || o.chunk > o.ctx || o.chunk > 65535 || o.ctx > 1048576 ||
      o.batch < 1 || o.batch > 1024 || o.cache < 0 || o.versions < 1 ||
      o.versions > 32 || o.lr <= 0 || o.rank < 1 || o.rank > 1024 ||
      (o.gemm_mode != "bf16" && o.gemm_mode != "strict") ||
      o.max_tokens < 1 || o.max_tokens > 1048576)
    throw std::invalid_argument(
        "invalid model/port/chunk/ctx/batch/cache/versions/lr/rank/gemm-mode/max-tokens");
  if (!o.alpha_set)
    o.alpha = float(o.rank);
  return o;
}
struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    gpu_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
              "RL stream");
  }
  ~Stream() { cudaStreamDestroy(value); }
};
template <class T>
std::vector<T> download(const DeviceBuffer<T> &d, cudaStream_t s) {
  std::vector<T> h(d.n);
  gpu_check(cudaMemcpyAsync(h.data(), d.p, d.n * sizeof(T),
                            cudaMemcpyDeviceToHost, s),
            "download snapshot");
  gpu_check(cudaStreamSynchronize(s), "snapshot ready");
  return h;
}
template <class T>
void upload(DeviceBuffer<T> &d, const std::vector<T> &h, cudaStream_t s) {
  d.resize(h.size(), "upload snapshot");
  gpu_check(cudaMemcpyAsync(d.p, h.data(), h.size() * sizeof(T),
                            cudaMemcpyHostToDevice, s),
            "upload snapshot");
  gpu_check(cudaStreamSynchronize(s), "snapshot uploaded");
}
struct CachedState {
  std::vector<float> state, logits;
  std::vector<bf16> shifts;
};
struct Policy {
  std::string version, directory;
  uint64_t step = 0;
  FrozenModelView view;
  DeviceBuffer<float> initial;
  DeviceBuffer<bf16> adapter;
  rwkv7_server::PrefixCache<CachedState> cache;
  explicit Policy(size_t capacity) : cache(capacity) {}
};
struct Sample {
  std::vector<int> tokens;
  std::vector<float> weights;
  float temperature;
};
// Stop conditions of one completions request. A row stops on the token that
// completes a match; that token stays in tokens/logprobs and the cache.
struct StopRules {
  std::set<int> ids;
  std::vector<std::vector<int>> sequences;
  std::vector<std::string> strings;
};
// One prompt of a completions request, decoded in its own BatchDecoder row.
struct Rollout {
  std::vector<int> prompt, generated;
  std::vector<int64_t> prefix;
  std::vector<float> logprobs;
  std::string stop_tail; // same bounded byte suffix as the inference engine
  std::mt19937_64 rng;
  size_t cached = 0;
  int limit = 0;
  bool active = true;
  std::string finish = "length";
  Json::Value matched;
  int next = 0;
  float next_logprob = 0;
};
std::pair<int, float> sample(const float *logits, int vocab, float temperature,
                             std::mt19937_64 &rng) {
  // Ordinary softmax on the CPU: no sampler kernel, filters or penalties.
  const float maximum = *std::max_element(logits, logits + vocab);
  std::vector<float> weights(vocab);
  double sum = 0;
  for (int i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i]))
      throw std::runtime_error("nonfinite rollout logits");
    weights[i] = std::exp((logits[i] - maximum) / temperature);
    sum += weights[i];
  }
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  int id = distribution(rng);
  float logprob =
      float((double(logits[id]) - maximum) / temperature - std::log(sum));
  return {id, logprob};
}
// Runs f(0..n-1) on up to hardware_concurrency threads; rethrows the first
// exception after all threads finish. f must only touch its own index.
template <class F> void parallel_for(size_t n, const F &f) {
  const size_t workers =
      std::min<size_t>(n, std::max(1u, std::thread::hardware_concurrency()));
  std::atomic<size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto work = [&] {
    for (size_t i; (i = next++) < n;) {
      try {
        f(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error)
          error = std::current_exception();
      }
    }
  };
  std::vector<std::thread> pool;
  for (size_t w = 1; w < workers; ++w)
    pool.emplace_back(work);
  work();
  for (auto &t : pool)
    t.join();
  if (error)
    std::rethrow_exception(error);
}

class Runtime {
  Options o_;
  bool miss_;
  // Exactly one Model owns all backbone allocations. Views only borrow them.
  Model model_;
  Stream stream_;
  FrozenModelView learner_;
  DeviceBuffer<float> initial_, gradient_, adam_m_, adam_v_;
  DeviceBuffer<bf16> fixed_adapter_;
  std::shared_ptr<const rwkv7_miss::Package> fixed_package_;
  std::unique_ptr<rwkv_bf16_miss::Trainer> trainer_;
  std::unique_ptr<Executor> actor_exec_, learner_exec_;
  // Batched decode rows of the current completions request size; rebuilt only
  // when a request has a different number of prompts.
  FrozenModelView actor_view_;
  std::unique_ptr<BatchDecoder> decoder_;
  rwkv7_server::TrieTokenizer tokenizer_;
  std::deque<std::shared_ptr<Policy>> policies_;
  // Startup policy; never retired, so clients can score against it (KL).
  std::shared_ptr<Policy> reference_;
  std::string run_id_ = rwkv7_miss::new_runtime_identity();
  uint64_t step_ = 0, version_ = 0;

  void copy(void *to, const void *from, size_t n) {
    gpu_check(
        cudaMemcpyAsync(to, from, n, cudaMemcpyDeviceToDevice, stream_.value),
        "copy policy");
  }
  void sync() {
    gpu_check(cudaStreamSynchronize(stream_.value), "RL synchronize");
  }
  std::vector<int> tokens(const Json::Value &v, size_t max) const {
    require(v.isArray() && !v.empty() && v.size() <= max,
            "tokens must be a nonempty bounded array");
    std::vector<int> result;
    result.reserve(v.size());
    for (const auto &t : v) {
      require(t.isInt() && t.asInt() >= 0 && t.asInt() < learner_.vocab,
              "token out of vocabulary");
      result.push_back(t.asInt());
    }
    return result;
  }
  std::shared_ptr<Policy> policy(const Json::Value &r) const {
    if (!r.isMember("policy_version"))
      return policies_.back();
    require(r["policy_version"].isString(), "policy_version must be a string");
    for (const auto &p : policies_)
      if (p->version == r["policy_version"].asString())
        return p;
    if (reference_->version == r["policy_version"].asString())
      return reference_;
    throw HttpError(409, "policy version is unknown or retired");
  }
  std::shared_ptr<Policy> snapshot() {
    auto p = std::make_shared<Policy>(o_.cache);
    p->version = run_id_ + ":" + std::to_string(version_);
    p->step = step_;
    p->view = learner_;
    p->initial.resize(initial_.n, "actor initial state");
    copy(p->initial.p, initial_.p, initial_.n * 4);
    if (trainer_)
      p->adapter = trainer_->snapshot(p->view);
    sync();
    return p;
  }
  // row < 0 caches the actor executor's state, otherwise that decoder row.
  void cache_put(Policy &p, const std::vector<int64_t> &prefix,
                 std::vector<float> logits, int row = -1) {
    if (!o_.cache)
      return;
    auto c = std::make_shared<CachedState>();
    if (row < 0) {
      c->state = download(actor_exec_->state(), stream_.value);
      c->shifts = download(actor_exec_->shifts(), stream_.value);
    } else
      decoder_->download(row, c->state, c->shifts);
    c->logits = std::move(logits);
    p.cache.put(p.version, prefix, std::move(c));
  }
  StopRules stop_rules(const Json::Value &r) const {
    StopRules rules;
    const bool numeric = r.isMember("stop_token_ids") ||
                         r.isMember("stop_token_sequences");
    require(!r.isMember("stop") || !numeric,
            "stop and token stop rules are mutually exclusive");
    if (r.isMember("stop_token_ids")) {
      require(r["stop_token_ids"].isArray(), "stop_token_ids must be an array");
      for (const auto &id : r["stop_token_ids"]) {
        require(id.isInt() && id.asInt() >= 0 && id.asInt() < learner_.vocab,
                "invalid stop token");
        rules.ids.insert(id.asInt());
      }
    }
    if (r.isMember("stop_token_sequences")) {
      const auto &list = r["stop_token_sequences"];
      require(list.isArray() && list.size() <= 16,
              "stop_token_sequences must be an array of at most 16 sequences");
      for (const auto &sequence : list) {
        require(sequence.isArray() && !sequence.empty() && sequence.size() <= 64,
                "each stop token sequence must hold 1..64 tokens");
        rules.sequences.push_back(tokens(sequence, 64));
      }
    }
    if (r.isMember("stop"))
      rules.strings = text_stops(r["stop"]);
    else if (!numeric)
      rules.strings = o_.stop;
    return rules;
  }
  // Checks the newest generated token; matches only generated tokens/text.
  bool stopped(const StopRules &rules,
               const rwkv7_server::TextStopMatcher &matcher, Rollout &row) const {
    const int id = row.generated.back();
    if (rules.ids.count(id)) {
      row.matched = id;
      return true;
    }
    for (const auto &sequence : rules.sequences)
      if (row.generated.size() >= sequence.size() &&
          std::equal(sequence.rbegin(), sequence.rend(),
                     row.generated.rbegin())) {
        row.matched = Json::Value(Json::arrayValue);
        for (int t : sequence)
          row.matched.append(t);
        return true;
      }
    // Reuse inference semantics exactly: match only the generated suffix at
    // a token boundary, preserving raw UTF-8 bytes and request order.
    auto matched = matcher.append(row.stop_tail, tokenizer_.decode(id));
    if (!matched.empty()) {
      row.matched = std::move(matched);
      row.stop_tail.clear();
      return true;
    }
    return false;
  }

  // State-mode optimizer checkpoint: the FP32 trainable state, both Adam
  // moments and the step. MiSS mode uses Trainer::checkpoint() instead.
  std::vector<std::pair<const char *, DeviceBuffer<float> *>>
  state_training_tensors() {
    return {{"initial_state", &initial_}, {"adam_m", &adam_m_},
            {"adam_v", &adam_v_}};
  }
  void save_state_training(const std::string &dir) {
    const auto temp = dir + ".tmp";
    if (std::filesystem::exists(dir) || std::filesystem::exists(temp))
      throw HttpError(409, "checkpoint destination exists");
    std::filesystem::create_directories(temp);
    try {
      std::vector<llm_infer::WriteTensor> ts;
      for (auto [name, buffer] : state_training_tensors()) {
        llm_infer::WriteTensor t{name, {int(buffer->n)}, false, {}};
        t.data.resize(buffer->n * 4);
        gpu_check(cudaMemcpyAsync(t.data.data(), buffer->p, buffer->n * 4,
                                  cudaMemcpyDeviceToHost, stream_.value),
                  "download checkpoint");
        sync();
        ts.push_back(std::move(t));
      }
      llm_infer::write_pth(temp + "/training.pth", ts);
      Json::Value m;
      m["kind"] = "online_rl_state_checkpoint";
      m["format_version"] = 1;
      m["step"] = Json::UInt64(step_);
      m["base_fingerprint"] = model_.base_fingerprint();
      m["tensor_digest"] =
          rwkv_bf16_miss::fingerprint_file(temp + "/training.pth");
      std::ofstream f(temp + "/checkpoint.json");
      f << m;
      f.close();
      if (!f)
        throw std::runtime_error("failed to write checkpoint metadata");
      std::filesystem::rename(temp, dir);
    } catch (...) {
      std::filesystem::remove_all(temp);
      throw;
    }
  }
  uint64_t load_state_training(const std::string &dir) {
    Json::Value m;
    std::ifstream f(dir + "/checkpoint.json");
    if (!f || !(f >> m) || m["kind"] != "online_rl_state_checkpoint" ||
        m["format_version"] != 1 || !m["step"].isUInt64())
      throw std::runtime_error("--resume needs a state-mode online RL checkpoint");
    if (m["base_fingerprint"].asString() != model_.base_fingerprint())
      throw std::runtime_error("checkpoint backbone fingerprint mismatch");
    if (m["tensor_digest"].asString() !=
        rwkv_bf16_miss::fingerprint_file(dir + "/training.pth"))
      throw std::runtime_error("checkpoint digest mismatch");
    auto a = llm_infer::PthArchive::open(dir + "/training.pth", true);
    if (!a.ok())
      throw std::runtime_error(a.status().message());
    auto records = llm_infer::parse_pth_tensor_records(a.value());
    if (!records.ok())
      throw std::runtime_error(records.status().message());
    for (auto [name, buffer] : state_training_tensors()) {
      const llm_infer::TensorRecord *record = nullptr;
      for (const auto &t : records.value())
        if (t.name == name)
          record = &t;
      if (!record || record->shape != std::vector<int64_t>{int64_t(buffer->n)} ||
          record->dtype != llm_infer::TensorDType::kFloat32)
        throw std::runtime_error(std::string("checkpoint tensor mismatch: ") +
                                 name);
      auto d = llm_infer::load_tensor_select(a.value(), *record, false, false,
                                             true);
      if (!d.ok())
        throw std::runtime_error(d.status().message());
      for (float x : d.value().values)
        if (!std::isfinite(x))
          throw std::runtime_error("nonfinite checkpoint");
      upload(*buffer, d.value().values, stream_.value);
    }
    return m["step"].asUInt64();
  }

public:
  Runtime(Options o, bool miss)
      : o_(std::move(o)), miss_(miss), model_(o_.model),
        learner_(model_.state_tuning_model_view()) {
    if (tokenizer_.load(o_.vocab) != rwkv7_server::kTokenizerSuccess)
      throw std::runtime_error("failed to load vocabulary");
    if (!o_.resume.empty()) {
      const auto dir = std::filesystem::path(o_.resume);
      if (!std::filesystem::is_directory(dir / "training"))
        throw std::invalid_argument(
            "--resume needs a publish directory saved with checkpoint:true");
      if (o_.adapter.empty() && std::filesystem::exists(dir / "adapter.pth"))
        o_.adapter = (dir / "adapter.pth").string();
      if (o_.state.empty() && std::filesystem::exists(dir / "state.pth"))
        o_.state = (dir / "state.pth").string();
    }
    initial_.resize(size_t(learner_.layers) * learner_.heads * 4096,
                    "learner initial state");
    initial_.zero("zero initial state");
    if (!o_.state.empty())
      initial_ = model_.load_state(o_.state);
    std::shared_ptr<const rwkv7_miss::Package> package;
    if (!o_.adapter.empty()) {
      package = rwkv7_miss::load_package(o_.adapter);
      if (package->base != model_.base_fingerprint())
        throw std::invalid_argument("adapter backbone fingerprint mismatch");
      if (!o_.rank_set)
        o_.rank = package->manifest["rank"].asInt();
      if (!o_.alpha_set)
        o_.alpha = package->manifest["alpha"].asFloat();
      if (!o_.targets_set) {
        std::set<int> selected;
        for (const auto &t : package->tensors)
          selected.insert(t.target);
        o_.targets.clear();
        for (int t : selected) {
          if (!o_.targets.empty())
            o_.targets += ",";
          o_.targets += rwkv_bf16_miss::target_names[t];
        }
      }
    }
    if (miss_) {
      Json::Value config;
      config["base_fingerprint"] = model_.base_fingerprint();
      // Trainer::resume() compares this whole object with the checkpoint's.
      config["training_dtype"] = "bfloat16";
      config["kind"] = "online_rl";
      config["rank"] = o_.rank;
      config["alpha"] = o_.alpha;
      config["targets"] = o_.targets;
      trainer_ = std::make_unique<rwkv_bf16_miss::Trainer>(
          learner_, o_.rank, o_.alpha, o_.targets, o_.chunk, config,
          stream_.value);
      if (package)
        trainer_->load_adapter(*package);
    } else {
      for (auto *d : {&gradient_, &adam_m_, &adam_v_}) {
        d->resize(initial_.n, "state Adam");
        d->zero("zero state Adam");
      }
      if (package) {
        fixed_package_ = package;
        std::vector<bf16> values(package->data.size());
        for (size_t i = 0; i < values.size(); ++i)
          values[i] =
              to_bf16(package->manifest["dtype"] == "bfloat16"
                          ? llm_infer::bf16_bits_to_float(package->data[i])
                          : llm_infer::f16_bits_to_float(package->data[i]));
        upload(fixed_adapter_, values, stream_.value);
        for (const auto &t : package->tensors) {
          if (t.layer >= learner_.layers)
            throw std::invalid_argument("adapter layer mismatch");
          auto &b = learner_.blocks[t.layer];
          if (t.in !=
                  (t.target == rwkv_bf16_miss::FfnValue ? b.ffn : b.channels) ||
              t.out !=
                  (t.target == rwkv_bf16_miss::FfnKey ? b.ffn : b.channels))
            throw std::invalid_argument("adapter dimensions mismatch");
          b.miss[t.target] = {t.in,
                              t.out,
                              t.rank,
                              package->scale,
                              fixed_adapter_.p + t.offset,
                              nullptr};
        }
      }
    }
    learner_exec_ =
        std::make_unique<Executor>(learner_, o_.chunk, stream_.value);
    auto actor_view = learner_;
    for (auto &b : actor_view.blocks)
      for (auto &linear : b.miss)
        linear.gradient = nullptr;
    actor_exec_ =
        std::make_unique<Executor>(actor_view, o_.chunk, stream_.value);
    actor_view_ = actor_view;
    if (trainer_)
      trainer_->bind_tapes(learner_exec_->tapes());
    if (!o_.resume.empty()) {
      const auto dir = (std::filesystem::path(o_.resume) / "training").string();
      if (trainer_) {
        int epoch = 0;
        uint64_t row = 0;
        trainer_->resume(dir, step_, epoch, row, initial_.p, initial_.n);
      } else
        step_ = load_state_training(dir);
      std::cout << "Resumed optimizer state at step " << step_ << std::endl;
    }
    policies_.push_back(snapshot());
    reference_ = policies_.back();
  }
  Json::Value health() const {
    Json::Value r;
    r["status"] = "ok";
    r["mode"] = miss_ ? "miss" : "state";
    r["dtype"] = "bfloat16";
    r["wkv_kernel"] = "rwkv7_statepassing_clampw";
    r["wkv_io_dtype"] = "bfloat16";
    r["wkv_state_dtype"] = "float32";
    r["gemm_mode"] = o_.gemm_mode;
    r["gemm_tile_rows"] = 32;
    r["backbone_loads"] = 1;
    r["step"] = Json::UInt64(step_);
    r["policy_version"] = policies_.back()->version;
    r["published_step"] = Json::UInt64(policies_.back()->step);
    r["reference_version"] = reference_->version;
    r["reference_step"] = Json::UInt64(reference_->step);
    r["chunk"] = o_.chunk;
    r["ctx"] = o_.ctx;
    r["vocab_size"] = learner_.vocab;
    r["max_batch"] = o_.batch;
    r["default_max_tokens"] = o_.max_tokens;
    r["default_stop"] = Json::Value(Json::arrayValue);
    for (const auto &stop : o_.stop)
      r["default_stop"].append(stop);
    r["policies"] = Json::Value(Json::arrayValue);
    for (const auto &p : policies_) {
      Json::Value v;
      v["version"] = p->version;
      v["step"] = Json::UInt64(p->step);
      v["cached_prefixes"] = Json::UInt64(p->cache.size());
      r["policies"].append(v);
    }
    return r;
  }
  Json::Value tokenize(const Json::Value &r) {
    require(r["text"].isString(), "text must be a string");
    Json::Value out;
    out["tokens"] = Json::Value(Json::arrayValue);
    for (int t : tokenizer_.encode(r["text"].asString()))
      out["tokens"].append(t);
    return out;
  }
  Json::Value detokenize(const Json::Value &r) {
    Json::Value out;
    auto text = tokenizer_.decode(tokens(r["tokens"], o_.ctx));
    out["text"] = rwkv7_server::take_complete_utf8(text, true);
    return out;
  }
  Json::Value completions(const Json::Value &r) {
    auto p = policy(r);
    const auto &prompts = r["prompts"];
    require(prompts.isArray() && !prompts.empty() &&
                prompts.size() <= unsigned(o_.batch),
            "invalid prompts batch");
    const int requested_max_new =
        number(r, "max_tokens", o_.max_tokens, 0, 1048576);
    int max_new = 0, active = 0;
    float temperature = real(r, "temperature", 1.0f, 0.01f, 100.0f);
    bool use_cache = flag(r, "use_cache", true);
    const StopRules rules = stop_rules(r);
    const rwkv7_server::TextStopMatcher matcher(rules.strings);
    require(!r.isMember("seed") || r["seed"].isUInt64(),
            "seed must be a nonnegative integer");
    // Row i draws from its own stream seeded by (seed, i), so a prompt's
    // sample does not depend on the other prompts in the request.
    const uint64_t seed =
        r.isMember("seed")
            ? r["seed"].asUInt64()
            : (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::vector<Rollout> rows(prompts.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      auto &row = rows[i];
      row.prompt = tokens(prompts[Json::ArrayIndex(i)], o_.ctx);
      // A safety cap need not fit in the remaining context. Stop each row
      // at its own capacity, without truncating any existing history.
      row.limit = std::min(requested_max_new, o_.ctx - int(row.prompt.size()));
      row.active = row.limit > 0;
      active += row.active;
      max_new = std::max(max_new, row.limit);
      std::seed_seq seq{uint32_t(seed), uint32_t(seed >> 32), uint32_t(i)};
      row.rng.seed(seq);
    }
    auto started = Clock::now();
    const int V = learner_.vocab, B = int(rows.size());
    // Prefill each prompt with the batch=1 executor (and its prefix cache),
    // then move its state into decoder row i.
    if (max_new > 0 && (!decoder_ || decoder_->rows() != B)) {
      decoder_.reset();
      decoder_ = std::make_unique<BatchDecoder>(actor_view_, B, stream_.value);
    }
    std::vector<float> logits(size_t(B) * V);
    for (int i = 0; i < B; ++i) {
      auto &row = rows[i];
      row.prefix.assign(row.prompt.begin(), row.prompt.end());
      auto hit = use_cache ? p->cache.find(p->version, row.prefix)
                           : rwkv7_server::PrefixCache<CachedState>::Match{};
      std::vector<float> last;
      if (hit.value) {
        upload(actor_exec_->state(), hit.value->state, stream_.value);
        upload(actor_exec_->shifts(), hit.value->shifts, stream_.value);
        last = hit.value->logits;
      } else
        actor_exec_->reset(p->initial.p);
      for (size_t off = hit.length; off < row.prompt.size(); off += o_.chunk)
        actor_exec_->forward(
            p->view, row.prompt.data() + off,
            int(std::min(size_t(o_.chunk), row.prompt.size() - off)),
            off + o_.chunk >= row.prompt.size() ? LogitsMode::Last
                                                : LogitsMode::None);
      if (hit.length < row.prompt.size()) {
        last = actor_exec_->last_logits();
        if (use_cache)
          cache_put(*p, row.prefix, last);
      }
      row.cached = hit.length;
      std::copy(last.begin(), last.end(), logits.begin() + size_t(i) * V);
      if (max_new > 0)
        decoder_->load(i, actor_exec_->state().p, actor_exec_->shifts().p);
    }
    sync();
    std::vector<int> feed(B, 0);
    for (int j = 0; j < max_new && active; ++j) {
      parallel_for(rows.size(), [&](size_t i) {
        auto &row = rows[i];
        if (row.active)
          std::tie(row.next, row.next_logprob) =
              sample(logits.data() + i * V, V, temperature, row.rng);
      });
      for (int i = 0; i < B; ++i) {
        auto &row = rows[i];
        if (!row.active)
          continue; // finished rows keep stepping on their last token
        row.generated.push_back(row.next);
        row.logprobs.push_back(row.next_logprob);
        row.prefix.push_back(row.next);
        feed[i] = row.next;
      }
      // Advance through the FINAL generated token too: the cached state and
      // next-token logits must describe the entire returned token history.
      decoder_->forward(p->view, feed.data());
      logits = decoder_->logits();
      for (int i = 0; i < B; ++i) {
        auto &row = rows[i];
        if (!row.active)
          continue;
        const bool stop = stopped(rules, matcher, row);
        if (stop)
          row.finish = "stop";
        if (stop || int(row.generated.size()) == row.limit) {
          row.active = false;
          --active;
          if (use_cache)
            cache_put(*p, row.prefix,
                      std::vector<float>(logits.begin() + size_t(i) * V,
                                         logits.begin() + size_t(i + 1) * V),
                      i);
        }
      }
    }
    Json::Value out;
    out["policy_version"] = p->version;
    out["step"] = Json::UInt64(p->step);
    out["temperature"] = temperature;
    out["max_tokens"] = requested_max_new;
    out["completions"] = Json::Value(Json::arrayValue);
    size_t cached = 0, prefilled = 0, generated = 0;
    for (const auto &row : rows) {
      Json::Value item;
      item["tokens"] = Json::Value(Json::arrayValue);
      item["logprobs"] = Json::Value(Json::arrayValue);
      for (size_t k = 0; k < row.generated.size(); ++k) {
        item["tokens"].append(row.generated[k]);
        item["logprobs"].append(row.logprobs[k]);
      }
      item["cached_prompt_tokens"] = Json::UInt64(row.cached);
      item["prefill_tokens"] = Json::UInt64(row.prompt.size() - row.cached);
      item["finish_reason"] = row.finish;
      item["max_tokens"] = row.limit;
      const bool truncated = requested_max_new > 0 && row.finish == "length";
      item["truncated"] = truncated;
      if (truncated)
        item["truncation_reason"] =
            row.limit < requested_max_new ? "context_length" : "max_tokens";
      if (!row.matched.isNull())
        item["matched_stop"] = row.matched;
      auto text = tokenizer_.decode(row.generated);
      item["text"] = rwkv7_server::take_complete_utf8(text, true);
      out["completions"].append(item);
      cached += row.cached;
      prefilled += row.prompt.size() - row.cached;
      generated += row.generated.size();
    }
    out["cached_prompt_tokens"] = Json::UInt64(cached);
    out["prefill_tokens"] = Json::UInt64(prefilled);
    out["generated_tokens"] = Json::UInt64(generated);
    out["seconds"] =
        std::chrono::duration<double>(Clock::now() - started).count();
    return out;
  }
  // Teacher-forced log probabilities under one policy, e.g. the reference
  // policy for a client-side KL penalty.
  Json::Value logprobs(const Json::Value &r) {
    auto p = policy(r);
    const auto &sequences = r["sequences"];
    require(sequences.isArray() && !sequences.empty() &&
                sequences.size() <= unsigned(o_.batch),
            "invalid sequences batch");
    float temperature = real(r, "temperature", 1.0f, 0.01f, 100.0f);
    std::vector<std::vector<int>> batch;
    for (const auto &v : sequences) {
      batch.push_back(tokens(v, o_.ctx));
      require(batch.back().size() >= 2, "each sequence needs two tokens");
    }
    auto started = Clock::now();
    Json::Value out;
    out["policy_version"] = p->version;
    out["step"] = Json::UInt64(p->step);
    out["temperature"] = temperature;
    out["logprobs"] = Json::Value(Json::arrayValue);
    for (const auto &ids : batch) {
      Json::Value row(Json::arrayValue);
      for (float v : actor_exec_->score(p->view, ids, p->initial.p, temperature))
        row.append(v);
      out["logprobs"].append(row);
    }
    out["seconds"] =
        std::chrono::duration<double>(Clock::now() - started).count();
    return out;
  }
  Json::Value train(const Json::Value &r) {
    require(r["policy_version"].isString(),
            "train requires policy_version from rollout");
    auto p = policy(r);
    if (p != policies_.back() || p->step != step_)
      throw HttpError(409, "stale rollout: publish the learner and collect a "
                           "new on-policy batch");
    const auto &rows = r["samples"];
    require(rows.isArray() && !rows.empty() &&
                rows.size() <= unsigned(o_.batch),
            "invalid training batch");
    float lr = real(r, "learning_rate", o_.lr, 1e-12f, 10.0f);
    float temperature = real(r, "temperature", 1.0f, 0.01f, 100.0f);
    std::vector<Sample> batch;
    size_t kept = 0, positions = 0;
    bool any_weight = false;
    for (const auto &row : rows) {
      require(row.isObject(), "sample must be an object");
      Sample s;
      s.temperature = real(row, "temperature", temperature, 0.01f, 100.0f);
      auto append = [&](const Json::Value &turn, const Json::Value &fallback) {
        require(turn.isObject(), "turn must be an object");
        auto ids = tokens(turn["tokens"], o_.ctx);
        const auto &mask = turn["loss_mask"];
        require(mask.isArray() && mask.size() == ids.size(),
                "loss_mask must align with tokens");
        const auto &adv =
            turn.isMember("advantages") ? turn["advantages"] : fallback;
        require(adv.isNumeric() || (adv.isArray() && adv.size() == ids.size()),
                "advantages must be a signed scalar or token-aligned array");
        require(s.tokens.size() + ids.size() <= size_t(o_.ctx),
                "training history of " +
                    std::to_string(s.tokens.size() + ids.size()) +
                    " tokens exceeds ctx " + std::to_string(o_.ctx) +
                    "; no truncation is performed (raise --ctx)");
        for (size_t i = 0; i < ids.size(); ++i) {
          const auto &m = mask[Json::ArrayIndex(i)];
          require(m.isBool() ||
                      (m.isInt() && (m.asInt() == 0 || m.asInt() == 1)),
                  "loss_mask entries must be boolean or 0/1");
          const auto &a = adv.isArray() ? adv[Json::ArrayIndex(i)] : adv;
          require(a.isNumeric() && std::isfinite(a.asFloat()) &&
                      std::abs(a.asFloat()) <= 1e6f,
                  "invalid signed advantage");
          bool selected = m.asBool() && !s.tokens.empty();
          if (selected)
            ++kept;
          float weight = selected ? a.asFloat() : 0.0f;
          any_weight |= weight != 0;
          s.tokens.push_back(ids[i]);
          s.weights.push_back(weight);
        }
      };
      if (row.isMember("turns")) {
        require(!row.isMember("tokens") && row["turns"].isArray() &&
                    !row["turns"].empty(),
                "use tokens or nonempty turns");
        require(!row["advantages"].isArray(),
                "multi-turn sample-level advantages must be scalar; put arrays "
                "inside each turn");
        for (const auto &turn : row["turns"])
          append(turn, row["advantages"]);
      } else
        append(row, Json::Value());
      require(s.tokens.size() >= 2,
              "training sample needs at least two tokens");
      positions += s.tokens.size() - 1;
      batch.push_back(std::move(s));
    }
    require(kept > 0, "batch has no selected prediction targets");
    Json::Value out;
    out["policy_version"] = p->version;
    out["trained_tokens"] = Json::UInt64(kept);
    out["total_tokens"] = Json::UInt64(positions);
    out["learning_rate"] = lr;
    auto started = Clock::now();
    float loss = 0;
    double grad_norm = 0;
    if (any_weight) {
      if (trainer_)
        trainer_->zero_gradient();
      else
        gpu_check(cudaMemsetAsync(gradient_.p, 0, gradient_.n * sizeof(float),
                                  stream_.value), "clear state gradient");
      for (const auto &s : batch) {
        loss += learner_exec_->backward(learner_, s.tokens, s.weights,
                                        initial_.p, kept, s.temperature);
        if (!miss_) {
          reduce_state_gradient_f32(
              stream_.value, 1, learner_.layers * learner_.heads, 64,
              learner_exec_->state_gradient(), gradient_.p, true);
        }
      }
      if (!std::isfinite(loss))
        throw std::runtime_error(
            "nonfinite batch loss; optimizer not executed");
      if (trainer_)
        grad_norm = trainer_->update(lr, step_ + 1);
      else {
        // Accumulate samples on GPU and transfer only once for validation.
        const auto accumulated = download(gradient_, stream_.value);
        for (float v : accumulated) {
          if (!std::isfinite(v) || std::abs(v) > 1e18f)
            throw std::runtime_error(
                "nonfinite/overflow state gradient; optimizer not executed");
          grad_norm += double(v) * v;
        }
        grad_norm = std::sqrt(grad_norm);
        AdamConfig adam;
        adam.learning_rate = lr;
        adam_update_state_f32(stream_.value, initial_.p, gradient_.p, adam_m_.p,
                              adam_v_.p, initial_.n, step_ + 1, adam, true);
      }
      sync();
      ++step_;
    }
    out["updated"] = any_weight;
    out["publish_required"] = any_weight;
    out["step"] = Json::UInt64(step_);
    out["loss"] = loss;
    out["grad_norm"] = grad_norm;
    out["seconds"] =
        std::chrono::duration<double>(Clock::now() - started).count();
    return out;
  }
  Json::Value publish(const Json::Value &r) {
    require(r["expected_step"].isUInt64(), "publish requires expected_step");
    if (r["expected_step"].asUInt64() != step_)
      throw HttpError(409, "learner step changed");
    // checkpoint:true implies save and adds the optimizer state.
    bool checkpoint = flag(r, "checkpoint", false);
    bool save = flag(r, "save", false) || checkpoint;
    const bool changed = policies_.back()->step != step_;
    std::shared_ptr<Policy> next;
    if (changed) {
      ++version_;
      next = snapshot();
    } else
      next = policies_.back();
    Json::Value out;
    if (!next->directory.empty())
      out["directory"] = next->directory;
    if (save && next->directory.empty()) {
      auto file_id = run_id_;
      std::replace(file_id.begin(), file_id.end(), ':', '_');
      const auto dir = std::filesystem::path(o_.output) /
                       (file_id + "-step-" + std::to_string(step_));
      const auto temp = dir.string() + ".tmp";
      if (std::filesystem::exists(dir) || std::filesystem::exists(temp))
        throw HttpError(409, "policy export destination exists");
      std::filesystem::create_directories(temp);
      try {
        save_state_checkpoint_pth(temp + "/state.pth", stream_.value,
                                  learner_.layers, learner_.heads, 64,
                                  initial_.p);
        if (trainer_)
          trainer_->export_adapter(temp + "/adapter.pth");
        else if (fixed_package_) {
          rwkv7_miss::save_adapter_pth(
              temp + "/adapter.pth", fixed_package_->manifest,
              fixed_package_->tensors, fixed_package_->data);
        }
        Json::Value meta;
        meta["policy_version"] = next->version;
        meta["step"] = Json::UInt64(step_);
        meta["mode"] = miss_ ? "miss" : "state";
        meta["gemm_mode"] = o_.gemm_mode;
        meta["base_fingerprint"] = model_.base_fingerprint();
        std::ofstream f(temp + "/policy.json");
        f << meta;
        f.close();
        if (!f)
          throw std::runtime_error("failed to write policy metadata");
        std::filesystem::rename(temp, dir);
        next->directory = dir.string();
        out["directory"] = next->directory;
      } catch (...) {
        std::filesystem::remove_all(temp);
        throw;
      }
    }
    if (checkpoint) {
      const auto dir = std::filesystem::path(next->directory) / "training";
      if (!std::filesystem::exists(dir)) {
        if (trainer_)
          trainer_->checkpoint(dir.string(), step_, 0, 0, initial_.p,
                               initial_.n);
        else
          save_state_training(dir.string());
      }
      out["checkpoint"] = dir.string();
    }
    // The single worker finishes old requests before this atomic publication.
    // Explicit old-version requests keep their own immutable weights and trie.
    if (changed)
      policies_.push_back(std::move(next));
    while (policies_.size() > size_t(o_.versions))
      policies_.pop_front();
    out["policy_version"] = policies_.back()->version;
    out["step"] = Json::UInt64(step_);
    return out;
  }
};

// Bounded single GPU worker: Drogon IO loops only enqueue/deliver JSON. This
// intentionally serializes rollout, train and publish without duplicate models
// or concurrent access to reusable CUDA workspaces.
class Worker {
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_;
  bool stop_ = false;
  std::thread thread_;

public:
  explicit Worker(bool strict)
      : thread_([this, strict] {
          set_consistent_forward(true, !strict);
          for (;;) {
            std::function<void()> job;
            {
              std::unique_lock<std::mutex> lock(mutex_);
              cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
              if (stop_ && queue_.empty())
                return;
              job = std::move(queue_.front());
              queue_.pop_front();
            }
            job();
          }
        }) {}
  ~Worker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
  bool enqueue(std::function<void()> job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || queue_.size() >= 32)
      return false;
    queue_.push_back(std::move(job));
    cv_.notify_one();
    return true;
  }
};
void respond(const std::function<void(const drogon::HttpResponsePtr &)> &cb,
             const Json::Value &body, int status = 200) {
  auto response = drogon::HttpResponse::newHttpJsonResponse(body);
  response->setStatusCode(static_cast<drogon::HttpStatusCode>(status));
  cb(response);
}
} // namespace
int run(int argc, char **argv, bool miss) {
  try {
    Options o = parse(argc, argv, miss);
    Runtime runtime(o, miss);
    Worker worker(o.gemm_mode == "strict");
    auto route = [&](const std::string &path, drogon::HttpMethod method,
                     auto action) {
      drogon::app().registerHandler(
          path,
          [&, action, method](
              const drogon::HttpRequestPtr &req,
              std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
            auto body = req->getJsonObject();
            if (method == drogon::Post && (!body || !body->isObject())) {
              Json::Value error;
              error["error"] = "expected a JSON object";
              respond(callback, error, 400);
              return;
            }
            Json::Value value = body ? *body : Json::Value(Json::objectValue);
            auto cb = std::move(callback);
            if (!worker.enqueue([&, action, value = std::move(value), cb] {
                  try {
                    respond(cb, action(value));
                  } catch (const HttpError &e) {
                    Json::Value error;
                    error["error"] = e.what();
                    respond(cb, error, e.status);
                  } catch (const std::exception &e) {
                    Json::Value error;
                    error["error"] = e.what();
                    respond(cb, error, 500);
                  }
                })) {
              Json::Value error;
              error["error"] = "GPU work queue is full";
              respond(cb, error, 503);
            }
          },
          {method});
    };
    route("/health", drogon::Get,
          [&](const Json::Value &) { return runtime.health(); });
    route("/v1/tokenize", drogon::Post,
          [&](const Json::Value &r) { return runtime.tokenize(r); });
    route("/v1/detokenize", drogon::Post,
          [&](const Json::Value &r) { return runtime.detokenize(r); });
    route("/v1/completions", drogon::Post,
          [&](const Json::Value &r) { return runtime.completions(r); });
    route("/v1/logprobs", drogon::Post,
          [&](const Json::Value &r) { return runtime.logprobs(r); });
    route("/v1/train", drogon::Post,
          [&](const Json::Value &r) { return runtime.train(r); });
    route("/v1/publish", drogon::Post,
          [&](const Json::Value &r) { return runtime.publish(r); });
    std::cout << "Online RL " << (miss ? "MiSS" : "state") << " ready http://"
              << o.host << ':' << o.port << std::endl;
    drogon::app()
        // GPU jobs can run for minutes without socket traffic. Drogon's default
        // 60-second idle timeout otherwise closes a live training request.
        .setIdleConnectionTimeout(0)
        .setClientMaxBodySize(16 * 1024 * 1024)
        .setThreadNum(2)
        .addListener(o.host, o.port)
        .run();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "online RL server: " << e.what() << '\n';
    return 1;
  }
}
} // namespace rwkv_online_rl
