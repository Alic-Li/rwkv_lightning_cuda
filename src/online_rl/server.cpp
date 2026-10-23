#include "server.hpp"
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include "rwkv/inference/prefix_cache.hpp"
#include "rwkv/inference/rwkv_tokenizer.hpp"
#include "rwkv/inference/utf8.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/version.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <drogon/drogon.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
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
struct Options {
  std::string model, adapter, state, host = "127.0.0.1",
                                     output = "online-rl-output";
  std::string vocab = RWKV_STATE_TUNE_DEFAULT_VOCAB, targets = "all";
  int port = 8848, chunk = 64, ctx = 4096, batch = 32, cache = 16, versions = 2,
      rank = 16;
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
    else if (key == "--vocab")
      o.vocab = v;
    else if (key == "--host")
      o.host = v;
    else if (key == "--output")
      o.output = v;
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
      o.versions > 32 || o.lr <= 0 || o.rank < 1 || o.rank > 1024)
    throw std::invalid_argument(
        "invalid model/port/chunk/ctx/batch/cache/versions/lr/rank");
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
  rwkv7_server::TrieTokenizer tokenizer_;
  std::deque<std::shared_ptr<Policy>> policies_;
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
  void cache_put(Policy &p, const std::vector<int64_t> &prefix,
                 const std::vector<float> &logits) {
    if (!o_.cache)
      return;
    auto c = std::make_shared<CachedState>();
    c->state = download(actor_exec_->state(), stream_.value);
    c->shifts = download(actor_exec_->shifts(), stream_.value);
    c->logits = logits;
    p.cache.put(p.version, prefix, std::move(c));
  }
  std::pair<int, float> sample(const std::vector<float> &logits,
                               float temperature, std::mt19937_64 &rng) {
    // Ordinary softmax on the CPU: no sampler kernel, filters or penalties.
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> weights(logits.size());
    double sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
      if (!std::isfinite(logits[i]))
        throw std::runtime_error("nonfinite rollout logits");
      weights[i] = std::exp((logits[i] - maximum) / temperature);
      sum += weights[i];
    }
    std::discrete_distribution<int> distribution(weights.begin(),
                                                 weights.end());
    int id = distribution(rng);
    float logprob =
        float((double(logits[id]) - maximum) / temperature - std::log(sum));
    return {id, logprob};
  }

public:
  Runtime(Options o, bool miss)
      : o_(std::move(o)), miss_(miss), model_(o_.model),
        learner_(model_.state_tuning_model_view()) {
    if (tokenizer_.load(o_.vocab) != rwkv7_server::kTokenizerSuccess)
      throw std::runtime_error("failed to load vocabulary");
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
    if (trainer_)
      trainer_->bind_tapes(learner_exec_->tapes());
    policies_.push_back(snapshot());
  }
  Json::Value health() const {
    Json::Value r;
    r["status"] = "ok";
    r["mode"] = miss_ ? "miss" : "state";
    r["dtype"] = "bfloat16";
    r["backbone_loads"] = 1;
    r["step"] = Json::UInt64(step_);
    r["policy_version"] = policies_.back()->version;
    r["published_step"] = Json::UInt64(policies_.back()->step);
    r["chunk"] = o_.chunk;
    r["ctx"] = o_.ctx;
    r["vocab_size"] = learner_.vocab;
    r["max_batch"] = o_.batch;
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
    int max_new = number(r, "max_tokens", 32, 0, o_.ctx);
    float temperature = real(r, "temperature", 1.0f, 0.01f, 100.0f);
    bool use_cache = flag(r, "use_cache", true);
    std::vector<std::vector<int>> batch;
    for (const auto &v : prompts) {
      batch.push_back(tokens(v, o_.ctx));
      require(batch.back().size() + max_new <= size_t(o_.ctx),
              "prompt + max_tokens exceeds ctx");
    }
    std::set<int> stops;
    if (r.isMember("stop_token_ids")) {
      require(r["stop_token_ids"].isArray(), "stop_token_ids must be an array");
      for (const auto &id : r["stop_token_ids"]) {
        require(id.isInt() && id.asInt() >= 0 && id.asInt() < learner_.vocab,
                "invalid stop token");
        stops.insert(id.asInt());
      }
    } else
      stops.insert(0);
    require(!r.isMember("seed") || r["seed"].isUInt64(),
            "seed must be a nonnegative integer");
    std::mt19937_64 rng(r.isMember("seed") ? r["seed"].asUInt64()
                                           : std::random_device{}());
    auto started = Clock::now();
    Json::Value out;
    out["policy_version"] = p->version;
    out["step"] = Json::UInt64(p->step);
    out["temperature"] = temperature;
    out["completions"] = Json::Value(Json::arrayValue);
    size_t cached = 0, prefilled = 0, generated = 0;
    for (const auto &prompt : batch) {
      std::vector<int64_t> prefix(prompt.begin(), prompt.end());
      auto hit = use_cache ? p->cache.find(p->version, prefix)
                           : rwkv7_server::PrefixCache<CachedState>::Match{};
      std::vector<float> logits;
      if (hit.value) {
        upload(actor_exec_->state(), hit.value->state, stream_.value);
        upload(actor_exec_->shifts(), hit.value->shifts, stream_.value);
        logits = hit.value->logits;
      } else
        actor_exec_->reset(p->initial.p);
      for (size_t off = hit.length; off < prompt.size(); off += o_.chunk) {
        actor_exec_->forward(
            p->view, prompt.data() + off,
            int(std::min(size_t(o_.chunk), prompt.size() - off)));
      }
      if (hit.length < prompt.size())
        logits = actor_exec_->last_logits();
      if (use_cache && hit.length < prompt.size())
        cache_put(*p, prefix, logits);
      Json::Value row;
      row["tokens"] = Json::Value(Json::arrayValue);
      row["logprobs"] = Json::Value(Json::arrayValue);
      row["cached_prompt_tokens"] = Json::UInt64(hit.length);
      row["prefill_tokens"] = Json::UInt64(prompt.size() - hit.length);
      row["finish_reason"] = "length";
      std::vector<int> generated_ids;
      for (int j = 0; j < max_new; ++j) {
        auto [id, logprob] = sample(logits, temperature, rng);
        row["tokens"].append(id);
        row["logprobs"].append(logprob);
        generated_ids.push_back(id);
        prefix.push_back(id);
        // Advance through the FINAL generated token too: the cached state and
        // next-token logits must describe the entire returned token history.
        actor_exec_->forward(p->view, &id, 1);
        logits = actor_exec_->last_logits();
        if (stops.count(id)) {
          row["finish_reason"] = "stop";
          break;
        }
      }
      if (use_cache && !generated_ids.empty())
        cache_put(*p, prefix, logits);
      auto text = tokenizer_.decode(generated_ids);
      row["text"] = rwkv7_server::take_complete_utf8(text, true);
      out["completions"].append(row);
      cached += hit.length;
      prefilled += prompt.size() - hit.length;
      generated += generated_ids.size();
    }
    out["cached_prompt_tokens"] = Json::UInt64(cached);
    out["prefill_tokens"] = Json::UInt64(prefilled);
    out["generated_tokens"] = Json::UInt64(generated);
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
                "training history exceeds ctx; no truncation is performed");
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
      std::vector<float> accumulated(miss_ ? 0 : initial_.n, 0.0f);
      for (const auto &s : batch) {
        loss += learner_exec_->backward(learner_, s.tokens, s.weights,
                                        initial_.p, kept, s.temperature);
        if (!miss_) {
          std::vector<float> g(initial_.n);
          gpu_check(cudaMemcpyAsync(g.data(), learner_exec_->state_gradient(),
                                    g.size() * 4, cudaMemcpyDeviceToHost,
                                    stream_.value),
                    "read state gradient");
          sync();
          for (size_t i = 0; i < g.size(); ++i) {
            accumulated[i] += g[i];
            if (!std::isfinite(accumulated[i]) ||
                std::abs(accumulated[i]) > 1e18f)
              throw std::runtime_error(
                  "nonfinite/overflow state gradient; optimizer not executed");
          }
        }
      }
      if (!std::isfinite(loss))
        throw std::runtime_error(
            "nonfinite batch loss; optimizer not executed");
      if (trainer_)
        grad_norm = trainer_->update(lr, step_ + 1);
      else {
        for (float v : accumulated)
          grad_norm += double(v) * v;
        grad_norm = std::sqrt(grad_norm);
        upload(gradient_, accumulated, stream_.value);
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
    bool save = flag(r, "save", false);
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
  Worker()
      : thread_([this] {
          set_consistent_forward(true);
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
    Worker worker;
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
