#include "rwkv/bf16_training/trainer.hpp"
#include "rwkv/bf16_training/checkpoint_config.hpp"
#include "rwkv/io/adapter_layout.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
namespace rwkv_bf16_miss {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace
Trainer::Trainer(rwkv_bf16_training::FrozenModelView &model, int rank,
                 float alpha, const std::string &targets, int time,
                 Json::Value config, cudaStream_t stream)
    : stream_(stream), config_(config), nora_(config["method"] == "nora"),
      rank_(rank), alpha_(alpha) {
  if (rank < 1 || rank > 1024 || !std::isfinite(alpha))
    throw std::invalid_argument("invalid adapter rank/alpha");
  if (nora_ && (alpha != rank || config_["initialization"] != "nora_init"))
    throw std::invalid_argument("NoRA requires nora_init and alpha/rank = 1");
  if (time < 1 || time > 65535)
    throw std::invalid_argument("adapter chunk must be 1..65535");
  std::set<int> selected;
  if (targets == "all")
    for (int i = 0; i < TargetCount; ++i)
      selected.insert(i);
  else {
    std::istringstream list(targets);
    std::string t;
    while (std::getline(list, t, ',')) {
      int i = 0;
      for (; i < TargetCount; ++i)
        if (t == target_names[i])
          break;
      if (i == TargetCount || !selected.insert(i).second)
        throw std::invalid_argument("unknown/duplicate adapter target: " + t);
    }
  }
  if (selected.empty())
    throw std::invalid_argument("empty adapter targets");
  size_t count = 0;
  reduced_.resize(model.layers);
  for (int l = 0; l < model.layers; ++l)
    for (int i : selected) {
      const auto &w = model.blocks[l];
      HostTensor t{l,
                   i,
                   i == FfnValue ? w.ffn : w.channels,
                   i == FfnKey ? w.ffn : w.channels,
                   rank,
                   count};
      count += size_t(t.out) * rank;
      if (nora_) {
        t.a_offset = count;
        count += size_t(rank) * t.in;
      }
      targets_.push_back(t);
      reduced_[l][i].resize(size_t(time) * rank, "adapter latent tape");
    }
  master_.resize(count, "adapter FP32 master");
  gradient_.resize(count, "adapter FP32 gradient");
  m_.resize(count, "adapter Adam m");
  v_.resize(count, "adapter Adam v");
  forward_.resize(count, "adapter BF16 factors");
  for (auto *p : {master_.p, gradient_.p, m_.p, v_.p})
    check(cudaMemsetAsync(p, 0, count * 4, stream_));
  check(cudaMemsetAsync(forward_.p, 0, count * 2, stream_));
  if (nora_) {
    // Random Kaiming-uniform directions, normalized along rank (each input
    // column), exactly once. B remains zero; both factors are then trainable.
    std::mt19937_64 rng(config_.get("seed", Json::UInt64(1234)).asUInt64());
    std::vector<float> initial(count, 0);
    for (const auto &t : targets_) {
      const float bound = 1 / std::sqrt(float(t.in));
      for (int k = 0; k < t.in; ++k) {
        double norm = 0;
        for (int j = 0; j < rank; ++j) {
          float v = float(double(rng() >> 11) * 0x1.0p-53 * 2 - 1) * bound;
          initial[t.a_offset + size_t(j) * t.in + k] = v;
          norm += double(v) * v;
        }
        const float denom = float(std::max(std::sqrt(norm), 1e-12));
        for (int j = 0; j < rank; ++j)
          initial[t.a_offset + size_t(j) * t.in + k] /= denom;
      }
    }
    check(cudaMemcpyAsync(master_.p, initial.data(), count * 4,
                          cudaMemcpyHostToDevice, stream_));
    cast_master(stream_, master_.p, forward_.p, count);
    check(cudaStreamSynchronize(stream_));
  }
  for (const auto &t : targets_)
    model.blocks[t.layer].miss[t.target] = {
        t.in,
        t.out,
        rank,
        alpha / rank,
        forward_.p + t.offset,
        gradient_.p + t.offset,
        nora_ ? forward_.p + t.a_offset : nullptr,
        nora_ ? gradient_.p + t.a_offset : nullptr};
}
void Trainer::bind_tapes(
    std::vector<rwkv_bf16_training::BlockTapeView> &tapes) {
  for (const auto &t : targets_)
    tapes[t.layer].miss_reduced[t.target] = reduced_[t.layer][t.target].p;
}
double Trainer::update(float lr, uint64_t step) {
  // Check the whole accumulated gradient before touching any optimizer tensor.
  std::vector<float> g(gradient_.n);
  check(cudaMemcpyAsync(g.data(), gradient_.p, g.size() * 4,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  double norm = 0;
  for (float x : g) {
    norm += double(x) * x;
    if (!std::isfinite(x) || std::abs(x) > 1e18f)
      throw std::runtime_error(
          "nonfinite/overflow adapter gradient; optimizer not executed");
  }
  rwkv_bf16_training::AdamConfig adam;
  adam.learning_rate = lr;
  rwkv_bf16_training::adam_update_state_f32(
      stream_, master_.p, gradient_.p, m_.p, v_.p, master_.n, step, adam, true);
  cast_master(stream_, master_.p, forward_.p, master_.n);
  return std::sqrt(norm);
}
void Trainer::zero_gradient() {
  check(cudaMemsetAsync(gradient_.p, 0, gradient_.n * 4, stream_));
}
void Trainer::load_adapter(const rwkv7_miss::Package &package) {
  if ((package.manifest["method"] == "nora") != nora_ ||
      package.base != config_["base_fingerprint"].asString() ||
      package.manifest["rank"].asInt() != rank_ ||
      package.scale != alpha_ / rank_ ||
      package.tensors.size() != targets_.size())
    throw std::invalid_argument(
        "cold adapter base/rank/scale/targets mismatch");
  std::vector<float> values(master_.n);
  for (const auto &t : targets_) {
    auto it = std::find_if(package.tensors.begin(), package.tensors.end(),
                           [&](const auto &p) {
                             return t.layer == p.layer && t.target == p.target;
                           });
    if (it == package.tensors.end() || it->in != t.in || it->out != t.out ||
        it->rank != t.rank)
      throw std::invalid_argument("cold adapter tensor mismatch");
    rwkv_adapter_detail::components(
        t, nora_, target_names,
        [&](const std::string &name, int rows, int cols, size_t offset) {
          size_t source =
              nora_ && name.back() == 'A' ? it->a_offset : it->offset;
          for (size_t j = 0; j < size_t(rows) * cols; ++j) {
            auto bits = package.data.at(source + j);
            values[offset + j] = package.manifest["dtype"] == "bfloat16"
                                     ? llm_infer::bf16_bits_to_float(bits)
                                     : llm_infer::f16_bits_to_float(bits);
          }
        });
  }
  check(cudaMemcpyAsync(master_.p, values.data(), values.size() * 4,
                        cudaMemcpyHostToDevice, stream_));
  for (auto *p : {gradient_.p, m_.p, v_.p})
    check(cudaMemsetAsync(p, 0, master_.n * 4, stream_));
  cast_master(stream_, master_.p, forward_.p, master_.n);
  check(cudaStreamSynchronize(stream_));
}
rwkv_bf16_training::DeviceBuffer<bf16>
Trainer::snapshot(rwkv_bf16_training::FrozenModelView &model) const {
  rwkv_bf16_training::DeviceBuffer<bf16> result;
  result.resize(forward_.n, "actor BF16 adapter snapshot");
  check(cudaMemcpyAsync(result.p, forward_.p, forward_.n * 2,
                        cudaMemcpyDeviceToDevice, stream_));
  for (const auto &t : targets_) {
    auto &linear = model.blocks[t.layer].miss[t.target];
    linear.d = result.p + t.offset;
    linear.gradient = nullptr;
    linear.a = nora_ ? result.p + t.a_offset : nullptr;
    linear.gradient_a = nullptr;
  }
  check(cudaStreamSynchronize(stream_));
  return result;
}
void Trainer::checkpoint(const std::string &dir, uint64_t step, int epoch,
                         uint64_t row, const float *initial, size_t states) {
  if (std::filesystem::exists(dir) || std::filesystem::exists(dir + ".tmp"))
    throw std::runtime_error("checkpoint destination exists");
  std::filesystem::create_directories(dir + ".tmp");
  std::vector<llm_infer::WriteTensor> ts;
  auto add = [&](const std::string &n, const float *p, std::vector<int> shape,
                 size_t count) {
    llm_infer::WriteTensor t{n, shape, false, {}};
    t.data.resize(count * 4);
    check(cudaMemcpyAsync(t.data.data(), p, count * 4, cudaMemcpyDeviceToHost,
                          stream_));
    check(cudaStreamSynchronize(stream_));
    ts.push_back(std::move(t));
  };
  for (const auto &t : targets_)
    for (const auto &p : std::vector<std::pair<std::string, const float *>>{
             {"master", master_.p},
             {"gradient", gradient_.p},
             {"adam_m", m_.p},
             {"adam_v", v_.p}})
      rwkv_adapter_detail::components(
          t, nora_, target_names,
          [&](const std::string &name, int rows, int cols, size_t offset) {
            add(name + "." + p.first, p.second + offset, {rows, cols},
                size_t(rows) * cols);
          });
  add("initial_state", initial, {int(states)}, states);
  std::vector<uint16_t> forward(forward_.n);
  check(cudaMemcpyAsync(forward.data(), forward_.p, forward.size() * 2,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  Json::Value manifest;
  if (nora_)
    manifest["method"] = "nora";
  manifest["dtype"] = "bfloat16";
  manifest["rank"] = rank_;
  manifest["alpha"] = alpha_;
  manifest["scale"] = alpha_ / rank_;
  manifest["base_fingerprint"] = config_["base_fingerprint"];
  manifest = adapter_manifest(manifest, targets_, forward);
  manifest["tensor_source"] = "master";
  llm_infer::write_pth(dir + ".tmp/training.pth", ts,
                       Json::writeString(Json::StreamWriterBuilder{}, manifest),
                       nora_ ? "archive/nora.json" : "archive/miss.json");
  Json::Value m;
  m["kind"] = nora_ ? "nora_training_checkpoint" : "miss_training_checkpoint";
  m["format_version"] = 1;
  m["config"] = config_;
  m["step"] = Json::UInt64(step);
  m["epoch"] = epoch;
  m["row"] = Json::UInt64(row);
  m["tensor_digest"] = fingerprint_file(dir + ".tmp/training.pth");
  std::mt19937_64 rng(config_["seed"].asUInt64());
  std::ostringstream rng_state;
  rng_state << rng;
  m["rng"] = rng_state.str();
  m["scheduler"] = config_["planned_steps"];
  {
    std::ofstream f(dir + ".tmp/checkpoint.json");
    f << m;
    if (!f)
      throw std::runtime_error("checkpoint metadata write failed");
  }
  std::filesystem::rename(dir + ".tmp", dir);
}
void Trainer::resume(const std::string &dir, uint64_t &step, int &epoch,
                     uint64_t &row, float *initial, size_t states) {
  Json::Value m;
  std::ifstream f(dir + "/checkpoint.json");
  if (!f)
    throw std::runtime_error("resume requires a training checkpoint directory");
  f >> m;
  if (m["kind"] !=
          (nora_ ? "nora_training_checkpoint" : "miss_training_checkpoint") ||
      m["format_version"] != 1 || m["config"]["training_dtype"] != "bfloat16" ||
      !checkpoint_config_matches(m["config"], config_))
    throw std::runtime_error(
        "resume requires a matching BF16 training checkpoint/config");
  if (m["tensor_digest"].asString() != fingerprint_file(dir + "/training.pth"))
    throw std::runtime_error("checkpoint digest mismatch");
  std::mt19937_64 rng;
  std::istringstream rs(m["rng"].asString());
  if (!(rs >> rng))
    throw std::runtime_error("invalid RNG state");
  auto a = llm_infer::PthArchive::open(dir + "/training.pth", true);
  if (!a.ok())
    throw std::runtime_error(a.status().message());
  auto records = llm_infer::parse_pth_tensor_records(a.value());
  if (!records.ok())
    throw std::runtime_error(records.status().message());
  if (records.value().size() != targets_.size() * (nora_ ? 8 : 4) + 1)
    throw std::runtime_error("checkpoint tensor count");
  auto load = [&](const std::string &n, float *p, std::vector<int64_t> shape) {
    const llm_infer::TensorRecord *r = nullptr;
    for (const auto &t : records.value())
      if (t.name == n) {
        if (r)
          throw std::runtime_error("duplicate checkpoint tensor");
        r = &t;
      }
    if (!r || r->shape != shape || r->dtype != llm_infer::TensorDType::kFloat32)
      throw std::runtime_error("checkpoint tensor mismatch: " + n);
    auto d = llm_infer::load_tensor_select(a.value(), *r, false, false, true);
    if (!d.ok())
      throw std::runtime_error(d.status().message());
    for (float x : d.value().values)
      if (!std::isfinite(x))
        throw std::runtime_error("nonfinite checkpoint");
    check(cudaMemcpyAsync(p, d.value().values.data(),
                          d.value().values.size() * 4, cudaMemcpyHostToDevice,
                          stream_));
    check(cudaStreamSynchronize(stream_));
  };
  for (const auto &t : targets_)
    for (const auto &p :
         std::vector<std::pair<std::string, float *>>{{"master", master_.p},
                                                      {"gradient", gradient_.p},
                                                      {"adam_m", m_.p},
                                                      {"adam_v", v_.p}})
      rwkv_adapter_detail::components(
          t, nora_, target_names,
          [&](const std::string &name, int rows, int cols, size_t offset) {
            load(name + "." + p.first, p.second + offset, {rows, cols});
          });
  load("initial_state", initial, {int64_t(states)});
  cast_master(stream_, master_.p, forward_.p, master_.n);
  step = m["step"].asUInt64();
  epoch = m["epoch"].asInt();
  row = m["row"].asUInt64();
}
void Trainer::export_adapter(const std::string &dir) {
  std::vector<uint16_t> data(forward_.n);
  check(cudaMemcpyAsync(data.data(), forward_.p, data.size() * 2,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  Json::Value m;
  if (nora_)
    m["method"] = "nora";
  m["dtype"] = "bfloat16";
  m["rank"] = rank_;
  m["alpha"] = alpha_;
  m["scale"] = alpha_ / rank_;
  m["base_fingerprint"] = config_["base_fingerprint"];
  save_adapter_pth(dir, m, targets_, data);
}
} // namespace rwkv_bf16_miss
