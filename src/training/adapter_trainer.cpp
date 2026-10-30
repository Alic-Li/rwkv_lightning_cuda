#include "rwkv/bf16_training/trainer.hpp"
#include "gpu_check.hpp"
#include "rwkv/io/adapter_layout.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <sstream>
namespace rwkv_bf16_miss {
using detail::check;
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
} // namespace rwkv_bf16_miss
