#pragma once
#include "rwkv/bf16_training/training.hpp"
#include <cstring>
#include <random>

// Small deterministic model, owned entirely by the test. No external weights
// or archive loader are needed to exercise the training executor in CTest.
struct Bf16ModelFixture {
  rwkv_bf16_training::FrozenModelView view;
  std::vector<rwkv_bf16_training::DeviceBuffer<bf16>> storage;
  std::vector<std::uint16_t> embedding;
  std::mt19937 rng{1234};

  const bf16 *weight(size_t n, float value = 0, bool random = true) {
    std::normal_distribution<float> dist(0, .06f);
    std::vector<bf16> host(n);
    for (auto &x : host)
      x = to_bf16(random ? dist(rng) : value);
    storage.emplace_back();
    auto &buffer = storage.back();
    buffer.resize(n, "fixture weight");
    rwkv_bf16_training::gpu_check(
        cudaMemcpy(buffer.p, host.data(), n * sizeof(bf16), cudaMemcpyHostToDevice),
        "upload fixture weight");
    return buffer.p;
  }

  explicit Bf16ModelFixture(int layers = 2, bool residual_only = false) {
    constexpr int C = 64, F = 128, R = 16, V = 256;
    view.layers = layers;
    view.channels = C;
    view.heads = 1;
    view.vocab = V;
    embedding.resize(V * C);
    std::normal_distribution<float> dist(0, 1);
    for (auto &x : embedding) {
      const bf16 value = to_bf16(dist(rng));
      std::memcpy(&x, &value, sizeof(x));
    }
    view.cpu_emb_ln0_bf16 = embedding.data();
    view.cpu_emb_ln0_elements = embedding.size();
    view.ln_out_weight = weight(C, 1, false);
    view.ln_out_bias = weight(C, 0, false);
    view.head_weight_orig = weight(V * C);
    view.blocks.resize(layers);
    for (auto &w : view.blocks) {
      w.channels = C;
      w.heads = 1;
      w.ffn = F;
      w.rank_w = w.rank_a = w.rank_g = w.rank_v = R;
      w.ln1_weight = w.ln2_weight = weight(C, 1, false);
      w.ln1_bias = w.ln2_bias = weight(C, 0, false);
      w.mix_r = w.mix_w = w.mix_k = w.mix_v = w.mix_a = w.mix_g =
          weight(C, .5f, false);
      w.receptance = weight(C * C);
      w.key = weight(C * C);
      w.value = weight(C * C);
      w.output = weight(C * C, 0, !residual_only);
      w.w1 = weight(C * R);
      w.w2 = weight(R * C);
      w.w0 = weight(C, 0, false);
      w.a1 = weight(C * R);
      w.a2 = weight(R * C);
      w.a0 = weight(C, 0, false);
      w.g1 = weight(C * R);
      w.g2 = weight(R * C);
      w.v1 = weight(C * R);
      w.v2 = weight(R * C);
      w.v0 = weight(C, 0, false);
      w.k_k = w.k_a = weight(C, 1, false);
      w.r_k = weight(C);
      w.att_group_norm_weight = weight(C, 1, false);
      w.att_group_norm_bias = weight(C, 0, false);
      w.ffn_mix = weight(C, .5f, false);
      w.ffn_key = weight(F * C);
      w.ffn_value = weight(F * C, 0, !residual_only);
    }
  }
};
