#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rwkv/runtime/rwkv_server_backend.hpp"
#include "rwkv_quantized.hpp"
#include "test_common.hpp"

namespace {

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;
  bool int4 = false;
  float fill = 0.0f;
};

std::size_t elements(const std::vector<std::int64_t> &shape) {
  std::size_t count = 1;
  for (const auto dim : shape)
    count *= static_cast<std::size_t>(dim);
  return count;
}

void append_tensor(llm_infer::QuantizedWriter &writer, const TensorSpec &spec,
                   int bits, int G, bool reference) {
  const std::size_t count = elements(spec.shape);
  std::vector<std::uint8_t> data;
  std::vector<std::uint16_t> scales;
  auto dtype = llm_infer::QuantizedDType::kBFloat16;
  if (spec.int4) {
    const int N = spec.shape[0], K = spec.shape[1], groups = (K + G - 1) / G;
    scales.resize(bits == 4 ? std::size_t(N) * groups : N);
    for (std::size_t i = 0; i < scales.size(); ++i)
      scales[i] = llm_infer::float_to_f16_bits(float(i % 5 + 1) /
                                               (bits == 4 ? 1024 : 16384));
    data.assign(reference   ? count * 2
                : bits == 4 ? std::size_t(N) * ((K + 1) / 2)
                            : count,
                0);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        const int q = bits == 4 ? ((n * 17 + k * 11) % 16) - 8
                                : ((n * 17 + k * 11) % 256) - 128;
        if (reference) {
          const float scale = llm_infer::f16_bits_to_float(
              scales[bits == 4 ? n * groups + k / G : n]);
          const auto h = llm_infer::float_to_f16_bits(q * scale);
          data[(std::size_t(n) * K + k) * 2] = h;
          data[(std::size_t(n) * K + k) * 2 + 1] = h >> 8;
        } else if (bits == 4)
          data[std::size_t(n) * ((K + 1) / 2) + k / 2] |= (q & 15)
                                                          << (4 * (k % 2));
        else
          data[std::size_t(n) * K + k] = static_cast<std::uint8_t>(q);
      }
    dtype = reference   ? llm_infer::QuantizedDType::kFloat16
            : bits == 4 ? llm_infer::QuantizedDType::kInt4
                        : llm_infer::QuantizedDType::kInt8;
    if (reference)
      scales.clear();
  } else {
    data.resize(count * 2);
    for (std::size_t i = 0; i < count; ++i) {
      float value = spec.fill + float(int((i * 13 + 7) % 29) - 14) * 0.002f;
      if (spec.name == "blocks.0.att.w0")
        value = -2.0f;
      const auto h = llm_infer::float_to_bf16_bits(value);
      data[i * 2] = h;
      data[i * 2 + 1] = h >> 8;
    }
  }
  TEST_CHECK(writer
                 .append(spec.name, dtype, spec.shape, data, scales,
                         bits == 4 && !reference && spec.int4 ? G : 0)
                 .ok_status());
}

std::vector<TensorSpec> model_tensors() {
  constexpr int C = 256;
  constexpr int F = 512;
  constexpr int V = 256;
  constexpr int R = 32;
  constexpr int H = C / 64;
  std::vector<TensorSpec> result{
      {"emb.weight", {V, C}, false, 0.0f},
      {"ln_out.weight", {C}, false, 1.0f},
      {"ln_out.bias", {C}, false, 0.0f},
      {"head.weight", {V, C}, true, 0.0f},
  };
  const auto add = [&](const char *suffix, std::vector<std::int64_t> shape,
                       bool int4 = false, float fill = 0.0f) {
    result.push_back(
        {std::string("blocks.0.") + suffix, std::move(shape), int4, fill});
  };
  add("ln0.weight", {C}, false, 1.0f);
  add("ln0.bias", {C});
  add("ln1.weight", {C}, false, 1.0f);
  add("ln1.bias", {C});
  add("ln2.weight", {C}, false, 1.0f);
  add("ln2.bias", {C});
  for (const char *name :
       {"att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g"}) {
    add(name, {C});
  }
  for (const char *name : {"att.receptance.weight", "att.key.weight",
                           "att.value.weight", "att.output.weight"}) {
    add(name, {C, C}, true);
  }
  add("att.w0", {C});
  add("att.w1", {C, R});
  add("att.w2", {R, C});
  add("att.a0", {C});
  add("att.a1", {C, R});
  add("att.a2", {R, C});
  add("att.g1", {C, R});
  add("att.g2", {R, C});
  add("att.k_k", {C});
  add("att.k_a", {C});
  add("att.r_k", {H, 64});
  add("att.ln_x.weight", {C}, false, 1.0f);
  add("att.ln_x.bias", {C});
  add("ffn.x_k", {C});
  add("ffn.key.weight", {F, C}, true);
  add("ffn.value.weight", {C, F}, true);
  const auto first_layer = result;
  for (const auto &tensor : first_layer) {
    if (tensor.name.rfind("blocks.0.", 0) != 0 ||
        tensor.name.rfind("blocks.0.ln0.", 0) == 0)
      continue;
    auto next = tensor;
    next.name.replace(0, 9, "blocks.1.");
    result.push_back(std::move(next));
  }
  result.push_back({"blocks.1.att.v0", {C}});
  result.push_back({"blocks.1.att.v1", {C, R}});
  result.push_back({"blocks.1.att.v2", {R, C}});
  return result;
}

} // namespace

int main() {
  if (!rwkv_test::cuda_device_available())
    return 77;
  const auto base = rwkv_test::unique_temp_path("hip_quant_model");
  const auto quant = base.string() + ".rwkvq",
             ref = base.string() + "-ref.rwkvq";
  struct Cleanup {
    std::string a, b;
    ~Cleanup() {
      std::error_code e;
      std::filesystem::remove(a, e);
      std::filesystem::remove(b, e);
    }
  } cleanup{quant, ref};
  for (int bits : {4, 8})
    for (int G : {32, 128}) {
      const auto specs = model_tensors();
      {
        llm_infer::QuantizedWriter qw(quant, specs.size()),
            fw(ref, specs.size());
        for (const auto &spec : specs) {
          append_tensor(qw, spec, bits, G, false);
          append_tensor(fw, spec, bits, G, true);
        }
        TEST_CHECK(qw.close().ok_status());
        TEST_CHECK(fw.close().ok_status());
      }
      for (bool wkv32 : {false, true}) {
        rwkv7_server::ModelBackend qm(quant, wkv32, true, "no-fc", "", false);
        rwkv7_server::ModelBackend fm(ref, wkv32, true, "no-fc", "", false);
        for (int B : {1, 2})
          for (int T : {1, 7, 17, 33}) {
            auto qs = qm.create_state(B), fs = fm.create_state(B);
            std::vector<std::vector<int64_t>> tokens(B,
                                                     std::vector<int64_t>(T));
            for (int b = 0; b < B; ++b)
              for (int t = 0; t < T; ++t)
                tokens[b][t] = (b * 7 + t * 3) % 256;
            rwkv7_server::DeviceLogits ql, fl;
            auto compare = [&]() {
              auto qv =
                  rwkv_test::copy_device_buffer(ql.values, "quant logits");
              auto fv =
                  rwkv_test::copy_device_buffer(fl.values, "reference logits");
              TEST_EQ(qv.size(), fv.size());
              float signal = 0;
              for (std::size_t i = 0; i < qv.size(); ++i) {
                TEST_CHECK(std::isfinite(qv[i]) && std::isfinite(fv[i]));
                signal = std::max(signal, std::abs(fv[i]));
                if (std::abs(qv[i] - fv[i]) > 0.015f + 0.015f * std::abs(fv[i]))
                  throw std::runtime_error("quantized model differs from "
                                           "dequantized FP16 reference");
              }
              TEST_CHECK(signal > 0.001f);
            };
            qm.forward_prefill(tokens, qs, ql);
            fm.forward_prefill(tokens, fs, fl);
            compare();
            for (int step = 0; step < 3; ++step) {
              std::vector<int64_t> next(B, step + 11);
              qm.forward_decode(next, qs, ql);
              fm.forward_decode(next, fs, fl);
              compare();
            }
          }
      }
    }
  std::cout << "W4/W8 nonzero model prefill/decode matches dequantized FP16 "
               "reference\n";
}
