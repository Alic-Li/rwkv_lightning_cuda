#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rwkv_quantized.hpp"
#include "rwkv_server_backend.hpp"
#include "test_common.hpp"

namespace {

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;
  bool int4 = false;
  float fill = 0.0f;
};

std::size_t elements(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const auto dim : shape) count *= static_cast<std::size_t>(dim);
  return count;
}

void append_tensor(llm_infer::QuantizedWriter& writer, const TensorSpec& spec) {
  const std::size_t count = elements(spec.shape);
  if (spec.int4) {
    const std::size_t rows = static_cast<std::size_t>(spec.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(spec.shape[1]);
    std::vector<std::uint8_t> data(rows * ((cols + 1) / 2), 0);
    std::vector<std::uint16_t> scales(rows * ((cols + 127) / 128), 0x3c00);
    TEST_CHECK(writer.append(spec.name, llm_infer::QuantizedDType::kInt4, spec.shape, data, scales, 128)
                   .ok_status());
    return;
  }
  const std::uint16_t bits = llm_infer::float_to_bf16_bits(spec.fill);
  std::vector<std::uint8_t> data(count * 2);
  for (std::size_t i = 0; i < count; ++i) {
    data[i * 2] = static_cast<std::uint8_t>(bits);
    data[i * 2 + 1] = static_cast<std::uint8_t>(bits >> 8);
  }
  TEST_CHECK(writer.append(spec.name, llm_infer::QuantizedDType::kBFloat16, spec.shape, data).ok_status());
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
  const auto add = [&](const char* suffix, std::vector<std::int64_t> shape, bool int4 = false,
                       float fill = 0.0f) {
    result.push_back({std::string("blocks.0.") + suffix, std::move(shape), int4, fill});
  };
  add("ln0.weight", {C}, false, 1.0f);
  add("ln0.bias", {C});
  add("ln1.weight", {C}, false, 1.0f);
  add("ln1.bias", {C});
  add("ln2.weight", {C}, false, 1.0f);
  add("ln2.bias", {C});
  for (const char* name : {"att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g"}) {
    add(name, {C});
  }
  for (const char* name :
       {"att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight"}) {
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
  return result;
}

}  // namespace

int main() {
  if (!rwkv_test::cuda_device_available()) return 77;
  std::filesystem::path path = rwkv_test::unique_temp_path("rwkv_w4a16_model");
  path += ".rwkvq";
  try {
    const auto tensors = model_tensors();
    llm_infer::QuantizedWriter writer(path.string(), static_cast<std::uint32_t>(tensors.size()));
    for (const auto& tensor : tensors) append_tensor(writer, tensor);
    TEST_CHECK(writer.close().ok_status());

    auto model = std::make_shared<rwkv7_server::ModelBackend>(path.string(), false, true, "no-fc", "", false);
    auto state = model->create_state(1);
    rwkv7_server::DeviceLogits logits;
    model->forward_prefill({{0}}, state, logits);
    model->forward_decode({0}, state, logits);
    TEST_EQ(logits.rows, 1);
    TEST_EQ(logits.vocab_size, 256);
    const auto values = rwkv_test::copy_device_buffer(logits.values, "copy W4 model logits");
    for (const float value : values) TEST_CHECK(std::isfinite(value));
    std::filesystem::remove(path);
    return 0;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}
