#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "rwkv7_fast_v4_common.hpp"

namespace rwkv7_server {

struct GenerateOptions {
  int max_tokens = 1024;
  std::vector<int64_t> stop_tokens{0, 261, 24281};
  double temperature = 1.0;
  int top_k = 20;
  double top_p = 0.6;
  double alpha_presence = 1.0;
  double alpha_frequency = 0.1;
  double alpha_decay = 0.996;
  bool pad_zero = true;
};

struct GenerationState {
  int batch_size = 0;
  rwkv7_fast_v4::DeviceBuffer<half> shift;
  rwkv7_fast_v4::DeviceBuffer<half> wkv_state;
  rwkv7_fast_v4::DeviceBuffer<int> elapsed;

  GenerationState() = default;
  GenerationState(const GenerationState&) = delete;
  GenerationState& operator=(const GenerationState&) = delete;
  GenerationState(GenerationState&&) noexcept = default;
  GenerationState& operator=(GenerationState&&) noexcept = default;
};

class ModelBackend {
 public:
  explicit ModelBackend(std::string model_path);
  ~ModelBackend();

  ModelBackend(const ModelBackend&) = delete;
  ModelBackend& operator=(const ModelBackend&) = delete;
  ModelBackend(ModelBackend&&) noexcept;
  ModelBackend& operator=(ModelBackend&&) noexcept;

  GenerationState create_state(int batch_size) const;
  std::vector<float> forward_prefill(
      const std::vector<std::vector<int64_t>>& token_batches,
      GenerationState& state) const;
  std::vector<float> forward_decode(
      const std::vector<int64_t>& token_batch,
      GenerationState& state) const;

  int vocab_size() const;
  const std::string& model_path() const;
  const std::string& model_name() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rwkv7_server
