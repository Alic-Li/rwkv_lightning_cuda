#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"
#include "rwkv/runtime/rwkv_state_tuning.hpp"

namespace rwkv7_server {

struct PrefillCapacity {
  std::size_t free_vram_bytes = 0;
  std::size_t total_vram_bytes = 0;
  std::size_t reserve_vram_bytes = 0;
  std::size_t bytes_per_batch = 0;
  int max_batch_size = 0;
};

enum class ThinkType {
  None,
  Fast,
  Free,
  PreferChinese,
  En,
  EnShort,
  EnLong,
};

struct GenerateOptions {
  std::shared_ptr<const rwkv7_miss::AdapterHandle> adapter;
  int max_tokens = 8192;
  std::vector<int64_t> stop_tokens{0, 261, 24281};
  double temperature = 1.0;
  int top_k = 20;
  double top_p = 0.3;
  double alpha_presence = 2.0;
  double alpha_frequency = 0.2;
  double alpha_decay = 0.996;
  bool force_reasoning = false;
  int force_reasoning_token_offset = 0;
};

struct GenerationState {
  std::shared_ptr<const rwkv7_miss::AdapterHandle> adapter;
  std::shared_ptr<const rwkv7_miss::GpuLease> adapter_gpu;
  std::string effective_key;
  std::string adapter_identity;
  std::string initial_state_identity = "zero";
  bool advanced = false;
  bool adapter_cold = false;
  double adapter_h2d_ms = 0;
  std::size_t observed_device_vram_peak = 0;
  std::chrono::steady_clock::time_point request_started{};
  int batch_size = 0;
  bool wkv32 = false;
  rwkv7_fast_v4::DeviceBuffer<half> shift;
  rwkv7_fast_v4::DeviceBuffer<half> wkv_state;
  rwkv7_fast_v4::DeviceBuffer<float> wkv_state32;
  rwkv7_fast_v4::DeviceBuffer<int> elapsed;

  GenerationState() = default;
  GenerationState(const GenerationState&) = delete;
  GenerationState& operator=(const GenerationState&) = delete;
  GenerationState(GenerationState&&) noexcept = default;
  GenerationState& operator=(GenerationState&&) noexcept = default;
};

inline void
bind_adapter(GenerationState &state,
             const std::shared_ptr<const rwkv7_miss::AdapterHandle> &adapter) {
  const auto id = adapter ? adapter->identity : std::string{};
  if (state.advanced && state.adapter_identity != id)
    throw std::runtime_error("state belongs to another adapter/version/scale");
  if (state.adapter_identity != id)
    state.adapter_gpu.reset();
  state.adapter = adapter;
  state.adapter_identity = id;
}

struct DeviceLogits {
  int rows = 0;
  int vocab_size = 0;
  rwkv7_fast_v4::DeviceBuffer<float> values;
  // Backend-owned scratch follows the logits/request lifetime. This keeps
  // prefill/decode reuse without leaving a high-water allocation per worker.
  std::shared_ptr<void> backend_workspace;

  DeviceLogits() = default;
  DeviceLogits(const DeviceLogits&) = delete;
  DeviceLogits& operator=(const DeviceLogits&) = delete;
  DeviceLogits(DeviceLogits&&) noexcept = default;
  DeviceLogits& operator=(DeviceLogits&&) noexcept = default;
};

class IModelBackend {
 public:
  virtual ~IModelBackend() = default;

  virtual GenerationState create_state(int batch_size) const = 0;
  // Load a serialized PyTorch state. Backends that do not support this format
  // retain the normal zero-initialized state behavior only when no path is set.
  virtual GenerationState load_state_from_pth(const std::string&, int batch_size) const {
    throw std::runtime_error("PTH state loading is not supported by this backend");
  }
  virtual void forward_prefill(
      const std::vector<std::vector<int64_t>>& token_batches,
      GenerationState& state,
      DeviceLogits& logits) const = 0;
  virtual void forward_decode(
      const std::vector<int64_t>& token_batch,
      GenerationState& state,
      DeviceLogits& logits) const = 0;
  virtual void copy_state_slice(
      const GenerationState& src,
      int src_offset,
      GenerationState& dst,
      int dst_offset,
      int count) const = 0;
  virtual void copy_logits_slice(
      const DeviceLogits& src,
      int src_offset,
      DeviceLogits& dst,
      int dst_offset,
      int count) const = 0;
  virtual PrefillCapacity query_prefill_capacity(int prefill_chunk_size) const = 0;

  virtual int vocab_size() const = 0;
  virtual const std::string& model_path() const = 0;
  virtual const std::string& model_name() const = 0;
  virtual std::string runtime_identity() const { return model_path(); }
};

class ModelBackend final : public IModelBackend {
 public:
  explicit ModelBackend(
      std::string model_path,
      bool use_wkv32 = false,
      bool chunk_load = false,
      std::string cmix_sparse = "no-fc",
      std::string tune_cache = {},
      bool retune = false,
      std::string tune_cache_directory = {});
  ~ModelBackend() override;

  ModelBackend(const ModelBackend&) = delete;
  ModelBackend& operator=(const ModelBackend&) = delete;
  ModelBackend(ModelBackend&&) noexcept;
  ModelBackend& operator=(ModelBackend&&) noexcept;

  GenerationState create_state(int batch_size) const override;
  GenerationState load_state_from_pth(const std::string& path, int batch_size) const override;
  void forward_prefill(
      const std::vector<std::vector<int64_t>>& token_batches,
      GenerationState& state,
      DeviceLogits& logits) const override;
  void forward_decode(
      const std::vector<int64_t>& token_batch,
      GenerationState& state,
      DeviceLogits& logits) const override;
  void copy_state_slice(
      const GenerationState& src,
      int src_offset,
      GenerationState& dst,
      int dst_offset,
      int count) const override;
  void copy_logits_slice(
      const DeviceLogits& src,
      int src_offset,
      DeviceLogits& dst,
      int dst_offset,
      int count) const override;
  PrefillCapacity query_prefill_capacity(int prefill_chunk_size) const override;

  int vocab_size() const override;
  const std::string& model_path() const override;
  const std::string& model_name() const override;
  std::string runtime_identity() const override;
  const std::string &base_fingerprint() const;

  // Read-only FP16 views for the optional state-tuning sidecar. The returned
  // pointers remain owned by this backend and are valid for its lifetime.
  rwkv7_state_tuning::FrozenModelView state_tuning_model_view() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rwkv7_server
