#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"

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
  std::string adapter_id;
  std::shared_ptr<const rwkv7_miss::AdapterHandle> adapter;
  int max_tokens = 8192;
  std::vector<int64_t> stop_tokens{0, 261, 24281};
  // Presence selects text matching, including an explicitly empty list.
  std::optional<std::vector<std::string>> stop;
  double temperature = 1.0;
  int top_k = 20;
  double top_p = 0.3;
  double alpha_presence = 2.0;
  double alpha_frequency = 0.2;
  double alpha_decay = 0.996;
  bool force_reasoning = false;
  int force_reasoning_token_offset = 0;
};

// Adapter selection for a packed continuous-decode batch. Rows are independent
// requests, so each row names its own GPU lease (null = base weights) and the
// backend applies every adapter in one pass over the shared base forward.
struct RowAdapters {
  std::vector<std::shared_ptr<const rwkv7_miss::GpuLease>> leases;
  std::vector<float> scales;
  // Set whenever leases/scales change; the backend then refreshes its device
  // mirror (row -> slot map and per-layer/target slot table) once.
  bool dirty = true;
  int slots = 0;
  int max_rank = 0;
  std::vector<std::uint8_t> present;  // [layer * TargetCount + target]
  rwkv7_fast_v4::DeviceBuffer<int> row_slot;
  rwkv7_fast_v4::DeviceBuffer<rwkv7_miss::Linear> table;
};

struct GenerationState {
  std::shared_ptr<const rwkv7_miss::AdapterHandle> adapter;
  std::shared_ptr<const rwkv7_miss::GpuLease> adapter_gpu;
  std::string effective_key;
  std::string adapter_identity;
  std::string initial_state_identity = "zero";
  bool advanced = false;
  // Bounded decoded suffix retained across pause/resume of logit generation.
  std::string text_stop_tail;
  bool adapter_cold = false;
  double adapter_h2d_ms = 0;
  std::chrono::steady_clock::time_point request_started{};
  int batch_size = 0;
  // Physical rows per layer block of the device buffers; 0 means batch_size.
  // A packed decode group keeps a larger capacity so rows stay in their slots
  // while the live row count changes.
  int row_capacity = 0;
  bool wkv32 = false;
  rwkv7_fast_v4::DeviceBuffer<half> shift;
  rwkv7_fast_v4::DeviceBuffer<half> wkv_state;
  rwkv7_fast_v4::DeviceBuffer<float> wkv_state32;
  rwkv7_fast_v4::DeviceBuffer<int> elapsed;
  // Packed decode groups only; when set it replaces `adapter` in forward.
  std::shared_ptr<RowAdapters> row_adapters;

  GenerationState() = default;
  GenerationState(const GenerationState&) = delete;
  GenerationState& operator=(const GenerationState&) = delete;
  GenerationState(GenerationState&&) noexcept = default;
  GenerationState& operator=(GenerationState&&) noexcept = default;
};

inline int state_row_stride(const GenerationState& state) {
  return state.row_capacity > 0 ? state.row_capacity : state.batch_size;
}

// One device-to-device copy; a null source zero-fills the destination.
struct DeviceCopy {
  const void* src = nullptr;
  void* dst = nullptr;
  std::size_t bytes = 0;
};

// Runs each phase after the previous one and waits for completion. Copies in
// one phase may run concurrently, so they must not overlap. This portable
// fallback issues one synchronous call per copy.
inline void run_device_copies_serially(const std::vector<std::vector<DeviceCopy>>& phases) {
  for (const auto& phase : phases) {
    for (const auto& copy : phase) {
      if (copy.bytes == 0) continue;
      rwkv7_fast_v4::check_cuda(
          copy.src ? cudaMemcpy(copy.dst, copy.src, copy.bytes, cudaMemcpyDeviceToDevice)
                   : cudaMemset(copy.dst, 0, copy.bytes),
          "run device copy");
    }
  }
}

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
  // Validate the bound adapter against this model and make it GPU resident.
  // Callers run this on the request thread so a cold H2D upload, cudaMalloc
  // or base-model hashing never stalls the shared decode loop.
  virtual void prepare_adapter(GenerationState&) const {}
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
  // Appends the copies that move one recurrent-state row (no identity
  // metadata) for a later run_device_copies. The default copies immediately
  // through copy_state_slice, which keeps only the order among state rows;
  // callers preserve the destination metadata themselves.
  virtual void append_state_row_copies(
      const GenerationState& src, int src_row,
      GenerationState& dst, int dst_row,
      std::vector<DeviceCopy>&) const {
    copy_state_slice(src, src_row, dst, dst_row, 1);
  }
  virtual void run_device_copies(const std::vector<std::vector<DeviceCopy>>& phases) const {
    run_device_copies_serially(phases);
  }
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
  void prepare_adapter(GenerationState& state) const override;
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
  void append_state_row_copies(
      const GenerationState& src, int src_row,
      GenerationState& dst, int dst_row,
      std::vector<DeviceCopy>& out) const override;
  void run_device_copies(const std::vector<std::vector<DeviceCopy>>& phases) const override;
  PrefillCapacity query_prefill_capacity(int prefill_chunk_size) const override;

  int vocab_size() const override;
  const std::string& model_path() const override;
  const std::string& model_name() const override;
  std::string runtime_identity() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rwkv7_server
