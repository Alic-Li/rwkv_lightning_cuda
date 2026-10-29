#pragma once
#include "rwkv/bf16_training/format.hpp"
#include "rwkv/bf16_training/training.hpp"
namespace rwkv7_miss { struct Package; }
namespace rwkv_bf16_miss {
class Trainer {
public:
  Trainer(rwkv_bf16_training::FrozenModelView &, int rank, float alpha,
          const std::string &targets, int time, Json::Value config,
          cudaStream_t stream);
  void bind_tapes(std::vector<rwkv_bf16_training::BlockTapeView> &);
  double update(float lr, std::uint64_t step);
  void zero_gradient();
  // Load validated package into the existing FP32 master/BF16 forward weights.
  // Cold starts intentionally zero Adam moments; resume() restores optimizer state.
  void load_adapter(const rwkv7_miss::Package &);
  // Only adapter tensors are copied. Frozen backbone pointers remain shared.
  rwkv_bf16_training::DeviceBuffer<bf16>
  snapshot(rwkv_bf16_training::FrozenModelView &) const;
  void checkpoint(const std::string &, std::uint64_t step, int epoch,
                  std::uint64_t row, const float *initial, std::size_t states);
  void resume(const std::string &, std::uint64_t &step, int &epoch,
              std::uint64_t &row, float *initial, std::size_t states);
  void export_adapter(const std::string &);

private:
  cudaStream_t stream_;
  // Missing optional-key reads must not change later checkpoint configs.
  const Json::Value config_;
  bool nora_;
  int rank_;
  float alpha_;
  std::vector<HostTensor> targets_;
  rwkv_bf16_training::DeviceBuffer<float> master_, gradient_, m_, v_;
  rwkv_bf16_training::DeviceBuffer<bf16> forward_;
  std::vector<std::array<rwkv_bf16_training::DeviceBuffer<float>, TargetCount>>
      reduced_;
};
} // namespace rwkv_bf16_miss
