#pragma once
#include "format.hpp"
#include "training.hpp"
namespace rwkv_bf16_miss {
class Trainer {
public:
  Trainer(rwkv_bf16_training::FrozenModelView &, int rank, float alpha,
          const std::string &targets, int time, Json::Value config,
          cudaStream_t stream);
  void bind_tapes(std::vector<rwkv_bf16_training::BlockTapeView> &);
  void update(float lr, std::uint64_t step);
  void checkpoint(const std::string &, std::uint64_t step, int epoch,
                  std::uint64_t row, const float *initial, std::size_t states);
  void resume(const std::string &, std::uint64_t &step, int &epoch,
              std::uint64_t &row, float *initial, std::size_t states);
  void export_adapter(const std::string &);

private:
  cudaStream_t stream_;
  Json::Value config_;
  int rank_;
  float alpha_;
  std::vector<HostTensor> targets_;
  rwkv_bf16_training::DeviceBuffer<float> master_, gradient_, m_, v_;
  rwkv_bf16_training::DeviceBuffer<bf16> forward_;
  std::vector<std::array<rwkv_bf16_training::DeviceBuffer<float>, TargetCount>>
      reduced_;
};
} // namespace rwkv_bf16_miss
