#pragma once
#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/runtime/rwkv_state_tuning.hpp"
namespace rwkv7_miss {
class Trainer {
public:
  Trainer(rwkv7_state_tuning::FrozenModelView &, int rank, float alpha,
          const std::string &targets, int time, Json::Value config,
          cudaStream_t stream);
  void bind_tapes(std::vector<rwkv7_state_tuning::BlockTapeView> &);
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
  rwkv7_fast_v4::DeviceBuffer<float> master_, gradient_, m_, v_;
  rwkv7_fast_v4::DeviceBuffer<half> forward_;
  std::vector<std::array<rwkv7_fast_v4::DeviceBuffer<float>, TargetCount>>
      reduced_;
};
} // namespace rwkv7_miss
