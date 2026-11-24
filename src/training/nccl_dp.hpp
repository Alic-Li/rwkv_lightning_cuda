#pragma once
#include "rwkv/bf16_training/runtime.hpp"
#ifdef RWKV_USE_HIP
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif
#include <string>
#include <vector>

namespace rwkv_dp {
// Linux single-node launcher: fork BEFORE any CUDA initialization. Each child
// owns one device/context/model; the parent supervises and reaps every rank.
// Returns -1 in workers, otherwise the aggregate exit status in the parent.
int launch(const std::string &devices);
int rank();
int size();
class Group {
public:
  Group();
  ~Group();
  Group(const Group &) = delete;
  Group &operator=(const Group &) = delete;
  void sum(float *values, size_t count, cudaStream_t stream);
  float sum_loss(float value, cudaStream_t stream);
  float sum_loss_and_times(float local_loss, float local_compute_s,
                           std::vector<float> &rank_compute_s,
                           cudaStream_t stream);
  void barrier(cudaStream_t stream);

private:
  ncclComm_t comm_ = nullptr;
  rwkv_bf16_training::DeviceBuffer<float> scalar_;
};
// Deterministic longest-first assignment balancing tokens, chunk sweeps, and
// optional measured per-rank inverse speed factors (relative sec/unit-load).
// Every rank sees the same global batch and same inv_speeds, so assignments are
// bit-identical across ranks.
std::vector<int> assign(const std::vector<size_t> &lengths, int ranks,
                        int micro_bsz,
                        const std::vector<double> *inv_speeds = nullptr,
                        std::vector<size_t> *out_raw_loads = nullptr);
void update_speed_model(const std::vector<size_t> &raw_loads,
                        const std::vector<float> &rank_compute_s,
                        std::vector<double> &inv_speeds, bool &ready);
} // namespace rwkv_dp
