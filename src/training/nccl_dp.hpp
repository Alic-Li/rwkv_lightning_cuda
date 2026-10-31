#pragma once
#include "rwkv/bf16_training/runtime.hpp"
#include <nccl.h>
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
  void barrier(cudaStream_t stream);

private:
  ncclComm_t comm_ = nullptr;
  rwkv_bf16_training::DeviceBuffer<float> scalar_;
};
// Deterministic longest-first assignment balancing tokens and chunk sweeps.
// Every rank sees the same global batch; only the assigned samples execute on
// GPU.
std::vector<int> assign(const std::vector<size_t> &lengths, int ranks,
                        int micro_bsz);
} // namespace rwkv_dp
