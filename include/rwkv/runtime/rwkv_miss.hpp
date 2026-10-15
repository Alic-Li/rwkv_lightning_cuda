#pragma once
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <array>
#include <cstddef>

namespace rwkv7_miss {
// Original logical layout D[out,rank]. Ragged last blocks are zero padded:
// S[t,j] = sum_{k < in, k % rank == j} X[t,k]. No fixed A is stored.
enum Target { Receptance, Key, Value, Output, FfnKey, FfnValue, TargetCount };
struct Linear {
  int in = 0, out = 0, rank = 0;
  float scale = 1;
  const half *d = nullptr;
};
using Block = std::array<Linear, TargetCount>;
// Inference uses fused reduction/project/add when reduced is null.
// Optional rows*rank FP32 scratch is retained for forward diagnostics.
void forward(cudaStream_t, int rows, const Linear &, const half *x, half *y,
             float *reduced = nullptr);
} // namespace rwkv7_miss
