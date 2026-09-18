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
  float *gradient = nullptr; // optional, FP32 accumulator; never a base dW
};
using Block = std::array<Linear, TargetCount>;
// reduced may be null for inference (fused reduction/project/add), otherwise
// contains rows*rank FP32 values retained for backward/recomputed per chunk.
void forward(cudaStream_t, int rows, const Linear &, const half *x, half *y,
             float *reduced = nullptr);
// Adds adapter contribution to the already-computed frozen-base dx.
void backward(cudaStream_t, int rows, const Linear &, const float *reduced,
              const half *dy, half *dx);
void cast_master(cudaStream_t, const float *, half *, std::size_t);
} // namespace rwkv7_miss
