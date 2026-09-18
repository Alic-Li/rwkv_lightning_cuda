#pragma once
#include "runtime.hpp"
#include <array>
#include <cstddef>

namespace rwkv_bf16_miss {
// Original logical layout D[out,rank]. Ragged last blocks are zero padded:
// S[t,j] = sum_{k < in, k % rank == j} X[t,k]. No fixed A is stored.
enum Target { Receptance, Key, Value, Output, FfnKey, FfnValue, TargetCount };
struct Linear {
  int in = 0, out = 0, rank = 0;
  float scale = 1;
  const bf16 *d = nullptr;
  float *gradient = nullptr; // optional, FP32 accumulator; never a base dW
};
using Block = std::array<Linear, TargetCount>;
// reduced may be null for inference (fused reduction/project/add), otherwise
// contains rows*rank FP32 values retained for backward/recomputed per chunk.
void forward(cudaStream_t, int rows, const Linear &, const bf16 *x, bf16 *y,
             float *reduced = nullptr);
// Adds adapter contribution to the already-computed frozen-base dx.
void backward(cudaStream_t, int rows, const Linear &, const float *reduced,
              const bf16 *dy, bf16 *dx);
void cast_master(cudaStream_t, const float *, bf16 *, std::size_t);
} // namespace rwkv_bf16_miss
