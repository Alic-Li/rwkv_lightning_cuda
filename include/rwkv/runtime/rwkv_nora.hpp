#pragma once
#include "rwkv/runtime/rwkv_miss.hpp"
namespace rwkv7_nora {
// Shared adapter view: d is NoRA B[out,rank], a is A[rank,in].
void forward(cudaStream_t, int rows, const rwkv7_miss::Linear &, const half *x,
             half *y, float *latent = nullptr);
} // namespace rwkv7_nora
