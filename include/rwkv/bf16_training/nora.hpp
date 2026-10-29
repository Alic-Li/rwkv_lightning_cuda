#pragma once
#include "rwkv/bf16_training/training.hpp"
namespace rwkv_bf16_nora {
void forward(cudaStream_t, int, const rwkv_bf16_miss::Linear &, const bf16 *,
             bf16 *, float *);
void backward(cudaStream_t, int, const rwkv_bf16_miss::Linear &, const bf16 *x,
              const float *latent, const bf16 *dy, bf16 *dx,
              float *latent_gradient);
// Reconstructs inputs from the chunk tape plus its incoming shifts. Only one
// block's scratch is live; no rows*in activation per adapter/layer is retained.
void backward_target(cudaStream_t,
                     const rwkv_bf16_training::FrozenBlockWeights &,
                     const rwkv_bf16_training::BlockTapeView &, int target,
                     const bf16 *dy, bf16 *dx);
} // namespace rwkv_bf16_nora
