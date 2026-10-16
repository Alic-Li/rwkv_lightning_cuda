#pragma once

#include <cstddef>
#include <cstdint>

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

namespace rwkv_sampling {

// One independent configuration per live row; uploaded once per sampling step.
struct RowSamplingOptions {
  float presence, frequency, decay, log2_inv_temperature;
  int top_k;
  float top_p;
  int force_reasoning_mask;
};

cudaError_t sample_rows_raw(
    const float* logits, float* penalties, int* outputs, void* states,
    float* probs, int batch_size, int vocab_size,
    const RowSamplingOptions* options, cudaStream_t stream);

std::size_t rand_state_bytes(int batch_size);

cudaError_t setup_rand_raw(
    void* states,
    std::uint64_t seed,
    int batch_size,
    cudaStream_t stream);

// logits is [B,T,V]; samples only the last time step. probs and penalties
// each contain B*V floats, outputs B ints. Concurrent calls need separate
// outputs, scratch, penalties and RNG states.
cudaError_t batch_sampling_repetition_temperature_topk_topp_raw(
    const float* logits,
    float* penalties,
    int* outputs,
    void* states,
    float* probs,
    int batch_size,
    int time_steps,
    int vocab_size,
    double presence_penalty,
    double repetition_penalty,
    double penalty_decay,
    double temperature,
    int top_k,
    double top_p,
    cudaStream_t stream);

// logits is [B,T,V], probs is [B,V], outputs is [B].
cudaError_t batch_sampling_temperature_topk_topp_raw(
    const float* logits,
    int* outputs,
    void* states,
    float* probs,
    int batch_size,
    int time_steps,
    int vocab_size,
    double temperature,
    int top_k,
    double top_p,
    cudaStream_t stream);

}  // namespace rwkv_sampling
