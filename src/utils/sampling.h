#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace rwkv_sampling {

std::size_t rand_state_bytes(int batch_size);

cudaError_t setup_rand_raw(
    void* states,
    std::uint64_t seed,
    int batch_size,
    cudaStream_t stream);

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

}  // namespace rwkv_sampling
