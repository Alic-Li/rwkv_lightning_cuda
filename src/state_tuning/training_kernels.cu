#include "rwkv_state_tuning.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace rwkv7_state_tuning {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, offset));
  }
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ float block_reduce(float value, float *scratch, bool maximum) {
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = maximum ? warp_max(value) : warp_sum(value);
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane]
                                          : (maximum ? -FLT_MAX : 0.0f);
  if (warp == 0)
    value = maximum ? warp_max(value) : warp_sum(value);
  if (threadIdx.x == 0)
    scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

__global__ void xent_kernel(int vocab, const half *__restrict__ logits,
                            const int *__restrict__ targets, int ignore_index,
                            float scale, float *__restrict__ loss,
                            half *__restrict__ d_logits) {
  const int row = blockIdx.x;
  const int target = targets[row];
  const long long base = static_cast<long long>(row) * vocab;
  __shared__ float scratch[8];
  float local_max = -FLT_MAX;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    local_max = fmaxf(local_max, __half2float(logits[base + token]));
  }
  const float max_logit = block_reduce(local_max, scratch, true);
  float local_sum = 0.0f;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    local_sum += expf(__half2float(logits[base + token]) - max_logit);
  }
  const float sum = block_reduce(local_sum, scratch, false);
  const bool ignored = target == ignore_index;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    const float probability =
        expf(__half2float(logits[base + token]) - max_logit) / sum;
    const float gradient =
        ignored ? 0.0f
                : (probability - (token == target ? 1.0f : 0.0f)) * scale;
    d_logits[base + token] = __float2half_rn(gradient);
  }
  if (threadIdx.x == 0 && !ignored) {
    atomicAdd(loss,
              (logf(sum) + max_logit - __half2float(logits[base + target])) *
                  scale);
  }
}

__global__ void reduce_state_kernel(int B, std::size_t lane,
                                    const float *__restrict__ source,
                                    float *__restrict__ destination,
                                    bool accumulate) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= lane)
    return;
  float value = accumulate ? destination[i] : 0.0f;
  for (int batch = 0; batch < B; ++batch) {
    value += source[static_cast<std::size_t>(batch) * lane + i];
  }
  destination[i] = value;
}

__global__ void adam_kernel(float *__restrict__ parameter,
                            float *__restrict__ gradient,
                            float *__restrict__ moment1,
                            float *__restrict__ moment2, std::size_t elements,
                            float lr, float beta1, float beta2, float inv_bias1,
                            float inv_bias2, float epsilon, float weight_decay,
                            bool zero_gradient) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  float g = gradient[i];
  if (weight_decay != 0.0f)
    g += weight_decay * parameter[i];
  const float m = beta1 * moment1[i] + (1.0f - beta1) * g;
  const float v = beta2 * moment2[i] + (1.0f - beta2) * g * g;
  moment1[i] = m;
  moment2[i] = v;
  parameter[i] -= lr * (m * inv_bias1) / (sqrtf(v * inv_bias2) + epsilon);
  if (zero_gradient)
    gradient[i] = 0.0f;
}

int blocks(std::size_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

} // namespace

void cross_entropy_forward_backward_f16(cudaStream_t stream, int rows,
                                        int vocab, const half *logits,
                                        const int *targets, int ignore_index,
                                        float *loss, half *d_logits,
                                        float gradient_scale) {
  if (rows <= 0 || vocab <= 0 || !logits || !targets || !loss || !d_logits) {
    throw std::invalid_argument("invalid cross entropy arguments");
  }
  int valid = rows;
  if (ignore_index >= 0) {
    // The normal state-tuning path uses dense labels. Keeping ignore support
    // exact would require a tiny count reduction, so use a count supplied by
    // the caller only when no ignore index is active.
    throw std::invalid_argument(
        "ignore_index is not supported by the first exact short-context path");
  }
  const cudaError_t error = cudaMemsetAsync(loss, 0, sizeof(float), stream);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  xent_kernel<<<rows, kThreads, 0, stream>>>(
      vocab, logits, targets, ignore_index, gradient_scale / valid, loss,
      d_logits);
}

void reduce_state_gradient_f32(cudaStream_t stream, int batch, int heads,
                               int head_size, const float *per_batch_gradient,
                               float *state_gradient, bool accumulate) {
  if (batch <= 0 || heads <= 0 || head_size <= 0 || !per_batch_gradient ||
      !state_gradient) {
    throw std::invalid_argument("invalid state gradient reduction arguments");
  }
  const std::size_t lane =
      static_cast<std::size_t>(heads) * head_size * head_size;
  reduce_state_kernel<<<blocks(lane), kThreads, 0, stream>>>(
      batch, lane, per_batch_gradient, state_gradient, accumulate);
}

void adam_update_state_f32(cudaStream_t stream, float *time_state,
                           float *gradient, float *moment1, float *moment2,
                           std::size_t elements, std::uint64_t step,
                           const AdamConfig &config, bool zero_gradient) {
  if (!time_state || !gradient || !moment1 || !moment2 || elements == 0 ||
      step == 0 || config.learning_rate <= 0.0f || config.beta1 < 0.0f ||
      config.beta1 >= 1.0f || config.beta2 < 0.0f || config.beta2 >= 1.0f ||
      config.epsilon <= 0.0f) {
    throw std::invalid_argument("invalid state Adam arguments");
  }
  const float inv_bias1 =
      1.0f / (1.0f - std::pow(config.beta1, static_cast<float>(step)));
  const float inv_bias2 =
      1.0f / (1.0f - std::pow(config.beta2, static_cast<float>(step)));
  adam_kernel<<<blocks(elements), kThreads, 0, stream>>>(
      time_state, gradient, moment1, moment2, elements, config.learning_rate,
      config.beta1, config.beta2, inv_bias1, inv_bias2, config.epsilon,
      config.weight_decay, zero_gradient);
}

} // namespace rwkv7_state_tuning
