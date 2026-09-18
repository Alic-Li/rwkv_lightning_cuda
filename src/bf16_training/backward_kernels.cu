// Copied from src/state_tuning; independent BF16 training implementation.
#include "training.hpp"

#include "runtime.hpp"

#include <algorithm>
#include <stdexcept>

#include "ops.hpp"

namespace rwkv_bf16_training {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float *scratch) {
  __syncthreads(); // Previous reduction readers must finish before reuse.
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
  if (warp == 0)
    value = warp_sum(value);
  if (threadIdx.x == 0)
    scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

__global__ void norm_backward_kernel(int group_size, int group_count,
                                     const bf16 *__restrict__ x,
                                     const bf16 *__restrict__ weight,
                                     const bf16 *__restrict__ dy,
                                     bf16 *__restrict__ dx, float eps) {
  const int group = blockIdx.x;
  const int offset = group * group_size;
  const int weight_offset = (group % group_count) * group_size;
  float local_x = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float value = to_float(x[offset + i]);
    local_x += value;
  }
  __shared__ float scratch[8];
  const float sum_x = block_sum(local_x, scratch);
  const float mean = sum_x / group_size;
  float local_x2 = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float centered = to_float(x[offset + i]) - mean;
    local_x2 += centered * centered;
  }
  const float variance = block_sum(local_x2, scratch) / group_size;
  const float rstd = rsqrtf(variance + eps);

  float local_d = 0.0f;
  float local_dxhat_xhat = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float xhat = (to_float(x[offset + i]) - mean) * rstd;
    const float scaled_dy =
        to_float(dy[offset + i]) * to_float(weight[weight_offset + i]);
    local_d += scaled_dy;
    local_dxhat_xhat += scaled_dy * xhat;
  }
  const float sum_d = block_sum(local_d, scratch);
  const float sum_d_xhat = block_sum(local_dxhat_xhat, scratch);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float xhat = (to_float(x[offset + i]) - mean) * rstd;
    const float scaled_dy =
        to_float(dy[offset + i]) * to_float(weight[weight_offset + i]);
    const float value = rstd * (scaled_dy - sum_d / group_size -
                                xhat * sum_d_xhat / group_size);
    dx[offset + i] = to_bf16(value);
  }
}

struct SixPointers {
  const bf16 *p0;
  const bf16 *p1;
  const bf16 *p2;
  const bf16 *p3;
  const bf16 *p4;
  const bf16 *p5;
};

__device__ __forceinline__ const bf16 *select_ptr(SixPointers p, int i) {
  switch (i) {
  case 0:
    return p.p0;
  case 1:
    return p.p1;
  case 2:
    return p.p2;
  case 3:
    return p.p3;
  case 4:
    return p.p4;
  default:
    return p.p5;
  }
}

__global__ void time_mix_backward_kernel(int B, int T, int C, int branches,
                                         SixPointers grads, SixPointers mixes,
                                         bf16 *__restrict__ dx) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long elements = static_cast<long long>(B) * T * C;
  if (idx >= elements)
    return;
  const int c = static_cast<int>(idx % C);
  const int t = static_cast<int>((idx / C) % T);
  float value = 0.0f;
  for (int branch = 0; branch < branches; ++branch) {
    const bf16 *grad = select_ptr(grads, branch);
    const float mix = to_float(select_ptr(mixes, branch)[c]);
    value += to_float(grad[idx]) * (1.0f - mix);
    if (t + 1 < T)
      value += to_float(grad[idx + C]) * mix;
  }
  dx[idx] = to_bf16(value);
}

__global__ void shift_backward_kernel(int B, int T, int C, int branches,
                                      SixPointers grads, SixPointers mixes,
                                      bf16 *__restrict__ d_shift) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long elements = static_cast<long long>(B) * C;
  if (idx >= elements)
    return;
  const int c = static_cast<int>(idx % C);
  const int batch = static_cast<int>(idx / C);
  const long long first = static_cast<long long>(batch) * T * C + c;
  float value = 0.0f;
  for (int branch = 0; branch < branches; ++branch) {
    value += to_float(select_ptr(grads, branch)[first]) *
             to_float(select_ptr(mixes, branch)[c]);
  }
  d_shift[idx] = to_bf16(value);
}

enum class ElementOp { ReluSquare, Sigmoid, Tanh, Add, AddInplace };

__global__ void last_shift_kernel(int T, int C, int count, bf16 *dx,
                                  const bf16 *shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const long long j = (static_cast<long long>(i / C) * T + T - 1) * C + i % C;
    dx[j] = to_bf16(to_float(dx[j]) + to_float(shift[i]));
  }
}

template <ElementOp Op>
__global__ void element_kernel(const bf16 *__restrict__ x,
                               const bf16 *__restrict__ y,
                               bf16 *__restrict__ out, std::size_t elements) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  const float xv = to_float(x[i]);
  const float yv = y ? to_float(y[i]) : 0.0f;
  float result;
  if constexpr (Op == ElementOp::ReluSquare) {
    result = xv > 0.0f ? 2.0f * xv * yv : 0.0f;
  } else if constexpr (Op == ElementOp::Sigmoid) {
    result = xv * (1.0f - xv) * yv;
  } else if constexpr (Op == ElementOp::Tanh) {
    result = (1.0f - xv * xv) * yv;
  } else {
    result = xv + yv;
  }
  out[i] = to_bf16(result);
}

void require_dims(int rows, int in_features, int out_features) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    throw std::invalid_argument(
        "state-tuning linear dimensions must be positive");
  }
}

SixPointers pack(const bf16 *const *pointers, int count) {
  SixPointers result{};
  const bf16 **fields[] = {&result.p0, &result.p1, &result.p2,
                           &result.p3, &result.p4, &result.p5};
  for (int i = 0; i < count; ++i) {
    if (!pointers[i])
      throw std::invalid_argument("null time-mix branch pointer");
    *fields[i] = pointers[i];
  }
  return result;
}

int blocks_for(std::size_t elements) {
  return static_cast<int>((elements + kThreads - 1) / kThreads);
}

} // namespace

void linear_orig_backward_input_bf16(cudaStream_t stream, int rows,
                                     int in_features, int out_features,
                                     const bf16 *dy,
                                     const bf16 *frozen_weight_orig, bf16 *dx) {
  require_dims(rows, in_features, out_features);
  if (!dy || !frozen_weight_orig || !dx)
    throw std::invalid_argument("null linear pointer");
  rwkv7_v3a_linear_bf16_launch(stream, rows, out_features, in_features, dy,
                               frozen_weight_orig, dx);
}

void add_last_shift_gradient(cudaStream_t stream, int batch, int time,
                             int channels, bf16 *dx, const bf16 *dshift) {
  if (dshift)
    last_shift_kernel<<<blocks_for(batch * channels), kThreads, 0, stream>>>(
        time, channels, batch * channels, dx, dshift);
}

void linear_runtime_backward_input_bf16(cudaStream_t stream, int rows,
                                        int in_features, int out_features,
                                        const bf16 *dy,
                                        const bf16 *frozen_weight_runtime,
                                        bf16 *dx) {
  require_dims(rows, in_features, out_features);
  if (!dy || !frozen_weight_runtime || !dx)
    throw std::invalid_argument("null linear pointer");
  rwkv7_v3a_linear_bf16_orig_launch(stream, rows, out_features, in_features, dy,
                                    frozen_weight_runtime, dx);
}

void layer_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    const bf16 *x, const bf16 *frozen_weight,
                                    const bf16 *dy, bf16 *dx, float eps) {
  if (rows <= 0 || channels <= 0 || !x || !frozen_weight || !dy || !dx ||
      eps <= 0.0f) {
    throw std::invalid_argument("invalid layer-norm backward arguments");
  }
  norm_backward_kernel<<<rows, kThreads, 0, stream>>>(
      channels, 1, x, frozen_weight, dy, dx, eps);
}

void group_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    int groups, const bf16 *x,
                                    const bf16 *frozen_weight, const bf16 *dy,
                                    bf16 *dx, float eps) {
  if (rows <= 0 || channels <= 0 || groups <= 0 || channels % groups != 0 ||
      !x || !frozen_weight || !dy || !dx || eps <= 0.0f) {
    throw std::invalid_argument("invalid group-norm backward arguments");
  }
  const int group_size = channels / groups;
  norm_backward_kernel<<<rows * groups, kThreads, 0, stream>>>(
      group_size, groups, x, frozen_weight, dy, dx, eps);
}

void time_mix_backward_input_bf16(cudaStream_t stream, int batch, int time,
                                  int channels, int branches,
                                  const bf16 *const *branch_grads,
                                  const bf16 *const *frozen_mixes, bf16 *dx,
                                  bf16 *d_shift) {
  if (batch <= 0 || time <= 0 || channels <= 0 || branches <= 0 ||
      branches > 6 || !branch_grads || !frozen_mixes || !dx) {
    throw std::invalid_argument("invalid time-mix backward arguments");
  }
  const SixPointers grads = pack(branch_grads, branches);
  const SixPointers mixes = pack(frozen_mixes, branches);
  const std::size_t elements =
      static_cast<std::size_t>(batch) * time * channels;
  time_mix_backward_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      batch, time, channels, branches, grads, mixes, dx);
  if (d_shift) {
    const std::size_t shift_elements =
        static_cast<std::size_t>(batch) * channels;
    shift_backward_kernel<<<blocks_for(shift_elements), kThreads, 0, stream>>>(
        batch, time, channels, branches, grads, mixes, d_shift);
  }
}

void relu_square_backward_input_bf16(cudaStream_t stream, const bf16 *x,
                                     const bf16 *dy, bf16 *dx,
                                     std::size_t elements) {
  element_kernel<ElementOp::ReluSquare>
      <<<blocks_for(elements), kThreads, 0, stream>>>(x, dy, dx, elements);
}

void sigmoid_backward_input_bf16(cudaStream_t stream, const bf16 *sigmoid_y,
                                 const bf16 *dy, bf16 *dx,
                                 std::size_t elements) {
  element_kernel<ElementOp::Sigmoid>
      <<<blocks_for(elements), kThreads, 0, stream>>>(sigmoid_y, dy, dx,
                                                      elements);
}

void tanh_backward_input_bf16(cudaStream_t stream, const bf16 *tanh_y,
                              const bf16 *dy, bf16 *dx, std::size_t elements) {
  element_kernel<ElementOp::Tanh>
      <<<blocks_for(elements), kThreads, 0, stream>>>(tanh_y, dy, dx, elements);
}

void add_bf16(cudaStream_t stream, const bf16 *a, const bf16 *b, bf16 *out,
              std::size_t elements) {
  element_kernel<ElementOp::Add>
      <<<blocks_for(elements), kThreads, 0, stream>>>(a, b, out, elements);
}

void add_inplace_bf16(cudaStream_t stream, bf16 *destination,
                      const bf16 *source, std::size_t elements) {
  element_kernel<ElementOp::AddInplace>
      <<<blocks_for(elements), kThreads, 0, stream>>>(destination, source,
                                                      destination, elements);
}

} // namespace rwkv_bf16_training
