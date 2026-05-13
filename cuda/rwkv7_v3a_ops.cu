#include <assert.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "rwkv7_fast_v4_kernels.cuh"

using dtype = half;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s failed with cublas status %d\n", what, static_cast<int>(status));
    abort();
  }
}

inline cublasHandle_t blas_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h = nullptr;
    check_cublas(cublasCreate(&h), "cublasCreate");
    return h;
  }();
  return handle;
}

inline cublasLtHandle_t blaslt_handle() {
  static cublasLtHandle_t handle = [] {
    cublasLtHandle_t h = nullptr;
    check_cublas(cublasLtCreate(&h), "cublasLtCreate");
    return h;
  }();
  return handle;
}

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

__global__ void bf16_to_f16_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const half h = __float2half_rn(bf16_bits_to_float_dev(src[i]));
    dst[i] = *reinterpret_cast<const uint16_t*>(&h);
  }
}

__global__ void bf16_to_f16_transpose_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int rows,
    int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) {
    const half h = __float2half_rn(bf16_bits_to_float_dev(src[static_cast<int64_t>(r) * cols + c]));
    dst[static_cast<int64_t>(c) * rows + r] = *reinterpret_cast<const uint16_t*>(&h);
  }
}

__global__ void f16_transpose_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int rows,
    int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols) {
    dst[static_cast<int64_t>(c) * rows + r] = src[static_cast<int64_t>(r) * cols + c];
  }
}

template <int Threads>
__device__ __forceinline__ float block_sum_t(float x) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (Threads / 32)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__global__ void emb_ln0_bf16_to_f16_kernel(
    int V,
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ out,
    float eps) {
  // Two-pass stats: bf16 inputs are converted to fp32 for mean, then reread for variance.
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (tok >= V) {
    return;
  }
  const uint16_t* er = emb + static_cast<int64_t>(tok) * C;
  float sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    sum += bf16_bits_to_float_dev(er[c]);
  }
  const float mean = block_sum_t<256>(sum) / static_cast<float>(C);
  float var = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    const float d = bf16_bits_to_float_dev(er[c]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(block_sum_t<256>(var) / static_cast<float>(C) + eps);
  uint16_t* yr = out + static_cast<int64_t>(tok) * C;
  for (int c = tid; c < C; c += blockDim.x) {
    const float x = bf16_bits_to_float_dev(er[c]);
    const float w = bf16_bits_to_float_dev(weight[c]);
    const float b = bf16_bits_to_float_dev(bias[c]);
    const half y = __float2half_rn((x - mean) * rstd * w + b);
    yr[c] = *reinterpret_cast<const uint16_t*>(&y);
  }
}

__global__ void add_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ y,
    dtype* __restrict__ out,
    int64_t n_pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_pairs) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    const float2 yv = __half22float2(reinterpret_cast<const __half2*>(y)[i]);
    reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(xv.x + yv.x, xv.y + yv.y);
  }
}

__global__ void advance_i32_kernel(int* __restrict__ x, int amount, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += amount;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n = blockIdx.x;
  const int m = blockIdx.y;
  if (m >= M || n >= N) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = weight_t + static_cast<int64_t>(n) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + (k2 << 1)));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + (k2 << 1)));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N)
          ? __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n) * K + k))
          : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + k)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    xv.x = apply_act<Act>(xv.x);
    xv.y = apply_act<Act>(xv.y);
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    int Kv,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ v1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
          *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

__global__ void layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float var = sum_var * inv_c;
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

__global__ void add_layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      sum += v.x + v.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float dx = v.x - mean;
      const float dy = v.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (v.x - mean) * rstd * w.x + b.x,
          (v.y - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(x_out + base)[idx] = __floats2half2_rn(sx, sy);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + b.x,
          (sy - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * LN_SMALL_C;
  const int64_t dst = bidx * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x,
          (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) = __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

} // namespace

void rwkv7_v4_bf16_to_f16_launch(
    cudaStream_t stream, const uint16_t* src_bf16, uint16_t* dst_f16, long long elems) {
  constexpr int threads = 256;
  bf16_to_f16_kernel<<<static_cast<int>(ceil_div(elems, threads)), threads, 0, stream>>>(
      src_bf16, dst_f16, elems);
}

void rwkv7_v4_bf16_to_f16_transpose_launch(
    cudaStream_t stream, const uint16_t* src_bf16, uint16_t* dst_f16, int rows, int cols) {
  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>(ceil_div(cols, block.x)), static_cast<unsigned>(ceil_div(rows, block.y)));
  bf16_to_f16_transpose_kernel<<<grid, block, 0, stream>>>(src_bf16, dst_f16, rows, cols);
}

void rwkv7_v4_f16_transpose_launch(
    cudaStream_t stream, const uint16_t* src_f16, uint16_t* dst_f16, int rows, int cols) {
  dim3 block(16, 16);
  dim3 grid(static_cast<unsigned>(ceil_div(cols, block.x)), static_cast<unsigned>(ceil_div(rows, block.y)));
  f16_transpose_kernel<<<grid, block, 0, stream>>>(src_f16, dst_f16, rows, cols);
}

void rwkv7_v4_emb_ln0_bf16_to_f16_launch(
    cudaStream_t stream, int V, int C,
    const uint16_t* emb_bf16, const uint16_t* weight_bf16, const uint16_t* bias_bf16,
    uint16_t* out_f16, float eps) {
  assert(C == LN_SMALL_C);
  emb_ln0_bf16_to_f16_kernel<<<V, 256, 0, stream>>>(
      V, C, emb_bf16, weight_bf16, bias_bf16, out_f16, eps);
}

void rwkv7_v3a_add_f16_launch(cudaStream_t stream, const half* x, const half* y, half* out, long long elems) {
  assert((elems & 1) == 0);
  constexpr int threads = 256;
  const int64_t pairs = elems >> 1;
  add_f16_kernel<<<static_cast<int>(ceil_div(pairs, threads)), threads, 0, stream>>>(
      x, y, out, pairs);
}

void rwkv7_v3a_advance_i32_launch(cudaStream_t stream, int* x, int amount, long long elems) {
  constexpr int threads = 256;
  advance_i32_kernel<<<static_cast<int>(ceil_div(elems, threads)), threads, 0, stream>>>(
      x, amount, elems);
}

void rwkv7_v3a_layer_norm_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* weight, const half* bias, half* y, float eps) {
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<rows, LN_SMALL512_THREADS, 0, stream>>>(
          x, weight, bias, y, rows, eps);
    } else if (rows >= 512) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<rows, LN_SMALL512_THREADS, 0, stream>>>(
          x, weight, bias, y, rows, eps);
    } else {
      layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<rows, LN_SMALL_THREADS, 0, stream>>>(
          x, weight, bias, y, rows, eps);
    }
    return;
  }
  layer_norm_f16_kernel<<<rows, LN_THREADS, 0, stream>>>(C, x, weight, bias, y, rows, eps);
}

void rwkv7_v3a_add_layer_norm_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* residual, const half* weight, const half* bias,
    half* x_out, half* y, float eps) {
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<rows, LN_SMALL512_THREADS, 0, stream>>>(
          x, residual, weight, bias, x_out, y, rows, eps);
    } else if (rows >= 512) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<rows, LN_SMALL512_THREADS, 0, stream>>>(
          x, residual, weight, bias, x_out, y, rows, eps);
    } else {
      add_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<rows, LN_SMALL_THREADS, 0, stream>>>(
          x, residual, weight, bias, x_out, y, rows, eps);
    }
    return;
  }
  add_layer_norm_f16_kernel<<<rows, LN_THREADS, 0, stream>>>(
      C, x, residual, weight, bias, x_out, y, rows, eps);
}

void rwkv7_v3a_add_last_layer_norm_f16_launch(
    cudaStream_t stream, int B, int T,
    const half* x, const half* residual, const half* weight, const half* bias,
    half* y, float eps) {
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<B, LN_SMALL512_THREADS, 0, stream>>>(
        x, residual, weight, bias, y, B, T, eps);
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<B, LN_SMALL512_THREADS, 0, stream>>>(
        x, residual, weight, bias, y, B, T, eps);
  } else {
    add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<B, LN_SMALL_THREADS, 0, stream>>>(
        x, residual, weight, bias, y, B, T, eps);
  }
}

void rwkv7_v3a_add_layer_norm_cmix_mix_f16_launch(
    cudaStream_t stream, int rows,
    const half* x, const half* residual, half* shift_state,
    const half* weight, const half* bias, const half* x_k,
    half* x_out, half* mixed, float eps) {
  add_layer_norm_cmix_mix_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<rows, LN_SMALL_THREADS, 0, stream>>>(
      x, residual, shift_state, weight, bias, x_k, x_out, mixed, rows, eps);
}

void rwkv7_v3a_add_layer_norm_tmix_mix6_f16_launch(
    cudaStream_t stream, int rows,
    const half* x, const half* residual, half* shift_state,
    const half* weight, const half* bias,
    const half* x_r, const half* x_w, const half* x_k,
    const half* x_v, const half* x_a, const half* x_g,
    half* x_out, half* out_r, half* out_w, half* out_k,
    half* out_v, half* out_a, half* out_g, float eps) {
  add_layer_norm_tmix_mix6_f16_scalar_stats_kernel<LN_SMALL_THREADS><<<rows, LN_SMALL_THREADS, 0, stream>>>(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g,
      x_out, out_r, out_w, out_k, out_v, out_a, out_g, rows, eps);
}

void rwkv7_v3a_linear_t_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t, half* y) {
  if (K <= 512 && N >= 1024 && M <= 4) {
    if (M == 1) {
      linear_t_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
          M, K, N, x, weight_t, y);
    } else {
      linear_t_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
          M, K, N, x, weight_t, y);
    }
  } else if (K >= 1024) {
    linear_t_f16_kernel<256><<<dim3(N, M, 1), 256, 0, stream>>>(M, K, N, x, weight_t, y);
  } else {
    linear_t_f16_kernel<128><<<dim3(N, M, 1), 128, 0, stream>>>(M, K, N, x, weight_t, y);
  }
}

void rwkv7_v3a_linear_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight, half* y) {
  if (M == 0 || N == 0 || K == 0) {
    return;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = blas_handle();
  check_cublas(cublasSetStream(handle, stream), "linear_f16 cublasSetStream");
  // Row-major y[M,N] = x[M,K] @ weight[K,N] is column-major
  // y^T[N,M] = weight^T[N,K] @ x^T[K,M].
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      N,
      M,
      K,
      &alpha,
      weight,
      CUDA_R_16F,
      N,
      x,
      CUDA_R_16F,
      K,
      &beta,
      y,
      CUDA_R_16F,
      N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16 cublasGemmEx");
}

void rwkv7_v3a_linear_f16_orig_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig, half* y) {
  if (M == 0 || N == 0 || K == 0) {
    return;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = blas_handle();
  check_cublas(cublasSetStream(handle, stream), "linear_f16_orig cublasSetStream");
  // weight_orig is row-major [N,K], i.e. column-major [K,N].
  // y[M,N] = x[M,K] @ weight_orig[N,K]^T.
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      N,
      M,
      K,
      &alpha,
      weight_orig,
      CUDA_R_16F,
      K,
      x,
      CUDA_R_16F,
      K,
      &beta,
      y,
      CUDA_R_16F,
      N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16_orig cublasGemmEx");
}

void rwkv7_v3a_linear_f16_orig_lt_cfg_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    void* workspace, std::size_t workspace_bytes, int algo_index, half* y) {
  if (M == 0 || N == 0 || K == 0) {
    return;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  cublasLtHandle_t handle = blaslt_handle();
  check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "linear_f16_orig_lt desc");
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "linear_f16_orig_lt transa");
  check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "linear_f16_orig_lt transb");
  check_cublas(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, K, N, K), "linear_f16_orig_lt a layout");
  check_cublas(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, K, M, K), "linear_f16_orig_lt b layout");
  check_cublas(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, N, M, N), "linear_f16_orig_lt c layout");
  check_cublas(cublasLtMatmulPreferenceCreate(&pref), "linear_f16_orig_lt preference");
  check_cublas(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
               "linear_f16_orig_lt workspace");

  cublasLtMatmulHeuristicResult_t heuristics[64];
  int returned = 0;
  check_cublas(cublasLtMatmulAlgoGetHeuristic(
      handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref,
      64, heuristics, &returned), "linear_f16_orig_lt heuristic");
  if (returned <= 0 || algo_index < 0 || algo_index >= returned) {
    fprintf(stderr, "linear_f16_orig_lt invalid algo_index=%d returned=%d\n", algo_index, returned);
    abort();
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasLtMatmul(
      handle,
      op_desc,
      &alpha,
      weight_orig,
      a_desc,
      x,
      b_desc,
      &beta,
      y,
      c_desc,
      y,
      c_desc,
      &heuristics[algo_index].algo,
      workspace,
      workspace_bytes,
      stream),
      "linear_f16_orig_lt matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
}

template <int RowTile, int OutTile>
void linear_orig_rows_launch_impl(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig, half* y) {
  linear_orig_rows_f16_kernel<128, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), 128, 0, stream>>>(
      M, K, N, x, weight_orig, y);
}

template <int Threads, int RowTile, int OutTile>
void linear_orig_rows_cfg_launch_impl(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig, half* y) {
  linear_orig_rows_f16_kernel<Threads, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), Threads, 0, stream>>>(
      M, K, N, x, weight_orig, y);
}

void rwkv7_v3a_linear_orig_rows_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    int row_tile, int out_tile, half* y) {
  if (row_tile == 1 && out_tile == 2) return linear_orig_rows_launch_impl<1, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 1 && out_tile == 4) return linear_orig_rows_launch_impl<1, 4>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 1 && out_tile == 8) return linear_orig_rows_launch_impl<1, 8>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 1 && out_tile == 16) return linear_orig_rows_launch_impl<1, 16>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 2 && out_tile == 2) return linear_orig_rows_launch_impl<2, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 2 && out_tile == 4) return linear_orig_rows_launch_impl<2, 4>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 2 && out_tile == 8) return linear_orig_rows_launch_impl<2, 8>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 3 && out_tile == 2) return linear_orig_rows_launch_impl<3, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 3 && out_tile == 4) return linear_orig_rows_launch_impl<3, 4>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 3 && out_tile == 8) return linear_orig_rows_launch_impl<3, 8>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 4 && out_tile == 2) return linear_orig_rows_launch_impl<4, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 4 && out_tile == 4) return linear_orig_rows_launch_impl<4, 4>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 4 && out_tile == 8) return linear_orig_rows_launch_impl<4, 8>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 8 && out_tile == 2) return linear_orig_rows_launch_impl<8, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 8 && out_tile == 4) return linear_orig_rows_launch_impl<8, 4>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 16 && out_tile == 1) return linear_orig_rows_launch_impl<16, 1>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 16 && out_tile == 2) return linear_orig_rows_launch_impl<16, 2>(stream, M, K, N, x, weight_orig, y);
  if (row_tile == 16 && out_tile == 4) return linear_orig_rows_launch_impl<16, 4>(stream, M, K, N, x, weight_orig, y);
  assert(false && "unsupported linear_orig_rows_f16 row_tile/out_tile");
}

void rwkv7_v3a_linear_orig_rows_cfg_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    int threads, int row_tile, int out_tile, half* y) {
  if (threads == 64 && row_tile == 1 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<64, 1, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<64, 1, 8>(stream, M, K, N, x, weight_orig, y);
  if (threads == 128 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<128, 1, 8>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<32, 4, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<64, 4, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 96 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<96, 4, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<32, 4, 8>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<64, 4, 8>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<32, 8, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<64, 8, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<32, 2, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<64, 2, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<32, 3, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<64, 3, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 96 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_launch_impl<96, 3, 4>(stream, M, K, N, x, weight_orig, y);
  if (threads == 32 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<32, 3, 8>(stream, M, K, N, x, weight_orig, y);
  if (threads == 64 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_launch_impl<64, 3, 8>(stream, M, K, N, x, weight_orig, y);
  assert(false && "unsupported linear_orig_rows_cfg_f16 threads/row_tile/out_tile");
}

void rwkv7_v3a_linear_t_act_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t, int act, half* y) {
  if (act == 1) {
    if (M == 1) {
      linear_t_act_f16_ntile_scalar_kernel<128, 2, 1><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(M, K, N, x, weight_t, y);
    } else {
      linear_t_act_f16_ntile_kernel<128, 4, 1><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(M, K, N, x, weight_t, y);
    }
  } else {
    if (M == 1) {
      linear_t_act_f16_ntile_scalar_kernel<128, 2, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(M, K, N, x, weight_t, y);
    } else {
      linear_t_act_f16_ntile_kernel<128, 4, 2><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(M, K, N, x, weight_t, y);
    }
  }
}

void rwkv7_v3a_linear_wag_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg,
    const half* xw, const half* xa, const half* xg,
    const half* w1_t, const half* a1_t, const half* g1_t,
    half* w1, half* a1, half* g1) {
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  linear_wag_rank_in_f16_kernel<256><<<dim3(Rmax, M, 3), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rmax, xw, xa, xg, w1_t, a1_t, g1_t, w1, a1, g1);
}

void rwkv7_v3a_linear_wagv_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg, int Rv,
    const half* xw, const half* xa, const half* xg, const half* xv,
    const half* w1_t, const half* a1_t, const half* g1_t, const half* v1_t,
    half* w1, half* a1, half* g1, half* v1) {
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t, w1, a1, g1, v1);
}

void rwkv7_v3a_linear_wag_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg,
    const half* w1, const half* a1, const half* g1,
    const half* w2_t, const half* a2_t, const half* g2_t,
    half* w, half* a, half* g) {
  linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
      M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t, w, a, g);
}

void rwkv7_v3a_linear_wagv_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg, int Kv,
    const half* w1, const half* a1, const half* g1, const half* v1,
    const half* w2_t, const half* a2_t, const half* g2_t, const half* v2_t,
    const half* v, const half* v_first, const half* v0,
    half* w, half* a, half* g, half* v_out) {
  linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
      M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0, w, a, g, v_out);
}

void rwkv7_v3a_linear_t_vres_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t,
    const half* v, const half* v_first, const half* v0, half* y) {
  if (M == 1) {
    linear_t_vres_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x, weight_t, v, v_first, v0, y);
  } else {
    linear_t_vres_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x, weight_t, v, v_first, v0, y);
  }
}
