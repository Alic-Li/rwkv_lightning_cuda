#pragma once
#include "rwkv/bf16_training/runtime.hpp"
#include "rwkv/runtime/nora_gemm.cuh"

namespace rwkv_bf16_nora {
namespace detail {
#ifndef RWKV_USE_HIP
// A warp owns one latent coordinate. Unlike a 16x16 output tile, this
// exposes many CTAs even for [512,16] projections. The reduction tree does
// not depend on the number of packed rows; all arithmetic stays FP32.
template <bool TR>
__global__ void project(int rows, int rank, int width, const bf16 *x,
                        const bf16 *weight, float *out, float scale) {
  const int lane = threadIdx.x % 32;
  const int item = blockIdx.x * 4 + threadIdx.x / 32;
  if (item >= rows * rank)
    return;
  const int row = item / rank, col = item % rank;
  float sum = 0;
  for (int k = lane; k < width; k += 32) {
    const float a = to_float(x[size_t(row) * width + k]);
    const float b =
        to_float(weight[TR ? size_t(col) * width + k : size_t(k) * rank + col]);
    sum = fmaf(a, b, sum);
  }
  for (int offset = 16; offset; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane == 0)
    out[item] = scale * sum;
}

template <int Rank>
__global__ void project_nt(int rows, int width, const bf16 *x,
                           const bf16 *weight, float *out, float scale) {
  constexpr int Parts = 128 / Rank;
  __shared__ float partial[Parts][Rank];
  const int row = blockIdx.x, col = threadIdx.x % Rank;
  const int part = threadIdx.x / Rank;
  float sum = 0;
  for (int k = part; k < width; k += Parts)
    sum = fmaf(to_float(x[size_t(row) * width + k]),
               to_float(weight[size_t(k) * Rank + col]), sum);
  partial[part][col] = sum;
  __syncthreads();
  if (part == 0) {
    float result = 0;
#pragma unroll
    for (int p = 0; p < Parts; ++p)
      result += partial[p][col];
    out[size_t(row) * Rank + col] = scale * result;
  }
}

// Rank is the short reduction dimension. Load the weight tile coalesced
// once, then each thread owns eight outputs and accumulates in rank order.
// No atomics, temporary buffer or extra BF16 rounding of FP32 latents.
template <bool TR, int Rank>
__global__ void expand(int rows, int cols, const float *x, const bf16 *weight,
                       bf16 *out, float scale) {
  constexpr int Columns = 64, Rows = 16;
  __shared__ float weights[Rank][Columns];
  __shared__ float latent[Rows][Rank];
  const int col0 = blockIdx.x * Columns, row0 = blockIdx.y * Rows;
  for (int i = threadIdx.x; i < Rank * Columns; i += blockDim.x) {
    const int c = TR ? i / Rank : i % Columns;
    const int k = TR ? i % Rank : i / Columns;
    weights[k][c] = col0 + c < cols
                        ? to_float(weight[TR ? size_t(col0 + c) * Rank + k
                                             : size_t(k) * cols + col0 + c])
                        : 0;
  }
  for (int i = threadIdx.x; i < Rows * Rank; i += blockDim.x)
    latent[i / Rank][i % Rank] =
        row0 + i / Rank < rows ? x[size_t(row0 + i / Rank) * Rank + i % Rank]
                               : 0;
  __syncthreads();
  for (int i = threadIdx.x; i < Rows * Columns; i += blockDim.x) {
    const int r = i / Columns, c = i % Columns;
    if (row0 + r < rows && col0 + c < cols) {
      float sum = 0;
#pragma unroll
      for (int k = 0; k < Rank; ++k)
        sum = fmaf(latent[r][k], weights[k][c], sum);
      const size_t at = size_t(row0 + r) * cols + col0 + c;
      out[at] = to_bf16(fmaf(scale, sum, to_float(out[at])));
    }
  }
}
#endif

template <bool TR>
void projection(cudaStream_t stream, int rows, int rank, int width,
                const bf16 *x, const bf16 *weight, float *out,
                float scale = 1) {
#ifndef RWKV_USE_HIP
  if (width >= 256) {
    if constexpr (TR) {
      if (rank <= 32) {
        project<TR><<<(rows * rank + 3) / 4, 128, 0, stream>>>(
            rows, rank, width, x, weight, out, scale);
        return;
      }
    } else if (rank == 16) {
      project_nt<16>
          <<<rows, 128, 0, stream>>>(rows, width, x, weight, out, scale);
      return;
    }
  }
#endif
  rwkv_nora_detail::gemm<false, TR, false>(stream, rows, rank, width, x, weight,
                                           out, scale);
}
template <bool TR>
void expansion(cudaStream_t stream, int rows, int cols, int rank,
               const float *x, const bf16 *weight, bf16 *out, float scale = 1) {
#ifndef RWKV_USE_HIP
  if (rank == 16) {
    expand<TR, 16>
        <<<dim3((cols + 63) / 64, (rows + 15) / 16), 128, 0, stream>>>(
            rows, cols, x, weight, out, scale);
    return;
  }
#endif
  rwkv_nora_detail::gemm<false, TR, true>(stream, rows, cols, rank, x, weight,
                                          out, scale);
}
} // namespace detail
} // namespace rwkv_bf16_nora
