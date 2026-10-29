#pragma once
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <cmath>
#include <cstddef>
// Portable CUDA/HIP tiled products. Every output has one owner, with FP32
// accumulation and a single final cast; no atomics or dense delta-W workspace.
namespace rwkv_nora_detail {
template <class T> __device__ float value(T x) { return float(x); }
template <class L, class R, class O, bool TL, bool TR, bool Add>
__global__ void product(int m, int n, int k, const L *left, const R *right,
                        O *out, float scale) {
  constexpr int tile = 16;
  int y = threadIdx.x / tile, x = threadIdx.x % tile;
  int row = blockIdx.y * tile + y, col = blockIdx.x * tile + x;
  __shared__ float a[tile][tile], b[tile][tile];
  float sum = 0;
  for (int start = 0; start < k; start += tile) {
    int ak = start + x, bk = start + y;
    a[y][x] =
        row < m && ak < k
            ? value(left[TL ? size_t(ak) * m + row : size_t(row) * k + ak])
            : 0;
    b[y][x] =
        col < n && bk < k
            ? value(right[TR ? size_t(col) * k + bk : size_t(bk) * n + col])
            : 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < tile; ++j)
      sum = fmaf(a[y][j], b[j][x], sum);
    __syncthreads();
  }
  if (row < m && col < n) {
    size_t i = size_t(row) * n + col;
    out[i] = O(fmaf(scale, sum, Add ? value(out[i]) : 0.0f));
  }
}
template <bool TL, bool TR, bool Add, class L, class R, class O>
void gemm(cudaStream_t stream, int m, int n, int k, const L *a, const R *b,
          O *out, float scale = 1) {
  product<L, R, O, TL, TR, Add>
      <<<dim3((n + 15) / 16, (m + 15) / 16), 256, 0, stream>>>(m, n, k, a, b,
                                                               out, scale);
}
// No scratch allocation in decode or forward-only callers. Rank projection is
// shared within each output tile, like the existing MiSS inference kernel.
template <class T>
__global__ void fused(int in, int out, int rank, float scale, const T *a,
                      const T *b, const T *x, T *y) {
  extern __shared__ float s[];
  __shared__ float partial[256];
  int row = blockIdx.y;
  // Sixteen latent coordinates per CTA tile, sixteen lanes per dot
  // product. Coalesced A/X loads and two barriers per tile, including decode.
  int jlane = threadIdx.x / 16, klane = threadIdx.x % 16;
  for (int start = 0; start < rank; start += 16) {
    int j = start + jlane;
    float sum = 0;
    if (j < rank)
      for (int k = klane; k < in; k += 16)
        sum = fmaf(value(x[size_t(row) * in + k]), value(a[size_t(j) * in + k]),
                   sum);
    partial[threadIdx.x] = sum;
    __syncthreads();
    if (klane == 0 && j < rank) {
      sum = 0;
      for (int k = 0; k < 16; ++k)
        sum += partial[jlane * 16 + k];
      s[j] = sum;
    }
    __syncthreads();
  }
  int o = blockIdx.x * 256 + threadIdx.x;
  if (o < out) {
    float sum = 0;
    for (int j = 0; j < rank; ++j)
      sum = fmaf(s[j], value(b[size_t(o) * rank + j]), sum);
    size_t i = size_t(row) * out + o;
    y[i] = T(fmaf(scale, sum, value(y[i])));
  }
}
} // namespace rwkv_nora_detail
