#include "rwkv_w4a16.cuh"
#include "rwkv_w8a16.cuh"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
int ceil_div(int x, int y) { return x / y + (x % y != 0); }
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("quantized buffer size overflow");
  return a * b;
}
void check_group(int g) {
  if (g != 32 && g != 128)
    throw std::invalid_argument("W4 group must be 32 or 128");
}

// Explicit 32-lane subgroups also keep the scalar paths valid on wave64.
template <int Bits>
__device__ float weight_at(const unsigned char *q, const half *s, int n, int k,
                           int K, int N, int G, bool kn) {
  if constexpr (Bits == 4) {
    const unsigned b = q[std::size_t(n) * ((std::size_t(K) + 1) / 2) + k / 2];
    const int v = int((b >> (4 * (k & 1))) & 15);
    return ((v ^ 8) - 8) *
           __half2float(s[std::size_t(n) * ((K + G - 1) / G) + k / G]);
  } else {
    const auto *w = reinterpret_cast<const signed char *>(q);
    return float(w[kn ? std::size_t(k) * N + n : std::size_t(n) * K + k]) *
           __half2float(s[n]);
  }
}

template <int Bits, int R>
__global__ void quant_gemv(int M, int K, int N, const half *x,
                           const unsigned char *q, const half *scales, half *y,
                           int G, bool kn, float *partial, int splits) {
  const int lane = threadIdx.x % 32;
  const int n = blockIdx.x * 4 + threadIdx.x / 32;
  const int m0 = blockIdx.y * R;
  if (n >= N)
    return;
  const int begin = (std::int64_t(K) * blockIdx.z / splits);
  const int end = (std::int64_t(K) * (blockIdx.z + 1) / splits);
  float sum[R] = {};
  if (!kn && K % 8 == 0 && begin % 8 == 0 && end % 8 == 0 &&
      reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
      reinterpret_cast<std::uintptr_t>(q) % 8 == 0) {
    for (int k = begin + lane * 8; k < end; k += 256) {
      unsigned long long packed;
      if constexpr (Bits == 4)
        packed = *reinterpret_cast<const unsigned *>(
            q + std::size_t(n) * (K / 2) + k / 2);
      else
        packed = *reinterpret_cast<const unsigned long long *>(
            q + std::size_t(n) * K + k);
      const float scale = __half2float(
          scales[Bits == 4 ? std::size_t(n) * ((K + G - 1) / G) + k / G : n]);
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (m0 + r < M) {
          const int4 raw =
              *reinterpret_cast<const int4 *>(x + std::size_t(m0 + r) * K + k);
          const half *xv = reinterpret_cast<const half *>(&raw);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int v = (packed >> (Bits * j)) & ((1 << Bits) - 1);
            const int w = (v ^ (1 << (Bits - 1))) - (1 << (Bits - 1));
            sum[r] = fmaf(__half2float(xv[j]), w * scale, sum[r]);
          }
        }
    }
  } else
    for (int k = begin + lane; k < end; k += 32) {
      const float w = weight_at<Bits>(q, scales, n, k, K, N, G, kn);
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (m0 + r < M)
          sum[r] =
              fmaf(__half2float(x[std::size_t(m0 + r) * K + k]), w, sum[r]);
    }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    for (int d = 16; d; d >>= 1)
      sum[r] += __shfl_down(sum[r], d, 32);
    if (lane == 0 && m0 + r < M) {
      const std::size_t i = std::size_t(m0 + r) * N + n;
      if (splits == 1)
        y[i] = __float2half_rn(sum[r]);
      else
        partial[std::size_t(blockIdx.z) * M * N + i] = sum[r];
    }
  }
}

// Portable tiled fallback. Weights stay compressed in global memory and are
// dequantized into a bounded LDS tile; no full FP16 weight copy is allocated.
template <int Bits>
__global__ void quant_gemm(int M, int K, int N, const half *x,
                           const unsigned char *q, const half *scales, half *y,
                           int G, bool kn, float *partial, int splits) {
#if defined(__gfx1100__) && !defined(RWKV_QUANT_SCALAR_ONLY)
  // RDNA3 WMMA: A lanes carry rows, B lanes carry columns; both halves
  // of the wave replicate the 16-element input fragments. FP32 accumulators
  // map to rows 2*i + lane/16 and column lane%16 (AMD GPUOpen WMMA guide).
  using half16 = _Float16 __attribute__((ext_vector_type(16)));
  using float8 = float __attribute__((ext_vector_type(8)));
  __shared__ _Float16 a_tile[16][72], b_tile[64][72];
  const int lane = threadIdx.x % 32;
  const int m0 = blockIdx.y * 16,
            n0 = blockIdx.x * 64 + (threadIdx.x / 32) * 16;
  const int tiles = (K + 63) / 64;
  const int begin = std::int64_t(tiles) * blockIdx.z / splits;
  const int end = std::int64_t(tiles) * (blockIdx.z + 1) / splits;
  float8 acc = {};
  for (int t = begin; t < end; ++t) {
    for (int i = threadIdx.x; i < 16 * 64; i += 128) {
      const int m = m0 + i / 64, k = t * 64 + i % 64;
      a_tile[i / 64][i % 64] =
          m < M && k < K ? _Float16(__half2float(x[std::size_t(m) * K + k]))
                         : _Float16(0);
    }
    for (int i = threadIdx.x; i < 64 * 64; i += 128) {
      const int n = blockIdx.x * 64 + i / 64, k = t * 64 + i % 64;
      b_tile[i / 64][i % 64] =
          n < N && k < K
              ? _Float16(weight_at<Bits>(q, scales, n, k, K, N, G, kn))
              : _Float16(0);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
      half16 a, b;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        a[j] = a_tile[lane % 16][kk + j];
        b[j] = b_tile[(threadIdx.x / 32) * 16 + lane % 16][kk + j];
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int m = m0 + 2 * j + lane / 16, n = n0 + lane % 16;
    if (m < M && n < N) {
      const std::size_t i = std::size_t(m) * N + n;
      if (splits == 1)
        y[i] = __float2half_rn(acc[j]);
      else
        partial[std::size_t(blockIdx.z) * M * N + i] = acc[j];
    }
  }
#else
  __shared__ float a[16][16], b[16][16];
  const int row = threadIdx.x / 16, col = threadIdx.x % 16;
  const int tiles = (K + 15) / 16;
  const int begin = std::int64_t(tiles) * blockIdx.z / splits;
  const int end = std::int64_t(tiles) * (blockIdx.z + 1) / splits;
  for (int nt = 0; nt < 4; ++nt) {
    const int m = blockIdx.y * 16 + row, n = blockIdx.x * 64 + nt * 16 + col;
    float sum[2] = {};
    for (int t = begin; t < end; ++t) {
      for (int r = 0; r < 2; ++r) {
        const int rr = row + r * 8, mm = m + r * 8;
        const int ka = t * 16 + col, kb = t * 16 + rr;
        a[rr][col] =
            mm < M && ka < K ? __half2float(x[std::size_t(mm) * K + ka]) : 0;
        b[rr][col] = n < N && kb < K
                         ? weight_at<Bits>(q, scales, n, kb, K, N, G, kn)
                         : 0;
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < 16; ++k)
        for (int r = 0; r < 2; ++r)
          sum[r] = fmaf(a[row + r * 8][k], b[k][col], sum[r]);
      __syncthreads();
    }
    for (int r = 0; r < 2; ++r)
      if (m + r * 8 < M && n < N) {
        const std::size_t i = std::size_t(m + r * 8) * N + n;
        if (splits == 1)
          y[i] = __float2half_rn(sum[r]);
        else
          partial[std::size_t(blockIdx.z) * M * N + i] = sum[r];
      }
  }
#endif
}

__global__ void reduce(half *y, const float *partial, std::size_t count,
                       int splits) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float sum = 0;
  for (int s = 0; s < splits; ++s)
    sum += partial[std::size_t(s) * count + i];
  y[i] = __float2half_rn(sum);
}

template <int Bits>
void linear(hipStream_t stream, int M, int K, int N, const half *x,
            const unsigned char *q, const half *s, half *y, int G, bool kn,
            void *workspace, std::size_t bytes, int force_split) {
  if (M <= 0 || K <= 0 || N <= 0)
    return;
  if (!x || !q || !s || !y || force_split < 0)
    throw std::invalid_argument("invalid quantized linear arguments");
  if (ceil_div(M, 4) > 65535 || (std::size_t(M) * N + 255) / 256 > 2147483647)
    throw std::invalid_argument("quantized linear grid too large");
  const int tiles = ceil_div(K, M <= 4 ? G : 64);
  int splits = force_split ? force_split : 1;
  if (!force_split && M > 4) {
    const std::size_t blocks = std::size_t(ceil_div(N, 64)) * ceil_div(M, 16);
    while (splits < 16 && blocks * splits < 512 && tiles / (splits * 2) >= 4)
      splits *= 2;
    while (splits > 1 &&
           (!workspace || bytes < rwkv7_w4a16_workspace_bytes(M, N, splits)))
      splits /= 2;
  }
  if (splits > tiles || splits > 65535)
    throw std::invalid_argument(
        "quantized split exceeds K tiles or grid limit");
  const std::size_t needed = rwkv7_w4a16_workspace_bytes(M, N, splits);
  if (splits > 1 &&
      (!workspace ||
       reinterpret_cast<std::uintptr_t>(workspace) % alignof(float) ||
       bytes < needed))
    throw std::invalid_argument(
        "split-K requires a sufficiently large float-aligned workspace");
  auto *p = static_cast<float *>(workspace);
  if (M == 1)
    quant_gemv<Bits, 1><<<dim3(ceil_div(N, 4), 1, splits), 128, 0, stream>>>(
        M, K, N, x, q, s, y, G, kn, p, splits);
  else if (M <= 4)
    quant_gemv<Bits, 4>
        <<<dim3(ceil_div(N, 4), ceil_div(M, 4), splits), 128, 0, stream>>>(
            M, K, N, x, q, s, y, G, kn, p, splits);
  else
    quant_gemm<Bits>
        <<<dim3(ceil_div(N, 64), ceil_div(M, 16), splits), 128, 0, stream>>>(
            M, K, N, x, q, s, y, G, kn, p, splits);
  if (splits > 1) {
    const std::size_t count = std::size_t(M) * N;
    reduce<<<(count + 255) / 256, 256, 0, stream>>>(y, p, count, splits);
  }
}

template <int G>
__global__ void quantize(const half *w, unsigned char *q, half *scales, int N,
                         int K) {
  const int lane = threadIdx.x % 32;
  const std::size_t groups = (std::size_t(K) + G - 1) / G;
  const std::size_t id = std::size_t(blockIdx.x) * 4 + threadIdx.x / 32;
  if (id >= std::size_t(N) * groups)
    return;
  const int n = id / groups, begin = (id % groups) * G;
  float amax = 0;
  for (int j = lane; j < G && begin + j < K; j += 32)
    amax = fmaxf(amax, fabsf(__half2float(w[std::size_t(n) * K + begin + j])));
  for (int d = 16; d; d >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, d, 32));
  const half hs =
      __float2half_rn(amax == 0 ? 1.0f : fmaxf(amax / 7.0f, 0x1p-24f));
  const float s = __half2float(hs);
  if (!lane)
    scales[id] = hs;
  for (int j = lane * 2; j < G && begin + j < K; j += 64) {
    int lo =
        int(nearbyintf(__half2float(w[std::size_t(n) * K + begin + j]) / s));
    int hi = begin + j + 1 < K
                 ? int(nearbyintf(
                       __half2float(w[std::size_t(n) * K + begin + j + 1]) / s))
                 : 0;
    lo = max(-7, min(7, lo));
    hi = max(-7, min(7, hi));
    q[std::size_t(n) * ((std::size_t(K) + 1) / 2) + (begin + j) / 2] =
        (lo & 15) | ((hi & 15) << 4);
  }
}
} // namespace

std::size_t rwkv7_w4a16_weight_bytes(int N, int K) {
  return N > 0 && K > 0 ? product(N, (std::size_t(K) + 1) / 2) : 0;
}
std::size_t rwkv7_w4a16_scale_count(int N, int K, int G) {
  check_group(G);
  return N > 0 && K > 0 ? product(N, ceil_div(K, G)) : 0;
}
std::size_t rwkv7_w4a16_workspace_bytes(int M, int N, int splits) {
  return M > 0 && N > 0 && splits > 1
             ? product(product(M, N), product(splits, sizeof(float)))
             : 0;
}
void rwkv7_w4a16_quantize_launch(hipStream_t stream, const half *w,
                                 std::uint8_t *q, half *s, int N, int K,
                                 int G) {
  if (N <= 0 || K <= 0)
    return;
  check_group(G);
  if (!w || !q || !s)
    throw std::invalid_argument("null quantize buffer");
  const auto blocks = (rwkv7_w4a16_scale_count(N, K, G) + 3) / 4;
  if (blocks > 2147483647)
    throw std::invalid_argument("quantize grid too large");
  if (G == 32)
    quantize<32><<<blocks, 128, 0, stream>>>(w, q, s, N, K);
  else
    quantize<128><<<blocks, 128, 0, stream>>>(w, q, s, N, K);
}
void rwkv7_w4a16_linear_launch(hipStream_t stream, int M, int K, int N,
                               const half *x, const std::uint8_t *q,
                               const half *s, half *y, int G, void *workspace,
                               std::size_t bytes, int splits) {
  if (M <= 0 || K <= 0 || N <= 0)
    return;
  check_group(G);
  linear<4>(stream, M, K, N, x, q, s, y, G, false, workspace, bytes, splits);
}
void rwkv7_w8a16_linear_launch(hipStream_t stream, int M, int K, int N,
                               const half *x, const std::int8_t *q,
                               const half *s, W8BLayout layout, half *y,
                               void *workspace, std::size_t bytes, int splits) {
  if (M <= 0 || K <= 0 || N <= 0)
    return;
  if (layout != W8BLayout::NK && layout != W8BLayout::KN)
    throw std::invalid_argument(
        "HIP W8 supports raw NK/KN; CUDA PackedNK is not a HIP layout");
  linear<8>(stream, M, K, N, x, reinterpret_cast<const unsigned char *>(q), s,
            y, 128, layout == W8BLayout::KN, workspace, bytes, splits);
}
