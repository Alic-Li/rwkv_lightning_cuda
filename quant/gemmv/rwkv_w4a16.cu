#include <mma.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "rwkv_w4a16.cuh"

namespace {
int ceil_div(int x, int y) { return x / y + (x % y != 0); }
void check_group(int g) {
  if (g != 32 && g != 128)
    throw std::invalid_argument("W4A16 group_size must be 32 or 128");
}
std::size_t checked_bytes(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("W4A16 buffer size overflow");
  return a * b;
}
__device__ __forceinline__ int signed_nibble(unsigned q) {
  return int(q ^ 8u) - 8;
}

template <int G>
__global__ void quantize(const half *w, unsigned char *q, half *scales, int N,
                         int K) {
  const int lane = threadIdx.x & 31;
  const std::size_t groups = (std::size_t(K) + G - 1) / G;
  const std::size_t id = std::size_t(blockIdx.x) * 4 + threadIdx.x / 32;
  if (id >= std::size_t(N) * groups)
    return;
  const int n = id / groups, begin = (id % groups) * G;
  float amax = 0;
  for (int j = lane; j < G && begin + j < K; j += 32)
    amax = fmaxf(amax, fabsf(__half2float(w[std::size_t(n) * K + begin + j])));
  for (int d = 16; d; d >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
  const half hs = __float2half_rn(
      amax == 0 ? 1.0f : fmaxf(__fdiv_rn(amax, 7.0f), 0x1p-24f));
  const float s = __half2float(hs);
  if (lane == 0)
    scales[id] = hs;
  for (int j = lane * 2; j < G && begin + j < K; j += 64) {
    int lo = __float2int_rn(
        __fdiv_rn(__half2float(w[std::size_t(n) * K + begin + j]), s));
    int hi = 0;
    if (begin + j + 1 < K)
      hi = __float2int_rn(
          __fdiv_rn(__half2float(w[std::size_t(n) * K + begin + j + 1]), s));
    lo = max(-7, min(7, lo));
    hi = max(-7, min(7, hi));
    q[std::size_t(n) * ((std::size_t(K) + 1) / 2) + (begin + j) / 2] =
        (lo & 15) | ((hi & 15) << 4);
  }
}

// Four warps compute four output channels; reuse each packed byte/scale
// across R independent requests, with coalesced pair loads along K.
template <int G, int R>
__global__ void gemv(int M, int K, int N, const half *x, const unsigned char *q,
                     const half *scales, half *y, float *partial, int splits) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 4 + threadIdx.x / 32;
  const int m0 = blockIdx.y * R;
  if (n >= N)
    return;
  const int groups = (std::size_t(K) + G - 1) / G;
  const int begin =
      (groups / splits * blockIdx.z + min(int(blockIdx.z), groups % splits));
  const int end = (groups / splits * (blockIdx.z + 1) +
                   min(int(blockIdx.z + 1), groups % splits));
  float sum[R] = {};
  for (int g = begin; g < end; ++g) {
    const float s = __half2float(scales[std::size_t(n) * groups + g]);
    for (int j = lane * 2; j < G; j += 64) {
      const int k = g * G + j;
      if (k < K) {
        const unsigned v =
            q[std::size_t(n) * ((std::size_t(K) + 1) / 2) + k / 2];
        const float w0 = signed_nibble(v & 15) * s;
        const float w1 = signed_nibble(v >> 4) * s;
#pragma unroll
        for (int r = 0; r < R; ++r)
          if (m0 + r < M) {
            sum[r] =
                fmaf(__half2float(x[std::size_t(m0 + r) * K + k]), w0, sum[r]);
            if (k + 1 < K)
              sum[r] = fmaf(__half2float(x[std::size_t(m0 + r) * K + k + 1]),
                            w1, sum[r]);
          }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    for (int d = 16; d; d >>= 1)
      sum[r] += __shfl_down_sync(0xffffffff, sum[r], d);
    if (lane == 0 && m0 + r < M) {
      const std::size_t i = std::size_t(m0 + r) * N + n;
      if (splits == 1)
        y[i] = __float2half_rn(sum[r]);
      else
        partial[std::size_t(blockIdx.z) * M * N + i] = sum[r];
    }
  }
}

// Aligned decode: each lane loads eight weights and eight activations at once.
template <int G, int R>
__global__ void gemv_aligned(int M, int K, int N, const half *__restrict__ x,
                             const unsigned char *__restrict__ q,
                             const half *__restrict__ scales,
                             half *__restrict__ y, float *__restrict__ partial,
                             int splits) {
  const int lane = threadIdx.x & 31, n = blockIdx.x * 4 + threadIdx.x / 32;
  if (n >= N)
    return;
  const int m0 = blockIdx.y * R;
  const int groups = K / G;
  const int begin =
      (groups / splits * blockIdx.z + min(int(blockIdx.z), groups % splits)) *
      G;
  const int end =
      begin + (groups / splits + (int(blockIdx.z) < groups % splits)) * G;
  float sum[R] = {};
  for (int k = begin + lane * 8; k < end; k += 256) {
    const unsigned bits = *reinterpret_cast<const unsigned *>(
        q + std::size_t(n) * (K / 2) + k / 2);
    const float scale = __half2float(scales[std::size_t(n) * groups + k / G]);
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (m0 + r < M) {
        const int4 raw =
            *reinterpret_cast<const int4 *>(x + std::size_t(m0 + r) * K + k);
        const half *xv = reinterpret_cast<const half *>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          sum[r] = fmaf(__half2float(xv[i]),
                        signed_nibble((bits >> (4 * i)) & 15) * scale, sum[r]);
      }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    for (int d = 16; d; d >>= 1)
      sum[r] += __shfl_down_sync(0xffffffff, sum[r], d);
    if (lane == 0 && m0 + r < M) {
      const std::size_t i = std::size_t(m0 + r) * N + n;
      if (splits == 1)
        y[i] = __float2half_rn(sum[r]);
      else
        partial[std::size_t(blockIdx.z) * M * N + i] = sum[r];
    }
  }
}

// 16x64 output tile, four warps. Packed weights are unpacked directly into
// shared memory; FP16 Tensor Core multiplication uses FP32 accumulators.
template <int G, bool Aligned>
__global__ void gemm(int M, int K, int N, const half *x, const unsigned char *q,
                     const half *scales, half *y, float *partial, int splits) {
  using namespace nvcuda;
  __shared__ __align__(32) half a[16 * 72];
  __shared__ __align__(32) half b[64 * 72];
  __shared__ __align__(32) float c[16 * 64];
  const int tid = threadIdx.x, warp = tid / 32;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 64;
  const int tiles = (std::size_t(K) + 63) / 64;
  const int begin =
      (tiles / splits * blockIdx.z + min(int(blockIdx.z), tiles % splits));
  const int end = (tiles / splits * (blockIdx.z + 1) +
                   min(int(blockIdx.z + 1), tiles % splits));
  const int groups = (std::size_t(K) + G - 1) / G;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int tile = begin; tile < end; ++tile) {
    for (int i = tid; i < 16 * 64; i += 128) {
      const int m = m0 + i / 64, k = tile * 64 + i % 64;
      a[i / 64 * 72 + i % 64] =
          m < M && k < K ? x[std::size_t(m) * K + k] : __float2half(0);
    }
    if constexpr (Aligned) {
      for (int i = tid; i < 64 * 8; i += 128) {
        const int n = n0 + i / 8, k = tile * 64 + (i % 8) * 8;
        const unsigned v = *reinterpret_cast<const unsigned *>(
            q + std::size_t(n) * (K / 2) + k / 2);
        const half2 s = __half2half2(scales[std::size_t(n) * groups + k / G]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const half2 pair = __halves2half2(
              __int2half_rn(signed_nibble((v >> (j * 8)) & 15)),
              __int2half_rn(signed_nibble((v >> (j * 8 + 4)) & 15)));
          *reinterpret_cast<half2 *>(b + (i / 8) * 72 + (i % 8) * 8 + j * 2) =
              __hmul2(pair, s);
        }
      }
    } else {
      for (int i = tid; i < 64 * 32; i += 128) {
        const int n = n0 + i / 32, k = tile * 64 + (i % 32) * 2;
        unsigned v = 0;
        float s = 0;
        if (n < N && k < K) {
          v = q[std::size_t(n) * ((std::size_t(K) + 1) / 2) + k / 2];
          s = __half2float(scales[std::size_t(n) * groups + k / G]);
        }
        b[(i / 32) * 72 + (i % 32) * 2] =
            __float2half_rn(signed_nibble(v & 15) * s);
        b[(i / 32) * 72 + (i % 32) * 2 + 1] =
            __float2half_rn(k + 1 < K ? signed_nibble(v >> 4) * s : 0.0f);
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 64; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
      wmma::load_matrix_sync(af, a + k, 72);
      wmma::load_matrix_sync(bf, b + warp * 16 * 72 + k, 72);
      wmma::mma_sync(acc, af, bf, acc);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(c + warp * 16, acc, 64, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 16 * 64; i += 128) {
    const int m = m0 + i / 64, n = n0 + i % 64;
    if (m < M && n < N) {
      const std::size_t out = std::size_t(m) * N + n;
      if (splits == 1)
        y[out] = __float2half_rn(c[i]);
      else
        partial[std::size_t(blockIdx.z) * M * N + out] = c[i];
    }
  }
}
__device__ __forceinline__ half2 dequant_pair(unsigned byte, half2 scale) {
  const unsigned bits =
      (((byte & 15) | ((byte & 240) << 12)) ^ 0x00080008u) | 0x64006400u;
  const unsigned magic = 0x64086408u;
  return __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&bits),
                         *reinterpret_cast<const half2 *>(&magic)),
                 scale);
}

// Register-dequantized m16n8k16 path, following the existing W8 MMA approach.
// Each warp owns 16 columns and reuses B across 16- or 64-row request tiles.
template <int G, int BM>
__global__ void mma_aligned(int M, int K, int N, const half *__restrict__ x,
                            const unsigned char *__restrict__ q,
                            const half *__restrict__ scales,
                            half *__restrict__ y, float *__restrict__ partial,
                            int splits) {
#if __CUDA_ARCH__ >= 750
  __shared__ __align__(16) half a[BM * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * 64;
  const int tiles = K / 64, groups = (K + G - 1) / G;
  const int begin =
      tiles / splits * blockIdx.z + min(int(blockIdx.z), tiles % splits);
  const int end = begin + tiles / splits + (int(blockIdx.z) < tiles % splits);
  float acc[BM / 16][2][4] = {};
  for (int tile = begin; tile < end; ++tile) {
    for (int i = tid; i < BM * 8; i += 128) {
      const int row = i / 8, chunk = i % 8;
      int4 xv = make_int4(0, 0, 0, 0);
      if (m0 + row < M)
        xv = *reinterpret_cast<const int4 *>(x + std::size_t(m0 + row) * K +
                                             tile * 64 + chunk * 8);
      *reinterpret_cast<int4 *>(a + row * 64 + (chunk ^ (row & 7)) * 8) = xv;
    }
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int n = n0 + warp * 16 + jj * 8 + lane / 4;
      const uint2 raw = *reinterpret_cast<const uint2 *>(
          q + std::size_t(n) * (K / 2) + tile * 32 + (lane & 3) * 8);
      const half2 s0 =
          __half2half2(scales[std::size_t(n) * groups + tile * 64 / G]);
      const half2 s1 =
          G == 32 ? __half2half2(scales[std::size_t(n) * groups + tile * 2 + 1])
                  : s0;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const unsigned lo = __shfl_sync(0xffffffff, raw.x, (lane & ~3) + kk);
        const unsigned hi = __shfl_sync(0xffffffff, raw.y, (lane & ~3) + kk);
        const half2 b0 = dequant_pair(lo >> ((lane & 3) * 8), kk < 2 ? s0 : s1);
        const half2 b1 = dequant_pair(hi >> ((lane & 3) * 8), kk < 2 ? s0 : s1);
        const unsigned bbits0 = *reinterpret_cast<const unsigned *>(&b0),
                       bbits1 = *reinterpret_cast<const unsigned *>(&b1);
#pragma unroll
        for (int mm = 0; mm < BM / 16; ++mm) {
          unsigned av[4];
          const int row = mm * 16 + lane % 16;
          const int chunk = kk * 2 + lane / 16;
          const unsigned addr =
              __cvta_generic_to_shared(a + row * 64 + (chunk ^ (row & 7)) * 8);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
              : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3])
              : "r"(addr));
#if __CUDA_ARCH__ >= 800
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]),
                         "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
                       : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]),
                         "r"(bbits0), "r"(bbits1));
#else
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                       : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]),
                         "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
                       : "r"(av[0]), "r"(av[1]), "r"(bbits0));
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                       : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]),
                         "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
                       : "r"(av[2]), "r"(av[3]), "r"(bbits1));
#endif
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mm = 0; mm < BM / 16; ++mm) {
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int m = m0 + mm * 16 + lane / 4 + j * 8;
        const int n = n0 + warp * 16 + jj * 8 + (lane & 3) * 2;
        if (m < M) {
          const std::size_t i = std::size_t(m) * N + n;
          if (splits == 1)
            *reinterpret_cast<half2 *>(y + i) =
                __floats2half2_rn(acc[mm][jj][j * 2], acc[mm][jj][j * 2 + 1]);
          else
            *reinterpret_cast<float2 *>(partial +
                                        std::size_t(blockIdx.z) * M * N + i) =
                make_float2(acc[mm][jj][j * 2], acc[mm][jj][j * 2 + 1]);
        }
      }
    }
  }
#endif
}

__global__ void reduce(const float *partial, half *y, std::size_t count,
                       int splits) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float v = 0;
  for (int s = 0; s < splits; ++s)
    v += partial[std::size_t(s) * count + i];
  y[i] = __float2half_rn(v);
}
template <int G>
void launch(cudaStream_t stream, int M, int K, int N, const half *x,
            const unsigned char *q, const half *scale, half *y, float *tmp,
            int splits) {
  const bool aligned_mma = K % 64 == 0 && N % 64 == 0 &&
                           reinterpret_cast<std::uintptr_t>(q) % 8 == 0 &&
                           reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
                           reinterpret_cast<std::uintptr_t>(y) % 4 == 0 &&
                           reinterpret_cast<std::uintptr_t>(tmp) % 8 == 0;
  if (M > 1 && aligned_mma && (M > 4 || splits > 1)) {
    if (M <= 16)
      mma_aligned<G, 16>
          <<<dim3(N / 64, ceil_div(M, 16), splits), 128, 0, stream>>>(
              M, K, N, x, q, scale, y, tmp, splits);
    else
      mma_aligned<G, 64>
          <<<dim3(N / 64, ceil_div(M, 64), splits), 128, 0, stream>>>(
              M, K, N, x, q, scale, y, tmp, splits);
  } else if (M <= 4 && K % G == 0 &&
             reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
             reinterpret_cast<std::uintptr_t>(q) % 4 == 0) {
    const dim3 grid(ceil_div(N, 4), 1, splits);
    if (M == 1)
      gemv_aligned<G, 1>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
    else if (M == 2)
      gemv_aligned<G, 2>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
    else
      gemv_aligned<G, 4>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
  } else if (M <= 4) {
    const dim3 grid(ceil_div(N, 4), 1, splits);
    if (M == 1)
      gemv<G, 1>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
    else if (M == 2)
      gemv<G, 2>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
    else
      gemv<G, 4>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
  } else {
    const dim3 grid(ceil_div(N, 64), ceil_div(M, 16), splits);
    if (K % 64 == 0 && N % 64 == 0 &&
        reinterpret_cast<std::uintptr_t>(q) % 4 == 0)
      gemm<G, true>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
    else
      gemm<G, false>
          <<<grid, 128, 0, stream>>>(M, K, N, x, q, scale, y, tmp, splits);
  }
  if (splits > 1) {
    const std::size_t count = std::size_t(M) * N;
    reduce<<<(count + 255) / 256, 256, 0, stream>>>(tmp, y, count, splits);
  }
}
} // namespace

std::size_t rwkv7_w4a16_weight_bytes(int N, int K) {
  return N > 0 && K > 0 ? checked_bytes(N, (std::size_t(K) + 1) / 2) : 0;
}
std::size_t rwkv7_w4a16_scale_count(int N, int K, int group_size) {
  check_group(group_size);
  return N > 0 && K > 0 ? checked_bytes(N, ceil_div(K, group_size)) : 0;
}
std::size_t rwkv7_w4a16_workspace_bytes(int M, int N, int split_k) {
  return M > 0 && N > 0 && split_k > 1
             ? checked_bytes(checked_bytes(M, N),
                             checked_bytes(split_k, sizeof(float)))
             : 0;
}
void rwkv7_w4a16_quantize_launch(cudaStream_t stream, const half *weight_nk,
                                 std::uint8_t *qweight, half *scale, int N,
                                 int K, int group_size) {
  if (N <= 0 || K <= 0)
    return;
  check_group(group_size);
  if (!weight_nk || !qweight || !scale)
    throw std::invalid_argument("W4A16 null quantize buffer");
  const auto blocks = (rwkv7_w4a16_scale_count(N, K, group_size) + 3) / 4;
  if (blocks > 2147483647)
    throw std::invalid_argument("W4A16 quantize grid too large");
  if (group_size == 32)
    quantize<32><<<blocks, 128, 0, stream>>>(weight_nk, qweight, scale, N, K);
  else
    quantize<128><<<blocks, 128, 0, stream>>>(weight_nk, qweight, scale, N, K);
}
void rwkv7_w4a16_linear_launch(cudaStream_t stream, int M, int K, int N,
                               const half *x, const std::uint8_t *qweight,
                               const half *scale, half *y, int group_size,
                               void *workspace, std::size_t workspace_bytes,
                               int force_split_k) {
  if (M <= 0 || K <= 0 || N <= 0)
    return;
  check_group(group_size);
  if (!x || !qweight || !scale || !y || force_split_k < 0)
    throw std::invalid_argument("W4A16 invalid linear arguments");
  if (ceil_div(M, 16) > 65535 || (std::size_t(M) * N + 255) / 256 > 2147483647)
    throw std::invalid_argument("W4A16 linear grid too large");
  const int tiles = ceil_div(K, M <= 4 ? group_size : 64);
  int splits = force_split_k;
  if (splits > tiles || splits > 65535)
    throw std::invalid_argument("W4A16 split_k exceeds K tiles or grid limit");
  if (!splits) {
    splits = 1;
    const std::size_t blocks =
        M == 1 ? ceil_div(N, 4)
               : std::size_t(ceil_div(N, 64)) * ceil_div(M, M <= 16 ? 16 : 64);
    // Large batches already expose parallelism; cap scratch traffic there.
    const int max_splits = M > 64 ? 8 : 16;
    while (M > 1 && splits < max_splits && blocks * splits < 2048 &&
           tiles / (splits * 2) >= (M <= 4    ? 2
                                    : M <= 32 ? 4
                                              : 8))
      splits *= 2;
    while (splits > 1 &&
           (!workspace ||
            workspace_bytes < rwkv7_w4a16_workspace_bytes(M, N, splits)))
      splits /= 2;
  }
  if (splits > 1 &&
      (!workspace ||
       reinterpret_cast<std::uintptr_t>(workspace) % alignof(float) ||
       workspace_bytes < rwkv7_w4a16_workspace_bytes(M, N, splits)))
    throw std::invalid_argument("W4A16 split-K needs a separate, sufficiently "
                                "large float-aligned workspace");
  if (group_size == 32)
    launch<32>(stream, M, K, N, x, qweight, scale, y,
               static_cast<float *>(workspace), splits);
  else
    launch<128>(stream, M, K, N, x, qweight, scale, y,
                static_cast<float *>(workspace), splits);
}
