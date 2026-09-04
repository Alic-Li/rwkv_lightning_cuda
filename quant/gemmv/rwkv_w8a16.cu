#include "rwkv_w8a16.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#include <cuda_runtime.h>
#include <mma.h>

namespace {

using namespace nvcuda;

const W8A16DeviceInfo& cached_device_info() {
  static const W8A16DeviceInfo info = [] {
    W8A16DeviceInfo result;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return result;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return result;
    if (cudaDeviceGetAttribute(&result.sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
      result.sm_count = properties.multiProcessorCount;
    }
    if (cudaDeviceGetAttribute(&result.compute_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      result.compute_major = properties.major;
    }
    std::snprintf(result.name, sizeof(result.name), "%s", properties.name);
    return result;
  }();
  return info;
}

struct W8A16TuneKey {
  int K;
  int N;
  int layout;
  int m_bucket;

  bool operator<(const W8A16TuneKey& other) const {
    if (K != other.K) return K < other.K;
    if (N != other.N) return N < other.N;
    if (layout != other.layout) return layout < other.layout;
    return m_bucket < other.m_bucket;
  }
};

std::mutex g_tuning_mutex;
std::map<W8A16TuneKey, int> g_tuning_splits;

int tune_m_bucket(int M) {
  if (M <= 8) return 8;
  if (M <= 16) return 16;
  if (M <= 32) return 32;
  if (M <= 64) return 64;
  if (M <= 128) return 128;
  return 0;
}

int layout_key(W8BLayout layout) {
  return static_cast<int>(layout);
}

int lookup_tuned_split(int K, int N, W8BLayout layout, int M) {
  const int m_bucket = tune_m_bucket(M);
  if (m_bucket == 0) return 0;
  std::lock_guard<std::mutex> lock(g_tuning_mutex);
  const auto it = g_tuning_splits.find({K, N, layout_key(layout), m_bucket});
  return it == g_tuning_splits.end() ? 0 : it->second;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ std::int8_t int4_byte(const int4& value, int index) {
  return reinterpret_cast<const std::int8_t*>(&value)[index];
}

template <int BLOCK_N, int BLOCK_K>
union W8WeightTile {
  half nk[BLOCK_N][BLOCK_K];
  half kn[BLOCK_K][BLOCK_N];
};

template <int BLOCK_N, int BLOCK_K, bool Async>
struct W8QTileStorage {};

template <int BLOCK_N, int BLOCK_K>
struct W8QTileStorage<BLOCK_N, BLOCK_K, true> {
  alignas(16) std::int8_t data[BLOCK_N * BLOCK_K];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* shared_ptr, const void* global_ptr) {
  const unsigned shared_address = __cvta_generic_to_shared(shared_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared_address), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group 0;");
}
#else
__device__ __forceinline__ void cp_async_16(void* shared_ptr, const void* global_ptr) {
  *reinterpret_cast<int4*>(shared_ptr) = *reinterpret_cast<const int4*>(global_ptr);
}

__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait() {}
#endif

template <int BLOCK_N, int BLOCK_K, W8BLayout Layout, int Threads>
__device__ __forceinline__ void prefetch_q_tile(
    std::int8_t* q_tile,
    int block_k,
    int block_n,
    int K,
    int N,
    const std::int8_t* weight,
    int tid) {
  for (int vector_index = tid; vector_index < (BLOCK_N * BLOCK_K) / 16; vector_index += Threads) {
    const int first = vector_index * 16;
    int n;
    int k;
    if constexpr (Layout == W8BLayout::NK) {
      n = first / BLOCK_K;
      k = first - n * BLOCK_K;
      cp_async_16(q_tile + first,
                  weight + static_cast<std::int64_t>(block_n + n) * K + block_k + k);
    } else {
      k = first / BLOCK_N;
      n = first - k * BLOCK_N;
      cp_async_16(q_tile + first,
                  weight + static_cast<std::int64_t>(block_k + k) * N + block_n + n);
    }
  }
  cp_async_commit();
}

template <int BLOCK_N, int BLOCK_K, W8BLayout Layout, int Threads>
__device__ __forceinline__ void convert_q_tile(
    const std::int8_t* q_tile,
    W8WeightTile<BLOCK_N, BLOCK_K>& weight_tile,
    int tid) {
  for (int vector_index = tid; vector_index < (BLOCK_N * BLOCK_K) / 16; vector_index += Threads) {
    const int first = vector_index * 16;
    const int4 q = *reinterpret_cast<const int4*>(q_tile + first);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q_index = first + i * 2;
      int qn;
      int qk;
      if constexpr (Layout == W8BLayout::NK) {
        qn = q_index / BLOCK_K;
        qk = q_index - qn * BLOCK_K;
      } else {
        qn = q_index / BLOCK_N;
        qk = q_index - qn * BLOCK_N;
      }
      half* tile_value = Layout == W8BLayout::NK ? &weight_tile.nk[qn][qk] : &weight_tile.kn[qn][qk];
      *reinterpret_cast<half2*>(tile_value) =
          __halves2half2(__int2half_rn(static_cast<int>(int4_byte(q, i * 2))),
                         __int2half_rn(static_cast<int>(int4_byte(q, i * 2 + 1))));
    }
  }
}

template <int Threads, int OutTile, int MaxM>
__global__ __launch_bounds__(Threads, 4) void w8a16_gemv_kernel(
    int M,
    int K,
    int N,
    const half* __restrict__ x,
    const std::int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    half* __restrict__ y) {
  const int n0 = static_cast<int>(blockIdx.x) * OutTile;
  float acc[MaxM][OutTile];
#pragma unroll
  for (int m = 0; m < MaxM; ++m) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) acc[m][j] = 0.0f;
  }

  for (int k0 = static_cast<int>(threadIdx.x) * 16; k0 < K; k0 += Threads * 16) {
    int4 w[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      w[j] = *reinterpret_cast<const int4*>(weight + static_cast<std::int64_t>(n0 + j) * K + k0);
    }
#pragma unroll
    for (int m = 0; m < MaxM; ++m) {
      if (m < M) {
        const int4 x0 = *reinterpret_cast<const int4*>(x + static_cast<std::int64_t>(m) * K + k0);
        const int4 x1 = *reinterpret_cast<const int4*>(x + static_cast<std::int64_t>(m) * K + k0 + 8);
        const half* x0h = reinterpret_cast<const half*>(&x0);
        const half* x1h = reinterpret_cast<const half*>(&x1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float xv = __half2float(x0h[i]);
#pragma unroll
          for (int j = 0; j < OutTile; ++j) {
            acc[m][j] = fmaf(xv, static_cast<float>(int4_byte(w[j], i)), acc[m][j]);
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float xv = __half2float(x1h[i]);
#pragma unroll
          for (int j = 0; j < OutTile; ++j) {
            acc[m][j] = fmaf(xv, static_cast<float>(int4_byte(w[j], i + 8)), acc[m][j]);
          }
        }
      }
    }
  }

  __shared__ float partial[Threads / 32][MaxM][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int m = 0; m < MaxM; ++m) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float sum = warp_sum(acc[m][j]);
      if (lane == 0) partial[warp][m][j] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int m = 0; m < MaxM; ++m) {
      if (m >= M) break;
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < Threads / 32; ++w) sum += partial[w][m][j];
        y[static_cast<std::int64_t>(m) * N + n0 + j] =
            __float2half_rn(sum * __half2float(scales[n0 + j]));
      }
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, W8BLayout Layout, int THREADS, bool Async>
__global__ __launch_bounds__(THREADS, 2) void w8a16_gemm_kernel(
    int M,
    int K,
    int N,
    const half* __restrict__ x,
    const std::int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    half* __restrict__ y) {
  constexpr int WARP_M_TILES = BLOCK_M == 64 ? 2 : 1;
  constexpr int WARP_N_TILES = BLOCK_N == 128 ? 2 : (BLOCK_M >= 32 ? 2 : 1);
  constexpr int WARPS_M = BLOCK_M / (16 * WARP_M_TILES);
  constexpr int WARPS_N = BLOCK_N / (16 * WARP_N_TILES);
  constexpr int OUTPUT_N = BLOCK_N == 128 ? 64 : BLOCK_N;
  constexpr int GROUP_WARPS = BLOCK_N == 128 ? WARPS_N / 2 : WARPS_N;
  static_assert(THREADS == WARPS_M * WARPS_N * 32, "WMMA tile requires one warp per tile group");
  __shared__ half x_tile[BLOCK_M][BLOCK_K];
  __shared__ W8WeightTile<BLOCK_N, BLOCK_K> weight_tile;
  __shared__ float output_tile[BLOCK_M][OUTPUT_N];
  __shared__ W8QTileStorage<BLOCK_N, BLOCK_K, Async> q_storage;

  const int block_m = static_cast<int>(blockIdx.y) * BLOCK_M;
  const int block_n = static_cast<int>(blockIdx.x) * BLOCK_N;
  const int tid = threadIdx.x;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_M_TILES][WARP_N_TILES];
#pragma unroll
  for (int wm = 0; wm < WARP_M_TILES; ++wm)
#pragma unroll
    for (int wn = 0; wn < WARP_N_TILES; ++wn) wmma::fill_fragment(acc[wm][wn], 0.0f);

  if constexpr (Async) {
    prefetch_q_tile<BLOCK_N, BLOCK_K, Layout, THREADS>(
        q_storage.data, 0, block_n, K, N, weight, tid);
    cp_async_wait();
    __syncthreads();
    convert_q_tile<BLOCK_N, BLOCK_K, Layout, THREADS>(q_storage.data, weight_tile, tid);
    __syncthreads();
  }

  for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
    for (int index = tid; index < (BLOCK_M * BLOCK_K) / 2; index += THREADS) {
      const int m = index / (BLOCK_K / 2);
      const int k = (index - m * (BLOCK_K / 2)) * 2;
      const int global_m = block_m + m;
      *reinterpret_cast<half2*>(&x_tile[m][k]) =
          global_m < M ? *reinterpret_cast<const half2*>(x + static_cast<std::int64_t>(global_m) * K + block_k + k)
                       : __float2half2_rn(0.0f);
    }
    if constexpr (!Async) {
    for (int vector_index = tid; vector_index < (BLOCK_N * BLOCK_K) / 16; vector_index += THREADS) {
      const int first = vector_index * 16;
      int n;
      int k;
      if constexpr (Layout == W8BLayout::NK) {
        n = first / BLOCK_K;
        k = first - n * BLOCK_K;
      } else {
        k = first / BLOCK_N;
        n = first - k * BLOCK_N;
      }
      const int global_n = block_n + n;
      const int global_k = block_k + k;
      int4 q;
      if constexpr (Layout == W8BLayout::NK) {
        q = *reinterpret_cast<const int4*>(weight + static_cast<std::int64_t>(global_n) * K + global_k);
      } else {
        q = *reinterpret_cast<const int4*>(weight + static_cast<std::int64_t>(global_k) * N + global_n);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q_index = first + i * 2;
        int qn;
        int qk;
        if constexpr (Layout == W8BLayout::NK) {
          qn = q_index / BLOCK_K;
          qk = q_index - qn * BLOCK_K;
        } else {
          qn = q_index / BLOCK_N;
          qk = q_index - qn * BLOCK_N;
        }
        half* tile_value = Layout == W8BLayout::NK ? &weight_tile.nk[qn][qk] : &weight_tile.kn[qn][qk];
        *reinterpret_cast<half2*>(tile_value) =
            __halves2half2(__int2half_rn(static_cast<int>(int4_byte(q, i * 2))),
                           __int2half_rn(static_cast<int>(int4_byte(q, i * 2 + 1))));
      }
    }
    }
    if constexpr (Async) {
      if (block_k + BLOCK_K < K) {
        prefetch_q_tile<BLOCK_N, BLOCK_K, Layout, THREADS>(
            q_storage.data, block_k + BLOCK_K,
            block_n, K, N, weight, tid);
      }
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    if constexpr (Layout == W8BLayout::NK) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WARP_M_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[WARP_N_TILES];
#pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk += 16) {
#pragma unroll
        for (int wm = 0; wm < WARP_M_TILES; ++wm) {
          wmma::load_matrix_sync(a_frag[wm], &x_tile[(warp_m * WARP_M_TILES + wm) * 16][kk], BLOCK_K);
        }
#pragma unroll
        for (int wn = 0; wn < WARP_N_TILES; ++wn) {
          wmma::load_matrix_sync(b_frag[wn], &weight_tile.nk[(warp_n * WARP_N_TILES + wn) * 16][kk], BLOCK_K);
        }
#pragma unroll
        for (int wm = 0; wm < WARP_M_TILES; ++wm) {
#pragma unroll
          for (int wn = 0; wn < WARP_N_TILES; ++wn) {
            wmma::mma_sync(acc[wm][wn], a_frag[wm], b_frag[wn], acc[wm][wn]);
          }
        }
      }
    } else {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WARP_M_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WARP_N_TILES];
#pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk += 16) {
#pragma unroll
        for (int wm = 0; wm < WARP_M_TILES; ++wm) {
          wmma::load_matrix_sync(a_frag[wm], &x_tile[(warp_m * WARP_M_TILES + wm) * 16][kk], BLOCK_K);
        }
#pragma unroll
        for (int wn = 0; wn < WARP_N_TILES; ++wn) {
          wmma::load_matrix_sync(b_frag[wn], &weight_tile.kn[kk][(warp_n * WARP_N_TILES + wn) * 16], BLOCK_N);
        }
#pragma unroll
        for (int wm = 0; wm < WARP_M_TILES; ++wm) {
#pragma unroll
          for (int wn = 0; wn < WARP_N_TILES; ++wn) {
            wmma::mma_sync(acc[wm][wn], a_frag[wm], b_frag[wn], acc[wm][wn]);
          }
        }
      }
    }
    __syncthreads();
    if constexpr (Async) {
      if (block_k + BLOCK_K < K) {
        cp_async_wait();
        __syncthreads();
        convert_q_tile<BLOCK_N, BLOCK_K, Layout, THREADS>(
            q_storage.data, weight_tile, tid);
        __syncthreads();
      }
    }
  }

  const int warp = tid >> 5;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  if constexpr (BLOCK_N == 128) {
    for (int group = 0; group < 2; ++group) {
#pragma unroll
      for (int wm = 0; wm < WARP_M_TILES; ++wm) {
#pragma unroll
        for (int wn = 0; wn < WARP_N_TILES; ++wn) {
          if (warp_n / GROUP_WARPS == group) {
            wmma::store_matrix_sync(
                &output_tile[(warp_m * WARP_M_TILES + wm) * 16]
                             [(warp_n % GROUP_WARPS * WARP_N_TILES + wn) * 16],
                acc[wm][wn], OUTPUT_N, wmma::mem_row_major);
          }
        }
      }
      __syncthreads();
      for (int index = tid; index < (BLOCK_M * OUTPUT_N) / 2; index += THREADS) {
        const int m = index / (OUTPUT_N / 2);
        const int n = (index - m * (OUTPUT_N / 2)) * 2;
        const int global_m = block_m + m;
        if (global_m < M) {
          const int global_n = block_n + group * OUTPUT_N + n;
          const float2 values = make_float2(output_tile[m][n], output_tile[m][n + 1]);
          const float2 scale_values = __half22float2(
              *reinterpret_cast<const half2*>(scales + global_n));
          *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(global_m) * N + global_n) =
              __floats2half2_rn(values.x * scale_values.x, values.y * scale_values.y);
        }
      }
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int wm = 0; wm < WARP_M_TILES; ++wm) {
#pragma unroll
      for (int wn = 0; wn < WARP_N_TILES; ++wn) {
        wmma::store_matrix_sync(
            &output_tile[(warp_m * WARP_M_TILES + wm) * 16][(warp_n * WARP_N_TILES + wn) * 16],
            acc[wm][wn], OUTPUT_N, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int index = tid; index < (BLOCK_M * OUTPUT_N) / 2; index += THREADS) {
      const int m = index / (OUTPUT_N / 2);
      const int n = (index - m * (OUTPUT_N / 2)) * 2;
      const int global_m = block_m + m;
      if (global_m < M) {
        const int global_n = block_n + n;
        const float2 values = make_float2(output_tile[m][n], output_tile[m][n + 1]);
        const float2 scale_values = __half22float2(
            *reinterpret_cast<const half2*>(scales + global_n));
        *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(global_m) * N + global_n) =
            __floats2half2_rn(values.x * scale_values.x, values.y * scale_values.y);
      }
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREADS, bool Async = false>
void launch_gemm(
    cudaStream_t stream,
    int M,
    int K,
    int N,
    const half* x,
    const std::int8_t* weight,
    const half* scales,
    W8BLayout layout,
    half* y) {
  const dim3 grid(static_cast<unsigned>(N / BLOCK_N), static_cast<unsigned>((M + BLOCK_M - 1) / BLOCK_M), 1);
  if (layout == W8BLayout::NK) {
    w8a16_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, W8BLayout::NK, THREADS, Async><<<grid, THREADS, 0, stream>>>(M, K, N, x, weight, scales, y);
  } else {
    w8a16_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, W8BLayout::KN, THREADS, Async><<<grid, THREADS, 0, stream>>>(M, K, N, x, weight, scales, y);
  }
}

struct W8FragA {
  half2 value[4];
};

struct W8FragB {
  half2 value[2];
};

struct W8FragC {
  float value[4];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <bool EvictFirst>
__device__ __forceinline__ void w8_cp_async_16(
    void* shared_ptr, const void* global_ptr, bool pred) {
  const unsigned shared_address = __cvta_generic_to_shared(shared_ptr);
  if constexpr (EvictFirst) {
    asm volatile(
        "{ .reg .pred p; .reg .b64 policy; setp.ne.b32 p, %0, 0; "
        "createpolicy.fractional.L2::evict_first.b64 policy, 1.0; "
        "@p cp.async.cg.shared.global.L2::cache_hint [%1], [%2], 16, policy; }"
        :: "r"(static_cast<int>(pred)), "r"(shared_address), "l"(global_ptr));
  } else {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %0, 0; "
        "@p cp.async.cg.shared.global [%1], [%2], 16; }"
        :: "r"(static_cast<int>(pred)), "r"(shared_address), "l"(global_ptr));
  }
}

__device__ __forceinline__ void w8_cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int Wait>
__device__ __forceinline__ void w8_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(Wait));
}

__device__ __forceinline__ void w8_ldmatrix(W8FragA& frag, const void* shared_ptr) {
  uint32_t* values = reinterpret_cast<uint32_t*>(frag.value);
  const unsigned shared_address = __cvta_generic_to_shared(shared_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
      : "r"(shared_address));
}

__device__ __forceinline__ W8FragB w8_dequant_packed(uint32_t word) {
  const uint32_t lo_bits = __byte_perm(word, 0x64646464u, 0x4140u);
  const uint32_t hi_bits = __byte_perm(word, 0x64646464u, 0x4342u);
  constexpr uint32_t kMagic1152 = 0x64806480u;
  W8FragB result;
  result.value[0] = __hsub2(
      *reinterpret_cast<const half2*>(&lo_bits),
      *reinterpret_cast<const half2*>(&kMagic1152));
  result.value[1] = __hsub2(
      *reinterpret_cast<const half2*>(&hi_bits),
      *reinterpret_cast<const half2*>(&kMagic1152));
  return result;
}

__device__ __forceinline__ int w8_raw_physical_col(int row, int col) {
  return ((col >> 4) ^ (row & 3)) * 16 + (col & 15);
}

__device__ __forceinline__ W8FragB w8_dequant_raw(const uint8_t* tile, int stride, int k, int n) {
  const auto shifted = [&](int row) -> uint32_t {
    const int value = static_cast<int>(static_cast<int8_t>(tile[row * stride + w8_raw_physical_col(row, n)])) + 128;
    return static_cast<uint32_t>(value & 0xff);
  };
  const uint32_t word = shifted(k) | (shifted(k + 1) << 8) |
                        (shifted(k + 8) << 16) | (shifted(k + 9) << 24);
  return w8_dequant_packed(word);
}

__device__ __forceinline__ half* w8_x_addr(half* stage, int row, int chunk) {
  const int physical_chunk = chunk ^ (row & 7);
  return stage + row * 64 + physical_chunk * 8;
}

__device__ __forceinline__ void w8_mma(
    const W8FragA& a, const W8FragB& b, W8FragC& c) {
  const uint32_t* av = reinterpret_cast<const uint32_t*>(a.value);
  const uint32_t* bv = reinterpret_cast<const uint32_t*>(b.value);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(c.value[0]), "=f"(c.value[1]), "=f"(c.value[2]), "=f"(c.value[3])
      : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]),
        "r"(bv[0]), "r"(bv[1]),
        "f"(c.value[0]), "f"(c.value[1]), "f"(c.value[2]), "f"(c.value[3]));
}
#endif

template <int BLOCK_M, bool Packed>
__global__ __launch_bounds__(256, 1) void w8a16_mma_kernel(
    int M,
    int K,
    int N,
    const half* __restrict__ x,
    const std::int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    int k_begin,
    int k_end,
    half* __restrict__ y,
    float* __restrict__ scratch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int THREADS = 256;
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int STAGES = 4;
  constexpr int M_SUBS = BLOCK_M / 16;
  constexpr int REDUCTION_J = BLOCK_M >= 128 ? 4 : 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wn = warp & 1;
  const int wk = warp >> 1;
  const int block_m = static_cast<int>(blockIdx.x) * BLOCK_M;
  const int block_n = static_cast<int>(blockIdx.y) * BN;
  const int total_k_tiles = k_end - k_begin;
  const int tiles_per_split = (total_k_tiles + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z);
  const int local_k_begin = k_begin + static_cast<int>(blockIdx.z) * tiles_per_split;
  const int local_k_end = min(k_end, local_k_begin + tiles_per_split);

  const std::size_t x_stage_bytes = static_cast<std::size_t>(BLOCK_M) * BK * sizeof(half);
  const std::size_t b_stage_bytes = static_cast<std::size_t>(BK) * BN * sizeof(std::uint8_t);
  extern __shared__ unsigned char shared[];
  half* x_shared = reinterpret_cast<half*>(shared);
  std::uint8_t* b_shared = shared + STAGES * x_stage_bytes;
  float* reduction = reinterpret_cast<float*>(b_shared + STAGES * b_stage_bytes);

  W8FragC acc[M_SUBS][4];
#pragma unroll
  for (int m = 0; m < M_SUBS; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][j].value[i] = 0.0f;
    }
  }

  const int k_tiles = max(0, local_k_end - local_k_begin);
  auto fetch_stage = [&](int stage, int tile, bool pred) {
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const int x_segments = BLOCK_M * 8;
    for (int index = tid; index < x_segments; index += THREADS) {
      const int row = index / 8;
      const int chunk = index & 7;
      const int global_row = block_m + row;
      const int global_k = (local_k_begin + tile) * BK + chunk * 8;
      w8_cp_async_16<false>(
          w8_x_addr(x_stage, row, chunk),
          x + static_cast<std::int64_t>(global_row) * K + global_k,
          pred && global_row < M);
    }
    std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    if constexpr (Packed) {
      const int nt = block_n / BN;
      const int kt16 = (local_k_begin + tile) * 4;
      const std::int64_t word_base =
          (static_cast<std::int64_t>(nt) * (K / 16) + kt16) * 8 * 32;
      w8_cp_async_16<true>(
          b_stage + tid * 16,
          weight + word_base * 4 + static_cast<std::int64_t>(tid) * 16,
          pred);
    } else {
      const int row = tid / 4;
      const int chunk = tid & 3;
      const int global_k = (local_k_begin + tile) * BK + row;
      w8_cp_async_16<false>(
          b_stage + row * BN + w8_raw_physical_col(row, chunk * 16),
          weight + static_cast<std::int64_t>(global_k) * N + block_n + chunk * 16,
          pred);
    }
    w8_cp_async_commit();
  };

  for (int stage = 0; stage < STAGES - 1; ++stage) {
    fetch_stage(stage, stage, stage < k_tiles);
  }
  for (int tile = 0; tile < k_tiles; ++tile) {
    w8_cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int next = tile + STAGES - 1;
    fetch_stage((tile + STAGES - 1) % STAGES, next, next < k_tiles);
    const int stage = tile % STAGES;
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    W8FragA a[M_SUBS];
#pragma unroll
    for (int m = 0; m < M_SUBS; ++m) {
      const int row = m * 16 + (lane & 15);
      const int chunk = wk * 2 + (lane >> 4);
      w8_ldmatrix(a[m], w8_x_addr(x_stage, row, chunk));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      W8FragB b;
      if constexpr (Packed) {
        const int word_offset = (wk * 8 + wn * 4 + j) * 32 + lane;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(b_stage + word_offset * 4);
        b = w8_dequant_packed(word);
      } else {
        const int n0 = wn * 32 + j * 8 + (lane >> 2);
        b = w8_dequant_raw(b_stage, BN, wk * 16 + ((lane & 3) * 2), n0);
      }
#pragma unroll
      for (int m = 0; m < M_SUBS; ++m) w8_mma(a[m], b, acc[m][j]);
    }
    __syncthreads();
  }
  w8_cp_async_wait<0>();
  __syncthreads();

  const int warp_slot = wk * 2 + wn;
  if constexpr (REDUCTION_J == 4) {
    for (int m = 0; m < M_SUBS; ++m) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::size_t index = (((static_cast<std::size_t>(j) * 8 + warp_slot) * 32 + lane) * 4 + i);
          reduction[index] = acc[m][j].value[i];
        }
      }
      __syncthreads();
      if (wk == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          W8FragC total;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float sum = 0.0f;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
              const std::size_t index = (((static_cast<std::size_t>(j) * 8 + part * 2 + wn) * 32 + lane) * 4 + i);
              sum += reduction[index];
            }
            total.value[i] = sum;
          }
          const int g = lane >> 2;
          const int r = lane & 3;
          const int col = block_n + wn * 32 + j * 8 + 2 * r;
          const int row0 = block_m + m * 16 + g;
          const int row1 = row0 + 8;
          const bool split = scratch != nullptr;
          if (row0 < M) {
            if (split) {
              atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col, total.value[0]);
              atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col + 1, total.value[1]);
            } else {
              const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
              *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row0) * N + col) =
                  __floats2half2_rn(total.value[0] * s.x, total.value[1] * s.y);
            }
          }
          if (row1 < M) {
            if (split) {
              atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col, total.value[2]);
              atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col + 1, total.value[3]);
            } else {
              const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
              *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row1) * N + col) =
                  __floats2half2_rn(total.value[2] * s.x, total.value[3] * s.y);
            }
          }
        }
      }
      __syncthreads();
    }
  } else for (int m = 0; m < M_SUBS; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::size_t index = ((static_cast<std::size_t>(warp_slot) * 32 + lane) * 4 + i);
        reduction[index] = acc[m][j].value[i];
      }
      __syncthreads();
      if (wk == 0) {
        W8FragC total;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float sum = 0.0f;
#pragma unroll
          for (int part = 0; part < 4; ++part) {
            const std::size_t index = ((static_cast<std::size_t>(part * 2 + wn) * 32 + lane) * 4 + i);
            sum += reduction[index];
          }
          total.value[i] = sum;
        }
        const int g = lane >> 2;
        const int r = lane & 3;
        const int col = block_n + wn * 32 + j * 8 + 2 * r;
        const int row0 = block_m + m * 16 + g;
        const int row1 = row0 + 8;
        const bool split = scratch != nullptr;
        if (row0 < M) {
          if (split) {
            atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col, total.value[0]);
            atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col + 1, total.value[1]);
          } else {
            const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
            *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row0) * N + col) =
                __floats2half2_rn(total.value[0] * s.x, total.value[1] * s.y);
          }
        }
        if (row1 < M) {
          if (split) {
            atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col, total.value[2]);
            atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col + 1, total.value[3]);
          } else {
            const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
            *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row1) * N + col) =
                __floats2half2_rn(total.value[2] * s.x, total.value[3] * s.y);
          }
        }
      }
      __syncthreads();
    }
  }
#else
  (void)M; (void)K; (void)N; (void)x; (void)weight; (void)scales;
  (void)k_begin; (void)k_end; (void)y; (void)scratch;
#endif
}

template <bool Packed>
__global__ __launch_bounds__(256, 2) void w8a16_mma_bm64_bn128_kernel(
    int M,
    int K,
    int N,
    const half* __restrict__ x,
    const std::int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    int k_begin,
    int k_end,
    half* __restrict__ y,
    float* __restrict__ scratch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int THREADS = 256;
  constexpr int BLOCK_M = 64;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int STAGES = 3;
  constexpr int N_GROUPS_PER_WARP = 4;
  constexpr int WORDS_PER_64 = (BK / 16) * 8 * 32;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;
  const int block_m = static_cast<int>(blockIdx.x) * BLOCK_M;
  const int block_n = static_cast<int>(blockIdx.y) * BN;
  const int total_k_tiles = k_end - k_begin;
  const int tiles_per_split = (total_k_tiles + static_cast<int>(gridDim.z) - 1) /
                              static_cast<int>(gridDim.z);
  const int local_k_begin = k_begin + static_cast<int>(blockIdx.z) * tiles_per_split;
  const int local_k_end = min(k_end, local_k_begin + tiles_per_split);

  const std::size_t x_stage_bytes = static_cast<std::size_t>(BLOCK_M) * BK * sizeof(half);
  extern __shared__ unsigned char shared[];
  half* x_shared = reinterpret_cast<half*>(shared);
  std::uint8_t* b_shared = shared + STAGES * x_stage_bytes;

  W8FragC acc[2][N_GROUPS_PER_WARP];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][n].value[i] = 0.0f;
    }
  }

  const int k_tiles = max(0, local_k_end - local_k_begin);
  auto fetch_stage = [&](int stage, int tile, bool pred) {
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const int x_segments = BLOCK_M * 8;
    for (int index = tid; index < x_segments; index += THREADS) {
      const int row = index / 8;
      const int chunk = index & 7;
      const int global_row = block_m + row;
      const int global_k = (local_k_begin + tile) * BK + chunk * 8;
      const bool valid = pred && global_row < M;
      if (valid) {
        w8_cp_async_16<false>(
            w8_x_addr(x_stage, row, chunk),
            x + static_cast<std::int64_t>(global_row) * K + global_k,
            true);
      } else {
        *reinterpret_cast<uint4*>(w8_x_addr(x_stage, row, chunk)) = make_uint4(0, 0, 0, 0);
      }
    }
    std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    const int b_segments = (BK * BN) / 16;
    for (int vector = tid; vector < b_segments; vector += THREADS) {
      if constexpr (Packed) {
        const int group = vector / 256;
        const int local = vector & 255;
        const int nt = block_n / 64 + group;
        const int kt16 = (local_k_begin + tile) * 4;
        const std::int64_t word_base =
            (static_cast<std::int64_t>(nt) * (K / 16) + kt16) * 8 * 32;
        w8_cp_async_16<true>(
            b_stage + static_cast<std::size_t>(group) * BK * 64 + local * 16,
            weight + word_base * 4 + static_cast<std::int64_t>(local) * 16,
            pred);
      } else {
        const int chunks_per_row = BN / 16;
        const int row = vector / chunks_per_row;
        const int chunk = vector & (chunks_per_row - 1);
        const int global_k = (local_k_begin + tile) * BK + row;
        w8_cp_async_16<false>(
            b_stage + row * BN + w8_raw_physical_col(row, chunk * 16),
            weight + static_cast<std::int64_t>(global_k) * N + block_n + chunk * 16,
            pred);
      }
    }
    w8_cp_async_commit();
  };

  for (int stage = 0; stage < STAGES - 1; ++stage) {
    fetch_stage(stage, stage, stage < k_tiles);
  }
  for (int tile = 0; tile < k_tiles; ++tile) {
    w8_cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int next = tile + STAGES - 1;
    fetch_stage((tile + STAGES - 1) % STAGES, next, next < k_tiles);
    const int stage = tile % STAGES;
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    for (int k_part = 0; k_part < 4; ++k_part) {
      W8FragA a[2];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = (warp_m * 2 + m) * 16 + (lane & 15);
        const int chunk = k_part * 2 + (lane >> 4);
        w8_ldmatrix(a[m], w8_x_addr(x_stage, row, chunk));
      }
      W8FragB b[N_GROUPS_PER_WARP];
#pragma unroll
      for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
        const int n_group = warp_n * N_GROUPS_PER_WARP + n;
        if constexpr (Packed) {
          const int word_offset = (n_group / 8) * WORDS_PER_64 +
                                  (k_part * 8 + (n_group & 7)) * 32 + lane;
          const uint32_t word = *reinterpret_cast<const uint32_t*>(b_stage + word_offset * 4);
          b[n] = w8_dequant_packed(word);
        } else {
          const int n0 = n_group * 8 + (lane >> 2);
          b[n] = w8_dequant_raw(b_stage, BN, k_part * 16 + ((lane & 3) * 2), n0);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < N_GROUPS_PER_WARP; ++n) w8_mma(a[m], b[n], acc[m][n]);
      }
    }
    __syncthreads();
  }
  w8_cp_async_wait<0>();
  __syncthreads();

  const int g = lane >> 2;
  const int r = lane & 3;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
      const int col = block_n + (warp_n * N_GROUPS_PER_WARP + n) * 8 + 2 * r;
      const int row0 = block_m + (warp_m * 2 + m) * 16 + g;
      const int row1 = row0 + 8;
      const W8FragC& total = acc[m][n];
      const bool split = scratch != nullptr;
      if (row0 < M) {
        if (split) {
          atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col, total.value[0]);
          atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col + 1, total.value[1]);
        } else {
          const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
          *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row0) * N + col) =
              __floats2half2_rn(total.value[0] * s.x, total.value[1] * s.y);
        }
      }
      if (row1 < M) {
        if (split) {
          atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col, total.value[2]);
          atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col + 1, total.value[3]);
        } else {
          const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
          *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row1) * N + col) =
              __floats2half2_rn(total.value[2] * s.x, total.value[3] * s.y);
        }
      }
    }
  }
#else
  (void)M; (void)K; (void)N; (void)x; (void)weight; (void)scales;
  (void)k_begin; (void)k_end; (void)y; (void)scratch;
#endif
}

template <bool Packed, int BN>
__global__ __launch_bounds__(512, 1) void w8a16_mma_bm128_kernel(
    int M,
    int K,
    int N,
    const half* __restrict__ x,
    const std::int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    int k_begin,
    int k_end,
    half* __restrict__ y,
    float* __restrict__ scratch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int THREADS = 512;
  constexpr int BLOCK_M = 128;
  constexpr int BK = 64;
  constexpr int STAGES = 4;
  constexpr int N_GROUPS_PER_WARP = BN / 32;
  constexpr int WORDS_PER_64 = (BK / 16) * 8 * 32;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;
  const int block_m = static_cast<int>(blockIdx.x) * BLOCK_M;
  const int block_n = static_cast<int>(blockIdx.y) * BN;
  const int total_k_tiles = k_end - k_begin;
  const int tiles_per_split = (total_k_tiles + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z);
  const int local_k_begin = k_begin + static_cast<int>(blockIdx.z) * tiles_per_split;
  const int local_k_end = min(k_end, local_k_begin + tiles_per_split);
  const std::size_t x_stage_bytes = static_cast<std::size_t>(BLOCK_M) * BK * sizeof(half);
  extern __shared__ unsigned char shared[];
  half* x_shared = reinterpret_cast<half*>(shared);
  std::uint8_t* b_shared = shared + STAGES * x_stage_bytes;

  W8FragC acc[2][N_GROUPS_PER_WARP];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][n].value[i] = 0.0f;
    }
  }

  const int k_tiles = max(0, local_k_end - local_k_begin);
  auto fetch_stage = [&](int stage, int tile, bool pred) {
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const int x_segments = BLOCK_M * 8;
    for (int index = tid; index < x_segments; index += THREADS) {
      const int row = index / 8;
      const int chunk = index & 7;
      const int global_row = block_m + row;
      const int global_k = (local_k_begin + tile) * BK + chunk * 8;
      const bool valid = pred && global_row < M;
      if (valid) {
        w8_cp_async_16<false>(
            w8_x_addr(x_stage, row, chunk),
            x + static_cast<std::int64_t>(global_row) * K + global_k,
            true);
      } else {
        *reinterpret_cast<uint4*>(w8_x_addr(x_stage, row, chunk)) = make_uint4(0, 0, 0, 0);
      }
    }
    std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    if (tid < (BK * BN) / 16) {
      if constexpr (Packed) {
        const int group = tid / 256;
        const int local = tid & 255;
        const int nt = block_n / 64 + group;
        const int kt16 = (local_k_begin + tile) * 4;
        const std::int64_t word_base =
            (static_cast<std::int64_t>(nt) * (K / 16) + kt16) * 8 * 32;
        w8_cp_async_16<false>(
            b_stage + static_cast<std::size_t>(group) * BK * 64 + local * 16,
            weight + word_base * 4 + static_cast<std::int64_t>(local) * 16,
            pred);
      } else {
        const int chunks_per_row = BN / 16;
        const int row = tid / chunks_per_row;
        const int chunk = tid % chunks_per_row;
        const int global_k = (local_k_begin + tile) * BK + row;
        w8_cp_async_16<false>(
            b_stage + row * BN + w8_raw_physical_col(row, chunk * 16),
            weight + static_cast<std::int64_t>(global_k) * N + block_n + chunk * 16,
            pred);
      }
    }
    w8_cp_async_commit();
  };

  for (int stage = 0; stage < STAGES - 1; ++stage) {
    fetch_stage(stage, stage, stage < k_tiles);
  }
  for (int tile = 0; tile < k_tiles; ++tile) {
    w8_cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int next = tile + STAGES - 1;
    fetch_stage((tile + STAGES - 1) % STAGES, next, next < k_tiles);
    const int stage = tile % STAGES;
    half* x_stage = x_shared + static_cast<std::size_t>(stage) * BLOCK_M * BK;
    const std::uint8_t* b_stage = b_shared + static_cast<std::size_t>(stage) * BK * BN;
    for (int k_part = 0; k_part < 4; ++k_part) {
      W8FragA a[2];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = (warp_m * 2 + m) * 16 + (lane & 15);
        const int chunk = k_part * 2 + (lane >> 4);
        w8_ldmatrix(a[m], w8_x_addr(x_stage, row, chunk));
      }
      W8FragB b[N_GROUPS_PER_WARP];
#pragma unroll
      for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
        const int n_group = warp_n * N_GROUPS_PER_WARP + n;
        if constexpr (Packed) {
          const int word_offset = (n_group / 8) * WORDS_PER_64 +
                                  (k_part * 8 + (n_group & 7)) * 32 + lane;
          const uint32_t word = *reinterpret_cast<const uint32_t*>(b_stage + word_offset * 4);
          b[n] = w8_dequant_packed(word);
        } else {
          const int n0 = n_group * 8 + (lane >> 2);
          b[n] = w8_dequant_raw(b_stage, BN, k_part * 16 + ((lane & 3) * 2), n0);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < N_GROUPS_PER_WARP; ++n) w8_mma(a[m], b[n], acc[m][n]);
      }
    }
    __syncthreads();
  }
  w8_cp_async_wait<0>();
  __syncthreads();

  const int g = lane >> 2;
  const int r = lane & 3;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < N_GROUPS_PER_WARP; ++n) {
      const int col = block_n + (warp_n * N_GROUPS_PER_WARP + n) * 8 + 2 * r;
      const int row0 = block_m + (warp_m * 2 + m) * 16 + g;
      const int row1 = row0 + 8;
      const W8FragC& total = acc[m][n];
      const bool split = scratch != nullptr;
      if (row0 < M) {
        if (split) {
          atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col, total.value[0]);
          atomicAdd(scratch + static_cast<std::int64_t>(row0) * N + col + 1, total.value[1]);
        } else {
          const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
          *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row0) * N + col) =
              __floats2half2_rn(total.value[0] * s.x, total.value[1] * s.y);
        }
      }
      if (row1 < M) {
        if (split) {
          atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col, total.value[2]);
          atomicAdd(scratch + static_cast<std::int64_t>(row1) * N + col + 1, total.value[3]);
        } else {
          const float2 s = __half22float2(*reinterpret_cast<const half2*>(scales + col));
          *reinterpret_cast<half2*>(y + static_cast<std::int64_t>(row1) * N + col) =
              __floats2half2_rn(total.value[2] * s.x, total.value[3] * s.y);
        }
      }
    }
  }
#else
  (void)M; (void)K; (void)N; (void)x; (void)weight; (void)scales;
  (void)k_begin; (void)k_end; (void)y; (void)scratch;
#endif
}

__global__ void w8a16_zero_f32_kernel(float* data, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) data[index] = 0.0f;
}

__global__ void w8a16_scale_epilogue_kernel(
    const float* scratch, const half* scales, half* y, int M, int N) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(M) * N;
  if (index >= count) return;
  const int n = static_cast<int>(index % N);
  y[index] = __float2half_rn(scratch[index] * __half2float(scales[n]));
}

template <bool Packed>
void launch_mma_bm64_bn128(
    cudaStream_t stream,
    int M,
    int K,
    int N,
    const half* x,
    const std::int8_t* weight,
    const half* scales,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int force_split_k) {
  constexpr int BK = 64;
  constexpr int BN = 128;
  constexpr int STAGES = 3;
  const int k_tiles_total = K / BK;
  const int m_tiles = (M + 63) / 64;
  const int n_tiles = N / BN;
  int split_k = 1;
  if (force_split_k > 0) {
    split_k = force_split_k;
  } else if (const int tuned = lookup_tuned_split(K, N, Packed ? W8BLayout::PackedNK : W8BLayout::KN, M); tuned > 0) {
    split_k = tuned;
  } else if (workspace == nullptr || workspace_bytes < static_cast<std::size_t>(M) * N * sizeof(float)) {
    split_k = 1;
  } else {
    while (n_tiles * m_tiles * split_k < 2 * std::max(1, cached_device_info().sm_count) &&
           split_k * 2 <= k_tiles_total / 4) split_k *= 2;
  }
  if (split_k < 1) split_k = 1;
  if (split_k > k_tiles_total) split_k = k_tiles_total;
  float* scratch = nullptr;
  if (split_k > 1) {
    const std::size_t count = static_cast<std::size_t>(M) * N;
    const std::size_t required = count * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
      std::fprintf(stderr, "W8A16 split-K workspace too small: need %zu bytes, have %zu\n",
                   required, workspace_bytes);
      std::exit(1);
    }
    scratch = reinterpret_cast<float*>(workspace);
    constexpr int threads = 256;
    w8a16_zero_f32_kernel<<<static_cast<int>((count + threads - 1) / threads), threads, 0, stream>>>(
        scratch, count);
  }
  const std::size_t x_stage_bytes = static_cast<std::size_t>(64) * BK * sizeof(half);
  const std::size_t b_stage_bytes = static_cast<std::size_t>(BK) * BN;
  const std::size_t shared_bytes = STAGES * (x_stage_bytes + b_stage_bytes);
  static bool shared_limit_set = false;
  if (!shared_limit_set) {
    const cudaError_t status = cudaFuncSetAttribute(
        w8a16_mma_bm64_bn128_kernel<Packed>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (status != cudaSuccess) {
      std::fprintf(stderr, "set W8A16 BM64 BN128 shared memory failed: %s\n", cudaGetErrorString(status));
      std::exit(1);
    }
    shared_limit_set = true;
  }
  const dim3 grid(static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_tiles), static_cast<unsigned>(split_k));
  w8a16_mma_bm64_bn128_kernel<Packed><<<grid, 256, shared_bytes, stream>>>(
      M, K, N, x, weight, scales, 0, k_tiles_total, y, scratch);
  if (scratch != nullptr) {
    constexpr int threads = 256;
    const std::size_t count = static_cast<std::size_t>(M) * N;
    w8a16_scale_epilogue_kernel<<<static_cast<int>((count + threads - 1) / threads), threads, 0, stream>>>(
        scratch, scales, y, M, N);
  }
}

template <bool Packed, int BN>
void launch_mma_bm128(
    cudaStream_t stream,
    int M,
    int K,
    int N,
    const half* x,
    const std::int8_t* weight,
    const half* scales,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int force_split_k) {
  constexpr int BK = 64;
  constexpr int STAGES = 4;
  const int k_tiles_total = K / BK;
  const int m_tiles = (M + 127) / 128;
  const int n_tiles = N / BN;
  int split_k = 1;
  if (force_split_k > 0) {
    split_k = force_split_k;
  } else if (const int tuned = lookup_tuned_split(K, N, Packed ? W8BLayout::PackedNK : W8BLayout::KN, M); tuned > 0) {
    split_k = tuned;
  } else if (workspace == nullptr || workspace_bytes < static_cast<std::size_t>(M) * N * sizeof(float)) {
    split_k = 1;
  } else {
    while (n_tiles * m_tiles * split_k < 2 * std::max(1, cached_device_info().sm_count) &&
           split_k * 2 <= k_tiles_total / 4) split_k *= 2;
  }
  if (split_k < 1) split_k = 1;
  if (split_k > k_tiles_total) split_k = k_tiles_total;
  float* scratch = nullptr;
  if (split_k > 1) {
    const std::size_t required = static_cast<std::size_t>(M) * N * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
      std::fprintf(stderr, "W8A16 split-K workspace too small: need %zu bytes, have %zu\n",
                   required, workspace_bytes);
      std::exit(1);
    }
    scratch = reinterpret_cast<float*>(workspace);
    constexpr int threads = 256;
    w8a16_zero_f32_kernel<<<static_cast<int>((required / sizeof(float) + threads - 1) / threads), threads, 0, stream>>>(
        scratch, required / sizeof(float));
  }
  const std::size_t x_stage_bytes = static_cast<std::size_t>(128) * BK * sizeof(half);
  const std::size_t b_stage_bytes = static_cast<std::size_t>(BK) * BN;
  const std::size_t shared_bytes = STAGES * (x_stage_bytes + b_stage_bytes);
  static bool shared_limit_set = false;
  if (!shared_limit_set) {
    const cudaError_t status = cudaFuncSetAttribute(
        w8a16_mma_bm128_kernel<Packed, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (status != cudaSuccess) {
      std::fprintf(stderr, "set W8A16 BM128 shared memory failed: %s\n", cudaGetErrorString(status));
      std::exit(1);
    }
    shared_limit_set = true;
  }
  const dim3 grid(static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_tiles), static_cast<unsigned>(split_k));
  w8a16_mma_bm128_kernel<Packed, BN><<<grid, 512, shared_bytes, stream>>>(
      M, K, N, x, weight, scales, 0, k_tiles_total, y, scratch);
  if (scratch != nullptr) {
    constexpr int threads = 256;
    w8a16_scale_epilogue_kernel<<<static_cast<int>((static_cast<std::size_t>(M) * N + threads - 1) / threads), threads, 0, stream>>>(
        scratch, scales, y, M, N);
  }
}

template <int BLOCK_M, bool Packed>
void launch_mma_gemm(
    cudaStream_t stream,
    int M,
    int K,
    int N,
    const half* x,
    const std::int8_t* weight,
    const half* scales,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int force_split_k) {
  constexpr int BK = 64;
  constexpr int BN = 64;
  constexpr int STAGES = 4;
  const int k_tiles_total = K / BK;
  const int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
  const int n_tiles = N / BN;
  int split_k = 1;
  if (force_split_k > 0) {
    split_k = force_split_k;
  } else if (const int tuned = lookup_tuned_split(K, N, Packed ? W8BLayout::PackedNK : W8BLayout::KN, M); tuned > 0) {
    split_k = tuned;
  } else if (workspace == nullptr || workspace_bytes < static_cast<std::size_t>(M) * N * sizeof(float)) {
    split_k = 1;
  } else {
    while (n_tiles * m_tiles * split_k < 2 * std::max(1, cached_device_info().sm_count) &&
           split_k * 2 <= k_tiles_total / 4) split_k *= 2;
  }
  if (split_k < 1) split_k = 1;
  if (split_k > k_tiles_total) split_k = k_tiles_total;
  float* scratch = nullptr;
  if (split_k > 1) {
    const std::size_t required = static_cast<std::size_t>(M) * N * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
      std::fprintf(stderr, "W8A16 split-K workspace too small: need %zu bytes, have %zu\n",
                   required, workspace_bytes);
      std::exit(1);
    }
    scratch = reinterpret_cast<float*>(workspace);
    constexpr int threads = 256;
    w8a16_zero_f32_kernel<<<static_cast<int>((static_cast<std::size_t>(M) * N + threads - 1) / threads), threads, 0, stream>>>(
        scratch, static_cast<std::size_t>(M) * N);
  }
  const std::size_t x_stage_bytes = static_cast<std::size_t>(BLOCK_M) * BK * sizeof(half);
  const std::size_t b_stage_bytes = static_cast<std::size_t>(BK) * BN;
  constexpr int reduction_j = BLOCK_M >= 128 ? 4 : 1;
  const std::size_t shared_bytes = STAGES * (x_stage_bytes + b_stage_bytes) +
                                   static_cast<std::size_t>(reduction_j) * 8 * 32 * 4 * sizeof(float);
  static bool shared_limit_set = false;
  if (!shared_limit_set) {
    cudaFuncSetAttribute(w8a16_mma_kernel<BLOCK_M, Packed>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
    shared_limit_set = true;
  }
  const dim3 grid(static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_tiles), static_cast<unsigned>(split_k));
  w8a16_mma_kernel<BLOCK_M, Packed><<<grid, 256, shared_bytes, stream>>>(
      M, K, N, x, weight, scales, 0, k_tiles_total, y, scratch);
  if (scratch != nullptr) {
    constexpr int threads = 256;
    w8a16_scale_epilogue_kernel<<<static_cast<int>((static_cast<std::size_t>(M) * N + threads - 1) / threads), threads, 0, stream>>>(
        scratch, scales, y, M, N);
  }
}

}  // namespace

W8A16DeviceInfo rwkv7_w8a16_device_info() {
  return cached_device_info();
}

void rwkv7_w8a16_tuning_reset() {
  std::lock_guard<std::mutex> lock(g_tuning_mutex);
  g_tuning_splits.clear();
}

void rwkv7_w8a16_tuning_set(int K, int N, W8BLayout layout, int m_bucket, int split_k) {
  const int bucket = tune_m_bucket(m_bucket);
  if (K <= 0 || N <= 0 || bucket == 0 || split_k <= 0) return;
  std::lock_guard<std::mutex> lock(g_tuning_mutex);
  g_tuning_splits[{K, N, layout_key(layout), bucket}] = split_k;
}

int rwkv7_w8a16_tuning_get(int K, int N, W8BLayout layout, int M) {
  return lookup_tuned_split(K, N, layout, M);
}

void rwkv7_w8a16_linear_launch(
    cudaStream_t stream,
    int M,
    int K,
    int N,
    const half* x,
    const std::int8_t* qweight,
    const half* scale,
    W8BLayout layout,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int force_split_k) {
  if (M <= 0 || K <= 0 || N <= 0) return;
  assert(N % 64 == 0);
  assert(K % 64 == 0);
  const int major = cached_device_info().compute_major;
  if (major >= 8 && (layout == W8BLayout::PackedNK || layout == W8BLayout::KN)) {
    // This exact M=32 boundary is measured, not a typo: BM64xBN128 regresses
    // M=16 by 2.60% (28.7038 -> 29.4487 ms) but improves M=32 by 1.17%
    // (34.2084 -> 33.8133 ms), placing the crossover between 16 and 32.
    // M=17..31 remain conservatively on gemm<32>; dynamic batching should A/B
    // M=20/24/28 before widening this condition.
    if (M == 32 && N % 128 == 0) {
      if (layout == W8BLayout::PackedNK) {
        launch_mma_bm64_bn128<true>(stream, M, K, N, x, qweight, scale, y,
                                    workspace, workspace_bytes, force_split_k);
      } else {
        launch_mma_bm64_bn128<false>(stream, M, K, N, x, qweight, scale, y,
                                     workspace, workspace_bytes, force_split_k);
      }
      return;
    }
    if (M <= 16) {
      if (layout == W8BLayout::PackedNK) {
        launch_mma_gemm<16, true>(stream, M, K, N, x, qweight, scale, y,
                                  workspace, workspace_bytes, force_split_k);
      } else {
        launch_mma_gemm<16, false>(stream, M, K, N, x, qweight, scale, y,
                                   workspace, workspace_bytes, force_split_k);
      }
    } else if (M <= 32) {
      if (layout == W8BLayout::PackedNK) {
        launch_mma_gemm<32, true>(stream, M, K, N, x, qweight, scale, y,
                                  workspace, workspace_bytes, force_split_k);
      } else {
        launch_mma_gemm<32, false>(stream, M, K, N, x, qweight, scale, y,
                                   workspace, workspace_bytes, force_split_k);
      }
    } else if (M <= 64) {
      if (layout == W8BLayout::PackedNK) {
        if (N % 128 == 0) {
          launch_mma_bm64_bn128<true>(stream, M, K, N, x, qweight, scale, y,
                                      workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_gemm<64, true>(stream, M, K, N, x, qweight, scale, y,
                                    workspace, workspace_bytes, force_split_k);
        }
      } else {
        if (N % 128 == 0) {
          launch_mma_bm64_bn128<false>(stream, M, K, N, x, qweight, scale, y,
                                       workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_gemm<64, false>(stream, M, K, N, x, qweight, scale, y,
                                     workspace, workspace_bytes, force_split_k);
        }
      }
    } else if (M <= 128) {
      if (N % 128 == 0) {
        if (layout == W8BLayout::PackedNK) {
          launch_mma_bm128<true, 128>(stream, M, K, N, x, qweight, scale, y,
                                      workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_bm128<false, 128>(stream, M, K, N, x, qweight, scale, y,
                                       workspace, workspace_bytes, force_split_k);
        }
      } else {
        if (layout == W8BLayout::PackedNK) {
          launch_mma_bm128<true, 64>(stream, M, K, N, x, qweight, scale, y,
                                     workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_bm128<false, 64>(stream, M, K, N, x, qweight, scale, y,
                                      workspace, workspace_bytes, force_split_k);
        }
      }
    } else {
      if (layout == W8BLayout::PackedNK) {
        if (N % 128 == 0) {
          launch_mma_bm128<true, 128>(stream, M, K, N, x, qweight, scale, y,
                                      workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_gemm<64, true>(stream, M, K, N, x, qweight, scale, y,
                                    workspace, workspace_bytes, force_split_k);
        }
      } else {
        if (N % 128 == 0) {
          launch_mma_bm128<false, 128>(stream, M, K, N, x, qweight, scale, y,
                                       workspace, workspace_bytes, force_split_k);
        } else {
          launch_mma_gemm<64, false>(stream, M, K, N, x, qweight, scale, y,
                                     workspace, workspace_bytes, force_split_k);
        }
      }
    }
    return;
  }
  if (layout == W8BLayout::PackedNK) {
    std::fprintf(stderr, "Packed W8A16 requires compute capability 8.0 or newer\n");
    std::exit(1);
  }
  if (layout == W8BLayout::NK && M <= 4) {
    if (M == 1) {
      assert(K % (128 * 16) == 0 && N % 4 == 0);
      w8a16_gemv_kernel<128, 4, 1><<<N / 4, 128, 0, stream>>>(M, K, N, x, qweight, scale, y);
    } else if (M == 2) {
      assert(K % (128 * 16) == 0 && N % 4 == 0);
      w8a16_gemv_kernel<128, 4, 2><<<N / 4, 128, 0, stream>>>(M, K, N, x, qweight, scale, y);
    } else if (M <= 4) {
      assert(K % (128 * 16) == 0 && N % 4 == 0);
      w8a16_gemv_kernel<128, 4, 4><<<N / 4, 128, 0, stream>>>(M, K, N, x, qweight, scale, y);
    }
    return;
  }
  if (N <= 4096 || (N % 128) != 0) {
    const bool use_nk_bk128 = layout == W8BLayout::NK && K >= 4096 && (K % 128) == 0;
    if (M <= 16) {
      if (use_nk_bk128) {
        launch_gemm<16, 64, 128, 128>(stream, M, K, N, x, qweight, scale, layout, y);
      } else {
        launch_gemm<16, 64, 64, 128, true>(stream, M, K, N, x, qweight, scale, layout, y);
      }
    } else if (M <= 32) {
      if (use_nk_bk128) {
        launch_gemm<32, 64, 128, 128>(stream, M, K, N, x, qweight, scale, layout, y);
      } else {
        launch_gemm<32, 64, 64, 128, true>(stream, M, K, N, x, qweight, scale, layout, y);
      }
    } else {
      if (use_nk_bk128) {
        launch_gemm<64, 64, 128, 128>(stream, M, K, N, x, qweight, scale, layout, y);
      } else {
        launch_gemm<64, 64, 64, 128, true>(stream, M, K, N, x, qweight, scale, layout, y);
      }
    }
    return;
  }
  if (M <= 16) {
    if (layout == W8BLayout::KN && K >= 8192 && (K % 128) == 0) {
      launch_gemm<16, 128, 128, 128>(stream, M, K, N, x, qweight, scale, layout, y);
    } else if (layout == W8BLayout::NK) {
      launch_gemm<16, 128, 64, 128, true>(stream, M, K, N, x, qweight, scale, layout, y);
    } else {
      launch_gemm<16, 128, 64, 128>(stream, M, K, N, x, qweight, scale, layout, y);
    }
  } else if (M <= 32) {
    if (layout == W8BLayout::KN && K >= 8192 && (K % 128) == 0) {
      launch_gemm<32, 128, 128, 256>(stream, M, K, N, x, qweight, scale, layout, y);
    } else if (layout == W8BLayout::NK) {
      launch_gemm<32, 128, 64, 256, true>(stream, M, K, N, x, qweight, scale, layout, y);
    } else {
      launch_gemm<32, 128, 64, 256>(stream, M, K, N, x, qweight, scale, layout, y);
    }
  } else {
    if (layout == W8BLayout::KN && K >= 8192 && (K % 128) == 0) {
      launch_gemm<32, 128, 128, 256>(stream, M, K, N, x, qweight, scale, layout, y);
    } else if (layout == W8BLayout::NK) {
      launch_gemm<64, 128, 64, 256, true>(stream, M, K, N, x, qweight, scale, layout, y);
    } else {
      launch_gemm<64, 128, 64, 256>(stream, M, K, N, x, qweight, scale, layout, y);
    }
  }
}
