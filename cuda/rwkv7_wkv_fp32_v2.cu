#include <assert.h>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "rwkv7_fast_v4_kernels.cuh"

namespace {

constexpr int N = 64;
constexpr int WARP_THREADS = 32;
constexpr int BLOCK_THREADS = 32;
constexpr int VALUE_TILE = 4;
constexpr int KEYS_PER_WARP = WARP_THREADS / VALUE_TILE;
constexpr float W_SCALE_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;

#ifdef _IO_FP16_
using io_t = __half;
__device__ __forceinline__ float io_to_float(io_t x) { return __half2float(x); }
__device__ __forceinline__ io_t float_to_io(float x) { return __float2half_rn(x); }
#else
using io_t = float;
__device__ __forceinline__ float io_to_float(float x) { return x; }
__device__ __forceinline__ float float_to_io(float x) { return x; }
#endif

__device__ __forceinline__ float w_eff(float w) {
  return exp2f(W_SCALE_LOG2_E / (1.0f + exp2f(NLOG2_E * w)));
}

__device__ __forceinline__ float load_io(const io_t* ptr, int64_t idx) {
  return io_to_float(__ldg(ptr + idx));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float warp_sum_broadcast(float x) {
  return __shfl_sync(0xffffffffu, warp_sum(x), 0);
}

__device__ __forceinline__ float block_sum_broadcast(float x) {
  __syncthreads(); // Protect the previous broadcast from the next reduction.
  __shared__ float partial[BLOCK_THREADS / WARP_THREADS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (BLOCK_THREADS / WARP_THREADS)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

// One warp owns four adjacent V columns. Groups of four lanes share a K index,
// preserving coalesced [K,V] state traffic for the latency-sensitive T=1 path.
__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32_kv_tile_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int h = static_cast<int>(blockIdx.x);
  const int b_id = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int value_lane = lane & (VALUE_TILE - 1);
  const int key_lane = lane / VALUE_TILE;
  const int value_index = static_cast<int>(blockIdx.z) * VALUE_TILE + value_lane;
  const int64_t token_base = static_cast<int64_t>(b_id) * C + h * N;
  float* state_base = state_ptr +
      (static_cast<int64_t>(b_id) * H + h) * N * N;

  float a_state = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < N; key_base += KEYS_PER_WARP) {
    const int key_index = key_base + key_lane;
    float contribution = state_base[key_index * N + value_index] *
        load_io(a_ptr, token_base + key_index);
#pragma unroll
    for (int offset = WARP_THREADS / 2; offset >= VALUE_TILE; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < VALUE_TILE) {
      a_state += contribution;
    }
  }
  a_state = __shfl_sync(0xffffffffu, a_state, value_lane);
  float value = lane < VALUE_TILE
      ? load_io(v_ptr, token_base + value_index)
      : 0.0f;
  value = __shfl_sync(0xffffffffu, value, value_lane);

  float result = 0.0f;
#pragma unroll
  for (int key_base = 0; key_base < N; key_base += KEYS_PER_WARP) {
    const int key_index = key_base + key_lane;
    const int64_t input_index = token_base + key_index;
    const int state_index = key_index * N + value_index;
    const float updated = state_base[state_index] *
            w_eff(load_io(w_ptr, input_index)) +
        load_io(b_ptr, input_index) * a_state +
        load_io(k_ptr, input_index) * value;
    state_base[state_index] = updated;
    float contribution = load_io(r_ptr, input_index) * updated;
#pragma unroll
    for (int offset = WARP_THREADS / 2; offset >= VALUE_TILE; offset >>= 1) {
      contribution += __shfl_down_sync(0xffffffffu, contribution, offset);
    }
    if (lane < VALUE_TILE) {
      result += contribution;
    }
  }
  if (lane < VALUE_TILE) {
    y_ptr[token_base + value_index] = float_to_io(result);
  }
}

template <int HeadSize>
__launch_bounds__(HeadSize, 2)
__global__ void wkv_fp32_v2_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int c_base = h * HeadSize;
  const int64_t bt_base = static_cast<int64_t>(b_id) * T * C + c_base;
  // Physical state ABI is [B,H,K,V]. Thread i owns one V column so the
  // per-key loads/stores are coalesced across adjacent threads.
  float* state_base = state_ptr +
      (static_cast<int64_t>(b_id) * H + h) * HeadSize * HeadSize;

  float state[HeadSize];
#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state[j] = state_base[j * HeadSize + i];
  }

  __shared__ float r[HeadSize];
  __shared__ float w[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  for (int t = 0; t < T; ++t) {
    const int64_t idx = bt_base + static_cast<int64_t>(t) * C + i;
    __syncthreads();
    r[i] = load_io(r_ptr, idx);
    w[i] = w_eff(load_io(w_ptr, idx));
    k[i] = load_io(k_ptr, idx);
    a[i] = load_io(a_ptr, idx);
    b[i] = load_io(b_ptr, idx);
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      sa += state[j] * a[j];
    }

    const float vi = load_io(v_ptr, idx);
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      float s = state[j];
      s = s * w[j] + sa * b[j] + k[j] * vi;
      y += s * r[j];
      state[j] = s;
    }
    y_ptr[idx] = float_to_io(y);
  }

#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state_base[j * HeadSize + i] = state[j];
  }
}

__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32_v2_small_warp_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int64_t state_base =
      (static_cast<int64_t>(b_id) * H + h) * N * N;

  for (int t = 0; t < T; ++t) {
    const int64_t token = (static_cast<int64_t>(b_id) * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = lane; j < N; j += WARP_THREADS) {
      sa += state_ptr[state_base + j * N + row] * load_io(a_ptr, token + j);
    }
    sa = warp_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = lane; j < N; j += WARP_THREADS) {
      const int64_t idx = token + j;
      const int64_t state_index = state_base + j * N + row;
      const float s = state_ptr[state_index] * w_eff(load_io(w_ptr, idx)) + vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_index] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = warp_sum(yy);
    if (lane == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
  }
}

__global__ __launch_bounds__(BLOCK_THREADS, 4) void wkv_fp32_v2_short_block_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int c_base = h * N;
  const int64_t state_base =
      (static_cast<int64_t>(b_id) * H + h) * N * N;

  for (int t = 0; t < T; ++t) {
    const int64_t token = (static_cast<int64_t>(b_id) * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      sa += state_ptr[state_base + j * N + row] * load_io(a_ptr, token + j);
    }
    sa = block_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      const int64_t idx = token + j;
      const int64_t state_index = state_base + j * N + row;
      const float s = state_ptr[state_index] * w_eff(load_io(w_ptr, idx)) + vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_index] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = block_sum_broadcast(yy);
    if (tid == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
    __syncthreads();
  }
}

}  // namespace

void rwkv7_wkv_fp32io16_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int H,
    int mode,
    float* state,
    const half* r,
    const half* w,
    const half* k,
    const half* v,
    const half* a,
    const half* b,
    half* y) {
  if (B <= 0 || T <= 0 || H <= 0 || C != static_cast<int64_t>(H) * N || B > 65535 || H > 65535)
    throw std::invalid_argument("invalid WKV dimensions or grid size");
  if (T == 1 && (mode == 2 || (mode == 0 && B == 1))) {
    wkv_fp32_kv_tile_kernel<<<dim3(H, B, N / VALUE_TILE), dim3(WARP_THREADS), 0, stream>>>(
        C,
        H,
        state,
        reinterpret_cast<const io_t*>(r),
        reinterpret_cast<const io_t*>(w),
        reinterpret_cast<const io_t*>(k),
        reinterpret_cast<const io_t*>(v),
        reinterpret_cast<const io_t*>(a),
        reinterpret_cast<const io_t*>(b),
        reinterpret_cast<io_t*>(y));
    return;
  }
  // The old automatic row-kernel thresholds were tuned for physical [V,K].
  // Keep that compatibility path only when explicitly requested with mode 2.
  const bool use_small = mode == 2;
  if (mode == 3) {
    wkv_fp32_v2_short_block_kernel<<<dim3(N, H, B), dim3(BLOCK_THREADS), 0, stream>>>(
        T,
        C,
        H,
        state,
        reinterpret_cast<const io_t*>(r),
        reinterpret_cast<const io_t*>(w),
        reinterpret_cast<const io_t*>(k),
        reinterpret_cast<const io_t*>(v),
        reinterpret_cast<const io_t*>(a),
        reinterpret_cast<const io_t*>(b),
        reinterpret_cast<io_t*>(y));
  } else if (use_small) {
    wkv_fp32_v2_small_warp_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0, stream>>>(
        T,
        C,
        H,
        state,
        reinterpret_cast<const io_t*>(r),
        reinterpret_cast<const io_t*>(w),
        reinterpret_cast<const io_t*>(k),
        reinterpret_cast<const io_t*>(v),
        reinterpret_cast<const io_t*>(a),
        reinterpret_cast<const io_t*>(b),
        reinterpret_cast<io_t*>(y));
  } else {
    wkv_fp32_v2_kernel<N><<<dim3(B * H), dim3(N), 0, stream>>>(
        T,
        C,
        H,
        state,
        reinterpret_cast<const io_t*>(r),
        reinterpret_cast<const io_t*>(w),
        reinterpret_cast<const io_t*>(k),
        reinterpret_cast<const io_t*>(v),
        reinterpret_cast<const io_t*>(a),
        reinterpret_cast<const io_t*>(b),
        reinterpret_cast<io_t*>(y));
  }
}
