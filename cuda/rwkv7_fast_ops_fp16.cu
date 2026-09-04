#include <assert.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "rwkv7_fast_v4_kernels.cuh"

using dtype = half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__global__ void i8_pack_kernel(
    const std::int8_t* __restrict__ src_nk,
    std::uint32_t* __restrict__ dst_words,
    int N,
    int K,
    int words) {
  const int word_index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_index >= words) return;
  const int words_per_nt = (K / 16) * 8 * 32;
  const int nt = word_index / words_per_nt;
  int rem = word_index - nt * words_per_nt;
  const int kt = rem / (8 * 32);
  rem -= kt * 8 * 32;
  const int ns = rem / 32;
  const int lane = rem - ns * 32;
  const int g = lane / 4;
  const int r = lane & 3;
  const int n = nt * 64 + ns * 8 + g;
  const int k0 = kt * 16 + 2 * r;
  const std::int8_t* row = src_nk + static_cast<std::int64_t>(n) * K;
  const auto encode = [](std::int8_t value) -> std::uint32_t {
    return static_cast<std::uint32_t>(static_cast<unsigned int>(static_cast<int>(value) + 128) & 0xffu);
  };
  dst_words[word_index] = encode(row[k0]) | (encode(row[k0 + 1]) << 8) |
                          (encode(row[k0 + 8]) << 16) | (encode(row[k0 + 9]) << 24);
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void tmix_mix6_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int b = static_cast<int>(bt / T);
  const int t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  __half2 prev2;
  if (t == 0) {
    prev2 = load_h2(shift_state + static_cast<int64_t>(b) * C + c);
  } else {
    prev2 = load_h2(x + idx - C);
  }

  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float dx0 = prev.x - cur.x;
  const float dx1 = prev.y - cur.y;

  const float2 xr = __half22float2(load_h2(x_r + c));
  const float2 xw = __half22float2(load_h2(x_w + c));
  const float2 xk = __half22float2(load_h2(x_k + c));
  const float2 xv = __half22float2(load_h2(x_v + c));
  const float2 xa = __half22float2(load_h2(x_a + c));
  const float2 xg = __half22float2(load_h2(x_g + c));

  store_h2(out_r + idx, cur.x + dx0 * xr.x, cur.y + dx1 * xr.y);
  store_h2(out_w + idx, cur.x + dx0 * xw.x, cur.y + dx1 * xw.y);
  store_h2(out_k + idx, cur.x + dx0 * xk.x, cur.y + dx1 * xk.y);
  store_h2(out_v + idx, cur.x + dx0 * xv.x, cur.y + dx1 * xv.y);
  store_h2(out_a + idx, cur.x + dx0 * xa.x, cur.y + dx1 * xa.y);
  store_h2(out_g + idx, cur.x + dx0 * xg.x, cur.y + dx1 * xg.y);
}

__global__ void update_shift_state_last_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int b = static_cast<int>(pair_idx / c_pairs);
  const int c = static_cast<int>(pair_idx - static_cast<int64_t>(b) * c_pairs) << 1;
  const int64_t src_idx = (static_cast<int64_t>(b) * T + (T - 1)) * C + c;
  *reinterpret_cast<__half2*>(shift_state + static_cast<int64_t>(b) * C + c) = load_h2(x + src_idx);
}

__global__ void tmix_kk_a_gate_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka,
    int64_t bth_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t bth = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (bth >= bth_size) {
    return;
  }

  const int64_t h = bth % H;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = h * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx, kv.x * fmaf(av0, ka.x, 1.0f - ka.x), kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
}

__global__ void tmix_lnx_rkvres_xg_kernel(
    int C,
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  __shared__ float partial[2];
  const int bth = blockIdx.x;
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = lane >> 5;
  const int warp_lane = lane & 31;
  const int h = bth % H;
  const int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t idx = base + lane;
  const int64_t c = cbase + lane;

  const float xv = load_h1(x + idx);
  float sum = xv;
  sum = warp_sum(sum);
  if (warp_lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
  __syncthreads();

  const float d = xv - mean;
  float ss = d * d;
  ss = warp_sum(ss);
  if (warp_lane == 0) {
    partial[warp] = ss;
  }
  __syncthreads();
  const float var = (partial[0] + partial[1]) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);
  __syncthreads();

  const float rv = load_h1(r + idx);
  const float kv = load_h1(k + idx);
  const float vv = load_h1(v + idx);
  float dot = rv * kv * load_h1(r_k + c);
  dot = warp_sum(dot);
  if (warp_lane == 0) {
    partial[warp] = dot;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float y = (d * rstd * load_h1(weight + c) + load_h1(bias + c) + rkv * vv)
                  * load_h1(g + idx);
  store_h1(out + idx, y);
}

__global__ void tmix_vres_gate_kernel(
    int C,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    const dtype* __restrict__ v12,
    dtype* __restrict__ out,
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx % static_cast<int64_t>(C));
  const float vv = load_h1(v + idx);
  const float gate = sigmoid_fast(load_h1(v0 + c) + load_h1(v12 + idx));
  store_h1(out + idx, fmaf(load_h1(v_first + idx) - vv, gate, vv));
}

template<int THREADS>
__global__ void cmix_sparse_up_one_kernel(
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x);
  const auto p2 = reinterpret_cast<const __half2*>(shift_state);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + f, relu * relu);
    }
  }
}

template<int THREADS>
__global__ void cmix_sparse_up_rows_kernel(
    int T,
    int C,
    int F,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act) {
  const int f = blockIdx.x;
  const int row = blockIdx.y;
  const int b = row / T;
  const int t = row - b * T;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row) * C);
  const auto p2 = (t == 0)
      ? reinterpret_cast<const __half2*>(shift_state + static_cast<int64_t>(b) * C)
      : reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row - 1) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + static_cast<int64_t>(row) * F + f, relu * relu);
    }
  }
}

__global__ void cmix_sparse_copy_zero_one_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = C / 8;
  if (i < n4) {
    reinterpret_cast<int4*>(shift_state)[i] = reinterpret_cast<const int4*>(x)[i];
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

__global__ void cmix_sparse_copy_zero_rows_kernel(
    int B,
    int T,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    dtype* __restrict__ out,
    int64_t out_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < out_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
  const int64_t state_vec4 = static_cast<int64_t>(B) * (C / 8);
  if (i < state_vec4) {
    const int b = static_cast<int>(i / (C / 8));
    const int c4 = static_cast<int>(i - static_cast<int64_t>(b) * (C / 8));
    reinterpret_cast<int4*>(shift_state)[i] =
        reinterpret_cast<const int4*>(x + (static_cast<int64_t>(b) * T + (T - 1)) * C)[c4];
  }
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

__global__ void cmix_mix_kernel(
    int T,
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int c_pairs = C >> 1;
  const int64_t bt = pair_idx / c_pairs;
  const int c = static_cast<int>(pair_idx - bt * c_pairs) << 1;
  const int b = static_cast<int>(bt / T);
  const int t = static_cast<int>(bt - static_cast<int64_t>(b) * T);
  const int64_t idx = bt * C + c;

  const __half2 cur2 = load_h2(x + idx);
  const __half2 prev2 = (t == 0) ? load_h2(shift_state + static_cast<int64_t>(b) * C + c) : load_h2(x + idx - C);
  const float2 cur = __half22float2(cur2);
  const float2 prev = __half22float2(prev2);
  const float2 mix = __half22float2(load_h2(x_k + c));
  store_h2(out + idx, cur.x + (prev.x - cur.x) * mix.x, cur.y + (prev.y - cur.y) * mix.y);
}

__global__ void relu_square_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  const float x0 = fmaxf(v.x, 0.0f);
  const float x1 = fmaxf(v.y, 0.0f);
  store_h2(out + idx, x0 * x0, x1 * x1);
}

__global__ void act_tanh_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, tanhf(v.x), tanhf(v.y));
}

__global__ void act_sigmoid_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 v = __half22float2(load_h2(x + idx));
  store_h2(out + idx, sigmoid_fast(v.x), sigmoid_fast(v.y));
}

__global__ void add_vec_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ vec,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>((pair_idx % (C >> 1)) << 1);
  const int64_t idx = pair_idx * 2;
  const float2 xv = __half22float2(load_h2(x + idx));
  const float2 vv = __half22float2(load_h2(vec + c));
  store_h2(out + idx, xv.x + vv.x, xv.y + vv.y);
}

__global__ void f16_to_f32_kernel(const dtype* __restrict__ src, float* __restrict__ dst, int64_t elems) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elems) {
    return;
  }
  dst[idx] = load_h1(src + idx);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_one_kernel(
    int C,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* act_row = act + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act_row + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_one_kernel(
    int C,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(preact + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2), acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float v = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(v * v);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float v = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(v * v);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C + c_block * (2 * THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C + c_block * (2 * THREADS) + tid * 2),
      acc);
}

__device__ __forceinline__ __half2 load_i8_pair_as_half2(
    const std::int8_t* value_i8, std::int64_t index) {
  const char2 q = *reinterpret_cast<const char2*>(value_i8 + index);
  return __floats2half2_rn(static_cast<float>(q.x), static_cast<float>(q.y));
}

__device__ __forceinline__ float2 load_i8_pair_as_float2(
    const std::int8_t* value_i8, std::int64_t index) {
  const char2 q = *reinterpret_cast<const char2*>(value_i8 + index);
  return make_float2(static_cast<float>(q.x), static_cast<float>(q.y));
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_one_i8_kernel(
    int C,
    const dtype* __restrict__ preact,
    const std::int8_t* __restrict__ value_i8,
    const dtype* __restrict__ scale,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];
  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const int c_base = c_block * (2 * FFN_SPMV_THREADS);
  if (tid < FFN_TILE / 2) {
    const float2 values = __half22float2(load_h2(preact + start_f + tid * 2));
    const float relu0 = fmaxf(values.x, 0.0f);
    const float relu1 = fmaxf(values.y, 0.0f);
    vec_slice[tid * 2] = __float2half_rn(relu0 * relu0);
    vec_slice[tid * 2 + 1] = __float2half_rn(relu1 * relu1);
  }
  __syncthreads();
  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) warp_counts[warp_id] = __popc(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();
  if (tid < FFN_TILE && nonzero) nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  __syncthreads();
  float2 acc = make_float2(0.0f, 0.0f);
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const float vec = __half2float(vec_slice[nnz_ids[i]]);
    const float2 mat = __half22float2(load_i8_pair_as_half2(
        value_i8, static_cast<std::int64_t>(actual_f) * C + c_base + tid * 2));
    acc.x = fmaf(vec, mat.x, acc.x);
    acc.y = fmaf(vec, mat.y, acc.y);
  }
  const float2 s2 = __half22float2(*reinterpret_cast<const __half2*>(scale + c_base + tid * 2));
  atomicAdd(reinterpret_cast<__half2*>(out + c_base + tid * 2),
            __floats2half2_rn(acc.x * s2.x, acc.y * s2.y));
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_i8_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const std::int8_t* __restrict__ value_i8,
    const dtype* __restrict__ scale,
    dtype* __restrict__ out) {
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];
  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const int c_base = c_block * (2 * FFN_SPMV_THREADS);
  const dtype* pre_row = preact + static_cast<std::int64_t>(row) * F;
  if (tid < FFN_TILE / 2) {
    const float2 values = __half22float2(load_h2(pre_row + start_f + tid * 2));
    const float relu0 = fmaxf(values.x, 0.0f);
    const float relu1 = fmaxf(values.y, 0.0f);
    vec_slice[tid * 2] = __float2half_rn(relu0 * relu0);
    vec_slice[tid * 2 + 1] = __float2half_rn(relu1 * relu1);
  }
  __syncthreads();
  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) warp_counts[warp_id] = __popc(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();
  if (tid < FFN_TILE && nonzero) nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  __syncthreads();
  float2 acc = make_float2(0.0f, 0.0f);
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const float vec = __half2float(vec_slice[nnz_ids[i]]);
    const float2 mat = __half22float2(load_i8_pair_as_half2(
        value_i8, static_cast<std::int64_t>(actual_f) * C + c_base + tid * 2));
    acc.x = fmaf(vec, mat.x, acc.x);
    acc.y = fmaf(vec, mat.y, acc.y);
  }
  const float2 s2 = __half22float2(*reinterpret_cast<const __half2*>(scale + c_base + tid * 2));
  atomicAdd(reinterpret_cast<__half2*>(out + static_cast<std::int64_t>(row) * C + c_base + tid * 2),
            __floats2half2_rn(acc.x * s2.x, acc.y * s2.y));
}

__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_i8_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const std::int8_t* __restrict__ value_i8,
    const dtype* __restrict__ scale,
    dtype* __restrict__ out) {
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];
  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const int c_base = c_block * (2 * THREADS);
  const dtype* pre_row = preact + static_cast<std::int64_t>(row) * F;
#pragma unroll
  for (int u = 0; u < 1; ++u) {
    const int local_f = tid * 2 + u * (TILE / 2);
    const float2 values = __half22float2(load_h2(pre_row + start_f + local_f));
    const float relu0 = fmaxf(values.x, 0.0f);
    const float relu1 = fmaxf(values.y, 0.0f);
    vec_slice[local_f] = __float2half_rn(relu0 * relu0);
    vec_slice[local_f + 1] = __float2half_rn(relu1 * relu1);
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) nnz_ids[warp_prefix[group] + local_pos] = local_f;
  }
  __syncthreads();
  float2 acc = make_float2(0.0f, 0.0f);
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const float vec = __half2float(vec_slice[local_f]);
    const float2 mat = __half22float2(load_i8_pair_as_half2(
        value_i8, static_cast<std::int64_t>(actual_f) * C + c_base + tid * 2));
    acc.x = fmaf(vec, mat.x, acc.x);
    acc.y = fmaf(vec, mat.y, acc.y);
  }
  const float2 s2 = __half22float2(*reinterpret_cast<const __half2*>(scale + c_base + tid * 2));
  atomicAdd(reinterpret_cast<__half2*>(out + static_cast<std::int64_t>(row) * C + c_base + tid * 2),
            __floats2half2_rn(acc.x * s2.x, acc.y * s2.y));
}

template <int F_TILE, int C_TILE, bool DirectFloatWeights = false>
__global__ __launch_bounds__(256, 6) void cmix_sparse_spmv_relu_rows_t1024_i8_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const std::int8_t* __restrict__ value_i8,
    const dtype* __restrict__ scale,
    dtype* __restrict__ out) {
  constexpr int THREADS = 256;
  constexpr int TILE = F_TILE;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];
  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  constexpr int C_PARTS = C_TILE / (2 * THREADS);
  const int c_base = c_block * C_TILE;
  const dtype* pre_row = preact + static_cast<std::int64_t>(row) * F;
#pragma unroll
  for (int u = 0; u < TILE / 512; ++u) {
    const int local_f = tid * 2 + u * (TILE / 2);
    const float2 values = __half22float2(load_h2(pre_row + start_f + local_f));
    const float relu0 = fmaxf(values.x, 0.0f);
    const float relu1 = fmaxf(values.y, 0.0f);
    vec_slice[local_f] = __float2half_rn(relu0 * relu0);
    vec_slice[local_f + 1] = __float2half_rn(relu1 * relu1);
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < TILE / 256; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < TILE / 256; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) nnz_ids[warp_prefix[group] + local_pos] = local_f;
  }
  __syncthreads();
  float2 acc[C_PARTS];
#pragma unroll
  for (int part = 0; part < C_PARTS; ++part) acc[part] = make_float2(0.0f, 0.0f);
  for (int i = 0; i < nnz_count; ++i) {
    const int local_f = nnz_ids[i];
    const int actual_f = start_f + local_f;
    const float vec = __half2float(vec_slice[local_f]);
#pragma unroll
    for (int part = 0; part < C_PARTS; ++part) {
      const int c_offset = c_base + part * (2 * THREADS) + tid * 2;
      float2 mat;
      if constexpr (DirectFloatWeights) {
        mat = load_i8_pair_as_float2(
            value_i8, static_cast<std::int64_t>(actual_f) * C + c_offset);
      } else {
        mat = __half22float2(load_i8_pair_as_half2(
            value_i8, static_cast<std::int64_t>(actual_f) * C + c_offset));
      }
      acc[part].x = fmaf(vec, mat.x, acc[part].x);
      acc[part].y = fmaf(vec, mat.y, acc[part].y);
    }
  }
#pragma unroll
  for (int part = 0; part < C_PARTS; ++part) {
    const int c_offset = c_base + part * (2 * THREADS) + tid * 2;
    const float2 s2 = __half22float2(*reinterpret_cast<const __half2*>(scale + c_offset));
    atomicAdd(reinterpret_cast<__half2*>(out + static_cast<std::int64_t>(row) * C + c_offset),
              __floats2half2_rn(acc[part].x * s2.x, acc[part].y * s2.y));
  }
}

__global__ void cmix_stats_relu2_kernel(
    const dtype* __restrict__ preact,
    unsigned long long* __restrict__ nonzero,
    unsigned long long* __restrict__ total,
    unsigned int* __restrict__ max_bits,
    long long elements) {
  const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = __half2float(preact[index]);
  const float relu2 = value > 0.0f ? value * value : 0.0f;
  atomicAdd(total, 1ULL);
  if (relu2 > 0.0f) atomicAdd(nonzero, 1ULL);
  atomicMax(max_bits, __float_as_uint(relu2));
}

}  // namespace

void rwkv7_tmix_mix6_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    const half* x,
    half* shift_state,
    const half* x_r,
    const half* x_w,
    const half* x_k,
    const half* x_v,
    const half* x_a,
    const half* x_g,
    half* out_r,
    half* out_w,
    half* out_k,
    half* out_v,
    half* out_a,
    half* out_g) {
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  tmix_mix6_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      T, C,
      x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      out_r, out_w, out_k, out_v, out_a, out_g,
      total_pairs);
  const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
  update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
      T, C, x, shift_state, state_pairs);
}

void rwkv7_tmix_kk_a_gate_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int H,
    const half* k,
    const half* k_k,
    const half* a0,
    const half* a12,
    const half* k_a,
    half* new_k,
    half* neg_kk,
    half* kka) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  const int blocks = static_cast<int>(ceil_div(bth_size, static_cast<int64_t>(WARPS_PER_BLOCK)));
  tmix_kk_a_gate_kernel<<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
      H,
      k, k_k, a0, a12, k_a, new_k, neg_kk, kka,
      bth_size);
}

void rwkv7_tmix_lnx_rkvres_xg_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int H,
    const half* x,
    const half* r,
    const half* k,
    const half* v,
    const half* r_k,
    const half* weight,
    const half* bias,
    const half* g,
    half* out) {
  (void)C;
  assert(C == H * HEAD_SIZE);
  const int64_t bth_size = static_cast<int64_t>(B) * T * H;
  tmix_lnx_rkvres_xg_kernel<<<static_cast<int>(bth_size), HEAD_SIZE, 0, stream>>>(
      C, H, x, r, k, v, r_k, weight, bias, g, out, bth_size);
}

void rwkv7_tmix_vres_gate_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    const half* v,
    const half* v_first,
    const half* v0,
    const half* v12,
    half* out) {
  const int64_t total = static_cast<int64_t>(B) * T * C;
  constexpr int threads = 256;
  tmix_vres_gate_kernel<<<static_cast<int>(ceil_div(total, threads)), threads, 0, stream>>>(
      C, v, v_first, v0, v12, out, total);
}

void rwkv7_cmix_sparse_one_launch(
    cudaStream_t stream,
    int C,
    int F,
    const half* x,
    half* shift_state,
    const half* x_k,
    const half* key_fc,
    const half* value_fc,
    half* act_scratch,
    half* out) {
  cmix_sparse_up_one_kernel<64><<<F, 64, 0, stream>>>(
      C, x, shift_state, x_k, key_fc, act_scratch);
  cmix_sparse_copy_zero_one_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(
      x, shift_state, out, C);
  cmix_sparse_spmv_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C, act_scratch, value_fc, out);
}

void rwkv7_cmix_sparse_rows_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int F,
    const half* x,
    half* shift_state,
    const half* x_k,
    const half* key_fc,
    const half* value_fc,
    half* act_scratch,
    half* out) {
  const int rows = B * T;
  cmix_sparse_up_rows_kernel<64><<<dim3(F, rows, 1), 64, 0, stream>>>(
      T, C, F, x, shift_state, x_k, key_fc, act_scratch);
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  cmix_sparse_copy_zero_rows_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(
      B, T, C, x, shift_state, out, out_vec4);
  cmix_sparse_spmv_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C, F, act_scratch, value_fc, out);
}

void rwkv7_cmix_sparse_down_relu_one_launch(
    cudaStream_t stream,
    int C,
    int F,
    const half* preact,
    const half* value_fc,
    half* out) {
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out, C / 8);
  cmix_sparse_spmv_relu_one_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C, preact, value_fc, out);
}

void rwkv7_cmix_sparse_down_relu_rows_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int F,
    const half* preact,
    const half* value_fc,
    half* out) {
  const int rows = B * T;
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out, out_vec4);
  cmix_sparse_spmv_relu_rows_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C, F, preact, value_fc, out);
}

void rwkv7_cmix_sparse_down_relu_rows_t512_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int F,
    const half* preact,
    const half* value_fc,
    half* out) {
  const int rows = B * T;
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out, out_vec4);
  cmix_sparse_spmv_relu_rows_t512_kernel<<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
      C, F, preact, value_fc, out);
}

void rwkv7_cmix_sparse_down_relu_one_i8_launch(
    cudaStream_t stream,
    int C,
    int F,
    const half* preact,
    const std::int8_t* value_i8,
    const half* scale,
    half* out) {
  zero_vec4_kernel<<<(C / 8 + 127) / 128, 128, 0, stream>>>(out, C / 8);
  cmix_sparse_spmv_relu_one_i8_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), 1), FFN_SPMV_THREADS, 0, stream>>>(
      C, preact, value_i8, scale, out);
}

void rwkv7_cmix_sparse_down_relu_rows_i8_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int F,
    const half* preact,
    const std::int8_t* value_i8,
    const half* scale,
    half* out) {
  const int rows = B * T;
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out, out_vec4);
  cmix_sparse_spmv_relu_rows_i8_kernel<<<dim3(F / FFN_TILE, C / (2 * FFN_SPMV_THREADS), rows), FFN_SPMV_THREADS, 0, stream>>>(
      C, F, preact, value_i8, scale, out);
}

void rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    int F,
    const half* preact,
    const std::int8_t* value_i8,
    const half* scale,
    half* out) {
  const int rows = B * T;
  if (F < 512 || (F % 512) != 0) {
    rwkv7_cmix_sparse_down_relu_rows_i8_launch(stream, B, T, C, F, preact, value_i8, scale, out);
    return;
  }
  const int64_t out_vec4 = static_cast<int64_t>(rows) * (C / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(out_vec4, 128)), 128, 0, stream>>>(out, out_vec4);
  if (F < 1024 || (F % 1024) != 0) {
    cmix_sparse_spmv_relu_rows_t512_i8_kernel<<<dim3(F / 512, C / 512, rows), 256, 0, stream>>>(
        C, F, preact, value_i8, scale, out);
  } else if (rows >= 64 && (C % 4096) == 0) {
    if (rows > 64) {
      cmix_sparse_spmv_relu_rows_t1024_i8_kernel<1024, 4096, true><<<dim3(F / 1024, C / 4096, rows), 256, 0, stream>>>(
          C, F, preact, value_i8, scale, out);
    } else {
      cmix_sparse_spmv_relu_rows_t1024_i8_kernel<1024, 4096><<<dim3(F / 1024, C / 4096, rows), 256, 0, stream>>>(
          C, F, preact, value_i8, scale, out);
    }
  } else {
    cmix_sparse_spmv_relu_rows_t1024_i8_kernel<1024, 512><<<dim3(F / 1024, C / 512, rows), 256, 0, stream>>>(
        C, F, preact, value_i8, scale, out);
  }
}

void rwkv7_cmix_stats_relu2_launch(
    cudaStream_t stream,
    int rows,
    int F,
    const half* preact,
    unsigned long long* nonzero,
    unsigned long long* total,
    unsigned int* max_bits) {
  const long long elements = static_cast<long long>(rows) * F;
  constexpr int threads = 256;
  cmix_stats_relu2_kernel<<<static_cast<int>((elements + threads - 1) / threads), threads, 0, stream>>>(
      preact, nonzero, total, max_bits, elements);
}

void rwkv7_cmix_mix_launch(
    cudaStream_t stream,
    int B,
    int T,
    int C,
    const half* x,
    half* shift_state,
    const half* x_k,
    half* out) {
  constexpr int threads = 256;
  const int64_t total_pairs = static_cast<int64_t>(B) * T * (C / 2);
  cmix_mix_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      T, C, x, shift_state, x_k, out, total_pairs);
  const int64_t state_pairs = static_cast<int64_t>(B) * (C / 2);
  update_shift_state_last_kernel<<<static_cast<int>(ceil_div(state_pairs, threads)), threads, 0, stream>>>(
      T, C, x, shift_state, state_pairs);
}

void rwkv7_relu_square_launch(cudaStream_t stream, const half* x, half* out, long long elems) {
  constexpr int threads = 256;
  const int64_t total_pairs = elems / 2;
  relu_square_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x, out, total_pairs);
}

void rwkv7_act_tanh_launch(cudaStream_t stream, const half* x, half* out, long long elems) {
  constexpr int threads = 256;
  const int64_t total_pairs = elems / 2;
  act_tanh_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x, out, total_pairs);
}

void rwkv7_act_sigmoid_launch(cudaStream_t stream, const half* x, half* out, long long elems) {
  constexpr int threads = 256;
  const int64_t total_pairs = elems / 2;
  act_sigmoid_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x, out, total_pairs);
}

void rwkv7_add_vec_launch(
    cudaStream_t stream, int C, const half* x, const half* vec, half* out, long long elems) {
  constexpr int threads = 256;
  const int64_t total_pairs = elems / 2;
  add_vec_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      C, x, vec, out, total_pairs);
}

void rwkv7_v4_f16_to_f32_launch(
    cudaStream_t stream, const half* src_f16, float* dst_f32, long long elems) {
  constexpr int threads = 256;
  f16_to_f32_kernel<<<static_cast<int>(ceil_div(elems, static_cast<int64_t>(threads))), threads, 0, stream>>>(
      src_f16, dst_f32, elems);
}

void rwkv7_v4_i8_pack_launch(
    cudaStream_t stream, const std::int8_t* src_nk, std::int8_t* dst_packed, int N, int K) {
  if (N <= 0 || K <= 0) return;
  if ((N % 64) != 0 || (K % 16) != 0) {
    std::fprintf(stderr, "i8 pack requires N %% 64 == 0 and K %% 16 == 0 (N=%d K=%d)\n", N, K);
    std::exit(1);
  }
  const int words = N * K / 4;
  constexpr int threads = 256;
  i8_pack_kernel<<<static_cast<int>(ceil_div(words, threads)), threads, 0, stream>>>(
      src_nk, reinterpret_cast<std::uint32_t*>(dst_packed), N, K, words);
}
