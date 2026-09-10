#include "rwkv_state_tuning.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace rwkv7_state_tuning {
namespace {

constexpr int kN = 64;
constexpr float kWScale = -0.6065306597126334f; // -exp(-0.5)

template <typename T> __device__ __forceinline__ float as_float(T value);
template <> __device__ __forceinline__ float as_float(half value) {
  return __half2float(value);
}
template <> __device__ __forceinline__ float as_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ half from_float(float value) {
  return __float2half_rn(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float effective_w(float raw) {
  const float sigmoid = 1.0f / (1.0f + expf(-raw));
  return expf(kWScale * sigmoid);
}

template <typename IO>
__global__ __launch_bounds__(kN, 2) void exact_forward_kernel(
    int T, int H, const float *__restrict__ initial_state,
    const IO *__restrict__ r_ptr, const IO *__restrict__ w_ptr,
    const IO *__restrict__ k_ptr, const IO *__restrict__ v_ptr,
    const IO *__restrict__ a_ptr, const IO *__restrict__ b_ptr,
    IO *__restrict__ y_ptr, float *__restrict__ final_state,
    float *__restrict__ state_before) {
  const int batch = blockIdx.y;
  const int head = blockIdx.x;
  const int value = threadIdx.x;
  const long long bh = static_cast<long long>(batch) * H + head;
  const long long state_base = bh * kN * kN;
  float state[kN];
#pragma unroll
  for (int key = 0; key < kN; ++key) {
    state[key] = initial_state[state_base + key * kN + value];
  }

  __shared__ float r[kN], w[kN], k[kN], a[kN], b[kN];
  for (int t = 0; t < T; ++t) {
    const long long token =
        (static_cast<long long>(batch) * T + t) * H * kN + head * kN;
    const long long tape_base = (bh * T + t) * kN * kN;
#pragma unroll
    for (int key = 0; key < kN; ++key) {
      state_before[tape_base + key * kN + value] = state[key];
    }
    __syncthreads();
    r[value] = as_float(r_ptr[token + value]);
    w[value] = effective_w(as_float(w_ptr[token + value]));
    k[value] = as_float(k_ptr[token + value]);
    a[value] = as_float(a_ptr[token + value]);
    b[value] = as_float(b_ptr[token + value]);
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int key = 0; key < kN; ++key)
      sa += state[key] * a[key];
    const float vv = as_float(v_ptr[token + value]);
    float yy = 0.0f;
#pragma unroll
    for (int key = 0; key < kN; ++key) {
      state[key] = state[key] * w[key] + sa * b[key] + k[key] * vv;
      yy += state[key] * r[key];
    }
    y_ptr[token + value] = from_float<IO>(yy);
  }
#pragma unroll
  for (int key = 0; key < kN; ++key) {
    final_state[state_base + key * kN + value] = state[key];
  }
}

template <typename IO>
__global__ __launch_bounds__(kN, 1) void exact_backward_kernel(
    int T, int H, const IO *__restrict__ r_ptr, const IO *__restrict__ w_ptr,
    const IO *__restrict__ k_ptr, const IO *__restrict__ v_ptr,
    const IO *__restrict__ a_ptr, const IO *__restrict__ b_ptr,
    const IO *__restrict__ dy_ptr, const float *__restrict__ ds_final,
    const float *__restrict__ state_before, float *__restrict__ ds_initial,
    IO *__restrict__ dr_ptr, IO *__restrict__ dw_ptr, IO *__restrict__ dk_ptr,
    IO *__restrict__ dv_ptr, IO *__restrict__ da_ptr, IO *__restrict__ db_ptr) {
  const int batch = blockIdx.y;
  const int head = blockIdx.x;
  const int value = threadIdx.x;
  const long long bh = static_cast<long long>(batch) * H + head;
  const long long state_base = bh * kN * kN;
  float dstate[kN];
#pragma unroll
  for (int key = 0; key < kN; ++key) {
    dstate[key] = ds_final ? ds_final[state_base + key * kN + value] : 0.0f;
  }

  __shared__ float r[kN], w[kN], w_sig[kN], k[kN], v[kN], a[kN], b[kN], dy[kN];
  __shared__ float sa[kN], dsa[kN];
  __shared__ float dnew[kN * kN];

  for (int t = T - 1; t >= 0; --t) {
    const long long token =
        (static_cast<long long>(batch) * T + t) * H * kN + head * kN;
    const long long tape_base = (bh * T + t) * kN * kN;
    __syncthreads();
    r[value] = as_float(r_ptr[token + value]);
    const float raw_w = as_float(w_ptr[token + value]);
    w_sig[value] = 1.0f / (1.0f + expf(-raw_w));
    w[value] = expf(kWScale * w_sig[value]);
    k[value] = as_float(k_ptr[token + value]);
    v[value] = as_float(v_ptr[token + value]);
    a[value] = as_float(a_ptr[token + value]);
    b[value] = as_float(b_ptr[token + value]);
    dy[value] = as_float(dy_ptr[token + value]);
    __syncthreads(); // a[] is read across both warps below.
    float local_sa = 0.0f;
#pragma unroll
    for (int key = 0; key < kN; ++key) {
      local_sa += state_before[tape_base + key * kN + value] * a[key];
    }
    sa[value] = local_sa;
    __syncthreads();

    float local_dsa = 0.0f;
#pragma unroll
    for (int key = 0; key < kN; ++key) {
      const float d = dstate[key] + r[key] * dy[value];
      dnew[key * kN + value] = d;
      local_dsa += d * b[key];
    }
    dsa[value] = local_dsa;
    __syncthreads();

    // A thread represents one channel. The same index is a K row for the
    // row reductions and a V column for the column reductions.
    float dr = 0.0f;
    float dw_eff = 0.0f;
    float dk = 0.0f;
    float db = 0.0f;
    float da = 0.0f;
    float dv = 0.0f;
#pragma unroll
    for (int q = 0; q < kN; ++q) {
      const float previous_row = state_before[tape_base + value * kN + q];
      const float updated_row =
          previous_row * w[value] + sa[q] * b[value] + k[value] * v[q];
      const float row_grad = dnew[value * kN + q];
      dr += updated_row * dy[q];
      dw_eff += row_grad * previous_row;
      dk += row_grad * v[q];
      db += row_grad * sa[q];
      da += previous_row * dsa[q];
      dv += dnew[q * kN + value] * k[q];
    }
    const float sig = w_sig[value];
    dr_ptr[token + value] = from_float<IO>(dr);
    dw_ptr[token + value] =
        from_float<IO>(kWScale * dw_eff * w[value] * sig * (1.0f - sig));
    dk_ptr[token + value] = from_float<IO>(dk);
    dv_ptr[token + value] = from_float<IO>(dv);
    da_ptr[token + value] = from_float<IO>(da);
    db_ptr[token + value] = from_float<IO>(db);

#pragma unroll
    for (int key = 0; key < kN; ++key) {
      dstate[key] = dnew[key * kN + value] * w[key] + a[key] * dsa[value];
    }
  }
#pragma unroll
  for (int key = 0; key < kN; ++key) {
    ds_initial[state_base + key * kN + value] = dstate[key];
  }
}

template <typename IO>
void launch_forward(cudaStream_t stream, const WkvShape &s, const float *s0,
                    const void *r, const void *w, const void *k, const void *v,
                    const void *a, const void *b, void *y, float *sT,
                    WkvTapeView tape) {
  exact_forward_kernel<IO><<<dim3(s.heads, s.batch), kN, 0, stream>>>(
      s.time, s.heads, s0, static_cast<const IO *>(r),
      static_cast<const IO *>(w), static_cast<const IO *>(k),
      static_cast<const IO *>(v), static_cast<const IO *>(a),
      static_cast<const IO *>(b), static_cast<IO *>(y), sT, tape.state_before);
}

template <typename IO>
void launch_backward(cudaStream_t stream, const WkvShape &s, const void *r,
                     const void *w, const void *k, const void *v, const void *a,
                     const void *b, const void *dy, const float *dsT,
                     WkvTapeView tape, float *ds0, void *dr, void *dw, void *dk,
                     void *dv, void *da, void *db) {
  exact_backward_kernel<IO><<<dim3(s.heads, s.batch), kN, 0, stream>>>(
      s.time, s.heads, static_cast<const IO *>(r), static_cast<const IO *>(w),
      static_cast<const IO *>(k), static_cast<const IO *>(v),
      static_cast<const IO *>(a), static_cast<const IO *>(b),
      static_cast<const IO *>(dy), dsT, tape.state_before, ds0,
      static_cast<IO *>(dr), static_cast<IO *>(dw), static_cast<IO *>(dk),
      static_cast<IO *>(dv), static_cast<IO *>(da), static_cast<IO *>(db));
}

void validate(const WkvShape &shape, WkvTapeView tape) {
  const std::size_t required = wkv_tape_elements(shape);
  if (!tape.state_before || tape.state_before_elements < required) {
    throw std::invalid_argument("state-tuning WKV tape is null or too small");
  }
}

} // namespace

void wkv_forward(cudaStream_t stream, IoType io_type, const WkvShape &shape,
                 const float *initial_state, const void *r, const void *w,
                 const void *k, const void *v, const void *a, const void *b,
                 void *y, float *final_state, WkvTapeView tape) {
  validate(shape, tape);
  if (!initial_state || !r || !w || !k || !v || !a || !b || !y ||
      !final_state) {
    throw std::invalid_argument("null state-tuning WKV forward pointer");
  }
  if (io_type == IoType::F16) {
    launch_forward<half>(stream, shape, initial_state, r, w, k, v, a, b, y,
                         final_state, tape);
  } else {
    launch_forward<__nv_bfloat16>(stream, shape, initial_state, r, w, k, v, a,
                                  b, y, final_state, tape);
  }
}

void wkv_backward_input(cudaStream_t stream, IoType io_type,
                        const WkvShape &shape, const void *r, const void *w,
                        const void *k, const void *v, const void *a,
                        const void *b, const void *dy, const float *ds_final,
                        WkvTapeView tape, float *ds_initial, void *dr, void *dw,
                        void *dk, void *dv, void *da, void *db) {
  validate(shape, tape);
  if (!r || !w || !k || !v || !a || !b || !dy || !ds_initial || !dr || !dw ||
      !dk || !dv || !da || !db) {
    throw std::invalid_argument("null state-tuning WKV backward pointer");
  }
  if (io_type == IoType::F16) {
    launch_backward<half>(stream, shape, r, w, k, v, a, b, dy, ds_final, tape,
                          ds_initial, dr, dw, dk, dv, da, db);
  } else {
    launch_backward<__nv_bfloat16>(stream, shape, r, w, k, v, a, b, dy,
                                   ds_final, tape, ds_initial, dr, dw, dk, dv,
                                   da, db);
  }
}

} // namespace rwkv7_state_tuning
