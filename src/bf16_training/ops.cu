// Training-only BF16 copies of the inference linear/norm/mix semantics.
#include "ops.hpp"
#ifdef RWKV_USE_HIP
#include <hipblas/hipblas.h>
#else
#include <cublas_v2.h>
#endif
#include <cmath>
namespace rwkv_bf16_training {
namespace {
__device__ float sum(float v, float *s) {
  __syncthreads();
  int i = threadIdx.x;
  s[i] = v;
  __syncthreads();
  for (int k = blockDim.x / 2; k; k /= 2) {
    if (i < k)
      s[i] += s[i + k];
    __syncthreads();
  }
  return s[0];
}
__global__ void element(const bf16 *x, const bf16 *y, bf16 *z, long long n,
                        int C, int op) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = to_float(x[i]);
  if (op == 0)
    v += to_float(y[i]);
  else if (op == 1)
    v = tanhf(v);
  else if (op == 2)
    v = 1 / (1 + expf(-v));
  else if (op == 3)
    v = fmaxf(v, 0) * fmaxf(v, 0);
  else
    v += to_float(y[i % C]);
  z[i] = to_bf16(v);
}
__global__ void norm(int C, const bf16 *x, const bf16 *w, const bf16 *b,
                     bf16 *y, float eps) {
  __shared__ float s[256];
  size_t off = size_t(blockIdx.x) * C;
  float v = 0;
  for (int i = threadIdx.x; i < C; i += blockDim.x)
    v += to_float(x[off + i]);
  float mean = sum(v, s) / C;
  v = 0;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float d = to_float(x[off + i]) - mean;
    v += d * d;
  }
  float inv = rsqrtf(sum(v, s) / C + eps);
  for (int i = threadIdx.x; i < C; i += blockDim.x)
    y[off + i] = to_bf16((to_float(x[off + i]) - mean) * inv * to_float(w[i]) +
                         to_float(b[i]));
}
// Shift is updated by a separate kernel, after all t=0 readers finish.
__global__ void mix(int B, int T, int C, const bf16 *x, const bf16 *shift,
                    const bf16 *w, bf16 *y) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(B) * T * C)
    return;
  int c = i % C, t = (i / C) % T, b = i / (size_t(T) * C);
  float v = to_float(x[i]);
  float p = to_float(t ? x[i - C] : shift[b * C + c]);
  y[i] = to_bf16(v + (p - v) * to_float(w[c]));
}
__global__ void shift_last(int B, int T, int C, const bf16 *x, bf16 *s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B * C)
    s[i] = x[(size_t(i / C) * T + T - 1) * C + i % C];
}
__global__ void kk(int H, const bf16 *k, const bf16 *kw, const bf16 *a0,
                   const bf16 *a12, const bf16 *ka, bf16 *nk, bf16 *neg,
                   bf16 *kka) {
  __shared__ float s[64];
  int j = threadIdx.x, c = (blockIdx.x % H) * 64 + j;
  size_t i = size_t(blockIdx.x) * 64 + j;
  float v = to_float(k[i]), u = v * to_float(kw[c]);
  float inv = 1 / fmaxf(sqrtf(sum(u * u, s)), 1e-12f);
  float a = 1 / (1 + expf(-to_float(a0[c]) - to_float(a12[i])));
  float b = to_float(ka[c]);
  nk[i] = to_bf16(v * (1 - b + a * b));
  neg[i] = to_bf16(-u * inv);
  kka[i] = to_bf16(u * inv * a);
}
__global__ void post(int H, const bf16 *x, const bf16 *r, const bf16 *k,
                     const bf16 *v, const bf16 *rk, const bf16 *w,
                     const bf16 *b, const bf16 *g, bf16 *y) {
  __shared__ float s[64];
  int j = threadIdx.x, c = (blockIdx.x % H) * 64 + j;
  size_t i = size_t(blockIdx.x) * 64 + j;
  float a = to_float(x[i]);
  float mean = sum(a, s) / 64;
  float d = a - mean;
  float inv = rsqrtf(sum(d * d, s) / 64 + 64e-5f);
  float q = sum(to_float(r[i]) * to_float(k[i]) * to_float(rk[c]), s);
  y[i] =
      to_bf16((d * inv * to_float(w[c]) + to_float(b[c]) + q * to_float(v[i])) *
              to_float(g[i]));
}
void gemm(cudaStream_t st, int M, int K, int N, const bf16 *x, const bf16 *w,
          bf16 *y, bool orig) {
  float a = 1, b = 0;
#ifdef RWKV_USE_HIP
  struct Handle {
    hipblasHandle_t h;
    Handle() {
      if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipblasCreate");
    }
    ~Handle() { hipblasDestroy(h); }
  };
  static thread_local Handle h;
  hipblasSetStream(h.h, st);
  auto e = hipblasGemmEx(h.h, orig ? HIPBLAS_OP_T : HIPBLAS_OP_N, HIPBLAS_OP_N,
                         N, M, K, &a, w, HIPBLAS_R_16B, orig ? K : N, x,
                         HIPBLAS_R_16B, K, &b, y, HIPBLAS_R_16B, N,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  if (e != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 hipblasGemmEx");
#else
  struct Handle {
    cublasHandle_t h;
    Handle() {
      if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasCreate");
    }
    ~Handle() { cublasDestroy(h); }
  };
  static thread_local Handle h;
  cublasSetStream(h.h, st);
  auto e = cublasGemmEx(h.h, orig ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, N,
                        M, K, &a, w, CUDA_R_16BF, orig ? K : N, x, CUDA_R_16BF,
                        K, &b, y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 cublasGemmEx");
#endif
}
} // namespace
void rwkv7_v3a_linear_bf16_launch(cudaStream_t s, int M, int K, int N,
                                  const bf16 *x, const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, false);
}
void rwkv7_v3a_linear_bf16_orig_launch(cudaStream_t s, int M, int K, int N,
                                       const bf16 *x, const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, true);
}
void rwkv7_v3a_layer_norm_bf16_launch(cudaStream_t s, int rows, int C,
                                      const bf16 *x, const bf16 *w,
                                      const bf16 *b, bf16 *y, float eps) {
  norm<<<rows, 256, 0, s>>>(C, x, w, b, y, eps);
}
void rwkv7_v3a_add_bf16_launch(cudaStream_t s, const bf16 *x, const bf16 *y,
                               bf16 *z, long long n) {
  element<<<(n + 255) / 256, 256, 0, s>>>(x, y, z, n, 0, 0);
}
void rwkv7_act_tanh_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                           long long n) {
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 1);
}
void rwkv7_act_sigmoid_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                              long long n) {
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 2);
}
void rwkv7_relu_square_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                              long long n) {
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 3);
}
void rwkv7_add_vec_launch(cudaStream_t s, int C, const bf16 *x, const bf16 *v,
                          bf16 *y, long long n) {
  element<<<(n + 255) / 256, 256, 0, s>>>(x, v, y, n, C, 4);
}
void rwkv7_cmix_mix_launch(cudaStream_t s, int B, int T, int C, const bf16 *x,
                           bf16 *sh, const bf16 *w, bf16 *y) {
  mix<<<(size_t(B) * T * C + 255) / 256, 256, 0, s>>>(B, T, C, x, sh, w, y);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, x, sh);
}
void rwkv7_tmix_mix6_launch(cudaStream_t s, int B, int T, int C, const bf16 *x,
                            bf16 *sh, const bf16 *wr, const bf16 *ww,
                            const bf16 *wk, const bf16 *wv, const bf16 *wa,
                            const bf16 *wg, bf16 *r, bf16 *w, bf16 *k, bf16 *v,
                            bf16 *a, bf16 *g) {
  const bf16 *ws[] = {wr, ww, wk, wv, wa, wg};
  bf16 *ys[] = {r, w, k, v, a, g};
  for (int j = 0; j < 6; ++j)
    mix<<<(size_t(B) * T * C + 255) / 256, 256, 0, s>>>(B, T, C, x, sh, ws[j],
                                                        ys[j]);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, x, sh);
}
void rwkv7_tmix_kk_a_gate_launch(cudaStream_t s, int B, int T, int C, int H,
                                 const bf16 *k, const bf16 *kw, const bf16 *a0,
                                 const bf16 *a12, const bf16 *ka, bf16 *nk,
                                 bf16 *neg, bf16 *kka) {
  kk<<<B * T * H, 64, 0, s>>>(H, k, kw, a0, a12, ka, nk, neg, kka);
}
void rwkv7_tmix_lnx_rkvres_xg_launch(cudaStream_t s, int B, int T, int C, int H,
                                     const bf16 *x, const bf16 *r,
                                     const bf16 *k, const bf16 *v,
                                     const bf16 *rk, const bf16 *w,
                                     const bf16 *b, const bf16 *g, bf16 *y) {
  post<<<B * T * H, 64, 0, s>>>(H, x, r, k, v, rk, w, b, g, y);
}
} // namespace rwkv_bf16_training
namespace rwkv_bf16_training {
__global__ void transpose_kernel(int rows, int cols, const bf16 *x, bf16 *y) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < size_t(rows) * cols)
    y[(i % cols) * rows + i / cols] = x[i];
}
void transpose_bf16(int rows, int cols, const bf16 *x, bf16 *y) {
  transpose_kernel<<<(size_t(rows) * cols + 255) / 256, 256>>>(rows, cols, x,
                                                               y);
  gpu_check(cudaDeviceSynchronize(), "transpose BF16 weight");
}
} // namespace rwkv_bf16_training
