// Training-only BF16 copies of the inference linear/norm/mix semantics.
#include "rwkv/bf16_training/ops.hpp"
#ifdef RWKV_USE_HIP
#include <hipblas/hipblas.h>
#else
#include <cublasLt.h>
#include <cublas_v2.h>
#include <map>
#include <tuple>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
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
// Packed calls pass offsets with B lanes and T total rows.
__global__ void mix(int B, int T, int C, const int *offsets, const bf16 *x,
                    const bf16 *shift, const bf16 *w, bf16 *y) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(offsets ? 1 : B) * T * C)
    return;
  int c = i % C;
  LaneRow at = lane_row(offsets, B, T, i / C);
  float v = to_float(x[i]);
  float p = to_float(!at.first ? x[i - C] : shift[at.lane * C + c]);
  y[i] = to_bf16(v + (p - v) * to_float(w[c]));
}
// An empty packed lane keeps its incoming shift.
__global__ void shift_last(int B, int T, int C, const int *offsets,
                           const bf16 *x, bf16 *s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * C)
    return;
  int b = i / C;
  if (!offsets)
    s[i] = x[(size_t(b) * T + T - 1) * C + i % C];
  else if (offsets[b + 1] > offsets[b])
    s[i] = x[(size_t(offsets[b + 1]) - 1) * C + i % C];
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
thread_local bool consistent_forward = false;
thread_local bool tensor_forward = false;
thread_local int pin_rows = 64;
void gemm_raw(cudaStream_t st, int M, int K, int N, const bf16 *x,
              const bf16 *w, bf16 *y, bool orig) {
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
  // Keep FP32 reductions while allowing BF16 Tensor Cores in online mode.
  if (cublasSetMathMode(h.h, tensor_forward
                                ? CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
                                : CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 cublasSetMathMode");
  auto e = cublasGemmEx(
      h.h, orig ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, w,
      CUDA_R_16BF, orig ? K : N, x, CUDA_R_16BF, K, &b, y, CUDA_R_16BF, N,
      consistent_forward && !tensor_forward ? CUBLAS_COMPUTE_32F_PEDANTIC
                                             : CUBLAS_COMPUTE_32F,
      consistent_forward && !tensor_forward ? CUBLAS_GEMM_DEFAULT
                                             : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 cublasGemmEx");
#endif
}
#ifndef RWKV_USE_HIP
// Consistent policies pin one cuBLASLt algorithm per (K, N, layout, compute):
// chosen once at pin_rows rows, without workspace, split-K or any reduction
// scheme. Each output row is then computed by the same kernel with the same
// K order whatever M is, so prefill, decode, teacher forcing and packed lanes
// agree bit for bit, and a GEMM reads its weights once instead of per tile.
// An algorithm that cannot run is an error, never a silent fallback.
struct LtCheck {
  void operator()(cublasStatus_t status, const char *what) const {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(std::string("cuBLASLt ") + what + ": " +
                               std::to_string(int(status)));
  }
} lt_check;
struct LtLayouts {
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  // Row-major y[M,N] = x[M,K] W as column-major y^T = W^T x^T.
  LtLayouts(int M, int K, int N, bool orig) {
    lt_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, orig ? K : N,
                                        orig ? N : K, orig ? K : N),
             "weight layout");
    lt_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, M, K),
             "input layout");
    lt_check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, N),
             "output layout");
  }
  ~LtLayouts() {
    for (auto layout : {a, b, c})
      if (layout)
        cublasLtMatrixLayoutDestroy(layout);
  }
};
struct LtGemm {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
};
void gemm_pinned(cudaStream_t st, int M, int K, int N, const bf16 *x,
                 const bf16 *w, bf16 *y, bool orig) {
  // Pinned algorithms assume 16-byte aligned rows (every model width is a
  // multiple of 8 BF16 values).
  if (K % 8 || N % 8 ||
      (reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(w) |
       reinterpret_cast<std::uintptr_t>(y)) % 16)
    throw std::invalid_argument("consistent GEMM needs 16-byte aligned rows");
  struct Handle {
    cublasLtHandle_t h = nullptr;
    std::map<std::tuple<int, int, bool, bool, int>, LtGemm> pinned;
    Handle() { lt_check(cublasLtCreate(&h), "create"); }
    ~Handle() {
      for (auto &item : pinned)
        cublasLtMatmulDescDestroy(item.second.desc);
      cublasLtDestroy(h);
    }
  };
  static thread_local Handle h;
  auto [it, fresh] =
      h.pinned.try_emplace(std::make_tuple(K, N, orig, tensor_forward, pin_rows));
  LtGemm &g = it->second;
  if (fresh) {
    try {
      lt_check(cublasLtMatmulDescCreate(&g.desc,
                                        tensor_forward
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_32F_PEDANTIC,
                                        CUDA_R_32F),
               "matmul descriptor");
      const cublasOperation_t ta = orig ? CUBLAS_OP_T : CUBLAS_OP_N,
                              tb = CUBLAS_OP_N;
      lt_check(cublasLtMatmulDescSetAttribute(
                   g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta),
               "transpose A");
      lt_check(cublasLtMatmulDescSetAttribute(
                   g.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb),
               "transpose B");
      cublasLtMatmulPreference_t pref = nullptr;
      lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
      const size_t workspace = 0;
      const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE,
                     alignment = 16;
      cublasLtMatmulHeuristicResult_t result{};
      int found = 0;
      cublasStatus_t status =
          cublasLtMatmulPreferenceSetAttribute(
              pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
              sizeof workspace);
      for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        if (status == CUBLAS_STATUS_SUCCESS)
          status = cublasLtMatmulPreferenceSetAttribute(
              pref, attribute, &alignment, sizeof alignment);
      if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
            sizeof reduction);
      if (status == CUBLAS_STATUS_SUCCESS) {
        LtLayouts l(pin_rows, K, N, orig);
        status = cublasLtMatmulAlgoGetHeuristic(h.h, g.desc, l.a, l.b, l.c,
                                                l.c, pref, 1, &result, &found);
      }
      cublasLtMatmulPreferenceDestroy(pref);
      lt_check(status, "algorithm heuristic");
      if (!found)
        throw std::runtime_error("cuBLASLt found no pinned BF16 algorithm");
      g.algo = result.algo;
    } catch (...) {
      if (g.desc)
        cublasLtMatmulDescDestroy(g.desc);
      h.pinned.erase(it);
      throw;
    }
  }
  const float one = 1, zero = 0;
  LtLayouts l(M, K, N, orig);
  lt_check(cublasLtMatmul(h.h, g.desc, &one, w, l.a, x, l.b, &zero, y, l.c, y,
                          l.c, &g.algo, nullptr, 0, st),
           "pinned BF16 matmul");
}
#endif
// Online BF16 and strict policies make each row independent of the GEMM row
// count (CUDA: pinned cuBLASLt algorithm; HIP: fixed 32-row tiles) to keep
// prefill, decode and teacher forcing consistent. WKV always sees the true
// sequence length. The default policy lets cuBLAS choose per call.
void gemm(cudaStream_t st, int M, int K, int N, const bf16 *x, const bf16 *w,
          bf16 *y, bool orig) {
  if (!consistent_forward) {
    gemm_raw(st, M, K, N, x, w, y, orig);
    return;
  }
#ifndef RWKV_USE_HIP
  gemm_pinned(st, M, K, N, x, w, y, orig);
#else
  constexpr int tile = 32;
  thread_local DeviceBuffer<bf16> padded_x, padded_y;
  for (int row = 0; row < M; row += tile) {
    int count = std::min(tile, M - row);
    // Reuse staging buffers sized for the largest linear width.
    if (padded_x.n < size_t(tile) * K)
      padded_x.resize(size_t(tile) * K, "BF16 fixed-tile input");
    if (padded_y.n < size_t(tile) * N)
      padded_y.resize(size_t(tile) * N, "BF16 fixed-tile output");
    gpu_check(cudaMemcpyAsync(padded_x.p, x + size_t(row) * K,
                              size_t(count) * K * 2, cudaMemcpyDeviceToDevice,
                              st),
              "pad GEMM input");
    if (count < tile)
      gpu_check(cudaMemsetAsync(padded_x.p + size_t(count) * K, 0,
                                size_t(tile - count) * K * 2, st),
                "zero GEMM padding");
    gemm_raw(st, tile, K, N, padded_x.p, w, padded_y.p, orig);
    gpu_check(cudaMemcpyAsync(y + size_t(row) * N, padded_y.p,
                              size_t(count) * N * 2, cudaMemcpyDeviceToDevice,
                              st),
              "copy GEMM output");
  }
#endif
}
} // namespace
void set_consistent_forward(bool enabled, bool tensor_cores, int rows) {
  if (rows < 1)
    throw std::invalid_argument("pinned GEMM rows must be positive");
  consistent_forward = enabled;
  tensor_forward = tensor_cores;
  pin_rows = rows;
}
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
                           bf16 *sh, const bf16 *w, bf16 *y,
                           const int *offsets) {
  const size_t n = size_t(offsets ? 1 : B) * T * C;
  mix<<<(n + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh, w, y);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh);
}
void rwkv7_tmix_mix6_launch(cudaStream_t s, int B, int T, int C, const bf16 *x,
                            bf16 *sh, const bf16 *wr, const bf16 *ww,
                            const bf16 *wk, const bf16 *wv, const bf16 *wa,
                            const bf16 *wg, bf16 *r, bf16 *w, bf16 *k, bf16 *v,
                            bf16 *a, bf16 *g, const int *offsets) {
  const bf16 *ws[] = {wr, ww, wk, wv, wa, wg};
  bf16 *ys[] = {r, w, k, v, a, g};
  const size_t n = size_t(offsets ? 1 : B) * T * C;
  for (int j = 0; j < 6; ++j)
    mix<<<(n + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh, ws[j], ys[j]);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh);
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
