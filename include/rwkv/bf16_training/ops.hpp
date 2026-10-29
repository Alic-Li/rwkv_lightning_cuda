#pragma once
#include "rwkv/bf16_training/runtime.hpp"
namespace rwkv_bf16_training {
// Thread-local GEMM policy. enabled makes every output row independent of the
// GEMM row count (pinned per-shape algorithm on CUDA, 32-row tiles on HIP)
// with FP32 reductions; tensor_cores selects BF16 Tensor Cores, otherwise
// pedantic FP32 accumulation (strict, bitwise comparison mode). Disabled, the
// default policy, lets cuBLAS choose a kernel per call. pin_rows is the row
// count each shape's kernel is chosen for: decode-heavy callers keep 64, the
// offline trainer uses 256. Results are consistent only among GEMMs made under
// the same pin_rows.
void set_consistent_forward(bool enabled, bool tensor_cores = false,
                            int pin_rows = 64);
// Token-shift mixes. With device offsets[B + 1], B is the lane count, T the
// packed row count, and shift_state holds one row per lane.
void rwkv7_tmix_mix6_launch(cudaStream_t stream, int B, int T, int C,
                            const bf16 *x, bf16 *shift_state, const bf16 *x_r,
                            const bf16 *x_w, const bf16 *x_k, const bf16 *x_v,
                            const bf16 *x_a, const bf16 *x_g, bf16 *out_r,
                            bf16 *out_w, bf16 *out_k, bf16 *out_v, bf16 *out_a,
                            bf16 *out_g, const int *offsets = nullptr);
void rwkv7_tmix_kk_a_gate_launch(cudaStream_t stream, int B, int T, int C,
                                 int H, const bf16 *k, const bf16 *k_k,
                                 const bf16 *a0, const bf16 *a12,
                                 const bf16 *k_a, bf16 *new_k, bf16 *neg_kk,
                                 bf16 *kka);
void rwkv7_tmix_lnx_rkvres_xg_launch(cudaStream_t stream, int B, int T, int C,
                                     int H, const bf16 *x, const bf16 *r,
                                     const bf16 *k, const bf16 *v,
                                     const bf16 *r_k, const bf16 *weight,
                                     const bf16 *bias, const bf16 *g,
                                     bf16 *out);
void rwkv7_cmix_mix_launch(cudaStream_t stream, int B, int T, int C,
                           const bf16 *x, bf16 *shift_state, const bf16 *x_k,
                           bf16 *out, const int *offsets = nullptr);
void rwkv7_relu_square_launch(cudaStream_t stream, const bf16 *x, bf16 *out,
                              long long elems);
void rwkv7_act_tanh_launch(cudaStream_t stream, const bf16 *x, bf16 *out,
                           long long elems);
void rwkv7_act_sigmoid_launch(cudaStream_t stream, const bf16 *x, bf16 *out,
                              long long elems);
void rwkv7_add_vec_launch(cudaStream_t stream, int C, const bf16 *x,
                          const bf16 *vec, bf16 *out, long long elems);
void rwkv7_v3a_add_bf16_launch(cudaStream_t stream, const bf16 *x,
                               const bf16 *y, bf16 *out, long long elems);
void rwkv7_v3a_layer_norm_bf16_launch(cudaStream_t stream, int rows, int C,
                                      const bf16 *x, const bf16 *weight,
                                      const bf16 *bias, bf16 *y, float eps);
void rwkv7_v3a_linear_bf16_launch(cudaStream_t stream, int M, int K, int N,
                                  const bf16 *x, const bf16 *weight, bf16 *y);
void rwkv7_v3a_linear_bf16_orig_launch(cudaStream_t stream, int M, int K, int N,
                                       const bf16 *x, const bf16 *weight_orig,
                                       bf16 *y);
} // namespace rwkv_bf16_training
