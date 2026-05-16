#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstddef>

void rwkv7_v4_bf16_to_f16_launch(
    cudaStream_t stream, const uint16_t* src_bf16, uint16_t* dst_f16, long long elems);

void rwkv7_v4_bf16_to_f16_transpose_launch(
    cudaStream_t stream, const uint16_t* src_bf16, uint16_t* dst_f16, int rows, int cols);

void rwkv7_v4_f16_transpose_launch(
    cudaStream_t stream, const uint16_t* src_f16, uint16_t* dst_f16, int rows, int cols);

void rwkv7_v4_emb_ln0_bf16_to_f16_launch(
    cudaStream_t stream, int V, int C,
    const uint16_t* emb_bf16, const uint16_t* weight_bf16, const uint16_t* bias_bf16,
    uint16_t* out_f16, float eps);

void rwkv7_v4_f16_to_f32_launch(
    cudaStream_t stream, const half* src_f16, float* dst_f32, long long elems);

void rwkv7_wkv_fp16_seq_launch(
    cudaStream_t stream, int B, int T, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t);

void rwkv7_wkv_fp16_seq_w0_launch(
    cudaStream_t stream, int B, int T, int C, int H,
    half* state, const half* r, const half* w, const half* w0, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t);

void rwkv7_wkv_fp16_one_launch(
    cudaStream_t stream, int B, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t);

void rwkv7_wkv_fp32io16_launch(
    cudaStream_t stream, int B, int T, int C, int H, int mode,
    float* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y);

void rwkv7_tmix_mix6_launch(
    cudaStream_t stream, int B, int T, int C,
    const half* x, half* shift_state,
    const half* x_r, const half* x_w, const half* x_k,
    const half* x_v, const half* x_a, const half* x_g,
    half* out_r, half* out_w, half* out_k,
    half* out_v, half* out_a, half* out_g);

void rwkv7_tmix_kk_a_gate_launch(
    cudaStream_t stream, int B, int T, int C, int H,
    const half* k, const half* k_k, const half* a0,
    const half* a12, const half* k_a,
    half* new_k, half* neg_kk, half* kka);

void rwkv7_tmix_lnx_rkvres_xg_launch(
    cudaStream_t stream, int B, int T, int C, int H,
    const half* x, const half* r, const half* k, const half* v,
    const half* r_k, const half* weight, const half* bias,
    const half* g, half* out);

void rwkv7_tmix_vres_gate_launch(
    cudaStream_t stream, int B, int T, int C,
    const half* v, const half* v_first, const half* v0,
    const half* v12, half* out);

void rwkv7_cmix_mix_launch(
    cudaStream_t stream, int B, int T, int C,
    const half* x, half* shift_state, const half* x_k, half* out);

void rwkv7_relu_square_launch(cudaStream_t stream, const half* x, half* out, long long elems);
void rwkv7_act_tanh_launch(cudaStream_t stream, const half* x, half* out, long long elems);
void rwkv7_act_sigmoid_launch(cudaStream_t stream, const half* x, half* out, long long elems);

void rwkv7_add_vec_launch(
    cudaStream_t stream, int C, const half* x, const half* vec, half* out, long long elems);

void rwkv7_cmix_sparse_down_relu_one_launch(
    cudaStream_t stream, int C, int F, const half* preact, const half* value_fc, half* out);

void rwkv7_cmix_sparse_down_relu_rows_launch(
    cudaStream_t stream, int B, int T, int C, int F,
    const half* preact, const half* value_fc, half* out);

void rwkv7_cmix_sparse_down_relu_rows_t512_launch(
    cudaStream_t stream, int B, int T, int C, int F,
    const half* preact, const half* value_fc, half* out);

void rwkv7_cmix_sparse_one_launch(
    cudaStream_t stream, int C, int F,
    const half* x, half* shift_state, const half* x_k,
    const half* key_fc, const half* value_fc,
    half* act_scratch, half* out);

void rwkv7_cmix_sparse_rows_launch(
    cudaStream_t stream, int B, int T, int C, int F,
    const half* x, half* shift_state, const half* x_k,
    const half* key_fc, const half* value_fc,
    half* act_scratch, half* out);

void rwkv7_v3a_add_f16_launch(
    cudaStream_t stream, const half* x, const half* y, half* out, long long elems);

void rwkv7_v3a_advance_i32_launch(cudaStream_t stream, int* x, int amount, long long elems);

void rwkv7_v3a_layer_norm_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* weight, const half* bias, half* y, float eps);

void rwkv7_v3a_add_layer_norm_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* residual, const half* weight, const half* bias,
    half* x_out, half* y, float eps);

void rwkv7_v3a_add_last_layer_norm_f16_launch(
    cudaStream_t stream, int B, int T, int C,
    const half* x, const half* residual, const half* weight, const half* bias,
    half* y, float eps);

void rwkv7_v3a_add_layer_norm_cmix_mix_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* residual, half* shift_state,
    const half* weight, const half* bias, const half* x_k,
    half* x_out, half* mixed, float eps);

void rwkv7_v3a_add_layer_norm_tmix_mix6_f16_launch(
    cudaStream_t stream, int rows, int C,
    const half* x, const half* residual, half* shift_state,
    const half* weight, const half* bias,
    const half* x_r, const half* x_w, const half* x_k,
    const half* x_v, const half* x_a, const half* x_g,
    half* x_out, half* out_r, half* out_w, half* out_k,
    half* out_v, half* out_a, half* out_g, float eps);

void rwkv7_v3a_linear_t_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t, half* y);

void rwkv7_v3a_linear_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight, half* y);

void rwkv7_v3a_linear_f16_orig_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig, half* y);

void rwkv7_v3a_linear_f16_orig_lt_cfg_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    void* workspace, std::size_t workspace_bytes, int algo_index, half* y);

void rwkv7_v3a_linear_orig_rows_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    int row_tile, int out_tile, half* y);

void rwkv7_v3a_linear_orig_rows_cfg_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    int threads, int row_tile, int out_tile, half* y);

void rwkv7_v3a_linear_orig_rows_exact_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_orig,
    int threads, int out_tile, bool use4, half* y);

void rwkv7_v3a_linear_t_act_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t, int act, half* y);

void rwkv7_v3a_linear_wag_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg,
    const half* xw, const half* xa, const half* xg,
    const half* w1_t, const half* a1_t, const half* g1_t,
    half* w1, half* a1, half* g1);

void rwkv7_v3a_linear_wagv_rank_in_f16_launch(
    cudaStream_t stream, int M, int K, int Rw, int Ra, int Rg, int Rv,
    const half* xw, const half* xa, const half* xg, const half* xv,
    const half* w1_t, const half* a1_t, const half* g1_t, const half* v1_t,
    half* w1, half* a1, half* g1, half* v1);

void rwkv7_v3a_linear_wag_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg,
    const half* w1, const half* a1, const half* g1,
    const half* w2_t, const half* a2_t, const half* g2_t,
    half* w, half* a, half* g);

void rwkv7_v3a_linear_wagv_rank_out_f16_launch(
    cudaStream_t stream, int M, int C, int Kw, int Ka, int Kg, int Kv,
    const half* w1, const half* a1, const half* g1, const half* v1,
    const half* w2_t, const half* a2_t, const half* g2_t, const half* v2_t,
    const half* v, const half* v_first, const half* v0,
    half* w, half* a, half* g, half* v_out);

void rwkv7_v3a_linear_t_vres_f16_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const half* weight_t,
    const half* v, const half* v_first, const half* v0, half* y);
