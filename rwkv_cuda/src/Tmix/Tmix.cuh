#ifndef TMIX_CUH
#define TMIX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// TMix 模块 CUDA 实现
// 支持 fp16 精度

// 临时缓冲区结构 - 用于 tmix_one_fp16
// 所有缓冲区大小都是 C = H * N
struct TmixOneTempBuffers {
    half* xx;            // [C]
    half* xr;            // [C]
    half* xw;            // [C]
    half* xk;            // [C]
    half* xv;            // [C]
    half* xa;            // [C]
    half* xg;            // [C]
    half* r;             // [C]
    half* w_intermediate;// [C]
    half* w;             // [C]
    half* k;             // [C]
    half* v;             // [C]
    half* a_intermediate;// [C]
    half* a;             // [C]
    half* g_intermediate;// [C]
    half* g;             // [C]
    half* kk;            // [C]
    half* k_scaled;      // [C]
    half* kka;           // [C]
    half* v_intermediate;// [C]
    half* v_sigmoid;     // [C]
    half* xx_wkv;        // [C]
    half* xx_gn;         // [C]
    half* xx_final;      // [C]
    half* g_scaled;      // [C]
    half* neg_kk;        // [C]
};

// 临时缓冲区结构 - 用于 tmix_seq_fp16 和 tmix_seq_batch_fp16
// 所有缓冲区大小都是 total_size = T * C (seq) 或 B * T * C (batch)
struct TmixSeqTempBuffers {
    half* xx;            // [total_size]
    half* xr;            // [total_size]
    half* xw;            // [total_size]
    half* xk;            // [total_size]
    half* xv;            // [total_size]
    half* xa;            // [total_size]
    half* xg;            // [total_size]
    half* r;             // [total_size]
    half* w_intermediate;// [total_size]
    half* w;             // [total_size]
    half* k;             // [total_size]
    half* v;             // [total_size]
    half* a_intermediate;// [total_size]
    half* a;             // [total_size]
    half* g_intermediate;// [total_size]
    half* g;             // [total_size]
    half* kk;            // [total_size]
    half* k_scaled;      // [total_size]
    half* kka;           // [total_size]
    half* v_intermediate;// [total_size]
    half* v_sigmoid;     // [total_size]
    half* xx_wkv;        // [total_size]
    half* xx_gn;         // [total_size]
    half* xx_final;      // [total_size]
    half* g_scaled;      // [total_size]
    half* neg_kk;        // [total_size]
    // 序列特有缓冲区
    half* x_concat;      // [total_size] - 用于 seq
    half* x_r_expanded;  // [total_size] - 用于 seq/batch
    half* x_w_expanded;  // [total_size] - 用于 seq/batch
    half* x_k_expanded;  // [total_size] - 用于 seq/batch
    half* x_v_expanded;  // [total_size] - 用于 seq/batch
    half* x_a_expanded;  // [total_size] - 用于 seq/batch
    half* x_g_expanded;  // [total_size] - 用于 seq/batch
    half* k_k_expanded;  // [total_size] - 用于 seq/batch
    half* k_a_expanded;  // [total_size] - 用于 seq/batch
    half* v_first_expanded;// [total_size] - 用于 seq/batch
    half* x_prev_expanded;// [C] - 用于 seq
};

// RWKV_x070_TMix_one - 单token处理
// layer_id: 层ID
// H: 头数
// N: 头大小
// x: [C] - 当前输入
// x_prev: [C] - 上一个输入（会被修改）
// v_first: [C] - 第一个v值（会被修改）
// state: [H, N, N] - 状态矩阵（会被修改）
// x_r, x_w, x_k, x_v, x_a, x_g: [C] - 时间混合参数
// w0, w1, w2: [C] - w相关权重
// a0, a1, a2: [C] - a相关权重
// v0, v1, v2: [C] - v相关权重
// g1, g2: [C] - g相关权重
// k_k, k_a, r_k: [C] - k和r相关参数
// R_, K_, V_, O_: [C, C] - 线性层权重矩阵
// ln_w, ln_b: [C] - layer norm权重和偏置
// elapsed_t: [1] - 经过的时间
// temp_buf: 临时缓冲区（必须预分配，大小由 C = H * N 决定）
// output: [C] - 输出
void tmix_one_fp16(
    int layer_id, int H, int N,
    const half* x,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixOneTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream = nullptr
);

// RWKV_x070_TMix_seq - 序列处理
// x: [T, C] - 输入序列
// x_prev: [C] - 上一个输入（会被修改）
// v_first: [C] - 第一个v值（会被修改）
// state: [H, N, N] - 状态矩阵（会被修改）
// temp_buf: 临时缓冲区（必须预分配，大小由 T * C 决定）
// output: [T, C] - 输出序列
void tmix_seq_fp16(
    int layer_id, int H, int N,
    const half* x, int T,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixSeqTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream = nullptr
);

// RWKV_x070_TMix_seq_batch - 批量序列处理
// x: [B, T, C] - 输入批次
// x_prev: [B, C] - 上一个输入（会被修改）
// v_first: [B, C] - 第一个v值（会被修改）
// state: [B, H, N, N] - 状态矩阵（会被修改）
// temp_buf: 临时缓冲区（必须预分配，大小由 B * T * C 决定）
// output: [B, T, C] - 输出批次
void tmix_seq_batch_fp16(
    int layer_id, int H, int N,
    const half* x, int B, int T,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixSeqTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream = nullptr
);

#endif // TMIX_CUH

