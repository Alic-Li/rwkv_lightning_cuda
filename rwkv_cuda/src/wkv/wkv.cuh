#ifndef WKV_H
#define WKV_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// WKV 算子的 CUDA 实现
// 支持 fp16 精度

// 前向传播 - 单个 token (batch)
// B: batch size
// C: channel size (H * N, where H is num heads, N is head size)
// H: number of heads
// state: [B, H, N, N] - 状态矩阵
// r: [B, C] - receptance
// w: [B, C] - weight
// k: [B, C] - key
// v: [B, C] - value
// a: [B, C] - attention parameter a
// b: [B, C] - attention parameter b
// y: [B, C] - output
// elapsed_t: [B] - elapsed time
void wkv_forward_one(
    int B, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t,
    cudaStream_t stream = nullptr
);

// 前向传播 - 序列 (batch)
// B: batch size
// T: sequence length
// C: channel size (H * N)
// H: number of heads
// state: [B, H, N, N] - 状态矩阵
// r: [B, T, C] - receptance
// w: [B, T, C] - weight
// k: [B, T, C] - key
// v: [B, T, C] - value
// a: [B, T, C] - attention parameter a
// b: [B, T, C] - attention parameter b
// y: [B, T, C] - output
// elapsed_t: [B] - elapsed time
void wkv_forward_seq(
    int B, int T, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t,
    cudaStream_t stream = nullptr
);

#endif // WKV_H

