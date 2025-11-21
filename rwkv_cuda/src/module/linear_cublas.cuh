#ifndef LINEAR_CUH
#define LINEAR_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Linear 层（矩阵乘法）CUDA 实现
// 支持 fp16 精度

// 矩阵乘法: [M, K] @ [K, N] → [M, N]
// A: [M, K] - 输入矩阵（行主序）
// W: [K, N] - 权重矩阵（行主序，即 W^T 是列主序）
// bias: [N] - 偏置向量（可选，可为 nullptr）
// C: [M, N] - 输出矩阵（行主序）
// stream: CUDA 流（可选）
void linear_fp16(
    const half* A,        // [M, K]
    const half* W,        // [K, N]
    const half* bias,     // [N] or nullptr
    half* C,              // [M, N]
    int M, int N, int K,
    cudaStream_t stream = nullptr
);

// 向量矩阵乘法: [K] @ [K, N] → [N]
// 这是 linear_fp16 的特例，M=1
inline void linear_vec_fp16(
    const half* x,        // [K]
    const half* W,        // [K, N]
    const half* bias,     // [N] or nullptr
    half* y,              // [N]
    int N, int K,
    cudaStream_t stream = nullptr
) {
    linear_fp16(x, W, bias, y, 1, N, K, stream);
}

#endif // LINEAR_CUH

