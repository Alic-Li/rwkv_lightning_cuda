#ifndef NORM_CUH
#define NORM_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Norm 模块 CUDA 实现
// 支持 fp16 精度，使用 Half8 向量化优化

// L2 Normalization: output = input / sqrt(sum(input^2) + epsilon)
// input: [M, N] - 输入矩阵（行主序）
// output: [M, N] - 输出矩阵（行主序）
// epsilon: 防止除零的小值（默认 1e-5）
// stream: CUDA 流（可选）
void l2_normalize_fp16(
    const half* input,     // [M, N]
    half* output,          // [M, N]
    int M, int N,
    float epsilon = 1e-5f,
    cudaStream_t stream = nullptr
);

// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * gamma + beta
// input: [M, N] - 输入矩阵（行主序）
// gamma: [N] - scale 参数
// beta: [N] - shift 参数（可选，可为 nullptr）
// output: [M, N] - 输出矩阵（行主序）
// eps: epsilon 值（默认 1e-5）
// stream: CUDA 流（可选）
void layer_norm_half8_fp16(
    const half* input,     // [M, N]
    const half* gamma,     // [N]
    const half* beta,      // [N] or nullptr
    half* output,          // [M, N]
    int M, int N,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

// Group Normalization: 对每个组进行归一化
// input: [B*G, C/G] - 输入数据（已按组展开，行主序）
// gamma: [C] - scale 参数
// beta: [C] - shift 参数（可选，可为 nullptr）
// output: [B*G, C/G] - 输出数据（行主序）
// batch_size: 批次大小 B
// num_groups: 组数 G
// channels_per_group: 每组通道数 C/G
// eps: epsilon 值（默认 1e-5）
// stream: CUDA 流（可选）
void group_norm_half8_fp16(
    const half* input,     // [B*G, C/G]
    const half* gamma,     // [C]
    const half* beta,      // [C] or nullptr
    half* output,          // [B*G, C/G]
    int batch_size,        // B
    int num_groups,        // G
    int channels_per_group, // C/G
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

#endif // NORM_CUH
