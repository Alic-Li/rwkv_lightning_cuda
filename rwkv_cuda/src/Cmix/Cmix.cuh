#ifndef CMIX_CUH
#define CMIX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CMix 模块 CUDA 实现
// 支持 fp16 精度

// 临时缓冲区结构 - 用于 cmix_one_fp16
struct CmixOneTempBuffers {
    half* xx;        // [C]
    half* k;         // [C]
    half* k_linear;  // [D]
};

// 临时缓冲区结构 - 用于 cmix_seq_fp16 和 cmix_seq_batch_fp16
struct CmixSeqTempBuffers {
    half* xx;        // [total_size] (total_size = T*C for seq, B*T*C for batch)
    half* k;         // [total_size]
    half* k_linear;  // [total_k_size] (total_k_size = T*D for seq, B*T*D for batch)
};

// CMix one: 单个 token 处理
// x: [C] - 输入向量
// x_prev: [2, C] - 前一个状态（只使用 x_prev[1]）
// x_k: [C] - 键缩放因子
// K_: [C, D] - 键权重矩阵（行主序）
// V_: [D, C] - 值权重矩阵（行主序）
// temp_buf: 临时缓冲区（必须预分配）
// output: [C] - 输出向量
// 注意：x_prev[1] 会被修改（in-place）
void cmix_one_fp16(
    const half* x,           // [C]
    half* x_prev,            // [2, C] (只使用和修改 x_prev[1])
    const half* x_k,         // [C]
    const half* K_,          // [C, D]
    const half* V_,          // [D, C]
    CmixOneTempBuffers* temp_buf,
    half* output,            // [C]
    int C, int D,
    cudaStream_t stream = nullptr
);

// CMix seq: 序列处理
// x: [T, C] - 输入序列
// x_prev: [2, C] - 前一个状态（只使用 x_prev[1]）
// x_k: [C] - 键缩放因子
// K_: [C, D] - 键权重矩阵（行主序）
// V_: [D, C] - 值权重矩阵（行主序）
// temp_buf: 临时缓冲区（必须预分配，大小由 T*C 和 T*D 决定）
// output: [T, C] - 输出序列
// 注意：x_prev[1] 会被修改（in-place）
void cmix_seq_fp16(
    const half* x,           // [T, C]
    half* x_prev,            // [2, C] (只使用和修改 x_prev[1])
    const half* x_k,         // [C]
    const half* K_,          // [C, D]
    const half* V_,          // [D, C]
    CmixSeqTempBuffers* temp_buf,
    half* output,            // [T, C]
    int T, int C, int D,
    cudaStream_t stream = nullptr
);

// CMix seq_batch: 批量序列处理
// x: [B, T, C] - 输入批量序列
// x_prev: [2, B, C] - 前一个状态（只使用 x_prev[1]）
// x_k: [C] - 键缩放因子
// K_: [C, D] - 键权重矩阵（行主序）
// V_: [D, C] - 值权重矩阵（行主序）
// temp_buf: 临时缓冲区（必须预分配，大小由 B*T*C 和 B*T*D 决定）
// output: [B, T, C] - 输出批量序列
// 注意：x_prev[1] 会被修改（in-place）
void cmix_seq_batch_fp16(
    const half* x,           // [B, T, C]
    half* x_prev,            // [2, B, C] (只使用和修改 x_prev[1])
    const half* x_k,         // [C]
    const half* K_,          // [C, D]
    const half* V_,          // [D, C]
    CmixSeqTempBuffers* temp_buf,
    half* output,            // [B, T, C]
    int B, int T, int C, int D,
    cudaStream_t stream = nullptr
);

#endif // CMIX_CUH

