#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 生成四种维度的零张量
// 针对half类型的特化版本
// 1D zeros
void zeros_fp16(half* output, int dim0, cudaStream_t stream);
// 2D zeros
void zeros_fp16(half* output, int dim0, int dim1, cudaStream_t stream);
// 3D zeros
void zeros_fp16(half* output, int dim0, int dim1, int dim2, cudaStream_t stream);
// 4D zeros
void zeros_fp16(half* output, int dim0, int dim1, int dim2, int dim3, cudaStream_t stream);

// 针对float类型的特化版本
// 1D zeros
void zeros_fp32(float* output, int dim0, cudaStream_t stream);
// 2D zeros
void zeros_fp32(float* output, int dim0, int dim1, cudaStream_t stream);
// 3D zeros
void zeros_fp32(float* output, int dim0, int dim1, int dim2, cudaStream_t stream);
// 4D zeros
void zeros_fp32(float* output, int dim0, int dim1, int dim2, int dim3, cudaStream_t stream);


// ==================== 激活函数 ====================

// torch.tanh: 双曲正切激活函数
void tanh_fp16(
    const half* input,
    half* output,
    size_t size,
    cudaStream_t stream = nullptr
);

// torch.sigmoid: Sigmoid 激活函数
void sigmoid_fp16(
    const half* input,
    half* output,
    size_t size,
    cudaStream_t stream = nullptr
);

// ==================== 拼接操作 ====================

// torch.cat: 在指定维度拼接两个张量
// dim=0: 在第一个维度拼接 [M1, N] + [M2, N] -> [M1+M2, N]
// dim=1: 在第二个维度拼接 [M, N1] + [M, N2] -> [M, N1+N2]
void cat_fp16(
    const half* tensor1,  // 第一个张量
    const half* tensor2,  // 第二个张量
    half* output,         // 输出张量（必须已分配内存）
    int dim,              // 拼接维度 (0 或 1)
    int M, int N,         // 张量维度
    int size1,            // 第一个张量在 dim 维度的大小
    int size2,            // 第二个张量在 dim 维度的大小
    cudaStream_t stream = nullptr
);

// ==================== 逐元素操作 ====================

// 逐元素减法: output = a - b
void element_wise_sub_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream = nullptr
);

// 逐元素加法: output = a + b
void element_wise_add_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream = nullptr
);

// 逐元素乘法: output = a * b
void element_wise_mul_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream = nullptr
);

// 融合乘加: output = a * b + c
void element_wise_fma_fp16(
    const half* a, const half* b, const half* c, half* output, int size, cudaStream_t stream = nullptr
);

// 缩放加法: output = a + b * scale
void element_wise_add_scaled_fp16(
    const half* a, const half* b, const half* scale, half* output, int size, cudaStream_t stream = nullptr
);

// 乘加: output = a * b + c
void element_wise_mul_add_fp16(
    const half* a, const half* b, const half* c, half* output, int size, cudaStream_t stream = nullptr
);

// 缩放加: output = a * scale + b
void element_wise_scale_add_fp16(
    const half* a, const half* scale, const half* b, half* output, int size, cudaStream_t stream = nullptr
);

// 逐元素减法（常量）: output = a - val
void element_wise_sub_const_fp16(
    const half* a, float val, half* output, int size, cudaStream_t stream = nullptr
);

// 逐元素加法（常量）: output = a + val
void element_wise_add_const_fp16(
    const half* a, float val, half* output, int size, cudaStream_t stream = nullptr
);

// 取反: output = -a
void negate_fp16(
    const half* a, half* output, int size, cudaStream_t stream = nullptr
);

// ==================== 复制操作 ====================

// 复制: dst = src
void copy_fp16(
    const half* src, half* dst, int size, cudaStream_t stream = nullptr
);

// ==================== 归约操作 ====================

// 在最后一个维度求和: [H, N] -> [H, 1]
void sum_reduce_last_dim_fp16(
    const half* input, half* output, int H, int N, cudaStream_t stream = nullptr
);

// 序列归约: [T*H, N] -> [T*H]
void sum_reduce_last_dim_seq_fp16(
    const half* input, half* output, int T, int H, int N, cudaStream_t stream = nullptr
);

// 批次归约: [B*T*H, N] -> [B*T*H]
void sum_reduce_last_dim_batch_fp16(
    const half* input, half* output, int B, int T, int H, int N, cudaStream_t stream = nullptr
);

// ==================== 广播操作 ====================

// 广播: [H, 1] -> [H, N]
void broadcast_fp16(
    const half* input, half* output, int H, int N, cudaStream_t stream = nullptr
);

// 序列广播: [T*H] -> [T*H*N]
void broadcast_seq_fp16(
    const half* input, half* output, int T, int H, int N, cudaStream_t stream = nullptr
);

// 批次广播: [B*T*H] -> [B*T*H*N]
void broadcast_batch_fp16(
    const half* input, half* output, int B, int T, int H, int N, cudaStream_t stream = nullptr
);

// 广播 v_first: [C] -> [T, C]
void broadcast_v_first_fp16(
    const half* v_first, half* output, int T, int C, cudaStream_t stream = nullptr
);

// 批次广播 v_first: [B, C] -> [B, T, C]
void broadcast_v_first_batch_fp16(
    const half* v_first, half* output, int B, int T, int C, cudaStream_t stream = nullptr
);

// ==================== 批次操作 ====================

// 创建拼接批次: 将 x_prev 和 x 拼接成 [B, T, C]
void create_concat_batch_fp16(
    const half* x_prev, const half* x, half* output, int B, int T, int C, cudaStream_t stream = nullptr
);

// 更新 x_prev: 从 x 的最后一个时间步更新 x_prev
void update_x_prev_batch_fp16(
    const half* x, half* x_prev, int B, int T, int C, cudaStream_t stream = nullptr
);

#endif // UTILS_CUH

/*
// 创建空张量
half* x = empty_fp16(10, 256);  // 类似 torch.empty((10, 256))

// 创建与输入相同形状的空张量
half* y = empty_like_fp16(input, size);  // 类似 torch.empty_like(input)

// 拼接两个张量
cat_fp16(tensor1, tensor2, output, 0, M, N, M1, M2);  // dim=0
cat_fp16(tensor1, tensor2, output, 1, M, N, N1, N2);  // dim=1
*/