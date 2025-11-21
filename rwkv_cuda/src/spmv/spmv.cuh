#ifndef SPMV_CUH
#define SPMV_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Sparse Matrix-Vector Multiplication (SPMV)
// 稀疏矩阵向量乘法
// vec: [D] - 输入向量（稀疏，大部分为0）
// mat: [D, C] - 权重矩阵（行主序）
// out: [C] - 输出向量

#define SPMV_BLOCKDIM 128
#define SPMV_MAXNPERBLOCK 64

void spmv_forward_fp16(
    int D, int C,
    const half* vec,    // [D]
    const half* mat,    // [D, C]
    half* out,          // [C]
    cudaStream_t stream = nullptr
);

#endif // SPMV_CUH

