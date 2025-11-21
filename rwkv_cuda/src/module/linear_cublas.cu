#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "linear_cublas.cuh"
#include <cuda_fp16.h>
#include <cublas_v2.h>

// C++ interface implementation
void linear_fp16(
    const half* A,
    const half* W,
    const half* bias,
    half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    
    if (stream != nullptr) {
        cublasSetStream(handle, stream);
    }

    // 准备 __half 类型参数
    const __half half_alpha = __float2half(1.0f);
    const __half half_beta = (bias != nullptr) ? __float2half(1.0f) : __float2half(0.0f);
    
    // 如果有偏置，则先复制偏置到输出
    if (bias != nullptr) {
        if (M == 1) {
            // 向量情况：直接复制偏置
            cudaMemcpyAsync(C, bias, N * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        } else {
            // 矩阵情况：广播偏置到每一行
            for (int i = 0; i < M; ++i) {
                cudaMemcpyAsync(C + i * N, bias, N * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
    
    // 使用 cublasHgemm 执行矩阵乘法
    // 输入是行主序：A [M, K], W [K, N]
    // 输出是行主序：C [M, N]
    // cuBLAS 使用列主序，所以需要计算 C^T = W^T * A^T
    // 即：cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, alpha, W, K, A, M, beta, C, N)
    cublasHgemm(
        handle,
        CUBLAS_OP_T,           // transa: W^T (W 是 [K, N] 行主序，转置后是 [N, K] 列主序)
        CUBLAS_OP_T,           // transb: A^T (A 是 [M, K] 行主序，转置后是 [K, M] 列主序)
        N,                     // m: C^T 的行数 = C 的列数
        M,                     // n: C^T 的列数 = C 的行数
        K,                     // k: 公共维度
        &half_alpha,           // alpha
        (const __half*)W,      // A (在 cuBLAS 中，这是 W^T)
        K,                     // lda: W 的行主序行数 = K
        (const __half*)A,      // B (在 cuBLAS 中，这是 A^T)
        M,                     // ldb: A 的行主序行数 = M
        &half_beta,            // beta
        (__half*)C,            // C (输出)
        N                      // ldc: C 的行主序行数 = N
    );
}