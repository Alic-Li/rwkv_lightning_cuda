#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "Cmix.cuh"
#include "../module/linear_cublas.cuh"
#include "../spmv/spmv.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// 辅助宏：检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            assert(false); \
        } \
    } while(0)

// ReLU kernel: relu(x) = max(0, x)
__global__ void relu_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half val = input[idx];
        output[idx] = __hgt(val, __float2half(0.0f)) ? val : __float2half(0.0f);
    }
}

// Square kernel: square(x) = x * x
__global__ void square_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half val = input[idx];
        output[idx] = __hmul(val, val);
    }
}

// Subtraction kernel: output = a - b
__global__ void sub_kernel(const half* a, const half* b, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hsub(a[idx], b[idx]);
    }
}

// Element-wise multiply and add kernel: output = a + b * c (broadcast c)
__global__ void broadcast_mul_add_kernel(
    const half* a, const half* b, const half* c,
    half* output, int size, int c_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c_idx = idx % c_size;
        output[idx] = __hadd(a[idx], __hmul(b[idx], c[c_idx]));
    }
}

// CMix one implementation
void cmix_one_fp16(
    const half* x,
    half* x_prev,
    const half* x_k,
    const half* K_,
    const half* V_,
    CmixOneTempBuffers* temp_buf,
    half* output,
    int C, int D,
    cudaStream_t stream
) {
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查权重指针
    if (K_ == nullptr || V_ == nullptr || x_k == nullptr) {
        fprintf(stderr, "Error: Required weights missing for cmix_one.\n");
        fprintf(stderr, "  K_=%p, V_=%p, x_k=%p\n", K_, V_, x_k);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for cmix_one.\n");
        fprintf(stderr, "  x=%p, x_prev=%p, output=%p, temp_buf=%p\n",
                x, x_prev, output, temp_buf);
        exit(1);
    }
    
    // 检查参数范围
    if (C <= 0 || D <= 0) {
        fprintf(stderr, "Error: Invalid dimensions for cmix_one. C=%d, D=%d\n", C, D);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *d_xx = temp_buf->xx;
    half *d_k = temp_buf->k;
    half *d_k_linear = temp_buf->k_linear;
    
    const int threads = 256;
    dim3 block_size(threads);
    
    // 1. xx = x_prev[1] - x
    int blocks = (C + threads - 1) / threads;
    sub_kernel<<<blocks, block_size, 0, stream>>>(x_prev + C, x, d_xx, C);
    
    // 2. x_prev[1] = x (in-place update)
    CUDA_CHECK(cudaMemcpyAsync(x_prev + C, x, C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    
    // 3. k = x + xx * x_k
    broadcast_mul_add_kernel<<<blocks, block_size, 0, stream>>>(
        x, d_xx, x_k, d_k, C, C
    );
    
    // 4. k = relu(linear(k, K_)) ** 2
    // 4.1 linear: k [C] @ K_ [C, D] -> k_linear [D]
    linear_vec_fp16(d_k, K_, nullptr, d_k_linear, D, C, stream);
    
    // 4.2 relu
    int D_blocks = (D + threads - 1) / threads;
    relu_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D);
    
    // 4.3 square
    square_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D);
    
    // 5. kv = SPMV_OP(k, V_) 或 k @ V_
    // k [D] @ V_ [D, C] -> kv [C]
    // 使用SPMV因为k是稀疏的（relu后很多0）
    // 先清零输出（SPMV使用atomicAdd）
    CUDA_CHECK(cudaMemsetAsync(output, 0, C * sizeof(half), stream));
    spmv_forward_fp16(D, C, d_k_linear, V_, output, stream);
}

// CMix seq implementation
void cmix_seq_fp16(
    const half* x,
    half* x_prev,
    const half* x_k,
    const half* K_,
    const half* V_,
    CmixSeqTempBuffers* temp_buf,
    half* output,
    int T, int C, int D,
    cudaStream_t stream
) {
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查权重指针
    if (K_ == nullptr || V_ == nullptr || x_k == nullptr) {
        fprintf(stderr, "Error: Required weights missing for cmix_seq.\n");
        fprintf(stderr, "  K_=%p, V_=%p, x_k=%p\n", K_, V_, x_k);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for cmix_seq.\n");
        fprintf(stderr, "  x=%p, x_prev=%p, output=%p, temp_buf=%p\n",
                x, x_prev, output, temp_buf);
        exit(1);
    }
    
    // 检查参数范围
    if (T <= 0 || C <= 0 || D <= 0) {
        fprintf(stderr, "Error: Invalid dimensions for cmix_seq. T=%d, C=%d, D=%d\n", T, C, D);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *d_xx = temp_buf->xx;
    half *d_k = temp_buf->k;
    half *d_k_linear = temp_buf->k_linear;
    
    const int threads = 256;
    dim3 block_size(threads);
    
    // 1. xx = cat([x_prev[1], x[:-1]]) - x
    // 先复制 x_prev[1] 到 xx 的第一行
    CUDA_CHECK(cudaMemcpyAsync(d_xx, x_prev + C, C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    // 复制 x[:-1] 到 xx 的剩余部分
    if (T > 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_xx + C, x, (T - 1) * C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }
    // 计算 xx - x
    int total_size = T * C;
    int blocks = (total_size + threads - 1) / threads;
    sub_kernel<<<blocks, block_size, 0, stream>>>(d_xx, x, d_xx, total_size);
    
    // 2. x_prev[1] = x[-1]
    CUDA_CHECK(cudaMemcpyAsync(x_prev + C, x + (T - 1) * C, C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    
    // 3. k = x + xx * x_k (broadcast x_k)
    broadcast_mul_add_kernel<<<blocks, block_size, 0, stream>>>(
        x, d_xx, x_k, d_k, total_size, C
    );
    
    // 4. k = relu(linear(k, K_)) ** 2
    // 4.1 linear: k [T, C] @ K_ [C, D] -> k_linear [T, D]
    linear_fp16(d_k, K_, nullptr, d_k_linear, T, D, C, stream);
    
    // 4.2 relu
    int D_total = T * D;
    int D_blocks = (D_total + threads - 1) / threads;
    relu_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D_total);
    
    // 4.3 square
    square_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D_total);
    
    // 5. kv = k @ V_
    // k [T, D] @ V_ [D, C] -> kv [T, C]
    linear_fp16(d_k_linear, V_, nullptr, output, T, C, D, stream);
}

// CMix seq_batch implementation
void cmix_seq_batch_fp16(
    const half* x,
    half* x_prev,
    const half* x_k,
    const half* K_,
    const half* V_,
    CmixSeqTempBuffers* temp_buf,
    half* output,
    int B, int T, int C, int D,
    cudaStream_t stream
) {
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查权重指针
    if (K_ == nullptr || V_ == nullptr || x_k == nullptr) {
        fprintf(stderr, "Error: Required weights missing for cmix_seq_batch.\n");
        fprintf(stderr, "  K_=%p, V_=%p, x_k=%p\n", K_, V_, x_k);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for cmix_seq_batch.\n");
        fprintf(stderr, "  x=%p, x_prev=%p, output=%p, temp_buf=%p\n",
                x, x_prev, output, temp_buf);
        exit(1);
    }
    
    // 检查参数范围
    if (B <= 0 || T <= 0 || C <= 0 || D <= 0) {
        fprintf(stderr, "Error: Invalid dimensions for cmix_seq_batch. B=%d, T=%d, C=%d, D=%d\n", B, T, C, D);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *d_xx = temp_buf->xx;
    half *d_k = temp_buf->k;
    half *d_k_linear = temp_buf->k_linear;
    
    const int threads = 256;
    dim3 block_size(threads);
    int total_size = B * T * C;
    
    // 1. xx = cat([x_prev[1], x[:,:-1]], dim=1) - x
    // 对每个batch处理
    for (int b = 0; b < B; b++) {
        size_t offset = (size_t)B * C + (size_t)b * C;
        
        // 复制 x_prev[1][b] 到 d_xx[b] 的第一行
        CUDA_CHECK(cudaMemcpyAsync(
            d_xx + b * T * C,
            x_prev + offset,
            C * sizeof(half),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        
        // 复制 x[b, :-1, :] 到 d_xx[b] 的剩余部分 (第2到T行)
        if (T > 1) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_xx + b * T * C + C,
                x + b * T * C,
                (T - 1) * C * sizeof(half),
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
    }
    // 计算 xx - x
    int blocks = (total_size + threads - 1) / threads;
    sub_kernel<<<blocks, block_size, 0, stream>>>(d_xx, x, d_xx, total_size);
    
    // 2. x_prev[1] = x[:,-1]
    for (int b = 0; b < B; b++) {
        size_t offset = (size_t)B * C + (size_t)b * C;

        CUDA_CHECK(cudaMemcpyAsync(
            x_prev + offset, // 使用修正后的 offset
            x + b * T * C + (T - 1) * C,
            C * sizeof(half),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
    
    // 3. k = x + xx * x_k (broadcast x_k)
    broadcast_mul_add_kernel<<<blocks, block_size, 0, stream>>>(
        x, d_xx, x_k, d_k, total_size, C
    );
    
    // 4. k = relu(linear(k, K_)) ** 2
    // 4.1 linear: k [B*T, C] @ K_ [C, D] -> k_linear [B*T, D]
    linear_fp16(d_k, K_, nullptr, d_k_linear, B * T, D, C, stream);
    
    // 4.2 relu
    int D_total = B * T * D;
    int D_blocks = (D_total + threads - 1) / threads;
    relu_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D_total);
    
    // 4.3 square
    square_kernel<<<D_blocks, block_size, 0, stream>>>(d_k_linear, d_k_linear, D_total);
    
    // 5. kv = k @ V_
    // k [B*T, D] @ V_ [D, C] -> kv [B*T, C]
    linear_fp16(d_k_linear, V_, nullptr, output, B * T, C, D, stream);
}
