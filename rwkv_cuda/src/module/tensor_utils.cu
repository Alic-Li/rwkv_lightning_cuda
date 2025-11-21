#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "tensor_utils.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


template<typename T>
__global__ void zeros_kernel_nd(
    T* output,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = T(0);
    }
}

// 1D zeros
template<typename T>
void zeros_1d(T* output, int dim0, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim0 + threads - 1) / threads;
    zeros_kernel_nd<<<blocks, threads, 0, stream>>>(output, dim0);
}

// 2D zeros
template<typename T>
void zeros_2d(T* output, int dim0, int dim1, cudaStream_t stream) {
    int threads = 256;
    int size = dim0 * dim1;
    int blocks = (size + threads - 1) / threads;
    zeros_kernel_nd<<<blocks, threads, 0, stream>>>(output, size);
}

// 3D zeros
template<typename T>
void zeros_3d(T* output, int dim0, int dim1, int dim2, cudaStream_t stream) {
    int threads = 256;
    int size = dim0 * dim1 * dim2;
    int blocks = (size + threads - 1) / threads;
    zeros_kernel_nd<<<blocks, threads, 0, stream>>>(output, size);
}

// 4D zeros
template<typename T>
void zeros_4d(T* output, int dim0, int dim1, int dim2, int dim3, cudaStream_t stream) {
    int threads = 256;
    int size = dim0 * dim1 * dim2 * dim3;
    int blocks = (size + threads - 1) / threads;
    zeros_kernel_nd<<<blocks, threads, 0, stream>>>(output, size);
}

// 针对half类型的特化版本
// 1D zeros
void zeros_fp16(half* output, int dim0, cudaStream_t stream) {
    zeros_1d<half>(output, dim0, stream);
}
// 2D zeros
void zeros_fp16(half* output, int dim0, int dim1, cudaStream_t stream) {
    zeros_2d<half>(output, dim0, dim1, stream);
}
// 3D zeros
void zeros_fp16(half* output, int dim0, int dim1, int dim2, cudaStream_t stream) {
    zeros_3d<half>(output, dim0, dim1, dim2, stream);
}
// 4D zeros
void zeros_fp16(half* output, int dim0, int dim1, int dim2, int dim3, cudaStream_t stream) {
    zeros_4d<half>(output, dim0, dim1, dim2, dim3, stream);
}

// 针对float类型的特化版本
// 1D zeros
void zeros_fp32(float* output, int dim0, cudaStream_t stream) {
    zeros_1d<float>(output, dim0, stream);
}
// 2D zeros
void zeros_fp32(float* output, int dim0, int dim1, cudaStream_t stream) {
    zeros_2d<float>(output, dim0, dim1, stream);
}
// 3D zeros
void zeros_fp32(float* output, int dim0, int dim1, int dim2, cudaStream_t stream) {
    zeros_3d<float>(output, dim0, dim1, dim2, stream);
}
// 4D zeros
void zeros_fp32(float* output, int dim0, int dim1, int dim2, int dim3, cudaStream_t stream) {
    zeros_4d<float>(output, dim0, dim1, dim2, dim3, stream);
}


// tanh
__global__ void tanh_kernel(
    const half* input,
    half* output,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(tanhf(val));
    }
}

void tanh_fp16(
    const half* input,
    half* output,
    size_t size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tanh_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// sigmoid
__global__ void sigmoid_kernel(
    const half* input,
    half* output,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        // sigmoid(x) = 1 / (1 + exp(-x))
        output[idx] = __float2half(1.0f / (1.0f + expf(-val)));
    }
}

void sigmoid_fp16(
    const half* input,
    half* output,
    size_t size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}

// cat: 在 dim=0 维度拼接（在第一个维度拼接）
// tensor1: [M1, N], tensor2: [M2, N] -> output: [M1+M2, N]
__global__ void cat_dim0_kernel(
    const half* tensor1,
    const half* tensor2,
    half* output,
    int M1, int M2, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_M = M1 + M2;
    int total_size = total_M * N;
    
    if (idx < total_size) {
        int m = idx / N;
        int n = idx % N;
        
        if (m < M1) {
            output[idx] = tensor1[m * N + n];
        } else {
            output[idx] = tensor2[(m - M1) * N + n];
        }
    }
}

// cat: 在 dim=1 维度拼接（在第二个维度拼接）
// tensor1: [M, N1], tensor2: [M, N2] -> output: [M, N1+N2]
__global__ void cat_dim1_kernel(
    const half* tensor1,
    const half* tensor2,
    half* output,
    int M, int N1, int N2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_N = N1 + N2;
    int total_size = M * total_N;
    
    if (idx < total_size) {
        int m = idx / total_N;
        int n = idx % total_N;
        
        if (n < N1) {
            output[idx] = tensor1[m * N1 + n];
        } else {
            output[idx] = tensor2[m * N2 + (n - N1)];
        }
    }
}

void cat_fp16(
    const half* tensor1,
    const half* tensor2,
    half* output,
    int dim,
    int M, int N,
    int size1,
    int size2,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks;
    
    if (dim == 0) {
        // 在第一个维度拼接: [M1, N] + [M2, N] -> [M1+M2, N]
        int total_M = size1 + size2;
        int total_size = total_M * N;
        blocks = (total_size + threads - 1) / threads;
        cat_dim0_kernel<<<blocks, threads, 0, stream>>>(tensor1, tensor2, output, size1, size2, N);
    } else if (dim == 1) {
        // 在第二个维度拼接: [M, N1] + [M, N2] -> [M, N1+N2]
        int total_N = size1 + size2;
        int total_size = M * total_N;
        blocks = (total_size + threads - 1) / threads;
        cat_dim1_kernel<<<blocks, threads, 0, stream>>>(tensor1, tensor2, output, M, size1, size2);
    } else {
        // 不支持的维度
        return;
    }
}
// Element-wise operations kernels
__global__ void element_wise_sub_kernel(
    const half* a, const half* b, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void element_wise_add_kernel(
    const half* a, const half* b, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void element_wise_mul_kernel(
    const half* a, const half* b, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void element_wise_fma_kernel(
    const half* a, const half* b, const half* c, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hfma(a[idx], b[idx], c[idx]);
    }
}

__global__ void element_wise_add_scaled_kernel(
    const half* a, const half* b, const half* scale, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(a[idx], __hmul(b[idx], scale[idx]));
    }
}

__global__ void element_wise_mul_add_kernel(
    const half* a, const half* b, const half* c, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(__hmul(a[idx], b[idx]), c[idx]);
    }
}

__global__ void element_wise_scale_add_kernel(
    const half* a, const half* scale, const half* b, half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(__hmul(a[idx], scale[idx]), b[idx]);
    }
}

// Copy kernel
__global__ void copy_kernel(const half* src, half* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Sum reduction kernel for [H, N] -> [H, 1]
__global__ void sum_reduce_last_dim_kernel(
    const half* input, half* output, int H, int N
) {
    int h = blockIdx.x;
    if (h >= H) return;
    
    typedef cub::BlockReduce<half, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    half thread_sum = __float2half(0.0f);
    int tid = threadIdx.x;
    int start_idx = h * N;
    
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum = __hadd(thread_sum, input[start_idx + i]);
    }
    
    half block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    if (tid == 0) {
        output[h] = block_sum;
    }
}

// Broadcast kernel: [H, 1] -> [H, N]
__global__ void broadcast_kernel(
    const half* input, half* output, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * N;
    if (idx < total) {
        int h = idx / N;
        output[idx] = input[h];
    }
}

// Element-wise operations with constants
__global__ void element_wise_sub_const_kernel(const half* a, float val, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hsub(a[idx], __float2half(val));
    }
}

__global__ void element_wise_add_const_kernel(const half* a, float val, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(a[idx], __float2half(val));
    }
}

__global__ void negate_kernel(const half* a, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hneg(a[idx]);
    }
}

// Broadcast v_first for seq
__global__ void broadcast_v_first_kernel(const half* v_first, half* output, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * C) {
        output[idx] = v_first[idx % C];
    }
}

// Sum reduction for seq: [T*H, N] -> [T*H]
__global__ void sum_reduce_last_dim_seq_kernel(
    const half* input, half* output, int T, int H, int N
) {
    int th = blockIdx.x;
    if (th >= T * H) return;
    
    typedef cub::BlockReduce<half, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    half thread_sum = __float2half(0.0f);
    int tid = threadIdx.x;
    int start_idx = th * N;
    
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum = __hadd(thread_sum, input[start_idx + i]);
    }
    
    half block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    if (tid == 0) {
        output[th] = block_sum;
    }
}

// Broadcast for seq: [T*H] -> [T*H*N]
__global__ void broadcast_seq_kernel(
    const half* input, half* output, int T, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * H * N;
    if (idx < total) {
        int th = idx / N;
        output[idx] = input[th];
    }
}

// Batch operations
__global__ void create_concat_batch_kernel(
    const half* x_prev, const half* x, half* output, int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx < total) {
        int b = idx / (T * C);
        int t = (idx / C) % T;
        int c = idx % C;
        
        if (t == 0) {
            output[idx] = x_prev[b * C + c];
        } else {
            output[idx] = x[(b * (T-1) + (t-1)) * C + c];
        }
    }
}

__global__ void update_x_prev_batch_kernel(const half* x, half* x_prev, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * C) {
        int b = idx / C;
        int c = idx % C;
        x_prev[b * C + c] = x[(b * T + (T-1)) * C + c];
    }
}

__global__ void broadcast_v_first_batch_kernel(const half* v_first, half* output, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * C) {
        int b = idx / (T * C);
        int c = idx % C;
        output[idx] = v_first[b * C + c];
    }
}

__global__ void sum_reduce_last_dim_batch_kernel(
    const half* input, half* output, int B, int T, int H, int N
) {
    int bth = blockIdx.x;
    if (bth >= B * T * H) return;
    
    typedef cub::BlockReduce<half, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    half thread_sum = __float2half(0.0f);
    int tid = threadIdx.x;
    int start_idx = bth * N;
    
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum = __hadd(thread_sum, input[start_idx + i]);
    }
    
    half block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    if (tid == 0) {
        output[bth] = block_sum;
    }
}

__global__ void broadcast_batch_kernel(
    const half* input, half* output, int B, int T, int H, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * H * N;
    if (idx < total) {
        int bth = idx / N;
        output[idx] = input[bth];
    }
}

// Helper function to launch element-wise kernels
inline void launch_element_wise(int size, cudaStream_t stream, 
    void (*kernel)(const half*, const half*, half*, int),
    const half* a, const half* b, half* output) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads, 0, stream>>>(a, b, output, size);
}

inline void launch_element_wise_ternary(int size, cudaStream_t stream,
    void (*kernel)(const half*, const half*, const half*, half*, int),
    const half* a, const half* b, const half* c, half* output) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads, 0, stream>>>(a, b, c, output, size);
}

inline void launch_copy(int size, cudaStream_t stream,
    const half* src, half* dst) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, size);
}

// ==================== Element-wise operations ====================

void element_wise_sub_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_sub_kernel<<<blocks, threads, 0, stream>>>(a, b, output, size);
}

void element_wise_add_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_add_kernel<<<blocks, threads, 0, stream>>>(a, b, output, size);
}

void element_wise_mul_fp16(
    const half* a, const half* b, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_mul_kernel<<<blocks, threads, 0, stream>>>(a, b, output, size);
}

void element_wise_fma_fp16(
    const half* a, const half* b, const half* c, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_fma_kernel<<<blocks, threads, 0, stream>>>(a, b, c, output, size);
}

void element_wise_add_scaled_fp16(
    const half* a, const half* b, const half* scale, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_add_scaled_kernel<<<blocks, threads, 0, stream>>>(a, b, scale, output, size);
}

void element_wise_mul_add_fp16(
    const half* a, const half* b, const half* c, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_mul_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, output, size);
}

void element_wise_scale_add_fp16(
    const half* a, const half* scale, const half* b, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_scale_add_kernel<<<blocks, threads, 0, stream>>>(a, scale, b, output, size);
}

// ==================== Copy operation ====================

void copy_fp16(const half* src, half* dst, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, size);
}

// ==================== Reduction operations ====================

void sum_reduce_last_dim_fp16(
    const half* input, half* output, int H, int N, cudaStream_t stream
) {
    dim3 grid(H);
    dim3 block(256);
    sum_reduce_last_dim_kernel<<<grid, block, 0, stream>>>(input, output, H, N);
}

void sum_reduce_last_dim_seq_fp16(
    const half* input, half* output, int T, int H, int N, cudaStream_t stream
) {
    dim3 grid(T * H);
    dim3 block(256);
    sum_reduce_last_dim_seq_kernel<<<grid, block, 0, stream>>>(input, output, T, H, N);
}

void sum_reduce_last_dim_batch_fp16(
    const half* input, half* output, int B, int T, int H, int N, cudaStream_t stream
) {
    dim3 grid(B * T * H);
    dim3 block(256);
    sum_reduce_last_dim_batch_kernel<<<grid, block, 0, stream>>>(input, output, B, T, H, N);
}

// ==================== Broadcast operations ====================

void broadcast_fp16(
    const half* input, half* output, int H, int N, cudaStream_t stream
) {
    int threads = 256;
    int total = H * N;
    int blocks = (total + threads - 1) / threads;
    broadcast_kernel<<<blocks, threads, 0, stream>>>(input, output, H, N);
}

void broadcast_seq_fp16(
    const half* input, half* output, int T, int H, int N, cudaStream_t stream
) {
    int threads = 256;
    int total = T * H * N;
    int blocks = (total + threads - 1) / threads;
    broadcast_seq_kernel<<<blocks, threads, 0, stream>>>(input, output, T, H, N);
}

void broadcast_batch_fp16(
    const half* input, half* output, int B, int T, int H, int N, cudaStream_t stream
) {
    int threads = 256;
    int total = B * T * H * N;
    int blocks = (total + threads - 1) / threads;
    broadcast_batch_kernel<<<blocks, threads, 0, stream>>>(input, output, B, T, H, N);
}

void broadcast_v_first_fp16(
    const half* v_first, half* output, int T, int C, cudaStream_t stream
) {
    int threads = 256;
    int total = T * C;
    int blocks = (total + threads - 1) / threads;
    broadcast_v_first_kernel<<<blocks, threads, 0, stream>>>(v_first, output, T, C);
}

void broadcast_v_first_batch_fp16(
    const half* v_first, half* output, int B, int T, int C, cudaStream_t stream
) {
    int threads = 256;
    int total = B * T * C;
    int blocks = (total + threads - 1) / threads;
    broadcast_v_first_batch_kernel<<<blocks, threads, 0, stream>>>(v_first, output, B, T, C);
}

// ==================== Element-wise operations with constants ====================

void element_wise_sub_const_fp16(
    const half* a, float val, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_sub_const_kernel<<<blocks, threads, 0, stream>>>(a, val, output, size);
}

void element_wise_add_const_fp16(
    const half* a, float val, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    element_wise_add_const_kernel<<<blocks, threads, 0, stream>>>(a, val, output, size);
}

void negate_fp16(
    const half* a, half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    negate_kernel<<<blocks, threads, 0, stream>>>(a, output, size);
}

// ==================== Batch operations ====================

void create_concat_batch_fp16(
    const half* x_prev, const half* x, half* output, int B, int T, int C, cudaStream_t stream
) {
    int threads = 256;
    int total = B * T * C;
    int blocks = (total + threads - 1) / threads;
    create_concat_batch_kernel<<<blocks, threads, 0, stream>>>(x_prev, x, output, B, T, C);
}

void update_x_prev_batch_fp16(
    const half* x, half* x_prev, int B, int T, int C, cudaStream_t stream
) {
    int threads = 256;
    int total = B * C;
    int blocks = (total + threads - 1) / threads;
    update_x_prev_batch_kernel<<<blocks, threads, 0, stream>>>(x, x_prev, B, T, C);
}
