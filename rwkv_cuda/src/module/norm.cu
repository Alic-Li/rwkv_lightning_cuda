#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "norm.cuh"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <algorithm>

// 强制 16 字节对齐，对应 float4 的宽度
struct __align__(16) Half8 {
    half data[8];
};

// 将 8个half 在内部计算中转为 8个float，防炸精度
__device__ __forceinline__ void load_half8_as_float(const void* ptr, float dst[8]) {
    // float4 (128bit) 形式加载，减少指令数
    float4 raw = *reinterpret_cast<const float4*>(ptr);
    const half* h_ptr = reinterpret_cast<const half*>(&raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[i] = __half2float(h_ptr[i]);
    }
}

// 将 8个float 转回 half 并以 float4 形式写回
__device__ __forceinline__ void store_float_as_half8(void* ptr, float src[8]) {
    Half8 container;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        container.data[i] = __float2half(src[i]);
    }
    *reinterpret_cast<float4*>(ptr) = *reinterpret_cast<float4*>(&container);
}

template <int BLOCK_SIZE>
__global__ void L2NormalizeHalf8Kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int hidden_size,
    float epsilon)
{
    const int tid = threadIdx.x;
    const int row_idx = blockIdx.x;
    
    // 指针偏移到当前行
    const int offset = row_idx * hidden_size;
    const half* row_input = input + offset;
    half* row_output = output + offset;

    float sum_sq = 0.0f;
    
    // 每个线程每次处理 8 个元素（128bit）
    for (int i = tid * 8; i < hidden_size; i += BLOCK_SIZE * 8) {
        float val_f[8];
        load_half8_as_float(&row_input[i], val_f);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            sum_sq += val_f[k] * val_f[k];
        }
    }

    // Block 级归约 (求 SumSq)
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float row_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

    // 计算 Scale (Rsqrt)
    __shared__ float s_scale;
    if (tid == 0) {
        s_scale = rsqrtf(row_sum_sq + epsilon);
    }
    __syncthreads();

    // 向量化写回
    float scale = s_scale;
    for (int i = tid * 8; i < hidden_size; i += BLOCK_SIZE * 8) {
        float val_f[8];
        // 如果是 Memory Bound，寄存器够的话可以缓存
        // 但对于 Large Hidden Size，通常重新读取比溢出寄存器更快
        load_half8_as_float(&row_input[i], val_f);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            val_f[k] *= scale;
        }

        store_float_as_half8(&row_output[i], val_f);
    }
}

template <int BLOCK_SIZE>
__global__ void LayerNormHalf8Kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    int hidden_size,
    float epsilon)
{
    const int tid = threadIdx.x;
    const int row_idx = blockIdx.x;
    
    const int offset = row_idx * hidden_size;
    const half* row_input = input + offset;
    half* row_output = output + offset;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // 向量化读取并统计
    for (int i = tid * 8; i < hidden_size; i += BLOCK_SIZE * 8) {
        float val_f[8];
        load_half8_as_float(&row_input[i], val_f);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            local_sum += val_f[k];
            local_sq_sum += val_f[k] * val_f[k];
        }
    }

    // Block 归约
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float row_sum = BlockReduce(temp_storage).Sum(local_sum);
    __syncthreads(); // 重用 Shared Mem
    float row_sq_sum = BlockReduce(temp_storage).Sum(local_sq_sum);

    // 计算 Mean & InvStd
    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean = row_sum / hidden_size;
        float var = (row_sq_sum / hidden_size) - (s_mean * s_mean);
        s_inv_std = rsqrtf(max(var, 0.0f) + epsilon);
    }
    __syncthreads();

    // 应用并写回
    for (int i = tid * 8; i < hidden_size; i += BLOCK_SIZE * 8) {
        float val_f[8];
        float g_f[8];
        float b_f[8];

        load_half8_as_float(&row_input[i], val_f);
        load_half8_as_float(&gamma[i], g_f);
        
        if (beta != nullptr) {
            load_half8_as_float(&beta[i], b_f);
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                b_f[k] = 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            // (x - mean) * inv_std * gamma + beta
            val_f[k] = ((val_f[k] - s_mean) * s_inv_std) * g_f[k] + b_f[k];
        }

        store_float_as_half8(&row_output[i], val_f);
    }
}

template <int BLOCK_SIZE>
__global__ void GroupNormHalf8Kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma, // Shape: [C]
    const half* __restrict__ beta,  // Shape: [C]
    int num_groups,
    int channels_per_group,  // = C / G
    float epsilon)
{
    const int tid = threadIdx.x;
    
    int global_group_idx = blockIdx.x; 
    
    const int offset = global_group_idx * channels_per_group;
    
    const half* row_input = input + offset;
    half* row_output = output + offset;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid * 8; i < channels_per_group; i += BLOCK_SIZE * 8) {
        float val_f[8];
        load_half8_as_float(&row_input[i], val_f);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            local_sum += val_f[k];
            local_sq_sum += val_f[k] * val_f[k];
        }
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float grp_sum = BlockReduce(temp_storage).Sum(local_sum);
    __syncthreads();
    float grp_sq_sum = BlockReduce(temp_storage).Sum(local_sq_sum);

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean = grp_sum / channels_per_group;
        float var = (grp_sq_sum / channels_per_group) - (s_mean * s_mean);
        s_inv_std = rsqrtf(max(var, 0.0f) + epsilon);
    }
    __syncthreads();

    // 应用 Gamma/Beta
    int current_group_in_batch = global_group_idx % num_groups;
    
    // Gamma/Beta 的起始偏移 = group_id * (C/G)
    int affine_base_offset = current_group_in_batch * channels_per_group;

    for (int i = tid * 8; i < channels_per_group; i += BLOCK_SIZE * 8) {
        float val_f[8];
        float g_f[8];
        float b_f[8];

        load_half8_as_float(&row_input[i], val_f);
        
        // Gamma/Beta 根据 Channel 索引 affine_base_offset + i
        load_half8_as_float(&gamma[affine_base_offset + i], g_f);
        
        if (beta != nullptr) {
            load_half8_as_float(&beta[affine_base_offset + i], b_f);
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                b_f[k] = 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            val_f[k] = ((val_f[k] - s_mean) * s_inv_std) * g_f[k] + b_f[k];
        }

        store_float_as_half8(&row_output[i], val_f);
    }
}

// 选择合适的 BLOCK_SIZE
constexpr int BLOCK_SIZE = 256;

// L2 Normalization 包装函数
void l2_normalize_fp16(
    const half* input,
    half* output,
    int M, int N,
    float epsilon,
    cudaStream_t stream
) {
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    L2NormalizeHalf8Kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        input, output, N, epsilon
    );
}

// Layer Normalization 包装函数
void layer_norm_half8_fp16(
    const half* input,
    const half* gamma,
    const half* beta,
    half* output,
    int M, int N,
    float eps,
    cudaStream_t stream
) {
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    
    LayerNormHalf8Kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        input, output, gamma, beta, N, eps
    );
}

// Group Normalization 包装函数
void group_norm_half8_fp16(
    const half* input,
    const half* gamma,
    const half* beta,
    half* output,
    int batch_size,
    int num_groups,
    int channels_per_group,
    float eps,
    cudaStream_t stream
) {
    // 总组数 = batch_size * num_groups
    int total_groups = batch_size * num_groups;
    
    dim3 grid(total_groups);
    dim3 block(BLOCK_SIZE);
    
    GroupNormHalf8Kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        input, output, gamma, beta, num_groups, channels_per_group, eps
    );
}
