#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "wkv.cuh"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#ifndef HEAD_SIZE
#define HEAD_SIZE 64
#endif

#define N HEAD_SIZE

// 常量定义
constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769;
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))

// 序列前向传播 kernel
__global__ void kernel_forward_w0_fp16_dither_seq(
    const int B, const int T, const int C, const int H,
    half* __restrict__ _state, 
    const half* __restrict__ const _r, 
    const half* __restrict__ const _w, 
    const half* __restrict__ const _k, 
    const half* __restrict__ const _v, 
    const half* __restrict__ const _a, 
    const half* __restrict__ const _b,
    half* __restrict__ const _y, 
    const int* __restrict__ const _elapsed_t) {

    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ half2 state_smem[N][N / 2];

    _state += bbb * C * N + h * N * N;
    constexpr int ldg_size = sizeof(int4) / sizeof(half);
    
    // 加载状态到 shared memory
    #pragma unroll
    for (int j0 = 0; j0 < N / ldg_size; j0++) {
        int4 state_vec = ((int4*)_state)[j0 * N + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++) {
            int row = j0 * ldg_size + i * ldg_size / N;
            int col = i * ldg_size % N / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2*)&state_vec)[j1];
        }
    }
    __syncthreads();
    
    half2 state[N / 2];
    #pragma unroll
    for (int j = 0; j < N / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];
    
    __shared__ half2 r[N / 2], k[N / 2], w[N / 2], a[N / 2], b[N / 2];

    // 处理序列中的每个时间步
    for (int _t = 0; _t < T; _t++) {
        const int t = bbb * T * C + h * N + i + _t * C;
        __syncthreads();
        
        // 计算 w: exp2(nexp_half_log2_e / (1.0 + exp2(nlog2_e * w))) - 1.0 + rotator1
        ((half*)w)[i] = __float2half(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * __half2float(_w[t])))) - 1.0f + rotator1(_elapsed_t[bbb] + _t));
        ((half*)k)[i] = _k[t];
        ((half*)a)[i] = _a[t];
        ((half*)b)[i] = _b[t];
        ((half*)r)[i] = _r[t];
        __syncthreads();
        
        // 计算 sa = sum(a * state)
        half2 sa2 = __float2half2_rn(0.0f);
        #pragma unroll
        for (int j = 0; j < N / 2; j++)
            sa2 = __hadd2(sa2, __hmul2(a[j], state[j]));
        half sa = __hadd(__low2half(sa2), __high2half(sa2));
        sa2 = __half2half2(sa);

        half vv = _v[t];
        half2 vv2 = __half2half2(vv);
        half2 y2 = __float2half2_rn(0.0f);
        
        // 更新状态并计算输出
        #pragma unroll
        for (int j = 0; j < N / 2; j++) {
            half2& s = state[j];
            // s += s * w + k * v + sa * b
            s = __hadd2(s, __hadd2(__hmul2(s, w[j]), __hadd2(__hmul2(k[j], vv2), __hmul2(sa2, b[j]))));
            y2 = __hadd2(y2, __hmul2(s, r[j]));
        }
        _y[t] = __hadd(__low2half(y2), __high2half(y2));
    }
    
    // 写回状态到 shared memory
    #pragma unroll
    for (int j = 0; j < N / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    
    // 写回状态到 global memory
    #pragma unroll
    for (int j0 = 0; j0 < N / ldg_size; j0++) {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++) {
            int row = j0 * ldg_size + i * ldg_size / N;
            int col = i * ldg_size % N / 2 + j1;
            ((half2*)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4*)_state)[j0 * N + i] = state_vec;
    }
}

// 单个 token 前向传播 kernel
__global__ void kernel_forward_w0_fp16_dither_one(
    const int B, const int C, const int H,
    half* __restrict__ _state, 
    const half* __restrict__ const _r, 
    const half* __restrict__ const _w, 
    const half* __restrict__ const _k, 
    const half* __restrict__ const _v, 
    const half* __restrict__ const _a, 
    const half* __restrict__ const _b,
    half* __restrict__ const _y, 
    const int* __restrict__ const _elapsed_t) {
    
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ half2 state_smem[N][N / 2];

    _state += bbb * C * N + h * N * N;
    constexpr int ldg_size = sizeof(int4) / sizeof(half);
    
    // 加载状态到 shared memory
    #pragma unroll
    for (int j0 = 0; j0 < N / ldg_size; j0++) {
        int4 state_vec = ((int4*)_state)[j0 * N + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++) {
            int row = j0 * ldg_size + i * ldg_size / N;
            int col = i * ldg_size % N / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2*)&state_vec)[j1];
        }
    }
    __syncthreads();
    
    half2 state[N / 2];
    #pragma unroll
    for (int j = 0; j < N / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];
    
    __shared__ half2 r[N / 2], k[N / 2], w[N / 2], a[N / 2], b[N / 2];

    const int t = bbb * C + h * N + i;
    ((half*)w)[i] = __float2half(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * __half2float(_w[t])))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((half*)k)[i] = _k[t];
    ((half*)a)[i] = _a[t];
    ((half*)b)[i] = _b[t];
    ((half*)r)[i] = _r[t];
    __syncthreads();
    
    // 计算 sa = sum(a * state)
    half2 sa2 = __float2half2_rn(0.0f);
    #pragma unroll
    for (int j = 0; j < N / 2; j++)
        sa2 = __hadd2(sa2, __hmul2(a[j], state[j]));
    half sa = __hadd(__low2half(sa2), __high2half(sa2));
    sa2 = __half2half2(sa);

    half vv = _v[t];
    half2 vv2 = __half2half2(vv);
    half2 y2 = __float2half2_rn(0.0f);
    
    // 更新状态并计算输出
    #pragma unroll
    for (int j = 0; j < N / 2; j++) {
        half2& s = state[j];
        s = __hadd2(s, __hadd2(__hmul2(s, w[j]), __hadd2(__hmul2(k[j], vv2), __hmul2(sa2, b[j]))));
        y2 = __hadd2(y2, __hmul2(s, r[j]));
    }
    _y[t] = __hadd(__low2half(y2), __high2half(y2));

    // 写回状态
    #pragma unroll
    for (int j = 0; j < N / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    
    #pragma unroll
    for (int j0 = 0; j0 < N / ldg_size; j0++) {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++) {
            int row = j0 * ldg_size + i * ldg_size / N;
            int col = i * ldg_size % N / 2 + j1;
            ((half2*)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4*)_state)[j0 * N + i] = state_vec;
    }
}

// C++ 接口实现
void wkv_forward_seq(
    int B, int T, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t,
    cudaStream_t stream) {
    assert(H * N == C);
    dim3 grid(B * H);
    dim3 block(N);
    kernel_forward_w0_fp16_dither_seq<<<grid, block, 0, stream>>>(
        B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_forward_one(
    int B, int C, int H,
    half* state, const half* r, const half* w, const half* k, const half* v,
    const half* a, const half* b, half* y, const int* elapsed_t,
    cudaStream_t stream) {
    assert(H * N == C);
    dim3 grid(B * H);
    dim3 block(N);
    kernel_forward_w0_fp16_dither_one<<<grid, block, 0, stream>>>(
        B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

