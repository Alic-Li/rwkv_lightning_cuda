#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/Tmix/Tmix.cuh"

#ifndef HEAD_SIZE
#define HEAD_SIZE 64
#endif

// 辅助函数：检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 辅助函数：分配 tmix_one 临时缓冲区
void allocate_tmix_one_temp_buffers(int C, TmixOneTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xr, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xw, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xk, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xv, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xa, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xg, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->r, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kk, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_scaled, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kka, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_sigmoid, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_wkv, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_gn, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_final, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_scaled, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->neg_kk, C * sizeof(half)));
}

// 辅助函数：释放 tmix_one 临时缓冲区
void free_tmix_one_temp_buffers(TmixOneTempBuffers* buf) {
    cudaFree(buf->xx);
    cudaFree(buf->xr);
    cudaFree(buf->xw);
    cudaFree(buf->xk);
    cudaFree(buf->xv);
    cudaFree(buf->xa);
    cudaFree(buf->xg);
    cudaFree(buf->r);
    cudaFree(buf->w_intermediate);
    cudaFree(buf->w);
    cudaFree(buf->k);
    cudaFree(buf->v);
    cudaFree(buf->a_intermediate);
    cudaFree(buf->a);
    cudaFree(buf->g_intermediate);
    cudaFree(buf->g);
    cudaFree(buf->kk);
    cudaFree(buf->k_scaled);
    cudaFree(buf->kka);
    cudaFree(buf->v_intermediate);
    cudaFree(buf->v_sigmoid);
    cudaFree(buf->xx_wkv);
    cudaFree(buf->xx_gn);
    cudaFree(buf->xx_final);
    cudaFree(buf->g_scaled);
    cudaFree(buf->neg_kk);
}

// 辅助函数：分配 tmix_seq 临时缓冲区
void allocate_tmix_seq_temp_buffers(int total_size, int C, TmixSeqTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xr, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xw, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xv, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xa, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xg, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->r, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_scaled, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kka, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_sigmoid, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_wkv, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_gn, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_final, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_scaled, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->neg_kk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_concat, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_r_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_w_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_k_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_v_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_a_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_g_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_k_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_a_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_first_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_prev_expanded, C * sizeof(half)));
}

// 辅助函数：释放 tmix_seq 临时缓冲区
void free_tmix_seq_temp_buffers(TmixSeqTempBuffers* buf) {
    cudaFree(buf->xx);
    cudaFree(buf->xr);
    cudaFree(buf->xw);
    cudaFree(buf->xk);
    cudaFree(buf->xv);
    cudaFree(buf->xa);
    cudaFree(buf->xg);
    cudaFree(buf->r);
    cudaFree(buf->w_intermediate);
    cudaFree(buf->w);
    cudaFree(buf->k);
    cudaFree(buf->v);
    cudaFree(buf->a_intermediate);
    cudaFree(buf->a);
    cudaFree(buf->g_intermediate);
    cudaFree(buf->g);
    cudaFree(buf->kk);
    cudaFree(buf->k_scaled);
    cudaFree(buf->kka);
    cudaFree(buf->v_intermediate);
    cudaFree(buf->v_sigmoid);
    cudaFree(buf->xx_wkv);
    cudaFree(buf->xx_gn);
    cudaFree(buf->xx_final);
    cudaFree(buf->g_scaled);
    cudaFree(buf->neg_kk);
    cudaFree(buf->x_concat);
    cudaFree(buf->x_r_expanded);
    cudaFree(buf->x_w_expanded);
    cudaFree(buf->x_k_expanded);
    cudaFree(buf->x_v_expanded);
    cudaFree(buf->x_a_expanded);
    cudaFree(buf->x_g_expanded);
    cudaFree(buf->k_k_expanded);
    cudaFree(buf->k_a_expanded);
    cudaFree(buf->v_first_expanded);
    cudaFree(buf->x_prev_expanded);
}

// 辅助函数：初始化随机数据
void init_random_half(half* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = __float2half(dis(gen));
    }
}

// 辅助函数：打印数组（用于调试）
void print_array(const half* data, size_t size, const char* name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < std::min(size, (size_t)10); i++) {
        std::cout << __half2float(data[i]) << " ";
    }
    if (size > 10) std::cout << "...";
    std::cout << std::endl;
}

// 测试 tmix_one_fp16
void test_tmix_one(int H, int N) {
    std::cout << "\n=== Testing tmix_one_fp16 ===" << std::endl;
    std::cout << "H=" << H << ", N=" << N << std::endl;
    
    int C = H * N;
    int layer_id = 1;
    
    // 分配主机内存
    std::vector<half> h_x(C);
    std::vector<half> h_x_prev(C);
    std::vector<half> h_v_first(C);
    std::vector<half> h_state(H * N * N);
    std::vector<half> h_x_r(C), h_x_w(C), h_x_k(C), h_x_v(C), h_x_a(C), h_x_g(C);
    std::vector<half> h_w0(C), h_w1(C * C), h_w2(C * C);
    std::vector<half> h_a0(C), h_a1(C * C), h_a2(C * C);
    std::vector<half> h_v0(C), h_v1(C * C), h_v2(C * C);
    std::vector<half> h_g1(C * C), h_g2(C * C);
    std::vector<half> h_k_k(C), h_k_a(C), h_r_k(C);
    std::vector<half> h_R_(C * C), h_K_(C * C), h_V_(C * C), h_O_(C * C);
    std::vector<half> h_ln_w(C), h_ln_b(C);
    std::vector<half> h_output(C);
    std::vector<int> h_elapsed_t(1);
    
    // 初始化数据 - 使用合理的权重范围避免数值爆炸
    // 输入数据：[-1, 1]
    init_random_half(h_x.data(), C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), C, -1.0f, 1.0f);
    init_random_half(h_v_first.data(), C, -1.0f, 1.0f);
    // 状态初始化为 0
    for (int i = 0; i < H * N * N; i++) {
        h_state[i] = __float2half(0.0f);
    }
    // 时间混合参数：[-0.1, 0.1]
    init_random_half(h_x_r.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_w.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_v.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_g.data(), C, -0.1f, 0.1f);
    // 偏置：[-0.01, 0.01]
    init_random_half(h_w0.data(), C, -0.01f, 0.01f);
    init_random_half(h_a0.data(), C, -0.01f, 0.01f);
    init_random_half(h_v0.data(), C, -0.01f, 0.01f);
    // 权重矩阵：使用 Xavier 初始化，范围约为 [-sqrt(1/C), sqrt(1/C)]
    float weight_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_w1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_w2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_R_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_K_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_V_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_O_.data(), C * C, -weight_scale, weight_scale);
    // k_k, k_a, r_k：[-0.1, 0.1]
    init_random_half(h_k_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_k_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_r_k.data(), C, -0.1f, 0.1f);
    // Layer norm：weight [0.8, 1.2], bias [-0.01, 0.01]
    init_random_half(h_ln_w.data(), C, 0.8f, 1.2f);
    init_random_half(h_ln_b.data(), C, -0.01f, 0.01f);
    h_elapsed_t[0] = 0;
    
    // 分配设备内存
    half *d_x, *d_x_prev, *d_v_first, *d_state;
    half *d_x_r, *d_x_w, *d_x_k, *d_x_v, *d_x_a, *d_x_g;
    half *d_w0, *d_w1, *d_w2;
    half *d_a0, *d_a1, *d_a2;
    half *d_v0, *d_v1, *d_v2;
    half *d_g1, *d_g2;
    half *d_k_k, *d_k_a, *d_r_k;
    half *d_R_, *d_K_, *d_V_, *d_O_;
    half *d_ln_w, *d_ln_b;
    half *d_output;
    int *d_elapsed_t;
    
    CUDA_CHECK(cudaMalloc(&d_x, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_first, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_state, H * N * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_r, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_v, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_g, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_r_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_R_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_b, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_t, sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_first, h_v_first.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), H * N * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_r, h_x_r.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_w, h_x_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_v, h_x_v.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_a, h_x_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_g, h_x_g.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w0, h_w0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a0, h_a0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a1, h_a1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a2, h_a2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v0, h_v0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v1, h_v1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, h_v2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g1, h_g1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2, h_g2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_k, h_k_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_a, h_k_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_k, h_r_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R_, h_R_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O_, h_O_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_w, h_ln_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_b, h_ln_b.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elapsed_t, h_elapsed_t.data(), sizeof(int), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    TmixOneTempBuffers temp_buf;
    allocate_tmix_one_temp_buffers(C, &temp_buf);
    
    // 执行 kernel
    std::cout << "Running tmix_one_fp16..." << std::endl;
    tmix_one_fp16(
        layer_id, H, N,
        d_x, d_x_prev, d_v_first, d_state,
        d_x_r, d_x_w, d_x_k, d_x_v, d_x_a, d_x_g,
        d_w0, d_w1, d_w2,
        d_a0, d_a1, d_a2,
        d_v0, d_v1, d_v2,
        d_g1, d_g2,
        d_k_k, d_k_a, d_r_k,
        d_R_, d_K_, d_V_, d_O_,
        d_ln_w, d_ln_b,
        d_elapsed_t,
        &temp_buf,
        d_output,
        nullptr
    );
    
    // 释放临时缓冲区
    free_tmix_one_temp_buffers(&temp_buf);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, C * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "Test completed successfully!" << std::endl;
    print_array(h_output.data(), C, "Output");
    
    // 清理
    cudaFree(d_x);
    cudaFree(d_x_prev);
    cudaFree(d_v_first);
    cudaFree(d_state);
    cudaFree(d_x_r);
    cudaFree(d_x_w);
    cudaFree(d_x_k);
    cudaFree(d_x_v);
    cudaFree(d_x_a);
    cudaFree(d_x_g);
    cudaFree(d_w0);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_v0);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_g1);
    cudaFree(d_g2);
    cudaFree(d_k_k);
    cudaFree(d_k_a);
    cudaFree(d_r_k);
    cudaFree(d_R_);
    cudaFree(d_K_);
    cudaFree(d_V_);
    cudaFree(d_O_);
    cudaFree(d_ln_w);
    cudaFree(d_ln_b);
    cudaFree(d_output);
    cudaFree(d_elapsed_t);
}

// 测试 tmix_seq_fp16
void test_tmix_seq(int H, int N, int T) {
    std::cout << "\n=== Testing tmix_seq_fp16 ===" << std::endl;
    std::cout << "H=" << H << ", N=" << N << ", T=" << T << std::endl;
    
    int C = H * N;
    int layer_id = 1;
    
    // 分配主机内存
    std::vector<half> h_x(T * C);
    std::vector<half> h_x_prev(C);
    std::vector<half> h_v_first(C);
    std::vector<half> h_state(H * N * N);
    std::vector<half> h_x_r(C), h_x_w(C), h_x_k(C), h_x_v(C), h_x_a(C), h_x_g(C);
    std::vector<half> h_w0(C), h_w1(C * C), h_w2(C * C);
    std::vector<half> h_a0(C), h_a1(C * C), h_a2(C * C);
    std::vector<half> h_v0(C), h_v1(C * C), h_v2(C * C);
    std::vector<half> h_g1(C * C), h_g2(C * C);
    std::vector<half> h_k_k(C), h_k_a(C), h_r_k(C);
    std::vector<half> h_R_(C * C), h_K_(C * C), h_V_(C * C), h_O_(C * C);
    std::vector<half> h_ln_w(C), h_ln_b(C);
    std::vector<half> h_output(T * C);
    std::vector<int> h_elapsed_t(1);
    
    // 初始化数据 - 使用合理的权重范围避免数值爆炸
    init_random_half(h_x.data(), T * C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), C, -1.0f, 1.0f);
    init_random_half(h_v_first.data(), C, -1.0f, 1.0f);
    for (int i = 0; i < H * N * N; i++) {
        h_state[i] = __float2half(0.0f);
    }
    init_random_half(h_x_r.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_w.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_v.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_g.data(), C, -0.1f, 0.1f);
    init_random_half(h_w0.data(), C, -0.01f, 0.01f);
    init_random_half(h_a0.data(), C, -0.01f, 0.01f);
    init_random_half(h_v0.data(), C, -0.01f, 0.01f);
    float weight_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_w1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_w2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_R_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_K_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_V_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_O_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_k_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_k_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_r_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_ln_w.data(), C, 0.8f, 1.2f);
    init_random_half(h_ln_b.data(), C, -0.01f, 0.01f);
    h_elapsed_t[0] = 0;
    
    // 分配设备内存（类似上面的代码，但大小不同）
    half *d_x, *d_x_prev, *d_v_first, *d_state;
    half *d_x_r, *d_x_w, *d_x_k, *d_x_v, *d_x_a, *d_x_g;
    half *d_w0, *d_w1, *d_w2;
    half *d_a0, *d_a1, *d_a2;
    half *d_v0, *d_v1, *d_v2;
    half *d_g1, *d_g2;
    half *d_k_k, *d_k_a, *d_r_k;
    half *d_R_, *d_K_, *d_V_, *d_O_;
    half *d_ln_w, *d_ln_b;
    half *d_output;
    int *d_elapsed_t;
    
    CUDA_CHECK(cudaMalloc(&d_x, T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_first, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_state, H * N * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_r, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_v, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_g, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_r_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_R_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_b, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_t, sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), T * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_first, h_v_first.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), H * N * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_r, h_x_r.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_w, h_x_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_v, h_x_v.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_a, h_x_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_g, h_x_g.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w0, h_w0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a0, h_a0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a1, h_a1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a2, h_a2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v0, h_v0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v1, h_v1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, h_v2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g1, h_g1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2, h_g2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_k, h_k_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_a, h_k_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_k, h_r_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R_, h_R_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O_, h_O_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_w, h_ln_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_b, h_ln_b.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elapsed_t, h_elapsed_t.data(), sizeof(int), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    TmixSeqTempBuffers temp_buf;
    int total_size = T * C;
    allocate_tmix_seq_temp_buffers(total_size, C, &temp_buf);
    
    // 执行 kernel
    std::cout << "Running tmix_seq_fp16..." << std::endl;
    tmix_seq_fp16(
        layer_id, H, N,
        d_x, T,
        d_x_prev, d_v_first, d_state,
        d_x_r, d_x_w, d_x_k, d_x_v, d_x_a, d_x_g,
        d_w0, d_w1, d_w2,
        d_a0, d_a1, d_a2,
        d_v0, d_v1, d_v2,
        d_g1, d_g2,
        d_k_k, d_k_a, d_r_k,
        d_R_, d_K_, d_V_, d_O_,
        d_ln_w, d_ln_b,
        d_elapsed_t,
        &temp_buf,
        d_output,
        nullptr
    );
    
    // 释放临时缓冲区
    free_tmix_seq_temp_buffers(&temp_buf);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, T * C * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "Test completed successfully!" << std::endl;
    print_array(h_output.data(), T * C, "Output");
    
    // 清理
    cudaFree(d_x);
    cudaFree(d_x_prev);
    cudaFree(d_v_first);
    cudaFree(d_state);
    cudaFree(d_x_r);
    cudaFree(d_x_w);
    cudaFree(d_x_k);
    cudaFree(d_x_v);
    cudaFree(d_x_a);
    cudaFree(d_x_g);
    cudaFree(d_w0);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_v0);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_g1);
    cudaFree(d_g2);
    cudaFree(d_k_k);
    cudaFree(d_k_a);
    cudaFree(d_r_k);
    cudaFree(d_R_);
    cudaFree(d_K_);
    cudaFree(d_V_);
    cudaFree(d_O_);
    cudaFree(d_ln_w);
    cudaFree(d_ln_b);
    cudaFree(d_output);
    cudaFree(d_elapsed_t);
}

// 测试 tmix_seq_batch_fp16
void test_tmix_seq_batch(int H, int N, int B, int T) {
    std::cout << "\n=== Testing tmix_seq_batch_fp16 ===" << std::endl;
    std::cout << "H=" << H << ", N=" << N << ", B=" << B << ", T=" << T << std::endl;
    
    int C = H * N;
    int layer_id = 1;
    
    // 分配主机内存
    std::vector<half> h_x(B * T * C);
    std::vector<half> h_x_prev(B * C);
    std::vector<half> h_v_first(B * C);
    std::vector<half> h_state(B * H * N * N);
    std::vector<half> h_x_r(C), h_x_w(C), h_x_k(C), h_x_v(C), h_x_a(C), h_x_g(C);
    std::vector<half> h_w0(C), h_w1(C * C), h_w2(C * C);
    std::vector<half> h_a0(C), h_a1(C * C), h_a2(C * C);
    std::vector<half> h_v0(C), h_v1(C * C), h_v2(C * C);
    std::vector<half> h_g1(C * C), h_g2(C * C);
    std::vector<half> h_k_k(C), h_k_a(C), h_r_k(C);
    std::vector<half> h_R_(C * C), h_K_(C * C), h_V_(C * C), h_O_(C * C);
    std::vector<half> h_ln_w(C), h_ln_b(C);
    std::vector<half> h_output(B * T * C);
    std::vector<int> h_elapsed_t(B);
    
    // 初始化数据 - 使用合理的权重范围避免数值爆炸
    init_random_half(h_x.data(), B * T * C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), B * C, -1.0f, 1.0f);
    init_random_half(h_v_first.data(), B * C, -1.0f, 1.0f);
    for (int i = 0; i < B * H * N * N; i++) {
        h_state[i] = __float2half(0.0f);
    }
    init_random_half(h_x_r.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_w.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_v.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_x_g.data(), C, -0.1f, 0.1f);
    init_random_half(h_w0.data(), C, -0.01f, 0.01f);
    init_random_half(h_a0.data(), C, -0.01f, 0.01f);
    init_random_half(h_v0.data(), C, -0.01f, 0.01f);
    float weight_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_w1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_w2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_a2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_v2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g1.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_g2.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_R_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_K_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_V_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_O_.data(), C * C, -weight_scale, weight_scale);
    init_random_half(h_k_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_k_a.data(), C, -0.1f, 0.1f);
    init_random_half(h_r_k.data(), C, -0.1f, 0.1f);
    init_random_half(h_ln_w.data(), C, 0.8f, 1.2f);
    init_random_half(h_ln_b.data(), C, -0.01f, 0.01f);
    for (int i = 0; i < B; i++) {
        h_elapsed_t[i] = i;
    }
    
    // 分配设备内存
    half *d_x, *d_x_prev, *d_v_first, *d_state;
    half *d_x_r, *d_x_w, *d_x_k, *d_x_v, *d_x_a, *d_x_g;
    half *d_w0, *d_w1, *d_w2;
    half *d_a0, *d_a1, *d_a2;
    half *d_v0, *d_v1, *d_v2;
    half *d_g1, *d_g2;
    half *d_k_k, *d_k_a, *d_r_k;
    half *d_R_, *d_K_, *d_V_, *d_O_;
    half *d_ln_w, *d_ln_b;
    half *d_output;
    int *d_elapsed_t;
    
    CUDA_CHECK(cudaMalloc(&d_x, B * T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, B * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_first, B * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_state, B * H * N * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_r, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_v, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_g, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v0, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g1, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g2, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_r_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_R_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_, C * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ln_b, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, B * T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_t, B * sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * T * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), B * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_first, h_v_first.data(), B * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), B * H * N * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_r, h_x_r.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_w, h_x_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_v, h_x_v.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_a, h_x_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_g, h_x_g.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w0, h_w0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a0, h_a0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a1, h_a1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a2, h_a2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v0, h_v0.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v1, h_v1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v2, h_v2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g1, h_g1.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2, h_g2.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_k, h_k_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_a, h_k_a.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_k, h_r_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R_, h_R_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_O_, h_O_.data(), C * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_w, h_ln_w.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_b, h_ln_b.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elapsed_t, h_elapsed_t.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    TmixSeqTempBuffers temp_buf;
    int total_size = B * T * C;
    allocate_tmix_seq_temp_buffers(total_size, C, &temp_buf);
    
    // 执行 kernel
    std::cout << "Running tmix_seq_batch_fp16..." << std::endl;
    tmix_seq_batch_fp16(
        layer_id, H, N,
        d_x, B, T,
        d_x_prev, d_v_first, d_state,
        d_x_r, d_x_w, d_x_k, d_x_v, d_x_a, d_x_g,
        d_w0, d_w1, d_w2,
        d_a0, d_a1, d_a2,
        d_v0, d_v1, d_v2,
        d_g1, d_g2,
        d_k_k, d_k_a, d_r_k,
        d_R_, d_K_, d_V_, d_O_,
        d_ln_w, d_ln_b,
        d_elapsed_t,
        &temp_buf,
        d_output,
        nullptr
    );
    
    // 释放临时缓冲区
    free_tmix_seq_temp_buffers(&temp_buf);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, B * T * C * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "Test completed successfully!" << std::endl;
    print_array(h_output.data(), B * T * C, "Output");
    
    // 清理
    cudaFree(d_x);
    cudaFree(d_x_prev);
    cudaFree(d_v_first);
    cudaFree(d_state);
    cudaFree(d_x_r);
    cudaFree(d_x_w);
    cudaFree(d_x_k);
    cudaFree(d_x_v);
    cudaFree(d_x_a);
    cudaFree(d_x_g);
    cudaFree(d_w0);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_v0);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_g1);
    cudaFree(d_g2);
    cudaFree(d_k_k);
    cudaFree(d_k_a);
    cudaFree(d_r_k);
    cudaFree(d_R_);
    cudaFree(d_K_);
    cudaFree(d_V_);
    cudaFree(d_O_);
    cudaFree(d_ln_w);
    cudaFree(d_ln_b);
    cudaFree(d_output);
    cudaFree(d_elapsed_t);
}

int main() {
    std::cout << "=== TMix CUDA Test ===" << std::endl;
    
    int H = 16;  // 头数
    int N = 64;  // 头大小
    
    // 测试 tmix_one
    test_tmix_one(H, N);
    
    // 测试 tmix_seq
    int T = 10;
    test_tmix_seq(H, N, T);
    
    // 测试 tmix_seq_batch
    int B = 2;
    test_tmix_seq_batch(H, N, B, T);
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    return 0;
}

