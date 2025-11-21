#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/wkv/wkv.cuh"

#ifndef HEAD_SIZE
#define HEAD_SIZE 64
#endif
#define HEAD_SIZE_N HEAD_SIZE

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

// 测试 forward_one
void test_forward_one(int B, int C, int H) {
    std::cout << "\n=== Testing wkv_forward_one ===" << std::endl;
    std::cout << "B=" << B << ", C=" << C << ", H=" << H << std::endl;
    
    // 分配主机内存
    size_t state_size = B * H * HEAD_SIZE_N * HEAD_SIZE_N;
    size_t vec_size = B * C;
    size_t elapsed_size = B;
    
    std::vector<half> h_state(state_size);
    std::vector<half> h_r(vec_size);
    std::vector<half> h_w(vec_size);
    std::vector<half> h_k(vec_size);
    std::vector<half> h_v(vec_size);
    std::vector<half> h_a(vec_size);
    std::vector<half> h_b(vec_size);
    std::vector<half> h_y(vec_size);
    std::vector<int> h_elapsed_t(elapsed_size);
    
    // 初始化数据 
    // state 初始化为 0
    for (size_t i = 0; i < state_size; i++) {
        h_state[i] = __float2half(0.0f);
    }
    init_random_half(h_r.data(), vec_size, -0.1f, 0.1f);
    init_random_half(h_w.data(), vec_size, -0.1f, 0.1f);
    init_random_half(h_k.data(), vec_size, -0.1f, 0.1f);
    init_random_half(h_v.data(), vec_size, -0.1f, 0.1f);
    init_random_half(h_a.data(), vec_size, -0.1f, 0.1f);
    init_random_half(h_b.data(), vec_size, -0.1f, 0.1f);
    for (int i = 0; i < B; i++) {
        h_elapsed_t[i] = i;
    }
    
    // 分配设备内存
    half *d_state, *d_r, *d_w, *d_k, *d_v, *d_a, *d_b, *d_y;
    int *d_elapsed_t;
    
    CUDA_CHECK(cudaMalloc(&d_state, state_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_r, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, vec_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_t, elapsed_size * sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), state_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), vec_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elapsed_t, h_elapsed_t.data(), elapsed_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // 执行 kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    wkv_forward_one(B, C, H, d_state, d_r, d_w, d_k, d_v, d_a, d_b, d_y, d_elapsed_t);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, vec_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 打印部分结果
    print_array(h_y.data(), vec_size, "Output y");
    
    // 清理
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_elapsed_t));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "Test passed!" << std::endl;
}

// 测试 forward_seq
void test_forward_seq(int B, int T, int C, int H) {
    std::cout << "\n=== Testing wkv_forward_seq ===" << std::endl;
    std::cout << "B=" << B << ", T=" << T << ", C=" << C << ", H=" << H << std::endl;
    
    // 分配主机内存
    size_t state_size = B * H * HEAD_SIZE_N * HEAD_SIZE_N;
    size_t seq_size = B * T * C;
    size_t elapsed_size = B;
    
    std::vector<half> h_state(state_size);
    std::vector<half> h_r(seq_size);
    std::vector<half> h_w(seq_size);
    std::vector<half> h_k(seq_size);
    std::vector<half> h_v(seq_size);
    std::vector<half> h_a(seq_size);
    std::vector<half> h_b(seq_size);
    std::vector<half> h_y(seq_size);
    std::vector<int> h_elapsed_t(elapsed_size);
    
    // 初始化数据
    // state 初始化为 0
    for (size_t i = 0; i < state_size; i++) {
        h_state[i] = __float2half(0.0f);
    }
    init_random_half(h_r.data(), seq_size, -0.1f, 0.1f);
    init_random_half(h_w.data(), seq_size, -0.1f, 0.1f);
    init_random_half(h_k.data(), seq_size, -0.1f, 0.1f);
    init_random_half(h_v.data(), seq_size, -0.1f, 0.1f);
    init_random_half(h_a.data(), seq_size, -0.1f, 0.1f);
    init_random_half(h_b.data(), seq_size, -0.1f, 0.1f);
    for (int i = 0; i < B; i++) {
        h_elapsed_t[i] = i;
    }
    
    // 分配设备内存
    half *d_state, *d_r, *d_w, *d_k, *d_v, *d_a, *d_b, *d_y;
    int *d_elapsed_t;
    
    CUDA_CHECK(cudaMalloc(&d_state, state_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_r, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_a, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, seq_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_elapsed_t, elapsed_size * sizeof(int)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_state, h_state.data(), state_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), seq_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elapsed_t, h_elapsed_t.data(), elapsed_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // 执行 kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    wkv_forward_seq(B, T, C, H, d_state, d_r, d_w, d_k, d_v, d_a, d_b, d_y, d_elapsed_t);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, seq_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 打印部分结果
    print_array(h_y.data(), std::min(seq_size, (size_t)20), "Output y (first 20)");
    
    // 清理
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_elapsed_t));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "Test passed!" << std::endl;
}

int main() {
    std::cout << "RWKV WKV Operator CUDA Test" << std::endl;
    std::cout << "============================" << std::endl;
    
    // 检查 CUDA 设备
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // 测试参数
    int H = 4;  // number of heads
    int head_size = HEAD_SIZE;  // head size
    int C = H * head_size;  // channel size
    
    // 测试 1: forward_one, B=1
    test_forward_one(1, C, H);
    
    // 测试 2: forward_one, B=4
    test_forward_one(4, C, H);
    
    // 测试 3: forward_seq, B=1, T=10
    test_forward_seq(1, 10, C, H);
    
    // 测试 4: forward_seq, B=2, T=5
    test_forward_seq(2, 5, C, H);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}

