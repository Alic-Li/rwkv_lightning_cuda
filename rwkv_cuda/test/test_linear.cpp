#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/module/linear_cublas.cuh"

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
void print_array(const half* data, size_t size, const char* name, size_t max_print = 10) {
    if (name && strlen(name) > 0) {
        std::cout << name << ": ";
    }
    for (size_t i = 0; i < std::min(size, max_print); i++) {
        std::cout << __half2float(data[i]) << " ";
    }
    if (size > max_print) std::cout << "...";
    std::cout << std::endl;
}

// 计算相对误差
float compute_error(const half* ref, const half* out, size_t size) {
    float max_diff = 0.0f;
    float max_val = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float ref_val = __half2float(ref[i]);
        float out_val = __half2float(out[i]);
        float diff = std::abs(ref_val - out_val);
        max_diff = std::max(max_diff, diff);
        max_val = std::max(max_val, std::abs(ref_val));
    }
    
    return max_val > 1e-6f ? (max_diff / max_val) : max_diff;
}

// 测试批量矩阵乘法: [M, K] @ [K, N] → [M, N]
void test_batch_gemm(int M, int K, int N, bool use_bias = false) {
    std::cout << "\n=== Testing Batch GEMM ===" << std::endl;
    std::cout << "Shape: [" << M << ", " << K << "] @ [" << K << ", " << N << "] → [" << M << ", " << N << "]" << std::endl;
    std::cout << "Use bias: " << (use_bias ? "Yes" : "No") << std::endl;
    
    size_t A_size = M * K;
    size_t W_size = K * N;
    size_t C_size = M * N;
    size_t bias_size = N;
    
    // 分配主机内存
    std::vector<half> h_A(A_size);
    std::vector<half> h_W(W_size);
    std::vector<half> h_bias(use_bias ? bias_size : 0);
    std::vector<half> h_C_gpu(C_size);
    std::vector<half> h_C_cpu(C_size);
    
    // 初始化数据 - 使用合理的权重范围避免数值爆炸
    // 输入数据：[-1, 1]
    init_random_half(h_A.data(), A_size, -1.0f, 1.0f);
    // 权重矩阵：使用 Xavier 初始化，范围约为 [-sqrt(1/K), sqrt(1/K)]
    float weight_scale = 1.0f / sqrtf((float)K);
    init_random_half(h_W.data(), W_size, -weight_scale, weight_scale);
    // 偏置：[-0.01, 0.01]
    if (use_bias) {
        init_random_half(h_bias.data(), bias_size, -0.01f, 0.01f);
    }
    
    // 分配设备内存
    half *d_A, *d_W, *d_bias, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_W, W_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, C_size * sizeof(half)));
    if (use_bias) {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    }
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), W_size * sizeof(half), cudaMemcpyHostToDevice));
    if (use_bias) {
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // 执行 GPU kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    linear_fp16(d_A, d_W, use_bias ? d_bias : nullptr, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, C_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU result sample: ";
    print_array(h_C_gpu.data(), std::min(C_size, (size_t)10), "", 10);
    std::cout << "✓ GPU execution completed!" << std::endl;
    
    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_C));
    if (use_bias) {
        CUDA_CHECK(cudaFree(d_bias));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试向量矩阵乘法: [K] @ [K, N] → [N]
void test_vec_gemm(int K, int N, bool use_bias = false) {
    std::cout << "\n=== Testing Vector-Matrix GEMM ===" << std::endl;
    std::cout << "Shape: [" << K << "] @ [" << K << ", " << N << "] → [" << N << "]" << std::endl;
    std::cout << "Use bias: " << (use_bias ? "Yes" : "No") << std::endl;
    
    size_t x_size = K;
    size_t W_size = K * N;
    size_t y_size = N;
    size_t bias_size = N;
    
    // 分配主机内存
    std::vector<half> h_x(x_size);
    std::vector<half> h_W(W_size);
    std::vector<half> h_bias(use_bias ? bias_size : 0);
    std::vector<half> h_y_gpu(y_size);
    std::vector<half> h_y_cpu(y_size);
    
    // 初始化数据 - 使用合理的权重范围避免数值爆炸
    // 输入数据：[-1, 1]
    init_random_half(h_x.data(), x_size, -1.0f, 1.0f);
    // 权重矩阵：使用 Xavier 初始化，范围约为 [-sqrt(1/K), sqrt(1/K)]
    float weight_scale = 1.0f / sqrtf((float)K);
    init_random_half(h_W.data(), W_size, -weight_scale, weight_scale);
    // 偏置：[-0.01, 0.01]
    if (use_bias) {
        init_random_half(h_bias.data(), bias_size, -0.01f, 0.01f);
    }
    
    // 分配设备内存
    half *d_x, *d_W, *d_bias, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_W, W_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(half)));
    if (use_bias) {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    }
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), W_size * sizeof(half), cudaMemcpyHostToDevice));
    if (use_bias) {
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // 执行 GPU kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    linear_vec_fp16(d_x, d_W, use_bias ? d_bias : nullptr, d_y, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, y_size * sizeof(half), cudaMemcpyDeviceToHost));

    std::cout << "GPU result sample: ";
    print_array(h_y_gpu.data(), std::min(y_size, (size_t)10), "", 10);
    std::cout << "✓ GPU execution completed!" << std::endl;
    
    // 清理
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_y));
    if (use_bias) {
        CUDA_CHECK(cudaFree(d_bias));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "RWKV Linear Module CUDA Test" << std::endl;
    std::cout << "==============================" << std::endl;
    
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
    
    // Warmup: 初始化 cuBLAS handle（懒加载）
    {
        std::cout << "\n=== Warmup (initializing cuBLAS) ===" << std::endl;
        size_t warmup_size = 64;
        half *d_A_warmup, *d_W_warmup, *d_C_warmup;
        CUDA_CHECK(cudaMalloc(&d_A_warmup, warmup_size * warmup_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_W_warmup, warmup_size * warmup_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C_warmup, warmup_size * warmup_size * sizeof(half)));
        
        // 执行一次小的矩阵乘法来触发 cuBLAS 初始化
        linear_fp16(d_A_warmup, d_W_warmup, nullptr, d_C_warmup, warmup_size, warmup_size, warmup_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_A_warmup));
        CUDA_CHECK(cudaFree(d_W_warmup));
        CUDA_CHECK(cudaFree(d_C_warmup));
        std::cout << "✓ Warmup completed!" << std::endl;
    }
    
    // 测试 1: [512, 1024] @ [1024, 65536] → [512, 65536]
    test_batch_gemm(512, 2048, 65536, false);
    test_batch_gemm(512, 2048, 65536, true);
    
    // 测试 2: [1024] @ [1024, 65536] → [65536]
    test_vec_gemm(2048, 65536, false);
    test_vec_gemm(2048, 65536, true);
    
    // 额外测试：较小的矩阵
    test_batch_gemm(512, 1024, 65536, false);
    test_vec_gemm(1024, 65536, false);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}

