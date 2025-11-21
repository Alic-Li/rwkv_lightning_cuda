#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/module/norm.cuh"

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

// 测试 L2 Normalization
void test_l2_normalize(int M, int N) {
    std::cout << "\n=== Testing L2 Normalize ===" << std::endl;
    std::cout << "Shape: [" << M << ", " << N << "]" << std::endl;
    
    size_t input_size = M * N;
    
    // 分配主机内存
    std::vector<half> h_input(input_size);
    std::vector<half> h_output_gpu(input_size);
    
    // 初始化数据
    init_random_half(h_input.data(), input_size);
    
    // 分配设备内存
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(half)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // 执行 GPU kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    l2_normalize_fp16(d_input, d_output, M, N, 1e-5f);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, input_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU result sample: ";
    print_array(h_output_gpu.data(), std::min(input_size, (size_t)10), "", 10);
    std::cout << "✓ GPU execution completed!" << std::endl;
    
    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试 Layer Normalization
void test_layer_norm_half8(int M, int N, bool use_beta = false) {
    std::cout << "\n=== Testing Layer Norm (Half8) ===" << std::endl;
    std::cout << "Shape: [" << M << ", " << N << "]" << std::endl;
    std::cout << "Use beta: " << (use_beta ? "Yes" : "No") << std::endl;
    
    size_t input_size = M * N;
    size_t param_size = N;
    
    // 分配主机内存
    std::vector<half> h_input(input_size);
    std::vector<half> h_gamma(param_size);
    std::vector<half> h_beta(use_beta ? param_size : 0);
    std::vector<half> h_output_gpu(input_size);
    
    // 初始化数据
    init_random_half(h_input.data(), input_size);
    // gamma 初始化为 1.0
    for (size_t i = 0; i < param_size; i++) {
        h_gamma[i] = __float2half(1.0f);
    }
    if (use_beta) {
        init_random_half(h_beta.data(), param_size, -0.1f, 0.1f);
    }
    
    // 分配设备内存
    half *d_input, *d_gamma, *d_beta, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(half)));
    if (use_beta) {
        CUDA_CHECK(cudaMalloc(&d_beta, param_size * sizeof(half)));
    }
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), param_size * sizeof(half), cudaMemcpyHostToDevice));
    if (use_beta) {
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), param_size * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // 执行 GPU kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    layer_norm_half8_fp16(d_input, d_gamma, use_beta ? d_beta : nullptr, d_output, M, N, 1e-5f);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, input_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU result sample: ";
    print_array(h_output_gpu.data(), std::min(input_size, (size_t)10), "", 10);
    std::cout << "✓ GPU execution completed!" << std::endl;
    
    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_output));
    if (use_beta) {
        CUDA_CHECK(cudaFree(d_beta));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试 Group Normalization
void test_group_norm_half8(int batch_size, int num_groups, int channels_per_group, bool use_beta = false) {
    std::cout << "\n=== Testing Group Norm (Half8) ===" << std::endl;
    std::cout << "Batch size: " << batch_size << ", Groups: " << num_groups 
              << ", Channels per group: " << channels_per_group << std::endl;
    std::cout << "Use beta: " << (use_beta ? "Yes" : "No") << std::endl;
    
    int total_groups = batch_size * num_groups;
    int total_channels = num_groups * channels_per_group;
    size_t input_size = total_groups * channels_per_group;
    size_t param_size = total_channels;
    
    // 分配主机内存
    std::vector<half> h_input(input_size);
    std::vector<half> h_gamma(param_size);
    std::vector<half> h_beta(use_beta ? param_size : 0);
    std::vector<half> h_output_gpu(input_size);
    
    // 初始化数据
    init_random_half(h_input.data(), input_size);
    // gamma 初始化为 1.0
    for (size_t i = 0; i < param_size; i++) {
        h_gamma[i] = __float2half(1.0f);
    }
    if (use_beta) {
        init_random_half(h_beta.data(), param_size, -0.1f, 0.1f);
    }
    
    // 分配设备内存
    half *d_input, *d_gamma, *d_beta, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(half)));
    if (use_beta) {
        CUDA_CHECK(cudaMalloc(&d_beta, param_size * sizeof(half)));
    }
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), param_size * sizeof(half), cudaMemcpyHostToDevice));
    if (use_beta) {
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), param_size * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // 执行 GPU kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    group_norm_half8_fp16(d_input, d_gamma, use_beta ? d_beta : nullptr, d_output,
                         batch_size, num_groups, channels_per_group, 1e-5f);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, input_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU result sample: ";
    print_array(h_output_gpu.data(), std::min(input_size, (size_t)10), "", 10);
    std::cout << "✓ GPU execution completed!" << std::endl;
    
    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_output));
    if (use_beta) {
        CUDA_CHECK(cudaFree(d_beta));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "RWKV Norm Module CUDA Test" << std::endl;
    std::cout << "===========================" << std::endl;
    
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
    
    // Warmup: 初始化 CUDA kernels（懒加载）
    {
        std::cout << "\n=== Warmup (initializing CUDA kernels) ===" << std::endl;
        size_t warmup_size = 64;
        half *d_input_warmup, *d_output_warmup, *d_gamma_warmup;
        CUDA_CHECK(cudaMalloc(&d_input_warmup, warmup_size * warmup_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_output_warmup, warmup_size * warmup_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_gamma_warmup, warmup_size * sizeof(half)));
        
        // 执行一次小的操作来触发 kernel 初始化
        l2_normalize_fp16(d_input_warmup, d_output_warmup, warmup_size, warmup_size, 1e-5f);
        layer_norm_half8_fp16(d_input_warmup, d_gamma_warmup, nullptr, d_output_warmup, warmup_size, warmup_size, 1e-5f);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_input_warmup));
        CUDA_CHECK(cudaFree(d_output_warmup));
        CUDA_CHECK(cudaFree(d_gamma_warmup));
        std::cout << "✓ Warmup completed!" << std::endl;
    }
    
    // 测试 L2 Normalization
    test_l2_normalize(1, 256);
    test_l2_normalize(512, 1024);
    test_l2_normalize(64, 4096);
    
    // 测试 Layer Normalization (Half8)
    test_layer_norm_half8(1, 256, false);
    test_layer_norm_half8(1, 256, true);
    test_layer_norm_half8(512, 1024, false);
    test_layer_norm_half8(512, 1024, true);
    test_layer_norm_half8(64, 4096, false);
    test_layer_norm_half8(64, 4096, true);
    
    // 测试 Group Normalization (Half8)
    test_group_norm_half8(1, 32, 128, false);
    test_group_norm_half8(1, 32, 128, true);
    test_group_norm_half8(8, 32, 128, false);
    test_group_norm_half8(8, 32, 128, true);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}

