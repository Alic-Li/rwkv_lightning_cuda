#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/utils/sampler.cuh"

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

// 测试采样函数
void test_sampler_simple() {
    std::cout << "\n=== Testing sampler_simple ===" << std::endl;
    
    const int vocab_size = 65536;
    std::vector<half> logits(vocab_size);
    
    // 初始化 logits（创建一个明显的峰值）
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = __float2half(0.0f);
    }
    logits[100] = __float2half(10.0f);  // 在位置 100 创建一个峰值
    
    // 分配 GPU 内存
    half* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // 测试 greedy decoding (noise=0, temp=1)
    int result = sampler_simple(d_logits, vocab_size, 0.0f, 1.0f);
    
    std::cout << "Greedy decoding result: " << result << " (expected: 100)" << std::endl;
    if (result == 100) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    // 测试带温度
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));
    result = sampler_simple(d_logits, vocab_size, 0.0f, 2.0f);
    std::cout << "With temp=2.0 result: " << result << std::endl;
    
    // 测试带噪声
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));
    result = sampler_simple(d_logits, vocab_size, 0.1f, 1.0f);
    std::cout << "With noise=0.1 result: " << result << std::endl;
    
    CUDA_CHECK(cudaFree(d_logits));
}

void test_sampler_simple_batch() {
    std::cout << "\n=== Testing sampler_simple_batch ===" << std::endl;
    
    const int batch_size = 4;
    const int vocab_size = 65536;
    std::vector<half> logits(batch_size * vocab_size);
    std::vector<int> output(batch_size);
    
    // 为每个 batch 创建不同的峰值
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < vocab_size; i++) {
            logits[b * vocab_size + i] = __float2half(0.0f);
        }
        logits[b * vocab_size + (b + 1) * 100] = __float2half(10.0f);  // 不同的峰值位置
    }
    
    // 分配 GPU 内存（只分配 logits，output 在 CPU）
    half* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), batch_size * vocab_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // 测试（output 是 CPU 指针，函数内部会直接写入）
    sampler_simple_batch(d_logits, output.data(), batch_size, vocab_size, 0.0f, 1.0f);
    
    std::cout << "Batch results: ";
    for (int i = 0; i < batch_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected: 100 200 300 400" << std::endl;
    
    bool passed = true;
    for (int i = 0; i < batch_size; i++) {
        if (output[i] != (i + 1) * 100) {
            passed = false;
            break;
        }
    }
    
    if (passed) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_logits));
}

int main(int argc, char* argv[]) {
    std::cout << "RWKV Util Functions Test" << std::endl;
    std::cout << "=========================" << std::endl;
    
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
    
    // 测试采样函数
    test_sampler_simple();
    test_sampler_simple_batch();
    
    std::cout << "\n✓ All tests completed!" << std::endl;
    return 0;
}

