#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/module/tensor_utils.cuh"

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

// 辅助函数：打印数组
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


/// 测试 zeros fp32 (多种维度)
void test_zeros_fp32() {
    std::cout << "\n=== Testing zeros fp32 (multiple dimensions) ===" << std::endl;
    
    // 1D测试: size=1024
    {
        std::cout << "\n--- Testing 1D zeros ---" << std::endl;
        int size = 1024;
        std::cout << "Size: " << size << std::endl;
        
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp32(d_output, size, 0); // 1D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<float> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (h_output[i] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 1D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 1D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 2D测试: [128, 256]
    {
        std::cout << "\n--- Testing 2D zeros ---" << std::endl;
        int dim0 = 128, dim1 = 256;
        int size = dim0 * dim1;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << "]" << std::endl;
        
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp32(d_output, dim0, dim1, 0); // 2D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<float> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (h_output[i] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 2D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 2D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 3D测试: [32, 64, 32]
    {
        std::cout << "\n--- Testing 3D zeros ---" << std::endl;
        int dim0 = 32, dim1 = 64, dim2 = 32;
        int size = dim0 * dim1 * dim2;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << ", " << dim2 << "]" << std::endl;
        
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp32(d_output, dim0, dim1, dim2, 0); // 3D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<float> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (h_output[i] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 3D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 3D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 4D测试: [16, 32, 16, 16]
    {
        std::cout << "\n--- Testing 4D zeros ---" << std::endl;
        int dim0 = 16, dim1 = 32, dim2 = 16, dim3 = 16;
        int size = dim0 * dim1 * dim2 * dim3;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << ", " << dim2 << ", " << dim3 << "]" << std::endl;
        
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp32(d_output, dim0, dim1, dim2, dim3, 0); // 4D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<float> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (h_output[i] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 4D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 4D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}
/// 测试 zeros fp16 (多种维度)
void test_zeros_fp16() {
    std::cout << "\n=== Testing zeros fp16 (multiple dimensions) ===" << std::endl;
    
    // 1D测试: size=1024
    {
        std::cout << "\n--- Testing 1D zeros ---" << std::endl;
        int size = 1024;
        std::cout << "Size: " << size << std::endl;
        
        half* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp16(d_output, size, 0); // 1D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<half> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (__half2float(h_output[i]) != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 1D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 1D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 2D测试: [128, 256]
    {
        std::cout << "\n--- Testing 2D zeros ---" << std::endl;
        int dim0 = 128, dim1 = 256;
        int size = dim0 * dim1;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << "]" << std::endl;
        
        half* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp16(d_output, dim0, dim1, 0); // 2D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<half> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (__half2float(h_output[i]) != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 2D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 2D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 3D测试: [32, 64, 32]
    {
        std::cout << "\n--- Testing 3D zeros ---" << std::endl;
        int dim0 = 32, dim1 = 64, dim2 = 32;
        int size = dim0 * dim1 * dim2;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << ", " << dim2 << "]" << std::endl;
        
        half* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp16(d_output, dim0, dim1, dim2, 0); // 3D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<half> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (__half2float(h_output[i]) != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 3D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 3D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 4D测试: [16, 32, 16, 16]
    {
        std::cout << "\n--- Testing 4D zeros ---" << std::endl;
        int dim0 = 16, dim1 = 32, dim2 = 16, dim3 = 16;
        int size = dim0 * dim1 * dim2 * dim3;
        std::cout << "Shape: [" << dim0 << ", " << dim1 << ", " << dim2 << ", " << dim3 << "]" << std::endl;
        
        half* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        zeros_fp16(d_output, dim0, dim1, dim2, dim3, 0); // 4D版本
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
        
        // 验证结果
        std::vector<half> h_output(size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
        
        bool all_zero = true;
        for (int i = 0; i < size; i++) {
            if (__half2float(h_output[i]) != 0.0f) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            std::cout << "✓ 4D Test passed!" << std::endl;
        } else {
            std::cout << "✗ 4D Test failed!" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// 测试 tanh
void test_tanh(size_t size) {
    std::cout << "\n=== Testing tanh ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_input(size);
    init_random_half(h_input.data(), size, -2.0f, 2.0f);
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    tanh_fp16(d_input, d_output, size);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    print_array(h_input.data(), std::min(size, (size_t)10), "Input");
    print_array(h_output.data(), std::min(size, (size_t)10), "Output");
    std::cout << "✓ Test completed!" << std::endl;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试 sigmoid
void test_sigmoid(size_t size) {
    std::cout << "\n=== Testing sigmoid ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_input(size);
    init_random_half(h_input.data(), size, -5.0f, 5.0f);
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    sigmoid_fp16(d_input, d_output, size);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    print_array(h_input.data(), std::min(size, (size_t)10), "Input");
    print_array(h_output.data(), std::min(size, (size_t)10), "Output");
    std::cout << "✓ Test completed!" << std::endl;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试 cat (dim=0)
void test_cat_dim0(int M1, int M2, int N) {
    std::cout << "\n=== Testing cat (dim=0) ===" << std::endl;
    std::cout << "Shape: [" << M1 << ", " << N << "] + [" << M2 << ", " << N << "] -> [" << (M1+M2) << ", " << N << "]" << std::endl;
    
    size_t size1 = M1 * N;
    size_t size2 = M2 * N;
    size_t output_size = (M1 + M2) * N;
    
    std::vector<half> h_tensor1(size1);
    std::vector<half> h_tensor2(size2);
    init_random_half(h_tensor1.data(), size1);
    init_random_half(h_tensor2.data(), size2);
    
    half *d_tensor1, *d_tensor2, *d_output;
    CUDA_CHECK(cudaMalloc(&d_tensor1, size1 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_tensor2, size2 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_tensor1, h_tensor1.data(), size1 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tensor2, h_tensor2.data(), size2 * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cat_fp16(d_tensor1, d_tensor2, d_output, 0, M1, N, M1, M2);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    std::vector<half> h_output(output_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    print_array(h_tensor1.data(), std::min(size1, (size_t)5), "Tensor1 (first row)");
    print_array(h_tensor2.data(), std::min(size2, (size_t)5), "Tensor2 (first row)");
    print_array(h_output.data(), std::min(output_size, (size_t)10), "Output (first row)");
    std::cout << "✓ Test completed!" << std::endl;
    
    CUDA_CHECK(cudaFree(d_tensor1));
    CUDA_CHECK(cudaFree(d_tensor2));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// 测试 cat (dim=1)
void test_cat_dim1(int M, int N1, int N2) {
    std::cout << "\n=== Testing cat (dim=1) ===" << std::endl;
    std::cout << "Shape: [" << M << ", " << N1 << "] + [" << M << ", " << N2 << "] -> [" << M << ", " << (N1+N2) << "]" << std::endl;
    
    size_t size1 = M * N1;
    size_t size2 = M * N2;
    size_t output_size = M * (N1 + N2);
    
    std::vector<half> h_tensor1(size1);
    std::vector<half> h_tensor2(size2);
    init_random_half(h_tensor1.data(), size1);
    init_random_half(h_tensor2.data(), size2);
    
    half *d_tensor1, *d_tensor2, *d_output;
    CUDA_CHECK(cudaMalloc(&d_tensor1, size1 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_tensor2, size2 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_tensor1, h_tensor1.data(), size1 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tensor2, h_tensor2.data(), size2 * sizeof(half), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cat_fp16(d_tensor1, d_tensor2, d_output, 1, M, N1, N1, N2);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "GPU execution time: " << elapsed_ms << " ms" << std::endl;
    
    std::vector<half> h_output(output_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 打印第一行
    print_array(h_tensor1.data(), std::min((size_t)N1, (size_t)5), "Tensor1 (first row)");
    print_array(h_tensor2.data(), std::min((size_t)N2, (size_t)5), "Tensor2 (first row)");
    print_array(h_output.data(), std::min((size_t)(N1+N2), (size_t)10), "Output (first row)");
    std::cout << "✓ Test completed!" << std::endl;
    
    CUDA_CHECK(cudaFree(d_tensor1));
    CUDA_CHECK(cudaFree(d_tensor2));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ==================== 逐元素操作测试 ====================

void test_element_wise_sub(int size) {
    std::cout << "\n=== Testing element_wise_sub ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_a(size), h_b(size);
    init_random_half(h_a.data(), size);
    init_random_half(h_b.data(), size);
    
    half *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    element_wise_sub_fp16(d_a, d_b, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < std::min(size, 100); i++) {
        float expected = __half2float(h_a[i]) - __half2float(h_b[i]);
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 0.01f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
}

void test_element_wise_add(int size) {
    std::cout << "\n=== Testing element_wise_add ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_a(size), h_b(size);
    init_random_half(h_a.data(), size);
    init_random_half(h_b.data(), size);
    
    half *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    element_wise_add_fp16(d_a, d_b, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < std::min(size, 100); i++) {
        float expected = __half2float(h_a[i]) + __half2float(h_b[i]);
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 0.01f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
}

void test_element_wise_mul(int size) {
    std::cout << "\n=== Testing element_wise_mul ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_a(size), h_b(size);
    init_random_half(h_a.data(), size);
    init_random_half(h_b.data(), size);
    
    half *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    element_wise_mul_fp16(d_a, d_b, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < std::min(size, 100); i++) {
        float expected = __half2float(h_a[i]) * __half2float(h_b[i]);
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 0.01f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
}

void test_copy(int size) {
    std::cout << "\n=== Testing copy ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_src(size);
    init_random_half(h_src.data(), size);
    
    half *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    copy_fp16(d_src, d_dst, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_dst(size);
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (__half2float(h_src[i]) != __half2float(h_dst[i])) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

void test_sum_reduce_last_dim(int H, int N) {
    std::cout << "\n=== Testing sum_reduce_last_dim ===" << std::endl;
    std::cout << "Shape: [" << H << ", " << N << "] -> [" << H << ", 1]" << std::endl;
    
    int input_size = H * N;
    std::vector<half> h_input(input_size);
    init_random_half(h_input.data(), input_size);
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, H * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    
    sum_reduce_last_dim_fp16(d_input, d_output, H, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(H);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, H * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int h = 0; h < H; h++) {
        float expected = 0.0f;
        for (int n = 0; n < N; n++) {
            expected += __half2float(h_input[h * N + n]);
        }
        float actual = __half2float(h_output[h]);
        if (std::abs(expected - actual) > 0.1f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_broadcast(int H, int N) {
    std::cout << "\n=== Testing broadcast ===" << std::endl;
    std::cout << "Shape: [" << H << ", 1] -> [" << H << ", " << N << "]" << std::endl;
    
    std::vector<half> h_input(H);
    init_random_half(h_input.data(), H);
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, H * N * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), H * sizeof(half), cudaMemcpyHostToDevice));
    
    broadcast_fp16(d_input, d_output, H, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(H * N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, H * N * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int h = 0; h < H; h++) {
        float expected = __half2float(h_input[h]);
        for (int n = 0; n < N; n++) {
            float actual = __half2float(h_output[h * N + n]);
            if (std::abs(expected - actual) > 0.001f) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_negate(int size) {
    std::cout << "\n=== Testing negate ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    
    std::vector<half> h_input(size);
    init_random_half(h_input.data(), size);
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    negate_fp16(d_input, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = -__half2float(h_input[i]);
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 0.001f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_element_wise_add_const(int size) {
    std::cout << "\n=== Testing element_wise_add_const ===" << std::endl;
    std::cout << "Size: " << size << ", Constant: 2.5" << std::endl;
    
    std::vector<half> h_input(size);
    init_random_half(h_input.data(), size);
    float constant = 2.5f;
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    element_wise_add_const_fp16(d_input, constant, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<half> h_output(size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < std::min(size, 100); i++) {
        float expected = __half2float(h_input[i]) + constant;
        float actual = __half2float(h_output[i]);
        if (std::abs(expected - actual) > 0.01f) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Test passed!" << std::endl;
    } else {
        std::cout << "✗ Test failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    std::cout << "RWKV Utils Module CUDA Test" << std::endl;
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

    test_zeros_fp32();
    test_zeros_fp16();
    
    test_tanh(1024);
    test_tanh(65536);
    
    test_sigmoid(1024);
    test_sigmoid(65536);
    
    test_cat_dim0(5, 3, 10);
    test_cat_dim0(100, 200, 256);
    
    test_cat_dim1(10, 5, 3);
    test_cat_dim1(512, 256, 256);
    
    // 逐元素操作测试
    test_element_wise_sub(1024);
    test_element_wise_sub(65536);
    
    test_element_wise_add(1024);
    test_element_wise_add(65536);
    
    test_element_wise_mul(1024);
    test_element_wise_mul(65536);
    
    // 复制操作测试
    test_copy(1024);
    test_copy(65536);
    
    // 归约操作测试
    test_sum_reduce_last_dim(32, 64);
    test_sum_reduce_last_dim(128, 256);
    
    // 广播操作测试
    test_broadcast(32, 64);
    test_broadcast(128, 256);
    
    // 其他操作测试
    test_negate(1024);
    test_negate(65536);
    
    test_element_wise_add_const(1024);
    test_element_wise_add_const(65536);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}
