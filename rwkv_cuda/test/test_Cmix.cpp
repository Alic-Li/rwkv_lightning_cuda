#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/Cmix/Cmix.cuh"

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

// 辅助函数：分配 cmix_one 临时缓冲区
void allocate_cmix_one_temp_buffers(int C, int D, CmixOneTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_linear, D * sizeof(half)));
}

// 辅助函数：释放 cmix_one 临时缓冲区
void free_cmix_one_temp_buffers(CmixOneTempBuffers* buf) {
    cudaFree(buf->xx);
    cudaFree(buf->k);
    cudaFree(buf->k_linear);
}

// 辅助函数：分配 cmix_seq 临时缓冲区
void allocate_cmix_seq_temp_buffers(int total_size, int total_k_size, CmixSeqTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_linear, total_k_size * sizeof(half)));
}

// 辅助函数：释放 cmix_seq 临时缓冲区
void free_cmix_seq_temp_buffers(CmixSeqTempBuffers* buf) {
    cudaFree(buf->xx);
    cudaFree(buf->k);
    cudaFree(buf->k_linear);
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

// 测试 cmix_one_fp16
void test_cmix_one(int C, int D) {
    std::cout << "\n=== Testing cmix_one_fp16 ===" << std::endl;
    std::cout << "C=" << C << ", D=" << D << std::endl;
    
    // 分配主机内存
    std::vector<half> h_x(C);
    std::vector<half> h_x_prev(2 * C);
    std::vector<half> h_x_k(C);
    std::vector<half> h_K_(C * D);
    std::vector<half> h_V_(D * C);
    std::vector<half> h_output(C);
    std::vector<half> h_output_ref(C);
    
    // 初始化数据
    init_random_half(h_x.data(), C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), 2 * C, -1.0f, 1.0f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    float k_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_K_.data(), C * D, -k_scale, k_scale);
    float v_scale = 1.0f / sqrtf((float)D);
    init_random_half(h_V_.data(), D * C, -v_scale, v_scale);
    
    // 分配设备内存
    half *d_x, *d_x_prev, *d_x_k, *d_K_, *d_V_, *d_output;
    CUDA_CHECK(cudaMalloc(&d_x, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, 2 * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, D * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, C * sizeof(half)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), 2 * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), D * C * sizeof(half), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    CmixOneTempBuffers temp_buf;
    allocate_cmix_one_temp_buffers(C, D, &temp_buf);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cmix_one_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, C, D);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 执行 kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cmix_one_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, C, D);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, C * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证输出
    bool has_nan = false;
    bool all_zero = true;
    float max_val = 0.0f, min_val = 0.0f;
    for (size_t i = 0; i < C; i++) {
        float val = __half2float(h_output[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan = true;
            break;
        }
        if (val != 0.0f) all_zero = false;
        max_val = std::max(max_val, std::abs(val));
        min_val = std::min(min_val, std::abs(val));
    }
    
    if (has_nan) {
        std::cerr << "ERROR: Output contains NaN or Inf!" << std::endl;
        return;
    }
    if (all_zero) {
        std::cerr << "WARNING: Output is all zeros!" << std::endl;
    }
    std::cout << "Output range: [" << -max_val << ", " << max_val << "]" << std::endl;
    
    // 打印部分结果
    print_array(h_output.data(), C, "Output");
    
    // 清理
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_prev));
    CUDA_CHECK(cudaFree(d_x_k));
    CUDA_CHECK(cudaFree(d_K_));
    CUDA_CHECK(cudaFree(d_V_));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "✓ Test passed!" << std::endl;
}

// 测试 cmix_seq_fp16
void test_cmix_seq(int T, int C, int D) {
    std::cout << "\n=== Testing cmix_seq_fp16 ===" << std::endl;
    std::cout << "T=" << T << ", C=" << C << ", D=" << D << std::endl;
    
    // 分配主机内存
    std::vector<half> h_x(T * C);
    std::vector<half> h_x_prev(2 * C);
    std::vector<half> h_x_k(C);
    std::vector<half> h_K_(C * D);
    std::vector<half> h_V_(D * C);
    std::vector<half> h_output(T * C);
    
    // 初始化数据
    init_random_half(h_x.data(), T * C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), 2 * C, -1.0f, 1.0f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    float k_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_K_.data(), C * D, -k_scale, k_scale);
    float v_scale = 1.0f / sqrtf((float)D);
    init_random_half(h_V_.data(), D * C, -v_scale, v_scale);
    
    // 分配设备内存
    half *d_x, *d_x_prev, *d_x_k, *d_K_, *d_V_, *d_output;
    CUDA_CHECK(cudaMalloc(&d_x, T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, 2 * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, D * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, T * C * sizeof(half)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), T * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), 2 * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), D * C * sizeof(half), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    CmixSeqTempBuffers temp_buf;
    int total_size = T * C;
    int total_k_size = T * D;
    allocate_cmix_seq_temp_buffers(total_size, total_k_size, &temp_buf);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cmix_seq_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, T, C, D);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 执行 kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cmix_seq_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, T, C, D);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, T * C * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证输出
    bool has_nan = false;
    bool all_zero = true;
    float max_val = 0.0f;
    for (size_t i = 0; i < T * C; i++) {
        float val = __half2float(h_output[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan = true;
            break;
        }
        if (val != 0.0f) all_zero = false;
        max_val = std::max(max_val, std::abs(val));
    }
    
    if (has_nan) {
        std::cerr << "ERROR: Output contains NaN or Inf!" << std::endl;
        return;
    }
    if (all_zero) {
        std::cerr << "WARNING: Output is all zeros!" << std::endl;
    }
    std::cout << "Output range: [" << -max_val << ", " << max_val << "]" << std::endl;
    
    // 打印部分结果
    print_array(h_output.data(), std::min((size_t)(T * C), (size_t)20), "Output (first 20)");
    
    // 释放临时缓冲区
    free_cmix_seq_temp_buffers(&temp_buf);
    
    // 清理
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_prev));
    CUDA_CHECK(cudaFree(d_x_k));
    CUDA_CHECK(cudaFree(d_K_));
    CUDA_CHECK(cudaFree(d_V_));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "✓ Test passed!" << std::endl;
}

// 测试 cmix_seq_batch_fp16
void test_cmix_seq_batch(int B, int T, int C, int D) {
    std::cout << "\n=== Testing cmix_seq_batch_fp16 ===" << std::endl;
    std::cout << "B=" << B << ", T=" << T << ", C=" << C << ", D=" << D << std::endl;
    
    // 分配主机内存
    std::vector<half> h_x(B * T * C);
    std::vector<half> h_x_prev(2 * B * C);
    std::vector<half> h_x_k(C);
    std::vector<half> h_K_(C * D);
    std::vector<half> h_V_(D * C);
    std::vector<half> h_output(B * T * C);
    
    // 初始化数据
    init_random_half(h_x.data(), B * T * C, -1.0f, 1.0f);
    init_random_half(h_x_prev.data(), 2 * B * C, -1.0f, 1.0f);
    init_random_half(h_x_k.data(), C, -0.1f, 0.1f);
    float k_scale = 1.0f / sqrtf((float)C);
    init_random_half(h_K_.data(), C * D, -k_scale, k_scale);
    float v_scale = 1.0f / sqrtf((float)D);
    init_random_half(h_V_.data(), D * C, -v_scale, v_scale);
    
    // 分配设备内存
    half *d_x, *d_x_prev, *d_x_k, *d_K_, *d_V_, *d_output;
    CUDA_CHECK(cudaMalloc(&d_x, B * T * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_prev, 2 * B * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x_k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K_, C * D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V_, D * C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, B * T * C * sizeof(half)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), B * T * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_prev, h_x_prev.data(), 2 * B * C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_k, h_x_k.data(), C * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_, h_K_.data(), C * D * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_, h_V_.data(), D * C * sizeof(half), cudaMemcpyHostToDevice));
    
    // 分配临时缓冲区
    CmixSeqTempBuffers temp_buf;
    int total_size = B * T * C;
    int total_k_size = B * T * D;
    allocate_cmix_seq_temp_buffers(total_size, total_k_size, &temp_buf);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cmix_seq_batch_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, B, T, C, D);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 执行 kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cmix_seq_batch_fp16(d_x, d_x_prev, d_x_k, d_K_, d_V_, &temp_buf, d_output, B, T, C, D);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel execution time: " << elapsed_ms << " ms" << std::endl;
    
    // 检查错误
    CUDA_CHECK(cudaGetLastError());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, B * T * C * sizeof(half), cudaMemcpyDeviceToHost));
    
    // 验证输出
    bool has_nan = false;
    bool all_zero = true;
    float max_val = 0.0f;
    for (size_t i = 0; i < B * T * C; i++) {
        float val = __half2float(h_output[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan = true;
            break;
        }
        if (val != 0.0f) all_zero = false;
        max_val = std::max(max_val, std::abs(val));
    }
    
    if (has_nan) {
        std::cerr << "ERROR: Output contains NaN or Inf!" << std::endl;
        return;
    }
    if (all_zero) {
        std::cerr << "WARNING: Output is all zeros!" << std::endl;
    }
    std::cout << "Output range: [" << -max_val << ", " << max_val << "]" << std::endl;
    
    // 打印部分结果
    print_array(h_output.data(), std::min((size_t)(B * T * C), (size_t)20), "Output (first 20)");
    
    // 释放临时缓冲区
    free_cmix_seq_temp_buffers(&temp_buf);
    
    // 清理
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_prev));
    CUDA_CHECK(cudaFree(d_x_k));
    CUDA_CHECK(cudaFree(d_K_));
    CUDA_CHECK(cudaFree(d_V_));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "✓ Test passed!" << std::endl;
}

int main() {
    std::cout << "=== CMix CUDA Test ===" << std::endl;
    
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
    
    int C = 1024;  // 通道数
    int D = 4096;  // 内部维度
    
    // 测试 cmix_one
    test_cmix_one(C, D);
    
    // 测试 cmix_seq
    int T = 10;
    test_cmix_seq(T, C, D);
    
    // 测试 cmix_seq_batch
    int B = 10;
    test_cmix_seq_batch(B, T, C, D);
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    return 0;
}

