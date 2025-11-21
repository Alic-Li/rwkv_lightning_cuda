#include "sampler.cuh"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>

void sampler_simple_batch(
    half* logits,
    int* output,
    int batch_size,
    int vocab_size,
    float noise,
    float temp
) {
    // 将 logits 从 GPU 复制到 CPU
    std::vector<half> cpu_logits_half(batch_size * vocab_size);
    cudaMemcpy(cpu_logits_half.data(), logits, batch_size * vocab_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    std::vector<float> cpu_logits(batch_size * vocab_size);
    
    // 处理每个 batch
    for (int b = 0; b < batch_size; b++) {
        float* batch_logits = cpu_logits.data() + b * vocab_size;
        
        // 复制并转换 half 到 float
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] = __half2float(cpu_logits_half[b * vocab_size + i]);
        }
        
        // 应用温度
        if (temp != 1.0f) {
            for (int i = 0; i < vocab_size; i++) {
                batch_logits[i] /= temp;
            }
        }
        
        // 添加噪声
        if (noise != 0.0f) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, noise);
            
            for (int i = 0; i < vocab_size; i++) {
                batch_logits[i] += dis(gen);
            }
        }
        
        // 找到最大值索引
        int max_idx = 0;
        float max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (batch_logits[i] > max_val) {
                max_val = batch_logits[i];
                max_idx = i;
            }
        }
        
        // output 在 CPU 上，直接赋值
        output[b] = max_idx;
        
        // 如果需要修改原始 logits（inplace），写回 GPU
        if (temp != 1.0f || noise != 0.0f) {
            for (int i = 0; i < vocab_size; i++) {
                cpu_logits_half[b * vocab_size + i] = __float2half(batch_logits[i]);
            }
        }
    }
    
    // 如果需要修改，写回 GPU
    if (temp != 1.0f || noise != 0.0f) {
        cudaMemcpy(logits, cpu_logits_half.data(), batch_size * vocab_size * sizeof(half), cudaMemcpyHostToDevice);
    }
}

int sampler_simple(
    half* logits,
    int vocab_size,
    float noise,
    float temp
) {
    // 将 logits 从 GPU 复制到 CPU
    std::vector<half> cpu_logits_half(vocab_size);
    cudaMemcpy(cpu_logits_half.data(), logits, vocab_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    std::vector<float> cpu_logits(vocab_size);
    
    // 复制并转换
    for (int i = 0; i < vocab_size; i++) {
        cpu_logits[i] = __half2float(cpu_logits_half[i]);
    }
    
    // 应用温度
    if (temp != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            cpu_logits[i] /= temp;
        }
    }
    
    // 添加噪声
    if (noise != 0.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, noise);
        
        for (int i = 0; i < vocab_size; i++) {
            cpu_logits[i] += dis(gen);
        }
    }
    
    // 找到最大值索引
    int max_idx = 0;
    float max_val = cpu_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (cpu_logits[i] > max_val) {
            max_val = cpu_logits[i];
            max_idx = i;
        }
    }
    
    // 如果需要修改原始 logits（inplace），写回 GPU
    if (temp != 1.0f || noise != 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            cpu_logits_half[i] = __float2half(cpu_logits[i]);
        }
        cudaMemcpy(logits, cpu_logits_half.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice);
    }
    
    return max_idx;
}

