#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdint>

// 采样函数（CPU 实现）
// sampler_simple_batch: 批量采样，返回 [batch_size] 形状
// logits: [batch_size, vocab_size] - 输入 logits（GPU，会被修改）
// output: [batch_size] - 输出的 token ID（CPU）
// noise: 噪声参数（默认 0.0）
// temp: 温度参数（默认 1.0）
void sampler_simple_batch(
    half* logits,           // [batch_size, vocab_size] (GPU，会被修改)
    int* output,            // [batch_size] 输出（CPU）
    int batch_size,
    int vocab_size,
    float noise = 0.0f,
    float temp = 1.0f
);

// sampler_simple: 单个样本采样，返回标量
// logits: [vocab_size] - 输入 logits（会被修改）
// noise: 噪声参数（默认 0.0）
// temp: 温度参数（默认 1.0）
// 返回: token ID
int sampler_simple(
    half* logits,           // [vocab_size] (会被修改)
    int vocab_size,
    float noise = 0.0f,
    float temp = 1.0f
);


#endif // SAMPLER_CUH

