#pragma once

#include <cstdint>
#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum class W8BLayout {
  NK,
  KN,
  PackedNK,
};

inline constexpr int kTuneVersion = 4;

struct W8A16DeviceInfo {
  int sm_count = 1;
  int compute_major = 0;
  char name[256] = {};
};

W8A16DeviceInfo rwkv7_w8a16_device_info();
void rwkv7_w8a16_tuning_reset();
void rwkv7_w8a16_tuning_set(int K, int N, W8BLayout layout, int m_bucket, int split_k);
int rwkv7_w8a16_tuning_get(int K, int N, W8BLayout layout, int M);

void rwkv7_v4_i8_pack_launch(
    cudaStream_t stream, const std::int8_t* src_nk, std::int8_t* dst_packed, int N, int K);

void rwkv7_w8a16_linear_launch(
    cudaStream_t stream, int M, int K, int N,
    const half* x, const std::int8_t* qweight, const half* scale,
    W8BLayout layout, half* y, void* workspace = nullptr,
    std::size_t workspace_bytes = 0, int force_split_k = 0);
