#pragma once
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// q[N,ceil(K/32),16+4*(bits-4)] : q4 bytes, bit4 uint32, bit5 uint32.
// sm[N,ceil(K/32)] : half2(scale, signed offset), expanded once on load.
// Graph safe, allocation free. Workspace stores split-K FP32 partial outputs.
std::size_t rwkv7_asym_workspace_bytes(int M, int N, int splits);
#ifndef RWKV_USE_HIP
void rwkv7_asym_linear_launch(cudaStream_t stream, int M, int K, int N,
    const half* x, const std::uint8_t* q, const half2* sm, half* y, int bits,
    void* workspace = nullptr, std::size_t workspace_bytes = 0, int force_split_k = 0);
#else
inline void rwkv7_asym_linear_launch(cudaStream_t, int, int, int, const half*,
    const std::uint8_t*, const half2*, half*, int, void* = nullptr, std::size_t = 0, int = 0) {
  throw std::runtime_error("SB6 inference currently requires the CUDA backend");
}
#endif
