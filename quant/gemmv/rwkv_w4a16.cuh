#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// X[M,K], Y[M,N], packed row-major W[N,ceil(K/2)]. Even k is the low
// nibble; each nibble is a signed two's-complement int4 (-8..7).
// FP16 scales[N,ceil(K/group_size)], group_size = 32 or 128.
// Dequantization: W[n,k] = signed_int4(q[n,k]) * scale[n,k/group_size].
// INT4 records in .rwkvq archives use this same device-ready NK layout.
std::size_t rwkv7_w4a16_weight_bytes(int N, int K);
std::size_t rwkv7_w4a16_scale_count(int N, int K, int group_size = 128);
std::size_t rwkv7_w4a16_workspace_bytes(int M, int N, int split_k);

// Quantize finite FP16 NK weights, round-to-nearest-even, clamp to [-7,7].
// Scales are rounded to FP16 before quantizing; zero groups use scale=1.
// Odd-K padding uses a zero high nibble. Input and output must not overlap.
void rwkv7_w4a16_quantize_launch(cudaStream_t stream, const half* weight_nk, std::uint8_t* qweight,
                                 half* scale, int N, int K, int group_size = 128);

// Asynchronous and CUDA-graph safe: no allocation, synchronization, atomics,
// device queries or global mutable state. Concurrent streams must use separate
// Y/workspaces; X/weights/scales can be shared. Workspace is float-aligned.
// split_k=0 selects a conservative shape-based split if workspace permits;
// split_k=1 disables splitting; explicit >1 requires the queried workspace.
// Invalid arguments throw std::invalid_argument; nonpositive shapes are no-ops.
void rwkv7_w4a16_linear_launch(cudaStream_t stream, int M, int K, int N, const half* x,
                               const std::uint8_t* qweight, const half* scale, half* y, int group_size = 128,
                               void* workspace = nullptr, std::size_t workspace_bytes = 0,
                               int force_split_k = 0);
