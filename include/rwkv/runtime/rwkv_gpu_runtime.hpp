#pragma once

#ifdef RWKV_USE_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaStreamSynchronize hipStreamSynchronize
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
