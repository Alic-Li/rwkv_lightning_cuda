#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef RWKV_USE_HIP
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <hip/hip_bfloat16.h>
using bf16 = hip_bfloat16;
#define __shfl_down_sync(mask, value, delta) __shfl_down(value, delta, 32)
// hip_bfloat16 is only a minimal POD when this header is compiled by a regular
// host compiler (rather than hipcc), so its C++ conversion operators are not
// available there.  Convert through the public bit representation so the same
// helpers work in both host-only translation units and HIP kernels.
__host__ __device__ inline float to_float(bf16 x) {
  union {
    unsigned int bits;
    float value;
  } converted = {static_cast<unsigned int>(x.data) << 16};
  return converted.value;
}
__host__ __device__ inline bf16 to_bf16(float x) {
  union {
    float value;
    unsigned int bits;
  } converted = {x};

  // Round finite values to nearest-even and retain a non-zero NaN payload,
  // matching ROCm's hip_bfloat16(float) conversion.
  if ((converted.bits & 0x7f800000u) != 0x7f800000u)
    converted.bits += 0x7fffu + ((converted.bits >> 16) & 1u);
  else if (converted.bits & 0xffffu)
    converted.bits |= 0x10000u;

  bf16 result{};
  result.data = static_cast<decltype(result.data)>(converted.bits >> 16);
  return result;
}
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;
__host__ __device__ inline float to_float(bf16 x) {
  return __bfloat162float(x);
}
__host__ __device__ inline bf16 to_bf16(float x) {
  return __float2bfloat16_rn(x);
}
#endif
namespace rwkv_bf16_training {
// Token position within a lane. Dense layouts pass offsets=nullptr and T rows
// per lane. Packed layouts pass device offsets[lanes + 1]: lane b owns rows
// [offsets[b], offsets[b + 1]) and may be empty; a valid row never maps to an
// empty lane.
struct LaneRow {
  int lane;
  bool first, last;
};
__host__ __device__ inline LaneRow lane_row(const int *offsets, int lanes,
                                            int T, long long row) {
  if (!offsets) {
    const int t = static_cast<int>(row % T);
    return {static_cast<int>(row / T), t == 0, t == T - 1};
  }
  int lo = 0, hi = lanes;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return {lo, offsets[lo] == row, offsets[lo + 1] == row + 1};
}
inline void gpu_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <class T> struct DeviceBuffer {
  T *p = nullptr;
  size_t n = 0;
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (p)
      cudaFree(p);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&b) noexcept
      : p(std::exchange(b.p, nullptr)), n(std::exchange(b.n, 0)) {}
  DeviceBuffer &operator=(DeviceBuffer &&b) noexcept {
    if (this != &b) {
      if (p)
        cudaFree(p);
      p = std::exchange(b.p, nullptr);
      n = std::exchange(b.n, 0);
    }
    return *this;
  }
  void resize(size_t count, const char *name) {
    if (count == n)
      return;
    T *next = nullptr;
    if (count)
      gpu_check(cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(T)),
                name);
    if (p)
      cudaFree(p);
    p = next;
    n = count;
  }
  void zero(const char *name) {
    if (n)
      gpu_check(cudaMemset(p, 0, n * sizeof(T)), name);
  }
};
} // namespace rwkv_bf16_training
