#pragma once
#include "rwkv/bf16_training/runtime.hpp"
#include <stdexcept>

namespace rwkv_bf16_miss {
namespace detail {
inline void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace detail
} // namespace rwkv_bf16_miss
