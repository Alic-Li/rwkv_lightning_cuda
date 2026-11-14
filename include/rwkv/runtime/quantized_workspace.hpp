#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

// HIP W8 split-K stores one FP32 output matrix per split.
inline std::size_t rwkv7_quantized_workspace_bytes(int m, int n, int splits) {
  if (m <= 0 || n <= 0 || splits <= 1)
    return 0;
  if (std::size_t(m) >
      std::numeric_limits<std::size_t>::max() / n / splits / sizeof(float))
    throw std::invalid_argument("quantized workspace overflow");
  return std::size_t(m) * n * splits * sizeof(float);
}
