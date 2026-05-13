#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "rwkv7_fast_v4_common.hpp"

namespace rwkv_test {

inline std::filesystem::path repo_root() {
  return std::filesystem::path(RWKV_TEST_REPO_ROOT);
}

inline std::filesystem::path vocab_path() {
  return repo_root() / "src" / "rwkv_vocab_v20230424.txt";
}

inline std::filesystem::path unique_temp_path(const std::string& stem) {
  const auto ts = std::to_string(
      static_cast<unsigned long long>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  return std::filesystem::temp_directory_path() / (stem + "_" + ts);
}

inline void require_cuda(cudaError_t err, const char* what) {
  rwkv7_fast_v4::check_cuda(err, what);
}

inline bool cuda_device_available() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

template <typename T>
std::vector<T> copy_device_buffer(const rwkv7_fast_v4::DeviceBuffer<T>& buffer, const char* label) {
  std::vector<T> host(buffer.n);
  if (buffer.n > 0) {
    require_cuda(cudaMemcpy(host.data(), buffer.p, buffer.n * sizeof(T), cudaMemcpyDeviceToHost), label);
  }
  return host;
}

template <typename T>
void copy_host_to_device(
    const std::vector<T>& host,
    rwkv7_fast_v4::DeviceBuffer<T>& buffer,
    const char* resize_label,
    const char* copy_label) {
  buffer.resize(host.size(), resize_label);
  if (!host.empty()) {
    require_cuda(cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), copy_label);
  }
}

inline std::vector<std::uint16_t> to_half_bits(const std::vector<float>& values) {
  std::vector<std::uint16_t> bits;
  bits.reserve(values.size());
  for (float value : values) {
    const half h = __float2half(value);
    bits.push_back(*reinterpret_cast<const std::uint16_t*>(&h));
  }
  return bits;
}

template <typename T>
std::string value_to_string(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

template <typename T>
std::string value_to_string(const std::vector<T>& values) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << value_to_string(values[i]);
  }
  oss << "]";
  return oss.str();
}

template <typename T, typename U>
void expect_eq_impl(const T& lhs, const U& rhs, const char* lhs_expr, const char* rhs_expr, const char* file, int line) {
  if (!(lhs == rhs)) {
    std::ostringstream oss;
    oss << file << ":" << line << " expected " << lhs_expr << " == " << rhs_expr
        << ", got " << value_to_string(lhs) << " vs " << value_to_string(rhs);
    throw std::runtime_error(oss.str());
  }
}

inline void expect_true_impl(bool value, const char* expr, const char* file, int line) {
  if (!value) {
    std::ostringstream oss;
    oss << file << ":" << line << " check failed: " << expr;
    throw std::runtime_error(oss.str());
  }
}

inline void expect_near_impl(double lhs, double rhs, double eps, const char* lhs_expr, const char* rhs_expr, const char* file, int line) {
  if (std::fabs(lhs - rhs) > eps) {
    std::ostringstream oss;
    oss << file << ":" << line << " expected " << lhs_expr << " ~= " << rhs_expr
        << " within " << eps << ", got " << lhs << " vs " << rhs;
    throw std::runtime_error(oss.str());
  }
}

template <typename Fn>
void expect_throw_impl(Fn&& fn, const char* expr, const char* file, int line) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) {
    std::ostringstream oss;
    oss << file << ":" << line << " expected exception from " << expr;
    throw std::runtime_error(oss.str());
  }
}

template <typename T>
bool contains(const std::vector<T>& values, const T& target) {
  return std::find(values.begin(), values.end(), target) != values.end();
}

}  // namespace rwkv_test

#define TEST_CHECK(expr) ::rwkv_test::expect_true_impl(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
#define TEST_EQ(lhs, rhs) ::rwkv_test::expect_eq_impl((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__)
#define TEST_NEAR(lhs, rhs, eps) ::rwkv_test::expect_near_impl((lhs), (rhs), (eps), #lhs, #rhs, __FILE__, __LINE__)
#define TEST_THROW(expr) ::rwkv_test::expect_throw_impl([&]() { expr; }, #expr, __FILE__, __LINE__)
