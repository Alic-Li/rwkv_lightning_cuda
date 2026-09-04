#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "rwkv7_fast_v4_kernels.cuh"
#include "test_common.hpp"

namespace {

using rwkv7_fast_v4::DeviceBuffer;

std::uint16_t to_half_bits(float value) {
  const half h = __float2half(value);
  return *reinterpret_cast<const std::uint16_t*>(&h);
}

float from_half_bits(std::uint16_t bits) {
  return __half2float(*reinterpret_cast<const half*>(&bits));
}

void check_linear(int M, int K, int N, W8BLayout layout) {
  std::mt19937 rng(static_cast<std::uint32_t>(M * 1009 + K * 17 + N));
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(M) * K);
  for (float& value : x) value = x_dist(rng);
  const std::vector<std::uint16_t> x_bits = rwkv_test::to_half_bits(x);

  std::vector<std::int8_t> q_nk(static_cast<std::size_t>(N) * K);
  std::vector<std::uint16_t> scale_bits(N);
  std::vector<float> scales(N);
  for (int n = 0; n < N; ++n) {
    scales[n] = 0.0025f + 0.0003f * static_cast<float>((n * 7) % 13);
    scale_bits[n] = to_half_bits(scales[n]);
    for (int k = 0; k < K; ++k) {
      const int value = ((n * 31 + k * 13) % 255) - 127;
      q_nk[static_cast<std::size_t>(n) * K + k] = static_cast<std::int8_t>(value);
    }
  }
  std::vector<std::int8_t> q;
  if (layout == W8BLayout::NK) {
    q = q_nk;
  } else {
    q.resize(static_cast<std::size_t>(K) * N);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) q[static_cast<std::size_t>(k) * N + n] = q_nk[static_cast<std::size_t>(n) * K + k];
    }
  }

  std::vector<float> reference(static_cast<std::size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(from_half_bits(x_bits[static_cast<std::size_t>(m) * K + k])) *
               q_nk[static_cast<std::size_t>(n) * K + k];
      }
      reference[static_cast<std::size_t>(m) * N + n] = static_cast<float>(sum * from_half_bits(scale_bits[n]));
    }
  }

  DeviceBuffer<std::uint16_t> dx;
  DeviceBuffer<std::int8_t> dq;
  DeviceBuffer<std::uint16_t> ds;
  DeviceBuffer<std::uint16_t> dy;
  rwkv_test::copy_host_to_device(x_bits, dx, "alloc test x", "copy test x");
  rwkv_test::copy_host_to_device(q, dq, "alloc test q", "copy test q");
  rwkv_test::copy_host_to_device(scale_bits, ds, "alloc test scales", "copy test scales");
  dy.resize(static_cast<std::size_t>(M) * N, "alloc test y");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dq.p,
                            reinterpret_cast<const half*>(ds.p), layout, reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaGetLastError(), "launch w8a16 linear test");
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync w8a16 linear test");
  const auto actual_bits = rwkv_test::copy_device_buffer(dy, "copy test y");
  for (std::size_t i = 0; i < actual_bits.size(); ++i) {
    const float actual = from_half_bits(actual_bits[i]);
    const float expected = reference[i];
    const float tolerance = 1.0e-2f * std::max(1.0f, std::fabs(expected)) + 1.0e-3f;
    if (std::fabs(actual - expected) > tolerance) {
      throw std::runtime_error("W8A16 linear mismatch at element " + std::to_string(i) +
                               " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
  }

  std::vector<std::uint16_t> shifted = scale_bits;
  std::rotate(shifted.begin(), shifted.begin() + 1, shifted.end());
  rwkv_test::copy_host_to_device(shifted, ds, "resize shifted scales", "copy shifted scales");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dq.p,
                            reinterpret_cast<const half*>(ds.p), layout, reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync shifted w8a16 test");
  const auto shifted_bits = rwkv_test::copy_device_buffer(dy, "copy shifted y");
  bool differs = false;
  for (std::size_t i = 0; i < shifted_bits.size(); ++i) {
    if (std::fabs(from_half_bits(shifted_bits[i]) - reference[i]) > 1.0e-2f) {
      differs = true;
      break;
    }
  }
  TEST_CHECK(differs);
}

void check_i8_pack() {
  constexpr int N = 64;
  constexpr int K = 48;
  std::vector<std::int8_t> source(static_cast<std::size_t>(N) * K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      source[static_cast<std::size_t>(n) * K + k] =
          static_cast<std::int8_t>(((n * 19 + k * 23) % 255) - 127);
    }
  }
  DeviceBuffer<std::int8_t> dsrc;
  DeviceBuffer<std::int8_t> dpacked;
  rwkv_test::copy_host_to_device(source, dsrc, "alloc pack source", "copy pack source");
  dpacked.resize(source.size(), "alloc pack output");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv_test::require_cuda(cudaGetLastError(), "launch i8 pack test");
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync i8 pack test");
  const auto packed = rwkv_test::copy_device_buffer(dpacked, "copy packed output");
  const auto* words = reinterpret_cast<const std::uint32_t*>(packed.data());
  const int words_per_nt = (K / 16) * 8 * 32;
  for (int n = 0; n < N; ++n) {
    const int nt = n / 64;
    const int ns = (n % 64) / 8;
    const int g = n % 8;
    for (int k = 0; k < K; ++k) {
      const int kt = k / 16;
      const int kk = k % 16;
      const int r = (kk % 8) / 2;
      const int byte = (kk >= 8 ? 2 : 0) + (kk & 1);
      const int lane = g * 4 + r;
      const int word_index = ((nt * (K / 16) + kt) * 8 + ns) * 32 + lane;
      const int unpacked = static_cast<int>((words[word_index] >> (8 * byte)) & 0xffu) - 128;
      TEST_CHECK(unpacked == static_cast<int>(source[static_cast<std::size_t>(n) * K + k]));
    }
  }
}

void check_packed_mma_simple() {
  constexpr int M = 16;
  constexpr int N = 64;
  constexpr int K = 64;
  std::vector<std::int8_t> source(static_cast<std::size_t>(N) * K, 1);
  std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(M) * K, to_half_bits(1.0f));
  std::vector<std::uint16_t> scales(N, to_half_bits(1.0f));
  DeviceBuffer<std::int8_t> dsrc;
  DeviceBuffer<std::int8_t> dpacked;
  DeviceBuffer<std::uint16_t> dx;
  DeviceBuffer<std::uint16_t> ds;
  DeviceBuffer<std::uint16_t> dy;
  rwkv_test::copy_host_to_device(source, dsrc, "alloc simple source", "copy simple source");
  dpacked.resize(source.size(), "alloc simple packed");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv_test::copy_host_to_device(x_bits, dx, "alloc simple x", "copy simple x");
  rwkv_test::copy_host_to_device(scales, ds, "alloc simple scales", "copy simple scales");
  dy.resize(static_cast<std::size_t>(M) * N, "alloc simple y");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync simple packed");
  const auto actual = rwkv_test::copy_device_buffer(dy, "copy simple y");
  std::cerr << "simple packed first=" << from_half_bits(actual[0]) << " row1=" << from_half_bits(actual[N]) << "\n";
  for (std::uint16_t bits : actual) {
    TEST_NEAR(from_half_bits(bits), 64.0f, 0.1f);
  }
  std::vector<std::int8_t> raw_kn(static_cast<std::size_t>(K) * N, 1);
  rwkv_test::copy_host_to_device(raw_kn, dsrc, "resize simple raw", "copy simple raw");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dsrc.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::KN,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync simple raw");
  const auto raw_actual = rwkv_test::copy_device_buffer(dy, "copy simple raw y");
  std::cerr << "simple raw first=" << from_half_bits(raw_actual[0]) << " row1=" << from_half_bits(raw_actual[N]) << "\n";
  for (std::uint16_t bits : raw_actual) {
    TEST_NEAR(from_half_bits(bits), 64.0f, 0.1f);
  }
  for (int qvalue : {-127, -1, 1, 127}) {
    std::fill(source.begin(), source.end(), static_cast<std::int8_t>(qvalue));
    rwkv_test::copy_host_to_device(source, dsrc, "resize q probe", "copy q probe");
    rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
    rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                              reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                              reinterpret_cast<half*>(dy.p));
    rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync q probe");
    const auto probe = rwkv_test::copy_device_buffer(dy, "copy q probe");
    std::cerr << "q probe " << qvalue << " -> " << from_half_bits(probe[0]) << "\n";
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) source[static_cast<std::size_t>(n) * K + k] =
        static_cast<std::int8_t>(((n * 19 + k * 23) % 255) - 127);
  }
  rwkv_test::copy_host_to_device(source, dsrc, "copy varied q", "copy varied q");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync varied q");
  const auto varied = rwkv_test::copy_device_buffer(dy, "copy varied output");
  float expected_varied = 0.0f;
  for (int k = 0; k < K; ++k) expected_varied += static_cast<float>(((k * 23) % 255) - 127);
  std::cerr << "varied q -> " << from_half_bits(varied[0]) << " expected=" << expected_varied << "\n";
  std::fill(source.begin(), source.end(), static_cast<std::int8_t>(1));
  for (int k = 0; k < K; ++k) x_bits[static_cast<std::size_t>(8) * K + k] = to_half_bits(static_cast<float>(k + 1));
  rwkv_test::copy_host_to_device(source, dsrc, "copy unit q", "copy unit q");
  rwkv_test::copy_host_to_device(x_bits, dx, "copy varied x", "copy varied x");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync varied x");
  const auto varied_x = rwkv_test::copy_device_buffer(dy, "copy varied x output");
  std::cerr << "varied x ->";
  for (int row = 0; row < M; ++row) std::cerr << " " << from_half_bits(varied_x[row * N]);
  std::cerr << " expected=" << (K * (K + 1) / 2) << "\n";
  std::fill(source.begin(), source.end(), static_cast<std::int8_t>(0));
  for (int n = 0; n < N; ++n) source[static_cast<std::size_t>(n) * K + (K - 1)] = 1;
  std::fill(x_bits.begin(), x_bits.end(), to_half_bits(1.0f));
  x_bits[static_cast<std::size_t>(1) * K + (K - 1)] = to_half_bits(64.0f);
  rwkv_test::copy_host_to_device(source, dsrc, "copy probe k", "copy probe k");
  rwkv_test::copy_host_to_device(x_bits, dx, "copy probe x", "copy probe x");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync probe k");
  const auto probe_k = rwkv_test::copy_device_buffer(dy, "copy probe k output");
  std::cerr << "probe k ->";
  for (int row = 0; row < M; ++row) {
    std::cerr << " row" << row << ":";
    for (int col = 0; col < 16; ++col) std::cerr << " " << from_half_bits(probe_k[row * N + col]);
  }
  std::cerr << "\n";
  std::fill(source.begin(), source.end(), static_cast<std::int8_t>(1));
  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < K; ++k) x_bits[static_cast<std::size_t>(row) * K + k] = to_half_bits(static_cast<float>(row + 1));
  }
  rwkv_test::copy_host_to_device(source, dsrc, "copy row q", "copy row q");
  rwkv_test::copy_host_to_device(x_bits, dx, "copy row x", "copy row x");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync row probe");
  const auto row_probe = rwkv_test::copy_device_buffer(dy, "copy row probe");
  std::cerr << "row probe ->";
  for (int row = 0; row < M; ++row) std::cerr << " " << from_half_bits(row_probe[row * N]);
  std::cerr << "\n";
  std::cerr << "frag debug:";
  for (int i = 0; i < 32; ++i) std::cerr << " " << from_half_bits(row_probe[i]);
  std::cerr << "\n";
  std::cerr << "acc debug:";
  for (int i = 0; i < 64; ++i) std::cerr << " " << from_half_bits(row_probe[1024 + i]);
  std::cerr << "\n";
}

void check_packed_mma_bn128() {
  constexpr int M = 64;
  constexpr int N = 128;
  constexpr int K = 4096;
  std::mt19937 rng(9917);
  std::uniform_int_distribution<int> q_dist(-127, 127);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  std::vector<std::int8_t> source(static_cast<std::size_t>(N) * K);
  std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(M) * K);
  std::vector<std::uint16_t> scales(N);
  for (std::int8_t& value : source) value = static_cast<std::int8_t>(q_dist(rng));
  for (std::uint16_t& value : x_bits) value = to_half_bits(x_dist(rng));
  for (int n = 0; n < N; ++n) scales[n] = to_half_bits(0.0025f + 0.0003f * (n % 13));
  DeviceBuffer<std::int8_t> dsrc;
  DeviceBuffer<std::int8_t> dpacked;
  DeviceBuffer<std::uint16_t> dx;
  DeviceBuffer<std::uint16_t> ds;
  DeviceBuffer<std::uint16_t> dy;
  rwkv_test::copy_host_to_device(source, dsrc, "alloc bn128 source", "copy bn128 source");
  dpacked.resize(source.size(), "alloc bn128 packed");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  rwkv_test::copy_host_to_device(x_bits, dx, "alloc bn128 x", "copy bn128 x");
  rwkv_test::copy_host_to_device(scales, ds, "alloc bn128 scales", "copy bn128 scales");
  dy.resize(static_cast<std::size_t>(M) * N, "alloc bn128 y");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaGetLastError(), "launch bn128 packed");
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync bn128 packed");
  const auto actual = rwkv_test::copy_device_buffer(dy, "copy bn128 y");
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (int k = 0; k < K; ++k) {
        expected += __half2float(*reinterpret_cast<const half*>(&x_bits[static_cast<std::size_t>(m) * K + k])) *
                    static_cast<float>(source[static_cast<std::size_t>(n) * K + k]);
      }
      expected *= from_half_bits(scales[n]);
      TEST_NEAR(from_half_bits(actual[static_cast<std::size_t>(m) * N + n]), expected,
                2.0e-2f * std::max(1.0f, std::fabs(expected)) + 1.0e-3f);
    }
  }
}

void debug_ldmatrix() {
  constexpr int M = 16;
  constexpr int N = 128;
  constexpr int K = 64;
  std::vector<std::int8_t> source(static_cast<std::size_t>(N) * K, 1);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(M) * K);
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k) x[static_cast<std::size_t>(row) * K + k] = to_half_bits(static_cast<float>(row + 1));
  std::vector<std::uint16_t> scales(N, to_half_bits(1.0f));
  DeviceBuffer<std::int8_t> ds, dp;
  DeviceBuffer<std::uint16_t> dx, dscale, dy;
  rwkv_test::copy_host_to_device(source, ds, "debug source", "debug source");
  dp.resize(source.size(), "debug packed");
  rwkv7_v4_i8_pack_launch(nullptr, ds.p, dp.p, N, K);
  rwkv_test::copy_host_to_device(x, dx, "debug x", "debug x");
  rwkv_test::copy_host_to_device(scales, dscale, "debug scale", "debug scale");
  dy.resize(static_cast<std::size_t>(M) * N + 8192, "debug y");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dp.p,
                            reinterpret_cast<const half*>(dscale.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "debug sync");
  const auto out = rwkv_test::copy_device_buffer(dy, "debug output");
  std::cerr << "ldmatrix raw:";
  for (int i = 0; i < 128; ++i) std::cerr << " " << from_half_bits(out[i]);
  std::cerr << "\n";
  std::cerr << "mma raw:";
  for (int i = 0; i < 32; ++i) std::cerr << " " << from_half_bits(out[512 + i]);
  std::cerr << "\n";
  std::cerr << "acc raw:";
  for (int i = 0; i < 256; ++i) std::cerr << " " << from_half_bits(out[4096 + i]);
  std::cerr << "\n";
  std::cerr << "red raw:";
  for (int i = 0; i < 256; ++i) std::cerr << " " << from_half_bits(out[8192 + i]);
  std::cerr << "\n";
}

void check_packed_mma() {
  constexpr int M = 8;
  constexpr int K = 4096;
  constexpr int N = 4096;
  std::mt19937 rng(71391);
  std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
  std::vector<float> x_float(static_cast<std::size_t>(M) * K);
  for (float& value : x_float) value = x_dist(rng);
  const auto x_bits = rwkv_test::to_half_bits(x_float);
  std::vector<std::int8_t> q(static_cast<std::size_t>(N) * K);
  std::vector<std::uint16_t> scale_bits(N);
  for (int n = 0; n < N; ++n) {
    scale_bits[n] = to_half_bits(0.001f + 0.0002f * static_cast<float>(n % 17));
    for (int k = 0; k < K; ++k) {
      q[static_cast<std::size_t>(n) * K + k] =
          static_cast<std::int8_t>(((n * 31 + k * 17) % 255) - 127);
    }
  }
  std::vector<float> reference(static_cast<std::size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(x_float[static_cast<std::size_t>(m) * K + k]) *
               q[static_cast<std::size_t>(n) * K + k];
      }
      reference[static_cast<std::size_t>(m) * N + n] =
          static_cast<float>(sum * from_half_bits(scale_bits[n]));
    }
  }
  DeviceBuffer<std::uint16_t> dx, ds, dy;
  DeviceBuffer<std::int8_t> dsrc, dpacked;
  rwkv_test::copy_host_to_device(x_bits, dx, "alloc packed x", "copy packed x");
  rwkv_test::copy_host_to_device(q, dsrc, "alloc packed source", "copy packed source");
  rwkv_test::copy_host_to_device(scale_bits, ds, "alloc packed scales", "copy packed scales");
  dpacked.resize(q.size(), "alloc packed weights");
  rwkv7_v4_i8_pack_launch(nullptr, dsrc.p, dpacked.p, N, K);
  dy.resize(static_cast<std::size_t>(M) * N, "alloc packed output");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync packed mma");
  const auto actual = rwkv_test::copy_device_buffer(dy, "copy packed output");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float tolerance = 2.0e-2f * std::max(1.0f, std::fabs(reference[i])) + 2.0e-3f;
    TEST_NEAR(from_half_bits(actual[i]), reference[i], tolerance);
  }
  DeviceBuffer<float> workspace;
  workspace.resize(static_cast<std::size_t>(M) * N, "alloc packed split workspace");
  rwkv7_w8a16_linear_launch(nullptr, M, K, N, reinterpret_cast<const half*>(dx.p), dpacked.p,
                            reinterpret_cast<const half*>(ds.p), W8BLayout::PackedNK,
                            reinterpret_cast<half*>(dy.p), workspace.p,
                            workspace.n * sizeof(float), 2);
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync packed split mma");
  const auto split_actual = rwkv_test::copy_device_buffer(dy, "copy packed split output");
  for (std::size_t i = 0; i < split_actual.size(); ++i) {
    const float tolerance = 2.0e-2f * std::max(1.0f, std::fabs(reference[i])) + 2.0e-3f;
    TEST_NEAR(from_half_bits(split_actual[i]), reference[i], tolerance);
  }
}

void check_sparse() {
  constexpr int C = 256;
  constexpr int F = 256;
  std::vector<float> preact(F);
  for (int f = 0; f < F; ++f) preact[f] = f % 3 == 0 ? 0.2f + 0.01f * f : -1.0f;
  std::vector<std::int8_t> q(static_cast<std::size_t>(F) * C);
  std::vector<std::uint16_t> scale_bits(C);
  for (int c = 0; c < C; ++c) {
    scale_bits[c] = to_half_bits(0.01f + 0.0001f * (c % 7));
    for (int f = 0; f < F; ++f) q[static_cast<std::size_t>(f) * C + c] = static_cast<std::int8_t>(((f * 11 + c * 5) % 127) - 63);
  }
  std::vector<float> reference(C, 0.0f);
  for (int c = 0; c < C; ++c) {
    for (int f = 0; f < F; ++f) {
      const float v = std::max(preact[f], 0.0f);
      reference[c] += v * v * static_cast<float>(q[static_cast<std::size_t>(f) * C + c]) * from_half_bits(scale_bits[c]);
    }
  }
  DeviceBuffer<std::uint16_t> dpre;
  DeviceBuffer<std::int8_t> dq;
  DeviceBuffer<std::uint16_t> ds;
  DeviceBuffer<std::uint16_t> dout;
  rwkv_test::copy_host_to_device(rwkv_test::to_half_bits(preact), dpre, "alloc sparse preact", "copy sparse preact");
  rwkv_test::copy_host_to_device(q, dq, "alloc sparse q", "copy sparse q");
  rwkv_test::copy_host_to_device(scale_bits, ds, "alloc sparse scales", "copy sparse scales");
  dout.resize(C, "alloc sparse output");
  rwkv7_cmix_sparse_down_relu_one_i8_launch(nullptr, C, F, reinterpret_cast<const half*>(dpre.p), dq.p,
                                             reinterpret_cast<const half*>(ds.p), reinterpret_cast<half*>(dout.p));
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync sparse test");
  const auto actual = rwkv_test::copy_device_buffer(dout, "copy sparse output");
  float rowmax = 0.0f;
  for (float value : reference) rowmax = std::max(rowmax, std::fabs(value));
  for (int c = 0; c < C; ++c) {
    TEST_NEAR(from_half_bits(actual[c]), reference[c], 2.0e-2f * rowmax + 1.0e-3f);
  }
}

void check_sparse_rows(int B, int T, bool use_t512) {
  constexpr int C = 512;
  constexpr int F = 512;
  const int rows = B * T;
  std::vector<float> preact(static_cast<std::size_t>(rows) * F);
  for (int row = 0; row < rows; ++row) {
    for (int f = 0; f < F; ++f) preact[static_cast<std::size_t>(row) * F + f] = (f + row) % 4 == 0 ? 0.001f * (f + 1) : -0.5f;
  }
  std::vector<std::int8_t> q(static_cast<std::size_t>(F) * C);
  std::vector<std::uint16_t> scale_bits(C);
  for (int c = 0; c < C; ++c) {
    scale_bits[c] = to_half_bits(0.01f + 0.0001f * (c % 7));
    for (int f = 0; f < F; ++f) q[static_cast<std::size_t>(f) * C + c] = static_cast<std::int8_t>(((f * 11 + c * 5) % 127) - 63);
  }
  DeviceBuffer<std::uint16_t> dpre;
  DeviceBuffer<std::int8_t> dq;
  DeviceBuffer<std::uint16_t> ds;
  DeviceBuffer<std::uint16_t> dout;
  rwkv_test::copy_host_to_device(rwkv_test::to_half_bits(preact), dpre, "alloc sparse rows preact", "copy sparse rows preact");
  rwkv_test::copy_host_to_device(q, dq, "alloc sparse rows q", "copy sparse rows q");
  rwkv_test::copy_host_to_device(scale_bits, ds, "alloc sparse rows scales", "copy sparse rows scales");
  dout.resize(static_cast<std::size_t>(rows) * C, "alloc sparse rows output");
  if (use_t512) {
    rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(nullptr, B, T, C, F,
                                                     reinterpret_cast<const half*>(dpre.p), dq.p,
                                                     reinterpret_cast<const half*>(ds.p), reinterpret_cast<half*>(dout.p));
  } else {
    rwkv7_cmix_sparse_down_relu_rows_i8_launch(nullptr, B, T, C, F,
                                                reinterpret_cast<const half*>(dpre.p), dq.p,
                                                reinterpret_cast<const half*>(ds.p), reinterpret_cast<half*>(dout.p));
  }
  rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync sparse rows test");
  const auto actual = rwkv_test::copy_device_buffer(dout, "copy sparse rows output");
  for (int row = 0; row < rows; ++row) {
    for (int c = 0; c < C; ++c) {
      float expected = 0.0f;
      for (int f = 0; f < F; ++f) {
        const float v = std::max(preact[static_cast<std::size_t>(row) * F + f], 0.0f);
        expected += v * v * static_cast<float>(q[static_cast<std::size_t>(f) * C + c]) * from_half_bits(scale_bits[c]);
      }
      TEST_NEAR(from_half_bits(actual[static_cast<std::size_t>(row) * C + c]), expected, 2.0e-2f * std::max(1.0f, std::fabs(expected)) + 1.0e-3f);
    }
  }
}

}  // namespace

int main() {
  if (!rwkv_test::cuda_device_available()) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 0;
  }
  try {
    check_i8_pack();
    check_packed_mma();
    check_packed_mma_bn128();
    for (int M : {1, 2, 4, 8, 12, 16, 32, 64, 100, 128}) {
      check_linear(M, 4096, 4096, W8BLayout::NK);
      if (M == 1 || M == 8 || M >= 9) {
        check_linear(M, 16384, 4096, W8BLayout::KN);
      }
    }
    check_sparse();
    check_sparse_rows(2, 1, false);
    check_sparse_rows(2, 4, true);
    std::cout << "w8a16 kernels ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
