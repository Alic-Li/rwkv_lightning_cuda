#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "rwkv_w4a16.cuh"

namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool b, const char* msg) {
  if (!b) throw std::runtime_error(msg);
}
template <class T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(std::size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
int unpack(unsigned char b, int k) {
  const int v = (b >> ((k & 1) * 4)) & 15;
  return (v ^ 8) - 8;
}
void run(int M, int K, int N, int G, bool raw = false) {
  std::mt19937 rng(M * 19 + K + N);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::vector<half> w(std::size_t(N) * K), x(std::size_t(M) * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      w[std::size_t(n) * K + k] = __float2half(n == 0 ? 0 : n == 1 ? 0x1p-24f : dist(rng) * (1 + k / G % 7));
  for (auto& v : x) v = __float2half(dist(rng));
  const std::size_t bytes = rwkv7_w4a16_weight_bytes(N, K);
  const int groups = (K + G - 1) / G;
  Buffer<half> dw(w.size()), dx(x.size()), ds(std::size_t(N) * groups), dy(std::size_t(M) * N);
  Buffer<unsigned char> dq(bytes);
  check(cudaMemcpy(dw.p, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dx.p, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice));
  cudaStream_t streams[2];
  for (auto& s : streams) check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  rwkv7_w4a16_quantize_launch(streams[0], dw.p, dq.p, ds.p, N, K, G);
  check(cudaStreamSynchronize(streams[0]));
  std::vector<unsigned char> q(bytes);
  std::vector<half> scales(std::size_t(N) * groups);
  check(cudaMemcpy(q.data(), dq.p, bytes, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(scales.data(), ds.p, scales.size() * sizeof(half), cudaMemcpyDeviceToHost));
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < groups; ++g) {
      float amax = 0;
      for (int k = g * G; k < std::min(K, (g + 1) * G); ++k)
        amax = std::max(amax, std::fabs(__half2float(w[std::size_t(n) * K + k])));
      const float expected_s = __half2float(__float2half(amax == 0 ? 1 : std::max(amax / 7, 0x1p-24f)));
      require(__half2float(scales[std::size_t(n) * groups + g]) == expected_s, "quantization scale mismatch");
      for (int k = g * G; k < std::min(K, (g + 1) * G); ++k) {
        const int expected_q = std::max(
            -7, std::min(7, int(std::nearbyint(__half2float(w[std::size_t(n) * K + k]) / expected_s))));
        if (unpack(q[std::size_t(n) * ((K + 1) / 2) + k / 2], k) != expected_q) {
          std::cerr << "quant M/K/N/G=" << M << '/' << K << '/' << N << '/' << G << " n/k=" << n << '/' << k
                    << " w=" << __half2float(w[std::size_t(n) * K + k]) << " s=" << expected_s
                    << " got=" << unpack(q[std::size_t(n) * ((K + 1) / 2) + k / 2], k)
                    << " expected=" << expected_q << '\n';
          throw std::runtime_error("quantization nibble mismatch");
        }
      }
      if (K & 1) require((q[std::size_t(n + 1) * ((K + 1) / 2) - 1] >> 4) == 0, "odd K padding mismatch");
    }
  if (raw) {  // All signed nibble values, including externally packed -8.
    for (std::size_t i = 0; i < bytes; ++i) q[i] = (i % 16) | ((15 - i % 16) << 4);
    check(cudaMemcpy(dq.p, q.data(), bytes, cudaMemcpyHostToDevice));
  }
  std::vector<double> ref(std::size_t(M) * N), abs_sum(ref.size());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double sum = 0, magnitude = 0;
      for (int k = 0; k < K; ++k) {
        const double v = double(__half2float(x[std::size_t(m) * K + k])) *
                         unpack(q[std::size_t(n) * ((K + 1) / 2) + k / 2], k) *
                         __half2float(scales[std::size_t(n) * groups + k / G]);
        sum += v;
        magnitude += std::fabs(v);
      }
      ref[std::size_t(m) * N + n] = sum;
      abs_sum[std::size_t(m) * N + n] = magnitude;
    }
  auto verify = [&](half* output) {
    std::vector<half> got(ref.size());
    check(cudaMemcpy(got.data(), output, got.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i) {
      // Account for FP16 staged weights, FP32 reduction and final FP16 rounding.
      const double tolerance = 0.0006 * abs_sum[i] + 0.0006 * std::fabs(ref[i]) + 0.002;
      if (!(std::fabs(__half2float(got[i]) - ref[i]) <= tolerance)) {
        std::cerr << "M/K/N/G=" << M << '/' << K << '/' << N << '/' << G << " index=" << i
                  << " got=" << __half2float(got[i]) << " ref=" << ref[i] << '\n';
        throw std::runtime_error("W4 linear reference mismatch");
      }
    }
  };
  const int splits = std::min(3, (K + (M <= 4 ? G : 64) - 1) / (M <= 4 ? G : 64));
  const auto wsbytes = rwkv7_w4a16_workspace_bytes(M, N, 32);
  Buffer<float> ws0(std::max<std::size_t>(1, wsbytes / 4)), ws1(std::max<std::size_t>(1, wsbytes / 4));
  Buffer<half> dy1(ref.size());
  for (int split : {1, splits, 0}) {
    rwkv7_w4a16_linear_launch(streams[0], M, K, N, dx.p, dq.p, ds.p, dy.p, G, ws0.p, wsbytes, split);
    rwkv7_w4a16_linear_launch(streams[1], M, K, N, dx.p, dq.p, ds.p, dy1.p, G, ws1.p, wsbytes, split);
    for (auto s : streams) check(cudaStreamSynchronize(s));
    verify(dy.p);
    verify(dy1.p);
  }
  rwkv7_w4a16_linear_launch(streams[0], M, K, N, dx.p, dq.p, ds.p, dy.p, G);
  check(cudaStreamSynchronize(streams[0]));
  verify(dy.p);
  // Capture both split compute and deterministic reduction, replay twice.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal));
  rwkv7_w4a16_linear_launch(streams[0], M, K, N, dx.p, dq.p, ds.p, dy.p, G, ws0.p, wsbytes, splits);
  check(cudaStreamEndCapture(streams[0], &graph));
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(exec, streams[0]));
  check(cudaStreamSynchronize(streams[0]));
  verify(dy.p);
  check(cudaGraphExecDestroy(exec));
  check(cudaGraphDestroy(graph));
  if (splits > 1) {
    bool threw = false;
    try {
      rwkv7_w4a16_linear_launch(nullptr, M, K, N, dx.p, dq.p, ds.p, dy.p, G, ws0.p,
                                rwkv7_w4a16_workspace_bytes(M, N, splits) - 1, splits);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "undersized workspace accepted");
  }
  for (auto s : streams) check(cudaStreamDestroy(s));
}
}  // namespace
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
    std::cout << "SKIP: no CUDA device\n";
    return 77;
  }
  try {
    require(rwkv7_w4a16_weight_bytes(3, 5) == 9, "weight size");
    require(rwkv7_w4a16_scale_count(3, 129, 128) == 6, "scale size");
    bool threw = false;
    try {
      rwkv7_w4a16_scale_count(1, 128, 64);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "invalid group accepted");
    rwkv7_w4a16_linear_launch(nullptr, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
    for (int g : {32, 128}) {
      for (int m : {1, 2, 3, 4, 5, 8, 16, 17, 32, 64, 100, 128}) run(m, 257, 67, g, m % 2 == 0);
      for (int k : {1, 2, 31, 32, 33, 63, 64, 127, 128, 129}) {
        run(1, k, 3, g);
        run(17, k, 65, g, true);
      }
      for (int m : {1, 2, 3, 4, 8, 16, 32, 64, 128}) run(m, 4096, 256, g);
      for (int m : {31, 33, 63, 65, 129, 256, 513, 1024}) run(m, 320, 128, g, true);
      run(17, 64, 64, g, true);
      run(8, 16384, 128, g);
    }
    std::cout << "W4A16 quantize/GEMV/Tensor Core/split-K/multi-stream/graph tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
