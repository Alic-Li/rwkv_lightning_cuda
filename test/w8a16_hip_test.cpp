#include "rwkv_w4a16.cuh"
#include "rwkv_w8a16.cuh"
#include "test_common.hpp"
#include <iostream>
#include <random>

using rwkv7_fast_v4::DeviceBuffer;
using rwkv_test::require_cuda;
namespace {
void run(int M, int K, int N, bool kn) {
  std::mt19937 gen(42 + M + K + N);
  std::vector<std::int8_t> q(std::size_t(N) * K);
  std::vector<half> x(std::size_t(M) * K), scales(N);
  for (auto &v : q)
    v = static_cast<std::int8_t>(gen() % 256 - 128);
  for (auto &v : x)
    v = __float2half(float(int(gen() % 201) - 100) / 100);
  for (int n = 0; n < N; ++n)
    scales[n] = __float2half(n == 0   ? 0
                             : n == 1 ? 0x1p-24f
                                      : float(n % 7 + 1) / 1024);
  std::vector<double> ref(std::size_t(M) * N), mag(ref.size());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const double v =
            double(__half2float(x[std::size_t(m) * K + k])) *
            q[kn ? std::size_t(k) * N + n : std::size_t(n) * K + k] *
            __half2float(scales[n]);
        ref[std::size_t(m) * N + n] += v;
        mag[std::size_t(m) * N + n] += std::abs(v);
      }
    }
  DeviceBuffer<std::int8_t> dq;
  DeviceBuffer<half> dx, ds, dy[2];
  DeviceBuffer<float> ws[2];
  rwkv_test::copy_host_to_device(q, dq, "alloc q", "copy q");
  rwkv_test::copy_host_to_device(x, dx, "alloc x", "copy x");
  rwkv_test::copy_host_to_device(scales, ds, "alloc s", "copy s");
  hipStream_t streams[2];
  const auto bytes = rwkv7_w4a16_workspace_bytes(M, N, 3);
  for (int i = 0; i < 2; ++i) {
    require_cuda(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking),
                 "create stream");
    dy[i].resize(ref.size(), "alloc y");
    ws[i].resize(bytes / 4, "alloc workspace");
  }
  auto launch = [&](int i, int split) {
    rwkv7_w8a16_linear_launch(streams[i], M, K, N, dx.p, dq.p, ds.p,
                              kn ? W8BLayout::KN : W8BLayout::NK, dy[i].p,
                              ws[i].p, bytes, split);
  };
  auto verify = [&](int i) {
    require_cuda(hipStreamSynchronize(streams[i]), "sync result");
    const auto y = rwkv_test::copy_device_buffer(dy[i], "copy y");
    for (std::size_t j = 0; j < y.size(); ++j) {
      const double tol = 0.0006 * mag[j] + 0.0006 * std::abs(ref[j]) + 0.002;
      if (!(std::abs(__half2float(y[j]) - ref[j]) <= tol))
        throw std::runtime_error(
            "W8 reference mismatch M/K/N=" + std::to_string(M) + "/" +
            std::to_string(K) + "/" + std::to_string(N));
    }
  };
  for (int split : {0, 1, K >= 384 ? 3 : 1}) {
    launch(0, split);
    launch(1, split);
    verify(0);
    verify(1);
  }
  hipGraph_t graph;
  hipGraphExec_t exec;
  require_cuda(
      hipStreamBeginCapture(streams[0], hipStreamCaptureModeThreadLocal),
      "capture");
  launch(0, K >= 384 ? 3 : 1);
  require_cuda(hipStreamEndCapture(streams[0], &graph), "end capture");
  require_cuda(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
               "instantiate");
  for (int i = 0; i < 2; ++i)
    require_cuda(hipGraphLaunch(exec, streams[0]), "replay");
  verify(0);
  require_cuda(hipGraphExecDestroy(exec), "destroy exec");
  require_cuda(hipGraphDestroy(graph), "destroy graph");
  for (auto s : streams)
    require_cuda(hipStreamDestroy(s), "destroy stream");
}
} // namespace
int main() {
  if (!rwkv_test::cuda_device_available())
    return 77;
  for (bool kn : {false, true}) {
    for (int m : {1, 2, 4, 5, 8, 16, 17, 32, 65, 128})
      run(m, 257, 67, kn);
    for (int k : {1, 31, 64, 129, 512, 4096}) {
      run(1, k, 65, kn);
      run(17, k, 64, kn);
    }
  }
  bool rejected = false;
  try {
    rwkv7_w8a16_linear_launch(nullptr, 1, 64, 64, nullptr, nullptr, nullptr,
                              W8BLayout::PackedNK, nullptr);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  TEST_CHECK(rejected);
  std::cout << "W8 HIP CPU-reference NK/KN, tails, split-K, streams and graph "
               "passed\n";
}
