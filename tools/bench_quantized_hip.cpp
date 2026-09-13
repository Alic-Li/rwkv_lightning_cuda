#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"
#include "rwkv_w4a16.cuh"
#include "rwkv_w8a16.cuh"
#include <iostream>
#include <random>
#include <vector>

using rwkv7_fast_v4::check_hip;
using rwkv7_fast_v4::DeviceBuffer;
int main() {
  hipDeviceProp_t device{};
  check_hip(hipGetDeviceProperties(&device, 0), "device");
  std::cout << "# " << device.name << ' ' << device.gcnArchName
            << "; warmup=3 iterations=20 seed=42\n";
  std::cout << "bits,M,K,N,microseconds\n";
  for (int K : {4096, 16384})
    for (int M : {1, 4, 16, 64}) {
      const int N = 4096;
      std::mt19937 gen(42);
      std::vector<half> x(std::size_t(M) * K),
          s(std::size_t(N) * ((K + 127) / 128));
      std::vector<unsigned char> q(std::size_t(N) * K);
      for (auto &v : x)
        v = __float2half(float(int(gen() % 201) - 100) / 100);
      for (auto &v : q)
        v = gen() % 256;
      for (auto &v : s)
        v = __float2half(0.01f);
      DeviceBuffer<half> dx, ds, dy;
      DeviceBuffer<float> workspace;
      const auto workspace_bytes = rwkv7_w4a16_workspace_bytes(M, N, 16);
      workspace.resize(workspace_bytes / sizeof(float), "split workspace");
      DeviceBuffer<unsigned char> dq;
      dx.resize(x.size(), "x");
      ds.resize(s.size(), "s");
      dq.resize(q.size(), "q");
      dy.resize(std::size_t(M) * N, "y");
      check_hip(hipMemcpy(dx.p, x.data(), x.size() * 2, hipMemcpyHostToDevice),
                "copy x");
      check_hip(hipMemcpy(ds.p, s.data(), s.size() * 2, hipMemcpyHostToDevice),
                "copy s");
      check_hip(hipMemcpy(dq.p, q.data(), q.size(), hipMemcpyHostToDevice),
                "copy q");
      hipEvent_t start, stop;
      check_hip(hipEventCreate(&start), "event");
      check_hip(hipEventCreate(&stop), "event");
      for (int bits : {4, 8}) {
        auto launch = [&]() {
          if (bits == 4)
            rwkv7_w4a16_linear_launch(nullptr, M, K, N, dx.p, dq.p, ds.p, dy.p,
                                      128, workspace.p, workspace_bytes);
          else
            rwkv7_w8a16_linear_launch(
                nullptr, M, K, N, dx.p,
                reinterpret_cast<const std::int8_t *>(dq.p), ds.p,
                W8BLayout::NK, dy.p, workspace.p, workspace_bytes);
        };
        for (int i = 0; i < 3; ++i)
          launch();
        check_hip(hipEventRecord(start), "start");
        for (int i = 0; i < 20; ++i)
          launch();
        check_hip(hipEventRecord(stop), "stop");
        check_hip(hipEventSynchronize(stop), "sync");
        float ms = 0;
        check_hip(hipEventElapsedTime(&ms, start, stop), "elapsed");
        std::cout << bits << ',' << M << ',' << K << ',' << N << ','
                  << ms * 1000 / 20 << '\n';
      }
      check_hip(hipEventDestroy(start), "event cleanup");
      check_hip(hipEventDestroy(stop), "event cleanup");
    }
}
