#include "rwkv_asym_a16.cuh"
#include "rwkv_w8a16.cuh"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
  T *p = nullptr;
  explicit Buffer(std::size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
};
} // namespace
int main() {
  try {
    std::cout << "bits,M,K,N,milliseconds\n";
    for (int K : {4096, 16384}) {
      const int N = 4096;
      const std::size_t groups = std::size_t(N) * (K / 32);
      Buffer<unsigned char> q(groups * 32);
      Buffer<half2> sm(groups);
      Buffer<half> s(groups), x(std::size_t(128) * K), y(std::size_t(128) * N);
      Buffer<float> ws(std::size_t(128) * N * 32);
      check(cudaMemset(q.p, 0x55, groups * 32));
      std::vector<half2> hm(groups, __floats2half2_rn(.01, -.1));
      std::vector<half> hs(groups, __float2half(.01));
      std::vector<half> hx(std::size_t(128) * K, __float2half(.05));
      check(cudaMemcpy(sm.p, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(s.p, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
      check(cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
      check(cudaDeviceSynchronize());
      for (int M : {1, 4, 16, 64, 128})
        for (int bits : {8, 4, 6}) {
          auto call = [&]() {
            if (bits != 8)
              rwkv7_asym_linear_launch(nullptr, M, K, N, x.p, q.p, sm.p, y.p,
                                       bits, ws.p,
                                       std::size_t(128) * N * 32 * 4);
            else
              rwkv7_w8a16_linear_launch(
                  nullptr, M, K, N, x.p,
                  reinterpret_cast<const std::int8_t *>(q.p), s.p,
                  W8BLayout::NK, y.p, ws.p, std::size_t(128) * N * 32 * 4);
          };
          for (int i = 0; i < 5; ++i)
            call();
          check(cudaDeviceSynchronize());
          cudaEvent_t a, b;
          check(cudaEventCreate(&a));
          check(cudaEventCreate(&b));
          check(cudaEventRecord(a));
          for (int i = 0; i < 50; ++i)
            call();
          check(cudaEventRecord(b));
          check(cudaEventSynchronize(b));
          float ms;
          check(cudaEventElapsedTime(&ms, a, b));
          std::cout << bits << ',' << M << ',' << K << ',' << N << ','
                    << ms / 50 << '\n';
          check(cudaEventDestroy(a));
          check(cudaEventDestroy(b));
        }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
