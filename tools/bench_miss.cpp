#include "rwkv/runtime/rwkv_miss.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace rwkv7_miss;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
  try {
    constexpr int K = 4096, N = 4096, R = 16, max_rows = 1024;
    half *x = nullptr, *d = nullptr, *g = nullptr, *y = nullptr, *dx = nullptr;
    float *s = nullptr, *dd = nullptr;
    check(cudaMalloc(&x, size_t(max_rows) * K * 2));
    check(cudaMalloc(&d, N * R * 2));
    check(cudaMalloc(&g, size_t(max_rows) * N * 2));
    check(cudaMalloc(&y, size_t(max_rows) * N * 2));
    check(cudaMalloc(&dx, size_t(max_rows) * K * 2));
    check(cudaMalloc(&s, max_rows * R * 4));
    check(cudaMalloc(&dd, N * R * 4));
    std::vector<half> input(size_t(max_rows) * K, __float2half(.01f)),
        weight(N * R, __float2half(.001f));
    check(
        cudaMemcpy(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    check(
        cudaMemcpy(g, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d, weight.data(), weight.size() * 2,
                     cudaMemcpyHostToDevice));
    check(cudaMemset(y, 0, input.size() * 2));
    check(cudaMemset(dx, 0, input.size() * 2));
    check(cudaMemset(dd, 0, N * R * 4));
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    Linear w{K, N, R, 1, d, dd};
    std::cout << "rows,forward_us,backward_us\n";
    for (int rows : {1, 64, 256, 1024}) {
      auto measure = [&](bool bwd) {
        for (int i = 0; i < 10; ++i)
          if (bwd)
            backward(nullptr, rows, w, s, g, dx);
          else
            forward(nullptr, rows, w, x, y, rows == 1 ? nullptr : s);
        check(cudaEventRecord(begin, nullptr));
        for (int i = 0; i < 100; ++i)
          if (bwd)
            backward(nullptr, rows, w, s, g, dx);
          else
            forward(nullptr, rows, w, x, y, rows == 1 ? nullptr : s);
        check(cudaEventRecord(end, nullptr));
        check(cudaEventSynchronize(end));
        float ms;
        check(cudaEventElapsedTime(&ms, begin, end));
        return ms * 10;
      };
      forward(nullptr, rows, w, x, y, s);
      float f = measure(false), b = measure(true);
      std::cout << rows << ',' << std::fixed << std::setprecision(3) << f << ','
                << b << '\n';
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    for (void *p : {(void *)x, (void *)d, (void *)g, (void *)y, (void *)dx,
                    (void *)s, (void *)dd})
      cudaFree(p);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
