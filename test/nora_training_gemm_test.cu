#include "nora/training_gemm.cuh"
#include "rwkv/bf16_training/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using namespace rwkv_bf16_training;
template <class T> void fill(DeviceBuffer<T> &b, size_t n, std::mt19937 &gen) {
  b.resize(n, "test");
  std::vector<T> h(n);
  std::uniform_real_distribution<float> dist(-.1f, .1f);
  for (auto &x : h)
    x = T(dist(gen));
  gpu_check(cudaMemcpy(b.p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice),
            "fill");
}
template <class T> std::vector<T> read(const DeviceBuffer<T> &b) {
  std::vector<T> h(b.n);
  gpu_check(cudaMemcpy(h.data(), b.p, b.n * sizeof(T), cudaMemcpyDeviceToHost),
            "read");
  return h;
}
template <class F> float time_us(F f) {
  cudaEvent_t a, b;
  gpu_check(cudaEventCreate(&a), "event");
  gpu_check(cudaEventCreate(&b), "event");
  for (int i = 0; i < 3; ++i)
    f();
  gpu_check(cudaEventRecord(a), "event");
  for (int i = 0; i < 20; ++i)
    f();
  gpu_check(cudaEventRecord(b), "event");
  gpu_check(cudaEventSynchronize(b), "event");
  float ms = 0;
  gpu_check(cudaEventElapsedTime(&ms, a, b), "time");
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms * 50;
}
template <bool TR> void projection(int rows, int rank, int width) {
  std::mt19937 gen(1234);
  DeviceBuffer<bf16> x, w;
  DeviceBuffer<float> a, b, c;
  fill(x, size_t(rows) * width, gen);
  fill(w, size_t(rank) * width, gen);
  a.resize(size_t(rows) * rank, "a");
  b.resize(a.n, "b");
  c.resize(a.n, "c");
  auto old = [&] {
    rwkv_nora_detail::gemm<false, TR, false>(nullptr, rows, rank, width, x.p,
                                             w.p, a.p, .7f);
  };
  auto fast = [&] {
    rwkv_bf16_nora::detail::projection<TR>(nullptr, rows, rank, width, x.p, w.p,
                                           b.p, .7f);
  };
  old();
  fast();
  auto aa = read(a), bb = read(b);
  double err = 0, norm = 0;
  for (size_t i = 0; i < aa.size(); ++i) {
    double d = double(aa[i]) - bb[i];
    err += d * d;
    norm += double(aa[i]) * aa[i];
  }
  double rel = std::sqrt(err / std::max(norm, 1e-30));
  if (!std::isfinite(rel) || rel > 3e-6)
    throw std::runtime_error("projection FP32 accuracy");
  const int split = rows / 2;
  if (split)
    rwkv_bf16_nora::detail::projection<TR>(nullptr, split, rank, width, x.p,
                                           w.p, c.p, .7f);
  rwkv_bf16_nora::detail::projection<TR>(nullptr, rows - split, rank, width,
                                         x.p + size_t(split) * width, w.p,
                                         c.p + size_t(split) * rank, .7f);
  auto cc = read(c);
  if (std::memcmp(bb.data(), cc.data(), bb.size() * 4))
    throw std::runtime_error("packing changed projection bits");
  if (rows >= 512 && rank == 16)
    std::cout << "projection TR=" << TR << " rows=" << rows
              << " width=" << width << " rel=" << rel
              << " reference_us=" << time_us(old)
              << " fast_us=" << time_us(fast) << '\n';
}
template <bool TR> void expansion(int rows, int rank, int cols) {
  std::mt19937 gen(5678);
  DeviceBuffer<float> x;
  DeviceBuffer<bf16> w, a, b;
  fill(x, size_t(rows) * rank, gen);
  fill(w, size_t(rank) * cols, gen);
  fill(a, size_t(rows) * cols, gen);
  b.resize(a.n, "b");
  gpu_check(cudaMemcpy(b.p, a.p, a.n * 2, cudaMemcpyDeviceToDevice), "copy");
  auto old = [&] {
    rwkv_nora_detail::gemm<false, TR, true>(nullptr, rows, cols, rank, x.p, w.p,
                                            a.p, -.7f);
  };
  auto fast = [&] {
    rwkv_bf16_nora::detail::expansion<TR>(nullptr, rows, cols, rank, x.p, w.p,
                                          b.p, -.7f);
  };
  old();
  fast();
  auto aa = read(a), bb = read(b);
  if (std::memcmp(aa.data(), bb.data(), aa.size() * 2))
    throw std::runtime_error("expansion differs from reference");
  if (rows >= 512 && rank == 16)
    std::cout << "expansion TR=" << TR << " rows=" << rows << " cols=" << cols
              << " reference_us=" << time_us(old)
              << " fast_us=" << time_us(fast) << '\n';
}
int main(int argc, char **argv) {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
      return 77;
    const bool quick = argc == 2 && std::strcmp(argv[1], "--quick") == 0;
    if (argc > 1 && !quick)
      throw std::invalid_argument("usage: rwkv_nora_training_gemm_test [--quick]");
    for (int rank : {1, 8, 16, 17, 32, 64})
      for (int rows : {1, 17})
        for (int width : {67, 257}) {
          projection<true>(rows, rank, width);
          projection<false>(rows, rank, width);
          expansion<true>(rows, rank, width);
          expansion<false>(rows, rank, width);
        }
    if (!quick)
      for (int rows : {512, 3072})
        for (int width : {2560, 10240}) {
          projection<true>(rows, 16, width);
          projection<false>(rows, 16, width);
          expansion<true>(rows, 16, width);
          expansion<false>(rows, 16, width);
        }
    std::cout << "NoRA optimized projections/expansions passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
