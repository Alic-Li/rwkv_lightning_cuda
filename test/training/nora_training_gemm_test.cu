#include "nora/training_gemm.cuh"
#include "rwkv/bf16_training/nora.hpp"
#include "rwkv/bf16_training/ops.hpp"
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
#ifdef RWKV_USE_HIP
void fused_forward_check() {
  constexpr int rows = 513, in = 4103, out = 4099, rank = 16;
  std::mt19937 gen(2718);
  DeviceBuffer<bf16> x, a, d, ref, got;
  fill(x, size_t(rows) * in, gen);
  fill(a, size_t(rank) * in, gen);
  fill(d, size_t(out) * rank, gen);
  fill(ref, size_t(rows) * out, gen);
  got.resize(ref.n, "fused result");
  gpu_check(cudaMemcpy(got.p, ref.p, ref.n * sizeof(bf16), cudaMemcpyDeviceToDevice),
            "copy fused input");
  rwkv_bf16_miss::Linear w{in, out, rank, .7f, d.p, nullptr, a.p, nullptr};
  auto old = [&] {
    rwkv_nora_detail::fused<<<dim3((out + 255) / 256, rows), 256, rank * sizeof(float)>>>(
        in, out, rank, w.scale, a.p, d.p, x.p, ref.p);
  };
  auto fast = [&] {
    rwkv_bf16_nora::forward(nullptr, rows, w, x.p, got.p, nullptr);
  };
  old();
  fast();
  auto expected = read(ref), actual = read(got);
  if (std::memcmp(expected.data(), actual.data(), ref.n * sizeof(bf16)))
    throw std::runtime_error("fused NoRA forward differs from reference");
  std::cout << "fused reference_us=" << time_us(old)
            << " fast_us=" << time_us(fast) << '\n';
}
#endif
void grad_weight_check(int rows, int rank, int width) {
  std::mt19937 gen(9012);
  DeviceBuffer<bf16> g, x;
  DeviceBuffer<float> s, da0, da1, db0, db1;
  fill(g, size_t(rows) * width, gen);
  fill(x, size_t(rows) * width, gen);
  fill(s, size_t(rows) * rank, gen);
  fill(da0, size_t(rank) * width, gen);
  fill(db0, size_t(width) * rank, gen);
  da1.resize(da0.n, "da1");
  db1.resize(db0.n, "db1");
  gpu_check(cudaMemcpy(da1.p, da0.p, da0.n * 4, cudaMemcpyDeviceToDevice), "copy da");
  gpu_check(cudaMemcpy(db1.p, db0.p, db0.n * 4, cudaMemcpyDeviceToDevice), "copy db");
  rwkv_nora_detail::gemm<true, false, true>(nullptr, width, rank, rows, g.p,
                                            s.p, db0.p, .7f);
  rwkv_bf16_nora::detail::grad_weight(nullptr, width, rank, rows, g.p, s.p,
                                      db1.p, .7f);
  rwkv_nora_detail::gemm<true, false, true>(nullptr, rank, width, rows, s.p,
                                            x.p, da0.p, 1.0f);
  rwkv_bf16_nora::detail::grad_weight(nullptr, rank, width, rows, s.p, x.p,
                                      da1.p, 1.0f);
  auto h_db0 = read(db0), h_db1 = read(db1);
  auto h_da0 = read(da0), h_da1 = read(da1);
  if (std::memcmp(h_db0.data(), h_db1.data(), h_db0.size() * 4))
    throw std::runtime_error("grad_weight dB differs from reference");
  if (std::memcmp(h_da0.data(), h_da1.data(), h_da0.size() * 4))
    throw std::runtime_error("grad_weight dA differs from reference");
}
void tensor_core_backward_check(int rows, int rank, int in, int out) {
  std::mt19937 gen(4242);
  DeviceBuffer<bf16> x, g, a, d, y0, y1, dx0, dx1;
  DeviceBuffer<float> s0, s1, ds0, ds1, db0, db1, da0, da1;
  fill(x, size_t(rows) * in, gen);
  fill(g, size_t(rows) * out, gen);
  fill(a, size_t(rank) * in, gen);
  fill(d, size_t(out) * rank, gen);
  fill(y0, size_t(rows) * out, gen);
  fill(dx0, size_t(rows) * in, gen);
  fill(db0, size_t(out) * rank, gen);
  fill(da0, size_t(rank) * in, gen);
  y1.resize(y0.n, "y1");
  dx1.resize(dx0.n, "dx1");
  db1.resize(db0.n, "db1");
  da1.resize(da0.n, "da1");
  s0.resize(size_t(rows) * rank, "s0");
  s1.resize(s0.n, "s1");
  ds0.resize(s0.n, "ds0");
  ds1.resize(s0.n, "ds1");
  gpu_check(cudaMemcpy(y1.p, y0.p, y0.n * 2, cudaMemcpyDeviceToDevice), "cy");
  gpu_check(cudaMemcpy(dx1.p, dx0.p, dx0.n * 2, cudaMemcpyDeviceToDevice), "cdx");
  gpu_check(cudaMemcpy(db1.p, db0.p, db0.n * 4, cudaMemcpyDeviceToDevice), "cdb");
  gpu_check(cudaMemcpy(da1.p, da0.p, da0.n * 4, cudaMemcpyDeviceToDevice), "cda");
  rwkv_bf16_miss::Linear w0{in, out, rank, 1.0f, d.p, db0.p, a.p, da0.p};
  rwkv_bf16_miss::Linear w1{in, out, rank, 1.0f, d.p, db1.p, a.p, da1.p};
  set_consistent_forward(true, false);
  rwkv_bf16_nora::forward(nullptr, rows, w0, x.p, y0.p, s0.p);
  rwkv_bf16_nora::backward(nullptr, rows, w0, x.p, s0.p, g.p, dx0.p, ds0.p);
  set_consistent_forward(false, false);
  rwkv_bf16_nora::forward(nullptr, rows, w1, x.p, y1.p, s1.p);
  rwkv_bf16_nora::backward(nullptr, rows, w1, x.p, s1.p, g.p, dx1.p, ds1.p);
  auto check_rel = [](const std::vector<float> &ref, const std::vector<float> &got,
                      double tol, const char *what) {
    double err = 0, norm = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double diff = double(ref[i]) - got[i];
      err += diff * diff;
      norm += double(ref[i]) * ref[i];
    }
    double rel = std::sqrt(err / std::max(norm, 1e-30));
    if (!std::isfinite(rel) || rel > tol)
      throw std::runtime_error(std::string(what) + " rel=" + std::to_string(rel));
  };
  check_rel(read(db0), read(db1), 2e-4, "tensor_core dB");
  check_rel(read(da0), read(da1), 2e-4, "tensor_core dA");
  check_rel(read(ds0), read(ds1), 2e-4, "tensor_core ds");
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
          grad_weight_check(rows, rank, width);
        }
    if (!quick)
      for (int rows : {512, 3072})
        for (int width : {2560, 10240}) {
          projection<true>(rows, 16, width);
          projection<false>(rows, 16, width);
          expansion<true>(rows, 16, width);
          expansion<false>(rows, 16, width);
          grad_weight_check(rows, 16, width);
        }
#ifdef RWKV_USE_HIP
    fused_forward_check();
#endif
    tensor_core_backward_check(257, 16, 256, 512);
    tensor_core_backward_check(512, 32, 4096, 4096);
    std::cout << "NoRA optimized projections/expansions passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
