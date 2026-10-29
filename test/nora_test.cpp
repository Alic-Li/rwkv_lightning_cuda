#include "rwkv/bf16_training/nora.hpp"
#include "rwkv/runtime/rwkv_nora.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using rwkv_bf16_training::DeviceBuffer;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> void upload(DeviceBuffer<T> &d, const std::vector<T> &h) {
  d.resize(h.size(), "NoRA test");
  check(
      cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> read(const DeviceBuffer<T> &d) {
  std::vector<T> h(d.n);
  check(
      cudaMemcpy(h.data(), d.p, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
void near(double a, double b, double tol) {
  if (!std::isfinite(a) || std::abs(a - b) > tol)
    throw std::runtime_error("NoRA mismatch: " + std::to_string(a) + " vs " +
                             std::to_string(b));
}
void numerical(int rows, int in, int out, int rank) {
  std::vector<bf16> x(rows * in), a(rank * in), b(out * rank), g(rows * out),
      y(rows * out), dx(rows * in);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = to_bf16(.1f * std::sin(i * .17f));
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = to_bf16(.1f * std::cos(i * .31f));
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = to_bf16(.1f * std::sin(i * .21f));
  for (size_t i = 0; i < g.size(); ++i)
    g[i] = to_bf16(.1f * std::cos(i * .11f));
  for (auto &v : y)
    v = to_bf16(.01f);
  for (auto &v : dx)
    v = to_bf16(.02f);
  DeviceBuffer<bf16> X, A, B, G, Y, DX;
  DeviceBuffer<float> S, DS, DA, DB;
  upload(X, x);
  upload(A, a);
  upload(B, b);
  upload(G, g);
  upload(Y, y);
  upload(DX, dx);
  S.resize(rows * rank, "S");
  DS.resize(rows * rank, "DS");
  upload(DA, std::vector<float>(a.size(), .03f));
  upload(DB, std::vector<float>(b.size(), .04f));
  rwkv_bf16_miss::Linear w{in, out, rank, 1, B.p, DB.p, A.p, DA.p};
  rwkv_bf16_miss::forward(nullptr, rows, w, X.p, Y.p, S.p);
  rwkv_bf16_nora::backward(nullptr, rows, w, X.p, S.p, G.p, DX.p, DS.p);
  auto yy = read(Y), ddx = read(DX);
  auto da = read(DA), db = read(DB);
  // Independent double-precision reference of the full BA product and its
  // derivatives; includes nonzero base dX and accumulated parameter gradients.
  std::vector<double> s(rows * rank), ds(rows * rank);
  for (int t = 0; t < rows; ++t)
    for (int j = 0; j < rank; ++j) {
      for (int k = 0; k < in; ++k)
        s[t * rank + j] +=
            double(to_float(x[t * in + k])) * to_float(a[j * in + k]);
      for (int o = 0; o < out; ++o)
        ds[t * rank + j] +=
            double(to_float(g[t * out + o])) * to_float(b[o * rank + j]);
    }
  for (int t = 0; t < rows; ++t) {
    for (int o = 0; o < out; ++o) {
      double v = to_float(y[t * out + o]);
      for (int j = 0; j < rank; ++j)
        v += s[t * rank + j] * to_float(b[o * rank + j]);
      near(to_float(yy[t * out + o]), v, 1e-5 + std::abs(v) * .004);
    }
    for (int k = 0; k < in; ++k) {
      double v = to_float(dx[t * in + k]);
      for (int j = 0; j < rank; ++j)
        v += ds[t * rank + j] * to_float(a[j * in + k]);
      near(to_float(ddx[t * in + k]), v, 1e-5 + std::abs(v) * .004);
    }
  }
  for (int j = 0; j < rank; ++j) {
    for (int k = 0; k < in; ++k) {
      double v = .03f;
      for (int t = 0; t < rows; ++t)
        v += ds[t * rank + j] * to_float(x[t * in + k]);
      near(da[j * in + k], v, 2e-6);
    }
    for (int o = 0; o < out; ++o) {
      double v = .04f;
      for (int t = 0; t < rows; ++t)
        v += s[t * rank + j] * to_float(g[t * out + o]);
      near(db[o * rank + j], v, 2e-6);
    }
  }
  // Same values fit exactly in FP16; verify separate and scratch-free paths.
  auto halfs = [](const auto &v) {
    std::vector<half> h;
    for (auto e : v)
      h.push_back(__float2half(to_float(e)));
    return h;
  };
  DeviceBuffer<half> HX, HA, HB, HY, HF;
  upload(HX, halfs(x));
  upload(HA, halfs(a));
  upload(HB, halfs(b));
  upload(HY, halfs(y));
  upload(HF, halfs(y));
  rwkv7_miss::Linear hw{in, out, rank, 1, HB.p, HA.p};
  rwkv7_miss::forward(nullptr, rows, hw, HX.p, HY.p, S.p);
  rwkv7_miss::forward(nullptr, rows, hw, HX.p, HF.p);
  auto hy = read(HY), hf = read(HF);
  for (size_t i = 0; i < hy.size(); ++i) {
    double v = to_float(y[i]);
    for (int j = 0; j < rank; ++j)
      v += s[(i / out) * rank + j] * to_float(b[(i % out) * rank + j]);
    near(__half2float(hy[i]), v, 1e-6 + std::abs(v) * .0006);
    near(__half2float(hf[i]), v, 1e-6 + std::abs(v) * .0006);
  }
}
int main() {
  try {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n)
      return 77;
    numerical(1, 67, 35, 1);
    numerical(17, 67, 35, 16);
    numerical(33, 19, 37, 32);
    numerical(3, 7, 5, 1024);
    std::cout << "NoRA forward/dA/dB/dX and FP16 inference passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
