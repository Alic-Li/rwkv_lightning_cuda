#include "rwkv7_fast_v4_common.hpp"
#include "rwkv_state_tuning.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using rwkv7_fast_v4::DeviceBuffer;
using namespace rwkv7_state_tuning;

void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> void upload(DeviceBuffer<T> &d, const std::vector<T> &h) {
  d.resize(h.size(), "test buffer");
  check(
      cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> download(const DeviceBuffer<T> &d) {
  std::vector<T> h(d.n);
  check(
      cudaMemcpy(h.data(), d.p, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
void near(double a, double b, double tolerance, const char *label) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
    throw std::runtime_error(std::string(label) + ": " + std::to_string(a) +
                             " vs " + std::to_string(b));
}

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
      return 77;
    constexpr int N = 64, T = 5, E = T * N, S = N * N;
    std::array<std::vector<double>, 6> x;
    std::array<DeviceBuffer<half>, 6> gpu, grad;
    for (int p = 0; p < 6; ++p) {
      x[p].resize(E);
      std::vector<half> h(E);
      for (int i = 0; i < E; ++i) {
        h[i] = __float2half(0.08f * std::sin(i * 0.17f + p));
        x[p][i] = __half2float(h[i]);
      }
      upload(gpu[p], h);
      grad[p].resize(E, "sequence gradient");
    }
    std::vector<float> initial(S), terminal(S);
    std::vector<half> upstream(E);
    for (int i = 0; i < S; ++i) {
      initial[i] = 0.03f * std::sin(i * 0.11f);
      terminal[i] = 0.01f * std::cos(i * 0.07f);
    }
    for (int i = 0; i < E; ++i)
      upstream[i] = __float2half(0.1f * std::cos(i * 0.13f));
    DeviceBuffer<float> s0, st, ds0, dst, tape;
    DeviceBuffer<half> y, dy;
    upload(s0, initial);
    upload(dst, terminal);
    upload(dy, upstream);
    st.resize(S, "final state");
    ds0.resize(S, "initial gradient");
    tape.resize(T * S, "tape");
    y.resize(E, "output");
    auto objective = [&]() {
      std::vector<double> state(initial.begin(), initial.end());
      double loss = 0;
      for (int t = 0; t < T; ++t)
        for (int v = 0; v < N; ++v) {
          double sa = 0, out = 0;
          for (int k = 0; k < N; ++k)
            sa += state[k * N + v] * x[4][t * N + k];
          for (int k = 0; k < N; ++k) {
            int i = t * N + k;
            state[k * N + v] =
                state[k * N + v] *
                    std::exp(-std::exp(-0.5) / (1 + std::exp(-x[1][i]))) +
                sa * x[5][i] + x[2][i] * x[3][t * N + v];
            out += state[k * N + v] * x[0][i];
          }
          loss += out * __half2float(upstream[t * N + v]);
        }
      for (int i = 0; i < S; ++i)
        loss += state[i] * terminal[i];
      return loss;
    };
    wkv_forward(nullptr, IoType::F16, {1, T, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, st.p,
                {tape.p, tape.n});
    wkv_backward_input(nullptr, IoType::F16, {1, T, 1, N}, gpu[0].p, gpu[1].p,
                       gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, dy.p, dst.p,
                       {tape.p, tape.n}, ds0.p, grad[0].p, grad[1].p, grad[2].p,
                       grad[3].p, grad[4].p, grad[5].p);
    for (int p = 0; p < 6; ++p) {
      auto actual = download(grad[p]);
      for (int i : {0, 31, 32, 63, 95, 159, 287, 319}) {
        double old = x[p][i], eps = 1e-5;
        x[p][i] = old + eps;
        double plus = objective();
        x[p][i] = old - eps;
        double minus = objective();
        x[p][i] = old;
        const double expected = (plus - minus) / (2 * eps);
        near(__half2float(actual[i]), expected,
             2e-6 + std::abs(expected) * 6e-4, "WKV input finite difference");
      }
    }
    const auto whole_ds = download(ds0);
    for (int i : {0, 31, 32, 63, 1024, 2048, 4095}) {
      float old = initial[i];
      initial[i] = old + 1e-3f;
      double plus = objective();
      initial[i] = old - 1e-3f;
      double minus = objective();
      initial[i] = old;
      near(whole_ds[i], (plus - minus) / 0.002, 2e-6,
           "WKV state finite difference");
    }
    // A nonzero terminal state adjoint must flow from chunk 1 into chunk 0.
    DeviceBuffer<float> boundary, short_tape, short_ds;
    boundary.resize(S, "boundary");
    short_tape.resize(3 * S, "short tape");
    short_ds.resize(S, "short ds");
    wkv_forward(nullptr, IoType::F16, {1, 2, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, boundary.p,
                {short_tape.p, short_tape.n});
    wkv_forward(nullptr, IoType::F16, {1, 3, 1, N}, boundary.p, gpu[0].p + 128,
                gpu[1].p + 128, gpu[2].p + 128, gpu[3].p + 128, gpu[4].p + 128,
                gpu[5].p + 128, y.p + 128, st.p, {short_tape.p, short_tape.n});
    wkv_backward_input(nullptr, IoType::F16, {1, 3, 1, N}, gpu[0].p + 128,
                       gpu[1].p + 128, gpu[2].p + 128, gpu[3].p + 128,
                       gpu[4].p + 128, gpu[5].p + 128, dy.p + 128, dst.p,
                       {short_tape.p, short_tape.n}, short_ds.p,
                       grad[0].p + 128, grad[1].p + 128, grad[2].p + 128,
                       grad[3].p + 128, grad[4].p + 128, grad[5].p + 128);
    wkv_forward(nullptr, IoType::F16, {1, 2, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, boundary.p,
                {short_tape.p, short_tape.n});
    wkv_backward_input(nullptr, IoType::F16, {1, 2, 1, N}, gpu[0].p, gpu[1].p,
                       gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, dy.p, short_ds.p,
                       {short_tape.p, short_tape.n}, ds0.p, grad[0].p,
                       grad[1].p, grad[2].p, grad[3].p, grad[4].p, grad[5].p);
    auto chunk_ds = download(ds0);
    for (int i = 0; i < S; ++i)
      near(chunk_ds[i], whole_ds[i], 1e-7, "chunk state adjoint");
    // Exercise shared reduction reuse across eight warps with nonzero mean.
    for (int width : {64, 4096}) {
      std::vector<half> hx(width), hw(width), hg(width);
      double mean = 0, variance = 0, sumg = 0, sumgx = 0;
      for (int i = 0; i < width; ++i) {
        hx[i] = __float2half(4.0f + 0.1f * std::sin(i * 0.3f));
        hw[i] = __float2half(0.8f + 0.1f * std::cos(i * 0.2f));
        hg[i] = __float2half(0.03f * std::sin(i * 0.1f));
        mean += __half2float(hx[i]);
      }
      mean /= width;
      for (auto value : hx)
        variance += std::pow(__half2float(value) - mean, 2);
      double inv = 1 / std::sqrt(variance / width + 1e-5);
      for (int i = 0; i < width; ++i) {
        double g = __half2float(hw[i]) * __half2float(hg[i]);
        sumg += g;
        sumgx += g * (__half2float(hx[i]) - mean) * inv;
      }
      DeviceBuffer<half> nx, nw, ng, nd;
      upload(nx, hx);
      upload(nw, hw);
      upload(ng, hg);
      nd.resize(width, "norm dx");
      for (int repeat = 0; repeat < 8; ++repeat) {
        layer_norm_backward_input_f16(nullptr, 1, width, nx.p, nw.p, ng.p, nd.p,
                                      1e-5f);
        auto actual = download(nd);
        for (int i = 0; i < width; ++i) {
          double g = __half2float(hw[i]) * __half2float(hg[i]);
          double expected =
              inv * (g - sumg / width -
                     (__half2float(hx[i]) - mean) * inv * sumgx / width);
          near(__half2float(actual[i]), expected,
               5e-5 + std::abs(expected) * 7e-4, "LN backward");
        }
      }
    }
    // Uniform logits: exact CE=log(V); normalization includes batch/chunk
    // scale.
    DeviceBuffer<half> logits, dlogits;
    DeviceBuffer<int> labels;
    DeviceBuffer<float> loss;
    upload(logits, std::vector<half>(3 * 1024, __float2half(2.0f)));
    upload(labels, std::vector<int>{0, 100, 1023});
    dlogits.resize(3 * 1024, "dlogits");
    loss.resize(1, "loss");
    cross_entropy_forward_backward_f16(nullptr, 3, 1024, logits.p, labels.p, -1,
                                       loss.p, dlogits.p, 0.5f);
    near(download(loss)[0], std::log(1024.0) * 0.5, 1e-5, "CE weighted loss");
    auto dg = download(dlogits);
    near(__half2float(dg[0]), (1.0 / 1024 - 1) / 6, 1e-4, "CE target gradient");
    near(__half2float(dg[1]), 1.0 / 6144, 1e-7, "CE other gradient");
    std::cout << "WKV finite differences, chunk adjoint, LN and CE passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
