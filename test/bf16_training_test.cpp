#include "training.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using rwkv_bf16_training::DeviceBuffer;
using namespace rwkv_bf16_training;

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
    std::array<DeviceBuffer<bf16>, 6> gpu, grad;
    for (int p = 0; p < 6; ++p) {
      x[p].resize(E);
      std::vector<bf16> h(E);
      for (int i = 0; i < E; ++i) {
        h[i] = to_bf16(0.08f * std::sin(i * 0.17f + p));
        x[p][i] = to_float(h[i]);
      }
      upload(gpu[p], h);
      grad[p].resize(E, "sequence gradient");
    }
    std::vector<float> initial(S), terminal(S);
    std::vector<bf16> upstream(E);
    for (int i = 0; i < S; ++i) {
      initial[i] = 0.03f * std::sin(i * 0.11f);
      terminal[i] = 0.01f * std::cos(i * 0.07f);
    }
    for (int i = 0; i < E; ++i)
      upstream[i] = to_bf16(0.1f * std::cos(i * 0.13f));
    DeviceBuffer<float> s0, st, ds0, dst, tape;
    DeviceBuffer<bf16> y, dy;
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
          loss += out * to_float(upstream[t * N + v]);
        }
      for (int i = 0; i < S; ++i)
        loss += state[i] * terminal[i];
      return loss;
    };
    wkv_forward(nullptr, IoType::BF16, {1, T, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, st.p,
                {tape.p, tape.n});
    wkv_backward_input(nullptr, IoType::BF16, {1, T, 1, N}, gpu[0].p, gpu[1].p,
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
        near(to_float(actual[i]), expected, 2e-6 + std::abs(expected) * 4e-3,
             "WKV input finite difference");
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
    // Simulate sequential layers reusing a tape: discard its old contents,
    // forward without recording, then replay before consuming gradients.
    const auto saved_y = download(y);
    const auto saved_final = download(st);
    std::array<std::vector<bf16>, 6> saved_grad;
    for (int p = 0; p < 6; ++p)
      saved_grad[p] = download(grad[p]);
    upload(tape, std::vector<float>(T * S, 123.0f));
    wkv_forward(nullptr, IoType::BF16, {1, T, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, st.p, {});
    auto no_tape_y = download(y);
    auto no_tape_final = download(st);
    for (int i = 0; i < E; ++i)
      near(to_float(no_tape_y[i]), to_float(saved_y[i]), 0,
           "no-tape forward output");
    for (int i = 0; i < S; ++i)
      near(no_tape_final[i], saved_final[i], 0, "no-tape final state");
    for (float v : download(tape))
      near(v, 123, 0, "no-tape recording disabled");
    wkv_forward(nullptr, IoType::BF16, {1, T, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, st.p,
                {tape.p, tape.n});
    wkv_backward_input(nullptr, IoType::BF16, {1, T, 1, N}, gpu[0].p, gpu[1].p,
                       gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, dy.p, dst.p,
                       {tape.p, tape.n}, ds0.p, grad[0].p, grad[1].p, grad[2].p,
                       grad[3].p, grad[4].p, grad[5].p);
    auto replay_ds = download(ds0);
    for (int i = 0; i < S; ++i)
      near(replay_ds[i], whole_ds[i], 0, "replay state gradient");
    for (int p = 0; p < 6; ++p) {
      auto actual = download(grad[p]);
      for (int i = 0; i < E; ++i)
        near(to_float(actual[i]), to_float(saved_grad[p][i]), 0,
             "replay input gradient");
    }
    // A nonzero terminal state adjoint must flow from chunk 1 into chunk 0.
    DeviceBuffer<float> boundary, short_tape, short_ds;
    boundary.resize(S, "boundary");
    short_tape.resize(3 * S, "short tape");
    short_ds.resize(S, "short ds");
    wkv_forward(nullptr, IoType::BF16, {1, 2, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, boundary.p,
                {short_tape.p, short_tape.n});
    wkv_forward(nullptr, IoType::BF16, {1, 3, 1, N}, boundary.p, gpu[0].p + 128,
                gpu[1].p + 128, gpu[2].p + 128, gpu[3].p + 128, gpu[4].p + 128,
                gpu[5].p + 128, y.p + 128, st.p, {short_tape.p, short_tape.n});
    wkv_backward_input(nullptr, IoType::BF16, {1, 3, 1, N}, gpu[0].p + 128,
                       gpu[1].p + 128, gpu[2].p + 128, gpu[3].p + 128,
                       gpu[4].p + 128, gpu[5].p + 128, dy.p + 128, dst.p,
                       {short_tape.p, short_tape.n}, short_ds.p,
                       grad[0].p + 128, grad[1].p + 128, grad[2].p + 128,
                       grad[3].p + 128, grad[4].p + 128, grad[5].p + 128);
    wkv_forward(nullptr, IoType::BF16, {1, 2, 1, N}, s0.p, gpu[0].p, gpu[1].p,
                gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, y.p, boundary.p,
                {short_tape.p, short_tape.n});
    wkv_backward_input(nullptr, IoType::BF16, {1, 2, 1, N}, gpu[0].p, gpu[1].p,
                       gpu[2].p, gpu[3].p, gpu[4].p, gpu[5].p, dy.p, short_ds.p,
                       {short_tape.p, short_tape.n}, ds0.p, grad[0].p,
                       grad[1].p, grad[2].p, grad[3].p, grad[4].p, grad[5].p);
    auto chunk_ds = download(ds0);
    for (int i = 0; i < S; ++i)
      near(chunk_ds[i], whole_ds[i], 1e-7, "chunk state adjoint");
    // Exercise shared reduction reuse across eight warps with nonzero mean.
    for (int width : {64, 4096}) {
      std::vector<bf16> hx(width), hw(width), hg(width);
      double mean = 0, variance = 0, sumg = 0, sumgx = 0;
      for (int i = 0; i < width; ++i) {
        hx[i] = to_bf16(4.0f + 0.1f * std::sin(i * 0.3f));
        hw[i] = to_bf16(0.8f + 0.1f * std::cos(i * 0.2f));
        hg[i] = to_bf16(0.03f * std::sin(i * 0.1f));
        mean += to_float(hx[i]);
      }
      mean /= width;
      for (auto value : hx)
        variance += std::pow(to_float(value) - mean, 2);
      double inv = 1 / std::sqrt(variance / width + 1e-5);
      for (int i = 0; i < width; ++i) {
        double g = to_float(hw[i]) * to_float(hg[i]);
        sumg += g;
        sumgx += g * (to_float(hx[i]) - mean) * inv;
      }
      DeviceBuffer<bf16> nx, nw, ng, nd;
      upload(nx, hx);
      upload(nw, hw);
      upload(ng, hg);
      nd.resize(width, "norm dx");
      for (int repeat = 0; repeat < 8; ++repeat) {
        layer_norm_backward_input_bf16(nullptr, 1, width, nx.p, nw.p, ng.p,
                                       nd.p, 1e-5f);
        auto actual = download(nd);
        for (int i = 0; i < width; ++i) {
          double g = to_float(hw[i]) * to_float(hg[i]);
          double expected =
              inv * (g - sumg / width -
                     (to_float(hx[i]) - mean) * inv * sumgx / width);
          near(to_float(actual[i]), expected,
               5e-5 + std::abs(expected) * 4.5e-3, "LN backward");
        }
      }
    }
    // BF16 preserves a real-vocabulary derivative at a long-batch scale
    // without loss scaling or a cast through FP16.
    DeviceBuffer<bf16> logits, dlogits;
    DeviceBuffer<int> labels;
    DeviceBuffer<float> loss;
    upload(logits, std::vector<bf16>(65536, to_bf16(0)));
    upload(labels, std::vector<int>{7});
    dlogits.resize(65536, "dlogits");
    loss.resize(1, "loss");
    cross_entropy_forward_backward_bf16(nullptr, 1, 65536, logits.p, labels.p,
                                        -1, loss.p, dlogits.p, 1.0f / 262144);
    auto g = download(dlogits);
    const double expected = 1.0 / 65536 / 262144;
    near(to_float(g[0]), expected, expected * .004, "BF16 non-target gradient");
    near(download(loss)[0], std::log(65536.0) / 262144, 1e-10,
         "BF16 mean loss");
    std::cout << "Native BF16 WKV finite differences, chunk adjoint, LN and "
                 "long-batch CE passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
