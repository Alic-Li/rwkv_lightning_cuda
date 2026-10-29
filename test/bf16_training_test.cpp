#include "rwkv/bf16_training/training.hpp"
#include <array>
#include <cmath>
#include <cstring>
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

// FP64 reference loss for one lane: sum(y * dy) + sum(final * dsT).
double wkv_reference(int T, const std::array<std::vector<double>, 6> &x,
                     const std::vector<float> &initial,
                     const std::vector<bf16> &upstream,
                     const std::vector<float> &terminal) {
  constexpr int N = 64;
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
  for (int i = 0; i < N * N; ++i)
    loss += state[i] * terminal[i];
  return loss;
}

template <class T> bool same_bits(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
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
    tape.resize(wkv_tape_elements({1, T, 1, N}), "tape");
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
    upload(tape, std::vector<float>(tape.n, 123.0f));
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
    short_tape.resize(wkv_tape_elements({1, 3, 1, N}), "short tape");
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
    // Rows past one checkpoint interval are replayed from a checkpoint.
    // 37 rows cross two interval boundaries and end mid-interval.
    {
      constexpr int LT = 37, LE = LT * N;
      std::array<std::vector<double>, 6> lx;
      std::array<DeviceBuffer<bf16>, 6> lgpu, lgrad;
      for (int p = 0; p < 6; ++p) {
        lx[p].resize(LE);
        std::vector<bf16> h(LE);
        for (int i = 0; i < LE; ++i) {
          h[i] = to_bf16(0.08f * std::sin(i * 0.23f + 2 * p + 1));
          lx[p][i] = to_float(h[i]);
        }
        upload(lgpu[p], h);
        lgrad[p].resize(LE, "long sequence gradient");
      }
      std::vector<bf16> lup(LE);
      for (int i = 0; i < LE; ++i)
        lup[i] = to_bf16(0.1f * std::cos(i * 0.19f));
      DeviceBuffer<bf16> ly, ldy;
      DeviceBuffer<float> lst, lds, ltape;
      upload(ldy, lup);
      ly.resize(LE, "long output");
      lst.resize(S, "long final");
      lds.resize(S, "long ds0");
      ltape.resize(wkv_tape_elements({1, LT, 1, N}), "long tape");
      auto run_dense = [&](int rows, int at, const float *init,
                           const float *ds_final, float *final_state,
                           float *ds_initial) {
        const WkvTapeView view{ltape.p, ltape.n};
        wkv_forward(nullptr, IoType::BF16, {1, rows, 1, N}, init,
                    lgpu[0].p + at, lgpu[1].p + at, lgpu[2].p + at,
                    lgpu[3].p + at, lgpu[4].p + at, lgpu[5].p + at,
                    ly.p + at, final_state, view);
        wkv_backward_input(nullptr, IoType::BF16, {1, rows, 1, N},
                           lgpu[0].p + at, lgpu[1].p + at, lgpu[2].p + at,
                           lgpu[3].p + at, lgpu[4].p + at, lgpu[5].p + at,
                           ldy.p + at, ds_final, view, ds_initial,
                           lgrad[0].p + at, lgrad[1].p + at, lgrad[2].p + at,
                           lgrad[3].p + at, lgrad[4].p + at, lgrad[5].p + at);
      };
      run_dense(LT, 0, s0.p, dst.p, lst.p, lds.p);
      std::array<std::vector<bf16>, 6> dense_grad;
      for (int p = 0; p < 6; ++p) {
        dense_grad[p] = download(lgrad[p]);
        for (int t : {0, 14, 15, 16, 17, 31, 32, 36})
          for (int c : {3, 40}) {
            const int i = t * N + c;
            double old = lx[p][i], eps = 1e-5;
            lx[p][i] = old + eps;
            double plus = wkv_reference(LT, lx, initial, lup, terminal);
            lx[p][i] = old - eps;
            double minus = wkv_reference(LT, lx, initial, lup, terminal);
            lx[p][i] = old;
            const double expected = (plus - minus) / (2 * eps);
            near(to_float(dense_grad[p][i]), expected,
                 2e-6 + std::abs(expected) * 4e-3,
                 "multi-interval WKV finite difference");
          }
      }
      const auto dense_ds = download(lds);
      const auto dense_y = download(ly);
      for (int i : {0, 63, 1024, 4095}) {
        std::vector<float> init(initial);
        init[i] += 1e-3f;
        double plus = wkv_reference(LT, lx, init, lup, terminal);
        init[i] -= 2e-3f;
        double minus = wkv_reference(LT, lx, init, lup, terminal);
        near(dense_ds[i], (plus - minus) / 0.002, 2e-6,
             "multi-interval WKV state finite difference");
      }

      // Packed lanes [0,5), [5,5) and [5,37) keep per-lane checkpoints:
      // each lane matches its own dense run bit for bit, and repeating the
      // packed pass is deterministic.
      const std::vector<int> lane_offsets{0, 5, 5, LT};
      DeviceBuffer<int> offsets;
      upload(offsets, lane_offsets);
      std::vector<float> lanes3(3 * S), dlanes3(3 * S);
      for (int l = 0; l < 3; ++l)
        for (int i = 0; i < S; ++i) {
          lanes3[l * S + i] = initial[i] * (1 + 0.25f * l);
          dlanes3[l * S + i] = terminal[i] * (1 - 0.25f * l);
        }
      DeviceBuffer<float> ps0, pdst, pst, pds, ptape;
      upload(ps0, lanes3);
      upload(pdst, dlanes3);
      pst.resize(3 * S, "packed final");
      pds.resize(3 * S, "packed ds0");
      const WkvShape packed{3, LT, 1, N, offsets.p};
      ptape.resize(wkv_tape_elements(packed), "packed tape");
      std::vector<std::vector<bf16>> first_run;
      std::vector<float> first_ds, first_final;
      for (int repeat = 0; repeat < 4; ++repeat) {
        upload(ptape, std::vector<float>(ptape.n, repeat ? NAN : 0.0f));
        const WkvTapeView view{ptape.p, ptape.n};
        wkv_forward(nullptr, IoType::BF16, packed, ps0.p, lgpu[0].p, lgpu[1].p,
                    lgpu[2].p, lgpu[3].p, lgpu[4].p, lgpu[5].p, ly.p, pst.p,
                    view);
        wkv_backward_input(nullptr, IoType::BF16, packed, lgpu[0].p, lgpu[1].p,
                           lgpu[2].p, lgpu[3].p, lgpu[4].p, lgpu[5].p, ldy.p,
                           pdst.p, view, pds.p, lgrad[0].p, lgrad[1].p,
                           lgrad[2].p, lgrad[3].p, lgrad[4].p, lgrad[5].p);
        std::vector<std::vector<bf16>> run;
        for (int p = 0; p < 6; ++p)
          run.push_back(download(lgrad[p]));
        run.push_back(download(ly));
        if (!repeat) {
          first_run = run;
          first_ds = download(pds);
          first_final = download(pst);
        } else {
          bool same = same_bits(download(pds), first_ds) &&
                      same_bits(download(pst), first_final);
          for (int p = 0; p < 7; ++p)
            same = same && same_bits(run[p], first_run[p]);
          if (!same)
            throw std::runtime_error("packed WKV backward is not deterministic");
        }
      }
      // Empty lane: state and adjoint pass through unchanged.
      for (int i = 0; i < S; ++i) {
        near(first_final[S + i], lanes3[S + i], 0, "empty lane final state");
        near(first_ds[S + i], dlanes3[S + i], 0, "empty lane state gradient");
      }
      for (int lane : {0, 2}) {
        const int at = lane_offsets[lane], rows = lane_offsets[lane + 1] - at;
        run_dense(rows, at * N, ps0.p + lane * S, pdst.p + lane * S, lst.p,
                  lds.p);
        if (!same_bits(std::vector<float>(first_final.begin() + lane * S,
                                          first_final.begin() + (lane + 1) * S),
                       download(lst)) ||
            !same_bits(std::vector<float>(first_ds.begin() + lane * S,
                                          first_ds.begin() + (lane + 1) * S),
                       download(lds)))
          throw std::runtime_error("packed lane state differs from dense run");
        const auto lane_y = download(ly);
        for (int p = 0; p < 7; ++p) {
          const auto dense = p < 6 ? download(lgrad[p]) : lane_y;
          if (std::memcmp(dense.data() + at * N, first_run[p].data() + at * N,
                          rows * N * sizeof(bf16)))
            throw std::runtime_error("packed lane gradient differs from dense");
        }
      }
      // The packed first lane starts from the dense initial state times 1.
      if (!same_bits(std::vector<bf16>(first_run[6].begin(),
                                       first_run[6].begin() + 5 * N),
                     std::vector<bf16>(dense_y.begin(),
                                       dense_y.begin() + 5 * N)))
        throw std::runtime_error("packed lane output differs from dense");
    }
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
    // Ignored rows are masked out of both loss and gradient; the per-row
    // weight is still gradient_scale / rows.
    DeviceBuffer<bf16> masked_logits, masked_dlogits;
    DeviceBuffer<int> masked_labels;
    upload(masked_logits, std::vector<bf16>(2 * 64, to_bf16(0)));
    upload(masked_labels, std::vector<int>{3, -1});
    masked_dlogits.resize(2 * 64, "masked dlogits");
    cross_entropy_forward_backward_bf16(nullptr, 2, 64, masked_logits.p,
                                        masked_labels.p, -1, loss.p,
                                        masked_dlogits.p, 2.0f);
    auto masked = download(masked_dlogits);
    near(to_float(masked[3]), 1.0 / 64 - 1, 4e-3, "masked kept target");
    for (int i = 64; i < 128; ++i)
      near(to_float(masked[i]), 0.0, 0.0, "masked row gradient");
    near(download(loss)[0], std::log(64.0), 1e-5, "masked mean loss");
    // Signed advantages and temperature: compare the existing CE kernel to
    // an independent FP64 softmax, including a zero-weight masked row.
    std::vector<bf16> signed_logits(3*64);
    for(int i=0;i<3*64;++i) signed_logits[i]=to_bf16(.03f*(i%64)-.5f);
    upload(masked_logits,signed_logits);upload(masked_labels,std::vector<int>{3,7,11});
    masked_dlogits.resize(3*64,"signed dlogits");
    DeviceBuffer<float> row_weights;upload(row_weights,std::vector<float>{1.5f,-2.f,0.f});
    cross_entropy_forward_backward_bf16(nullptr,3,64,masked_logits.p,masked_labels.p,
        -1,loss.p,masked_dlogits.p,1.0f,row_weights.p,.7f);
    auto signed_grad=download(masked_dlogits);double expected_loss=0;
    for(int row=0;row<3;++row) {
      double sum=0;for(int i=0;i<64;++i) sum+=std::exp(to_float(signed_logits[row*64+i])/.7);
      const int target=std::array<int,3>{3,7,11}[row];
      double weight=std::array<double,3>{1.5,-2,0}[row]/3;
      expected_loss+=weight*(std::log(sum)-to_float(signed_logits[row*64+target])/.7);
      for(int i=0;i<64;++i) {
        double p=std::exp(to_float(signed_logits[row*64+i])/.7)/sum;
        double g=weight*(p-(i==target))/.7;
        near(to_float(signed_grad[row*64+i]),g,1e-6+std::abs(g)*.004,"signed temperature CE gradient");
      }
    }
    near(download(loss)[0],expected_loss,2e-6,"signed CE loss");
    std::cout << "Native BF16 WKV finite differences (incl. checkpoint "
                 "replay, packed lanes, determinism), chunk adjoint, LN, "
                 "long-batch and masked CE passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
