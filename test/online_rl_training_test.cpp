#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
using namespace rwkv_bf16_training;
std::vector<float> read(const float *p, size_t n) {
  std::vector<float> out(n);
  gpu_check(cudaMemcpy(out.data(), p, n * 4, cudaMemcpyDeviceToHost),
            "read test gradient");
  return out;
}
void compare(const std::vector<float> &a, const std::vector<float> &b,
             double tolerance, const char *name) {
  double error = 0, norm = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    error += std::pow(a[i] - b[i], 2);
    norm += double(a[i]) * a[i];
  }
  double relative = std::sqrt(error / std::max(norm, 1e-30));
  std::cout << name << " relative_l2=" << relative
            << " norm=" << std::sqrt(norm) << '\n';
  if (!std::isfinite(relative) || relative > tolerance)
    throw std::runtime_error(name);
}
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: online_rl_training_test tiny-model.pth\n";
    return 77;
  }
  try {
    set_consistent_forward(true);
    Model model(argv[1]);
    auto view = model.state_tuning_model_view();
    if (view.channels > 128)
      throw std::invalid_argument("use a tiny fixture for this test");
    Json::Value config;
    config["base_fingerprint"] = model.base_fingerprint();
    rwkv_bf16_miss::Trainer trainer(view, 8, 8, "all", 31, config, nullptr);
    Executor full(view, 31, nullptr), chunked(view, 7, nullptr);
    DeviceBuffer<float> initial;
    initial.resize(size_t(view.layers) * view.heads * 4096, "test state");
    std::vector<float> host(initial.n);
    for (size_t i = 0; i < host.size(); ++i)
      host[i] = .02f * std::sin(i * .07f);
    gpu_check(cudaMemcpy(initial.p, host.data(), host.size() * 4,
                         cudaMemcpyHostToDevice),
              "initial");
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i)
      ids[i] = 98 + i % 3;
    // All supervision is in the last chunks/turn. Early masked context must
    // still transmit both recurrent and shift gradients to the initial state.
    std::vector<float> weights(32, 0);
    weights[22] = 1.5;
    weights[25] = -.3;
    weights[31] = 2;
    auto gradients = [&] {
      std::vector<float> result;
      for (const auto &b : view.blocks)
        for (const auto &w : b.miss) {
          auto g = read(w.gradient, size_t(w.out) * w.rank);
          result.insert(result.end(), g.begin(), g.end());
        }
      return result;
    };
    trainer.bind_tapes(full.tapes());
    trainer.zero_gradient();
    float lf = full.backward(view, ids, weights, initial.p, 3, .7f);
    auto gf = gradients(), sf = read(full.state_gradient(), initial.n);
    trainer.bind_tapes(chunked.tapes());
    trainer.zero_gradient();
    float lc = chunked.backward(view, ids, weights, initial.p, 3, .7f);
    compare(gf, gradients(), .04, "signed masked MiSS whole/chunks");
    compare(sf, read(chunked.state_gradient(), initial.n), .04,
            "initial-state whole/chunks");
    if (std::abs(lf - lc) > 1e-4 ||
        std::inner_product(sf.begin(), sf.end(), sf.begin(), 0.) < 1e-15)
      throw std::runtime_error(
          "lost early context gradient or inconsistent loss");
    for (float &w : weights)
      w = -w;
    trainer.bind_tapes(full.tapes());
    trainer.zero_gradient();
    float ln = full.backward(view, ids, weights, initial.p, 3, .7f);
    for (float &g : gf)
      g = -g;
    for (float &g : sf)
      g = -g;
    compare(gf, gradients(), 1e-5, "signed MiSS direction");
    compare(sf, read(full.state_gradient(), initial.n), 1e-5,
            "signed state direction");
    if (std::abs(lf + ln) > 1e-5)
      throw std::runtime_error("signed loss direction");
    std::cout << "Full cross-chunk/masked multi-turn gradients passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
