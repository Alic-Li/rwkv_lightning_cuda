// Padding-free packed micro-batches must match per-sample backward() summed
// over the same samples: loss, initial-state gradient and adapter gradients.
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
using namespace rwkv_bf16_training;
namespace {
std::vector<float> read(const float *p, size_t n) {
  std::vector<float> out(n);
  gpu_check(cudaMemcpy(out.data(), p, n * 4, cudaMemcpyDeviceToHost),
            "read test gradient");
  return out;
}
void compare(const std::vector<float> &a, const std::vector<float> &b,
             double tolerance, const std::string &name) {
  double error = 0, norm = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    error += std::pow(double(a[i]) - b[i], 2);
    norm += double(a[i]) * a[i];
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  std::cout << name << " relative_l2=" << relative
            << " norm=" << std::sqrt(norm) << '\n';
  if (!std::isfinite(relative) || relative > tolerance || norm == 0)
    throw std::runtime_error(name);
}
struct Batch {
  std::vector<std::vector<int>> tokens;
  std::vector<std::vector<float>> weights;
  size_t normalizer = 0;
};
// Lengths straddle chunk boundaries; prefixes are masked like prompt turns.
// A fully masked sample leaves its lane empty for the whole call.
Batch make_batch(const std::vector<int> &lengths, int vocab, unsigned seed) {
  std::mt19937 rng(seed);
  Batch batch;
  for (int length : lengths) {
    std::vector<int> ids(length);
    std::vector<float> weights(length, 0);
    for (int &id : ids)
      id = 33 + int(rng() % std::min(vocab - 33, 20000));
    const int prompt = length > 8 ? length / 3 : length;
    for (int i = prompt + 1; i < length; ++i)
      weights[i] = 1;
    for (int i = 1; i < length; ++i)
      batch.normalizer += weights[i] != 0;
    batch.tokens.push_back(ids);
    batch.weights.push_back(weights);
  }
  return batch;
}
std::vector<float> adapter_gradients(const FrozenModelView &view) {
  std::vector<float> result;
  for (const auto &b : view.blocks)
    for (const auto &w : b.miss) {
      if (w.gradient) {
        auto g = read(w.gradient, size_t(w.out) * w.rank);
        result.insert(result.end(), g.begin(), g.end());
      }
      if (w.gradient_a) {
        auto g = read(w.gradient_a, size_t(w.rank) * w.in);
        result.insert(result.end(), g.begin(), g.end());
      }
    }
  return result;
}
void check(Model &model, const std::string &method, double tolerance) {
  auto view = model.state_tuning_model_view();
  constexpr int chunk = 16, lanes = 4;
  std::unique_ptr<rwkv_bf16_miss::Trainer> trainer;
  if (method != "state") {
    Json::Value config;
    config["base_fingerprint"] = model.base_fingerprint();
    if (method == "nora") {
      config["method"] = "nora";
      config["initialization"] = "nora_init";
    }
    trainer = std::make_unique<rwkv_bf16_miss::Trainer>(
        view, 8, 8, "all", chunk * lanes, config, nullptr);
  }
  Executor single(view, chunk, nullptr), packed(view, chunk, nullptr, lanes);
  const size_t states = size_t(view.layers) * view.heads * 4096;
  DeviceBuffer<float> initial;
  initial.resize(states, "test state");
  std::vector<float> host(states);
  for (size_t i = 0; i < host.size(); ++i)
    host[i] = .02f * std::sin(i * .07f);
  gpu_check(cudaMemcpy(initial.p, host.data(), host.size() * 4,
                       cudaMemcpyHostToDevice),
            "initial");
  const std::vector<std::vector<int>> cases = {
      {37, 5, 50, 18}, {33, 6, 21}, {64, 16, 17, 2}};
  for (int round = 0; round < 2; ++round) {
    for (size_t c = 0; c < cases.size(); ++c) {
      const auto batch = make_batch(cases[c], view.vocab, unsigned(c + 7));
      const std::string name =
          method + " round" + std::to_string(round) + " case" + std::to_string(c);
      if (trainer) {
        trainer->bind_tapes(single.tapes());
        trainer->zero_gradient();
      }
      float expected_loss = 0;
      std::vector<float> expected_state(states, 0);
      for (size_t i = 0; i < batch.tokens.size(); ++i) {
        expected_loss += single.backward(view, batch.tokens[i],
                                         batch.weights[i], initial.p,
                                         batch.normalizer);
        const auto g = read(single.state_gradient(), states);
        for (size_t j = 0; j < states; ++j)
          expected_state[j] += g[j];
      }
      std::vector<float> expected_adapter;
      if (trainer) {
        expected_adapter = adapter_gradients(view);
        trainer->bind_tapes(packed.tapes());
        trainer->zero_gradient();
      }
      const float loss = packed.backward_packed(
          view, batch.tokens, batch.weights, initial.p, batch.normalizer);
      std::cout << name << " loss " << expected_loss << " vs " << loss << '\n';
      if (std::abs(loss - expected_loss) > tolerance * std::abs(expected_loss))
        throw std::runtime_error(name + " loss");
      compare(expected_state, read(packed.state_gradient(), states), tolerance,
              name + " dState");
      if (trainer)
        compare(expected_adapter, adapter_gradients(view), tolerance,
                name + " adapter");
    }
    // A second round after one update makes NoRA/MiSS D nonzero, so NoRA's
    // A gradient and adapter forward contributions are exercised too.
    if (!trainer)
      break;
    trainer->update(1e-2f, 1);
  }
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: packed_batch_test model.pth\n";
    return 77;
  }
  try {
    Model model(argv[1]);
    // Consistent GEMMs make every row independent of packing, so only the
    // adapter-gradient row summation order may differ. The auto policy lets
    // cuBLAS pick kernels per row count; on these random-token inputs that
    // BF16 rounding alone moves gradients by percents (even between chunk
    // sizes of backward()), so it is not compared here.
    for (bool tensor_cores : {false, true}) {
      set_consistent_forward(true, tensor_cores);
      for (const char *method : {"state", "miss", "nora"})
        check(model, method, 1e-5);
    }
    std::cout << "Packed micro-batch matches per-sample accumulation\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
