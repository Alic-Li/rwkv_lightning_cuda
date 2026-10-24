#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
using namespace rwkv_bf16_training;
// BF16 rounds between layers. Check distribution drift as well as logits;
// strict mode continues to require bitwise equality across all row shapes.
double compare(const std::vector<float> &a, const std::vector<float> &b,
               bool strict, const char *name) {
  double error = 0, max_abs = 0, za = 0, zb = 0;
  const float ma = *std::max_element(a.begin(), a.end());
  const float mb = *std::max_element(b.begin(), b.end());
  for (size_t i = 0; i < a.size(); ++i) {
    error += std::pow(double(a[i]) - b[i], 2);
    max_abs = std::max(max_abs, std::abs(double(a[i]) - b[i]));
    za += std::exp(double(a[i]) - ma);
    zb += std::exp(double(b[i]) - mb);
  }
  double kl = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
    kl += std::exp(la) * (la - lb);
  }
  double rms = std::sqrt(error / a.size());
  std::cout << name << ": max_abs=" << max_abs << " rms=" << rms << " kl=" << kl
            << std::endl;
  // KL is invariant to an unimportant common logit offset and downweights
  // tail tokens. Raw maximum/RMS logit differences remain diagnostics.
  if ((strict && a != b) || !std::isfinite(kl) || !std::isfinite(rms) ||
      kl > .02)
    throw std::runtime_error(name);
  return kl;
}
int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: online_rl_forward_test model adapter state [bf16]\n";
    return 77;
  }
  try {
    const bool strict = argc == 4;
    if (!strict && std::string(argv[4]) != "bf16")
      throw std::invalid_argument("expected bf16 mode");
    set_consistent_forward(true, !strict);
    Model model(argv[1]);
    auto m = model.state_tuning_model_view();
    auto package = rwkv7_miss::load_package(argv[2]);
    Json::Value config;
    config["base_fingerprint"] = model.base_fingerprint();
    rwkv_bf16_miss::Trainer trainer(m, package->manifest["rank"].asInt(),
                                    package->manifest["alpha"].asFloat(), "all",
                                    40, config, nullptr);
    trainer.load_adapter(*package);
    auto initial = model.load_state(argv[3]);
    auto actor = m;
    auto snapshot = trainer.snapshot(actor);
    Executor inference(actor, 40, nullptr), learner(m, 40, nullptr);
    trainer.bind_tapes(learner.tapes());
    std::vector<int> ids(40);
    for (size_t i = 0; i < ids.size(); ++i)
      ids[i] = 98 + i % 8;
    auto run = [&](Executor &e, const FrozenModelView &view, int chunk,
                   bool roundtrip, bool last_only = false) {
      e.reset(initial.p);
      for (int i = 0; i < int(ids.size()); i += chunk) {
        e.forward(view, ids.data() + i, std::min(chunk, int(ids.size()) - i),
                  !last_only                     ? LogitsMode::All
                  : i + chunk >= int(ids.size()) ? LogitsMode::Last
                                                 : LogitsMode::None);
        if (roundtrip) {
          std::vector<float> state(e.state().n);
          std::vector<bf16> shifts(e.shifts().n);
          gpu_check(cudaMemcpy(state.data(), e.state().p, state.size() * 4,
                               cudaMemcpyDeviceToHost),
                    "download state");
          gpu_check(cudaMemcpy(shifts.data(), e.shifts().p, shifts.size() * 2,
                               cudaMemcpyDeviceToHost),
                    "download shifts");
          gpu_check(cudaMemcpy(e.state().p, state.data(), state.size() * 4,
                               cudaMemcpyHostToDevice),
                    "upload state");
          gpu_check(cudaMemcpy(e.shifts().p, shifts.data(), shifts.size() * 2,
                               cudaMemcpyHostToDevice),
                    "upload shifts");
        }
      }
      return e.last_logits();
    };
    auto whole = run(inference, actor, 40, false),
         split = run(inference, actor, 2, false),
         cached = run(inference, actor, 2, true);
    if (split != cached)
      throw std::runtime_error("CPU prefix snapshots changed logits");
    if (whole != run(learner, m, 40, false))
      throw std::runtime_error("actor/learner same-shape forward differs");
    std::cout << "CPU cache roundtrip: bitwise equal; actor/learner same-shape "
                 "logits: bitwise equal\n";
    compare(whole, split, strict, "chunk 40 vs 2");
    compare(whole, run(inference, actor, 1, false), strict, "chunk 40 vs 1");
    compare(whole, run(inference, actor, 7, false), strict, "chunk 40 vs 7");
    compare(whole, run(inference, actor, 40, false, true), strict,
            "full vs final-only output head");
    compare(whole, run(inference, actor, 7, false, true), strict,
            "state-only prefill plus final output head");

    // Compare each row with its own batch=1 history, including unequal
    // prompt lengths. Strict mode additionally requires bitwise equality.
    const int rows = 3, steps = 4, V = m.vocab;
    double decode_kl = 0;
    BatchDecoder decoder(actor, rows, nullptr);
    std::vector<std::vector<int>> histories(rows);
    for (int r = 0; r < rows; ++r) {
      histories[r].assign(ids.begin(), ids.begin() + 5 + 7 * r);
      inference.reset(initial.p);
      inference.forward(actor, histories[r].data(), int(histories[r].size()));
      decoder.load(r, inference.state().p, inference.shifts().p);
    }
    for (int step = 0; step < steps; ++step) {
      std::vector<int> feed(rows);
      for (int r = 0; r < rows; ++r) {
        feed[r] = 100 + (r * 3 + step) % 8;
        histories[r].push_back(feed[r]);
      }
      decoder.forward(actor, feed.data());
      const auto batched = decoder.logits();
      for (int r = 0; r < rows; ++r) {
        inference.reset(initial.p);
        for (int t : histories[r])
          inference.forward(actor, &t, 1);
        const auto single = inference.last_logits();
        decode_kl +=
            compare(single,
                    std::vector<float>(batched.begin() + size_t(r) * V,
                                       batched.begin() + size_t(r + 1) * V),
                    strict, "batched decode vs batch=1");
      }
    }
    std::cout << "Mean batched/single decode KL=" << decode_kl / (rows * steps)
              << std::endl;
    if (decode_kl / (rows * steps) > .005)
      throw std::runtime_error("mean batched/single decode distribution drift");
    std::vector<float> state;
    std::vector<bf16> shifts;
    decoder.download(1, state, shifts);
    if (state.size() != inference.state().n ||
        shifts.size() != inference.shifts().n)
      throw std::runtime_error("decoder row download layout");

    // Teacher-forced scores are log softmax of the same forward logits.
    const float temperature = 0.7f;
    const auto scores = inference.score(actor, ids, initial.p, temperature);
    if (scores.size() != ids.size() - 1)
      throw std::runtime_error("score length");
    double worst = 0;
    inference.reset(initial.p);
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      inference.forward(actor, &ids[i], 1);
      const auto logits = inference.last_logits();
      const float top = *std::max_element(logits.begin(), logits.end());
      double sum = 0;
      for (float x : logits)
        sum += std::exp((double(x) - top) / temperature);
      const double expected =
          (double(logits[ids[i + 1]]) - top) / temperature - std::log(sum);
      worst = std::max(worst, std::abs(expected - scores[i]));
    }
    std::cout << "Teacher-forced score max_abs=" << worst << std::endl;
    if (worst > (strict ? 1e-3 : .25))
      throw std::runtime_error("teacher-forced score differs from logits");
    std::cout << "Batched decode validated for " << rows << " rows x " << steps
              << " steps; score max_abs=" << worst << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
