#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace rwkv_bf16_training;
int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: online_rl_forward_test model adapter state\n";
    return 77;
  }
  try {
    set_consistent_forward(true);
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
                   bool roundtrip) {
      e.reset(initial.p);
      for (int i = 0; i < int(ids.size()); i += chunk) {
        e.forward(view, ids.data() + i, std::min(chunk, int(ids.size()) - i));
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
    double max_abs = 0, squared = 0;
    for (size_t i = 0; i < whole.size(); ++i) {
      double d = std::abs(whole[i] - split[i]);
      max_abs = std::max(max_abs, d);
      squared += d * d;
    }
    double rms = std::sqrt(squared / whole.size());
    std::cout << "CPU cache roundtrip: bitwise equal; actor/learner same-shape "
                 "logits: bitwise equal\n"
              << "BF16 chunk 40 vs 2: max_abs=" << max_abs << " rms=" << rms
              << '\n';
    if (whole != split || whole != run(inference, actor, 1, false) ||
        whole != run(inference, actor, 7, false))
      throw std::runtime_error("fixed-tile chunk/decode policy changed");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
