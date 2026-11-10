#include <iostream>
#include <vector>

#include "rwkv/inference/rwkv_sampler.hpp"
#include "test_common.hpp"

namespace {

rwkv7_server::DeviceLogits make_logits(const std::vector<float>& host_logits) {
  rwkv7_server::DeviceLogits logits;
  logits.rows = 1;
  logits.vocab_size = static_cast<int>(host_logits.size());
  rwkv_test::copy_host_to_device(host_logits, logits.values, "alloc sampler test logits", "copy sampler test logits");
  return logits;
}

std::vector<float> copy_penalties(const rwkv7_server::SamplerPenaltyState& state) {
  return rwkv_test::copy_device_buffer(state.penalties, "copy sampler penalties");
}

}  // namespace

int main() {
  try {
    if (!rwkv_test::cuda_device_available()) {
      std::cout << "rwkv_sampler_test skipped: no CUDA device available\n";
      return 0;
    }

    {
      auto state = rwkv7_server::make_sampler_penalties(4);
      auto logits = make_logits({0.25f, 3.0f, 1.0f, -1.0f});
      rwkv7_server::GenerateOptions options;
      options.top_k = 0;
      options.top_p = 0.0;
      options.temperature = 1.0;
      TEST_EQ(rwkv7_server::sample_repetition_temperature_topk_topp(logits, state, options), 1);
    }

    {
      auto state = rwkv7_server::make_sampler_penalties(4);
      auto logits = make_logits({0.25f, 3.0f, 1.0f, -1.0f});
      rwkv7_server::GenerateOptions options;
      options.top_k = 0;
      options.top_p = 0.0;
      options.temperature = 1.0;
      options.alpha_presence = 0.0;
      options.alpha_frequency = 0.0;

      TEST_EQ(rwkv7_server::sample_repetition_temperature_topk_topp(logits, state, options), 1);
      auto penalties = copy_penalties(state);
      for (float penalty : penalties) {
        TEST_NEAR(penalty, 0.0, 1e-6);
      }
    }

    {
      auto state = rwkv7_server::make_sampler_penalties(4);
      auto logits = make_logits({100.0f, 0.0f, 0.0f, 0.0f});
      rwkv7_server::GenerateOptions options;
      options.top_k = 0;
      options.top_p = 0.0;
      options.temperature = 1.0;
      options.alpha_presence = 2.0;
      options.alpha_frequency = 5.0;
      options.alpha_decay = 0.5;

      TEST_EQ(rwkv7_server::sample_repetition_temperature_topk_topp(logits, state, options), 0);
      auto penalties = copy_penalties(state);
      TEST_NEAR(penalties.at(0), 2.0, 1e-4);
      TEST_NEAR(penalties.at(1), 0.0, 1e-6);

      TEST_EQ(rwkv7_server::sample_repetition_temperature_topk_topp(logits, state, options), 0);
      penalties = copy_penalties(state);
      TEST_NEAR(penalties.at(0), 6.0, 1e-4);
    }

    {
      auto state = rwkv7_server::make_sampler_penalties(4);
      auto logits = make_logits({1.0f, 2.0f, 3.0f, 4.0f});
      logits.rows = 2;
      TEST_THROW(rwkv7_server::sample_repetition_temperature_topk_topp(logits, state, rwkv7_server::GenerateOptions{}));
    }

    // Independent parameters must agree with isolated samplers using identical
    // RNG histories, including after a slot moves during compaction.
    {
      rwkv7_server::DeviceLogits logits;
      logits.rows = 3; logits.vocab_size = 4;
      rwkv_test::copy_host_to_device(std::vector<float>{2, 1, 0, -1, 2, 1, 0, -1, 2, 1, 0, -1},
                                    logits.values, "rows", "rows");
      auto batch = rwkv7_server::make_sampler_penalties(4, 3);
      std::vector<rwkv7_server::GenerateOptions> options(3);
      std::vector<rwkv7_server::SamplerPenaltyState> isolated;
      for (int i = 0; i < 3; ++i) {
        options[i].temperature = 0.2 + i;
        options[i].top_k = i + 2;
        options[i].top_p = 0.3 + i * 0.3;
        options[i].alpha_presence = 0.5 + i;
        options[i].alpha_frequency = 0.1 * i;
        options[i].alpha_decay = 0.7 + i * 0.1;
        isolated.push_back(rwkv7_server::make_sampler_penalties(4));
        rwkv7_server::copy_sampler_row(batch, i, isolated.back(), 0);
      }
      auto single = make_logits({2, 1, 0, -1});
      for (int step = 0; step < 32; ++step) {
        auto tokens = rwkv7_server::sample_rows(logits, batch, options, {step, step, step});
        for (int i = 0; i < 3; ++i)
          TEST_EQ(tokens[i], rwkv7_server::sample_repetition_temperature_topk_topp(
              single, isolated[i], options[i]));
      }
      auto moved = rwkv7_server::make_sampler_penalties(4, 1);
      rwkv7_server::copy_sampler_row(batch, 2, moved, 0);
      TEST_EQ(rwkv7_server::sample_rows(single, moved, {options[2]}, {32})[0],
              rwkv7_server::sample_repetition_temperature_topk_topp(single, isolated[2], options[2]));
      TEST_THROW(rwkv7_server::sample_rows(logits, batch, {options[0]}, {0}));
    }

    std::cout << "rwkv_sampler_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_sampler_test failed: " << e.what() << "\n";
    return 1;
  }
}
