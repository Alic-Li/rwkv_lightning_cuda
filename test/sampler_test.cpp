#include <iostream>
#include <vector>

#include "rwkv_sampler.hpp"
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

    std::cout << "rwkv_sampler_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_sampler_test failed: " << e.what() << "\n";
    return 1;
  }
}
