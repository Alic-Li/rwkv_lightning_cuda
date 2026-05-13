#pragma once

#include <cstdint>
#include <vector>

#include "rwkv_server_backend.hpp"

namespace rwkv7_server {

struct SamplerPenaltyState {
  int vocab_size = 0;
  rwkv7_fast_v4::DeviceBuffer<float> penalties;
  rwkv7_fast_v4::DeviceBuffer<float> probs;
  rwkv7_fast_v4::DeviceBuffer<int> outputs;
  rwkv7_fast_v4::DeviceBuffer<std::uint8_t> rand_states;
};

SamplerPenaltyState make_sampler_penalties(int vocab_size);

int sample_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options);

}  // namespace rwkv7_server
