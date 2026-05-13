#pragma once

#include <vector>

#include "rwkv_server_backend.hpp"

namespace rwkv7_server {

struct SamplerPenaltyState {
  std::vector<double> penalties;
};

SamplerPenaltyState make_sampler_penalties(int vocab_size);

int sample_repetition_temperature_topk_topp(
    const std::vector<float>& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options);

void update_sampler_penalties(
    int token,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options);

}  // namespace rwkv7_server
