#pragma once

#include <cstdint>
#include <vector>

#include "rwkv/runtime/rwkv_server_backend.hpp"

namespace rwkv7_server {

struct SamplerPenaltyState {
  rwkv7_fast_v4::DeviceBuffer<std::uint8_t> row_options;
  int batch_size = 0;
  int vocab_size = 0;
  rwkv7_fast_v4::DeviceBuffer<float> penalties;
  rwkv7_fast_v4::DeviceBuffer<float> probs;
  rwkv7_fast_v4::DeviceBuffer<int> outputs;
  rwkv7_fast_v4::DeviceBuffer<std::uint8_t> rand_states;
};

SamplerPenaltyState make_sampler_penalties(int vocab_size, int batch_size = 1);

int sample_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options);

std::vector<int> sample_batch_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options);

std::vector<int> sample_rows(
    const DeviceLogits& logits, SamplerPenaltyState& penalties,
    const std::vector<GenerateOptions>& options, const std::vector<int>& steps);

void copy_sampler_row(const SamplerPenaltyState& src, int src_row,
                      SamplerPenaltyState& dst, int dst_row);

}  // namespace rwkv7_server
