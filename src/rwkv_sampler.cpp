#include "rwkv_sampler.hpp"

#include <chrono>
#include <random>
#include <stdexcept>

#include <cuda_runtime.h>

#include "utils/sampling.h"

namespace rwkv7_server {
namespace {

std::uint64_t make_sampler_seed() {
  std::random_device rd;
  const auto time_seed = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd()) ^ time_seed;
}

}  // namespace

SamplerPenaltyState make_sampler_penalties(int vocab_size, int batch_size) {
  if (vocab_size <= 0) {
    throw std::runtime_error("invalid sampler vocab size");
  }
  if (batch_size <= 0) {
    throw std::runtime_error("invalid sampler batch size");
  }

  SamplerPenaltyState state;
  state.batch_size = batch_size;
  state.vocab_size = vocab_size;
  state.penalties.resize(static_cast<std::size_t>(batch_size) * vocab_size, "alloc sampler penalties");
  state.probs.resize(static_cast<std::size_t>(batch_size) * vocab_size, "alloc sampler probs");
  state.outputs.resize(batch_size, "alloc sampler outputs");
  state.rand_states.resize(rwkv_sampling::rand_state_bytes(batch_size), "alloc sampler rand states");
  state.penalties.zero("zero sampler penalties");

  rwkv7_fast_v4::check_cuda(
      rwkv_sampling::setup_rand_raw(state.rand_states.p, make_sampler_seed(), batch_size, 0),
      "setup sampler rand state");
  rwkv7_fast_v4::check_cuda(cudaStreamSynchronize(0), "sync sampler rand state");
  return state;
}

std::vector<int> sample_batch_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options) {
  if (logits.rows <= 0) {
    throw std::runtime_error("sampler expects at least one logits row");
  }
  if (logits.vocab_size <= 0 || logits.values.p == nullptr) {
    throw std::runtime_error("empty device logits");
  }
  if (penalties.vocab_size != logits.vocab_size) {
    throw std::runtime_error("sampler logits size does not match sampler state");
  }
  if (penalties.batch_size != logits.rows) {
    throw std::runtime_error("sampler batch size does not match logits rows");
  }

  const bool use_repetition_penalty =
      options.alpha_presence != 0.0 || options.alpha_frequency != 0.0;
  if (use_repetition_penalty) {
    rwkv7_fast_v4::check_cuda(
        rwkv_sampling::batch_sampling_repetition_temperature_topk_topp_raw(
            logits.values.p,
            penalties.penalties.p,
            penalties.outputs.p,
            penalties.rand_states.p,
            penalties.probs.p,
            logits.rows,
            1,
            logits.vocab_size,
            options.alpha_presence,
            options.alpha_frequency,
            options.alpha_decay,
            options.temperature,
            options.top_k,
            options.top_p,
            0),
        "launch sampler repetition kernel");
  } else {
    rwkv7_fast_v4::check_cuda(
        rwkv_sampling::batch_sampling_temperature_topk_topp_raw(
            logits.values.p,
            penalties.outputs.p,
            penalties.rand_states.p,
            penalties.probs.p,
            logits.rows,
            1,
            logits.vocab_size,
            options.temperature,
            options.top_k,
            options.top_p,
            0),
        "launch sampler kernel");
  }

  std::vector<int> tokens(static_cast<std::size_t>(logits.rows));
  rwkv7_fast_v4::check_cuda(
      cudaMemcpy(
          tokens.data(),
          penalties.outputs.p,
          static_cast<std::size_t>(logits.rows) * sizeof(int),
          cudaMemcpyDeviceToHost),
      "copy sampled tokens to host");
  return tokens;
}

int sample_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options) {
  if (logits.rows != 1) {
    throw std::runtime_error("sampler currently expects a single logits row");
  }
  return sample_batch_repetition_temperature_topk_topp(logits, penalties, options).front();
}

}  // namespace rwkv7_server
