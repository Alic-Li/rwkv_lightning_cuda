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

SamplerPenaltyState make_sampler_penalties(int vocab_size) {
  if (vocab_size <= 0) {
    throw std::runtime_error("invalid sampler vocab size");
  }

  SamplerPenaltyState state;
  state.vocab_size = vocab_size;
  state.penalties.resize(static_cast<std::size_t>(vocab_size), "alloc sampler penalties");
  state.probs.resize(static_cast<std::size_t>(vocab_size), "alloc sampler probs");
  state.outputs.resize(1, "alloc sampler outputs");
  state.rand_states.resize(rwkv_sampling::rand_state_bytes(1), "alloc sampler rand states");
  state.penalties.zero("zero sampler penalties");

  rwkv7_fast_v4::check_cuda(
      rwkv_sampling::setup_rand_raw(state.rand_states.p, make_sampler_seed(), 1, 0),
      "setup sampler rand state");
  rwkv7_fast_v4::check_cuda(cudaStreamSynchronize(0), "sync sampler rand state");
  return state;
}

int sample_repetition_temperature_topk_topp(
    const DeviceLogits& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options) {
  if (logits.rows != 1) {
    throw std::runtime_error("sampler currently expects a single logits row");
  }
  if (logits.vocab_size <= 0 || logits.values.p == nullptr) {
    throw std::runtime_error("empty device logits");
  }
  if (penalties.vocab_size != logits.vocab_size) {
    throw std::runtime_error("sampler logits size does not match sampler state");
  }

  rwkv7_fast_v4::check_cuda(
      rwkv_sampling::batch_sampling_repetition_temperature_topk_topp_raw(
          logits.values.p,
          penalties.penalties.p,
          penalties.outputs.p,
          penalties.rand_states.p,
          penalties.probs.p,
          1,
          1,
          logits.vocab_size,
          options.alpha_presence,
          options.alpha_frequency,
          options.alpha_decay,
          options.temperature,
          options.top_k,
          options.top_p,
          0),
      "launch sampler kernel");

  int token = 0;
  rwkv7_fast_v4::check_cuda(
      cudaMemcpy(&token, penalties.outputs.p, sizeof(token), cudaMemcpyDeviceToHost),
      "copy sampled token to host");
  return token;
}

}  // namespace rwkv7_server
