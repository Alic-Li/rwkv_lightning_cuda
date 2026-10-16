#include "rwkv/inference/rwkv_sampler.hpp"

#include <chrono>
#include <random>
#include <stdexcept>

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

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

std::vector<int> sample_rows(
    const DeviceLogits& logits, SamplerPenaltyState& penalties,
    const std::vector<GenerateOptions>& options, const std::vector<int>& steps) {
  if (logits.rows != penalties.batch_size || logits.vocab_size != penalties.vocab_size ||
      options.size() != static_cast<size_t>(logits.rows) || steps.size() != options.size())
    throw std::invalid_argument("row sampler shape mismatch");
  std::vector<rwkv_sampling::RowSamplingOptions> rows;
  for (size_t i = 0; i < options.size(); ++i) {
    const auto& o = options[i];
    if (!(o.temperature >= 0.001 && o.temperature <= 1000))
      throw std::invalid_argument("temperature must be in [0.001, 1000]");
    int k = o.top_k <= 0 || o.top_k > logits.vocab_size ? logits.vocab_size : o.top_k;
    double p = o.top_p < 0 || o.top_p > 1 ? 1 : o.top_p;
    if (p == 0) { k = 1; p = 1; }
    const int step = o.force_reasoning_token_offset + steps[i];
    rows.push_back({float(o.alpha_presence), float(o.alpha_frequency), float(o.alpha_decay),
                    float(1.44269504088896340736 / o.temperature), k, float(p),
                    o.force_reasoning && step >= 1 && step <= 2});
  }
  penalties.row_options.resize(rows.size() * sizeof(rows[0]), "allocate row sampling options");
  rwkv7_fast_v4::check_cuda(cudaMemcpy(penalties.row_options.p, rows.data(),
      rows.size() * sizeof(rows[0]), cudaMemcpyHostToDevice), "upload row sampling options");
  rwkv7_fast_v4::check_cuda(rwkv_sampling::sample_rows_raw(logits.values.p,
      penalties.penalties.p, penalties.outputs.p, penalties.rand_states.p, penalties.probs.p,
      logits.rows, logits.vocab_size,
      reinterpret_cast<const rwkv_sampling::RowSamplingOptions*>(penalties.row_options.p), 0),
      "sample independent rows");
  std::vector<int> tokens(logits.rows);
  rwkv7_fast_v4::check_cuda(cudaMemcpy(tokens.data(), penalties.outputs.p,
      tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), "download row samples");
  return tokens;
}

void copy_sampler_row(const SamplerPenaltyState& src, int src_row,
                      SamplerPenaltyState& dst, int dst_row) {
  if (src.vocab_size != dst.vocab_size || src_row < 0 || src_row >= src.batch_size ||
      dst_row < 0 || dst_row >= dst.batch_size)
    throw std::invalid_argument("invalid sampler row copy");
  rwkv7_fast_v4::check_cuda(cudaMemcpy(dst.penalties.p + size_t(dst_row) * dst.vocab_size,
      src.penalties.p + size_t(src_row) * src.vocab_size,
      src.vocab_size * sizeof(float), cudaMemcpyDeviceToDevice), "move row penalties");
  const auto bytes = rwkv_sampling::rand_state_bytes(1);
  rwkv7_fast_v4::check_cuda(cudaMemcpy(dst.rand_states.p + dst_row * bytes,
      src.rand_states.p + src_row * bytes, bytes, cudaMemcpyDeviceToDevice), "move row RNG");
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
