// Copied from src/state_tuning; independent BF16 training implementation.
#include "training.hpp"

#include "runtime.hpp"

#include <stdexcept>

#include "ops.hpp"

namespace rwkv_bf16_training {
namespace {

constexpr int kThreads = 256;
constexpr float kLnEps = 1.0e-5f;

__global__ void broadcast_state_kernel(int B, std::size_t lane,
                                       const float *__restrict__ source,
                                       float *__restrict__ destination) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(B) * lane;
  if (i < total)
    destination[i] = source[i % lane];
}

int blocks(std::size_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace

const bf16 *model_forward_state_tuning_bf16(
    cudaStream_t stream, int layers, int B, int T,
    const FrozenBlockWeights *weights, BlockTapeView *tapes, const bf16 *input,
    const float *time_state, bf16 *shifts, const bf16 *ln_out_weight,
    const bf16 *ln_out_bias, const bf16 *head_weight_orig, int vocab,
    bf16 *activation_ping, bf16 *activation_pong, float *state_batch,
    float *state_final, bf16 *block_workspace,
    std::size_t block_workspace_elements, bf16 *final_normalized, bf16 *logits,
    float *carried_states) {
  if (layers <= 0 || B <= 0 || T <= 0 || vocab <= 0 || !weights || !tapes ||
      !input || !time_state || !shifts || !ln_out_weight || !ln_out_bias ||
      !head_weight_orig || !activation_ping || !activation_pong ||
      !state_batch || !state_final || !block_workspace || !final_normalized ||
      !logits) {
    throw std::invalid_argument("invalid state-tuning model forward arguments");
  }
  const int C = weights[0].channels;
  const int H = weights[0].heads;
  if (H <= 0 || C <= 0 || C % H)
    throw std::invalid_argument("invalid state-tuning model head dimensions");
  const int N = C / H;
  if (N != 64)
    throw std::invalid_argument("state-tuning model requires N=64");
  const std::size_t row_elements = static_cast<std::size_t>(B) * T * C;
  const std::size_t state_lane = static_cast<std::size_t>(H) * N * N;
  if (!carried_states)
    check(cudaMemsetAsync(
        shifts, 0, static_cast<std::size_t>(layers) * 2 * B * C * sizeof(bf16),
        stream));
  check(cudaMemcpyAsync(activation_ping, input, row_elements * sizeof(bf16),
                        cudaMemcpyDeviceToDevice, stream));
  const bf16 *current = activation_ping;
  bf16 *next = activation_pong;
  for (int layer = 0; layer < layers; ++layer) {
    if (weights[layer].channels != C || weights[layer].heads != H) {
      throw std::invalid_argument("all forward blocks must share C and H");
    }
    const std::size_t required =
        block_forward_workspace_bf16_elements(B, T, weights[layer]);
    if (required > block_workspace_elements) {
      throw std::invalid_argument("model forward block workspace is too small");
    }
    if (carried_states) {
      check(cudaMemcpyAsync(
          state_batch, carried_states + layer * B * state_lane,
          B * state_lane * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else
      broadcast_state_kernel<<<blocks(static_cast<std::size_t>(B) * state_lane),
                               kThreads, 0, stream>>>(
          B, state_lane,
          time_state + static_cast<std::size_t>(layer) * state_lane,
          state_batch);
    bf16 *att_shift = shifts + static_cast<std::size_t>(layer) * 2 * B * C;
    bf16 *ffn_shift = att_shift + static_cast<std::size_t>(B) * C;
    block_forward_state_tuning_bf16(
        stream, B, T, current, att_shift, ffn_shift, state_batch, state_final,
        tapes[0].v_first, next, weights[layer], tapes[layer], block_workspace,
        block_workspace_elements, layer == 0);
    if (carried_states) {
      check(cudaMemcpyAsync(carried_states + layer * B * state_lane,
                            state_final, B * state_lane * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    }
    current = next;
    next = next == activation_ping ? activation_pong : activation_ping;
  }
  rwkv7_v3a_layer_norm_bf16_launch(stream, B * T, C, current, ln_out_weight,
                                   ln_out_bias, final_normalized, kLnEps);
  rwkv7_v3a_linear_bf16_orig_launch(stream, B * T, C, vocab, final_normalized,
                                    head_weight_orig, logits);
  return current;
}

} // namespace rwkv_bf16_training
