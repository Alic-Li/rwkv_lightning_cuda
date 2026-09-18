// Copied from src/state_tuning; independent BF16 training implementation.
#include "training.hpp"

#include <algorithm>
#include <stdexcept>

namespace rwkv_bf16_training {

const bf16 *model_output_backward_state_only(
    cudaStream_t stream, int layers, int batch, int time,
    const FrozenBlockWeights *weights, const BlockTapeView *tapes,
    BlockBackwardState *state_gradients, const bf16 *final_block_output,
    const bf16 *ln_out_weight, const bf16 *head_weight_orig, int vocab,
    const bf16 *d_logits, bf16 *gradient_ping, bf16 *gradient_pong,
    bf16 *block_workspace, std::size_t block_workspace_elements) {
  if (layers <= 0 || batch <= 0 || time <= 0 || !weights || !tapes ||
      !state_gradients || !final_block_output || !ln_out_weight ||
      !head_weight_orig || vocab <= 0 || !d_logits || !gradient_ping ||
      !gradient_pong) {
    throw std::invalid_argument("invalid model output backward arguments");
  }
  const int rows = batch * time;
  const int channels = weights[0].channels;
  linear_orig_backward_input_bf16(stream, rows, channels, vocab, d_logits,
                                  head_weight_orig, gradient_ping);
  layer_norm_backward_input_bf16(stream, rows, channels, final_block_output,
                                 ln_out_weight, gradient_ping, gradient_pong,
                                 1.0e-5f);
  return model_backward_state_only(
      stream, layers, weights, tapes, state_gradients, gradient_pong,
      gradient_ping, gradient_pong, block_workspace, block_workspace_elements);
}

const bf16 *model_backward_state_only(
    cudaStream_t stream, int layers, const FrozenBlockWeights *weights,
    const BlockTapeView *tapes, BlockBackwardState *state_gradients,
    const bf16 *output_gradient, bf16 *gradient_ping, bf16 *gradient_pong,
    bf16 *block_workspace, std::size_t block_workspace_elements) {
  if (layers <= 0 || !weights || !tapes || !state_gradients ||
      !output_gradient || !gradient_ping || !gradient_pong ||
      !block_workspace) {
    throw std::invalid_argument("invalid model backward arguments");
  }
  const int rows = tapes[0].batch * tapes[0].time;
  const int channels = weights[0].channels;
  if (rows <= 0 || channels <= 0) {
    throw std::invalid_argument("invalid model backward shape");
  }
  const std::size_t gradient_elements =
      static_cast<std::size_t>(rows) * channels;
  if (output_gradient != gradient_ping) {
    const cudaError_t copy_error = cudaMemcpyAsync(
        gradient_ping, output_gradient, gradient_elements * sizeof(bf16),
        cudaMemcpyDeviceToDevice, stream);
    if (copy_error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(copy_error));
    }
  }

  bf16 *v_first_gradient = state_gradients[0].v_first_grad;
  if (layers > 1 && !v_first_gradient) {
    throw std::invalid_argument(
        "multi-layer backward requires v_first gradient");
  }
  if (v_first_gradient) {
    const cudaError_t zero_error = cudaMemsetAsync(
        v_first_gradient, 0, gradient_elements * sizeof(bf16), stream);
    if (zero_error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(zero_error));
    }
  }

  const bf16 *current = gradient_ping;
  bf16 *next = gradient_pong;
  for (int layer = layers - 1; layer >= 0; --layer) {
    if (tapes[layer].batch * tapes[layer].time != rows ||
        weights[layer].channels != channels) {
      throw std::invalid_argument("all model backward layers must share B,T,C");
    }
    if (layers > 1)
      state_gradients[layer].v_first_grad = v_first_gradient;
    const std::size_t layer_workspace = block_backward_workspace_bf16_elements(
        tapes[layer].batch, tapes[layer].time, weights[layer]);
    if (layer_workspace > block_workspace_elements) {
      throw std::invalid_argument("model block workspace is too small");
    }
    block_backward_state_only(stream, current, next, state_gradients[layer],
                              tapes[layer], weights[layer], block_workspace,
                              block_workspace_elements, layer == 0);
    current = next;
    next = next == gradient_ping ? gradient_pong : gradient_ping;
  }
  return current;
}

} // namespace rwkv_bf16_training
