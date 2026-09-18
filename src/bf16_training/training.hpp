#pragma once

#include "miss.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime.hpp"
#include "rwkv/runtime/rwkv_state_tuning.hpp"

namespace rwkv_bf16_training {

// Reuse the existing standard state-passing operator, without a local copy.
// All training call sites explicitly select IoType::BF16.
using rwkv7_state_tuning::IoType;
using rwkv7_state_tuning::wkv_backward_input;
using rwkv7_state_tuning::wkv_forward;
using rwkv7_state_tuning::wkv_tape_elements;
using rwkv7_state_tuning::WkvShape;
using rwkv7_state_tuning::WkvTapeView;

// Forward Y=X*W^T, frozen W=[N,K]. Computes only dX=dY*W.
void linear_orig_backward_input_bf16(cudaStream_t stream, int rows,
                                     int in_features, int out_features,
                                     const bf16 *dy,
                                     const bf16 *frozen_weight_orig, bf16 *dx);

// Forward Y=X*W, runtime frozen W=[K,N]. Computes only dX=dY*W^T.
void linear_runtime_backward_input_bf16(cudaStream_t stream, int rows,
                                        int in_features, int out_features,
                                        const bf16 *dy,
                                        const bf16 *frozen_weight_runtime,
                                        bf16 *dx);

void layer_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    const bf16 *x, const bf16 *frozen_weight,
                                    const bf16 *dy, bf16 *dx, float eps);
void group_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    int groups, const bf16 *x,
                                    const bf16 *frozen_weight, const bf16 *dy,
                                    bf16 *dx, float eps);

// Backward of branches of x + (prev-x)*mix. d_shift may be null.
void time_mix_backward_input_bf16(cudaStream_t stream, int batch, int time,
                                  int channels, int branches,
                                  const bf16 *const *branch_grads,
                                  const bf16 *const *frozen_mixes, bf16 *dx,
                                  bf16 *d_shift);

void relu_square_backward_input_bf16(cudaStream_t stream, const bf16 *x,
                                     const bf16 *dy, bf16 *dx,
                                     std::size_t elements);
void sigmoid_backward_input_bf16(cudaStream_t stream, const bf16 *sigmoid_y,
                                 const bf16 *dy, bf16 *dx,
                                 std::size_t elements);
void tanh_backward_input_bf16(cudaStream_t stream, const bf16 *tanh_y,
                              const bf16 *dy, bf16 *dx, std::size_t elements);
void add_bf16(cudaStream_t stream, const bf16 *a, const bf16 *b, bf16 *out,
              std::size_t elements);
void add_inplace_bf16(cudaStream_t stream, bf16 *destination,
                      const bf16 *source, std::size_t elements);

// Non-owning views used by the state-only block executor. All matrices are in
// the same runtime layouts as rwkv7_fast_v4: the four attention projections
// and FFN key are original [N,K]; low-rank and FFN value matrices are runtime
// [K,N]. Biases are omitted because frozen affine biases do not affect dX.
struct FrozenBlockWeights {
  rwkv_bf16_miss::Block
      miss{}; // request/training-owned view, never LayerWeights
  int channels = 0;
  int heads = 0;
  int ffn = 0;
  int rank_w = 0;
  int rank_a = 0;
  int rank_g = 0;
  int rank_v = 0;
  const bf16 *ln1_weight = nullptr;
  const bf16 *ln1_bias = nullptr;
  const bf16 *ln2_weight = nullptr;
  const bf16 *ln2_bias = nullptr;
  const bf16 *mix_r = nullptr;
  const bf16 *mix_w = nullptr;
  const bf16 *mix_k = nullptr;
  const bf16 *mix_v = nullptr;
  const bf16 *mix_a = nullptr;
  const bf16 *mix_g = nullptr;
  const bf16 *receptance = nullptr;
  const bf16 *key = nullptr;
  const bf16 *value = nullptr;
  const bf16 *output = nullptr;
  const bf16 *w1 = nullptr;
  const bf16 *w2 = nullptr;
  const bf16 *w0 = nullptr;
  const bf16 *a1 = nullptr;
  const bf16 *a2 = nullptr;
  const bf16 *a0 = nullptr;
  const bf16 *g1 = nullptr;
  const bf16 *g2 = nullptr;
  const bf16 *v1 = nullptr;
  const bf16 *v2 = nullptr;
  const bf16 *v0 = nullptr;
  const bf16 *k_k = nullptr;
  const bf16 *k_a = nullptr;
  const bf16 *r_k = nullptr;
  const bf16 *att_group_norm_weight = nullptr;
  const bf16 *att_group_norm_bias = nullptr;
  const bf16 *ffn_mix = nullptr;
  const bf16 *ffn_key = nullptr;
  const bf16 *ffn_value = nullptr;
};

struct FrozenModelView {
  int layers = 0;
  int channels = 0;
  int heads = 0;
  int vocab = 0;
  const std::uint16_t *cpu_emb_ln0_bf16 = nullptr;
  std::size_t cpu_emb_ln0_elements = 0;
  const bf16 *ln_out_weight = nullptr;
  const bf16 *ln_out_bias = nullptr;
  const bf16 *head_weight_orig = nullptr;
  std::vector<FrozenBlockWeights> blocks;
};

// Activations required by one block backward. The state-tuning forward
// recorder owns these buffers; no activation is attached to inference state.
struct BlockTapeView {
  std::array<float *, rwkv_bf16_miss::TargetCount> miss_reduced{};
  int batch = 0;
  int time = 0;
  bf16 *x = nullptr;
  bf16 *ln1 = nullptr;
  bf16 *r = nullptr;
  bf16 *raw_w = nullptr;
  bf16 *k = nullptr;
  bf16 *v_base = nullptr;
  bf16 *v_first = nullptr;
  bf16 *v = nullptr;
  bf16 *alpha = nullptr;
  bf16 *neg_kk = nullptr;
  bf16 *wkv_k = nullptr;
  bf16 *kka = nullptr;
  bf16 *w1_tanh = nullptr;
  bf16 *g1_sigmoid = nullptr;
  bf16 *v_gate = nullptr;
  bf16 *wkv_y = nullptr;
  bf16 *att_group_norm = nullptr;
  bf16 *att_gate = nullptr;
  bf16 *x_after_att = nullptr;
  bf16 *ln2 = nullptr;
  bf16 *ffn_hid = nullptr;
  WkvTapeView wkv;
  // Optional replay mode: owned initial state and shared final-state scratch.
  // wkv storage may be shared by blocks executed sequentially on one stream.
  float *wkv_replay_initial = nullptr;
  float *wkv_replay_final = nullptr;
};

std::size_t
block_forward_workspace_bf16_elements(int batch, int time,
                                      const FrozenBlockWeights &weights);

// Reuses the inference BF16 kernels and records only the values required by
// block_backward_state_only. initial/final WKV states are FP32 [B,H,K,V].
// v_first is a shared [B,T,C] buffer: layer 0 writes it, later layers read it.
void block_forward_state_tuning_bf16(
    cudaStream_t stream, int batch, int time, const bf16 *input,
    bf16 *attention_shift, bf16 *ffn_shift, const float *initial_state,
    float *final_state, bf16 *v_first, bf16 *output,
    const FrozenBlockWeights &frozen_weights, BlockTapeView &tape,
    bf16 *workspace, std::size_t workspace_elements, bool layer_zero);

struct BlockBackwardState {
  const float *final_state_grad = nullptr; // optional [B,H,K,V]
  float *initial_state_grad = nullptr;     // required [B,H,K,V]
  bf16 *att_shift_grad = nullptr;          // optional [B,C]
  bf16 *ffn_shift_grad = nullptr;          // optional [B,C]
  bf16 *v_first_grad = nullptr;            // in/out accumulator [B,T,C]
  const bf16 *final_att_shift_grad = nullptr;
  const bf16 *final_ffn_shift_grad = nullptr;
};

std::size_t
block_backward_workspace_bf16_elements(int batch, int time,
                                       const FrozenBlockWeights &weights);

// grad_out/grad_in are [B,T,C]. The model executor calls this from layer N to
// layer 0 and ping-pongs those two long-lived buffers. workspace is ephemeral
// per-block scratch and is reused for every layer.
void block_backward_state_only(cudaStream_t stream, const bf16 *grad_out,
                               bf16 *grad_in, BlockBackwardState state_grad,
                               const BlockTapeView &tape,
                               const FrozenBlockWeights &frozen_weights,
                               bf16 *workspace, std::size_t workspace_elements,
                               bool layer_zero);

// Cross entropy over every row. loss must point to one FP32 device scalar and
// is overwritten with the row mean times gradient_scale. gradient_scale
// weights valid tokens across chunks/samples; it is not loss scaling.
// The dense path requires
// ignore_index < 0 (dense, equal-length batches). d_logits is BF16.
void cross_entropy_forward_backward_bf16(cudaStream_t stream, int rows,
                                         int vocab, const bf16 *logits,
                                         const int *targets, int ignore_index,
                                         float *loss, bf16 *d_logits,
                                         float gradient_scale = 1.0f);

// Short-context model forward. input is the existing emb+ln0 BF16 table
// result [B,T,C]. time_state is FP32 [L,H,K,V] and is broadcast across B.
// Without carried_states, shifts is zeroed and time_state is broadcast.
// With carried_states [L,B,H,K,V], both states and shifts are advanced in
// place. Tapes and all tape storage are caller-owned.
const bf16 *model_forward_state_tuning_bf16(
    cudaStream_t stream, int layers, int batch, int time,
    const FrozenBlockWeights *weights, BlockTapeView *tapes, const bf16 *input,
    const float *time_state, bf16 *shifts, const bf16 *ln_out_weight,
    const bf16 *ln_out_bias, const bf16 *head_weight_orig, int vocab,
    bf16 *activation_ping, bf16 *activation_pong, float *state_batch,
    float *state_final, bf16 *block_workspace,
    std::size_t block_workspace_elements, bf16 *final_normalized, bf16 *logits,
    float *carried_states = nullptr);

void add_last_shift_gradient(cudaStream_t stream, int batch, int time,
                             int channels, bf16 *dx, const bf16 *dshift);

// Frozen output head backward: logits -> ln_out -> last block. Returns the
// same ping/pong alias contract as model_backward_state_only.
const bf16 *model_output_backward_state_only(
    cudaStream_t stream, int layers, int batch, int time,
    const FrozenBlockWeights *weights, const BlockTapeView *tapes,
    BlockBackwardState *state_gradients, const bf16 *final_block_output,
    const bf16 *ln_out_weight, const bf16 *head_weight_orig, int vocab,
    const bf16 *d_logits, bf16 *gradient_ping, bf16 *gradient_pong,
    bf16 *block_workspace, std::size_t block_workspace_elements);

// Reverse-order block scheduler. gradient_ping and gradient_pong are the only
// two model-sized [B,T,C] gradient buffers. The return value aliases one of
// them and is the gradient at the embedding side.
const bf16 *model_backward_state_only(
    cudaStream_t stream, int layers, const FrozenBlockWeights *weights,
    const BlockTapeView *tapes, BlockBackwardState *state_gradients,
    const bf16 *output_gradient, bf16 *gradient_ping, bf16 *gradient_pong,
    bf16 *block_workspace, std::size_t block_workspace_elements);

// Reduces [B,H,K,V] to the trainable [H,K,V]. Set accumulate=true when
// accumulating micro-batches.
void reduce_state_gradient_f32(cudaStream_t stream, int batch, int heads,
                               int head_size, const float *per_batch_gradient,
                               float *state_gradient, bool accumulate);

// Each contiguous 64x64 state matrix is orthogonalized independently.
struct MuonConfig {
  float learning_rate = 0.02f;
  float momentum = 0.95f;
  int ns_steps = 5;
  bool nesterov = true;
  float weight_decay = 0.0f;
};

void muon_update_state_f32(cudaStream_t stream, float *time_state,
                           float *gradient, float *momentum,
                           std::size_t elements, const MuonConfig &config,
                           bool zero_gradient);

struct AdamConfig {
  float learning_rate = 1.0e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1.0e-8f;
  float weight_decay = 0.0f;
};

// The optimizer owns no model weights: only time_state, its gradient and the
// FP32 Adam moments are accepted.
void adam_update_state_f32(cudaStream_t stream, float *time_state,
                           float *gradient, float *moment1, float *moment2,
                           std::size_t elements, std::uint64_t step,
                           const AdamConfig &config, bool zero_gradient);

void save_state_checkpoint_pth(const std::string &path, cudaStream_t stream,
                               int layers, int heads, int head_size,
                               const float *runtime_time_state);

void save_state_checkpoint_pth_host(
    const std::string &path, int layers, int heads, int head_size,
    const std::vector<float> &runtime_time_state);

} // namespace rwkv_bf16_training
