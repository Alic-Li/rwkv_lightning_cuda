#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"

namespace rwkv7_state_tuning {

// This is a deliberately small pointer-based CUDA API. It has no torch
// dependency and, by construction, has nowhere to return a weight gradient.
enum class IoType : std::uint8_t { F16, BF16 };

struct WkvShape {
  int batch = 0;
  int time = 0;
  int heads = 0;
  int head_size = 64;
};

// Exact short-context tape: FP32 [B,H,T,K,V]. A later checkpoint/recompute
// implementation can change the storage policy without changing the API.
struct WkvTapeView {
  float *state_before = nullptr;
  std::size_t state_before_elements = 0;
};

std::size_t wkv_tape_elements(const WkvShape &shape);

// r/w/k/v/a/b/y are [B,T,H,N]. w is the raw clamp-w input including w0.
// Initial/final states use the inference physical ABI [B,H,K,V].
void wkv_forward(cudaStream_t stream, IoType io_type, const WkvShape &shape,
                 const float *initial_state, const void *r, const void *w,
                 const void *k, const void *v, const void *a, const void *b,
                 void *y, float *final_state, WkvTapeView tape);

// Input gradients only. ds_final may be null (zero). ds_initial is FP32;
// sequence gradients have io_type. No frozen-weight gradient is computed.
void wkv_backward_input(cudaStream_t stream, IoType io_type,
                        const WkvShape &shape, const void *r, const void *w,
                        const void *k, const void *v, const void *a,
                        const void *b, const void *dy, const float *ds_final,
                        WkvTapeView tape, float *ds_initial, void *dr, void *dw,
                        void *dk, void *dv, void *da, void *db);

// Forward Y=X*W^T, frozen W=[N,K]. Computes only dX=dY*W.
void linear_orig_backward_input_f16(cudaStream_t stream, int rows,
                                    int in_features, int out_features,
                                    const half *dy,
                                    const half *frozen_weight_orig, half *dx);

// Forward Y=X*W, runtime frozen W=[K,N]. Computes only dX=dY*W^T.
void linear_runtime_backward_input_f16(cudaStream_t stream, int rows,
                                       int in_features, int out_features,
                                       const half *dy,
                                       const half *frozen_weight_runtime,
                                       half *dx);

void layer_norm_backward_input_f16(cudaStream_t stream, int rows, int channels,
                                   const half *x, const half *frozen_weight,
                                   const half *dy, half *dx, float eps);
void group_norm_backward_input_f16(cudaStream_t stream, int rows, int channels,
                                   int groups, const half *x,
                                   const half *frozen_weight, const half *dy,
                                   half *dx, float eps);

// Backward of branches of x + (prev-x)*mix. d_shift may be null.
void time_mix_backward_input_f16(cudaStream_t stream, int batch, int time,
                                 int channels, int branches,
                                 const half *const *branch_grads,
                                 const half *const *frozen_mixes, half *dx,
                                 half *d_shift);

void relu_square_backward_input_f16(cudaStream_t stream, const half *x,
                                    const half *dy, half *dx,
                                    std::size_t elements);
void sigmoid_backward_input_f16(cudaStream_t stream, const half *sigmoid_y,
                                const half *dy, half *dx, std::size_t elements);
void tanh_backward_input_f16(cudaStream_t stream, const half *tanh_y,
                             const half *dy, half *dx, std::size_t elements);
void add_f16(cudaStream_t stream, const half *a, const half *b, half *out,
             std::size_t elements);
void add_inplace_f16(cudaStream_t stream, half *destination, const half *source,
                     std::size_t elements);

// Non-owning views used by the state-only block executor. All matrices are in
// the same runtime layouts as rwkv7_fast_v4: the four attention projections
// and FFN key are original [N,K]; low-rank and FFN value matrices are runtime
// [K,N]. Biases are omitted because frozen affine biases do not affect dX.
struct FrozenBlockWeights {
  int channels = 0;
  int heads = 0;
  int ffn = 0;
  int rank_w = 0;
  int rank_a = 0;
  int rank_g = 0;
  int rank_v = 0;
  const half *ln1_weight = nullptr;
  const half *ln1_bias = nullptr;
  const half *ln2_weight = nullptr;
  const half *ln2_bias = nullptr;
  const half *mix_r = nullptr;
  const half *mix_w = nullptr;
  const half *mix_k = nullptr;
  const half *mix_v = nullptr;
  const half *mix_a = nullptr;
  const half *mix_g = nullptr;
  const half *receptance = nullptr;
  const half *key = nullptr;
  const half *value = nullptr;
  const half *output = nullptr;
  const half *w1 = nullptr;
  const half *w2 = nullptr;
  const half *w0 = nullptr;
  const half *a1 = nullptr;
  const half *a2 = nullptr;
  const half *a0 = nullptr;
  const half *g1 = nullptr;
  const half *g2 = nullptr;
  const half *v1 = nullptr;
  const half *v2 = nullptr;
  const half *v0 = nullptr;
  const half *k_k = nullptr;
  const half *k_a = nullptr;
  const half *r_k = nullptr;
  const half *att_group_norm_weight = nullptr;
  const half *att_group_norm_bias = nullptr;
  const half *ffn_mix = nullptr;
  const half *ffn_key = nullptr;
  const half *ffn_value = nullptr;
};

struct FrozenModelView {
  int layers = 0;
  int channels = 0;
  int heads = 0;
  int vocab = 0;
  const std::uint16_t *cpu_emb_ln0_f16 = nullptr;
  std::size_t cpu_emb_ln0_elements = 0;
  const half *ln_out_weight = nullptr;
  const half *ln_out_bias = nullptr;
  const half *head_weight_orig = nullptr;
  std::vector<FrozenBlockWeights> blocks;
};

// Activations required by one block backward. The state-tuning forward
// recorder owns these buffers; no activation is attached to inference state.
struct BlockTapeView {
  int batch = 0;
  int time = 0;
  half *x = nullptr;
  half *ln1 = nullptr;
  half *r = nullptr;
  half *raw_w = nullptr;
  half *k = nullptr;
  half *v_base = nullptr;
  half *v_first = nullptr;
  half *v = nullptr;
  half *alpha = nullptr;
  half *neg_kk = nullptr;
  half *wkv_k = nullptr;
  half *kka = nullptr;
  half *w1_tanh = nullptr;
  half *g1_sigmoid = nullptr;
  half *v_gate = nullptr;
  half *wkv_y = nullptr;
  half *att_group_norm = nullptr;
  half *att_gate = nullptr;
  half *x_after_att = nullptr;
  half *ln2 = nullptr;
  half *ffn_hid = nullptr;
  WkvTapeView wkv;
};

std::size_t
block_forward_workspace_f16_elements(int batch, int time,
                                     const FrozenBlockWeights &weights);

// Reuses the inference FP16 kernels and records only the values required by
// block_backward_state_only. initial/final WKV states are FP32 [B,H,K,V].
// v_first is a shared [B,T,C] buffer: layer 0 writes it, later layers read it.
void block_forward_state_tuning_f16(
    cudaStream_t stream, int batch, int time, const half *input,
    half *attention_shift, half *ffn_shift, const float *initial_state,
    float *final_state, half *v_first, half *output,
    const FrozenBlockWeights &frozen_weights, BlockTapeView &tape,
    half *workspace, std::size_t workspace_elements, bool layer_zero);

struct BlockBackwardState {
  const float *final_state_grad = nullptr; // optional [B,H,K,V]
  float *initial_state_grad = nullptr;     // required [B,H,K,V]
  half *att_shift_grad = nullptr;          // optional [B,C]
  half *ffn_shift_grad = nullptr;          // optional [B,C]
  half *v_first_grad = nullptr;            // in/out accumulator [B,T,C]
  const half *final_att_shift_grad = nullptr;
  const half *final_ffn_shift_grad = nullptr;
};

std::size_t
block_backward_workspace_f16_elements(int batch, int time,
                                      const FrozenBlockWeights &weights);

// grad_out/grad_in are [B,T,C]. The model executor calls this from layer N to
// layer 0 and ping-pongs those two long-lived buffers. workspace is ephemeral
// per-block scratch and is reused for every layer.
void block_backward_state_only(cudaStream_t stream, const half *grad_out,
                               half *grad_in, BlockBackwardState state_grad,
                               const BlockTapeView &tape,
                               const FrozenBlockWeights &frozen_weights,
                               half *workspace, std::size_t workspace_elements,
                               bool layer_zero);

// Cross entropy over every row. loss must point to one FP32 device scalar and
// is overwritten with the row mean times gradient_scale (as is d_logits).
// The dense path requires
// ignore_index < 0 (dense, equal-length batches). d_logits is FP16.
void cross_entropy_forward_backward_f16(cudaStream_t stream, int rows,
                                        int vocab, const half *logits,
                                        const int *targets, int ignore_index,
                                        float *loss, half *d_logits,
                                        float gradient_scale = 1.0f);

// Short-context model forward. input is the existing emb+ln0 FP16 table
// result [B,T,C]. time_state is FP32 [L,H,K,V] and is broadcast across B.
// Without carried_states, shifts is zeroed and time_state is broadcast.
// With carried_states [L,B,H,K,V], both states and shifts are advanced in
// place. Tapes and all tape storage are caller-owned.
const half *model_forward_state_tuning_f16(
    cudaStream_t stream, int layers, int batch, int time,
    const FrozenBlockWeights *weights, BlockTapeView *tapes, const half *input,
    const float *time_state, half *shifts, const half *ln_out_weight,
    const half *ln_out_bias, const half *head_weight_orig, int vocab,
    half *activation_ping, half *activation_pong, float *state_batch,
    float *state_final, half *block_workspace,
    std::size_t block_workspace_elements, half *final_normalized, half *logits,
    float *carried_states = nullptr);

void add_last_shift_gradient(cudaStream_t stream, int batch, int time,
                             int channels, half *dx, const half *dshift);

// Frozen output head backward: logits -> ln_out -> last block. Returns the
// same ping/pong alias contract as model_backward_state_only.
const half *model_output_backward_state_only(
    cudaStream_t stream, int layers, int batch, int time,
    const FrozenBlockWeights *weights, const BlockTapeView *tapes,
    BlockBackwardState *state_gradients, const half *final_block_output,
    const half *ln_out_weight, const half *head_weight_orig, int vocab,
    const half *d_logits, half *gradient_ping, half *gradient_pong,
    half *block_workspace, std::size_t block_workspace_elements);

// Reverse-order block scheduler. gradient_ping and gradient_pong are the only
// two model-sized [B,T,C] gradient buffers. The return value aliases one of
// them and is the gradient at the embedding side.
const half *model_backward_state_only(
    cudaStream_t stream, int layers, const FrozenBlockWeights *weights,
    const BlockTapeView *tapes, BlockBackwardState *state_gradients,
    const half *output_gradient, half *gradient_ping, half *gradient_pong,
    half *block_workspace, std::size_t block_workspace_elements);

// Reduces [B,H,K,V] to the trainable [H,K,V]. Set accumulate=true when
// accumulating micro-batches.
void reduce_state_gradient_f32(cudaStream_t stream, int batch, int heads,
                               int head_size, const float *per_batch_gradient,
                               float *state_gradient, bool accumulate);

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

} // namespace rwkv7_state_tuning
