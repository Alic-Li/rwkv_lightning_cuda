#include "rwkv/runtime/rwkv_state_tuning.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace rwkv7_state_tuning {
namespace {

constexpr int kN = 64;
constexpr int kThreads = 256;
constexpr float kLnEps = 1.0e-5f;
constexpr float kGnEps = 64.0e-5f;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__global__ __launch_bounds__(kN) void post_backward_kernel(
    int H, const half *__restrict__ group_norm, const half *__restrict__ r,
    const half *__restrict__ k, const half *__restrict__ v,
    const half *__restrict__ r_k, const half *__restrict__ gate,
    const half *__restrict__ dz, half *__restrict__ d_group_norm,
    half *__restrict__ dr, half *__restrict__ dk, half *__restrict__ dv,
    half *__restrict__ d_gate) {
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int lane = threadIdx.x;
  const long long idx = static_cast<long long>(bth) * kN + lane;
  const int c = head * kN + lane;
  const float rv = __half2float(r[idx]);
  const float kv = __half2float(k[idx]);
  const float vv = __half2float(v[idx]);
  const float gv = __half2float(gate[idx]);
  const float upstream = __half2float(dz[idx]);
  float q_piece = rv * kv * __half2float(r_k[c]);
  float dq_piece = upstream * gv * vv;
  q_piece = warp_sum(q_piece);
  dq_piece = warp_sum(dq_piece);
  __shared__ float partial_q[2], partial_dq[2];
  if ((lane & 31) == 0) {
    partial_q[lane >> 5] = q_piece;
    partial_dq[lane >> 5] = dq_piece;
  }
  __syncthreads();
  const float q = partial_q[0] + partial_q[1];
  const float dq = partial_dq[0] + partial_dq[1];
  const float rk = __half2float(r_k[c]);
  d_group_norm[idx] = __float2half_rn(upstream * gv);
  dr[idx] = __float2half_rn(dq * kv * rk);
  dk[idx] = __float2half_rn(dq * rv * rk);
  dv[idx] = __float2half_rn(upstream * gv * q);
  d_gate[idx] =
      __float2half_rn(upstream * (__half2float(group_norm[idx]) + q * vv));
}

__global__ __launch_bounds__(kN) void kk_backward_kernel(
    int H, const half *__restrict__ original_k, const half *__restrict__ alpha,
    const half *__restrict__ neg_kk, const half *__restrict__ k_k,
    const half *__restrict__ k_a, const half *__restrict__ dk_direct,
    const half *__restrict__ d_new_k, const half *__restrict__ d_neg_kk,
    const half *__restrict__ d_kka, half *__restrict__ dk,
    half *__restrict__ d_a12) {
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int lane = threadIdx.x;
  const long long idx = static_cast<long long>(bth) * kN + lane;
  const int c = head * kN + lane;
  const float kval = __half2float(original_k[idx]);
  const float scale_k = __half2float(k_k[c]);
  const float u = kval * scale_k;
  const float kk = -__half2float(neg_kk[idx]);
  const float av = __half2float(alpha[idx]);
  const float ka = __half2float(k_a[c]);
  // Both the WKV recurrence and the post-WKV rkv residual consume new_k.
  // Combine those two edges before differentiating new_k =
  // original_k * (1 - k_a + alpha * k_a).
  const float dnew = __half2float(dk_direct[idx]) + __half2float(d_new_k[idx]);
  const float dkk =
      -__half2float(d_neg_kk[idx]) + __half2float(d_kka[idx]) * av;
  float norm_piece = u * u;
  float dot_piece = dkk * kk;
  norm_piece = warp_sum(norm_piece);
  dot_piece = warp_sum(dot_piece);
  __shared__ float norm_part[2], dot_part[2];
  if ((lane & 31) == 0) {
    norm_part[lane >> 5] = norm_piece;
    dot_part[lane >> 5] = dot_piece;
  }
  __syncthreads();
  const float norm = sqrtf(norm_part[0] + norm_part[1]);
  const float inv = 1.0f / fmaxf(norm, 1.0e-12f);
  const float dot = dot_part[0] + dot_part[1];
  const float du = norm > 1.0e-12f ? (dkk - kk * dot) * inv : dkk * inv;
  const float mix_scale = 1.0f - ka + av * ka;
  const float da = __half2float(d_kka[idx]) * kk + dnew * kval * ka;
  dk[idx] = __float2half_rn(dnew * mix_scale + du * scale_k);
  d_a12[idx] = __float2half_rn(da * av * (1.0f - av));
}

__global__ void
vres_backward_kernel(const half *__restrict__ v_base,
                     const half *__restrict__ v_first,
                     const half *__restrict__ gate, const half *__restrict__ dv,
                     half *__restrict__ dv_base, half *__restrict__ dv12,
                     half *__restrict__ dv_first, std::size_t elements) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  const float g = __half2float(gate[i]);
  const float upstream = __half2float(dv[i]);
  const float base = __half2float(v_base[i]);
  const float first = __half2float(v_first[i]);
  dv_base[i] = __float2half_rn(upstream * (1.0f - g));
  dv12[i] = __float2half_rn(upstream * (first - base) * g * (1.0f - g));
  if (dv_first) {
    dv_first[i] = __float2half_rn(__half2float(dv_first[i]) + upstream * g);
  }
}

struct Arena {
  half *data;
  std::size_t size;
  std::size_t offset = 0;

  half *take(std::size_t count) {
    if (offset + count > size) {
      throw std::invalid_argument("block backward workspace is too small");
    }
    half *result = data + offset;
    offset += count;
    return result;
  }
};

void copy(cudaStream_t stream, const half *source, half *destination,
          std::size_t n) {
  if (source == destination)
    return;
  const cudaError_t error = cudaMemcpyAsync(
      destination, source, n * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

void require_pointer(const void *pointer, const char *name) {
  if (!pointer) {
    throw std::invalid_argument(std::string("missing block tape/weight: ") +
                                name);
  }
}

int blocks(std::size_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

} // namespace

std::size_t
block_backward_workspace_elements_impl(int rows, const FrozenBlockWeights &w) {
  const std::size_t row = static_cast<std::size_t>(rows);
  return row * w.channels * 32 + row * w.ffn * 2 +
         row * (2 * w.rank_w + w.rank_a + 2 * w.rank_g + 2 * w.rank_v);
}

std::size_t block_backward_workspace_f16_elements(int batch, int time,
                                                  const FrozenBlockWeights &w) {
  if (batch <= 0 || time <= 0 || w.channels <= 0 || w.ffn <= 0 ||
      w.heads <= 0 || w.channels != w.heads * kN || w.rank_w <= 0 ||
      w.rank_a <= 0 || w.rank_g <= 0 || w.rank_v < 0) {
    throw std::invalid_argument("invalid block backward dimensions");
  }
  return block_backward_workspace_elements_impl(batch * time, w);
}

} // namespace rwkv7_state_tuning

namespace rwkv7_state_tuning {

void block_backward_state_only(cudaStream_t stream, const half *grad_out,
                               half *grad_in, BlockBackwardState state_grad,
                               const BlockTapeView &tape,
                               const FrozenBlockWeights &w, half *workspace,
                               std::size_t workspace_elements,
                               bool layer_zero) {
  const std::size_t required =
      block_backward_workspace_f16_elements(tape.batch, tape.time, w);
  if (!workspace || workspace_elements < required || !grad_out || !grad_in ||
      !state_grad.initial_state_grad) {
    throw std::invalid_argument("invalid block backward buffers");
  }
  if (tape.batch <= 0 || tape.time <= 0) {
    throw std::invalid_argument("invalid block tape shape");
  }
  for (const auto &item : {std::pair<const void *, const char *>{tape.x, "x"},
                           {tape.ln1, "ln1"},
                           {tape.r, "r"},
                           {tape.raw_w, "raw_w"},
                           {tape.k, "k"},
                           {tape.v_base, "v_base"},
                           {tape.wkv_k, "wkv_k"},
                           {tape.alpha, "alpha"},
                           {tape.neg_kk, "neg_kk"},
                           {tape.kka, "kka"},
                           {tape.w1_tanh, "w1_tanh"},
                           {tape.g1_sigmoid, "g1_sigmoid"},
                           {tape.v, "v"},
                           {tape.wkv_y, "wkv_y"},
                           {tape.att_group_norm, "att_group_norm"},
                           {tape.att_gate, "att_gate"},
                           {tape.x_after_att, "x_after_att"},
                           {tape.ln2, "ln2"},
                           {tape.ffn_hid, "ffn_hid"},
                           {w.ln1_weight, "ln1_weight"},
                           {w.ln2_weight, "ln2_weight"},
                           {w.receptance, "receptance"},
                           {w.key, "key"},
                           {w.value, "value"},
                           {w.output, "output"},
                           {w.w1, "w1"},
                           {w.w2, "w2"},
                           {w.a1, "a1"},
                           {w.a2, "a2"},
                           {w.g1, "g1"},
                           {w.g2, "g2"},
                           {w.k_k, "k_k"},
                           {w.k_a, "k_a"},
                           {w.r_k, "r_k"},
                           {w.att_group_norm_weight, "att_group_norm_weight"},
                           {w.ffn_mix, "ffn_mix"},
                           {w.ffn_key, "ffn_key"},
                           {w.ffn_value, "ffn_value"}}) {
    require_pointer(item.first, item.second);
  }
  if (!layer_zero) {
    require_pointer(tape.v_first, "v_first");
    require_pointer(tape.v_gate, "v_gate");
    require_pointer(w.v1, "v1");
    require_pointer(w.v2, "v2");
    if (w.rank_v <= 0)
      throw std::invalid_argument("nonzero layer requires rank_v");
  }

  const int rows = tape.batch * tape.time;
  const int C = w.channels;
  const std::size_t R = static_cast<std::size_t>(rows) * C;
  const std::size_t F = static_cast<std::size_t>(rows) * w.ffn;
  Arena arena{workspace, workspace_elements};
  half *grad_x_after = arena.take(R);
  half *d_act = arena.take(F);
  half *d_hid = arena.take(F);
  half *d_mixed = arena.take(R);
  half *d_ln2 = arena.take(R);
  half *d_post = arena.take(R);
  half *d_gn = arena.take(R);
  half *d_wkv_y = arena.take(R);
  half *post_dr = arena.take(R);
  half *post_dk = arena.take(R);
  half *post_dv = arena.take(R);
  half *d_gate = arena.take(R);
  half *wkv_dr = arena.take(R);
  half *d_w = arena.take(R);
  half *d_wkv_k = arena.take(R);
  half *d_v = arena.take(R);
  half *d_neg_kk = arena.take(R);
  half *d_kka = arena.take(R);
  half *d_k = arena.take(R);
  half *d_a12 = arena.take(R);
  half *d_v_base = arena.take(R);
  half *d_v12 = arena.take(R);
  half *dx_r = arena.take(R);
  half *dx_w = arena.take(R);
  half *dx_k = arena.take(R);
  half *dx_v = arena.take(R);
  half *dx_a = arena.take(R);
  half *dx_g = arena.take(R);
  half *d_ln1 = arena.take(R);
  half *temp0 = arena.take(R);
  half *temp1 = arena.take(R);

  // ChannelMix/FFN, followed by the residual into x_after_att.
  linear_runtime_backward_input_f16(stream, rows, w.ffn, C, grad_out,
                                    w.ffn_value, d_act);
  relu_square_backward_input_f16(stream, tape.ffn_hid, d_act, d_hid, F);
  linear_orig_backward_input_f16(stream, rows, C, w.ffn, d_hid, w.ffn_key,
                                 d_mixed);
  const half *ffn_grads[] = {d_mixed};
  const half *ffn_mixes[] = {w.ffn_mix};
  time_mix_backward_input_f16(stream, tape.batch, tape.time, C, 1, ffn_grads,
                              ffn_mixes, d_ln2, state_grad.ffn_shift_grad);
  add_last_shift_gradient(stream, tape.batch, tape.time, C, d_ln2,
                          state_grad.final_ffn_shift_grad);
  layer_norm_backward_input_f16(stream, rows, C, tape.x_after_att, w.ln2_weight,
                                d_ln2, temp0, kLnEps);
  add_f16(stream, grad_out, temp0, grad_x_after, R);

  // Attention output projection and fused post-WKV GroupNorm/rkv/gate.
  linear_orig_backward_input_f16(stream, rows, C, C, grad_x_after, w.output,
                                 d_post);
  post_backward_kernel<<<rows * w.heads, kN, 0, stream>>>(
      w.heads, tape.att_group_norm, tape.r, tape.wkv_k, tape.v, w.r_k,
      tape.att_gate, d_post, d_gn, post_dr, post_dk, post_dv, d_gate);
  group_norm_backward_input_f16(stream, rows, C, w.heads, tape.wkv_y,
                                w.att_group_norm_weight, d_gn, d_wkv_y, kGnEps);

  // Exact WKV reverse recurrence. The only persistent result is ds_initial.
  wkv_backward_input(stream, IoType::F16, {tape.batch, tape.time, w.heads, kN},
                     tape.r, tape.raw_w, tape.wkv_k, tape.v, tape.neg_kk,
                     tape.kka, d_wkv_y, state_grad.final_state_grad, tape.wkv,
                     state_grad.initial_state_grad, wkv_dr, d_w, d_wkv_k, d_v,
                     d_neg_kk, d_kka);
  add_inplace_f16(stream, wkv_dr, post_dr, R);
  add_inplace_f16(stream, d_v, post_dv, R);

  kk_backward_kernel<<<rows * w.heads, kN, 0, stream>>>(
      w.heads, tape.k, tape.alpha, tape.neg_kk, w.k_k, w.k_a, post_dk, d_wkv_k,
      d_neg_kk, d_kka, d_k, d_a12);

  // v_first is a cross-layer edge. Reverse layers accumulate it until layer 0.
  if (layer_zero) {
    copy(stream, d_v, d_v_base, R);
    if (state_grad.v_first_grad) {
      add_inplace_f16(stream, d_v_base, state_grad.v_first_grad, R);
    }
  } else {
    vres_backward_kernel<<<blocks(R), kThreads, 0, stream>>>(
        tape.v_base, tape.v_first, tape.v_gate, d_v, d_v_base, d_v12,
        state_grad.v_first_grad, R);
  }

  // Frozen low-rank branches: only their inputs receive gradients.
  half *d_w_tanh = arena.take(static_cast<std::size_t>(rows) * w.rank_w);
  half *d_w1 = arena.take(static_cast<std::size_t>(rows) * w.rank_w);
  linear_runtime_backward_input_f16(stream, rows, w.rank_w, C, d_w, w.w2,
                                    d_w_tanh);
  tanh_backward_input_f16(stream, tape.w1_tanh, d_w_tanh, d_w1,
                          static_cast<std::size_t>(rows) * w.rank_w);
  linear_runtime_backward_input_f16(stream, rows, C, w.rank_w, d_w1, w.w1,
                                    dx_w);

  half *d_a1 = arena.take(static_cast<std::size_t>(rows) * w.rank_a);
  linear_runtime_backward_input_f16(stream, rows, w.rank_a, C, d_a12, w.a2,
                                    d_a1);
  linear_runtime_backward_input_f16(stream, rows, C, w.rank_a, d_a1, w.a1,
                                    dx_a);

  half *d_g_sigmoid = arena.take(static_cast<std::size_t>(rows) * w.rank_g);
  half *d_g1 = arena.take(static_cast<std::size_t>(rows) * w.rank_g);
  linear_runtime_backward_input_f16(stream, rows, w.rank_g, C, d_gate, w.g2,
                                    d_g_sigmoid);
  sigmoid_backward_input_f16(stream, tape.g1_sigmoid, d_g_sigmoid, d_g1,
                             static_cast<std::size_t>(rows) * w.rank_g);
  linear_runtime_backward_input_f16(stream, rows, C, w.rank_g, d_g1, w.g1,
                                    dx_g);

  // Base r/k/v projections.
  linear_orig_backward_input_f16(stream, rows, C, C, wkv_dr, w.receptance,
                                 dx_r);
  linear_orig_backward_input_f16(stream, rows, C, C, d_k, w.key, dx_k);
  linear_orig_backward_input_f16(stream, rows, C, C, d_v_base, w.value, dx_v);
  if (!layer_zero) {
    half *d_v1 = arena.take(static_cast<std::size_t>(rows) * w.rank_v);
    half *rank_v_padding =
        arena.take(static_cast<std::size_t>(rows) * w.rank_v);
    (void)rank_v_padding;
    linear_runtime_backward_input_f16(stream, rows, w.rank_v, C, d_v12, w.v2,
                                      d_v1);
    linear_runtime_backward_input_f16(stream, rows, C, w.rank_v, d_v1, w.v1,
                                      temp0);
    add_inplace_f16(stream, dx_v, temp0, R);
  }

  const half *tmix_grads[] = {dx_r, dx_w, dx_k, dx_v, dx_a, dx_g};
  const half *tmix_weights[] = {w.mix_r, w.mix_w, w.mix_k,
                                w.mix_v, w.mix_a, w.mix_g};
  for (int i = 0; i < 6; ++i) {
    require_pointer(tmix_weights[i], "time-mix weight");
  }
  time_mix_backward_input_f16(stream, tape.batch, tape.time, C, 6, tmix_grads,
                              tmix_weights, d_ln1, state_grad.att_shift_grad);
  add_last_shift_gradient(stream, tape.batch, tape.time, C, d_ln1,
                          state_grad.final_att_shift_grad);
  layer_norm_backward_input_f16(stream, rows, C, tape.x, w.ln1_weight, d_ln1,
                                temp1, kLnEps);
  add_f16(stream, grad_x_after, temp1, grad_in, R);
}

} // namespace rwkv7_state_tuning
