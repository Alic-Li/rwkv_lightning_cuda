#include "rwkv_state_tuning.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "rwkv7_fast_v4_kernels.cuh"

namespace rwkv7_state_tuning {
namespace {

constexpr int kN = 64;
constexpr int kThreads = 256;
constexpr float kLnEps = 1.0e-5f;
constexpr float kGnEps = 64.0e-5f;

__global__ void alpha_kernel(int C, const half *__restrict__ a0,
                             const half *__restrict__ a12,
                             half *__restrict__ alpha, std::size_t elements) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  const int c = static_cast<int>(i % C);
  const float x = __half2float(a0[c]) + __half2float(a12[i]);
  alpha[i] = __float2half_rn(1.0f / (1.0f + expf(-x)));
}

__global__ void
vres_forward_kernel(int C, const half *__restrict__ base,
                    const half *__restrict__ first, const half *__restrict__ v0,
                    const half *__restrict__ v12, half *__restrict__ gate,
                    half *__restrict__ out, std::size_t elements) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  const int c = static_cast<int>(i % C);
  const float x = __half2float(v0[c]) + __half2float(v12[i]);
  const float g = 1.0f / (1.0f + expf(-x));
  const float b = __half2float(base[i]);
  gate[i] = __float2half_rn(g);
  out[i] = __float2half_rn(b + (__half2float(first[i]) - b) * g);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__global__ __launch_bounds__(kN) void group_norm_forward_kernel(
    int H, const half *__restrict__ x, const half *__restrict__ weight,
    const half *__restrict__ bias, half *__restrict__ y) {
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int lane = threadIdx.x;
  const long long i = static_cast<long long>(bth) * kN + lane;
  const int c = head * kN + lane;
  const float value = __half2float(x[i]);
  float sum = warp_sum(value);
  __shared__ float partial[2];
  if ((lane & 31) == 0)
    partial[lane >> 5] = sum;
  __syncthreads();
  const float mean = (partial[0] + partial[1]) / kN;
  __syncthreads();
  const float centered = value - mean;
  float square = warp_sum(centered * centered);
  if ((lane & 31) == 0)
    partial[lane >> 5] = square;
  __syncthreads();
  const float rstd = rsqrtf((partial[0] + partial[1]) / kN + kGnEps);
  y[i] = __float2half_rn(centered * rstd * __half2float(weight[c]) +
                         __half2float(bias[c]));
}

struct Arena {
  half *data;
  std::size_t size;
  std::size_t offset = 0;
  half *take(std::size_t count) {
    if (offset + count > size) {
      throw std::invalid_argument("block forward workspace is too small");
    }
    half *result = data + offset;
    offset += count;
    return result;
  }
};

int blocks(std::size_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

void copy(cudaStream_t stream, const half *source, half *destination,
          std::size_t n) {
  if (source == destination)
    return;
  const cudaError_t error = cudaMemcpyAsync(
      destination, source, n * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

void require(const void *pointer) {
  if (!pointer)
    throw std::invalid_argument("missing block forward buffer");
}

} // namespace

std::size_t block_forward_workspace_f16_elements(int batch, int time,
                                                 const FrozenBlockWeights &w) {
  if (batch <= 0 || time <= 0 || w.channels <= 0 || w.ffn <= 0 ||
      w.channels != w.heads * kN || w.rank_w <= 0 || w.rank_a <= 0 ||
      w.rank_g <= 0 || w.rank_v < 0) {
    throw std::invalid_argument("invalid block forward dimensions");
  }
  const std::size_t rows = static_cast<std::size_t>(batch) * time;
  return rows * w.channels * 14 + rows * w.ffn +
         rows * (w.rank_w + w.rank_a + w.rank_g + w.rank_v);
}

void block_forward_state_tuning_f16(cudaStream_t stream, int B, int T,
                                    const half *input, half *attention_shift,
                                    half *ffn_shift, const float *initial_state,
                                    float *final_state, half *v_first,
                                    half *output, const FrozenBlockWeights &w,
                                    BlockTapeView &tape, half *workspace,
                                    std::size_t workspace_elements,
                                    bool layer_zero) {
  const std::size_t required = block_forward_workspace_f16_elements(B, T, w);
  if (!input || !attention_shift || !ffn_shift || !initial_state ||
      !final_state || !v_first || !output || !workspace ||
      workspace_elements < required) {
    throw std::invalid_argument("invalid block forward arguments");
  }
  const int rows = B * T;
  const int C = w.channels;
  const std::size_t R = static_cast<std::size_t>(rows) * C;
  const std::size_t F = static_cast<std::size_t>(rows) * w.ffn;
  tape.batch = B;
  tape.time = T;
  for (void *pointer :
       {static_cast<void *>(tape.x), static_cast<void *>(tape.ln1),
        static_cast<void *>(tape.r), static_cast<void *>(tape.raw_w),
        static_cast<void *>(tape.k), static_cast<void *>(tape.v_base),
        static_cast<void *>(tape.v), static_cast<void *>(tape.alpha),
        static_cast<void *>(tape.neg_kk), static_cast<void *>(tape.wkv_k),
        static_cast<void *>(tape.kka), static_cast<void *>(tape.w1_tanh),
        static_cast<void *>(tape.g1_sigmoid), static_cast<void *>(tape.wkv_y),
        static_cast<void *>(tape.att_group_norm),
        static_cast<void *>(tape.att_gate),
        static_cast<void *>(tape.x_after_att), static_cast<void *>(tape.ln2),
        static_cast<void *>(tape.ffn_hid)}) {
    require(pointer);
  }
  for (const void *pointer :
       {static_cast<const void *>(w.ln1_weight),
        static_cast<const void *>(w.ln1_bias),
        static_cast<const void *>(w.ln2_weight),
        static_cast<const void *>(w.ln2_bias),
        static_cast<const void *>(w.mix_r),
        static_cast<const void *>(w.mix_w),
        static_cast<const void *>(w.mix_k),
        static_cast<const void *>(w.mix_v),
        static_cast<const void *>(w.mix_a),
        static_cast<const void *>(w.mix_g),
        static_cast<const void *>(w.receptance),
        static_cast<const void *>(w.key),
        static_cast<const void *>(w.value),
        static_cast<const void *>(w.output),
        static_cast<const void *>(w.w0),
        static_cast<const void *>(w.w1),
        static_cast<const void *>(w.w2),
        static_cast<const void *>(w.a0),
        static_cast<const void *>(w.a1),
        static_cast<const void *>(w.a2),
        static_cast<const void *>(w.g1),
        static_cast<const void *>(w.g2),
        static_cast<const void *>(w.k_k),
        static_cast<const void *>(w.k_a),
        static_cast<const void *>(w.r_k),
        static_cast<const void *>(w.att_group_norm_weight),
        static_cast<const void *>(w.att_group_norm_bias),
        static_cast<const void *>(w.ffn_mix),
        static_cast<const void *>(w.ffn_key),
        static_cast<const void *>(w.ffn_value)}) {
    require(pointer);
  }
  if (!layer_zero) {
    require(w.v0);
    require(w.v1);
    require(w.v2);
    require(tape.v_gate);
  }

  Arena arena{workspace, workspace_elements};
  half *xr = arena.take(R);
  half *xw = arena.take(R);
  half *xk = arena.take(R);
  half *xv = arena.take(R);
  half *xa = arena.take(R);
  half *xg = arena.take(R);
  half *w1 = arena.take(static_cast<std::size_t>(rows) * w.rank_w);
  half *a1 = arena.take(static_cast<std::size_t>(rows) * w.rank_a);
  half *g1 = arena.take(static_cast<std::size_t>(rows) * w.rank_g);
  half *w12 = arena.take(R);
  half *a12 = arena.take(R);
  half *v12 = arena.take(R);
  half *attention_post = arena.take(R);
  half *attention_out = arena.take(R);
  half *ffn_mixed = arena.take(R);
  half *ffn_act = arena.take(F);
  half *ffn_out = arena.take(R);

  copy(stream, input, tape.x, R);
  rwkv7_v3a_layer_norm_f16_launch(stream, rows, C, input, w.ln1_weight,
                                  w.ln1_bias, tape.ln1, kLnEps);
  rwkv7_tmix_mix6_launch(stream, B, T, C, tape.ln1, attention_shift, w.mix_r,
                         w.mix_w, w.mix_k, w.mix_v, w.mix_a, w.mix_g, xr, xw,
                         xk, xv, xa, xg);
  rwkv7_v3a_linear_f16_orig_launch(stream, rows, C, C, xr, w.receptance,
                                   tape.r);
  rwkv7_v3a_linear_f16_orig_launch(stream, rows, C, C, xk, w.key, tape.k);
  rwkv7_v3a_linear_f16_orig_launch(stream, rows, C, C, xv, w.value,
                                   tape.v_base);

  rwkv7_v3a_linear_f16_launch(stream, rows, C, w.rank_w, xw, w.w1, w1);
  rwkv7_act_tanh_launch(stream, w1, tape.w1_tanh,
                        static_cast<long long>(rows) * w.rank_w);
  rwkv7_v3a_linear_f16_launch(stream, rows, w.rank_w, C, tape.w1_tanh, w.w2,
                              w12);
  rwkv7_add_vec_launch(stream, C, w12, w.w0, tape.raw_w, R);

  rwkv7_v3a_linear_f16_launch(stream, rows, C, w.rank_a, xa, w.a1, a1);
  rwkv7_v3a_linear_f16_launch(stream, rows, w.rank_a, C, a1, w.a2, a12);
  alpha_kernel<<<blocks(R), kThreads, 0, stream>>>(C, w.a0, a12, tape.alpha, R);

  rwkv7_v3a_linear_f16_launch(stream, rows, C, w.rank_g, xg, w.g1, g1);
  rwkv7_act_sigmoid_launch(stream, g1, tape.g1_sigmoid,
                           static_cast<long long>(rows) * w.rank_g);
  rwkv7_v3a_linear_f16_launch(stream, rows, w.rank_g, C, tape.g1_sigmoid, w.g2,
                              tape.att_gate);

  if (layer_zero) {
    copy(stream, tape.v_base, v_first, R);
    copy(stream, tape.v_base, tape.v, R);
  } else {
    half *v1 = arena.take(static_cast<std::size_t>(rows) * w.rank_v);
    rwkv7_v3a_linear_f16_launch(stream, rows, C, w.rank_v, xv, w.v1, v1);
    rwkv7_v3a_linear_f16_launch(stream, rows, w.rank_v, C, v1, w.v2, v12);
    vres_forward_kernel<<<blocks(R), kThreads, 0, stream>>>(
        C, tape.v_base, v_first, w.v0, v12, tape.v_gate, tape.v, R);
  }
  tape.v_first = v_first;

  rwkv7_tmix_kk_a_gate_launch(stream, B, T, C, w.heads, tape.k, w.k_k, w.a0,
                              a12, w.k_a, tape.wkv_k, tape.neg_kk, tape.kka);
  wkv_forward(stream, IoType::F16, {B, T, w.heads, kN}, initial_state, tape.r,
              tape.raw_w, tape.wkv_k, tape.v, tape.neg_kk, tape.kka, tape.wkv_y,
              final_state, tape.wkv);

  group_norm_forward_kernel<<<rows * w.heads, kN, 0, stream>>>(
      w.heads, tape.wkv_y, w.att_group_norm_weight, w.att_group_norm_bias,
      tape.att_group_norm);
  rwkv7_tmix_lnx_rkvres_xg_launch(
      stream, B, T, C, w.heads, tape.wkv_y, tape.r, tape.wkv_k, tape.v, w.r_k,
      w.att_group_norm_weight, w.att_group_norm_bias, tape.att_gate,
      attention_post);
  rwkv7_v3a_linear_f16_orig_launch(stream, rows, C, C, attention_post, w.output,
                                   attention_out);
  rwkv7_v3a_add_f16_launch(stream, input, attention_out, tape.x_after_att, R);

  rwkv7_v3a_layer_norm_f16_launch(stream, rows, C, tape.x_after_att,
                                  w.ln2_weight, w.ln2_bias, tape.ln2, kLnEps);
  rwkv7_cmix_mix_launch(stream, B, T, C, tape.ln2, ffn_shift, w.ffn_mix,
                        ffn_mixed);
  rwkv7_v3a_linear_f16_orig_launch(stream, rows, C, w.ffn, ffn_mixed, w.ffn_key,
                                   tape.ffn_hid);
  rwkv7_relu_square_launch(stream, tape.ffn_hid, ffn_act, F);
  rwkv7_v3a_linear_f16_launch(stream, rows, w.ffn, C, ffn_act, w.ffn_value,
                              ffn_out);
  rwkv7_v3a_add_f16_launch(stream, tape.x_after_att, ffn_out, output, R);
}

} // namespace rwkv7_state_tuning
