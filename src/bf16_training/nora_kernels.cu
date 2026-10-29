#include "rwkv/bf16_training/nora.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/runtime/nora_gemm.cuh"
#include <cmath>
#include <stdexcept>
namespace rwkv_bf16_nora {
namespace {
void validate(int rows, const rwkv_bf16_miss::Linear &w) {
  if (!w.a || !w.d || rows < 1 || rows > 65535 || w.in < 1 || w.out < 1 ||
      w.rank < 1 || w.rank > 1024 || !std::isfinite(w.scale))
    throw std::invalid_argument("invalid NoRA shape/weights");
}
void checked() {
  rwkv_bf16_training::gpu_check(cudaGetLastError(), "NoRA kernel");
}
__global__ void replay_mix(int time, int C, int rows, const bf16 *x,
                           const bf16 *shift, const bf16 *mix, bf16 *y) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * C)
    return;
  int c = i % C, t = (i / C) % time, b = i / (size_t(time) * C);
  float v = to_float(x[i]), p = to_float(t ? x[i - C] : shift[b * C + c]);
  y[i] = to_bf16(v + (p - v) * to_float(mix[c]));
}
} // namespace
void forward(cudaStream_t stream, int rows, const rwkv_bf16_miss::Linear &w,
             const bf16 *x, bf16 *y, float *s) {
  validate(rows, w);
  if (!x || !y)
    throw std::invalid_argument("null NoRA forward input");
  using namespace rwkv_nora_detail;
  if (s) {
    gemm<false, true, false>(stream, rows, w.rank, w.in, x, w.a, s);
    gemm<false, true, true>(stream, rows, w.out, w.rank, s, w.d, y, w.scale);
  } else {
    fused<<<dim3((w.out + 255) / 256, rows), 256, w.rank * sizeof(float),
            stream>>>(w.in, w.out, w.rank, w.scale, w.a, w.d, x, y);
  }
  checked();
}
void backward(cudaStream_t stream, int rows, const rwkv_bf16_miss::Linear &w,
              const bf16 *x, const float *s, const bf16 *g, bf16 *dx,
              float *ds) {
  validate(rows, w);
  if (!g || !dx || !ds || (w.gradient && !s) || (w.gradient_a && !x))
    throw std::invalid_argument("null NoRA backward input/tape");
  using namespace rwkv_nora_detail;
  if (w.gradient)
    gemm<true, false, true>(stream, w.out, w.rank, rows, g, s, w.gradient,
                            w.scale);
  gemm<false, false, false>(stream, rows, w.rank, w.out, g, w.d, ds, w.scale);
  if (w.gradient_a)
    gemm<true, false, true>(stream, w.rank, w.in, rows, ds, x, w.gradient_a);
  gemm<false, false, true>(stream, rows, w.in, w.rank, ds, w.a, dx);
  checked();
}
void backward_target(cudaStream_t stream,
                     const rwkv_bf16_training::FrozenBlockWeights &w,
                     const rwkv_bf16_training::BlockTapeView &t, int target,
                     const bf16 *g, bf16 *dx) {
  const auto &a = w.miss[target];
  if (!a.a) {
    rwkv_bf16_miss::backward(stream, t.batch * t.time, a,
                             t.miss_reduced[target], g, dx);
    return;
  }
  int rows = t.batch * t.time, C = w.channels;
  bf16 *x = t.nora_input;
  if (!t.nora_ds || (a.gradient_a && !x))
    throw std::invalid_argument("missing NoRA replay scratch");
  if (a.gradient_a) {
    using namespace rwkv_bf16_training;
    if (target == 5)
      rwkv7_relu_square_launch(stream, t.ffn_hid, x, size_t(rows) * w.ffn);
    else if (target == 3)
      rwkv7_tmix_lnx_rkvres_xg_launch(
          stream, t.batch, t.time, C, w.heads, t.wkv_y, t.r, t.wkv_k, t.v,
          w.r_k, w.att_group_norm_weight, w.att_group_norm_bias, t.att_gate, x);
    else {
      const bf16 *mix = target == 0   ? w.mix_r
                        : target == 1 ? w.mix_k
                        : target == 2 ? w.mix_v
                                      : w.ffn_mix;
      const bf16 *shift = target == 4 ? t.nora_ffn_shift : t.nora_att_shift;
      if (!shift)
        throw std::invalid_argument("missing NoRA incoming shift");
      replay_mix<<<(size_t(rows) * C + 255) / 256, 256, 0, stream>>>(
          t.time, C, rows, target == 4 ? t.ln2 : t.ln1, shift, mix, x);
    }
  }
  backward(stream, rows, a, x, t.miss_reduced[target], g, dx, t.nora_ds);
}
} // namespace rwkv_bf16_nora
