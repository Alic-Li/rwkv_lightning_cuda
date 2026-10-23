#include "rwkv/bf16_training/executor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace rwkv_bf16_training {
namespace {
struct OwnedTape {
  DeviceBuffer<bf16> x, ln1, r, raw_w, k, v_base, v, alpha, neg_kk, wkv_k, kka,
      w1_tanh, g1_sigmoid, v_gate, wkv_y, att_group_norm, att_gate, x_after_att,
      ln2, ffn_hid;
  DeviceBuffer<float> wkv_initial;
  std::array<DeviceBuffer<float>, rwkv_bf16_miss::TargetCount> miss_reduced;
  BlockTapeView view;

  void allocate(int time, const FrozenBlockWeights &w, bf16 *v_first,
                float *shared_tape, float *shared_final) {
    const std::size_t row = static_cast<std::size_t>(time) * w.channels;
    auto alloc = [row](DeviceBuffer<bf16> &buffer, const char *name) {
      buffer.resize(row, name);
    };
    alloc(x, "tune tape x");
    alloc(ln1, "tune tape ln1");
    alloc(r, "tune tape r");
    alloc(raw_w, "tune tape raw_w");
    alloc(k, "tune tape k");
    alloc(v_base, "tune tape v_base");
    alloc(v, "tune tape v");
    alloc(alpha, "tune tape alpha");
    alloc(neg_kk, "tune tape neg_kk");
    alloc(wkv_k, "tune tape wkv_k");
    alloc(kka, "tune tape kka");
    w1_tanh.resize(static_cast<std::size_t>(time) * w.rank_w,
                   "tune tape w1 tanh");
    g1_sigmoid.resize(static_cast<std::size_t>(time) * w.rank_g,
                      "tune tape g1 sigmoid");
    if (w.rank_v > 0)
      alloc(v_gate, "tune tape v gate");
    alloc(wkv_y, "tune tape wkv y");
    alloc(att_group_norm, "tune tape att group norm");
    alloc(att_gate, "tune tape att gate");
    alloc(x_after_att, "tune tape x after att");
    alloc(ln2, "tune tape ln2");
    ffn_hid.resize(static_cast<std::size_t>(time) * w.ffn, "tune tape ffn hid");
    const rwkv_bf16_training::WkvShape shape{1, time, w.heads, 64};
    const auto tape_elements = rwkv_bf16_training::wkv_tape_elements(shape);
    wkv_initial.resize(static_cast<std::size_t>(w.heads) * 64 * 64,
                       "tune WKV replay initial");
    view.wkv_replay_initial = wkv_initial.p;
    view.wkv_replay_final = shared_final;
    view.batch = 1;
    view.time = time;
    view.x = x.p;
    view.ln1 = ln1.p;
    view.r = r.p;
    view.raw_w = raw_w.p;
    view.k = k.p;
    view.v_base = v_base.p;
    view.v_first = v_first;
    view.v = v.p;
    view.alpha = alpha.p;
    view.neg_kk = neg_kk.p;
    view.wkv_k = wkv_k.p;
    view.kka = kka.p;
    view.w1_tanh = w1_tanh.p;
    view.g1_sigmoid = g1_sigmoid.p;
    view.v_gate = v_gate.p;
    view.wkv_y = wkv_y.p;
    view.att_group_norm = att_group_norm.p;
    view.att_gate = att_gate.p;
    view.x_after_att = x_after_att.p;
    view.ln2 = ln2.p;
    view.ffn_hid = ffn_hid.p;
    view.wkv = {shared_tape, tape_elements};
    // Frozen adapters still use the same reduction order as the learner.
    // These scratch buffers belong to this executor, never another graph.
    for (int i = 0; i < rwkv_bf16_miss::TargetCount; ++i)
      if (w.miss[i].d && !w.miss[i].gradient) {
        miss_reduced[i].resize(size_t(time) * w.miss[i].rank,
                               "frozen MiSS reduced");
        view.miss_reduced[i] = miss_reduced[i].p;
      }
  }
};

} // namespace
struct Executor::Impl {
  int L, C, H, V, chunk, last_time = 0;
  size_t lane, states, shift_count;
  cudaStream_t stream;
  DeviceBuffer<bf16> input, ping, pong, norm, logits, dlogits, gping, gpong,
      shifts, shift_grad, first, first_grad, workspace, terminal_shift;
  DeviceBuffer<int> targets;
  DeviceBuffer<float> state_gradient, per_batch_gradient, state_batch,
      state_final, loss, shared_tape, shared_final, carried, weights;
  std::vector<OwnedTape> owned;
  std::vector<BlockTapeView> tapes;
  std::vector<BlockBackwardState> gradients;
  std::vector<DeviceBuffer<float>> checkpoints;
  std::vector<DeviceBuffer<bf16>> shift_checkpoints;
  std::vector<uint16_t> host_input;
  const bf16 *final_block = nullptr;
  Impl(const FrozenModelView &m, int time, cudaStream_t s)
      : L(m.layers), C(m.channels), H(m.heads), V(m.vocab), chunk(time),
        lane(size_t(H) * 4096), states(L * lane),
        shift_count(size_t(L) * 2 * C), stream(s) {
    if (time <= 0 || L <= 0 || H <= 0 || C != H * 64 || V <= 0)
      throw std::invalid_argument("invalid BF16 executor dimensions/chunk");
    const size_t rows = size_t(chunk) * C;
    for (auto *b :
         {&input, &ping, &pong, &norm, &gping, &gpong, &first, &first_grad})
      b->resize(rows, "executor activations");
    for (auto *b : {&shifts, &shift_grad, &terminal_shift})
      b->resize(shift_count, "executor shifts");
    logits.resize(size_t(chunk) * V, "executor logits");
    dlogits.resize(logits.n, "executor dlogits");
    targets.resize(chunk, "executor targets");
    weights.resize(chunk, "CE row weights");
    for (auto *b : {&state_gradient, &per_batch_gradient, &carried})
      b->resize(states, "executor states");
    for (auto *b : {&state_batch, &state_final, &shared_final})
      b->resize(lane, "executor state scratch");
    loss.resize(1, "executor loss");
    shared_tape.resize(wkv_tape_elements({1, chunk, H, 64}),
                       "shared WKV replay tape");
    owned.resize(L);
    tapes.resize(L);
    gradients.resize(L);
    size_t scratch = 0;
    for (int l = 0; l < L; ++l) {
      owned[l].allocate(chunk, m.blocks[l], first.p, shared_tape.p,
                        shared_final.p);
      tapes[l] = owned[l].view;
      scratch = std::max(
          {scratch,
           block_forward_workspace_bf16_elements(1, chunk, m.blocks[l]),
           block_backward_workspace_bf16_elements(1, chunk, m.blocks[l])});
      gradients[l] = {state_gradient.p + l * lane,
                      per_batch_gradient.p + l * lane,
                      shift_grad.p + size_t(l) * 2 * C,
                      shift_grad.p + size_t(l) * 2 * C + C,
                      first_grad.p,
                      terminal_shift.p + size_t(l) * 2 * C,
                      terminal_shift.p + size_t(l) * 2 * C + C};
    }
    workspace.resize(scratch, "executor block workspace");
  }
  void copy(void *to, const void *from, size_t n) {
    gpu_check(cudaMemcpyAsync(to, from, n, cudaMemcpyDeviceToDevice, stream),
              "copy boundary");
  }
  void zero(void *to, size_t n) {
    gpu_check(cudaMemsetAsync(to, 0, n, stream), "clear adjoint");
  }
  void sync() const {
    gpu_check(cudaStreamSynchronize(stream), "BF16 execution");
  }
  void forward(const FrozenModelView &m, const int *tokens, int T) {
    if (T <= 0 || T > chunk)
      throw std::invalid_argument("forward chunk length");
    host_input.resize(size_t(T) * C);
    for (int t = 0; t < T; ++t) {
      if (tokens[t] < 0 || tokens[t] >= V)
        throw std::invalid_argument("token out of range");
      std::copy_n(m.cpu_emb_ln0_bf16 + size_t(tokens[t]) * C, C,
                  host_input.data() + size_t(t) * C);
    }
    gpu_check(cudaMemcpyAsync(input.p, host_input.data(), host_input.size() * 2,
                              cudaMemcpyHostToDevice, stream),
              "upload embeddings");
    for (auto &t : tapes)
      t.time = T;
    final_block = model_forward_state_tuning_bf16(
        stream, L, 1, T, m.blocks.data(), tapes.data(), input.p, carried.p,
        shifts.p, m.ln_out_weight, m.ln_out_bias, m.head_weight_orig, V, ping.p,
        pong.p, state_batch.p, state_final.p, workspace.p, workspace.n, norm.p,
        logits.p, carried.p);
    last_time = T;
    sync(); // host input staging may be reused after returning
  }
};
Executor::Executor(const FrozenModelView &m, int chunk, cudaStream_t s)
    : impl_(std::make_unique<Impl>(m, chunk, s)) {}
Executor::~Executor() = default;
std::vector<BlockTapeView> &Executor::tapes() { return impl_->tapes; }
DeviceBuffer<float> &Executor::state() { return impl_->carried; }
DeviceBuffer<bf16> &Executor::shifts() { return impl_->shifts; }
const float *Executor::state_gradient() const {
  return impl_->state_gradient.p;
}
void Executor::reset(const float *initial) {
  auto &w = *impl_;
  w.copy(w.carried.p, initial, w.states * 4);
  w.zero(w.shifts.p, w.shift_count * 2);
}
void Executor::forward(const FrozenModelView &m, const int *tokens, int count) {
  impl_->forward(m, tokens, count);
}
std::vector<float> Executor::last_logits() const {
  auto &w = *impl_;
  if (!w.last_time)
    throw std::logic_error("no forward logits");
  std::vector<bf16> bits(w.V);
  gpu_check(cudaMemcpyAsync(bits.data(),
                            w.logits.p + size_t(w.last_time - 1) * w.V,
                            size_t(w.V) * 2, cudaMemcpyDeviceToHost, w.stream),
            "download logits");
  w.sync();
  std::vector<float> result(w.V);
  for (int i = 0; i < w.V; ++i)
    result[i] = to_float(bits[i]);
  return result;
}
float Executor::backward(const FrozenModelView &m,
                         const std::vector<int> &tokens,
                         const std::vector<float> &weights,
                         const float *initial, size_t normalizer,
                         float temperature) {
  auto &w = *impl_;
  if (tokens.size() < 2 || weights.size() != tokens.size() || !normalizer)
    throw std::invalid_argument("backward tokens/weights/normalizer");
  for (size_t i = 0; i < tokens.size(); ++i)
    if (tokens[i] < 0 || tokens[i] >= w.V || !std::isfinite(weights[i]))
      throw std::invalid_argument("invalid backward token/weight");
  const int length = int(tokens.size()) - 1,
            chunks = 1 + (length - 1) / w.chunk;
  w.checkpoints.resize(chunks);
  w.shift_checkpoints.resize(chunks);
  reset(initial);
  for (int j = 0; j < chunks; ++j) {
    w.checkpoints[j].resize(w.states, "WKV boundary checkpoint");
    w.shift_checkpoints[j].resize(w.shift_count, "shift boundary checkpoint");
    w.copy(w.checkpoints[j].p, w.carried.p, w.states * 4);
    w.copy(w.shift_checkpoints[j].p, w.shifts.p, w.shift_count * 2);
    if (j + 1 < chunks)
      w.forward(m, tokens.data() + j * w.chunk, w.chunk);
  }
  w.zero(w.state_gradient.p, w.states * 4);
  w.zero(w.terminal_shift.p, w.shift_count * 2);
  float total = 0;
  for (int j = chunks - 1; j >= 0; --j) {
    int off = j * w.chunk, T = std::min(w.chunk, length - off);
    w.copy(w.carried.p, w.checkpoints[j].p, w.states * 4);
    w.copy(w.shifts.p, w.shift_checkpoints[j].p, w.shift_count * 2);
    w.forward(m, tokens.data() + off, T);
    gpu_check(cudaMemcpyAsync(w.targets.p, tokens.data() + off + 1,
                              T * sizeof(int), cudaMemcpyHostToDevice,
                              w.stream),
              "upload targets");
    gpu_check(cudaMemcpyAsync(w.weights.p, weights.data() + off + 1,
                              T * sizeof(float), cudaMemcpyHostToDevice,
                              w.stream),
              "upload loss weights");
    cross_entropy_forward_backward_bf16(
        w.stream, T, w.V, w.logits.p, w.targets.p, -1, w.loss.p, w.dlogits.p,
        float(T) / normalizer, w.weights.p, temperature);
    model_output_backward_state_only(
        w.stream, w.L, 1, T, m.blocks.data(), w.tapes.data(),
        w.gradients.data(), w.final_block, m.ln_out_weight, m.head_weight_orig,
        w.V, w.dlogits.p, w.gping.p, w.gpong.p, w.workspace.p, w.workspace.n);
    w.copy(w.state_gradient.p, w.per_batch_gradient.p, w.states * 4);
    w.copy(w.terminal_shift.p, w.shift_grad.p, w.shift_count * 2);
    float loss;
    gpu_check(
        cudaMemcpyAsync(&loss, w.loss.p, 4, cudaMemcpyDeviceToHost, w.stream),
        "read loss");
    w.sync();
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite loss; optimizer not executed");
    total += loss;
  }
  return total;
}
} // namespace rwkv_bf16_training
