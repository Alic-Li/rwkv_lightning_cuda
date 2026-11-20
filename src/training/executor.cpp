#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace rwkv_bf16_training {
namespace {
struct OwnedTape {
  DeviceBuffer<bf16> x, ln1, r, raw_w, k, v_base, v, alpha, neg_kk, wkv_k, kka,
      w1_tanh, g1_sigmoid, v_gate, wkv_y, att_group_norm, att_gate, x_after_att,
      ln2, ffn_hid, nora_att_shift, nora_ffn_shift;
  DeviceBuffer<float> wkv_initial;
  std::array<DeviceBuffer<float>, rwkv_bf16_miss::TargetCount> miss_reduced;
  BlockTapeView view;

  // replay=false leaves WKV unrecorded (decode only, never differentiated).
  // time is the row capacity; lanes sizes the per-lane replay state/shifts.
  void allocate(int time, const FrozenBlockWeights &w, bf16 *v_first,
                float *shared_tape, float *shared_final, bool replay = true,
                int lanes = 1) {
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
    if (replay) {
      // time = lanes * chunk rows; the tape also holds per-lane scratch.
      const rwkv_bf16_training::WkvShape shape{lanes, time / lanes, w.heads,
                                               64};
      wkv_initial.resize(static_cast<std::size_t>(lanes) * w.heads * 64 * 64,
                         "tune WKV replay initial");
      view.wkv_replay_initial = wkv_initial.p;
      view.wkv_replay_final = shared_final;
      view.wkv = {shared_tape, rwkv_bf16_training::wkv_tape_elements(shape)};
    }
    if (replay && std::any_of(w.miss.begin(), w.miss.end(),
                              [](const auto &a) { return a.a != nullptr; })) {
      nora_att_shift.resize(size_t(lanes) * w.channels,
                            "NoRA incoming attention shift");
      nora_ffn_shift.resize(size_t(lanes) * w.channels,
                            "NoRA incoming FFN shift");
      view.nora_att_shift = nora_att_shift.p;
      view.nora_ffn_shift = nora_ffn_shift.p;
    }
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
  static constexpr int kOutputTileRows = 1024;
  // B lanes; states/shifts are lane-major within each layer: [L,B,...].
  int L, C, H, V, chunk, B, capacity, last_time = 0;
  size_t lane, states, shift_count;
  cudaStream_t stream;
  DeviceBuffer<bf16> input, ping, pong, norm, logits, gping, gpong,
      shifts, shift_grad, first, first_grad, workspace, terminal_shift,
      nora_input;
  DeviceBuffer<float> nora_ds;
  DeviceBuffer<int> targets, offsets, active_rows_dev;
  DeviceBuffer<float> state_gradient, per_batch_gradient, state_batch,
      state_final, loss, chunk_losses, shared_tape, shared_final, carried,
      weights, reduced_gradient;
  std::vector<OwnedTape> owned;
  std::vector<BlockTapeView> tapes;
  std::vector<BlockBackwardState> gradients;
  std::vector<DeviceBuffer<float>> checkpoints;
  std::vector<DeviceBuffer<bf16>> shift_checkpoints;
  std::vector<uint16_t> host_input;
  std::vector<float> host_losses;
  struct PinnedSlot {
    uint16_t *input = nullptr;
    int *offsets = nullptr;
    int *targets = nullptr;
    int *active_rows = nullptr;
    float *weights = nullptr;
    cudaEvent_t copied = nullptr;
    bool in_flight = false;
  } staging[2];
  int next_slot = 0;
  const bf16 *final_block = nullptr;
  Impl(const FrozenModelView &m, int time, cudaStream_t s, int lanes)
      : L(m.layers), C(m.channels), H(m.heads), V(m.vocab), chunk(time),
        B(lanes), capacity(time * lanes), lane(size_t(H) * 4096),
        states(size_t(L) * lanes * lane),
        shift_count(size_t(L) * 2 * lanes * C), stream(s) {
    if (time <= 0 || lanes <= 0 || L <= 0 || H <= 0 || C != H * 64 ||
        V <= 0 || size_t(time) * lanes > 65535)
      throw std::invalid_argument("invalid BF16 executor dimensions/chunk");
    const size_t rows = size_t(capacity) * C;
    for (auto *b :
         {&input, &ping, &pong, &norm, &gping, &gpong, &first, &first_grad})
      b->resize(rows, "executor activations");
    for (auto *b : {&shifts, &shift_grad, &terminal_shift})
      b->resize(shift_count, "executor shifts");
    const int logit_rows = std::max(chunk, std::min(capacity, kOutputTileRows));
    logits.resize(size_t(logit_rows) * V, "executor tiled logits");
    targets.resize(capacity, "executor targets");
    weights.resize(capacity, "CE row weights");
    active_rows_dev.resize(capacity, "active CE row indices");
    offsets.resize(B + 1, "packed lane offsets");
    for (auto *b : {&state_gradient, &per_batch_gradient, &carried})
      b->resize(states, "executor states");
    if (B > 1)
      reduced_gradient.resize(size_t(L) * lane, "executor lane-summed dState");
    for (auto *b : {&state_batch, &state_final, &shared_final})
      b->resize(B * lane, "executor state scratch");
    loss.resize(1, "executor loss");
    shared_tape.resize(wkv_tape_elements({B, chunk, H, 64}),
                       "shared WKV replay tape");
    for (auto &slot : staging) {
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.input),
                              rows * sizeof(uint16_t), cudaHostAllocDefault),
                "alloc pinned input");
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.offsets),
                              (B + 1) * sizeof(int), cudaHostAllocDefault),
                "alloc pinned offsets");
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.targets),
                              capacity * sizeof(int), cudaHostAllocDefault),
                "alloc pinned targets");
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.active_rows),
                              capacity * sizeof(int), cudaHostAllocDefault),
                "alloc pinned active rows");
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.weights),
                              capacity * sizeof(float), cudaHostAllocDefault),
                "alloc pinned weights");
      gpu_check(cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming),
                "create staging event");
    }
    owned.resize(L);
    tapes.resize(L);
    gradients.resize(L);
    size_t scratch = 0;
    for (int l = 0; l < L; ++l) {
      owned[l].allocate(capacity, m.blocks[l], first.p, shared_tape.p,
                        shared_final.p, true, B);
      tapes[l] = owned[l].view;
      scratch = std::max(
          {scratch,
           block_forward_workspace_bf16_elements(1, capacity, m.blocks[l]),
           block_backward_workspace_bf16_elements(1, capacity, m.blocks[l])});
      const size_t state_at = size_t(l) * B * lane, shift_at = size_t(l) * 2 * B * C;
      gradients[l] = {state_gradient.p + state_at,
                      per_batch_gradient.p + state_at,
                      l == 0 ? nullptr : shift_grad.p + shift_at,
                      shift_grad.p + shift_at + size_t(B) * C,
                      first_grad.p,
                      l == 0 ? nullptr : terminal_shift.p + shift_at,
                      terminal_shift.p + shift_at + size_t(B) * C};
    }
    size_t nora_in = 0, nora_rank = 0;
    for (const auto &b : m.blocks)
      for (const auto &a : b.miss)
        if (a.a) {
          nora_in = std::max(nora_in, size_t(capacity) * a.in);
          nora_rank = std::max(nora_rank, size_t(capacity) * a.rank);
        }
    if (nora_in) {
      nora_input.resize(nora_in, "NoRA shared input replay");
      nora_ds.resize(nora_rank, "NoRA shared latent adjoint");
      for (auto &t : tapes) {
        t.nora_input = nora_input.p;
        t.nora_ds = nora_ds.p;
      }
    }
    workspace.resize(scratch, "executor block workspace");
  }
  ~Impl() {
    for (auto &slot : staging) {
      if (slot.in_flight && slot.copied)
        cudaEventSynchronize(slot.copied);
      if (slot.copied)
        cudaEventDestroy(slot.copied);
      if (slot.input)
        cudaFreeHost(slot.input);
      if (slot.offsets)
        cudaFreeHost(slot.offsets);
      if (slot.targets)
        cudaFreeHost(slot.targets);
      if (slot.active_rows)
        cudaFreeHost(slot.active_rows);
      if (slot.weights)
        cudaFreeHost(slot.weights);
    }
  }
  PinnedSlot &acquire_slot() {
    PinnedSlot &s = staging[next_slot++ & 1];
    if (s.in_flight) {
      gpu_check(cudaEventSynchronize(s.copied), "wait staging slot");
      s.in_flight = false;
    }
    return s;
  }
  void record_slot(PinnedSlot &s) {
    gpu_check(cudaEventRecord(s.copied, stream), "record staging slot");
    s.in_flight = true;
  }
  void copy(void *to, const void *from, size_t n) {
    gpu_check(cudaMemcpyAsync(to, from, n, cudaMemcpyDeviceToDevice, stream),
              "copy boundary");
  }
  void carry_shift_gradient() {
    // Layer 0 attention shifts depend only on frozen embeddings. Their
    // gradients are not requested; carry only the written, live suffix.
    const size_t first = size_t(B) * C;
    copy(terminal_shift.p + first, shift_grad.p + first,
         (shift_count - first) * sizeof(bf16));
  }
  void reserve_checkpoints(int count) {
    if (checkpoints.size() < size_t(count)) {
      checkpoints.resize(count);
      shift_checkpoints.resize(count);
    }
    for (int j = 0; j < count; ++j) {
      checkpoints[j].resize(states, "WKV boundary checkpoint");
      shift_checkpoints[j].resize(shift_count, "shift boundary checkpoint");
    }
    if (chunk_losses.n < size_t(count))
      chunk_losses.resize(count, "chunk losses");
    if (host_losses.size() < size_t(count))
      host_losses.resize(count);
  }
  void zero(void *to, size_t n) {
    gpu_check(cudaMemsetAsync(to, 0, n, stream), "clear adjoint");
  }
  void sync() {
    gpu_check(cudaStreamSynchronize(stream), "BF16 execution");
    for (auto &slot : staging)
      slot.in_flight = false;
  }
  void single_lane() const {
    if (B != 1)
      throw std::logic_error("single-sequence executor call needs lanes=1");
  }
  // Advances every lane by its packed rows [lane_offsets[b],
  // lane_offsets[b + 1]) of tokens. Lanes without rows keep their state.
  void packed_forward(const FrozenModelView &m, const std::vector<int> &tokens,
                      const std::vector<int> &lane_offsets, bool with_logits,
                      bool sync_host = true, bool record_tape = true) {
    const int rows = int(tokens.size());
    if (rows <= 0 || rows > capacity || int(lane_offsets.size()) != B + 1 ||
        lane_offsets.front() != 0 || lane_offsets.back() != rows)
      throw std::invalid_argument("packed forward rows");
    PinnedSlot &slot = acquire_slot();
    for (int t = 0; t < rows; ++t)
      std::copy_n(m.cpu_emb_ln0_bf16 + size_t(tokens[t]) * C, C,
                  slot.input + size_t(t) * C);
    std::copy(lane_offsets.begin(), lane_offsets.end(), slot.offsets);
    gpu_check(cudaMemcpyAsync(input.p, slot.input, size_t(rows) * C * 2,
                              cudaMemcpyHostToDevice, stream),
              "upload packed embeddings");
    gpu_check(cudaMemcpyAsync(offsets.p, slot.offsets,
                              lane_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice, stream),
              "upload lane offsets");
    record_slot(slot);
    final_block = model_forward_state_tuning_bf16(
        stream, L, B, rows, m.blocks.data(), tapes.data(), input.p, carried.p,
        shifts.p, m.ln_out_weight, m.ln_out_bias, m.head_weight_orig, V, ping.p,
        pong.p, state_batch.p, state_final.p, workspace.p, workspace.n, norm.p,
        logits.p, carried.p, with_logits, offsets.p, record_tape);
    last_time = 0;
    if (sync_host)
      sync();
  }
  void forward(const FrozenModelView &m, const int *tokens, int T,
               LogitsMode mode = LogitsMode::All, bool sync_host = true,
               bool record_tape = true) {
    single_lane();
    if (T <= 0 || T > chunk)
      throw std::invalid_argument("forward chunk length");
    PinnedSlot &slot = acquire_slot();
    for (int t = 0; t < T; ++t) {
      if (tokens[t] < 0 || tokens[t] >= V)
        throw std::invalid_argument("token out of range");
      std::copy_n(m.cpu_emb_ln0_bf16 + size_t(tokens[t]) * C, C,
                  slot.input + size_t(t) * C);
    }
    gpu_check(cudaMemcpyAsync(input.p, slot.input, size_t(T) * C * 2,
                              cudaMemcpyHostToDevice, stream),
              "upload embeddings");
    record_slot(slot);
    for (auto &t : tapes)
      t.time = T;
    final_block = model_forward_state_tuning_bf16(
        stream, L, 1, T, m.blocks.data(), tapes.data(), input.p, carried.p,
        shifts.p, m.ln_out_weight, m.ln_out_bias, m.head_weight_orig, V, ping.p,
        pong.p, state_batch.p, state_final.p, workspace.p, workspace.n, norm.p,
        logits.p, carried.p, mode == LogitsMode::All, nullptr, record_tape);
    if (mode == LogitsMode::Last) {
      rwkv7_v3a_layer_norm_bf16_launch(
          stream, 1, C, final_block + size_t(T - 1) * C, m.ln_out_weight,
          m.ln_out_bias, norm.p, 1.0e-5f);
      rwkv7_v3a_linear_bf16_orig_launch(stream, 1, C, V, norm.p,
                                        m.head_weight_orig, logits.p);
    }
    last_time = mode == LogitsMode::None ? 0 : mode == LogitsMode::Last ? 1 : T;
    if (sync_host)
      sync();
  }
  void supervised_output_backward(const FrozenModelView &m, int rows,
                                  const int *chunk_targets,
                                  const float *scaled_row_weights,
                                  float temperature, float *chunk_loss) {
    PinnedSlot &slot = acquire_slot();
    int active = 0;
    for (int r = 0; r < rows; ++r) {
      if (scaled_row_weights[r] != 0.0f) {
        slot.active_rows[active] = r;
        slot.targets[active] = chunk_targets[r];
        slot.weights[active] = scaled_row_weights[r];
        ++active;
      }
    }
    if (active == 0) {
      zero(gping.p, size_t(rows) * C * sizeof(bf16));
      zero(chunk_loss, sizeof(float));
      model_backward_state_only(stream, L, m.blocks.data(), tapes.data(),
                                gradients.data(), gping.p, gping.p, gpong.p,
                                workspace.p, workspace.n, false);
      return;
    }
    if (active == rows) {
      gpu_check(cudaMemcpyAsync(targets.p, slot.targets, rows * sizeof(int),
                                cudaMemcpyHostToDevice, stream),
                "upload targets");
      gpu_check(cudaMemcpyAsync(weights.p, slot.weights, rows * sizeof(float),
                                cudaMemcpyHostToDevice, stream),
                "upload loss weights");
      record_slot(slot);
      for (int off = 0; off < rows; off += kOutputTileRows) {
        const int cnt = std::min(kOutputTileRows, rows - off);
        const bf16 *x_tile = final_block + size_t(off) * C;
        rwkv7_v3a_layer_norm_bf16_launch(stream, cnt, C, x_tile,
                                         m.ln_out_weight, m.ln_out_bias, norm.p,
                                         1.0e-5f);
        rwkv7_v3a_linear_bf16_orig_launch(stream, cnt, C, V, norm.p,
                                          m.head_weight_orig, logits.p);
        cross_entropy_forward_backward_bf16(
            stream, cnt, V, logits.p, targets.p + off, -1, chunk_loss, logits.p,
            float(cnt), weights.p + off, temperature, off > 0);
        linear_orig_backward_input_bf16(stream, cnt, C, V, logits.p,
                                        m.head_weight_orig, gping.p);
        layer_norm_backward_input_bf16(stream, cnt, C, x_tile, m.ln_out_weight,
                                       gping.p, gpong.p + size_t(off) * C,
                                       1.0e-5f);
      }
      model_backward_state_only(stream, L, m.blocks.data(), tapes.data(),
                                gradients.data(), gpong.p, gping.p, gpong.p,
                                workspace.p, workspace.n, false);
      return;
    }
    gpu_check(cudaMemcpyAsync(active_rows_dev.p, slot.active_rows,
                              active * sizeof(int), cudaMemcpyHostToDevice,
                              stream),
              "upload active row indices");
    gpu_check(cudaMemcpyAsync(targets.p, slot.targets, active * sizeof(int),
                              cudaMemcpyHostToDevice, stream),
              "upload active targets");
    gpu_check(cudaMemcpyAsync(weights.p, slot.weights, active * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "upload active loss weights");
    record_slot(slot);
    // Gather only supervised rows, stream through tiled L2-resident logits
    // in-place, then scatter back into [rows, C] with zeros at masked rows.
    rwkv7_gather_rows_bf16_launch(stream, active, C, final_block,
                                  active_rows_dev.p, gping.p);
    for (int off = 0; off < active; off += kOutputTileRows) {
      const int cnt = std::min(kOutputTileRows, active - off);
      const bf16 *x_tile = gping.p + size_t(off) * C;
      bf16 *norm_tile = norm.p + size_t(off) * C;
      rwkv7_v3a_layer_norm_bf16_launch(stream, cnt, C, x_tile, m.ln_out_weight,
                                       m.ln_out_bias, norm_tile, 1.0e-5f);
      rwkv7_v3a_linear_bf16_orig_launch(stream, cnt, C, V, norm_tile,
                                        m.head_weight_orig, logits.p);
      cross_entropy_forward_backward_bf16(
          stream, cnt, V, logits.p, targets.p + off, -1, chunk_loss, logits.p,
          float(cnt), weights.p + off, temperature, off > 0);
      linear_orig_backward_input_bf16(stream, cnt, C, V, logits.p,
                                      m.head_weight_orig, gpong.p);
      layer_norm_backward_input_bf16(stream, cnt, C, x_tile, m.ln_out_weight,
                                     gpong.p, norm_tile, 1.0e-5f);
    }
    rwkv7_scatter_rows_bf16_launch(stream, rows, active, C, norm.p,
                                   active_rows_dev.p, gpong.p);
    model_backward_state_only(stream, L, m.blocks.data(), tapes.data(),
                              gradients.data(), gpong.p, gping.p, gpong.p,
                              workspace.p, workspace.n, false);
  }
};
Executor::Executor(const FrozenModelView &m, int chunk, cudaStream_t s,
                   int lanes)
    : impl_(std::make_unique<Impl>(m, chunk, s, lanes)) {}
Executor::~Executor() = default;
std::vector<BlockTapeView> &Executor::tapes() { return impl_->tapes; }
DeviceBuffer<float> &Executor::state() { return impl_->carried; }
DeviceBuffer<bf16> &Executor::shifts() { return impl_->shifts; }
const float *Executor::state_gradient() const {
  return impl_->B == 1 ? impl_->state_gradient.p : impl_->reduced_gradient.p;
}
const float *Executor::lane_state_gradients() const {
  return impl_->state_gradient.p;
}
int Executor::lanes() const { return impl_->B; }
void Executor::reset(const float *initial) {
  auto &w = *impl_;
  // Broadcast [L,H,K,V] into every lane of [L,B,H,K,V].
  for (int b = 0; b < w.B; ++b)
    gpu_check(cudaMemcpy2DAsync(w.carried.p + size_t(b) * w.lane,
                                size_t(w.B) * w.lane * 4, initial, w.lane * 4,
                                w.lane * 4, w.L, cudaMemcpyDeviceToDevice,
                                w.stream),
              "broadcast initial state");
  w.zero(w.shifts.p, w.shift_count * 2);
}
void Executor::forward(const FrozenModelView &m, const int *tokens, int count,
                       LogitsMode mode) {
  impl_->forward(m, tokens, count, mode);
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
void Executor::reserve_sequence(int tokens) {
  if (tokens < 1)
    throw std::invalid_argument("reserve_sequence needs positive tokens");
  impl_->reserve_checkpoints(1 + (tokens - 1) / impl_->chunk);
}
float Executor::backward(const FrozenModelView &m,
                         const std::vector<int> &tokens,
                         const std::vector<float> &weights,
                         const float *initial, size_t normalizer,
                         float temperature, const std::vector<int> *label_ids) {
  auto &w = *impl_;
  w.single_lane();
  if (tokens.size() < 2 || weights.size() != tokens.size() || !normalizer ||
      !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("backward tokens/weights/normalizer");
  for (size_t i = 0; i < tokens.size(); ++i)
    if (tokens[i] < 0 || tokens[i] >= w.V || !std::isfinite(weights[i]))
      throw std::invalid_argument("invalid backward token/weight");
  if (label_ids) {
    if (label_ids->size() != tokens.size()) throw std::invalid_argument("label size mismatch");
    for (const int id : *label_ids) if (id < 0 || id >= w.V) throw std::invalid_argument("invalid label id");
  }
  const auto &labels = label_ids ? *label_ids : tokens;
  // A suffix after the last nonzero target has no loss and no terminal
  // adjoint. Earlier masked context must still participate in full BPTT.
  int length = int(tokens.size()) - 1;
  while (length > 0 && weights[length] == 0)
    --length;
  w.zero(w.state_gradient.p, w.states * 4);
  w.zero(w.terminal_shift.p, w.shift_count * 2);
  if (!length) {
    w.sync();
    return 0;
  }
  const int chunks = 1 + (length - 1) / w.chunk;
  w.reserve_checkpoints(chunks);
  reset(initial);
  for (int j = 0; j < chunks; ++j) {
    w.copy(w.checkpoints[j].p, w.carried.p, w.states * 4);
    w.copy(w.shift_checkpoints[j].p, w.shifts.p, w.shift_count * 2);
    if (j + 1 < chunks)
      w.forward(m, tokens.data() + j * w.chunk, w.chunk, LogitsMode::None,
                false, false);
  }
  std::vector<float> scaled_weights;
  for (int j = chunks - 1; j >= 0; --j) {
    int off = j * w.chunk, T = std::min(w.chunk, length - off);
    w.copy(w.carried.p, w.checkpoints[j].p, w.states * 4);
    w.copy(w.shifts.p, w.shift_checkpoints[j].p, w.shift_count * 2);
    const bool supervised =
        std::any_of(weights.begin() + off + 1, weights.begin() + off + T + 1,
                    [](float x) { return x != 0; });
    w.forward(m, tokens.data() + off, T, LogitsMode::None, false);
    if (supervised) {
      const float lane_scale = float(T) / normalizer / T;
      scaled_weights.resize(T);
      for (int t = 0; t < T; ++t)
        scaled_weights[t] = lane_scale * weights[off + 1 + t];
      w.supervised_output_backward(m, T, labels.data() + off + 1,
                                   scaled_weights.data(), temperature,
                                   w.chunk_losses.p + j);
    } else {
      // No local CE/head derivative, but recurrent and shift adjoints from
      // later supervised chunks still flow through every frozen block.
      w.zero(w.gping.p, size_t(T) * w.C * sizeof(bf16));
      w.zero(w.chunk_losses.p + j, sizeof(float));
      model_backward_state_only(w.stream, w.L, m.blocks.data(), w.tapes.data(),
                                w.gradients.data(), w.gping.p, w.gping.p,
                                w.gpong.p, w.workspace.p, w.workspace.n, false);
    }
    w.copy(w.state_gradient.p, w.per_batch_gradient.p, w.states * 4);
    if (j > 0)
      w.carry_shift_gradient();
  }
  gpu_check(cudaMemcpyAsync(w.host_losses.data(), w.chunk_losses.p,
                            chunks * sizeof(float), cudaMemcpyDeviceToHost,
                            w.stream),
            "read chunk losses");
  w.sync();
  float total = 0;
  for (int j = chunks - 1; j >= 0; --j) {
    const float loss = w.host_losses[j];
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite loss; optimizer not executed");
    total += loss;
  }
  return total;
}
float Executor::backward_packed(const FrozenModelView &m,
                                const std::vector<std::vector<int>> &tokens,
                                const std::vector<std::vector<float>> &weights,
                                const float *initial, size_t normalizer,
                                float temperature,
                                const std::vector<std::vector<int>> *label_ids) {
  auto &w = *impl_;
  const int n = int(tokens.size());
  if (n < 1 || n > w.B || weights.size() != tokens.size() || !normalizer ||
      !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("packed backward samples/normalizer");
  if (label_ids) {
    if (label_ids->size() != tokens.size()) throw std::invalid_argument("packed label size mismatch");
    for (size_t b = 0; b < tokens.size(); ++b) {
      if ((*label_ids)[b].size() != tokens[b].size()) throw std::invalid_argument("packed label length mismatch");
      for (const int id : (*label_ids)[b]) if (id < 0 || id >= w.V) throw std::invalid_argument("invalid packed label id");
    }
  }
  // Per lane, the same trimmed length as backward(): positions after the last
  // weighted target carry neither loss nor adjoint.
  std::vector<int> length(w.B, 0);
  int steps = 0;
  for (int b = 0; b < n; ++b) {
    const auto &tok = tokens[b];
    const auto &wt = weights[b];
    if (tok.size() < 2 || wt.size() != tok.size())
      throw std::invalid_argument("packed backward tokens/weights");
    for (size_t i = 0; i < tok.size(); ++i)
      if (tok[i] < 0 || tok[i] >= w.V || !std::isfinite(wt[i]))
        throw std::invalid_argument("invalid backward token/weight");
    int len = int(tok.size()) - 1;
    while (len > 0 && wt[len] == 0)
      --len;
    length[b] = len;
    steps = std::max(steps, (len + w.chunk - 1) / w.chunk);
  }
  w.zero(w.state_gradient.p, w.states * 4);
  w.zero(w.terminal_shift.p, w.shift_count * 2);
  float *summed = w.B == 1 ? w.state_gradient.p : w.reduced_gradient.p;
  if (!steps) {
    w.zero(summed, size_t(w.L) * w.lane * 4);
    w.sync();
    return 0;
  }
  // Step j holds lane b's tokens [j*chunk, j*chunk + T_b), packed lane-major.
  auto pack = [&](int j, std::vector<int> &offsets, std::vector<int> *inputs,
                  std::vector<int> *targets, std::vector<float> *row_weights) {
    offsets.assign(1, 0);
    if (inputs)
      inputs->clear();
    if (targets) {
      targets->clear();
      row_weights->clear();
    }
    for (int b = 0; b < w.B; ++b) {
      const int off = j * w.chunk;
      const int T = std::clamp(length[b] - off, 0, w.chunk);
      // backward() scales a T-row chunk by (T / normalizer) / T before the
      // row weight. Folding that same float product into each row (the CE
      // scale becomes rows / rows == 1) keeps packed rows bitwise equal to
      // per-sample rows rather than off by an ulp of a BF16 dLogit.
      const float lane_scale = float(T) / normalizer / T;
      for (int t = off; t < off + T; ++t) {
        if (inputs)
          inputs->push_back(tokens[b][t]);
        if (targets) {
          targets->push_back(label_ids ? (*label_ids)[b][t + 1] : tokens[b][t + 1]);
          row_weights->push_back(lane_scale * weights[b][t + 1]);
        }
      }
      offsets.push_back(offsets.back() + T);
    }
  };
  w.reserve_checkpoints(steps);
  reset(initial);
  std::vector<int> offsets, inputs, targets;
  std::vector<float> row_weights;
  for (int j = 0; j < steps; ++j) {
    w.copy(w.checkpoints[j].p, w.carried.p, w.states * 4);
    w.copy(w.shift_checkpoints[j].p, w.shifts.p, w.shift_count * 2);
    if (j + 1 < steps) {
      pack(j, offsets, &inputs, nullptr, nullptr);
      w.packed_forward(m, inputs, offsets, false, false, false);
    }
  }
  for (int j = steps - 1; j >= 0; --j) {
    pack(j, offsets, &inputs, &targets, &row_weights);
    const int rows = int(inputs.size());
    w.copy(w.carried.p, w.checkpoints[j].p, w.states * 4);
    w.copy(w.shifts.p, w.shift_checkpoints[j].p, w.shift_count * 2);
    const bool supervised = std::any_of(row_weights.begin(), row_weights.end(),
                                        [](float x) { return x != 0; });
    w.packed_forward(m, inputs, offsets, false, false);
    if (supervised) {
      w.supervised_output_backward(m, rows, targets.data(), row_weights.data(),
                                   temperature, w.chunk_losses.p + j);
    } else {
      w.zero(w.gping.p, size_t(rows) * w.C * sizeof(bf16));
      w.zero(w.chunk_losses.p + j, sizeof(float));
      model_backward_state_only(w.stream, w.L, m.blocks.data(), w.tapes.data(),
                                w.gradients.data(), w.gping.p, w.gping.p,
                                w.gpong.p, w.workspace.p, w.workspace.n, false);
    }
    w.copy(w.state_gradient.p, w.per_batch_gradient.p, w.states * 4);
    if (j > 0)
      w.carry_shift_gradient();
  }
  gpu_check(cudaMemcpyAsync(w.host_losses.data(), w.chunk_losses.p,
                            steps * sizeof(float), cudaMemcpyDeviceToHost,
                            w.stream),
            "read packed chunk losses");
  if (w.B > 1)
    for (int l = 0; l < w.L; ++l)
      reduce_state_gradient_f32(w.stream, w.B, w.H, 64,
                                w.state_gradient.p + size_t(l) * w.B * w.lane,
                                summed + size_t(l) * w.lane, false);
  w.sync();
  float total = 0;
  for (int j = steps - 1; j >= 0; --j) {
    const float loss = w.host_losses[j];
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite loss; optimizer not executed");
    total += loss;
  }
  return total;
}
std::vector<float> Executor::score(const FrozenModelView &m,
                                   const std::vector<int> &tokens,
                                   const float *initial, float temperature) {
  auto &w = *impl_;
  w.single_lane();
  if (tokens.size() < 2)
    throw std::invalid_argument("score needs at least two tokens");
  const int length = int(tokens.size()) - 1;
  std::vector<float> result(length);
  DeviceBuffer<float> out;
  out.resize(w.chunk, "teacher-forced log probabilities");
  reset(initial);
  for (int off = 0; off < length; off += w.chunk) {
    const int T = std::min(w.chunk, length - off);
    w.forward(m, tokens.data() + off, T);
    gpu_check(cudaMemcpyAsync(w.targets.p, tokens.data() + off + 1,
                              T * sizeof(int), cudaMemcpyHostToDevice,
                              w.stream),
              "upload score targets");
    target_logprobs_bf16(w.stream, T, w.V, w.logits.p, w.targets.p, out.p,
                         temperature);
    gpu_check(cudaMemcpyAsync(result.data() + off, out.p, T * sizeof(float),
                              cudaMemcpyDeviceToHost, w.stream),
              "download log probabilities");
    w.sync();
  }
  return result;
}

struct BatchDecoder::Impl {
  int L, C, H, V, B;
  size_t lane;
  cudaStream_t stream;
  DeviceBuffer<bf16> input, ping, pong, norm, logits, shifts, first, workspace;
  DeviceBuffer<float> carried, state_batch, state_final;
  std::vector<OwnedTape> owned;
  std::vector<BlockTapeView> tapes;
  std::vector<uint16_t> host_input;
  bool ready = false;
  Impl(const FrozenModelView &m, int rows, cudaStream_t s)
      : L(m.layers), C(m.channels), H(m.heads), V(m.vocab), B(rows),
        lane(size_t(H) * 4096), stream(s) {
    if (rows <= 0 || L <= 0 || H <= 0 || C != H * 64 || V <= 0)
      throw std::invalid_argument("invalid BF16 batch decoder dimensions");
    const size_t activations = size_t(B) * C;
    for (auto *b : {&input, &ping, &pong, &norm, &first})
      b->resize(activations, "decoder activations");
    logits.resize(size_t(B) * V, "decoder logits");
    shifts.resize(size_t(L) * 2 * B * C, "decoder shifts");
    carried.resize(size_t(L) * B * lane, "decoder states");
    for (auto *b : {&state_batch, &state_final})
      b->resize(size_t(B) * lane, "decoder state scratch");
    carried.zero("zero decoder states");
    shifts.zero("zero decoder shifts");
    owned.resize(L);
    tapes.resize(L);
    size_t scratch = 0;
    for (int l = 0; l < L; ++l) {
      // One token per row: a [B,1] step needs B rows of activation storage.
      owned[l].allocate(B, m.blocks[l], first.p, nullptr, nullptr, false);
      tapes[l] = owned[l].view;
      scratch = std::max(
          scratch, block_forward_workspace_bf16_elements(B, 1, m.blocks[l]));
    }
    workspace.resize(scratch, "decoder block workspace");
  }
  void sync() const {
    gpu_check(cudaStreamSynchronize(stream), "BF16 batch decode");
  }
};
BatchDecoder::BatchDecoder(const FrozenModelView &m, int rows, cudaStream_t s)
    : impl_(std::make_unique<Impl>(m, rows, s)) {}
BatchDecoder::~BatchDecoder() = default;
int BatchDecoder::rows() const { return impl_->B; }
void BatchDecoder::load(int row, const float *state, const bf16 *shifts) {
  auto &w = *impl_;
  if (row < 0 || row >= w.B || !state || !shifts)
    throw std::invalid_argument("decoder row");
  // Layer-major layouts: one strided 2D copy per tensor.
  gpu_check(cudaMemcpy2DAsync(w.carried.p + size_t(row) * w.lane,
                              size_t(w.B) * w.lane * 4, state, w.lane * 4,
                              w.lane * 4, w.L, cudaMemcpyDeviceToDevice,
                              w.stream),
            "load decoder state");
  gpu_check(cudaMemcpy2DAsync(w.shifts.p + size_t(row) * w.C,
                              size_t(w.B) * w.C * 2, shifts, size_t(w.C) * 2,
                              size_t(w.C) * 2, size_t(w.L) * 2,
                              cudaMemcpyDeviceToDevice, w.stream),
            "load decoder shifts");
}
void BatchDecoder::download(int row, std::vector<float> &state,
                            std::vector<bf16> &shifts) const {
  auto &w = *impl_;
  if (row < 0 || row >= w.B)
    throw std::invalid_argument("decoder row");
  state.resize(size_t(w.L) * w.lane);
  shifts.resize(size_t(w.L) * 2 * w.C);
  gpu_check(cudaMemcpy2DAsync(state.data(), w.lane * 4,
                              w.carried.p + size_t(row) * w.lane,
                              size_t(w.B) * w.lane * 4, w.lane * 4, w.L,
                              cudaMemcpyDeviceToHost, w.stream),
            "download decoder state");
  gpu_check(cudaMemcpy2DAsync(
                shifts.data(), size_t(w.C) * 2, w.shifts.p + size_t(row) * w.C,
                size_t(w.B) * w.C * 2, size_t(w.C) * 2, size_t(w.L) * 2,
                cudaMemcpyDeviceToHost, w.stream),
            "download decoder shifts");
  w.sync();
}
void BatchDecoder::forward(const FrozenModelView &m, const int *tokens) {
  auto &w = *impl_;
  w.host_input.resize(size_t(w.B) * w.C);
  for (int b = 0; b < w.B; ++b) {
    if (tokens[b] < 0 || tokens[b] >= w.V)
      throw std::invalid_argument("token out of range");
    std::copy_n(m.cpu_emb_ln0_bf16 + size_t(tokens[b]) * w.C, w.C,
                w.host_input.data() + size_t(b) * w.C);
  }
  gpu_check(cudaMemcpyAsync(w.input.p, w.host_input.data(),
                            w.host_input.size() * 2, cudaMemcpyHostToDevice,
                            w.stream),
            "upload decoder embeddings");
  model_forward_state_tuning_bf16(
      w.stream, w.L, w.B, 1, m.blocks.data(), w.tapes.data(), w.input.p,
      w.carried.p, w.shifts.p, m.ln_out_weight, m.ln_out_bias,
      m.head_weight_orig, w.V, w.ping.p, w.pong.p, w.state_batch.p,
      w.state_final.p, w.workspace.p, w.workspace.n, w.norm.p, w.logits.p,
      w.carried.p);
  w.ready = true;
  w.sync(); // host input staging may be reused after returning
}
std::vector<float> BatchDecoder::logits() const {
  auto &w = *impl_;
  if (!w.ready)
    throw std::logic_error("no decoder logits");
  std::vector<bf16> bits(size_t(w.B) * w.V);
  gpu_check(cudaMemcpyAsync(bits.data(), w.logits.p, bits.size() * 2,
                            cudaMemcpyDeviceToHost, w.stream),
            "download decoder logits");
  w.sync();
  std::vector<float> result(bits.size());
  for (size_t i = 0; i < bits.size(); ++i)
    result[i] = to_float(bits[i]);
  return result;
}
} // namespace rwkv_bf16_training
