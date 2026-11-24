#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
namespace rwkv_bf16_training {
namespace {
struct OwnedTape {
  DeviceBuffer<bf16> x, ln1, r, raw_w, k, v_base, v, alpha, neg_kk, wkv_k, kka,
      w1_tanh, g1_sigmoid, v_gate, wkv_y, att_group_norm, att_gate, x_after_att,
      ln2, ffn_hid, xr, xk, xv, att_post, ffn_mixed, nora_att_shift,
      nora_ffn_shift;
  DeviceBuffer<float> wkv_initial;
  std::array<DeviceBuffer<float>, rwkv_bf16_miss::TargetCount> miss_reduced;
  BlockTapeView view;

  void allocate(int time, const FrozenBlockWeights &w, bf16 *v_first,
                float *shared_tape, std::size_t shared_tape_elems,
                bool replay = true, float *shared_final = nullptr,
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
      alloc(xr, "tune tape xr");
      alloc(xk, "tune tape xk");
      alloc(xv, "tune tape xv");
      alloc(att_post, "tune tape att_post");
      alloc(ffn_mixed, "tune tape ffn_mixed");
      if (shared_final) {
        wkv_initial.resize(static_cast<std::size_t>(lanes) * w.heads * 64 * 64,
                           "tune WKV replay initial");
        view.wkv_replay_initial = wkv_initial.p;
        view.wkv_replay_final = shared_final;
        nora_att_shift.resize(size_t(lanes) * w.channels,
                              "NoRA incoming attention shift");
        nora_ffn_shift.resize(size_t(lanes) * w.channels,
                              "NoRA incoming FFN shift");
        view.nora_att_shift = nora_att_shift.p;
        view.nora_ffn_shift = nora_ffn_shift.p;
      } else {
        view.wkv_replay_initial = nullptr;
        view.wkv_replay_final = nullptr;
      }
      view.wkv = {shared_tape, shared_tape_elems};
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
    view.xr = xr.p;
    view.xk = xk.p;
    view.xv = xv.p;
    view.att_post = att_post.p;
    view.ffn_mixed = ffn_mixed.p;
    if (!replay) {
      for (int i = 0; i < rwkv_bf16_miss::TargetCount; ++i)
        if (w.miss[i].d && !w.miss[i].gradient) {
          miss_reduced[i].resize(size_t(time) * w.miss[i].rank,
                                 "frozen MiSS reduced");
          view.miss_reduced[i] = miss_reduced[i].p;
        }
    }
  }
};

struct FrozenMissLayerScratch {
  std::array<DeviceBuffer<float>, rwkv_bf16_miss::TargetCount> miss_reduced;
};

struct DeviceHwInfo {
  int sm_count = 108;
  int max_l2_lanes = 6;
};

DeviceHwInfo query_device_hw(int heads) {
  static thread_local int cached_dev = -1;
  static thread_local DeviceHwInfo cached{};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess)
    dev = 0;
  if (dev != cached_dev) {
    int sms = 0, l2 = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess ||
        sms <= 0)
      sms = 108;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev) !=
            cudaSuccess ||
        l2 <= 0)
      l2 = 40 * 1024 * 1024;
    const size_t per_lane =
        size_t(heads) * rwkv7_state_tuning::kWkvCheckpointInterval * 64 * 64 * 4;
    const size_t budget = (size_t(l2) * 7) / 8;
    const int max_lanes =
        std::clamp(int(budget / (per_lane ? per_lane : 1)), 2, 16);
    cached_dev = dev;
    cached = {sms, max_lanes};
  }
  return cached;
}

int wkv_sm_waves(int lanes, int heads, const DeviceHwInfo &hw) {
  if (lanes <= 0)
    return 0;
  auto waves_of = [&](int b) {
    return (b * heads + hw.sm_count - 1) / hw.sm_count;
  };
  if (lanes <= hw.max_l2_lanes)
    return waves_of(lanes);
  constexpr int kMaxB = 64;
  if (lanes <= kMaxB) {
    int dp[kMaxB + 1];
    dp[0] = 0;
    for (int r = 1; r <= lanes; ++r) {
      dp[r] = 1 << 28;
      const int lim = std::min(r, hw.max_l2_lanes);
      for (int s = lim; s >= 1; --s)
        dp[r] = std::min(dp[r], waves_of(s) + dp[r - s]);
    }
    return dp[lanes];
  }
  const int full = lanes / hw.max_l2_lanes;
  const int rem = lanes % hw.max_l2_lanes;
  return full * waves_of(hw.max_l2_lanes) + (rem ? waves_of(rem) : 0);
}

} // namespace
struct Executor::Impl {
  static constexpr int kOutputTileRows = 1024;
  int L, C, H, V, chunk, B, capacity, last_time = 0;
  size_t lane, states, shift_count;
  cudaStream_t stream;
  DeviceBuffer<bf16> input, ping, pong, norm, logits, gping, gpong,
      shifts, shift_grad, first, first_grad, workspace, terminal_shift,
      nora_input, replay_shifts;
  DeviceBuffer<float> nora_ds;
  DeviceBuffer<int> targets, offsets, active_rows_dev;
  DeviceBuffer<uint8_t> doc_start_dev, win_doc_start;
  DeviceBuffer<float> state_gradient, per_batch_gradient, doc_state_grad,
      doc_init_dev, state_batch, state_final, loss, chunk_losses, shared_tape,
      aux_wkv_tape, shared_final, carried, weights, reduced_gradient;
  OwnedTape shared_owned, aux_owned;
  FrozenBlockWeights proto_weights{};
  bool aux_allocated = false;
  std::vector<OwnedTape> full_owned;
  std::vector<FrozenMissLayerScratch> frozen_miss_scratch;
  std::vector<BlockTapeView> tapes;
  std::vector<BlockBackwardState> gradients;
  std::vector<DeviceBuffer<float>> checkpoints;
  std::vector<DeviceBuffer<bf16>> shift_checkpoints;
  // Windowed 1-pass layer-input cache: stores layers 1..L-1 (layer 0 uses
  // uploaded embeddings directly) and skips dead ffn_value GEMMs on recompute.
  DeviceBuffer<bf16> x_window, d_out_window, win_shift_init;
  DeviceBuffer<float> win_wkv_init;
  DeviceBuffer<int> win_offsets;
  int win_chunk_cap = 0;
  size_t win_slot_rows = 0; // total (layer_slot * row) capacity in x_window
  size_t win_dout_cap = 0;  // total row capacity in d_out_window / win_doc_start
  std::vector<uint16_t> host_input;
  std::vector<float> host_losses;
  struct PinnedSlot {
    uint16_t *input = nullptr;
    int *offsets = nullptr;
    uint8_t *doc_start = nullptr;
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
    replay_shifts.resize(size_t(4) * B * C, "executor layer replay shifts");
    const int logit_rows = std::max(chunk, std::min(capacity, kOutputTileRows));
    logits.resize(size_t(logit_rows) * V, "executor tiled logits");
    targets.resize(capacity, "executor targets");
    weights.resize(capacity, "CE row weights");
    active_rows_dev.resize(capacity, "active CE row indices");
    offsets.resize(B + 1, "packed lane offsets");
    doc_start_dev.resize(capacity, "packed document start flags");
    for (auto *b :
         {&state_gradient, &per_batch_gradient, &doc_state_grad, &carried})
      b->resize(states, "executor states");
    doc_init_dev.resize(size_t(L) * lane, "executor doc initial state");
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
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&slot.doc_start),
                              capacity * sizeof(uint8_t), cudaHostAllocDefault),
                "alloc pinned doc_start");
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
    proto_weights = m.blocks[0];
    for (int l = 1; l < L; ++l) {
      proto_weights.ffn = std::max(proto_weights.ffn, m.blocks[l].ffn);
      proto_weights.rank_w = std::max(proto_weights.rank_w, m.blocks[l].rank_w);
      proto_weights.rank_a = std::max(proto_weights.rank_a, m.blocks[l].rank_a);
      proto_weights.rank_g = std::max(proto_weights.rank_g, m.blocks[l].rank_g);
      proto_weights.rank_v = std::max(proto_weights.rank_v, m.blocks[l].rank_v);
    }
    shared_owned.allocate(capacity, proto_weights, first.p, shared_tape.p,
                          shared_tape.n, true);
    frozen_miss_scratch.resize(L);
    tapes.resize(L);
    gradients.resize(L);
    size_t scratch = 0;
    for (int l = 0; l < L; ++l) {
      tapes[l] = shared_owned.view;
      tapes[l].nora_att_shift = replay_shifts.p;
      tapes[l].nora_ffn_shift = replay_shifts.p + size_t(B) * C;
      for (int i = 0; i < rwkv_bf16_miss::TargetCount; ++i) {
        if (m.blocks[l].miss[i].d && !m.blocks[l].miss[i].gradient) {
          frozen_miss_scratch[l].miss_reduced[i].resize(
              size_t(capacity) * m.blocks[l].miss[i].rank,
              "frozen MiSS reduced");
          tapes[l].miss_reduced[i] =
              frozen_miss_scratch[l].miss_reduced[i].p;
        }
      }
      scratch = std::max(
          {scratch,
           block_forward_workspace_bf16_elements(1, capacity, m.blocks[l]),
           block_backward_workspace_bf16_elements(1, capacity, m.blocks[l])});
      const size_t state_at = size_t(l) * B * lane,
                   shift_at = size_t(l) * 2 * B * C;
      gradients[l] = {state_gradient.p + state_at,
                      per_batch_gradient.p + state_at,
                      l == 0 ? nullptr : shift_grad.p + shift_at,
                      shift_grad.p + shift_at + size_t(B) * C,
                      first_grad.p,
                      l == 0 ? nullptr : terminal_shift.p + shift_at,
                      terminal_shift.p + shift_at + size_t(B) * C,
                      nullptr};
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
    reserve_coarse_checkpoints(2);
  }
  ~Impl() {
    for (auto &slot : staging) {
      if (slot.in_flight && slot.copied)
        rwkv_gpu::check_cleanup(cudaEventSynchronize(slot.copied), "cudaEventSynchronize");
      if (slot.copied)
        rwkv_gpu::check_cleanup(cudaEventDestroy(slot.copied), "cudaEventDestroy");
      if (slot.input)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.input), "cudaFreeHost");
      if (slot.offsets)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.offsets), "cudaFreeHost");
      if (slot.doc_start)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.doc_start), "cudaFreeHost");
      if (slot.targets)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.targets), "cudaFreeHost");
      if (slot.active_rows)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.active_rows), "cudaFreeHost");
      if (slot.weights)
        rwkv_gpu::check_cleanup(cudaFreeHost(slot.weights), "cudaFreeHost");
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
    const size_t first_off = size_t(B) * C;
    copy(terminal_shift.p + first_off, shift_grad.p + first_off,
         (shift_count - first_off) * sizeof(bf16));
  }
  void reserve_coarse_checkpoints(int count) {
    if (checkpoints.size() < size_t(count)) {
      checkpoints.resize(count);
      shift_checkpoints.resize(count);
    }
    for (int j = 0; j < count; ++j) {
      checkpoints[j].resize(states, "WKV coarse window checkpoint");
      shift_checkpoints[j].resize(shift_count, "shift coarse window checkpoint");
    }
  }
  void reserve_loss_slots(int steps) {
    if (chunk_losses.n < size_t(steps))
      chunk_losses.resize(steps, "chunk losses");
    if (host_losses.size() < size_t(steps))
      host_losses.resize(steps);
  }
  void ensure_window_pool(int needed_chunks, size_t needed_rows) {
    needed_chunks = std::max(1, needed_chunks);
    needed_rows = std::max<size_t>(size_t(capacity), needed_rows);
    const size_t slots_s2 = size_t(std::max(1, (L - 1) / 2));
    const size_t cur_max_rows =
        std::min(win_dout_cap, win_slot_rows / slots_s2);
    if (needed_chunks <= win_chunk_cap && needed_rows <= cur_max_rows)
      return;

    const size_t dout_bytes_per_row = C * sizeof(bf16) + 1;
    const size_t bytes_per_chunk =
        states * sizeof(float) + shift_count * sizeof(bf16) +
        size_t(B + 1) * sizeof(int);

    const int old_chunk_cap = win_chunk_cap;
    const size_t old_dout_cap = win_dout_cap;
    win_chunk_cap = 0;
    win_dout_cap = win_slot_rows = 0;
    x_window = DeviceBuffer<bf16>{};
    d_out_window = DeviceBuffer<bf16>{};
    win_doc_start = DeviceBuffer<uint8_t>{};
    win_wkv_init = DeviceBuffer<float>{};
    win_shift_init = DeviceBuffer<bf16>{};
    win_offsets = DeviceBuffer<int>{};

    size_t free_b = 0, total_b = 0;
    gpu_check(cudaMemGetInfo(&free_b, &total_b), "query VRAM for window pool");
    size_t safety_reserve =
        std::max<size_t>(size_t(7) << 29, total_b / 28);
    if (const char *res_env = std::getenv("RWKV_VRAM_RESERVE_MIB")) {
      if (*res_env)
        safety_reserve = std::max<size_t>(
            safety_reserve, static_cast<size_t>(std::stoull(res_env)) << 20);
    }
    const size_t budget_b =
        free_b > safety_reserve ? (free_b - safety_reserve) : 0;

    const int max_afford_chunks = std::max(
        1, int((budget_b * 2 / 5) / (bytes_per_chunk ? bytes_per_chunk : 1)));
    const int target_chunks =
        std::max(1, std::min(std::max(old_chunk_cap, needed_chunks),
                             std::min(max_afford_chunks, 56)));
    const size_t chunk_bytes = size_t(target_chunks) * bytes_per_chunk;
    const size_t rem_after_chunks =
        budget_b > chunk_bytes ? (budget_b - chunk_bytes) : 0;

    // Allocate d_out + doc_start up to 327,680 rows (~2.5 GiB) and dedicate all
    // remaining VRAM to x_window so stride=1 covers ~200k+ rows and stride=2
    // covers ~327k+ rows in a single 1-pass window (even on 80G H100!).
    const size_t max_dout_afford = std::max<size_t>(
        size_t(capacity),
        rem_after_chunks / (dout_bytes_per_row + slots_s2 * C * sizeof(bf16)));
    const size_t target_dout = std::max<size_t>(
        size_t(capacity),
        std::min(std::max(needed_rows, old_dout_cap), max_dout_afford));
    const size_t dout_bytes = target_dout * dout_bytes_per_row;
    const size_t x_budget =
        rem_after_chunks > dout_bytes ? (rem_after_chunks - dout_bytes) : 0;
    const size_t slots_s1 = size_t(std::max(1, L - 1));
    const size_t max_slot_afford =
        std::max<size_t>(slots_s1 * size_t(capacity),
                         x_budget / (size_t(C) * sizeof(bf16)));
    const size_t target_slot_rows =
        std::min(max_slot_afford, slots_s1 * target_dout);

    DeviceBuffer<bf16> x, dout, shifts;
    DeviceBuffer<uint8_t> starts;
    DeviceBuffer<float> wkv;
    DeviceBuffer<int> offsets;
    x.resize(target_slot_rows * C, "window layer-input x pool");
    dout.resize(target_dout * C, "window head d_out pool");
    starts.resize(target_dout, "window doc_start pool");
    wkv.resize(size_t(target_chunks) * states, "window layer WKV initial states");
    shifts.resize(size_t(target_chunks) * shift_count,
                  "window layer incoming shifts");
    offsets.resize(size_t(target_chunks) * (B + 1), "window packed offsets");

    x_window = std::move(x);
    d_out_window = std::move(dout);
    win_doc_start = std::move(starts);
    win_wkv_init = std::move(wkv);
    win_shift_init = std::move(shifts);
    win_offsets = std::move(offsets);
    win_chunk_cap = target_chunks;
    win_dout_cap = target_dout;
    win_slot_rows = target_slot_rows;
  }
  bf16 *win_x_ptr(int stride, size_t eff_cap, int l, size_t row_off) {
    const size_t slot = (stride == 1) ? size_t(l - 1) : size_t(l / 2 - 1);
    return x_window.p + (slot * eff_cap + row_off) * C;
  }
  bf16 *win_dout_ptr(size_t row_off) { return d_out_window.p + row_off * C; }
  uint8_t *win_doc_ptr(size_t row_off) { return win_doc_start.p + row_off; }
  float *win_wkv_ptr(int slot, int l) {
    return win_wkv_init.p + (size_t(slot) * L + l) * B * lane;
  }
  bf16 *win_shift_ptr(int slot, int l) {
    return win_shift_init.p + (size_t(slot) * L + l) * 2 * B * C;
  }
  int *win_offsets_ptr(int slot) {
    return win_offsets.p + size_t(slot) * (B + 1);
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
  void upload_embeddings_and_offsets(const FrozenModelView &m, const int *toks,
                                     int rows, const int *host_offs,
                                     int offs_count, int *dev_offs_dst,
                                     const uint8_t *host_doc = nullptr,
                                     uint8_t *dev_doc_dst = nullptr) {
    PinnedSlot &slot = acquire_slot();
    for (int t = 0; t < rows; ++t) {
      if (toks[t] < 0 || toks[t] >= V)
        throw std::invalid_argument("token out of range");
      std::copy_n(m.cpu_emb_ln0_bf16 + size_t(toks[t]) * C, C,
                  slot.input + size_t(t) * C);
    }
    gpu_check(cudaMemcpyAsync(input.p, slot.input, size_t(rows) * C * 2,
                              cudaMemcpyHostToDevice, stream),
              "upload embeddings");
    if (host_offs && dev_offs_dst && offs_count > 0) {
      std::copy_n(host_offs, offs_count, slot.offsets);
      gpu_check(cudaMemcpyAsync(dev_offs_dst, slot.offsets,
                                size_t(offs_count) * sizeof(int),
                                cudaMemcpyHostToDevice, stream),
                "upload lane offsets");
    }
    if (host_doc && dev_doc_dst && rows > 0) {
      std::copy_n(host_doc, rows, slot.doc_start);
      gpu_check(cudaMemcpyAsync(dev_doc_dst, slot.doc_start,
                                size_t(rows) * sizeof(uint8_t),
                                cudaMemcpyHostToDevice, stream),
                "upload doc_start flags");
    }
    record_slot(slot);
  }
  static void assign_tape_activations(BlockTapeView &dst,
                                      const BlockTapeView &src) {
    const auto saved_miss = dst.miss_reduced;
    bf16 *const saved_nora_in = dst.nora_input;
    float *const saved_nora_ds = dst.nora_ds;
    dst = src;
    dst.miss_reduced = saved_miss;
    dst.nora_input = saved_nora_in;
    dst.nora_ds = saved_nora_ds;
  }
  void ensure_aux_tape() {
    if (aux_allocated || L <= 1)
      return;
    aux_wkv_tape.resize(wkv_tape_elements({B, chunk, H, 64}),
                        "aux WKV replay tape");
    aux_owned.allocate(capacity, proto_weights, first.p, aux_wkv_tape.p,
                       aux_wkv_tape.n, true);
    aux_allocated = true;
  }
  void bind_window_tapes(int x_stride) {
    for (int l = 0; l < L; ++l) {
      const bool use_aux = (x_stride == 2) && (L > 1) && ((l & 1) != 0);
      assign_tape_activations(tapes[l],
                              use_aux ? aux_owned.view : shared_owned.view);
      tapes[l].nora_att_shift =
          replay_shifts.p + (use_aux ? size_t(2) * B * C : 0);
      tapes[l].nora_ffn_shift =
          replay_shifts.p + (use_aux ? size_t(3) * B * C : size_t(B) * C);
    }
  }
  void ensure_full_tapes(const FrozenModelView &m) {
    if (full_owned.empty()) {
      std::vector<OwnedTape> pending(L);
      for (int l = 0; l < L; ++l)
        pending[l].allocate(capacity, m.blocks[l], first.p, shared_tape.p,
                            shared_tape.n, true, shared_final.p, B);
      full_owned.swap(pending);
    }
    for (int l = 0; l < L; ++l)
      assign_tape_activations(tapes[l], full_owned[l].view);
  }
  void forward(const FrozenModelView &m, const int *tokens, int T,
               LogitsMode mode = LogitsMode::All, bool sync_host = true,
               bool record_tape = true) {
    single_lane();
    if (T <= 0 || T > chunk)
      throw std::invalid_argument("forward chunk length");
    upload_embeddings_and_offsets(m, tokens, T, nullptr, 0, nullptr);
    if (record_tape) {
      ensure_full_tapes(m);
      for (auto &t : tapes) {
        t.time = T;
        t.doc_initial_state = nullptr;
      }
    } else {
      for (auto &t : tapes) {
        t.x = shared_owned.x.p;
        t.time = T;
        t.doc_initial_state = nullptr;
      }
    }
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
  void compute_output_head_grad(const FrozenModelView &m, int rows,
                                const bf16 *final_act, const int *chunk_targets,
                                const float *scaled_row_weights,
                                float temperature, float *chunk_loss_out,
                                bf16 *d_out_dst) {
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
      zero(d_out_dst, size_t(rows) * C * sizeof(bf16));
      zero(chunk_loss_out, sizeof(float));
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
        const bf16 *x_tile = final_act + size_t(off) * C;
        rwkv7_v3a_layer_norm_bf16_launch(stream, cnt, C, x_tile,
                                         m.ln_out_weight, m.ln_out_bias, norm.p,
                                         1.0e-5f);
        rwkv7_v3a_linear_bf16_orig_launch(stream, cnt, C, V, norm.p,
                                          m.head_weight_orig, logits.p);
        cross_entropy_forward_backward_bf16(
            stream, cnt, V, logits.p, targets.p + off, -1, chunk_loss_out,
            logits.p, float(cnt), weights.p + off, temperature, off > 0);
        linear_orig_backward_input_bf16(stream, cnt, C, V, logits.p,
                                        m.head_weight_orig, gping.p);
        layer_norm_backward_input_bf16(stream, cnt, C, x_tile, m.ln_out_weight,
                                       gping.p, d_out_dst + size_t(off) * C,
                                       1.0e-5f);
      }
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
    rwkv7_gather_rows_bf16_launch(stream, active, C, final_act,
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
          stream, cnt, V, logits.p, targets.p + off, -1, chunk_loss_out,
          logits.p, float(cnt), weights.p + off, temperature, off > 0);
      linear_orig_backward_input_bf16(stream, cnt, C, V, logits.p,
                                      m.head_weight_orig, gpong.p);
      layer_norm_backward_input_bf16(stream, cnt, C, x_tile, m.ln_out_weight,
                                     gpong.p, norm_tile, 1.0e-5f);
    }
    rwkv7_scatter_rows_bf16_launch(stream, rows, active, C, norm.p,
                                   active_rows_dev.p, d_out_dst);
  }
  void compute_layer0_v_first(const FrozenBlockWeights &w0, int B_arg,
                              int T_arg, int rows, const bf16 *x0,
                              const bf16 *att_shift0, const int *dev_offs,
                              const uint8_t *dev_doc) {
    rwkv7_v3a_layer_norm_bf16_launch(stream, rows, C, x0, w0.ln1_weight,
                                     w0.ln1_bias, shared_owned.ln1.p, 1.0e-5f);
    copy(replay_shifts.p, att_shift0, size_t(B_arg) * C * sizeof(bf16));
    rwkv7_cmix_mix_launch(stream, B_arg, T_arg, C, shared_owned.ln1.p,
                          replay_shifts.p, w0.mix_v, shared_owned.xv.p,
                          dev_offs, dev_doc);
    rwkv7_v3a_linear_bf16_orig_launch(stream, rows, C, C, shared_owned.xv.p,
                                      w0.value, first.p);
    rwkv_bf16_miss::forward(stream, rows, w0.miss[2], shared_owned.xv.p,
                            first.p, nullptr);
  }
  struct ChunkSpec {
    int rows = 0;
    int active_lanes = 1;
    bool packed = false;
    bool has_doc_start = false;
    std::vector<int> offsets;
    std::vector<uint8_t> doc_start;
    std::vector<int> inputs;
    std::vector<int> targets;
    std::vector<float> weights;
  };
  float execute_windowed_backward(const FrozenModelView &m,
                                  const std::vector<ChunkSpec> &chunks_vec,
                                  const float *initial, float temperature,
                                  Executor &outer, bool any_doc_start) {
    const int steps = int(chunks_vec.size());
    size_t total_rows = 0;
    for (const auto &c : chunks_vec)
      total_rows += size_t(c.rows);
    reserve_loss_slots(steps);
    ensure_window_pool(steps, total_rows);

    const size_t slots_s1 = size_t(std::max(1, L - 1));
    size_t cap_s1 = std::min(win_dout_cap, win_slot_rows / slots_s1);
    int x_stride = (L <= 1 || total_rows <= cap_s1) ? 1 : 2;
    if (x_stride == 2 && !aux_allocated && L > 1) {
      win_chunk_cap = 0;
      win_dout_cap = 0;
      win_slot_rows = 0;
      x_window = DeviceBuffer<bf16>{};
      d_out_window = DeviceBuffer<bf16>{};
      win_doc_start = DeviceBuffer<uint8_t>{};
      ensure_aux_tape();
      ensure_window_pool(steps, total_rows);
      cap_s1 = std::min(win_dout_cap, win_slot_rows / slots_s1);
    }
    const size_t cap_s2 =
        (L <= 2) ? win_dout_cap
                 : std::min(win_dout_cap, win_slot_rows / size_t((L - 1) / 2));
    const size_t eff_row_cap = (x_stride == 1) ? cap_s1 : cap_s2;

    struct Window {
      int begin = 0, end = 0;
    };
    std::vector<Window> windows;
    for (int j = 0; j < steps;) {
      int k = j;
      size_t rsum = 0;
      while (k < steps && (k - j) < win_chunk_cap &&
             rsum + size_t(chunks_vec[k].rows) <= eff_row_cap) {
        rsum += size_t(chunks_vec[k].rows);
        ++k;
      }
      if (k == j)
        throw std::logic_error("window capacity smaller than single chunk");
      windows.push_back({j, k});
      j = k;
    }

    const int num_win = int(windows.size());
    reserve_coarse_checkpoints(num_win);
    bind_window_tapes(x_stride);
    outer.reset(initial);

    if (any_doc_start) {
      copy(doc_init_dev.p, initial, size_t(L) * lane * sizeof(float));
      zero(doc_state_grad.p, states * sizeof(float));
      for (int l = 0; l < L; ++l) {
        tapes[l].doc_initial_state = doc_init_dev.p + size_t(l) * lane;
        gradients[l].doc_state_grad = doc_state_grad.p + size_t(l) * B * lane;
      }
    } else {
      for (int l = 0; l < L; ++l) {
        tapes[l].doc_initial_state = nullptr;
        gradients[l].doc_state_grad = nullptr;
      }
    }

    zero(per_batch_gradient.p, states * sizeof(float));
    zero(shift_grad.p, shift_count * sizeof(bf16));

    // Coarse Pass 1: ONLY runs when total_rows exceeds window pool (num_win > 1).
    for (int g = 0; g < num_win; ++g) {
      copy(checkpoints[g].p, carried.p, states * sizeof(float));
      copy(shift_checkpoints[g].p, shifts.p, shift_count * sizeof(bf16));
      if (g + 1 < num_win) {
        for (int j = windows[g].begin; j < windows[g].end; ++j) {
          const auto &ch = chunks_vec[j];
          const int B_arg = ch.packed ? ch.active_lanes : 1;
          upload_embeddings_and_offsets(
              m, ch.inputs.data(), ch.rows,
              ch.packed ? ch.offsets.data() : nullptr,
              ch.packed ? (B_arg + 1) : 0, ch.packed ? offsets.p : nullptr,
              ch.has_doc_start ? ch.doc_start.data() : nullptr,
              ch.has_doc_start ? doc_start_dev.p : nullptr);
          const int *offs_arg = ch.packed ? offsets.p : nullptr;
          const uint8_t *doc_arg = ch.has_doc_start ? doc_start_dev.p : nullptr;
          for (auto &t : tapes)
            t.x = shared_owned.x.p;
          // Advance each block explicitly with B_arg active lanes while keeping
          // full [L, B, ...] strides on carried and shifts.
          copy(ping.p, input.p, size_t(ch.rows) * C * sizeof(bf16));
          const bf16 *cur_act = ping.p;
          bf16 *nxt_act = pong.p;
          for (int l = 0; l < L; ++l) {
            float *wkv_state = carried.p + size_t(l) * B * lane;
            bf16 *att_s = shifts.p + size_t(l) * 2 * B * C;
            bf16 *ffn_s = att_s + size_t(B) * C;
            block_forward_state_tuning_bf16(
                stream, B_arg, ch.rows, cur_act, att_s, ffn_s, wkv_state,
                wkv_state, first.p, nxt_act, m.blocks[l], tapes[l], workspace.p,
                workspace.n, l == 0, offs_arg, false, doc_arg);
            cur_act = nxt_act;
            nxt_act = (nxt_act == ping.p) ? pong.p : ping.p;
          }
        }
      }
    }

    std::vector<size_t> slot_row_off;
    for (int g = num_win - 1; g >= 0; --g) {
      const int c_begin = windows[g].begin;
      const int c_end = windows[g].end;
      const int win_len = c_end - c_begin;
      slot_row_off.assign(win_len, 0);
      copy(carried.p, checkpoints[g].p, states * sizeof(float));
      copy(shifts.p, shift_checkpoints[g].p, shift_count * sizeof(bf16));

      size_t cur_off = 0;
      for (int j = c_begin; j < c_end; ++j) {
        const int slot = j - c_begin;
        const auto &ch = chunks_vec[j];
        slot_row_off[slot] = cur_off;
        const int B_arg = ch.packed ? ch.active_lanes : 1;
        int *dev_offs = ch.packed ? win_offsets_ptr(slot) : nullptr;
        uint8_t *dev_doc = ch.has_doc_start ? win_doc_ptr(cur_off) : nullptr;
        upload_embeddings_and_offsets(
            m, ch.inputs.data(), ch.rows,
            ch.packed ? ch.offsets.data() : nullptr,
            ch.packed ? (B_arg + 1) : 0, dev_offs,
            ch.has_doc_start ? ch.doc_start.data() : nullptr, dev_doc);
        copy(ping.p, input.p, size_t(ch.rows) * C * sizeof(bf16));
        const bf16 *cur_act = ping.p;
        bf16 *nxt_act = pong.p;
        for (int l = 0; l < L; ++l) {
          float *wkv_state = carried.p + size_t(l) * B * lane;
          bf16 *att_s = shifts.p + size_t(l) * 2 * B * C;
          bf16 *ffn_s = att_s + size_t(B) * C;
          if (l > 0 && (x_stride == 1 || (l & 1) == 0)) {
            copy(win_x_ptr(x_stride, eff_row_cap, l, cur_off), cur_act,
                 size_t(ch.rows) * C * sizeof(bf16));
          }
          copy(win_wkv_ptr(slot, l), wkv_state,
               size_t(B_arg) * lane * sizeof(float));
          copy(win_shift_ptr(slot, l), att_s,
               size_t(B_arg) * C * sizeof(bf16));
          copy(win_shift_ptr(slot, l) + size_t(B) * C, ffn_s,
               size_t(B_arg) * C * sizeof(bf16));
          tapes[l].x = shared_owned.x.p;
          block_forward_state_tuning_bf16(
              stream, B_arg, ch.rows, cur_act, att_s, ffn_s, wkv_state,
              wkv_state, first.p, nxt_act, m.blocks[l], tapes[l], workspace.p,
              workspace.n, l == 0, dev_offs, false, dev_doc);
          cur_act = nxt_act;
          nxt_act = (nxt_act == ping.p) ? pong.p : ping.p;
        }
        compute_output_head_grad(m, ch.rows, cur_act, ch.targets.data(),
                                 ch.weights.data(), temperature,
                                 chunk_losses.p + j, win_dout_ptr(cur_off));
        cur_off += size_t(ch.rows);
      }

      // Reverse chunk & layer backward over window g (stride=1 or stride=2):
      for (int j = c_end - 1; j >= c_begin; --j) {
        const int slot = j - c_begin;
        const auto &ch = chunks_vec[j];
        const size_t r_off = slot_row_off[slot];
        const int B_arg = ch.packed ? ch.active_lanes : 1;
        const int *dev_offs = ch.packed ? win_offsets_ptr(slot) : nullptr;
        const uint8_t *dev_doc =
            ch.has_doc_start ? win_doc_ptr(r_off) : nullptr;

        upload_embeddings_and_offsets(m, ch.inputs.data(), ch.rows, nullptr, 0,
                                      nullptr);
        copy(gping.p, win_dout_ptr(r_off), size_t(ch.rows) * C * sizeof(bf16));
        if (L > 1) {
          zero(first_grad.p, size_t(ch.rows) * C * sizeof(bf16));
          compute_layer0_v_first(m.blocks[0], B_arg, ch.rows, ch.rows, input.p,
                                 win_shift_ptr(slot, 0), dev_offs, dev_doc);
        }
        auto recompute_layer = [&](int l, bf16 *x_in, bf16 *y_out) {
          const size_t sh_off =
              ((x_stride == 2) && (L > 1) && ((l & 1) != 0))
                  ? size_t(2) * B * C
                  : 0;
          bf16 *att_rep = replay_shifts.p + sh_off;
          bf16 *ffn_rep = att_rep + size_t(B) * C;
          copy(att_rep, win_shift_ptr(slot, l),
               size_t(B_arg) * C * sizeof(bf16));
          copy(ffn_rep, win_shift_ptr(slot, l) + size_t(B) * C,
               size_t(B_arg) * C * sizeof(bf16));
          tapes[l].x = x_in;
          block_forward_state_tuning_bf16(
              stream, B_arg, ch.rows, x_in, att_rep, ffn_rep,
              win_wkv_ptr(slot, l), shared_final.p, first.p, y_out, m.blocks[l],
              tapes[l], workspace.p, workspace.n, l == 0, dev_offs, true,
              dev_doc);
        };
        const bf16 *cur_grad = gping.p;
        bf16 *nxt_grad = gpong.p;
        if (x_stride == 1) {
          for (int l = L - 1; l >= 0; --l) {
            bf16 *x_l =
                (l == 0) ? input.p : win_x_ptr(1, eff_row_cap, l, r_off);
            recompute_layer(l, x_l, nullptr);
            bf16 *grad_in = (l == 0) ? nullptr : nxt_grad;
            if (L > 1)
              gradients[l].v_first_grad = first_grad.p;
            block_backward_state_only(stream, cur_grad, grad_in, gradients[l],
                                      tapes[l], m.blocks[l], workspace.p,
                                      workspace.n, l == 0);
            cur_grad = grad_in;
            nxt_grad = (nxt_grad == gping.p) ? gpong.p : gping.p;
          }
        } else {
          int l_top = L - 1;
          if ((L & 1) != 0) {
            bf16 *x_l = (l_top == 0)
                            ? input.p
                            : win_x_ptr(2, eff_row_cap, l_top, r_off);
            recompute_layer(l_top, x_l, nullptr);
            bf16 *grad_in = (l_top == 0) ? nullptr : nxt_grad;
            if (L > 1)
              gradients[l_top].v_first_grad = first_grad.p;
            block_backward_state_only(
                stream, cur_grad, grad_in, gradients[l_top], tapes[l_top],
                m.blocks[l_top], workspace.p, workspace.n, l_top == 0);
            cur_grad = grad_in;
            nxt_grad = (nxt_grad == gping.p) ? gpong.p : gping.p;
            --l_top;
          }
          for (int l0 = l_top - 1; l0 >= 0; l0 -= 2) {
            const int l1 = l0 + 1;
            bf16 *x_l0 =
                (l0 == 0) ? input.p : win_x_ptr(2, eff_row_cap, l0, r_off);
            recompute_layer(l0, x_l0, ping.p);
            recompute_layer(l1, ping.p, nullptr);
            if (L > 1) {
              gradients[l1].v_first_grad = first_grad.p;
              gradients[l0].v_first_grad = first_grad.p;
            }
            block_backward_state_only(stream, cur_grad, nxt_grad, gradients[l1],
                                      tapes[l1], m.blocks[l1], workspace.p,
                                      workspace.n, false);
            cur_grad = nxt_grad;
            nxt_grad = (nxt_grad == gping.p) ? gpong.p : gping.p;
            bf16 *grad_in0 = (l0 == 0) ? nullptr : nxt_grad;
            block_backward_state_only(stream, cur_grad, grad_in0, gradients[l0],
                                      tapes[l0], m.blocks[l0], workspace.p,
                                      workspace.n, l0 == 0);
            cur_grad = grad_in0;
            nxt_grad = (nxt_grad == gping.p) ? gpong.p : gping.p;
          }
        }
        copy(state_gradient.p, per_batch_gradient.p, states * sizeof(float));
        if (j > 0)
          carry_shift_gradient();
      }
    }

    gpu_check(cudaMemcpyAsync(host_losses.data(), chunk_losses.p,
                              steps * sizeof(float), cudaMemcpyDeviceToHost,
                              stream),
              "read chunk losses");
    const float *grad_src = any_doc_start ? doc_state_grad.p : state_gradient.p;
    if (B > 1) {
      float *summed = reduced_gradient.p;
      for (int l = 0; l < L; ++l)
        reduce_state_gradient_f32(stream, B, H, 64,
                                  grad_src + size_t(l) * B * lane,
                                  summed + size_t(l) * lane, false);
    } else if (any_doc_start) {
      copy(state_gradient.p, doc_state_grad.p, states * sizeof(float));
    }
    sync();
    float total = 0;
    for (int j = steps - 1; j >= 0; --j) {
      const float lval = host_losses[j];
      if (!std::isfinite(lval))
        throw std::runtime_error("nonfinite loss; optimizer not executed");
      total += lval;
    }
    return total;
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
void Executor::prefill(const FrozenModelView &m, const int *tokens, int count,
                       LogitsMode mode) {
  impl_->forward(m, tokens, count, mode, true, false);
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
  const int typ_chunks =
      std::min(1 + (tokens - 1) / impl_->chunk, std::max(8, 18432 / impl_->chunk));
  const size_t est_rows =
      std::min<size_t>(size_t(tokens) * size_t(impl_->B), 262144);
  impl_->ensure_window_pool(typ_chunks, est_rows);
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
    if (label_ids->size() != tokens.size())
      throw std::invalid_argument("label size mismatch");
    for (const int id : *label_ids)
      if (id < 0 || id >= w.V)
        throw std::invalid_argument("invalid label id");
  }
  const auto &labels = label_ids ? *label_ids : tokens;
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
  std::vector<Impl::ChunkSpec> specs(chunks);
  for (int j = 0; j < chunks; ++j) {
    const int off = j * w.chunk;
    const int T = std::min(w.chunk, length - off);
    const float lane_scale = float(T) / normalizer / T;
    auto &sp = specs[j];
    sp.rows = T;
    sp.active_lanes = 1;
    sp.packed = false;
    sp.inputs.assign(tokens.begin() + off, tokens.begin() + off + T);
    sp.targets.assign(labels.begin() + off + 1, labels.begin() + off + 1 + T);
    sp.weights.resize(T);
    for (int t = 0; t < T; ++t)
      sp.weights[t] = lane_scale * weights[off + 1 + t];
  }
  return w.execute_windowed_backward(m, specs, initial, temperature, *this,
                                     false);
}
float Executor::backward_packed(const FrozenModelView &m,
                                const std::vector<std::vector<int>> &tokens,
                                const std::vector<std::vector<float>> &weights,
                                const float *initial, size_t normalizer,
                                float temperature,
                                const std::vector<std::vector<int>> *label_ids) {
  auto &w = *impl_;
  const int n = int(tokens.size());
  if (n < 1 || weights.size() != tokens.size() || !normalizer ||
      !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("packed backward samples/normalizer");
  if (label_ids) {
    if (label_ids->size() != tokens.size())
      throw std::invalid_argument("packed label size mismatch");
    for (size_t b = 0; b < tokens.size(); ++b) {
      if ((*label_ids)[b].size() != tokens[b].size())
        throw std::invalid_argument("packed label length mismatch");
      for (const int id : (*label_ids)[b])
        if (id < 0 || id >= w.V)
          throw std::invalid_argument("invalid packed label id");
    }
  }

  std::vector<int> sample_len(n, 0);
  int active_samples = 0;
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
    sample_len[b] = len;
    if (len > 0)
      ++active_samples;
  }

  w.zero(w.state_gradient.p, w.states * 4);
  w.zero(w.terminal_shift.p, w.shift_count * 2);
  float *summed = w.B == 1 ? w.state_gradient.p : w.reduced_gradient.p;
  if (!active_samples) {
    w.zero(summed, size_t(w.L) * w.lane * 4);
    w.sync();
    return 0;
  }

  // Legacy sample-ordered 1-sample-per-lane path when consistent GEMM mode is
  // enabled AND sample count fits in w.B (preserves bit-exact per-lane dState).
  if (consistent_forward_enabled() && n <= w.B) {
    int steps = 0;
    for (int b = 0; b < n; ++b)
      steps = std::max(steps, (sample_len[b] + w.chunk - 1) / w.chunk);
    std::vector<Impl::ChunkSpec> specs(steps);
    for (int j = 0; j < steps; ++j) {
      auto &sp = specs[j];
      sp.packed = true;
      sp.active_lanes = w.B;
      sp.offsets.assign(1, 0);
      for (int b = 0; b < w.B; ++b) {
        const int len = b < n ? sample_len[b] : 0;
        const int off = j * w.chunk;
        const int T = std::clamp(len - off, 0, w.chunk);
        const float lane_scale = float(T) / normalizer / T;
        for (int t = off; t < off + T; ++t) {
          sp.inputs.push_back(tokens[b][t]);
          sp.targets.push_back(label_ids ? (*label_ids)[b][t + 1]
                                         : tokens[b][t + 1]);
          sp.weights.push_back(lane_scale * weights[b][t + 1]);
        }
        sp.offsets.push_back(sp.offsets.back() + T);
      }
      sp.rows = sp.offsets.back();
    }
    return w.execute_windowed_backward(m, specs, initial, temperature, *this,
                                       false);
  }

  // 1D Virtual-Stream LPT Bin-Packing with doc_start state/shift resets:
  // - Eliminates ragged tail starvation by concatenating shorter samples onto
  //   nearly equal-length virtual streams.
  // - Selects virtual stream count S <= w.B that minimizes hardware SM waves +
  //   boundary overhead while keeping GEMMs saturated.
  // - Sorts virtual streams longest-first so each chunk j launches ONLY the
  //   active non-empty prefix B_act(j) <= S!
  std::vector<int> valid_idx;
  valid_idx.reserve(active_samples);
  for (int b = 0; b < n; ++b)
    if (sample_len[b] > 0)
      valid_idx.push_back(b);
  std::stable_sort(valid_idx.begin(), valid_idx.end(), [&](int a, int b) {
    return sample_len[a] > sample_len[b];
  });

  const int max_streams = std::min(int(valid_idx.size()), w.B);
  const DeviceHwInfo hw = query_device_hw(w.H);

  auto pack_onto = [&](int S, std::vector<std::vector<int>> &bins,
                       std::vector<int> &bin_lens) {
    bins.assign(S, {});
    bin_lens.assign(S, 0);
    for (int idx : valid_idx) {
      int best = 0;
      for (int s = 1; s < S; ++s)
        if (bin_lens[s] < bin_lens[best])
          best = s;
      bins[best].push_back(idx);
      bin_lens[best] += sample_len[idx];
    }
    std::vector<int> perm(S);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](int a, int b) { return bin_lens[a] > bin_lens[b]; });
    std::vector<std::vector<int>> sorted_bins(S);
    std::vector<int> sorted_lens(S);
    for (int s = 0; s < S; ++s) {
      sorted_bins[s] = std::move(bins[perm[s]]);
      sorted_lens[s] = bin_lens[perm[s]];
    }
    bins = std::move(sorted_bins);
    bin_lens = std::move(sorted_lens);
  };

  int best_S = max_streams;
  if (!consistent_forward_enabled() && max_streams > 1) {
    double best_cost = 1e30;
    std::vector<std::vector<int>> cand_bins;
    std::vector<int> cand_lens;
    for (int S = 1; S <= max_streams; ++S) {
      pack_onto(S, cand_bins, cand_lens);
      const int c_steps = (cand_lens[0] + w.chunk - 1) / w.chunk;
      double cost = 0.0;
      for (int j = 0; j < c_steps; ++j) {
        const int off = j * w.chunk;
        int b_act = 0;
        int rows_j = 0;
        int max_t = 0;
        for (int s = 0; s < S; ++s) {
          const int t = std::clamp(cand_lens[s] - off, 0, w.chunk);
          if (t > 0) {
            ++b_act;
            rows_j += t;
            max_t = std::max(max_t, t);
          }
        }
        // Each chunk sweeps all 14.4 GB of model weights once (memory-bound
        // below ~4096 rows, compute-bound above ~4096 rows), so merging short
        // samples without increasing chunk count is pure profit while doubling
        // stream length for 1-2 lonely tail streams is properly penalized.
        const double gemm_cost = std::max(double(rows_j), 4096.0);
        const double wkv_cost =
            double(wkv_sm_waves(b_act, w.H, hw)) * double(max_t) * 42.0;
        const double bound_cost = double(b_act) * 180.0 + 600.0;
        cost += gemm_cost + wkv_cost + bound_cost;
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_S = S;
      }
    }
  }

  std::vector<std::vector<int>> stream_samples;
  std::vector<int> stream_lens;
  pack_onto(best_S, stream_samples, stream_lens);

  struct FlatStream {
    std::vector<int> inputs;
    std::vector<int> targets;
    std::vector<float> weights;
    std::vector<uint8_t> doc_start;
  };
  std::vector<FlatStream> streams(best_S);
  const bool need_doc_start = (int(valid_idx.size()) > best_S);
  for (int s = 0; s < best_S; ++s) {
    auto &fs = streams[s];
    fs.inputs.reserve(stream_lens[s]);
    fs.targets.reserve(stream_lens[s]);
    fs.weights.reserve(stream_lens[s]);
    if (need_doc_start)
      fs.doc_start.reserve(stream_lens[s]);
    for (int idx : stream_samples[s]) {
      const int len = sample_len[idx];
      const float scale = 1.0f / float(normalizer);
      for (int t = 0; t < len; ++t) {
        fs.inputs.push_back(tokens[idx][t]);
        fs.targets.push_back(label_ids ? (*label_ids)[idx][t + 1]
                                       : tokens[idx][t + 1]);
        fs.weights.push_back(scale * weights[idx][t + 1]);
        if (need_doc_start)
          fs.doc_start.push_back(t == 0 ? 1 : 0);
      }
    }
  }

  const int steps = (stream_lens[0] + w.chunk - 1) / w.chunk;
  std::vector<Impl::ChunkSpec> specs(steps);
  for (int j = 0; j < steps; ++j) {
    const int off = j * w.chunk;
    int b_act = 0;
    while (b_act < best_S && stream_lens[b_act] > off)
      ++b_act;
    auto &sp = specs[j];
    sp.packed = true;
    sp.active_lanes = b_act;
    sp.has_doc_start = need_doc_start;
    sp.offsets.assign(1, 0);
    for (int s = 0; s < b_act; ++s) {
      const int T = std::clamp(stream_lens[s] - off, 0, w.chunk);
      sp.inputs.insert(sp.inputs.end(), streams[s].inputs.begin() + off,
                       streams[s].inputs.begin() + off + T);
      sp.targets.insert(sp.targets.end(), streams[s].targets.begin() + off,
                        streams[s].targets.begin() + off + T);
      sp.weights.insert(sp.weights.end(), streams[s].weights.begin() + off,
                        streams[s].weights.begin() + off + T);
      if (need_doc_start) {
        sp.doc_start.insert(sp.doc_start.end(),
                            streams[s].doc_start.begin() + off,
                            streams[s].doc_start.begin() + off + T);
      }
      sp.offsets.push_back(sp.offsets.back() + T);
    }
    sp.rows = sp.offsets.back();
  }
  return w.execute_windowed_backward(m, specs, initial, temperature, *this,
                                     need_doc_start);
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
    w.forward(m, tokens.data() + off, T, LogitsMode::All, true, false);
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
      owned[l].allocate(B, m.blocks[l], first.p, nullptr, 0, false);
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
      w.carried.p, true, nullptr, false);
  w.ready = true;
  w.sync();
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
