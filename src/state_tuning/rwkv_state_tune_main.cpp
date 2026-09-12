#include "dataset.hpp"
#include "rwkv/runtime/rwkv_server_backend.hpp"
#include "rwkv/runtime/rwkv_state_tuning.hpp"
#include "rwkv/inference/rwkv_tokenizer.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using rwkv7_fast_v4::DeviceBuffer;
using rwkv7_state_tuning::BlockBackwardState;
using rwkv7_state_tuning::BlockTapeView;
using rwkv7_state_tuning::FrozenBlockWeights;

struct Options {
  std::string model;
  std::string data;
  std::string output;
  std::string vocab = RWKV_STATE_TUNE_DEFAULT_VOCAB;
  int ctx = 128;
  int chunk = 64;
  int epochs = 1;
  int batch_size = 1;
  std::uint64_t max_steps = 0;
  float lr = 1.0f;
  float lr_final = 0.01f;
  std::uint64_t warmup_steps = 10;
  std::uint64_t save_every = 0;
  std::uint64_t seed = 1234;
};

[[noreturn]] void usage(const char *program, const std::string &error = {}) {
  if (!error.empty())
    std::cerr << "error: " << error << "\n\n";
  std::cerr
      << "Usage: " << program << " --model MODEL.pth --data TRAIN.jsonl "
      << "--output DIR [options]\n"
      << "  --vocab PATH          RWKV vocab file\n"
      << "  --ctx N               maximum tokens per JSONL sample (default "
         "128)\n"
      << "  --chunk N             recompute chunk length (default "
         "64)\n"
      << "  --epochs N            dataset passes (default 1)\n"
      << "  --batch-size N        samples per update, accumulated sequentially "
         "(default 1)\n"
      << "  --max-steps N         stop after N optimizer updates\n"
      << "  --lr FLOAT            initial learning rate (default 1.0)\n"
      << "  --lr-final FLOAT      final learning rate (default 0.01)\n"
      << "  --warmup-steps N      linear warmup updates (default 10)\n"
      << "  --save-every N        periodic checkpoint interval (default 0)\n"
      << "  --seed N              deterministic run seed (default 1234)\n";
  std::exit(error.empty() ? 0 : 2);
}

std::string value(int &i, int argc, char **argv) {
  if (++i >= argc)
    usage(argv[0], std::string("missing value for ") + argv[i - 1]);
  return argv[i];
}

template <typename T> T integer(const std::string &text, const char *name) {
  if (text.empty() || text.front() == '-')
    usage("rwkv_state_tune", std::string("invalid ") + name);
  std::size_t used = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(text, &used);
  } catch (...) {
    usage("rwkv_state_tune", std::string("invalid ") + name);
  }
  if (used != text.size() ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
    usage("rwkv_state_tune", std::string("invalid ") + name);
  return static_cast<T>(parsed);
}

float real(const std::string &text, const char *name) {
  std::size_t used = 0;
  float parsed = 0;
  try {
    parsed = std::stof(text, &used);
  } catch (...) {
    usage("rwkv_state_tune", std::string("invalid ") + name);
  }
  if (used != text.size() || !std::isfinite(parsed))
    usage("rwkv_state_tune", std::string("invalid ") + name);
  return parsed;
}

Options parse(int argc, char **argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      usage(argv[0]);
    else if (arg == "--model")
      out.model = value(i, argc, argv);
    else if (arg == "--data")
      out.data = value(i, argc, argv);
    else if (arg == "--output")
      out.output = value(i, argc, argv);
    else if (arg == "--vocab")
      out.vocab = value(i, argc, argv);
    else if (arg == "--ctx")
      out.ctx = integer<int>(value(i, argc, argv), "ctx");
    else if (arg == "--chunk")
      out.chunk = integer<int>(value(i, argc, argv), "chunk");
    else if (arg == "--epochs")
      out.epochs = integer<int>(value(i, argc, argv), "epochs");
    else if (arg == "--batch-size" || arg == "--batch")
      out.batch_size = integer<int>(value(i, argc, argv), "batch-size");
    else if (arg == "--max-steps")
      out.max_steps = integer<std::uint64_t>(value(i, argc, argv), "max-steps");
    else if (arg == "--lr")
      out.lr = real(value(i, argc, argv), "lr");
    else if (arg == "--lr-final")
      out.lr_final = real(value(i, argc, argv), "lr-final");
    else if (arg == "--warmup-steps")
      out.warmup_steps =
          integer<std::uint64_t>(value(i, argc, argv), "warmup-steps");
    else if (arg == "--save-every")
      out.save_every =
          integer<std::uint64_t>(value(i, argc, argv), "save-every");
    else if (arg == "--seed")
      out.seed = integer<std::uint64_t>(value(i, argc, argv), "seed");
    else
      usage(argv[0], "unknown option: " + arg);
  }
  if (out.model.empty() || out.data.empty() || out.output.empty())
    usage(argv[0], "--model, --data and --output are required");
  if (out.ctx <= 0 || out.chunk <= 0 || out.epochs <= 0 ||
      out.batch_size <= 0 || out.lr <= 0 || out.lr_final <= 0)
    usage(argv[0], "ctx/chunk/epochs/lr values must be positive");
  return out;
}

void check(cudaError_t error, const char *what) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(error));
}

struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
          "create training stream");
  }
  ~Stream() {
    if (value)
      cudaStreamDestroy(value);
  }
};

struct OwnedTape {
  DeviceBuffer<half> x, ln1, r, raw_w, k, v_base, v, alpha, neg_kk, wkv_k, kka,
      w1_tanh, g1_sigmoid, v_gate, wkv_y, att_group_norm, att_gate, x_after_att,
      ln2, ffn_hid;
  DeviceBuffer<float> wkv_states;
  BlockTapeView view;

  void allocate(int time, const FrozenBlockWeights &w, half *v_first) {
    const std::size_t row = static_cast<std::size_t>(time) * w.channels;
    auto alloc = [row](DeviceBuffer<half> &buffer, const char *name) {
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
    const rwkv7_state_tuning::WkvShape shape{1, time, w.heads, 64};
    wkv_states.resize(rwkv7_state_tuning::wkv_tape_elements(shape),
                      "tune WKV exact tape");
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
    view.wkv = {wkv_states.p, wkv_states.n};
  }
};

float learning_rate(const Options &o, std::uint64_t next_step,
                    std::uint64_t planned_steps) {
  if (o.warmup_steps && next_step <= o.warmup_steps)
    return o.lr * static_cast<float>(next_step) /
           static_cast<float>(o.warmup_steps);
  const std::uint64_t after_warmup =
      planned_steps > o.warmup_steps ? planned_steps - o.warmup_steps : 1;
  const std::uint64_t position =
      next_step > o.warmup_steps ? next_step - o.warmup_steps : 0;
  const float progress = std::min(1.0f, static_cast<float>(position) /
                                            static_cast<float>(after_warmup));
  return std::exp(std::log(o.lr) +
                  (std::log(o.lr_final) - std::log(o.lr)) * progress);
}

std::string checkpoint_name(const std::filesystem::path &directory,
                            std::uint64_t step) {
  std::ostringstream name;
  name << "state-step-" << std::setw(8) << std::setfill('0') << step << ".pth";
  return (directory / name.str()).string();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    const int max_time = std::min(options.ctx, options.chunk);
    const std::size_t dataset_rows =
        rwkv7_state_tuning::count_jsonl_rows(options.data);
    if (!dataset_rows)
      throw std::runtime_error("dataset has no JSONL rows");
    std::uint64_t planned =
        static_cast<std::uint64_t>(options.epochs) *
        ((dataset_rows + options.batch_size - 1) / options.batch_size);
    if (options.max_steps)
      planned = std::min(planned, options.max_steps);

    rwkv7_server::TrieTokenizer tokenizer;
    if (tokenizer.load(options.vocab) != rwkv7_server::kTokenizerSuccess)
      throw std::runtime_error("failed to load tokenizer: " + options.vocab);
    rwkv7_server::ModelBackend model(options.model, false, false, "off");
    const auto frozen = model.state_tuning_model_view();
    const int L = frozen.layers;
    const int C = frozen.channels;
    const int H = frozen.heads;
    const int V = frozen.vocab;
    if (L <= 0 || C <= 0 || H <= 0 || V <= 0 || C % H != 0 ||
        frozen.blocks.size() != static_cast<std::size_t>(L) ||
        frozen.cpu_emb_ln0_elements != static_cast<std::size_t>(V) * C) {
      throw std::runtime_error("invalid model dimensions for state tuning");
    }
    const int N = C / H;
    if (N != 64)
      throw std::runtime_error("state tuning currently requires head size 64");
    const std::size_t R = static_cast<std::size_t>(max_time) * C;
    const std::size_t state_lane = static_cast<std::size_t>(H) * N * N;

    Stream stream;
    DeviceBuffer<half> input, activation_ping, activation_pong, final_norm,
        logits, d_logits, grad_ping, grad_pong, shifts, shift_grad, v_first,
        v_first_grad, block_workspace;
    DeviceBuffer<int> targets;
    DeviceBuffer<float> time_state, state_gradient, adam_m, adam_v,
        per_batch_state_gradient, state_batch, state_final, loss;
    input.resize(R, "tune input");
    activation_ping.resize(R, "tune activation ping");
    activation_pong.resize(R, "tune activation pong");
    final_norm.resize(R, "tune final norm");
    logits.resize(static_cast<std::size_t>(max_time) * V, "tune logits");
    d_logits.resize(static_cast<std::size_t>(max_time) * V, "tune dlogits");
    grad_ping.resize(R, "tune grad ping");
    grad_pong.resize(R, "tune grad pong");
    shifts.resize(static_cast<std::size_t>(L) * 2 * C, "tune shifts");
    shift_grad.resize(static_cast<std::size_t>(L) * 2 * C,
                      "tune shift gradients");
    v_first.resize(R, "tune v_first");
    v_first_grad.resize(R, "tune v_first gradient");
    targets.resize(max_time, "tune targets");
    const std::size_t state_count = static_cast<std::size_t>(L) * state_lane;
    time_state.resize(state_count, "tune time_state");
    state_gradient.resize(state_count, "tune dState");
    adam_m.resize(state_count, "tune Adam m");
    adam_v.resize(state_count, "tune Adam v");
    per_batch_state_gradient.resize(state_count, "tune per-batch dState");
    state_batch.resize(state_lane, "tune expanded state");
    state_final.resize(state_lane, "tune final state");
    loss.resize(1, "tune loss");
    time_state.zero("zero tune time_state");
    state_gradient.zero("zero tune dState");
    adam_m.zero("zero tune Adam m");
    adam_v.zero("zero tune Adam v");

    std::vector<OwnedTape> owned(static_cast<std::size_t>(L));
    std::vector<BlockTapeView> tapes(static_cast<std::size_t>(L));
    std::vector<BlockBackwardState> backward(static_cast<std::size_t>(L));
    std::size_t workspace_count = 0;
    for (int layer = 0; layer < L; ++layer) {
      owned[static_cast<std::size_t>(layer)].allocate(
          max_time, frozen.blocks[static_cast<std::size_t>(layer)], v_first.p);
      tapes[static_cast<std::size_t>(layer)] =
          owned[static_cast<std::size_t>(layer)].view;
      workspace_count = std::max(
          workspace_count,
          rwkv7_state_tuning::block_forward_workspace_f16_elements(
              1, max_time, frozen.blocks[static_cast<std::size_t>(layer)]));
      workspace_count = std::max(
          workspace_count,
          rwkv7_state_tuning::block_backward_workspace_f16_elements(
              1, max_time, frozen.blocks[static_cast<std::size_t>(layer)]));
      backward[static_cast<std::size_t>(layer)] = {
          nullptr,
          per_batch_state_gradient.p +
              static_cast<std::size_t>(layer) * state_lane,
          shift_grad.p + static_cast<std::size_t>(layer) * 2 * C,
          shift_grad.p + static_cast<std::size_t>(layer) * 2 * C + C,
          v_first_grad.p};
    }
    block_workspace.resize(workspace_count, "tune block workspace");
    std::filesystem::create_directories(options.output);

    // Only chunk boundary recurrent states persist; all activation tapes are
    // overwritten and recomputed when traversing chunks in reverse.
    const std::size_t shift_count = static_cast<std::size_t>(L) * 2 * C;
    DeviceBuffer<float> carried;
    DeviceBuffer<half> terminal_shift;
    carried.resize(state_count, "carried WKV state");
    terminal_shift.resize(shift_count, "terminal shift gradient");
    const int max_chunks = 1 + (options.ctx - 1) / max_time;
    std::vector<DeviceBuffer<float>> checkpoints(max_chunks);
    std::vector<DeviceBuffer<half>> shift_checkpoints(max_chunks);
    for (int j = 0; j < max_chunks; ++j) {
      checkpoints[j].resize(state_count, "WKV boundary checkpoint");
      shift_checkpoints[j].resize(shift_count, "shift boundary checkpoint");
    }
    auto copy_device = [&](void *to, const void *from, std::size_t bytes) {
      check(cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToDevice,
                            stream.value),
            "copy chunk boundary");
    };
    auto zero_device = [&](void *to, std::size_t bytes) {
      check(cudaMemsetAsync(to, 0, bytes, stream.value),
            "clear chunk gradient");
    };
    for (int l = 0; l < L; ++l) {
      backward[l].final_state_grad = state_gradient.p + l * state_lane;
      backward[l].final_att_shift_grad = terminal_shift.p + l * 2 * C;
      backward[l].final_ffn_shift_grad = terminal_shift.p + l * 2 * C + C;
    }

    std::uint64_t global_step = 0;
    std::size_t skipped = 0;
    bool stop = false;
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t processed_tokens = 0;
    std::vector<float> host_gradient(state_count), accumulated(state_count);
    for (int epoch = 1; epoch <= options.epochs && !stop; ++epoch) {
      rwkv7_state_tuning::JsonlTextReader dataset(options.data);
      bool exhausted = false;
      while (!exhausted) {
        std::vector<std::vector<int>> batch;
        std::size_t batch_tokens = 0;
        while (batch.size() < static_cast<std::size_t>(options.batch_size)) {
          std::string text;
          if (!dataset.next(text)) {
            exhausted = true;
            break;
          }
          const auto encoded = tokenizer.encode(text);
          if (encoded.size() < 2) {
            ++skipped;
            continue;
          }
          const std::size_t count = std::min(
              encoded.size(), static_cast<std::size_t>(options.ctx) + 1);
          batch.emplace_back(encoded.begin(), encoded.begin() + count);
          batch_tokens += count - 1;
        }
        if (batch.empty())
          break;
        std::fill(accumulated.begin(), accumulated.end(), 0.0f);
        float batch_loss = 0.0f;
        for (const auto &tokens : batch) {
          const int length = static_cast<int>(tokens.size()) - 1;
          const int chunks = 1 + (length - 1) / max_time;
          copy_device(carried.p, time_state.p, state_count * sizeof(float));
          zero_device(shifts.p, shift_count * sizeof(half));
          std::vector<std::uint16_t> host_input;
          std::vector<int> host_targets;
          auto forward_chunk = [&](int j) -> const half * {
            const int offset = j * max_time;
            const int T = std::min(max_time, length - offset);
            host_input.resize(static_cast<std::size_t>(T) * C);
            host_targets.resize(T);
            for (int t = 0; t < T; ++t) {
              const int token = tokens[offset + t];
              const int target = tokens[offset + t + 1];
              if (token < 0 || token >= V || target < 0 || target >= V)
                throw std::runtime_error("out-of-range token or target");
              const auto *source =
                  frozen.cpu_emb_ln0_f16 + static_cast<std::size_t>(token) * C;
              std::copy(source, source + C,
                        host_input.data() + static_cast<std::size_t>(t) * C);
              host_targets[t] = target;
            }
            check(cudaMemcpyAsync(input.p, host_input.data(),
                                  host_input.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice, stream.value),
                  "copy input");
            check(cudaMemcpyAsync(targets.p, host_targets.data(),
                                  host_targets.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, stream.value),
                  "copy targets");
            for (auto &tape : tapes)
              tape.time = T;
            return rwkv7_state_tuning::model_forward_state_tuning_f16(
                stream.value, L, 1, T, frozen.blocks.data(), tapes.data(),
                input.p, time_state.p, shifts.p, frozen.ln_out_weight,
                frozen.ln_out_bias, frozen.head_weight_orig, V,
                activation_ping.p, activation_pong.p, state_batch.p,
                state_final.p, block_workspace.p, block_workspace.n,
                final_norm.p, logits.p, carried.p);
          };
          for (int j = 0; j < chunks; ++j) {
            copy_device(checkpoints[j].p, carried.p,
                        state_count * sizeof(float));
            copy_device(shift_checkpoints[j].p, shifts.p,
                        shift_count * sizeof(half));
            if (j + 1 < chunks) {
              forward_chunk(j);
              check(cudaStreamSynchronize(stream.value), "checkpoint forward");
            }
          }
          zero_device(state_gradient.p, state_count * sizeof(float));
          zero_device(terminal_shift.p, shift_count * sizeof(half));
          for (int j = chunks - 1; j >= 0; --j) {
            copy_device(carried.p, checkpoints[j].p,
                        state_count * sizeof(float));
            copy_device(shifts.p, shift_checkpoints[j].p,
                        shift_count * sizeof(half));
            const half *final_block = forward_chunk(j);
            const int T = std::min(max_time, length - j * max_time);
            rwkv7_state_tuning::cross_entropy_forward_backward_f16(
                stream.value, T, V, logits.p, targets.p, -1, loss.p, d_logits.p,
                static_cast<float>(T) / batch_tokens);
            rwkv7_state_tuning::model_output_backward_state_only(
                stream.value, L, 1, T, frozen.blocks.data(), tapes.data(),
                backward.data(), final_block, frozen.ln_out_weight,
                frozen.head_weight_orig, V, d_logits.p, grad_ping.p,
                grad_pong.p, block_workspace.p, block_workspace.n);
            copy_device(state_gradient.p, per_batch_state_gradient.p,
                        state_count * sizeof(float));
            copy_device(terminal_shift.p, shift_grad.p,
                        shift_count * sizeof(half));
            float chunk_loss;
            check(cudaMemcpyAsync(&chunk_loss, loss.p, sizeof(float),
                                  cudaMemcpyDeviceToHost, stream.value),
                  "copy chunk loss");
            check(cudaStreamSynchronize(stream.value), "chunk backward");
            if (!std::isfinite(chunk_loss))
              throw std::runtime_error(
                  "nonfinite loss before optimizer; state not updated");
            batch_loss += chunk_loss;
          }
          check(cudaMemcpyAsync(host_gradient.data(), state_gradient.p,
                                state_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream.value),
                "validate initial state gradient");
          check(cudaStreamSynchronize(stream.value), "read state gradient");
          for (std::size_t i = 0; i < state_count; ++i) {
            if (!std::isfinite(host_gradient[i]))
              throw std::runtime_error(
                  "nonfinite dState at layer " +
                  std::to_string(i / state_lane) +
                  "; optimizer not executed, reduce lr or inspect backward");
            accumulated[i] += host_gradient[i];
            if (!std::isfinite(accumulated[i]) ||
                std::abs(accumulated[i]) >
                    std::sqrt(std::numeric_limits<float>::max()))
              throw std::runtime_error(
                  "batch gradient overflow before optimizer");
          }
        }
        check(cudaMemcpyAsync(state_gradient.p, accumulated.data(),
                              state_count * sizeof(float),
                              cudaMemcpyHostToDevice, stream.value),
              "copy batch gradient");
        const std::uint64_t next_step = global_step + 1;
        const float lr = learning_rate(options, next_step, planned);
        rwkv7_state_tuning::AdamConfig adam;
        adam.learning_rate = lr;
        rwkv7_state_tuning::adam_update_state_f32(
            stream.value, time_state.p, state_gradient.p, adam_m.p, adam_v.p,
            state_count, next_step, adam, true);
        check(cudaStreamSynchronize(stream.value), "optimizer update");
        global_step = next_step;
        processed_tokens += batch_tokens;
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        const double fraction =
            std::min(1.0, static_cast<double>(global_step) / planned);
        const int filled = static_cast<int>(fraction * 24);
        std::cout << '\r' << '[' << std::string(filled, '=')
                  << std::string(24 - filled, ' ') << "] " << global_step << '/'
                  << planned << " epoch=" << epoch << '/' << options.epochs
                  << " loss=" << std::fixed << std::setprecision(4)
                  << batch_loss << " batch=" << batch.size()
                  << " tokens=" << batch_tokens
                  << " lr=" << std::setprecision(6) << lr
                  << " tok/s=" << std::setprecision(1)
                  << processed_tokens / elapsed
                  << " ETA=" << std::setprecision(0)
                  << elapsed / global_step * (planned - global_step) << "s   "
                  << std::flush;
        if (options.save_every && global_step % options.save_every == 0) {
          const auto path = checkpoint_name(options.output, global_step);
          rwkv7_state_tuning::save_state_checkpoint_pth(path, stream.value, L,
                                                        H, N, time_state.p);
          std::cout << "\nsaved: " << path << '\n';
        }
        if (options.max_steps && global_step >= options.max_steps) {
          stop = true;
          break;
        }
      }
    }
    std::cout << '\n';
    const std::string final_path =
        (std::filesystem::path(options.output) / "state-final.pth").string();
    rwkv7_state_tuning::save_state_checkpoint_pth(final_path, stream.value, L,
                                                  H, N, time_state.p);
    std::cout << "saved: " << final_path << '\n';
    std::cout << "complete: steps=" << global_step
              << " skipped_short_samples=" << skipped
              << " seed=" << options.seed << '\n';
  } catch (const std::exception &error) {
    std::cerr << "\nrwkv_state_tune: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
