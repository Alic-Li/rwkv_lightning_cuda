#include "../state_tuning/dataset.hpp"
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
#include "rwkv/bf16_training/trainer.hpp"
#endif
#include "rwkv/bf16_training/model.hpp"
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/inference/rwkv_tokenizer.hpp"
#include "rwkv/bf16_training/training.hpp"

#include "rwkv/bf16_training/runtime.hpp"
#include "rwkv/version.hpp"

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

using rwkv_bf16_training::DeviceBuffer;

// Target i of a sample predicts tokens[i + 1]; train[i + 1] says whether it
// contributes to the loss. Untrained prefixes still run forward and backward
// because the tuned initial state receives its gradient through them.
struct Sample {
  std::vector<int> tokens;
  std::vector<unsigned char> train;
};

struct Options {
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
  std::string state, resume, targets = "all";
  int rank = 16;
  float alpha = -1;
#endif
  std::string model;
  std::string data;
  std::string output;
  std::string vocab = RWKV_STATE_TUNE_DEFAULT_VOCAB;
  std::string optimizer = "adam";
  std::string gemm_mode = "bf16";
  bool chunk_load = false;
  int ctx = 128;
  int chunk = 64;
  int epochs = 1;
  int batch_size = 1;
  int micro_bsz = 1;
  std::uint64_t max_steps = 0;
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
  float lr = 1.0e-3f;
#else
  float lr = 1.0f;
#endif
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
  float lr_final = 1.0e-4f;
#else
  float lr_final = 0.01f;
#endif
  std::uint64_t warmup_steps = 10;
  std::uint64_t save_every = 0;
  std::uint64_t seed = 1234;
};

[[noreturn]] void usage(const char *program, const std::string &error = {}) {
  if (!error.empty())
    std::cerr << "error: " << error << "\n\n";
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
#ifdef RWKV_NORA_TUNE
  std::cerr << "NoRA-init: --rank 16 --alpha RANK --targets "
               "all|comma-separated-weight-names\n"
            << "  Random column-normalized A, zero B; Adam trains A and B. "
               "alpha must equal rank.\n"
            << "  --state initial.pth --resume checkpoint-directory\n"
            << "  default lr=0.001 lr-final=0.0001\n";
#else
  std::cerr << "MiSS: --rank 16 --alpha RANK --targets "
               "all|comma-separated-weight-names\n"
            << "      --state initial.pth --resume checkpoint-directory\n"
            << "      default lr=0.001 lr-final=0.0001; Adam updates only D\n";
#endif
#endif
  std::cerr
      << "Usage: " << program << " --model MODEL.pth --data TRAIN.jsonl "
      << "--output DIR [options]\n"
      << "  --wkv_tape            accepted for compatibility; tape replay is "
         "always enabled\n"
      << "  --chunk-load          accepted for compatibility; BF16 loading "
         "always streams\n"
      << "  --vocab PATH          RWKV vocab file\n"
      << "  --ctx N               maximum tokens per JSONL sample (default "
         "128)\n"
      << "  --chunk N             recompute chunk length (default "
         "64)\n"
      << "  --epochs N            dataset passes (default 1)\n"
      << "  --batch-size N        samples per update (default 1)\n"
      << "  --micro-bsz N         samples run together, packed without padding "
         "(default 1;\n"
      << "                        activation memory scales with N * chunk)\n"
      << "  --max-steps N         stop after N optimizer updates\n"
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
      << "  --optimizer adam      adapter optimizer\n"
      << "  --lr FLOAT            initial learning rate (default 0.001)\n"
      << "  --lr-final FLOAT      final learning rate (default 0.0001)\n"
#else
      << "  --optimizer adam|muon optimizer (default adam)\n"
      << "  --lr FLOAT            initial learning rate (default 1.0)\n"
      << "  --lr-final FLOAT      final learning rate (default 0.01)\n"
#endif
      << "  --gemm-mode MODE      bf16 (default): one pinned GEMM kernel per shape,\n"
         "                        so rows (and --micro-bsz packing) never change\n"
         "                        results; strict: same with pedantic FP32;\n"
         "                        auto: cuBLAS picks per call (previous default)\n"
      << "  --warmup-steps N      linear warmup updates (default 10)\n"
      << "  --save-every N        periodic checkpoint interval (default 0)\n"
      << "  --seed N              deterministic run seed (default 1234)\n"
      << "  --version, -v         print the version and exit\n"
      << "JSONL rows: {\"text\": ...} trains every token; "
         "{\"segments\": [{\"text\": ..., \"train\": bool}, ...]}\n"
      << "computes loss only on tokens from train=true segments\n";
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
    else if (rwkv::is_version_flag(arg))
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
#ifdef RWKV_NORA_TUNE
      std::exit(rwkv::print_version("rwkv_nora_tune"));
#else
      std::exit(rwkv::print_version("rwkv_miss_tune"));
#endif
#else
      std::exit(rwkv::print_version("rwkv_state_tune"));
#endif
    else if (arg == "--wkv_tape")
      continue; // Compatibility flag: shared WKV tape is always enabled.
    else if (arg == "--chunk-load")
      out.chunk_load = true;
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
    else if (arg == "--state")
      out.state = value(i, argc, argv);
    else if (arg == "--resume")
      out.resume = value(i, argc, argv);
    else if (arg == "--targets")
      out.targets = value(i, argc, argv);
    else if (arg == "--rank")
      out.rank = integer<int>(value(i, argc, argv), "rank");
    else if (arg == "--alpha")
      out.alpha = real(value(i, argc, argv), "alpha");
#endif
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
    else if (arg == "--micro-bsz")
      out.micro_bsz = integer<int>(value(i, argc, argv), "micro-bsz");
    else if (arg == "--max-steps")
      out.max_steps = integer<std::uint64_t>(value(i, argc, argv), "max-steps");
    else if (arg == "--optimizer")
      out.optimizer = value(i, argc, argv);
    else if (arg == "--gemm-mode")
      out.gemm_mode = value(i, argc, argv);
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
  if (out.optimizer != "adam" && out.optimizer != "muon")
    usage(argv[0], "optimizer must be adam or muon");
  if (out.gemm_mode != "bf16" && out.gemm_mode != "strict" &&
      out.gemm_mode != "auto")
    usage(argv[0], "gemm-mode must be bf16, strict or auto");
  if (out.model.empty() || out.data.empty() || out.output.empty())
    usage(argv[0], "--model, --data and --output are required");
  if (out.ctx <= 0 || out.chunk <= 0 || out.epochs <= 0 ||
      out.batch_size <= 0 || out.micro_bsz <= 0 || out.lr <= 0 ||
      out.lr_final <= 0)
    usage(argv[0], "ctx/chunk/epochs/batch/lr values must be positive");
  out.micro_bsz = std::min(out.micro_bsz, out.batch_size);
  if (static_cast<long long>(std::min(out.ctx, out.chunk)) * out.micro_bsz >
      65535)
    usage(argv[0], "micro-bsz * chunk must not exceed 65535 rows");
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
  if (out.rank < 1 || out.rank > 1024 || out.optimizer != "adam")
    usage(argv[0], "adapter tuning requires rank 1..1024 and Adam");
  if (out.alpha == -1)
    out.alpha = out.rank;
#ifdef RWKV_NORA_TUNE
  if (out.alpha != out.rank)
    usage(argv[0], "NoRA requires alpha/rank = 1");
#endif
#endif
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
    // Thread-local policy: all training GEMMs run on this thread. Kernels are
    // chosen for wide prefill-like GEMMs (no decode to keep fast) at a fixed
    // row count, so --chunk and --micro-bsz never change which kernel runs.
    rwkv_bf16_training::set_consistent_forward(
        options.gemm_mode != "auto", options.gemm_mode == "bf16", 256);
    const int max_time = std::min(options.ctx, options.chunk);
    const std::size_t dataset_rows =
        rwkv7_state_tuning::count_jsonl_rows(options.data);
    if (!dataset_rows)
      throw std::runtime_error("dataset has no JSONL rows");
    std::uint64_t planned =
        static_cast<std::uint64_t>(options.epochs) *
        ((dataset_rows + options.batch_size - 1) / options.batch_size);
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
    if (options.max_steps)
      planned = std::min(planned, options.max_steps);
#endif

    rwkv7_server::TrieTokenizer tokenizer;
    if (tokenizer.load(options.vocab) != rwkv7_server::kTokenizerSuccess)
      throw std::runtime_error("failed to load tokenizer: " + options.vocab);
    rwkv_bf16_training::Model model(options.model);
    auto frozen = model.state_tuning_model_view();
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
    const std::size_t state_lane = static_cast<std::size_t>(H) * N * N;

    Stream stream;
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
    Json::Value config;
    config["base_fingerprint"] = model.base_fingerprint();
    config["data_fingerprint"] = rwkv7_miss::fingerprint_file(options.data);
    config["vocab_fingerprint"] = rwkv7_miss::fingerprint_file(options.vocab);
    config["state_fingerprint"] =
        options.state.empty() ? "zero"
                              : rwkv7_miss::fingerprint_file(options.state);
    config["rank"] = options.rank;
    config["alpha"] = options.alpha;
    config["targets"] = options.targets;
    config["ctx"] = options.ctx;
    config["chunk"] = options.chunk;
    config["batch_size"] = options.batch_size;
    config["epochs"] = options.epochs;
    config["lr"] = options.lr;
    config["lr_final"] = options.lr_final;
    config["warmup_steps"] = Json::UInt64(options.warmup_steps);
    config["planned_steps"] = Json::UInt64(planned);
    config["seed"] = Json::UInt64(options.seed);
    config["training_dtype"] = "bfloat16";
#ifdef RWKV_NORA_TUNE
    config["method"] = "nora";
    config["initialization"] = "nora_init";
#endif
    // Adapter latent tapes hold one packed micro-batch chunk.
    rwkv_bf16_miss::Trainer miss(frozen, options.rank, options.alpha,
                                 options.targets,
                                 max_time * options.micro_bsz, config,
                                 stream.value);
#endif
    const std::size_t state_count = static_cast<std::size_t>(L) * state_lane;
    DeviceBuffer<float> time_state, state_gradient, adam_m, adam_v;
    time_state.resize(state_count, "tune time_state");
    state_gradient.resize(state_count, "tune dState");
    time_state.zero("zero tune time_state");
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
    adam_m.resize(state_count, "tune Adam m");
    adam_m.zero("zero tune Adam m");
    if (options.optimizer == "adam") {
      adam_v.resize(state_count, "tune Adam v");
      adam_v.zero("zero tune Adam v");
    }
#else
    if (!options.state.empty()) time_state = model.load_state(options.state);
#endif
    rwkv_bf16_training::Executor executor(frozen, max_time, stream.value,
                                          options.micro_bsz);
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
    miss.bind_tapes(executor.tapes());
#endif
    std::cout << "Optimizer: " << options.optimizer << '\n'
              << "GEMM mode: " << options.gemm_mode << '\n'
              << "Micro-bsz: " << options.micro_bsz
              << " packed lanes (no padding)\n"
              << "WKV tape: shared across layers (recompute)\n";
    std::filesystem::create_directories(options.output);

    std::uint64_t global_step = 0;
    std::size_t skipped = 0;
    bool stop = false;
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t processed_tokens = 0;
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
    std::vector<float> host_gradient(state_count), accumulated(state_count);
#endif
    int resume_epoch = 1;
    std::uint64_t resume_row = 0;
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
    if (!options.resume.empty())
      miss.resume(options.resume, global_step, resume_epoch, resume_row,
                  time_state.p, state_count);
    if (options.max_steps && global_step >= options.max_steps)
      stop = true;
#endif
    for (int epoch = resume_epoch; epoch <= options.epochs && !stop; ++epoch) {
      rwkv7_state_tuning::JsonlTextReader dataset(options.data);
      std::uint64_t data_row = 0;
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
      if (epoch == resume_epoch) {
        std::vector<rwkv7_state_tuning::TextSegment> skipped_row;
        while (data_row < resume_row) {
          if (!dataset.next(skipped_row))
            throw std::runtime_error("resume data progress exceeds dataset");
          ++data_row;
        }
      }
#endif
      bool exhausted = false;
      while (!exhausted) {
        std::vector<Sample> batch;
        // batch_tokens counts loss targets and normalises the gradient;
        // batch_positions counts every position run through the model.
        std::size_t batch_tokens = 0;
        std::size_t batch_positions = 0;
        while (batch.size() < static_cast<std::size_t>(options.batch_size)) {
          std::vector<rwkv7_state_tuning::TextSegment> segments;
          if (!dataset.next(segments)) {
            exhausted = true;
            break;
          }
          ++data_row;
          // Segments are tokenized independently so the mask follows their
          // boundaries exactly.
          Sample sample;
          for (const auto &segment : segments) {
            const auto encoded = tokenizer.encode(segment.text);
            sample.tokens.insert(sample.tokens.end(), encoded.begin(),
                                 encoded.end());
            sample.train.insert(sample.train.end(), encoded.size(),
                                segment.train ? 1 : 0);
          }
          std::size_t count = std::min(
              sample.tokens.size(), static_cast<std::size_t>(options.ctx) + 1);
          // Positions after the last trained target carry no gradient.
          while (count >= 2 && !sample.train[count - 1])
            --count;
          if (count < 2) {
            ++skipped;
            continue;
          }
          sample.tokens.resize(count);
          sample.train.resize(count);
          batch_tokens += static_cast<std::size_t>(
              std::count(sample.train.begin() + 1, sample.train.end(), 1));
          batch_positions += count - 1;
          batch.push_back(std::move(sample));
        }
        if (batch.empty())
          break;
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
        std::fill(accumulated.begin(), accumulated.end(), 0.0f);
#endif
        float batch_loss = 0.0f;
        for (std::size_t first = 0; first < batch.size();
             first += options.micro_bsz) {
          const std::size_t last = std::min(
              batch.size(), first + static_cast<std::size_t>(options.micro_bsz));
          if (options.micro_bsz == 1) {
            const auto &sample = batch[first];
            std::vector<float> weights(sample.train.begin(), sample.train.end());
            batch_loss += executor.backward(frozen, sample.tokens, weights,
                                             time_state.p, batch_tokens);
          } else {
            std::vector<std::vector<int>> tokens;
            std::vector<std::vector<float>> weights;
            for (std::size_t i = first; i < last; ++i) {
              tokens.push_back(batch[i].tokens);
              weights.emplace_back(batch[i].train.begin(), batch[i].train.end());
            }
            batch_loss += executor.backward_packed(frozen, tokens, weights,
                                                    time_state.p, batch_tokens);
          }
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
          // Per-sample gradients [L,lanes,H,K,V] are added in sample order,
          // exactly like single-sample steps, so --micro-bsz changes no bits.
          const std::size_t lanes = executor.lanes();
          host_gradient.resize(state_count * lanes);
          check(cudaMemcpyAsync(host_gradient.data(),
                                executor.lane_state_gradients(),
                                host_gradient.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream.value),
                "validate initial state gradient");
          check(cudaStreamSynchronize(stream.value), "read state gradient");
          for (std::size_t b = 0; b < last - first; ++b)
            for (std::size_t i = 0; i < state_count; ++i) {
              const float g =
                  host_gradient[(i / state_lane * lanes + b) * state_lane +
                                i % state_lane];
              if (!std::isfinite(g))
                throw std::runtime_error("nonfinite dState at layer " +
                                         std::to_string(i / state_lane) +
                                         "; optimizer not executed");
              accumulated[i] += g;
              if (!std::isfinite(accumulated[i]) ||
                  std::abs(accumulated[i]) >
                      std::sqrt(std::numeric_limits<float>::max()))
                throw std::runtime_error(
                    "batch gradient overflow before optimizer");
            }
#endif
        }
#if !defined(RWKV_MISS_TUNE) && !defined(RWKV_NORA_TUNE)
        check(cudaMemcpyAsync(state_gradient.p, accumulated.data(),
                              state_count * sizeof(float),
                              cudaMemcpyHostToDevice, stream.value),
              "copy batch gradient");
#endif
        const std::uint64_t next_step = global_step + 1;
        const float lr = learning_rate(options, next_step, planned);
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
        miss.update(lr, next_step);
#else
        if (options.optimizer == "muon") {
          rwkv_bf16_training::MuonConfig muon;
          muon.learning_rate = lr;
          rwkv_bf16_training::muon_update_state_f32(stream.value, time_state.p,
                                                    state_gradient.p, adam_m.p,
                                                    state_count, muon, true);
        } else {
          rwkv_bf16_training::AdamConfig adam;
          adam.learning_rate = lr;
          rwkv_bf16_training::adam_update_state_f32(
              stream.value, time_state.p, state_gradient.p, adam_m.p, adam_v.p,
              state_count, next_step, adam, true);
        }
#endif
        check(cudaStreamSynchronize(stream.value), "optimizer update");
        global_step = next_step;
        processed_tokens += batch_positions;
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
                  << " tokens=" << batch_positions
                  << " trained=" << batch_tokens
                  << " lr=" << std::setprecision(6) << lr
                  << " tok/s=" << std::setprecision(1)
                  << processed_tokens / elapsed
                  << " ETA=" << std::setprecision(0)
                  << elapsed / global_step * (planned - global_step) << "s   "
                  << std::flush;
        if (options.save_every && global_step % options.save_every == 0) {
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
          const auto path = (std::filesystem::path(options.output) /
                             ("checkpoint-" + std::to_string(global_step)))
                                .string();
          miss.checkpoint(path, global_step, epoch, data_row, time_state.p,
                          state_count);
#else
          const auto path = checkpoint_name(options.output, global_step);
          rwkv_bf16_training::save_state_checkpoint_pth(path, stream.value, L,
                                                        H, N, time_state.p);
#endif
          std::cout << "\nsaved: " << path << '\n';
        }
        if (options.max_steps && global_step >= options.max_steps) {
          stop = true;
          break;
        }
      }
    }
    std::cout << '\n';
#if defined(RWKV_MISS_TUNE) || defined(RWKV_NORA_TUNE)
    const std::string final_path =
        (std::filesystem::path(options.output) / "adapter-final.pth").string();
    miss.export_adapter(final_path);
#else
    const std::string final_path =
        (std::filesystem::path(options.output) / "state-final.pth").string();
    rwkv_bf16_training::save_state_checkpoint_pth(final_path, stream.value, L,
                                                  H, N, time_state.p);
#endif
    std::cout << "saved: " << std::filesystem::absolute(final_path).string()
              << '\n';
    std::cout << "complete: steps=" << global_step
              << " skipped_short_samples=" << skipped
              << " seed=" << options.seed << '\n';
  } catch (const std::exception &error) {
    std::cerr <<
#ifdef RWKV_NORA_TUNE
        "\nrwkv_nora_tune: "
#elif defined(RWKV_MISS_TUNE)
        "\nrwkv_miss_tune: "
#else
        "\nrwkv_state_tune: "
#endif
              << error.what() << '\n';
    return 1;
  }
  return 0;
}
