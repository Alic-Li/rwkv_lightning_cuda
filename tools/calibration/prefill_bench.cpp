#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rwkv_server_backend.hpp"

namespace {

constexpr int kPromptTokens = 1024;
constexpr int kChunkTokens = 128;
constexpr int kWarmups = 1;
constexpr int kRepeats = 3;
constexpr unsigned kSeed = 1380668983u;

double run_prefill(const rwkv7_server::ModelBackend& model, const std::vector<std::int64_t>& tokens) {
  auto state = model.create_state(1);
  rwkv7_server::DeviceLogits logits;
  const auto begin = std::chrono::steady_clock::now();
  for (int offset = 0; offset < kPromptTokens; offset += kChunkTokens) {
    std::vector<std::vector<std::int64_t>> chunk{
        std::vector<std::int64_t>(tokens.begin() + offset, tokens.begin() + offset + kChunkTokens)};
    model.forward_prefill(chunk, state, logits);
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: rwkv_prefill_bench MODEL [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n";
    return 2;
  }
  try {
    std::string cmix_sparse = "no-fc";
    std::string tune_cache;
    bool retune = false;
    for (int arg = 2; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (option == "--cmix-sparse" && arg + 1 < argc &&
          (std::string(argv[arg + 1]) == "no-fc" || std::string(argv[arg + 1]) == "off")) {
        cmix_sparse = argv[++arg];
      } else if (option == "--tune-cache" && arg + 1 < argc) {
        tune_cache = argv[++arg];
      } else if (option == "--retune") {
        retune = true;
      } else {
        std::cerr << "usage: rwkv_prefill_bench MODEL [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n";
        return 2;
      }
    }
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, cmix_sparse,
                                                               tune_cache, retune);
    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int> token_distribution(0, model->vocab_size() - 1);
    std::vector<std::int64_t> tokens(kPromptTokens);
    for (auto& token : tokens) token = token_distribution(rng);
    for (int i = 0; i < kWarmups; ++i) run_prefill(*model, tokens);
    std::array<double, kRepeats> samples{};
    for (double& sample : samples) sample = run_prefill(*model, tokens);
    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[kRepeats / 2];
    std::cout << "prefill_config prompt_tokens=" << kPromptTokens
              << " chunk_tokens=" << kChunkTokens
              << " warmups=" << kWarmups
              << " repeats=" << kRepeats
              << " seed=" << kSeed
              << " cmix_sparse=" << cmix_sparse << "\n";
    std::cout << "prefill_ms=" << median_ms
              << " first_token_ms=" << median_ms
              << " prefill_tok_s=" << (1000.0 * kPromptTokens / median_ms) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "prefill benchmark failed: " << error.what() << "\n";
    return 1;
  }
}
