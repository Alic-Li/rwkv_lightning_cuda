#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "rwkv/runtime/rwkv_server_backend.hpp"

namespace {

std::vector<int> parse_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const int value = std::stoi(item);
    if (value <= 0) throw std::invalid_argument("list values must be positive");
    values.push_back(value);
  }
  if (values.empty()) throw std::invalid_argument("empty list");
  return values;
}

constexpr int kPromptTokens = 32;
constexpr int kWarmups = 3;
constexpr int kRepeats = 10;

double median_ms(std::array<double, kRepeats> samples) {
  std::sort(samples.begin(), samples.end());
  return 0.5 * (samples[kRepeats / 2 - 1] + samples[kRepeats / 2]);
}

constexpr const char* kUsage =
    "usage: bench_model MODEL [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n"
    "                         [--batches 1,2,4,...] [--prefill-tokens T1,T2,...]\n"
    "  Default: decode step latency per batch size.\n"
    "  --prefill-tokens: also time forward_prefill of B rows x T tokens.\n";

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return 2;
  }
  try {
    std::string cmix_sparse = "no-fc";
    std::string tune_cache;
    bool retune = false;
    std::vector<int> batches{1, 2, 4, 8, 16, 32, 64};
    std::vector<int> prefill_tokens;
    for (int arg = 2; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (option == "--cmix-sparse" && arg + 1 < argc &&
          (std::string(argv[arg + 1]) == "no-fc" || std::string(argv[arg + 1]) == "off")) {
        cmix_sparse = argv[++arg];
      } else if (option == "--tune-cache" && arg + 1 < argc) {
        tune_cache = argv[++arg];
      } else if (option == "--retune") {
        retune = true;
      } else if (option == "--batches" && arg + 1 < argc) {
        batches = parse_list(argv[++arg]);
      } else if (option == "--prefill-tokens" && arg + 1 < argc) {
        prefill_tokens = parse_list(argv[++arg]);
      } else {
        std::cerr << kUsage;
        return 2;
      }
    }
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, cmix_sparse,
                                                               tune_cache, retune);
    std::cout << "bench_config prompt_tokens=" << kPromptTokens << " warmups=" << kWarmups
              << " repeats=" << kRepeats << " seed=1380668983 cmix_sparse=" << cmix_sparse << "\n";
    for (int batch : batches) {
      std::mt19937 rng(1380668983u);
      std::uniform_int_distribution<int> token_distribution(0, model->vocab_size() - 1);
      std::vector<std::vector<std::int64_t>> prompt(
          static_cast<std::size_t>(batch), std::vector<std::int64_t>(kPromptTokens));
      for (auto& row : prompt) {
        for (auto& token : row) token = token_distribution(rng);
      }
      std::vector<std::int64_t> tokens(static_cast<std::size_t>(batch));
      std::unordered_set<std::int64_t> used_tokens;
      for (auto& token : tokens) {
        do {
          token = token_distribution(rng);
        } while (!used_tokens.insert(token).second && used_tokens.size() < 60000);
      }
      auto state = model->create_state(batch);
      rwkv7_server::DeviceLogits logits;
      model->forward_prefill(prompt, state, logits);
      for (int i = 0; i < kWarmups; ++i) model->forward_decode(tokens, state, logits);
      std::array<double, kRepeats> samples{};
      for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        model->forward_decode(tokens, state, logits);
        const auto stop = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::milli>(stop - start).count();
      }
      const double decode_ms = median_ms(samples);
      std::cout << "batch=" << batch << " decode_ms=" << decode_ms
                << " tok_s=" << (1000.0 * batch / decode_ms) << "\n";
      for (int length : prefill_tokens) {
        std::vector<std::vector<std::int64_t>> chunk(
            static_cast<std::size_t>(batch), std::vector<std::int64_t>(static_cast<std::size_t>(length)));
        for (auto& row : chunk) {
          for (auto& token : row) token = token_distribution(rng);
        }
        for (int i = 0; i < kWarmups; ++i) model->forward_prefill(chunk, state, logits);
        for (double& sample : samples) {
          const auto start = std::chrono::steady_clock::now();
          model->forward_prefill(chunk, state, logits);
          const auto stop = std::chrono::steady_clock::now();
          sample = std::chrono::duration<double, std::milli>(stop - start).count();
        }
        const double prefill_ms = median_ms(samples);
        std::cout << "batch=" << batch << " prefill_tokens=" << length << " prefill_ms=" << prefill_ms
                  << " ms_per_row=" << prefill_ms / batch << "\n";
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
