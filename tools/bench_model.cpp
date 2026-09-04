#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "rwkv_server_backend.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bench_model MODEL [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n";
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
        std::cerr << "usage: bench_model MODEL [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n";
        return 2;
      }
    }
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, cmix_sparse,
                                                               tune_cache, retune);
    constexpr int prompt_tokens = 32;
    constexpr int warmups = 3;
    constexpr int repeats = 10;
    std::cout << "bench_config prompt_tokens=" << prompt_tokens << " warmups=" << warmups
              << " repeats=" << repeats << " seed=1380668983 cmix_sparse=" << cmix_sparse << "\n";
    for (int batch : {1, 2, 4, 8, 16, 32, 64}) {
      std::mt19937 rng(1380668983u);
      std::uniform_int_distribution<int> token_distribution(0, model->vocab_size() - 1);
      std::vector<std::vector<std::int64_t>> prompt(
          static_cast<std::size_t>(batch), std::vector<std::int64_t>(prompt_tokens));
      for (auto& row : prompt) {
        for (auto& token : row) token = token_distribution(rng);
      }
      std::vector<std::int64_t> tokens(static_cast<std::size_t>(batch));
      std::unordered_set<std::int64_t> used_tokens;
      for (auto& token : tokens) {
        do {
          token = token_distribution(rng);
        } while (!used_tokens.insert(token).second);
      }
      auto state = model->create_state(batch);
      rwkv7_server::DeviceLogits logits;
      model->forward_prefill(prompt, state, logits);
      for (int i = 0; i < warmups; ++i) model->forward_decode(tokens, state, logits);
      std::array<double, repeats> samples{};
      for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        model->forward_decode(tokens, state, logits);
        const auto stop = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::milli>(stop - start).count();
      }
      std::sort(samples.begin(), samples.end());
      const double median_ms = 0.5 * (samples[repeats / 2 - 1] + samples[repeats / 2]);
      std::cout << "batch=" << batch << " decode_ms=" << median_ms
                << " tok_s=" << (1000.0 * batch / median_ms) << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
