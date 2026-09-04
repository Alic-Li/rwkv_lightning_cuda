#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "rwkv_server_backend.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: rwkv_profile_model MODEL BATCH [--cmix-sparse no-fc|off] [--tune-cache PATH] [--retune]\n";
    return 2;
  }
  try {
    const int batch = std::stoi(argv[2]);
    if (batch <= 0) throw std::runtime_error("batch must be positive");
    std::string cmix_sparse = "no-fc";
    std::string tune_cache;
    bool retune = false;
    for (int arg = 3; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (option == "--cmix-sparse" && arg + 1 < argc &&
          (std::string(argv[arg + 1]) == "no-fc" || std::string(argv[arg + 1]) == "off")) {
        cmix_sparse = argv[++arg];
      } else if (option == "--tune-cache" && arg + 1 < argc) {
        tune_cache = argv[++arg];
      } else if (option == "--retune") {
        retune = true;
      } else {
        throw std::runtime_error("invalid tuning or cmix option");
      }
    }
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, cmix_sparse,
                                                               tune_cache, retune);
    std::mt19937 rng(1380668983u);
    std::uniform_int_distribution<int> token_distribution(0, model->vocab_size() - 1);
    std::vector<std::vector<std::int64_t>> prompt(
        static_cast<std::size_t>(batch), std::vector<std::int64_t>(32));
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
    model->forward_decode(tokens, state, logits);
    setenv("RWKV_PROFILE", "1", 1);
    std::cout << "profile_config batch=" << batch << " cmix_sparse=" << cmix_sparse << "\n";
    cudaProfilerStart();
    model->forward_decode(tokens, state, logits);
    cudaProfilerStop();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "profile failed: " << error.what() << "\n";
    return 1;
  }
}
