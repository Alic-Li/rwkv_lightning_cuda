#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rwkv_server_backend.hpp"
#include "rwkv_tokenizer.hpp"

namespace {

const std::vector<std::string> kPrompts{
    "请用一段简短中文说明矩阵乘法的作用。",
    "User: Explain why the sky appears blue.\nAssistant:",
    "```cpp\nint main() { return 0; }\n```\n请解释这段代码。",
};

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::int64_t argmax(const rwkv7_server::DeviceLogits& logits) {
  std::vector<float> values(static_cast<std::size_t>(logits.rows) * logits.vocab_size);
  check_cuda(cudaMemcpy(values.data(), logits.values.p, values.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "copy logits");
  std::size_t best = 0;
  for (std::size_t index = 1; index < values.size(); ++index) {
    if (values[index] > values[best]) best = index;
  }
  return static_cast<std::int64_t>(best);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: rwkv_cmix_stats MODEL Vocab [PROMPT_INDEX] [--tune-cache PATH] [--retune]\n";
    return 2;
  }
  try {
    int prompt_index = -1;
    int arg = 3;
    if (arg < argc && argv[arg][0] != '-') prompt_index = std::stoi(argv[arg++]);
    if (prompt_index < -1 || prompt_index >= static_cast<int>(kPrompts.size())) {
      throw std::runtime_error("prompt index must be -1, 0, 1, or 2");
    }
    std::string tune_cache;
    bool retune = false;
    for (; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (option == "--tune-cache" && arg + 1 < argc) tune_cache = argv[++arg];
      else if (option == "--retune") retune = true;
      else throw std::runtime_error("invalid tuning option");
    }
    rwkv7_server::TrieTokenizer tokenizer;
    if (tokenizer.load(argv[2]) != rwkv7_server::kTokenizerSuccess) {
      throw std::runtime_error("failed to load vocabulary");
    }
    setenv("RWKV_CMIX_STATS", "1", 1);
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, "no-fc",
                                                               tune_cache, retune);
    std::cout << "cmix_stats_config prompt_index=" << prompt_index << " decode_steps=64\n";
    for (int index = 0; index < static_cast<int>(kPrompts.size()); ++index) {
      if (prompt_index >= 0 && prompt_index != index) continue;
      const auto ids = tokenizer.encode(kPrompts[static_cast<std::size_t>(index)]);
      if (ids.empty()) throw std::runtime_error("prompt tokenization produced no tokens");
      auto state = model->create_state(1);
      rwkv7_server::DeviceLogits logits;
      std::vector<std::vector<std::int64_t>> prompt{std::vector<std::int64_t>(ids.begin(), ids.end())};
      model->forward_prefill(prompt, state, logits);
      const std::int64_t prompt_last = argmax(logits);
      std::int64_t token = prompt_last;
      for (int step = 0; step < 64; ++step) {
        model->forward_decode({token}, state, logits);
        token = argmax(logits);
      }
      std::cout << "cmix_stats_prompt index=" << index << " prompt_tokens=" << ids.size()
                << " first_decode_token=" << prompt_last << "\n";
    }
    model.reset();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cmix stats failed: " << error.what() << "\n";
    return 1;
  }
}
