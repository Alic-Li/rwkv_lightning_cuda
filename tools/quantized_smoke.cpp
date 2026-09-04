#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "rwkv_server_backend.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: quantized_smoke MODEL_RWKVQ [--tune-cache PATH] [--retune]\n";
    return 2;
  }
  try {
    std::string tune_cache;
    bool retune = false;
    for (int arg = 2; arg < argc; ++arg) {
      const std::string option = argv[arg];
      if (option == "--tune-cache" && arg + 1 < argc) tune_cache = argv[++arg];
      else if (option == "--retune") retune = true;
      else {
        std::cerr << "usage: quantized_smoke MODEL_RWKVQ [--tune-cache PATH] [--retune]\n";
        return 2;
      }
    }
    auto model = std::make_shared<rwkv7_server::ModelBackend>(argv[1], false, true, "no-fc",
                                                               tune_cache, retune);
    auto state = model->create_state(1);
    rwkv7_server::DeviceLogits logits;
    model->forward_prefill({{0}}, state, logits);
    model->forward_decode({0}, state, logits);
    std::vector<float> host(static_cast<std::size_t>(logits.rows) * logits.vocab_size);
    cudaMemcpy(host.data(), logits.values.p, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::size_t best = 0;
    for (std::size_t i = 1; i < host.size(); ++i) if (host[i] > host[best]) best = i;
    std::cout << "quantized smoke ok rows=" << logits.rows << " vocab=" << logits.vocab_size
              << " argmax=" << best << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "quantized smoke failed: " << error.what() << "\n";
    return 1;
  }
}
