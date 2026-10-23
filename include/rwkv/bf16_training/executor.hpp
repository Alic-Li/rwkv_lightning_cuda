#pragma once
#include "rwkv/bf16_training/training.hpp"
#include <memory>

namespace rwkv_bf16_training {
// Shared CLI/server execution workspace. Only recurrent boundary checkpoints
// survive a forward sweep; backward replays each chunk and propagates BOTH WKV
// and attention/FFN shift adjoints, including through masked prompt tokens.
class Executor {
public:
  Executor(const FrozenModelView &, int chunk, cudaStream_t);
  ~Executor();
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;
  std::vector<BlockTapeView> &tapes();
  DeviceBuffer<float> &state();
  DeviceBuffer<bf16> &shifts();
  const float *state_gradient() const;
  void reset(const float *initial);
  void forward(const FrozenModelView &, const int *tokens, int count);
  std::vector<float> last_logits() const;
  // weights[i] applies to tokens[i], predicted from tokens[i-1]. weights[0]
  // is unused. Normalizer is the batch count of selected targets, never the
  // signed sum of advantages. The initial-state gradient remains available.
  float backward(const FrozenModelView &, const std::vector<int> &tokens,
                 const std::vector<float> &weights, const float *initial,
                 size_t normalizer, float temperature = 1.0f);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv_bf16_training
