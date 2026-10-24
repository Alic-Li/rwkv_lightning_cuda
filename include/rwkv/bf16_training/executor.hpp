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
  // Teacher-forced log softmax(logits / temperature) of tokens[i + 1] given
  // tokens[0..i], for every i: tokens.size() - 1 values. Starts from
  // `initial` with zero shifts and uses the same forward as backward().
  std::vector<float> score(const FrozenModelView &,
                           const std::vector<int> &tokens, const float *initial,
                           float temperature = 1.0f);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Decodes `rows` independent sequences one token per row per step. States are
// carried as [L,rows,H,K,V] and shifts as [L,2,rows,C]; nothing is recorded
// for backward. With set_consistent_forward(true) every GEMM row is computed
// in the same fixed 32-row tile shape as Executor, so each row's logits match
// a batch=1 decode of that sequence bit for bit.
class BatchDecoder {
public:
  BatchDecoder(const FrozenModelView &, int rows, cudaStream_t);
  ~BatchDecoder();
  BatchDecoder(const BatchDecoder &) = delete;
  BatchDecoder &operator=(const BatchDecoder &) = delete;
  int rows() const;
  // Copies one sequence's [L,H,K,V] state and [L,2,C] shifts (device memory,
  // e.g. Executor::state()/shifts()) into `row`.
  void load(int row, const float *state, const bf16 *shifts);
  // Returns one row's state and shifts in the single-sequence layouts above.
  void download(int row, std::vector<float> &state,
                std::vector<bf16> &shifts) const;
  // Advances every row by tokens[row]. Callers feed any valid token to rows
  // they no longer read.
  void forward(const FrozenModelView &, const int *tokens);
  // rows * vocab logits of the last forward, row-major.
  std::vector<float> logits() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv_bf16_training
