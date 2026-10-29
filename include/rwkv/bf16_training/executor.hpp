#pragma once
#include "rwkv/bf16_training/training.hpp"
#include <memory>

namespace rwkv_bf16_training {
enum class LogitsMode { All, Last, None };
// Shared CLI/server execution workspace. Only recurrent boundary checkpoints
// survive a forward sweep; backward replays each chunk and propagates BOTH WKV
// and attention/FFN shift adjoints, including through masked prompt tokens.
//
// lanes > 1 reserves room for backward_packed() only: chunk * lanes rows of
// activations/logits and per-lane states. The single-sequence calls below
// require lanes == 1.
class Executor {
public:
  Executor(const FrozenModelView &, int chunk, cudaStream_t, int lanes = 1);
  ~Executor();
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;
  std::vector<BlockTapeView> &tapes();
  DeviceBuffer<float> &state();
  DeviceBuffer<bf16> &shifts();
  // [L,H,K,V] initial-state gradient of the last backward/backward_packed,
  // summed over lanes.
  const float *state_gradient() const;
  // After backward_packed: device [L,lanes,H,K,V] per-sample initial-state
  // gradients (unused lanes zero). Adding lanes in sample order reproduces
  // per-sample backward() accumulation bit for bit.
  const float *lane_state_gradients() const;
  int lanes() const;
  void reset(const float *initial);
  // Last computes only the final token's output head; None advances states
  // without an output head. last_logits() is unavailable after None.
  void forward(const FrozenModelView &, const int *tokens, int count,
               LogitsMode = LogitsMode::All);
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
  // Padding-free micro-batch: sample i runs in lane i (at most lanes()
  // samples). Each chunk step packs every lane's next <= chunk tokens back to
  // back; finished lanes contribute no rows. Equals the sum of backward() over
  // the samples with the same initial state and normalizer, up to summation
  // order. Returns the summed loss.
  float backward_packed(const FrozenModelView &,
                        const std::vector<std::vector<int>> &tokens,
                        const std::vector<std::vector<float>> &weights,
                        const float *initial, size_t normalizer,
                        float temperature = 1.0f);
  std::vector<float> score(const FrozenModelView &,
                           const std::vector<int> &tokens, const float *initial,
                           float temperature = 1.0f);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Decodes `rows` independent sequences one token per row per step. States are
// carried as [L,rows,H,K,V] and shifts as [L,2,rows,C]; nothing is recorded
// for backward. With set_consistent_forward(true, ...) every GEMM row is
// computed independently of the row count, as in Executor, so each row's
// logits match a batch=1 decode of that sequence bit for bit.
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
