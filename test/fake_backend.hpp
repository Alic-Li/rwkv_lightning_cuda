#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rwkv_server_backend.hpp"
#include "test_common.hpp"

namespace rwkv_test {

class FakeModelBackend final : public rwkv7_server::IModelBackend {
 public:
  explicit FakeModelBackend(std::vector<std::vector<float>> logits_steps, std::string name = "fake-model")
      : logits_steps_(std::move(logits_steps)),
        vocab_size_(logits_steps_.empty() ? 0 : static_cast<int>(logits_steps_.front().size())),
        model_name_(std::move(name)) {
    for (const auto& step : logits_steps_) {
      if (static_cast<int>(step.size()) != vocab_size_) {
        throw std::runtime_error("inconsistent fake backend vocab size");
      }
    }
  }

  rwkv7_server::GenerationState create_state(int batch_size) const override {
    rwkv7_server::GenerationState state;
    state.batch_size = batch_size;
    return state;
  }

  void forward_prefill(
      const std::vector<std::vector<int64_t>>& token_batches,
      rwkv7_server::GenerationState& state,
      rwkv7_server::DeviceLogits& logits) const override {
    prefill_batches_.push_back(token_batches);
    state.batch_size = static_cast<int>(token_batches.size());
    step_index_ = 0;
    write_step_logits(logits);
  }

  void forward_decode(
      const std::vector<int64_t>& token_batch,
      rwkv7_server::GenerationState&,
      rwkv7_server::DeviceLogits& logits) const override {
    decode_tokens_.push_back(token_batch);
    ++step_index_;
    write_step_logits(logits);
  }

  int vocab_size() const override {
    return vocab_size_;
  }

  const std::string& model_path() const override {
    return model_path_;
  }

  const std::string& model_name() const override {
    return model_name_;
  }

  const std::vector<std::vector<std::vector<int64_t>>>& prefill_batches() const {
    return prefill_batches_;
  }

  const std::vector<std::vector<int64_t>>& decode_tokens() const {
    return decode_tokens_;
  }

 private:
  void write_step_logits(rwkv7_server::DeviceLogits& logits) const {
    if (step_index_ >= logits_steps_.size()) {
      throw std::runtime_error("fake backend ran out of logits steps");
    }
    const auto& step = logits_steps_[step_index_];
    logits.rows = 1;
    logits.vocab_size = vocab_size_;
    copy_host_to_device(step, logits.values, "alloc fake logits", "copy fake logits");
  }

  std::vector<std::vector<float>> logits_steps_;
  int vocab_size_ = 0;
  std::string model_path_ = "fake://backend";
  std::string model_name_;
  mutable std::size_t step_index_ = 0;
  mutable std::vector<std::vector<std::vector<int64_t>>> prefill_batches_;
  mutable std::vector<std::vector<int64_t>> decode_tokens_;
};

inline std::vector<float> one_hot_logits(int vocab_size, int token_id, float hot = 100.0f, float cold = -100.0f) {
  std::vector<float> logits(static_cast<std::size_t>(vocab_size), cold);
  if (token_id >= 0 && token_id < vocab_size) {
    logits[static_cast<std::size_t>(token_id)] = hot;
  }
  return logits;
}

}  // namespace rwkv_test
