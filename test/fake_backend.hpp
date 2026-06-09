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
    write_step_logits(state.batch_size, logits);
  }

  void forward_decode(
      const std::vector<int64_t>& token_batch,
      rwkv7_server::GenerationState&,
      rwkv7_server::DeviceLogits& logits) const override {
    decode_tokens_.push_back(token_batch);
    ++step_index_;
    write_step_logits(static_cast<int>(token_batch.size()), logits);
  }

  void copy_state_slice(
      const rwkv7_server::GenerationState& src,
      int src_offset,
      rwkv7_server::GenerationState& dst,
      int dst_offset,
      int count) const override {
    if (src_offset < 0 || dst_offset < 0 || count <= 0) {
      throw std::runtime_error("invalid fake state slice range");
    }
    if (src_offset + count > src.batch_size || dst_offset + count > dst.batch_size) {
      throw std::runtime_error("fake state slice range exceeds batch size");
    }
  }

  void copy_logits_slice(
      const rwkv7_server::DeviceLogits& src,
      int src_offset,
      rwkv7_server::DeviceLogits& dst,
      int dst_offset,
      int count) const override {
    if (src_offset < 0 || dst_offset < 0 || count <= 0) {
      throw std::runtime_error("invalid fake logits slice range");
    }
    if (src_offset + count > src.rows || dst_offset + count > dst.rows || src.vocab_size != dst.vocab_size) {
      throw std::runtime_error("fake logits slice range exceeds row count");
    }

    std::vector<float> dst_host(static_cast<std::size_t>(dst.rows) * dst.vocab_size, 0.0f);
    if (dst.values.p != nullptr && dst.values.n == dst_host.size()) {
      dst_host = copy_device_buffer(dst.values, "copy fake dst logits");
    }
    const auto src_host = copy_device_buffer(src.values, "copy fake src logits");
    const std::size_t vocab = static_cast<std::size_t>(src.vocab_size);
    for (int row = 0; row < count; ++row) {
      const auto* src_begin = src_host.data() + (static_cast<std::size_t>(src_offset + row) * vocab);
      auto* dst_begin = dst_host.data() + (static_cast<std::size_t>(dst_offset + row) * vocab);
      std::copy(src_begin, src_begin + vocab, dst_begin);
    }
    copy_host_to_device(dst_host, dst.values, "alloc fake merged logits", "copy fake merged logits");
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
  void write_step_logits(int rows, rwkv7_server::DeviceLogits& logits) const {
    if (step_index_ >= logits_steps_.size()) {
      throw std::runtime_error("fake backend ran out of logits steps");
    }
    const auto& step = logits_steps_[step_index_];
    std::vector<float> repeated(static_cast<std::size_t>(rows) * vocab_size_);
    for (int row = 0; row < rows; ++row) {
      std::copy(
          step.begin(),
          step.end(),
          repeated.begin() + static_cast<std::size_t>(row) * vocab_size_);
    }
    logits.rows = rows;
    logits.vocab_size = vocab_size_;
    copy_host_to_device(repeated, logits.values, "alloc fake logits", "copy fake logits");
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
