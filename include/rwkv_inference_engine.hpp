#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rwkv_sampler.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv_tokenizer.hpp"

namespace rwkv7_server {

class InferenceEngine {
 public:
  using StreamCallback = std::function<bool(int, const std::string&)>;

  InferenceEngine(
      std::shared_ptr<IModelBackend> model,
      std::shared_ptr<TrieTokenizer> tokenizer,
      std::string model_name);

  std::vector<std::string> batch_generate(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options) const;

  std::vector<std::string> batch_generate_state(
      const std::vector<std::string>& prompts,
      GenerationState& state,
      const GenerateOptions& options) const;

  void batch_generate_stream(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  void batch_generate_state_stream(
      const std::vector<std::string>& prompts,
      GenerationState& state,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  std::string format_openai_prompt(
      const std::string& system,
      const std::vector<std::pair<std::string, std::string>>& messages,
      bool enable_think) const;

  int count_tokens(const std::string& text) const;

 std::shared_ptr<IModelBackend> model() const { return model_; }
 std::shared_ptr<TrieTokenizer> tokenizer() const { return tokenizer_; }
 const std::string& model_name() const { return model_name_; }

 private:
  std::vector<int64_t> encode_prompt(const std::string& prompt) const;
  std::string generate_one(
      const std::string& prompt,
      GenerationState& state,
      const GenerateOptions& options,
      const StreamCallback* emit,
      int stream_index,
      int chunk_size) const;

  std::shared_ptr<IModelBackend> model_;
  std::shared_ptr<TrieTokenizer> tokenizer_;
  std::string model_name_;
};

}  // namespace rwkv7_server
