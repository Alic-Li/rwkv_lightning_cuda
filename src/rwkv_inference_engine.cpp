#include "rwkv_inference_engine.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace rwkv7_server {
namespace {

bool is_utf8_continuation(unsigned char byte) {
  return (byte & 0xC0u) == 0x80u;
}

int utf8_sequence_length(unsigned char lead) {
  if (lead <= 0x7Fu) {
    return 1;
  }
  if (lead >= 0xC2u && lead <= 0xDFu) {
    return 2;
  }
  if (lead >= 0xE0u && lead <= 0xEFu) {
    return 3;
  }
  if (lead >= 0xF0u && lead <= 0xF4u) {
    return 4;
  }
  return 0;
}

bool is_valid_utf8_sequence(const std::string& text, size_t pos, int len) {
  const auto b0 = static_cast<unsigned char>(text[pos]);
  switch (len) {
    case 1:
      return true;
    case 2:
      return is_utf8_continuation(static_cast<unsigned char>(text[pos + 1]));
    case 3: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2)) {
        return false;
      }
      if (b0 == 0xE0u && b1 < 0xA0u) {
        return false;
      }
      if (b0 == 0xEDu && b1 >= 0xA0u) {
        return false;
      }
      return true;
    }
    case 4: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      const auto b3 = static_cast<unsigned char>(text[pos + 3]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2) || !is_utf8_continuation(b3)) {
        return false;
      }
      if (b0 == 0xF0u && b1 < 0x90u) {
        return false;
      }
      if (b0 == 0xF4u && b1 > 0x8Fu) {
        return false;
      }
      return true;
    }
    default:
      return false;
  }
}

std::string take_complete_utf8(std::string& pending, bool flush_all) {
  std::string out;
  size_t i = 0;
  size_t last_copy = 0;
  while (i < pending.size()) {
    const auto lead = static_cast<unsigned char>(pending[i]);
    const int len = utf8_sequence_length(lead);
    if (len == 0) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    if (i + static_cast<size_t>(len) > pending.size()) {
      break;
    }
    if (!is_valid_utf8_sequence(pending, i, len)) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    i += static_cast<size_t>(len);
  }
  out.append(pending, last_copy, i - last_copy);
  pending.erase(0, i);
  if (flush_all && !pending.empty()) {
    pending.clear();
  }
  return out;
}

}  // namespace

InferenceEngine::InferenceEngine(
    std::shared_ptr<ModelBackend> model,
    std::shared_ptr<TrieTokenizer> tokenizer,
    std::string model_name)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      model_name_(std::move(model_name)) {}

std::vector<int64_t> InferenceEngine::encode_prompt(const std::string& prompt, bool pad_zero) const {
  auto ids = tokenizer_->encode(prompt);
  std::vector<int64_t> out(ids.begin(), ids.end());
  if (pad_zero) {
    out.insert(out.begin(), 0);
  }
  if (out.empty()) {
    out.push_back(0);
  }
  return out;
}

std::string InferenceEngine::generate_one(
    const std::string& prompt,
    GenerationState& state,
    const GenerateOptions& options,
    const StreamCallback* emit,
    int stream_index,
    int chunk_size) const {
  const auto prompt_ids = encode_prompt(prompt, options.pad_zero);
  std::vector<std::vector<int64_t>> prefill_batch{prompt_ids};
  auto logits = model_->forward_prefill(prefill_batch, state);

  std::vector<int> token_buffer;
  std::string utf8_pending;
  std::string result;

  auto penalties = make_sampler_penalties(model_->vocab_size());
  for (int step = 0; step < options.max_tokens; ++step) {
    const int token = sample_repetition_temperature_topk_topp(logits, penalties, options);
    if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
      break;
    }

    update_sampler_penalties(token, penalties, options);
    token_buffer.push_back(token);
    result += tokenizer_->decode(token);

    if (emit != nullptr && chunk_size > 0 && token_buffer.size() >= static_cast<size_t>(chunk_size)) {
      utf8_pending += tokenizer_->decode(token_buffer);
      token_buffer.clear();
      const std::string chunk = take_complete_utf8(utf8_pending, false);
      if (!chunk.empty() && !(*emit)(stream_index, chunk)) {
        return result;
      }
    }

    logits = model_->forward_decode(std::vector<int64_t>{token}, state);
  }

  if (emit != nullptr) {
    if (!token_buffer.empty()) {
      utf8_pending += tokenizer_->decode(token_buffer);
      token_buffer.clear();
    }
    const std::string tail = take_complete_utf8(utf8_pending, true);
    if (!tail.empty()) {
      (*emit)(stream_index, tail);
    }
  }
  return result;
}

std::vector<std::string> InferenceEngine::batch_generate(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options) const {
  std::vector<std::string> outputs;
  outputs.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    auto state = model_->create_state(1);
    outputs.push_back(generate_one(prompt, state, options, nullptr, 0, 0));
  }
  return outputs;
}

std::vector<std::string> InferenceEngine::batch_generate_state(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options) const {
  if (prompts.size() != 1 || state.batch_size != 1) {
    throw std::runtime_error("stateful generation only supports single prompt batch");
  }
  return {generate_one(prompts.front(), state, options, nullptr, 0, 0)};
}

void InferenceEngine::batch_generate_stream(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  for (size_t i = 0; i < prompts.size(); ++i) {
    auto state = model_->create_state(1);
    generate_one(prompts[i], state, options, &emit, static_cast<int>(i), chunk_size);
  }
}

void InferenceEngine::batch_generate_state_stream(
    const std::vector<std::string>& prompts,
    GenerationState& state,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  if (prompts.size() != 1 || state.batch_size != 1) {
    throw std::runtime_error("stateful generation only supports single prompt batch");
  }
  generate_one(prompts.front(), state, options, &emit, 0, chunk_size);
}

std::string InferenceEngine::format_openai_prompt(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool enable_think) const {
  std::ostringstream oss;
  if (!system.empty()) {
    oss << "System: " << system << "\n\n";
  }
  for (const auto& [role, content] : messages) {
    oss << role << ": " << content << "\n\n";
  }
  oss << "Assistant:";
  if (enable_think) {
    oss << " <think>\n</think>\n";
  } else {
    oss << ' ';
  }
  return oss.str();
}

int InferenceEngine::count_tokens(const std::string& text) const {
  return static_cast<int>(tokenizer_->encode(text).size());
}

}  // namespace rwkv7_server
