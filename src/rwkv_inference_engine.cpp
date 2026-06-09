#include "rwkv_inference_engine.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rwkv7_server {
namespace {

constexpr int kPrefillChunkSize = 256;

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
    std::shared_ptr<IModelBackend> model,
    std::shared_ptr<TrieTokenizer> tokenizer,
    std::string model_name)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      model_name_(std::move(model_name)) {}

std::vector<int64_t> InferenceEngine::encode_prompt(const std::string& prompt) const {
  auto ids = tokenizer_->encode(prompt);
  std::vector<int64_t> out(ids.begin(), ids.end());
  if (out.empty()) {
    throw std::runtime_error("prompt is empty after tokenization");
  }
  return out;
}

std::vector<std::vector<int64_t>> InferenceEngine::encode_prompts_sorted(
    const std::vector<std::string>& prompts,
    std::vector<size_t>& sorted_to_original) const {
  std::vector<std::vector<int64_t>> encoded_prompts;
  encoded_prompts.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded_prompts.push_back(encode_prompt(prompt));
  }

  sorted_to_original.resize(prompts.size());
  std::iota(sorted_to_original.begin(), sorted_to_original.end(), 0);
  std::stable_sort(
      sorted_to_original.begin(),
      sorted_to_original.end(),
      [&](size_t lhs, size_t rhs) { return encoded_prompts[lhs].size() > encoded_prompts[rhs].size(); });

  std::vector<std::vector<int64_t>> sorted_prompt_ids;
  sorted_prompt_ids.reserve(prompts.size());
  for (size_t prompt_index : sorted_to_original) {
    sorted_prompt_ids.push_back(std::move(encoded_prompts[prompt_index]));
  }
  return sorted_prompt_ids;
}

void InferenceEngine::prefill_batch_chunked(
    const std::vector<std::vector<int64_t>>& sorted_prompt_ids,
    GenerationState& state,
    DeviceLogits& logits) const {
  const int batch_size = static_cast<int>(sorted_prompt_ids.size());
  if (batch_size <= 0) {
    throw std::runtime_error("prefill batch must not be empty");
  }
  if (state.batch_size != batch_size) {
    throw std::runtime_error("prefill state batch size mismatch");
  }

  std::vector<int> lengths;
  lengths.reserve(sorted_prompt_ids.size());
  std::vector<int> pos(sorted_prompt_ids.size(), 0);
  for (const auto& prompt_ids : sorted_prompt_ids) {
    lengths.push_back(static_cast<int>(prompt_ids.size()));
  }

  bool initialized_logits = false;
  while (true) {
    int active_count = 0;
    while (active_count < batch_size && pos[static_cast<size_t>(active_count)] < lengths[static_cast<size_t>(active_count)]) {
      ++active_count;
    }
    if (active_count == 0) {
      break;
    }

    int step = lengths[0] - pos[0];
    for (int i = 1; i < active_count; ++i) {
      step = std::min(step, lengths[static_cast<size_t>(i)] - pos[static_cast<size_t>(i)]);
    }
    step = std::min(step, kPrefillChunkSize);

    std::vector<std::vector<int64_t>> chunk_tokens;
    chunk_tokens.reserve(static_cast<size_t>(active_count));
    for (int i = 0; i < active_count; ++i) {
      const auto& prompt_ids = sorted_prompt_ids[static_cast<size_t>(i)];
      const int begin = pos[static_cast<size_t>(i)];
      chunk_tokens.emplace_back(
          prompt_ids.begin() + begin,
          prompt_ids.begin() + begin + step);
    }

    if (active_count == batch_size) {
      model_->forward_prefill(chunk_tokens, state, logits);
      initialized_logits = true;
    } else {
      auto sub_state = model_->create_state(active_count);
      model_->copy_state_slice(state, 0, sub_state, 0, active_count);
      DeviceLogits sub_logits;
      model_->forward_prefill(chunk_tokens, sub_state, sub_logits);
      model_->copy_state_slice(sub_state, 0, state, 0, active_count);
      if (!initialized_logits) {
        throw std::runtime_error("batched prefill logits were not initialized");
      }
      model_->copy_logits_slice(sub_logits, 0, logits, 0, active_count);
    }

    for (int i = 0; i < active_count; ++i) {
      pos[static_cast<size_t>(i)] += step;
    }
  }
}

std::string InferenceEngine::generate_one(
    const std::string& prompt,
    GenerationState& state,
    const GenerateOptions& options,
    const StreamCallback* emit,
    int stream_index,
    int chunk_size) const {
  const auto prompt_ids = encode_prompt(prompt);
  std::vector<std::vector<int64_t>> prefill_batch{prompt_ids};
  DeviceLogits logits;
  model_->forward_prefill(prefill_batch, state, logits);

  std::vector<int> token_buffer;
  std::string utf8_pending;
  std::string result;

  auto penalties = make_sampler_penalties(model_->vocab_size());
  for (int step = 0; step < options.max_tokens; ++step) {
    const int token = sample_repetition_temperature_topk_topp(logits, penalties, options);
    if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
      break;
    }

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

    model_->forward_decode(std::vector<int64_t>{token}, state, logits);
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
  if (prompts.empty()) {
    return {};
  }

  std::vector<size_t> sorted_to_original;
  const auto sorted_prompt_ids = encode_prompts_sorted(prompts, sorted_to_original);

  const int batch_size = static_cast<int>(prompts.size());
  auto state = model_->create_state(batch_size);
  DeviceLogits logits;
  prefill_batch_chunked(sorted_prompt_ids, state, logits);

  const int fallback_token = options.stop_tokens.empty() ? 0 : static_cast<int>(options.stop_tokens.front());
  auto penalties = make_sampler_penalties(model_->vocab_size(), batch_size);
  std::vector<std::string> sorted_outputs(prompts.size());
  std::vector<bool> finished(prompts.size(), false);
  std::vector<int64_t> decode_batch(prompts.size(), fallback_token);
  int active = batch_size;

  for (int step = 0; step < options.max_tokens && active > 0; ++step) {
    const auto sampled_tokens = sample_batch_repetition_temperature_topk_topp(logits, penalties, options);
    for (size_t i = 0; i < sampled_tokens.size(); ++i) {
      if (finished[i]) {
        decode_batch[i] = fallback_token;
        continue;
      }

      const int token = sampled_tokens[i];
      if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
        finished[i] = true;
        decode_batch[i] = fallback_token;
        --active;
        continue;
      }

      decode_batch[i] = token;
      sorted_outputs[i] += tokenizer_->decode(token);
    }

    if (active == 0) {
      break;
    }
    model_->forward_decode(decode_batch, state, logits);
  }

  std::vector<std::string> outputs(prompts.size());
  for (size_t i = 0; i < sorted_outputs.size(); ++i) {
    outputs[sorted_to_original[i]] = std::move(sorted_outputs[i]);
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
  if (prompts.empty()) {
    return;
  }

  std::vector<size_t> sorted_to_original;
  const auto sorted_prompt_ids = encode_prompts_sorted(prompts, sorted_to_original);

  const int batch_size = static_cast<int>(prompts.size());
  auto state = model_->create_state(batch_size);
  DeviceLogits logits;
  prefill_batch_chunked(sorted_prompt_ids, state, logits);

  const int fallback_token = options.stop_tokens.empty() ? 0 : static_cast<int>(options.stop_tokens.front());
  auto penalties = make_sampler_penalties(model_->vocab_size(), batch_size);
  std::vector<bool> finished(prompts.size(), false);
  std::vector<int64_t> decode_batch(prompts.size(), fallback_token);
  std::vector<std::vector<int>> token_buffers(prompts.size());
  std::vector<std::string> utf8_pending(prompts.size());
  int active = batch_size;

  for (int step = 0; step < options.max_tokens && active > 0; ++step) {
    const auto sampled_tokens = sample_batch_repetition_temperature_topk_topp(logits, penalties, options);
    for (size_t i = 0; i < sampled_tokens.size(); ++i) {
      if (finished[i]) {
        decode_batch[i] = fallback_token;
        continue;
      }

      const int token = sampled_tokens[i];
      if (std::find(options.stop_tokens.begin(), options.stop_tokens.end(), token) != options.stop_tokens.end()) {
        finished[i] = true;
        decode_batch[i] = fallback_token;
        --active;
        if (!token_buffers[i].empty()) {
          utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
          token_buffers[i].clear();
        }
        const std::string tail = take_complete_utf8(utf8_pending[i], true);
        if (!tail.empty() && !emit(static_cast<int>(sorted_to_original[i]), tail)) {
          return;
        }
        continue;
      }

      decode_batch[i] = token;
      token_buffers[i].push_back(token);

      if (chunk_size > 0 && token_buffers[i].size() >= static_cast<size_t>(chunk_size)) {
        utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
        token_buffers[i].clear();
        const std::string chunk = take_complete_utf8(utf8_pending[i], false);
        if (!chunk.empty() && !emit(static_cast<int>(sorted_to_original[i]), chunk)) {
          return;
        }
      }
    }

    if (active == 0) {
      break;
    }
    model_->forward_decode(decode_batch, state, logits);
  }

  for (size_t i = 0; i < prompts.size(); ++i) {
    if (!token_buffers[i].empty()) {
      utf8_pending[i] += tokenizer_->decode(token_buffers[i]);
      token_buffers[i].clear();
    }
    const std::string tail = take_complete_utf8(utf8_pending[i], true);
    if (!tail.empty() && !emit(static_cast<int>(sorted_to_original[i]), tail)) {
      return;
    }
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
  bool has_prefix = false;
  if (!system.empty()) {
    oss << "System: " << system;
    has_prefix = true;
  }
  for (const auto& [role, content] : messages) {
    if (has_prefix) {
      oss << "\n\n";
    }
    oss << role << ": " << content;
    has_prefix = true;
  }
  if (has_prefix) {
    oss << "\n\n";
  }
  oss << "Assistant: ";
  if (enable_think) {
    oss << "<think";
  } else {
    oss << "<think>\n</think>\n";
  }
  return oss.str();
}

int InferenceEngine::count_tokens(const std::string& text) const {
  return static_cast<int>(tokenizer_->encode(text).size());
}

}  // namespace rwkv7_server
