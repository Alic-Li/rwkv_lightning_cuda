#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "fake_backend.hpp"
#include "rwkv_inference_engine.hpp"
#include "rwkv_tokenizer.hpp"
#include "test_common.hpp"

namespace {

std::vector<std::vector<float>> build_logits_steps(
    const std::vector<int>& token_ids,
    int vocab_size) {
  std::vector<std::vector<float>> steps;
  steps.reserve(token_ids.size() + 1);
  for (int token_id : token_ids) {
    steps.push_back(rwkv_test::one_hot_logits(vocab_size, token_id));
  }
  steps.push_back(rwkv_test::one_hot_logits(vocab_size, 0));
  return steps;
}

}  // namespace

int main() {
  try {
    auto tokenizer = std::make_shared<rwkv7_server::TrieTokenizer>();
    TEST_EQ(tokenizer->load(rwkv_test::vocab_path().string()), rwkv7_server::kTokenizerSuccess);

    TEST_EQ(tokenizer->decode(tokenizer->encode("roundtrip")), std::string("roundtrip"));

    const std::string answer = "This is a test answer.";
    const auto answer_ids = tokenizer->encode(answer);
    TEST_CHECK(!answer_ids.empty());
    int vocab_size = 1;
    for (int token_id : answer_ids) {
      vocab_size = std::max(vocab_size, token_id + 1);
    }

    auto fake_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend");
    rwkv7_server::InferenceEngine engine(fake_backend, tokenizer, fake_backend->model_name());

    TEST_EQ(
        engine.format_openai_prompt(
            "SYS",
            {{"User", "Q1"}, {"Assistant", "A1"}, {"User", "Q2"}},
            false),
        std::string("System: SYS\n\nUser: Q1\n\nAssistant: A1\n\nUser: Q2\n\nAssistant: <think>\n</think>\n"));
    TEST_EQ(
        engine.format_openai_prompt("SYS", {{"User", "Q"}}, true),
        std::string("System: SYS\n\nUser: Q\n\nAssistant: <think"));
    TEST_EQ(engine.count_tokens(answer), static_cast<int>(answer_ids.size()));

    rwkv7_server::GenerationState invalid_state;
    invalid_state.batch_size = 2;
    TEST_THROW(engine.batch_generate_state({"prompt"}, invalid_state, rwkv7_server::GenerateOptions{}));
    TEST_THROW(engine.batch_generate_state_stream({"prompt", "prompt"}, invalid_state, rwkv7_server::GenerateOptions{}, 1, [](int, const std::string&) { return true; }));

    if (!rwkv_test::cuda_device_available()) {
      std::cout << "rwkv_inference_engine_test partial pass: CPU-only checks passed, GPU generation checks skipped\n";
      return 0;
    }

    rwkv7_server::GenerateOptions options;
    options.max_tokens = static_cast<int>(answer_ids.size()) + 1;
    options.stop_tokens = {0};
    options.top_k = 0;
    options.top_p = 0.0;
    options.temperature = 1.0;
    options.alpha_presence = 0.0;
    options.alpha_frequency = 0.0;
    options.alpha_decay = 1.0;

    const std::string prompt = "System: test\n\nUser: hi\n\nAssistant:";
    const auto prompt_ids = tokenizer->encode(prompt);
    const auto outputs = engine.batch_generate({prompt}, options);
    TEST_EQ(outputs.size(), static_cast<std::size_t>(1));
    TEST_EQ(outputs.front(), answer);
    TEST_EQ(fake_backend->prefill_batches().size(), static_cast<std::size_t>(1));
    TEST_EQ(fake_backend->prefill_batches().front().size(), static_cast<std::size_t>(1));
    TEST_CHECK(std::vector<int64_t>(
                   fake_backend->prefill_batches().front().front().begin(),
                   fake_backend->prefill_batches().front().front().end()) ==
               std::vector<int64_t>(prompt_ids.begin(), prompt_ids.end()));
    TEST_EQ(fake_backend->decode_tokens().size(), answer_ids.size());

    auto stream_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend-stream");
    rwkv7_server::InferenceEngine stream_engine(stream_backend, tokenizer, stream_backend->model_name());
    std::string streamed;
    std::vector<int> stream_indices;
    stream_engine.batch_generate_stream(
        {prompt},
        options,
        1,
        [&](int index, const std::string& chunk) {
          stream_indices.push_back(index);
          streamed += chunk;
          return true;
        });
    TEST_EQ(streamed, answer);
    TEST_CHECK(!stream_indices.empty());
    for (int index : stream_indices) {
      TEST_EQ(index, 0);
    }

    std::cout << "rwkv_inference_engine_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_inference_engine_test failed: " << e.what() << "\n";
    return 1;
  }
}
