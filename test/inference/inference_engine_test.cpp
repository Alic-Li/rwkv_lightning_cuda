#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "fake_backend.hpp"
#include "rwkv/inference/rwkv_inference_engine.hpp"
#include "rwkv/inference/rwkv_tokenizer.hpp"
#include "test_common.hpp"
#include "rwkv/inference/text_stop_matcher.hpp"

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

std::vector<std::vector<float>> build_force_reasoning_logits_steps() {
  // The GPU sampler operates on float4 vocabulary tiles.
  const int vocab_size = 756;
  std::vector<std::vector<float>> steps;
  for (size_t i = 0; i < 3; ++i) {
    std::vector<float> logits(static_cast<std::size_t>(vocab_size), -100.0f);
    logits[112 + i] = 90.0f;
    if (i == 1) {
      logits[111] = 100.0f;
    }
    if (i == 2) {
      logits[754] = 100.0f;
    }
    steps.push_back(std::move(logits));
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

    vocab_size = (vocab_size + 3) / 4 * 4;

    auto fake_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend");
    rwkv7_server::InferenceEngine engine(fake_backend, tokenizer, fake_backend->model_name());
    TEST_EQ(engine.prefill_chunk_size(), 128);
    TEST_THROW(rwkv7_server::InferenceEngine(fake_backend, tokenizer, "invalid-chunk", 0));

    TEST_EQ(
        engine.format_openai_prompt(
            "SYS",
            {{"User", "Q1"}, {"Assistant", "A1"}, {"User", "Q2"}},
            false),
        std::string("System: SYS\n\nUser: Q1\n\nAssistant: A1\n\nUser: Q2\n\nAssistant: <think></think"));
    TEST_EQ(
        engine.format_openai_prompt("SYS", {{"User", "Q"}}, true),
        std::string("System: SYS\n\nUser: Q\n\nAssistant: <think"));
    TEST_EQ(
        engine.format_openai_prompt("SYS", {{"User", "Q"}}, rwkv7_server::ThinkType::EnShort),
        std::string("System: SYS\n\nUser: Q (think a bit)\n\nAssistant: <think"));
    TEST_EQ(
        engine.format_openai_prompt("SYS", {{"User", "Q"}}, rwkv7_server::ThinkType::None),
        std::string("System: SYS\n\nUser: Q\n\nAssistant: "));
    TEST_EQ(engine.count_tokens(answer), static_cast<int>(answer_ids.size()));

    rwkv7_server::GenerationState invalid_state;
    invalid_state.batch_size = 2;
    TEST_THROW(engine.batch_generate_state({"prompt"}, invalid_state, rwkv7_server::GenerateOptions{}));
    TEST_THROW(engine.batch_generate_state_stream({"prompt", "prompt"}, invalid_state, rwkv7_server::GenerateOptions{}, 1, [](int, const std::string&) { return true; }));

    // Bounded byte tails work across token and UTF-8 boundaries.
    rwkv7_server::TextStopMatcher matcher({"</tool_call>", "\nUser:", "结束"});
    std::string tail;
    TEST_EQ(matcher.append(tail, std::string(10000, 'x') + "</tool_"), std::string{});
    TEST_CHECK(tail.size() <= std::string("</tool_call>").size());
    TEST_EQ(matcher.append(tail, "call>"), std::string("</tool_call>"));
    tail.clear();
    const std::string utf8_stop = "结束";
    TEST_EQ(matcher.append(tail, utf8_stop.substr(0, 2)), std::string{});
    TEST_EQ(matcher.append(tail, utf8_stop.substr(2)), utf8_stop);
    tail.clear();
    TEST_EQ(matcher.append(tail, "\nUser: more"), std::string{});
    TEST_EQ(matcher.append(tail, "\nUser:"), std::string("\nUser:"));
    rwkv7_server::TextStopMatcher overlap({"User:", "\nUser:"});
    tail.clear();
    TEST_EQ(overlap.append(tail, "\nUser:"), std::string("User:"));
    rwkv7_server::TextStopMatcher empty({});
    tail.clear();
    TEST_EQ(empty.append(tail, "anything"), std::string{});
    TEST_CHECK(tail.empty());

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

    auto chunk_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend-chunk-prefill");
    rwkv7_server::InferenceEngine chunk_engine(
        chunk_backend, tokenizer, chunk_backend->model_name(), 3);
    auto chunk_state = chunk_backend->create_state(1);
    rwkv7_server::DeviceLogits chunk_logits;
    const std::string chunk_prompt = prompt + prompt + prompt;
    const int chunk_prompt_tokens = chunk_engine.prefill_prompt(chunk_prompt, chunk_state, chunk_logits);
    TEST_CHECK(chunk_prompt_tokens > 3);
    TEST_CHECK(chunk_backend->prefill_batches().size() > 1);
    for (const auto& batch : chunk_backend->prefill_batches()) {
      TEST_EQ(batch.size(), static_cast<std::size_t>(1));
      TEST_CHECK(batch.front().size() <= static_cast<std::size_t>(3));
    }

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

    auto force_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_force_reasoning_logits_steps(),
        "fake-backend-force");
    rwkv7_server::InferenceEngine force_engine(force_backend, tokenizer, force_backend->model_name());
    rwkv7_server::GenerateOptions force_options = options;
    force_options.max_tokens = 4;
    force_options.force_reasoning = true;
    const auto force_outputs = force_engine.batch_generate({prompt}, force_options);
    TEST_EQ(force_outputs.size(), static_cast<std::size_t>(1));
    TEST_EQ(force_outputs.front(), std::string("opq"));

    auto batch_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend-batch");
    rwkv7_server::InferenceEngine batch_engine(batch_backend, tokenizer, batch_backend->model_name());
    const std::string longer_prompt = prompt + "\nextra";
    const auto batch_outputs = batch_engine.batch_generate({prompt, longer_prompt}, options);
    TEST_EQ(batch_outputs.size(), static_cast<std::size_t>(2));
    TEST_EQ(batch_outputs[0], answer);
    TEST_EQ(batch_outputs[1], answer);
    TEST_EQ(batch_backend->prefill_batches().size(), static_cast<std::size_t>(2));
    TEST_EQ(batch_backend->prefill_batches()[0].size(), static_cast<std::size_t>(2));
    TEST_EQ(batch_backend->prefill_batches()[1].size(), static_cast<std::size_t>(1));
    TEST_EQ(batch_backend->decode_tokens().size(), answer_ids.size());
    for (const auto& decode_step : batch_backend->decode_tokens()) {
      TEST_EQ(decode_step.size(), static_cast<std::size_t>(2));
    }

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

    auto batch_stream_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size),
        "fake-backend-stream-batch");
    rwkv7_server::InferenceEngine batch_stream_engine(
        batch_stream_backend, tokenizer, batch_stream_backend->model_name());
    std::vector<std::string> streamed_batch(2);
    batch_stream_engine.batch_generate_stream(
        {prompt, longer_prompt},
        options,
        1,
        [&](int index, const std::string& chunk) {
          streamed_batch.at(static_cast<std::size_t>(index)) += chunk;
          return true;
        });
    TEST_EQ(streamed_batch[0], answer);
    TEST_EQ(streamed_batch[1], answer);
    TEST_EQ(batch_stream_backend->prefill_batches().size(), static_cast<std::size_t>(2));
    TEST_EQ(batch_stream_backend->prefill_batches()[0].size(), static_cast<std::size_t>(2));
    TEST_EQ(batch_stream_backend->prefill_batches()[1].size(), static_cast<std::size_t>(1));

    // Exercise every generation entry point at the budget boundary. The final
    // sampled EOS still means stop, even when it uses the last allowed step.
    for (int budget : {0, 1, static_cast<int>(answer_ids.size()),
                       static_cast<int>(answer_ids.size()) + 1}) {
      auto bounded = options;
      bounded.max_tokens = budget;
      const std::string expected = budget > static_cast<int>(answer_ids.size())
          ? "stop" : "length";
      rwkv7_server::InferenceEngine::GenerationStats stats;
      engine.batch_generate({prompt}, bounded, &stats);
      TEST_EQ(stats.finish_reasons, std::vector<std::string>{expected});
      const auto accept = [](int, const std::string&) { return true; };
      stats = engine.batch_generate_stream({prompt}, bounded, 100, accept);
      TEST_EQ(stats.finish_reasons, std::vector<std::string>{expected});
      auto state = fake_backend->create_state(1);
      engine.batch_generate_state({prompt}, state, bounded, &stats);
      TEST_EQ(stats.finish_reasons, std::vector<std::string>{expected});
      state = fake_backend->create_state(1);
      stats = engine.batch_generate_state_stream({prompt}, state, bounded, 100, accept);
      TEST_EQ(stats.finish_reasons, std::vector<std::string>{expected});
      rwkv7_server::DeviceLogits logits;
      state = fake_backend->create_state(1);
      engine.prefill_prompt(prompt, state, logits);
      stats = engine.generate_from_logits_stream(state, logits, bounded, 100, accept);
      TEST_EQ(stats.finish_reasons, std::vector<std::string>{expected});
    }

    // Text stops span tokens and terminate before the output chunk fills.
    TEST_CHECK(answer_ids.size() >= 2);
    const std::string prefix = tokenizer->decode(std::vector<int>{answer_ids[0], answer_ids[1]});
    auto text_options = options;
    text_options.stop = std::vector<std::string>{prefix};
    // Text mode ignores even a conflicting legacy ID at the engine level.
    text_options.stop_tokens = {answer_ids[0]};
    for (int chunk_size : {0, 1, 100}) {
      std::vector<std::string> output(2);
      std::vector<int> stops(2, 0);
      const auto text_stats = engine.batch_generate_stream(
          {prompt, longer_prompt}, text_options, chunk_size,
          [&](int index, const std::string& chunk) {
            if (chunk.empty()) ++stops[index];
            else output[index] += chunk;
            return true;
          });
      TEST_EQ(output, (std::vector<std::string>{prefix, prefix}));
      TEST_EQ(stops, (std::vector<int>{1, 1}));
      TEST_EQ(text_stats.matched_stops, (std::vector<std::string>{prefix, prefix}));
      TEST_EQ(text_stats.finish_reasons, (std::vector<std::string>{"stop", "stop"}));
      TEST_EQ(text_stats.generated_tokens, 4);
    }
    rwkv7_server::InferenceEngine::GenerationStats text_stats;
    TEST_EQ(engine.batch_generate({prompt}, text_options, &text_stats),
            (std::vector<std::string>{prefix}));
    TEST_EQ(text_stats.matched_stops[0], prefix);
    for (int chunk_size : {0, 1, 100}) {
      text_options.max_tokens = 2;  // Match exactly at the budget boundary.
      auto state = fake_backend->create_state(1);
      std::string output;
      const auto collect = [&](int, const std::string& chunk) { output += chunk; return true; };
      text_stats = engine.batch_generate_state_stream({prompt}, state, text_options, chunk_size, collect);
      TEST_EQ(output, prefix);
      TEST_EQ(text_stats.matched_stops[0], prefix);
      TEST_EQ(text_stats.finish_reasons[0], std::string("stop"));
      TEST_EQ(fake_backend->decode_tokens().back()[0], answer_ids[1]);
      state = fake_backend->create_state(1);
      TEST_EQ(engine.batch_generate_state({prompt}, state, text_options, &text_stats)[0], prefix);
      TEST_EQ(text_stats.matched_stops[0], prefix);
      state = fake_backend->create_state(1);
      rwkv7_server::DeviceLogits logits;
      engine.prefill_prompt(prompt, state, logits);
      output.clear();
      text_stats = engine.generate_from_logits_stream(state, logits, text_options, chunk_size, collect);
      TEST_EQ(output, prefix);
      TEST_EQ(text_stats.matched_stops[0], prefix);
      TEST_EQ(text_stats.finish_reasons[0], std::string("stop"));
    }
    // A stop prefix generated before pause must still match after resume.
    auto resumed_state = fake_backend->create_state(1);
    rwkv7_server::DeviceLogits resumed_logits;
    engine.prefill_prompt(prompt, resumed_state, resumed_logits);
    int control_calls = 0;
    text_options.max_tokens = 8;
    text_stats = engine.generate_from_logits_stream(
        resumed_state, resumed_logits, text_options, 100,
        [](int, const std::string&) { return true; }, [&] { return ++control_calls > 1; });
    TEST_CHECK(text_stats.stopped);
    TEST_EQ(resumed_state.text_stop_tail, tokenizer->decode(answer_ids[0]));
    text_stats = engine.generate_from_logits_stream(
        resumed_state, resumed_logits, text_options, 100,
        [](int, const std::string&) { return true; });
    TEST_EQ(text_stats.matched_stops[0], prefix);
    TEST_EQ(text_stats.generated_tokens, 1);
    TEST_CHECK(resumed_state.text_stop_tail.empty());
    text_options.stop = std::vector<std::string>{};
    text_options.max_tokens = 2;
    engine.batch_generate({prompt}, text_options, &text_stats);
    TEST_EQ(text_stats.finish_reasons[0], std::string("length"));
    TEST_CHECK(text_stats.matched_stops[0].empty());

    auto bounded = options;
    bounded.max_tokens = 1;
    bounded.stop_tokens.clear();
    rwkv7_server::InferenceEngine::GenerationStats stats;
    engine.batch_generate({prompt}, bounded, &stats);
    TEST_EQ(stats.finish_reasons[0], std::string("length"));
    bounded.stop_tokens = {answer_ids.front()};
    engine.batch_generate({prompt}, bounded, &stats);
    TEST_EQ(stats.finish_reasons[0], std::string("stop"));
    TEST_EQ(stats.generated_tokens, 0);

    // Sorted row 1 is the shorter original prompt at index 0. Only that row
    // emits EOS; the longer prompt must keep its independent length reason.
    auto mixed_backend = std::make_shared<rwkv_test::FakeModelBackend>(
        build_logits_steps(answer_ids, vocab_size), "mixed-finish", 1);
    rwkv7_server::InferenceEngine mixed_engine(mixed_backend, tokenizer, "mixed-finish");
    bounded = options;
    bounded.max_tokens = 1;
    mixed_engine.batch_generate({prompt, longer_prompt}, bounded, &stats);
    TEST_EQ(stats.finish_reasons, (std::vector<std::string>{"stop", "length"}));
    stats = mixed_engine.batch_generate_stream(
        {prompt, longer_prompt}, bounded, 100,
        [](int, const std::string&) { return true; });
    TEST_EQ(stats.finish_reasons, (std::vector<std::string>{"stop", "length"}));
    text_options.stop = std::vector<std::string>{tokenizer->decode(answer_ids.front())};
    text_options.max_tokens = 2;
    auto mixed_state = mixed_backend->create_state(2);
    int notifications = 0;
    stats = mixed_engine.batch_generate_stream_with_state(
        {prompt, longer_prompt}, mixed_state, text_options, 100,
        [](int, const std::string&) { return true; }, {}, {}, std::nullopt,
        [&](const rwkv7_server::InferenceEngine::GenerationStats& row_stats) {
          ++notifications;
          TEST_EQ(row_stats.matched_stops[1], text_options.stop->front());
          TEST_EQ(row_stats.finish_reasons[1], std::string("stop"));
        });
    TEST_EQ(notifications, 1);
    TEST_EQ(stats.finish_reasons, (std::vector<std::string>{"length", "stop"}));
    TEST_CHECK(stats.matched_stops[0].empty());

    stats = engine.batch_generate_stream(
        {prompt}, options, 1, [](int, const std::string&) { return true; },
        [] { return true; });
    TEST_EQ(stats.finish_reasons[0], std::string("stop"));
    TEST_CHECK(stats.stopped);
    TEST_EQ(stats.generated_tokens, 0);
    stats = engine.batch_generate_stream(
        {prompt}, options, 1, [](int, const std::string&) { return false; });
    TEST_EQ(stats.finish_reasons[0], std::string("stop"));
    TEST_CHECK(stats.stopped);

    std::cout << "rwkv_inference_engine_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_inference_engine_test failed: " << e.what() << "\n";
    return 1;
  }
}
