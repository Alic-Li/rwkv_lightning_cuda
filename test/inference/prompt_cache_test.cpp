#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rwkv/inference/rwkv_inference_engine.hpp"
#include "test_common.hpp"

namespace {
// Every token updates a per-row recurrent value. Unlike a fixed-logit fake,
// this detects wrong row copies, duplicated tokens and post-decode snapshots.
class RecurrentBackend final : public rwkv7_server::IModelBackend {
 public:
  explicit RecurrentBackend(bool wkv32) : wkv32_(wkv32) {}
  mutable int prefilled = 0;
  std::string name = "recurrent-test";
  rwkv7_server::GenerationState create_state(int count) const override {
    rwkv7_server::GenerationState state;
    state.batch_size = count;
    state.wkv32 = wkv32_;
    state.shift.resize(count, "test shift");
    state.shift.zero("test shift");
    if (wkv32_) {
      state.wkv_state32.resize(count, "test wkv32");
      state.wkv_state32.zero("test wkv32");
    } else {
      state.wkv_state.resize(count, "test wkv16");
      state.wkv_state.zero("test wkv16");
    }
    rwkv_test::copy_host_to_device(std::vector<int>(count, 0), state.elapsed, "state", "state");
    return state;
  }
  void forward_prefill(const std::vector<std::vector<int64_t>>& rows,
                       rwkv7_server::GenerationState& state,
                       rwkv7_server::DeviceLogits& logits) const override {
    auto values = rwkv_test::copy_device_buffer(state.elapsed, "read state");
    const auto shift = read_half_bits(state.shift);
    std::vector<uint16_t> new_shift(rows.size()), new_wkv16(rows.size());
    std::vector<float> new_wkv32(rows.size());
    const auto wkv16 = state.wkv32 ? std::vector<uint16_t>{} : read_half_bits(state.wkv_state);
    const auto wkv32 = state.wkv32
        ? rwkv_test::copy_device_buffer(state.wkv_state32, "read wkv32") : std::vector<float>{};
    std::vector<float> output(rows.size() * vocab_size(), -100.0f);
    for (size_t i = 0; i < rows.size(); ++i) {
      int s = shift[i], w = state.wkv32 ? static_cast<int>(wkv32[i]) : wkv16[i];
      for (auto token : rows[i]) {
        values[i] = (values[i] * 31 + token + s + w) % 251;
        s = values[i] * 7 % 251;
        w = values[i] * 11 % 251;
      }
      new_shift[i] = static_cast<uint16_t>(s);
      new_wkv16[i] = static_cast<uint16_t>(w);
      new_wkv32[i] = static_cast<float>(w);
      prefilled += static_cast<int>(rows[i].size());
      output[i * vocab_size() + 1 + values[i]] = 100.0f;
    }
    state.advanced = true;
    rwkv_test::copy_host_to_device(values, state.elapsed, "state", "state");
    write_half_bits(new_shift, state.shift);
    if (state.wkv32) {
      rwkv_test::copy_host_to_device(new_wkv32, state.wkv_state32, "wkv32", "wkv32");
    } else {
      write_half_bits(new_wkv16, state.wkv_state);
    }
    logits.rows = static_cast<int>(rows.size());
    logits.vocab_size = vocab_size();
    rwkv_test::copy_host_to_device(output, logits.values, "logits", "logits");
  }
  void forward_decode(const std::vector<int64_t>& tokens, rwkv7_server::GenerationState& state,
                      rwkv7_server::DeviceLogits& logits) const override {
    std::vector<std::vector<int64_t>> rows;
    for (auto token : tokens) rows.push_back({token});
    const int before = prefilled;
    forward_prefill(rows, state, logits);
    prefilled = before;
  }
  void copy_state_slice(const rwkv7_server::GenerationState& src, int from,
                        rwkv7_server::GenerationState& dst, int to, int count) const override {
    const auto source = rwkv_test::copy_device_buffer(src.elapsed, "source state");
    auto dest = rwkv_test::copy_device_buffer(dst.elapsed, "dest state");
    std::copy_n(source.begin() + from, count, dest.begin() + to);
    rwkv_test::copy_host_to_device(dest, dst.elapsed, "state", "state");
    copy_buffer_slice(src.shift, from, dst.shift, to, count);
    if (src.wkv32) {
      copy_buffer_slice(src.wkv_state32, from, dst.wkv_state32, to, count);
    } else {
      copy_buffer_slice(src.wkv_state, from, dst.wkv_state, to, count);
    }
    dst.advanced = src.advanced;
    dst.adapter = src.adapter;
    dst.adapter_identity = src.adapter_identity;
    dst.effective_key = src.effective_key;
    dst.initial_state_identity = src.initial_state_identity;
  }
  void copy_logits_slice(const rwkv7_server::DeviceLogits& src, int from,
                         rwkv7_server::DeviceLogits& dst, int to, int count) const override {
    const auto source = rwkv_test::copy_device_buffer(src.values, "source logits");
    auto dest = rwkv_test::copy_device_buffer(dst.values, "dest logits");
    std::copy_n(source.begin() + from * vocab_size(), count * vocab_size(),
                dest.begin() + to * vocab_size());
    rwkv_test::copy_host_to_device(dest, dst.values, "logits", "logits");
  }
  rwkv7_server::PrefillCapacity query_prefill_capacity(int) const override { return {}; }
  int vocab_size() const override { return 256; }
  const std::string& model_name() const override { return name; }
  const std::string& model_path() const override { return name; }

 private:
  bool wkv32_;
  static std::vector<uint16_t> read_half_bits(const rwkv7_fast_v4::DeviceBuffer<half>& buffer) {
    std::vector<uint16_t> host(buffer.n);
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(host.data(), buffer.p, host.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "read test half bits");
    return host;
  }
  static void write_half_bits(const std::vector<uint16_t>& host,
                              rwkv7_fast_v4::DeviceBuffer<half>& buffer) {
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
        "write test half bits");
  }
  template <typename T>
  static void copy_buffer_slice(const rwkv7_fast_v4::DeviceBuffer<T>& src, int from,
                                rwkv7_fast_v4::DeviceBuffer<T>& dst, int to, int count) {
    rwkv7_fast_v4::check_cuda(
        cudaMemcpy(dst.p + to, src.p + from, count * sizeof(T), cudaMemcpyDeviceToDevice),
        "copy test state slice");
  }
};
}

int main() {
  try {
    if (!rwkv_test::cuda_device_available()) return 77;
    auto tokenizer = std::make_shared<rwkv7_server::TrieTokenizer>();
    TEST_EQ(tokenizer->load(rwkv_test::vocab_path().string()), rwkv7_server::kTokenizerSuccess);
    for (bool wkv32 : {false, true}) {
      auto backend = std::make_shared<RecurrentBackend>(wkv32);
      rwkv7_server::InferenceEngine engine(backend, tokenizer, "cache-test", 3, 4);
      rwkv7_server::GenerateOptions options;
      options.max_tokens = 3;
      options.stop_tokens.clear();
      options.top_k = 1;
      options.alpha_presence = options.alpha_frequency = 0;
      auto run = [&](rwkv7_server::InferenceEngine& target, const std::vector<std::string>& prompts,
                     bool cached, std::string state_id = "", std::string fingerprint = "zero") {
        auto state = backend->create_state(static_cast<int>(prompts.size()));
        state.initial_state_identity = fingerprint;
        if (fingerprint != "zero") {
          rwkv_test::copy_host_to_device(std::vector<int>(prompts.size(), 17), state.elapsed,
                                        "initial state", "initial state");
        }
        std::vector<std::string> output(prompts.size());
        const auto stats = target.batch_generate_stream_with_state(
            prompts, state, options, 1,
            [&](int row, const std::string& chunk) { output.at(row) += chunk; return true; },
            {}, {}, cached ? std::optional<std::string>(state_id) : std::nullopt);
        return std::make_pair(output, stats.cached_prompt_tokens);
      };
      const std::string prefix = "Hello world\n\n";
      const std::string extended = prefix + "Tool result: success\n\n";
      const auto ids = tokenizer->encode(prefix);
      const auto extended_ids = tokenizer->encode(extended);
      TEST_CHECK(std::equal(ids.begin(), ids.end(), extended_ids.begin()));
      const auto cold = run(engine, {prefix}, true);
      TEST_EQ(cold.second, 0);
      const int before_exact = backend->prefilled;
      const auto exact = run(engine, {prefix}, true);
      TEST_EQ(exact.first, cold.first);
      TEST_EQ(exact.second, static_cast<int>(ids.size()));
      TEST_EQ(backend->prefilled, before_exact);

      // Full hit + partial hit + miss, with suffix sorting unlike full-prompt sorting.
      const std::vector<std::string> mixed{prefix, extended, "A completely unrelated long question\n\n"};
      const int before_mixed = backend->prefilled;
      const auto warm = run(engine, mixed, true);
      int total = 0;
      for (const auto& prompt : mixed) total += engine.count_tokens(prompt);
      TEST_EQ(warm.second, static_cast<int>(2 * ids.size()));
      TEST_EQ(backend->prefilled - before_mixed, total - warm.second);
      TEST_EQ(warm.first, run(engine, mixed, false).first);
      const auto longer = extended + "Next tool result\n\n";
      const auto longest = run(engine, {longer}, true);
      TEST_EQ(longest.second, static_cast<int>(extended_ids.size()));
      TEST_EQ(longest.first, run(engine, {longer}, false).first);
      TEST_EQ(run(engine, {prefix}, true, "other-state").second, 0);
      TEST_EQ(run(engine, {prefix}, true, "other-state", "changed-file").second, 0);

      rwkv7_server::InferenceEngine initialized(backend, tokenizer, "initialized", 3, 4);
      run(initialized, {prefix}, true, "uploaded", "seeded-state");
      const auto seeded = run(initialized, {extended, "unrelated"}, true, "uploaded", "seeded-state");
      TEST_EQ(seeded.second, static_cast<int>(ids.size()));
      TEST_EQ(seeded.first,
              run(initialized, {extended, "unrelated"}, false, "uploaded", "seeded-state").first);

      // A string prefix is deliberately not a token prefix at a tokenizer boundary.
      std::string boundary_prefix, boundary_full;
      for (const std::string word : {"hello", "testing", "tokenization", "world"}) {
        for (size_t n = 1; n < word.size(); ++n) {
          const auto a = tokenizer->encode(word.substr(0, n)), b = tokenizer->encode(word);
          if (a.size() > b.size() || !std::equal(a.begin(), a.end(), b.begin())) {
            boundary_prefix = word.substr(0, n); boundary_full = word; break;
          }
        }
        if (!boundary_full.empty()) break;
      }
      TEST_CHECK(!boundary_full.empty());
      rwkv7_server::InferenceEngine boundary(backend, tokenizer, "boundary", 3, 2);
      run(boundary, {boundary_prefix}, true);
      const auto boundary_result = run(boundary, {boundary_full}, true);
      TEST_EQ(boundary_result.second, 0);
      TEST_EQ(boundary_result.first, run(boundary, {boundary_full}, false).first);

      rwkv7_server::InferenceEngine lru(backend, tokenizer, "lru", 3, 2);
      run(lru, {"alpha"}, true); run(lru, {"beta"}, true);
      TEST_CHECK(run(lru, {"alpha"}, true).second > 0);
      run(lru, {"gamma"}, true);
      TEST_CHECK(run(lru, {"alpha"}, true).second > 0);
      TEST_EQ(run(lru, {"beta"}, true).second, 0);
      rwkv7_server::InferenceEngine disabled(backend, tokenizer, "disabled", 3, 0);
      run(disabled, {prefix}, true);
      TEST_EQ(run(disabled, {prefix}, true).second, 0);

      // Adapter namespaces and engine instances cannot share cached states.
      auto package = std::make_shared<rwkv7_miss::Package>();
      package->version = "test-v1";
      options.max_tokens = 0;
      options.adapter = std::make_shared<rwkv7_miss::AdapterHandle>(package, 1.0f);
      TEST_EQ(run(engine, {prefix}, true).second, 0);
      TEST_CHECK(run(engine, {prefix}, true).second > 0);
      options.adapter_id = "same-weights-different-id";
      TEST_EQ(run(engine, {prefix}, true).second, 0);
      options.adapter = std::make_shared<rwkv7_miss::AdapterHandle>(package, 2.0f);
      TEST_EQ(run(engine, {prefix}, true).second, 0);
      rwkv7_server::InferenceEngine fresh(backend, tokenizer, "fresh", 3, 2);
      TEST_EQ(run(fresh, {prefix}, true).second, 0);
    }
    std::cout << "rwkv_prompt_cache_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
