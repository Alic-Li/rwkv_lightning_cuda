#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <set>
#include <thread>

#include "rwkv/inference/rwkv_inference_engine.hpp"
#include "test_common.hpp"

using namespace rwkv7_server;
namespace {
// A recurrent backend with real GPU buffers. Output and next state depend on
// every input token, the initial state and the bound adapter, not a global step.
class Backend final : public IModelBackend {
 public:
  struct Call { bool prefill; int rows; std::string adapter; std::thread::id thread; };
  mutable std::vector<Call> calls;
  mutable std::atomic<int> decode_steps{0};
  mutable std::atomic<bool> fail_next{false};
  mutable std::atomic<bool> fail_create{false}, fail_copy{false};
  std::string name = "continuous-test";
  GenerationState create_state(int n) const override {
    if (fail_create.exchange(false)) throw std::runtime_error("injected allocation failure");
    GenerationState s;
    s.batch_size = n;
    rwkv_test::copy_host_to_device(std::vector<int>(n, 0), s.elapsed, "state", "state");
    return s;
  }
  void forward_prefill(const std::vector<std::vector<int64_t>>& tokens,
                       GenerationState& s, DeviceLogits& out) const override {
    forward(tokens, s, out, true);
  }
  void forward_decode(const std::vector<int64_t>& tokens,
                      GenerationState& s, DeviceLogits& out) const override {
    ++decode_steps;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (fail_next.exchange(false)) throw std::runtime_error("injected decode failure");
    std::vector<std::vector<int64_t>> rows;
    for (auto t : tokens) rows.push_back({t});
    forward(rows, s, out, false);
  }
  void forward(const std::vector<std::vector<int64_t>>& tokens,
               GenerationState& s, DeviceLogits& out, bool prefill) const {
    const std::string identity = s.adapter ? s.adapter->identity : "";
    TEST_EQ(s.adapter_identity, identity);
    if (s.advanced) TEST_EQ(s.effective_key, identity + s.initial_state_identity);
    calls.push_back({prefill, int(tokens.size()), identity, std::this_thread::get_id()});
    auto values = rwkv_test::copy_device_buffer(s.elapsed, "read state");
    std::vector<float> logits(tokens.size() * vocab_size(), -100.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
      for (auto t : tokens[i]) values[i] = (values[i] * 17 + int(t) + (s.adapter ? 3 : 0)) % 23;
      logits[i * vocab_size() + 98 + values[i]] = 10.0f;
      logits[i * vocab_size() + 98 + (values[i] + 1) % 23] = 9.0f;
    }
    rwkv_test::copy_host_to_device(values, s.elapsed, "state", "state");
    out.rows = int(tokens.size()); out.vocab_size = vocab_size();
    rwkv_test::copy_host_to_device(logits, out.values, "logits", "logits");
    s.advanced = true;
    s.effective_key = identity + s.initial_state_identity;
  }
  void copy_state_slice(const GenerationState& src, int from, GenerationState& dst,
                        int to, int n) const override {
    if (fail_copy.exchange(false)) throw std::runtime_error("injected copy failure");
    if (dst.advanced) TEST_EQ(src.effective_key, dst.effective_key);
    rwkv7_fast_v4::check_cuda(cudaMemcpy(dst.elapsed.p + to, src.elapsed.p + from,
        n * sizeof(int), cudaMemcpyDeviceToDevice), "copy state");
    dst.adapter = src.adapter; dst.adapter_identity = src.adapter_identity;
    dst.effective_key = src.effective_key; dst.initial_state_identity = src.initial_state_identity;
    dst.advanced = src.advanced;
  }
  void copy_logits_slice(const DeviceLogits& src, int from, DeviceLogits& dst,
                         int to, int n) const override {
    rwkv7_fast_v4::check_cuda(cudaMemcpy(dst.values.p + to * vocab_size(),
        src.values.p + from * vocab_size(), n * vocab_size() * sizeof(float),
        cudaMemcpyDeviceToDevice), "copy logits");
  }
  PrefillCapacity query_prefill_capacity(int) const override { return {}; }
  int vocab_size() const override { return 256; }
  const std::string& model_name() const override { return name; }
  const std::string& model_path() const override { return name; }
};
struct Result { std::string text; int state; InferenceEngine::GenerationStats stats; };
}

int main() {
  try {
    if (!rwkv_test::cuda_device_available()) return 77;
    auto tokenizer = std::make_shared<TrieTokenizer>();
    TEST_EQ(tokenizer->load(rwkv_test::vocab_path().string()), kTokenizerSuccess);
    auto backend = std::make_shared<Backend>();
    InferenceEngine engine(backend, tokenizer, "continuous", 2, 0, 4);
    auto package = std::make_shared<rwkv7_miss::Package>();
    package->version = "test-adapter";
    const auto adapter = std::make_shared<rwkv7_miss::AdapterHandle>(package, 1.0f);
    auto run = [&](int id, const InferenceEngine::ControlCallback& stop = InferenceEngine::ControlCallback{}) {
      GenerateOptions o;
      o.max_tokens = id == 0 ? 50 : 5 + id;
      o.stop_tokens.clear(); o.top_p = 0; o.temperature = 0.2 + id * 0.1;
      o.alpha_presence = id % 3 == 0 ? 20 : 0;
      o.alpha_frequency = id % 2 ? 3 : 0;
      o.alpha_decay = 0.8 + id * 0.01;
      if (id % 2) o.adapter = adapter;
      auto state = backend->create_state(1);
      state.initial_state_identity = "initial-" + std::to_string(id);
      const std::string identity = state.initial_state_identity;
      rwkv_test::copy_host_to_device(std::vector<int>{id}, state.elapsed, "state", "state");
      Result result;
      result.stats = engine.batch_generate_stream_with_state(
          {id == 7 ? std::string(160, 'x') : "User: hello"}, state, o, 1,
          [&](int index, const std::string& text) { TEST_EQ(index, 0); result.text += text; return true; }, stop);
      result.state = rwkv_test::copy_device_buffer(state.elapsed, "final state")[0];
      TEST_EQ(state.initial_state_identity, identity);
      TEST_EQ(state.adapter_identity, o.adapter ? o.adapter->identity : "");
      TEST_EQ(state.effective_key, state.adapter_identity + identity);
      return result;
    };
    std::vector<Result> expected;
    for (int i = 0; i < 8; ++i) expected.push_back(run(i));
    backend->calls.clear(); backend->decode_steps = 0;
    std::vector<std::future<Result>> futures;
    // Begin one decode, then join it with short and chunked long prompts.
    futures.push_back(std::async(std::launch::async, [&] { return run(0); }));
    while (backend->decode_steps == 0) std::this_thread::yield();
    for (int i = 1; i < 8; ++i)
      futures.push_back(std::async(std::launch::async, [&, i] { return run(i); }));
    for (int i = 0; i < 8; ++i) {
      const auto actual = futures[i].get();
      TEST_EQ(actual.text, expected[i].text);
      TEST_EQ(actual.state, expected[i].state);
      TEST_EQ(actual.stats.generated_tokens, expected[i].stats.generated_tokens);
    }
    int largest_batch = 0;
    bool saw_decode = false, interleaved = false;
    std::set<std::thread::id> owners;
    for (const auto& call : backend->calls) {
      owners.insert(call.thread);
      if (!call.prefill) { saw_decode = true; largest_batch = std::max(largest_batch, call.rows); }
      else if (saw_decode) interleaved = true;
      TEST_CHECK(call.rows <= 4);
    }
    TEST_CHECK(largest_batch > 1);
    TEST_CHECK(interleaved);
    TEST_EQ(owners.size(), size_t(1));

    // Cancellation and callback exceptions must drain/export slots before the
    // caller destroys its state; a later request must still work.
    int checks = 0;
    auto cancelled = run(0, [&] { return ++checks > 3; });
    TEST_CHECK(cancelled.stats.stopped);
    TEST_EQ(cancelled.stats.generated_tokens, 3);
    auto survivor = std::async(std::launch::async, [&] { return run(0); });
    TEST_THROW(engine.batch_generate_stream({"hello"}, GenerateOptions{}, 1,
        [](int, const std::string&) -> bool { throw std::runtime_error("consumer failed"); }));
    TEST_EQ(survivor.get().text, expected[0].text);
    backend->fail_next = true;
    TEST_THROW(run(2));
    TEST_EQ(run(2).text, expected[2].text);
    TEST_THROW(run(2, []() -> bool { throw std::runtime_error("control failed"); }));
    TEST_EQ(run(2).text, expected[2].text);
    for (bool allocation : {true, false}) {
      auto state = backend->create_state(1);
      TEST_THROW(engine.batch_generate_stream_with_state({"hello"}, state, GenerateOptions{}, 1,
          [](int, const std::string&) { return true; }, {},
          [&](const InferenceEngine::GenerationStats&) {
            if (allocation) backend->fail_create = true;
            else backend->fail_copy = true;
          }));
      TEST_EQ(run(2).text, expected[2].text);
    }

    // A slow HTTP consumer must relinquish its slots without losing recurrent
    // state, penalty history or output. Another request can finish meanwhile.
    GenerateOptions slow_options;
    slow_options.max_tokens = 300; slow_options.stop_tokens.clear(); slow_options.top_p = 0;
    const auto slow_expected = engine.batch_generate({"slow"}, slow_options)[0];
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    const int before_slow = backend->decode_steps;
    auto slow = std::async(std::launch::async, [&] {
      bool first = true;
      std::string output;
      engine.batch_generate_stream({"slow"}, slow_options, 1,
          [&](int, const std::string& text) {
            if (first) { first = false; entered.set_value(); released.wait(); }
            output += text;
            return true;
          });
      return output;
    });
    entered.get_future().wait();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (backend->decode_steps - before_slow < 129 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto fast = std::async(std::launch::async, [&] { return run(2); });
    const bool independent = fast.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    release.set_value();
    TEST_CHECK(independent);
    TEST_EQ(fast.get().text, expected[2].text);
    TEST_EQ(slow.get(), slow_expected);

    std::cout << "continuous batching passed: largest merged batch=" << largest_batch << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
