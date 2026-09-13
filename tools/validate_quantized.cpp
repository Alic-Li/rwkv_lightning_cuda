#include "rwkv/inference/rwkv_tokenizer.hpp"
#include "rwkv/runtime/rwkv_server_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
struct Result {
  std::vector<std::vector<float>> logits;
  double prefill_ms = 0, decode_ms = 0;
};
std::vector<float> download(const rwkv7_server::DeviceLogits &logits) {
  std::vector<float> result(logits.values.n);
  rwkv7_fast_v4::check_cuda(cudaMemcpy(result.data(), logits.values.p,
                                       result.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost),
                            "copy logits");
  for (float v : result)
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite model logits");
  return result;
}
Result run(const char *path, const std::vector<int> &corpus,
           rwkv7_server::TrieTokenizer &tokenizer) {
  rwkv7_server::ModelBackend model(path, true, true, "no-fc", "", false);
  std::vector<std::vector<int64_t>> prompt(
      1, std::vector<int64_t>(corpus.begin(), corpus.begin() + 32));
  // Warm the exact prefill and decode shapes, then start from a fresh state.
  rwkv7_server::DeviceLogits logits;
  {
    auto warm = model.create_state(1);
    model.forward_prefill(prompt, warm, logits);
    model.forward_decode({corpus[32]}, warm, logits);
  }
  auto state = model.create_state(1);
  Result result;
  auto t = Clock::now();
  model.forward_prefill(prompt, state, logits);
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  result.logits.push_back(download(logits));
  std::vector<double> samples;
  for (int i = 32; i < 64; ++i) {
    t = Clock::now();
    model.forward_decode({corpus[i]}, state, logits);
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - t).count());
    result.logits.push_back(download(logits));
  }
  std::sort(samples.begin(), samples.end());
  result.decode_ms = (samples[15] + samples[16]) * 0.5;
  std::cout << std::setprecision(6) << "model=" << path
            << " prefill32_ms=" << result.prefill_ms
            << " decode_median_ms=" << result.decode_ms
            << " tok_s=" << 1000 / result.decode_ms << '\n';
  auto generated_state = model.create_state(1);
  const auto text =
      tokenizer.encode("User: 请用一句话解释什么是人工智能。\n\nAssistant:");
  model.forward_prefill({std::vector<int64_t>(text.begin(), text.end())},
                        generated_state, logits);
  std::vector<int> generated;
  for (int i = 0; i < 32; ++i) {
    auto values = download(logits);
    const int token =
        std::max_element(values.begin(), values.end()) - values.begin();
    if (token == 0)
      break;
    generated.push_back(token);
    model.forward_decode({token}, generated_state, logits);
  }
  std::cout << "greedy_text=" << tokenizer.decode(generated) << '\n';
  return result;
}
} // namespace
int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: rwkv_quantized_validate MODEL VOCAB "
                 "[BF16_REFERENCE_MODEL]\n";
    return 2;
  }
  try {
    rwkv7_server::TrieTokenizer tokenizer;
    if (tokenizer.load(argv[2]) != 0)
      throw std::runtime_error("failed to load vocabulary");
    auto corpus = tokenizer.encode(
        "Artificial intelligence is the study of building computer systems "
        "that can learn from data, understand language, and solve problems. A "
        "useful model should provide clear explanations, handle uncertainty, "
        "and respond to the actual question. "
        "人工智能可以帮助我们分析数据、学习知识和编写程序，但我们仍需要验证它给"
        "出的答案。量化通过降低模型权重的存储精度来减少显存占用。");
    if (corpus.size() < 64)
      throw std::runtime_error("validation corpus too short");
    auto candidate = run(argv[1], corpus, tokenizer);
    if (argc == 4) {
      auto reference = run(argv[3], corpus, tokenizer);
      double squared = 0, signal = 0;
      std::size_t count = 0, match = 0;
      for (std::size_t row = 0; row < candidate.logits.size(); ++row) {
        const auto &a = candidate.logits[row];
        const auto &b = reference.logits[row];
        if (a.size() != b.size())
          throw std::runtime_error("reference vocabulary mismatch");
        match += std::max_element(a.begin(), a.end()) - a.begin() ==
                 std::max_element(b.begin(), b.end()) - b.begin();
        for (std::size_t i = 0; i < a.size(); ++i) {
          const double d = double(a[i]) - b[i];
          squared += d * d;
          signal += double(b[i]) * b[i];
          ++count;
        }
      }
      std::cout << std::scientific << std::setprecision(8)
                << "teacher_forced_positions=" << candidate.logits.size()
                << " logit_mse=" << squared / count << " relative_rmse="
                << std::sqrt(squared / std::max(signal, 1e-30))
                << " top1_matches=" << match << '/' << candidate.logits.size()
                << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
