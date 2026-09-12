#include "dataset.hpp"
#include "rwkv/runtime/rwkv_state_tuning.hpp"

#include "rwkv/io/pth_archive.hpp"
#include "rwkv/io/pth_tensor.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace rwkv7_state_tuning;
  try {
    const WkvShape shape{2, 7, 3, 64};
    const std::size_t expected = static_cast<std::size_t>(2) * 3 * 7 * 64 * 64;
    if (wkv_tape_elements(shape) != expected) {
      throw std::runtime_error("unexpected exact WKV tape size");
    }
    bool rejected = false;
    try {
      (void)wkv_tape_elements({1, 1, 1, 32});
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("invalid head size was accepted");

    const auto checkpoint = std::filesystem::temp_directory_path() /
                            "rwkv_state_tuning_writer_test.pth";
    std::vector<float> runtime(2 * 2 * 2);
    for (std::size_t i = 0; i < runtime.size(); ++i)
      runtime[i] = static_cast<float>(i);
    save_state_checkpoint_pth_host(checkpoint.string(), 1, 2, 2, runtime);
    auto archive = llm_infer::PthArchive::open(checkpoint.string());
    if (!archive.ok())
      throw std::runtime_error(archive.status().message());
    auto records = llm_infer::parse_pth_tensor_records(archive.value());
    if (!records.ok() || records.value().size() != 1)
      throw std::runtime_error("checkpoint record parse failed");
    auto tensor =
        llm_infer::load_tensor_as_float(archive.value(), records.value()[0]);
    if (!tensor.ok() ||
        tensor.value().values != std::vector<float>({0, 2, 1, 3, 4, 6, 5, 7}))
      throw std::runtime_error("checkpoint state transpose mismatch");
    std::filesystem::remove(checkpoint);

    const auto dataset = std::filesystem::temp_directory_path() /
                         "rwkv_state_tuning_dataset_test.jsonl";
    {
      std::ofstream file(dataset);
      file << "{\"text\":\"hello\\nworld\"}\n";
      file << "{\"text\":\"\\u4f60\\u597d\"}\n";
    }
    rwkv7_state_tuning::JsonlTextReader reader(dataset.string());
    std::string text;
    if (!reader.next(text) || text != "hello\nworld")
      throw std::runtime_error("JSONL escaped text mismatch");
    if (!reader.next(text) || text != "你好" || reader.next(text))
      throw std::runtime_error("JSONL unicode text mismatch");
    if (rwkv7_state_tuning::count_jsonl_rows(dataset.string()) != 2)
      throw std::runtime_error("JSONL row count mismatch");
    std::filesystem::remove(dataset);
  } catch (const std::exception &error) {
    std::cerr << "state_tuning_api_test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
