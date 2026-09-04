#include <chrono>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "rwkv_quantized.hpp"
#include "test_common.hpp"

int main() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("rwkv_quantized_archive_test_" + std::to_string(stamp) + ".rwkvq");
  try {
    llm_infer::QuantizedWriter writer(path.string(), 2);
    const std::vector<std::uint8_t> quantized_data{0, 127, 129, 255};
    const std::vector<std::uint16_t> scales{0x3c00, 0x4000};
    TEST_CHECK(writer.append("blocks.0.att.key.weight", llm_infer::QuantizedDType::kInt8,
                             {2, 2}, quantized_data, scales).ok_status());
    const std::vector<std::uint8_t> bf16{0x00, 0x3f, 0x00, 0x40};
    TEST_CHECK(writer.append("ln_out.bias", llm_infer::QuantizedDType::kBFloat16,
                             {2}, bf16).ok_status());
    TEST_CHECK(writer.close().ok_status());

    auto archive = llm_infer::QuantizedArchive::open(path.string());
    TEST_CHECK(archive.ok());
    TEST_EQ(archive.value().records().size(), static_cast<std::size_t>(2));
    const auto* q_record = archive.value().find("blocks.0.att.key.weight");
    TEST_CHECK(q_record != nullptr);
    std::vector<std::uint8_t> quantized_data_read;
    std::vector<std::uint16_t> scales_read;
    TEST_CHECK(archive.value().read_data(*q_record, &quantized_data_read).ok_status());
    TEST_CHECK(archive.value().read_scales(*q_record, &scales_read).ok_status());
    TEST_CHECK(quantized_data_read == quantized_data);
    TEST_CHECK(scales_read == scales);
    std::filesystem::remove(path);
    return 0;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}
