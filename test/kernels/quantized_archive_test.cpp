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
    {
      llm_infer::QuantizedWriter writer(path.string(), 3);
      TEST_CHECK(!writer.append("bad.group", llm_infer::QuantizedDType::kInt4,
                                {1, 3}, {0, 0}, {0x3c00}, 64).ok_status());
      TEST_CHECK(!writer.append("bad.row.padding", llm_infer::QuantizedDType::kInt4,
                                {3, 3}, {0, 0, 0, 0, 0}, {0x3c00, 0x3c00, 0x3c00}, 32).ok_status());
      TEST_CHECK(!writer.append("bad.scale.count", llm_infer::QuantizedDType::kInt4,
                                {2, 129}, std::vector<std::uint8_t>(130),
                                {0x3c00, 0x3c00, 0x3c00}, 128).ok_status());
      const std::vector<std::uint8_t> quantized_data{0, 127, 129, 255};
      const std::vector<std::uint16_t> scales{0x3c00, 0x4000};
      TEST_CHECK(writer.append("blocks.0.att.key.weight", llm_infer::QuantizedDType::kInt8,
                               {2, 2}, quantized_data, scales).ok_status());
      const std::vector<std::uint8_t> bf16{0x00, 0x3f, 0x00, 0x40};
      TEST_CHECK(writer.append("ln_out.bias", llm_infer::QuantizedDType::kBFloat16,
                               {2}, bf16).ok_status());
      const std::vector<std::uint8_t> int4_data{0x71, 0x03, 0xaf, 0x06, 0xd8, 0x02};
      const std::vector<std::uint16_t> int4_scales{0x3c00, 0x3800, 0x3400};
      TEST_CHECK(writer.append("blocks.0.ffn.value.weight", llm_infer::QuantizedDType::kInt4,
                               {3, 3}, int4_data, int4_scales, 32).ok_status());
      TEST_CHECK(writer.close().ok_status());

      auto archive = llm_infer::QuantizedArchive::open(path.string());
      TEST_CHECK(archive.ok());
      TEST_EQ(archive.value().records().size(), static_cast<std::size_t>(3));
      const auto* q_record = archive.value().find("blocks.0.att.key.weight");
      TEST_CHECK(q_record != nullptr);
      std::vector<std::uint8_t> quantized_data_read;
      std::vector<std::uint16_t> scales_read;
      TEST_CHECK(archive.value().read_data(*q_record, &quantized_data_read).ok_status());
      TEST_CHECK(archive.value().read_scales(*q_record, &scales_read).ok_status());
      TEST_CHECK(quantized_data_read == quantized_data);
      TEST_CHECK(scales_read == scales);
      const auto* int4_record = archive.value().find("blocks.0.ffn.value.weight");
      TEST_CHECK(int4_record != nullptr);
      TEST_CHECK(int4_record->dtype == llm_infer::QuantizedDType::kInt4);
      TEST_EQ(int4_record->quant_group_size, static_cast<std::uint16_t>(32));
      TEST_EQ(int4_record->numel, static_cast<std::uint64_t>(9));
      std::vector<std::uint8_t> int4_data_read;
      std::vector<std::uint16_t> int4_scales_read;
      TEST_CHECK(archive.value().read_data(*int4_record, &int4_data_read).ok_status());
      TEST_CHECK(archive.value().read_scales(*int4_record, &int4_scales_read).ok_status());
      TEST_CHECK(int4_data_read == int4_data);
      TEST_CHECK(int4_scales_read == int4_scales);
    }  // Close file handles before deleting the fixture on Windows.
    std::filesystem::remove(path);
    return 0;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}
