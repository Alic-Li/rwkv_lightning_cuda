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
      llm_infer::QuantizedWriter writer(path.string(), 4);
      const std::vector<std::uint8_t> quantized_data{0, 127, 129, 255};
      const std::vector<std::uint16_t> scales{0x3c00, 0x4000};
      TEST_CHECK(writer.append("blocks.0.att.key.weight", llm_infer::QuantizedDType::kInt8,
                               {2, 2}, quantized_data, scales).ok_status());
      const std::vector<std::uint8_t> bf16{0x00, 0x3f, 0x00, 0x40};
      TEST_CHECK(writer.append("ln_out.bias", llm_infer::QuantizedDType::kBFloat16,
                               {2}, bf16).ok_status());
      const std::vector<std::uint16_t> metadata{0x3c03, 0xbc03, 3, 0, 0, 3, 0, 0};
      for (int bits : {4, 6}) {
        const auto dtype = static_cast<llm_infer::QuantizedDType>(bits);
        const std::vector<std::uint8_t> planes(2 * (16 + 4 * (bits - 4)), 0x55);
        TEST_CHECK(!writer.append("bad.sb6.rank", dtype, {33}, planes, metadata, 32).ok_status());
        TEST_CHECK(!writer.append("bad.sb6.group", dtype, {1, 33}, planes, metadata, 128).ok_status());
        TEST_CHECK(!writer.append("bad.sb6.metadata", dtype, {1, 33}, planes, {0}, 32).ok_status());
        auto bad = metadata;
        bad[0] = 0x7c00;  // Infinite d.
        TEST_CHECK(!writer.append("bad.sb6.inf", dtype, {1, 33}, planes, bad, 32).ok_status());
        bad = metadata; bad[0] = 0x7bff;  // Finite d, but d*qs overflows half.
        TEST_CHECK(!writer.append("bad.sb6.overflow", dtype, {1, 33}, planes, bad, 32).ok_status());
        TEST_CHECK(writer.append("sb6." + std::to_string(bits), dtype, {1, 33}, planes, metadata, 32).ok_status());
      }
      for (int retired : {3, 5})
        TEST_CHECK(!writer.append("retired", static_cast<llm_infer::QuantizedDType>(retired), {1, 1}, {0}, {}, 32).ok_status());
      TEST_CHECK(writer.close().ok_status());

      auto archive = llm_infer::QuantizedArchive::open(path.string());
      TEST_CHECK(archive.ok());
      TEST_EQ(archive.value().records().size(), static_cast<std::size_t>(4));
      const auto* q_record = archive.value().find("blocks.0.att.key.weight");
      TEST_CHECK(q_record != nullptr);
      std::vector<std::uint8_t> quantized_data_read;
      std::vector<std::uint16_t> scales_read;
      TEST_CHECK(archive.value().read_data(*q_record, &quantized_data_read).ok_status());
      TEST_CHECK(archive.value().read_scales(*q_record, &scales_read).ok_status());
      TEST_CHECK(quantized_data_read == quantized_data);
      TEST_CHECK(scales_read == scales);
      for (int bits : {4, 6}) {
        const auto* record = archive.value().find("sb6." + std::to_string(bits));
        TEST_CHECK(record != nullptr);
        TEST_EQ(llm_infer::asym_bits(record->dtype), bits);
        TEST_EQ(record->scale_count, static_cast<std::uint64_t>(8));
        std::vector<std::uint16_t> raw;
        TEST_CHECK(archive.value().read_scales(*record, &raw).ok_status());
        TEST_CHECK(raw == metadata);
        auto expanded = llm_infer::expand_asym_metadata(*record, raw);
        TEST_CHECK(expanded.ok());
        // +/-1.0029296875 * 3 is exactly halfway: ties-to-even rounds DOWN
        // in magnitude to +/-3.0078125, unlike the legacy away-from-zero rule.
        const std::vector<std::uint16_t> expected{0x4204, 0xc204, 0, 0x8000};
        TEST_CHECK(expanded.value() == expected);
      }
    }  // Close file handles before deleting the fixture on Windows.
    for (unsigned char retired : {3, 5}) {
      std::fstream mutate(path, std::ios::in | std::ios::out | std::ios::binary);
      mutate.seekp(20); mutate.put(retired); mutate.close();
      auto invalid = llm_infer::QuantizedArchive::open(path.string());
      TEST_CHECK(!invalid.ok());
      TEST_CHECK(invalid.status().message().find("retired") != std::string::npos);
    }
    std::filesystem::remove(path);
    return 0;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}
