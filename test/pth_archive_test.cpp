#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "pth_archive.hpp"
#include "test_common.hpp"

namespace {

void append_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  append_u16(out, static_cast<std::uint16_t>(value));
  append_u16(out, static_cast<std::uint16_t>(value >> 16));
}

struct TestEntry {
  std::string name;
  std::vector<std::uint8_t> data;
  std::uint32_t local_offset = 0;
};

std::vector<std::uint8_t> make_stored_zip(std::vector<TestEntry>* entries) {
  std::vector<std::uint8_t> out;
  for (auto& entry : *entries) {
    entry.local_offset = static_cast<std::uint32_t>(out.size());
    append_u32(out, 0x04034b50u);
    append_u16(out, 20);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u32(out, 0);
    append_u32(out, static_cast<std::uint32_t>(entry.data.size()));
    append_u32(out, static_cast<std::uint32_t>(entry.data.size()));
    append_u16(out, static_cast<std::uint16_t>(entry.name.size()));
    append_u16(out, 0);
    out.insert(out.end(), entry.name.begin(), entry.name.end());
    out.insert(out.end(), entry.data.begin(), entry.data.end());
  }

  const std::uint32_t central_offset = static_cast<std::uint32_t>(out.size());
  for (const auto& entry : *entries) {
    append_u32(out, 0x02014b50u);
    append_u16(out, 20);
    append_u16(out, 20);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u32(out, 0);
    append_u32(out, static_cast<std::uint32_t>(entry.data.size()));
    append_u32(out, static_cast<std::uint32_t>(entry.data.size()));
    append_u16(out, static_cast<std::uint16_t>(entry.name.size()));
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u16(out, 0);
    append_u32(out, 0);
    append_u32(out, entry.local_offset);
    out.insert(out.end(), entry.name.begin(), entry.name.end());
  }
  const std::uint32_t central_size = static_cast<std::uint32_t>(out.size()) - central_offset;
  append_u32(out, 0x06054b50u);
  append_u16(out, 0);
  append_u16(out, 0);
  append_u16(out, static_cast<std::uint16_t>(entries->size()));
  append_u16(out, static_cast<std::uint16_t>(entries->size()));
  append_u32(out, central_size);
  append_u32(out, central_offset);
  append_u16(out, 0);
  return out;
}

}  // namespace

int main() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("rwkv_pth_archive_test_" + std::to_string(stamp) + ".zip");
  try {
    std::vector<TestEntry> expected{
        {"model/data.pkl", {1, 2, 3, 4}, 0},
        {"model/data/0", {9, 8, 7, 6, 5}, 0},
    };
    const auto zip = make_stored_zip(&expected);
    {
      std::ofstream file(path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(zip.data()),
                 static_cast<std::streamsize>(zip.size()));
      TEST_CHECK(static_cast<bool>(file));
    }

    auto whole = llm_infer::PthArchive::open(path.string());
    auto chunked = llm_infer::PthArchive::open(path.string(), true);
    TEST_CHECK(whole.ok());
    TEST_CHECK(chunked.ok());
    TEST_CHECK(!whole.value().chunk_load());
    TEST_CHECK(chunked.value().chunk_load());
    TEST_EQ(whole.value().entries().size(), expected.size());
    TEST_EQ(chunked.value().entries().size(), expected.size());

    for (const auto& item : expected) {
      const auto* whole_entry = whole.value().find_entry(item.name);
      const auto* chunked_entry = chunked.value().find_entry(item.name);
      TEST_CHECK(whole_entry != nullptr);
      TEST_CHECK(chunked_entry != nullptr);
      auto whole_data = whole.value().read_stored_entry(*whole_entry);
      auto chunked_data = chunked.value().read_stored_entry(*chunked_entry);
      TEST_CHECK(whole_data.ok());
      TEST_CHECK(chunked_data.ok());
      TEST_CHECK(whole_data.value() == item.data);
      TEST_CHECK(chunked_data.value() == item.data);
      TEST_EQ(whole_entry->data_offset, chunked_entry->data_offset);
      std::vector<std::uint8_t> whole_range(2);
      std::vector<std::uint8_t> chunked_range(2);
      TEST_CHECK(whole.value().read_stored_entry_range(
          *whole_entry, 1, whole_range.data(), whole_range.size()).ok_status());
      TEST_CHECK(chunked.value().read_stored_entry_range(
          *chunked_entry, 1, chunked_range.data(), chunked_range.size()).ok_status());
      TEST_CHECK(whole_range == std::vector<std::uint8_t>(item.data.begin() + 1, item.data.begin() + 3));
      TEST_CHECK(chunked_range == whole_range);
    }
    const auto* entry = chunked.value().find_entry(expected.front().name);
    TEST_CHECK(entry != nullptr);
    TEST_CHECK(!chunked.value().stored_entry_view(*entry).ok());

    std::filesystem::remove(path);
    return 0;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}
