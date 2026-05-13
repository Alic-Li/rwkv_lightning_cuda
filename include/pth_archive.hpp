#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "status.hpp"

namespace llm_infer {

struct PthEntry {
  std::string name;
  std::uint16_t compression_method = 0;
  std::uint32_t crc32 = 0;
  std::uint64_t compressed_size = 0;
  std::uint64_t uncompressed_size = 0;
  std::uint64_t local_header_offset = 0;
  std::uint64_t data_offset = 0;

  bool is_stored() const {
    return compression_method == 0;
  }
};

struct PthEntryView {
  const std::uint8_t* data = nullptr;
  std::uint64_t size = 0;
};

class PthArchive {
public:
  static Result<PthArchive> open(const std::string& path);

  const std::string& path() const {
    return path_;
  }

  const std::vector<PthEntry>& entries() const {
    return entries_;
  }

  const PthEntry* find_entry(const std::string& name) const;

  Result<std::vector<std::uint8_t>> read_stored_entry(const PthEntry& entry) const;
  Result<PthEntryView> stored_entry_view(const PthEntry& entry) const;

private:
  std::string path_;
  std::vector<std::uint8_t> bytes_;
  std::vector<PthEntry> entries_;
};

}  // namespace llm_infer
