#include "pth_archive.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace llm_infer {
namespace {

constexpr std::uint32_t kEndOfCentralDirectory = 0x06054b50u;
constexpr std::uint32_t kZip64EndOfCentralDirectory = 0x06064b50u;
constexpr std::uint32_t kZip64EndOfCentralDirectoryLocator = 0x07064b50u;
constexpr std::uint32_t kCentralDirectoryFileHeader = 0x02014b50u;
constexpr std::uint32_t kLocalFileHeader = 0x04034b50u;

std::uint16_t read_u16(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  return static_cast<std::uint16_t>(bytes[off]) |
         (static_cast<std::uint16_t>(bytes[off + 1]) << 8);
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  return static_cast<std::uint32_t>(bytes[off]) |
         (static_cast<std::uint32_t>(bytes[off + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[off + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[off + 3]) << 24);
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  return static_cast<std::uint64_t>(read_u32(bytes, off)) |
         (static_cast<std::uint64_t>(read_u32(bytes, off + 4)) << 32);
}

std::string number(std::uint64_t value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

Result<std::vector<std::uint8_t>> read_all(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Status::error("failed to open file: " + path);
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size < 0) {
    return Status::error("failed to determine file size: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) {
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
      return Status::error("failed to read file: " + path);
    }
  }
  return bytes;
}

Result<std::size_t> find_eocd(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < 22) {
    return Status::error("file is too small to be a zip archive");
  }

  const std::size_t max_comment = 65535;
  const std::size_t min_pos = bytes.size() > (22 + max_comment) ? bytes.size() - (22 + max_comment) : 0;
  for (std::size_t pos = bytes.size() - 22;; --pos) {
    if (read_u32(bytes, pos) == kEndOfCentralDirectory) {
      return pos;
    }
    if (pos == min_pos) {
      break;
    }
  }
  return Status::error("end of central directory not found");
}

Status read_zip64_eocd(
    const std::vector<std::uint8_t>& bytes,
    std::size_t eocd,
    std::uint64_t* entry_count,
    std::uint64_t* cd_size,
    std::uint64_t* cd_offset) {
  if (eocd < 20 || read_u32(bytes, eocd - 20) != kZip64EndOfCentralDirectoryLocator) {
    return Status::error("zip64 end of central directory locator not found");
  }
  const std::size_t loc = eocd - 20;
  const std::uint32_t locator_disk = read_u32(bytes, loc + 4);
  const std::uint64_t zip64_eocd_offset = read_u64(bytes, loc + 8);
  const std::uint32_t total_disks = read_u32(bytes, loc + 16);
  if (locator_disk != 0 || total_disks != 1) {
    return Status::error("multi-disk zip64 archives are not supported");
  }
  if (zip64_eocd_offset + 56 > bytes.size()) {
    return Status::error("zip64 end of central directory is out of file bounds");
  }
  const std::size_t z = static_cast<std::size_t>(zip64_eocd_offset);
  if (read_u32(bytes, z) != kZip64EndOfCentralDirectory) {
    return Status::error("zip64 end of central directory signature mismatch");
  }
  const std::uint32_t disk_no = read_u32(bytes, z + 16);
  const std::uint32_t cd_disk = read_u32(bytes, z + 20);
  if (disk_no != 0 || cd_disk != 0) {
    return Status::error("multi-disk zip64 archives are not supported");
  }
  *entry_count = read_u64(bytes, z + 32);
  *cd_size = read_u64(bytes, z + 40);
  *cd_offset = read_u64(bytes, z + 48);
  return Status::ok();
}

Status apply_zip64_extra(
    const std::vector<std::uint8_t>& bytes,
    std::size_t extra_pos,
    std::uint16_t extra_len,
    bool need_uncompressed,
    bool need_compressed,
    bool need_local_offset,
    PthEntry* entry) {
  const std::size_t extra_end = extra_pos + extra_len;
  if (extra_end > bytes.size()) {
    return Status::error("central directory extra field is truncated: " + entry->name);
  }
  std::size_t pos = extra_pos;
  while (pos + 4 <= extra_end) {
    const std::uint16_t tag = read_u16(bytes, pos);
    const std::uint16_t size = read_u16(bytes, pos + 2);
    pos += 4;
    if (pos + size > extra_end) {
      return Status::error("central directory extra block is truncated: " + entry->name);
    }
    if (tag == 0x0001u) {
      std::size_t p = pos;
      auto take_u64 = [&]() -> Result<std::uint64_t> {
        if (p + 8 > pos + size) {
          return Status::error("zip64 extended information is truncated: " + entry->name);
        }
        std::uint64_t v = read_u64(bytes, p);
        p += 8;
        return v;
      };
      if (need_uncompressed) {
        auto r = take_u64();
        if (!r.ok()) return r.status();
        entry->uncompressed_size = r.value();
      }
      if (need_compressed) {
        auto r = take_u64();
        if (!r.ok()) return r.status();
        entry->compressed_size = r.value();
      }
      if (need_local_offset) {
        auto r = take_u64();
        if (!r.ok()) return r.status();
        entry->local_header_offset = r.value();
      }
      return Status::ok();
    }
    pos += size;
  }
  if (need_uncompressed || need_compressed || need_local_offset) {
    return Status::error("zip64 extended information missing: " + entry->name);
  }
  return Status::ok();
}

Status compute_data_offsets(const std::vector<std::uint8_t>& bytes, std::vector<PthEntry>* entries) {
  for (PthEntry& entry : *entries) {
    const std::uint64_t local = entry.local_header_offset;
    if (local + 30 > bytes.size()) {
      return Status::error("local header offset is out of range for entry: " + entry.name);
    }
    if (read_u32(bytes, static_cast<std::size_t>(local)) != kLocalFileHeader) {
      return Status::error("local header signature mismatch for entry: " + entry.name);
    }
    const std::uint16_t name_len = read_u16(bytes, static_cast<std::size_t>(local + 26));
    const std::uint16_t extra_len = read_u16(bytes, static_cast<std::size_t>(local + 28));
    entry.data_offset = local + 30 + name_len + extra_len;
    if (entry.data_offset + entry.compressed_size > bytes.size()) {
      return Status::error("entry data range is out of file bounds: " + entry.name);
    }
  }
  return Status::ok();
}

}  // namespace

Result<PthArchive> PthArchive::open(const std::string& path) {
  auto bytes_result = read_all(path);
  if (!bytes_result.ok()) {
    return bytes_result.status();
  }

  PthArchive archive;
  archive.path_ = path;
  archive.bytes_ = std::move(bytes_result.value());

  auto eocd_result = find_eocd(archive.bytes_);
  if (!eocd_result.ok()) {
    return eocd_result.status();
  }
  const std::size_t eocd = eocd_result.value();

  const std::uint16_t disk_no = read_u16(archive.bytes_, eocd + 4);
  const std::uint16_t cd_disk = read_u16(archive.bytes_, eocd + 6);
  std::uint64_t entry_count = read_u16(archive.bytes_, eocd + 10);
  std::uint64_t cd_size = read_u32(archive.bytes_, eocd + 12);
  std::uint64_t cd_offset = read_u32(archive.bytes_, eocd + 16);

  if (disk_no != 0 || cd_disk != 0) {
    return Status::error("multi-disk zip archives are not supported");
  }
  if (entry_count == 0xffffu || cd_size == 0xffffffffu || cd_offset == 0xffffffffu) {
    Status zip64_status = read_zip64_eocd(archive.bytes_, eocd, &entry_count, &cd_size, &cd_offset);
    if (!zip64_status.ok_status()) {
      return zip64_status;
    }
  }
  if (cd_offset + cd_size > archive.bytes_.size()) {
    return Status::error("central directory range is out of file bounds");
  }

  std::size_t pos = static_cast<std::size_t>(cd_offset);
  archive.entries_.reserve(static_cast<std::size_t>(entry_count));
  for (std::uint64_t i = 0; i < entry_count; ++i) {
    if (pos + 46 > archive.bytes_.size()) {
      return Status::error("central directory header is truncated");
    }
    if (read_u32(archive.bytes_, pos) != kCentralDirectoryFileHeader) {
      return Status::error("central directory signature mismatch at entry " + number(i));
    }

    PthEntry entry;
    entry.compression_method = read_u16(archive.bytes_, pos + 10);
    entry.crc32 = read_u32(archive.bytes_, pos + 16);
    const std::uint32_t compressed_size32 = read_u32(archive.bytes_, pos + 20);
    const std::uint32_t uncompressed_size32 = read_u32(archive.bytes_, pos + 24);
    const std::uint16_t name_len = read_u16(archive.bytes_, pos + 28);
    const std::uint16_t extra_len = read_u16(archive.bytes_, pos + 30);
    const std::uint16_t comment_len = read_u16(archive.bytes_, pos + 32);
    const std::uint32_t local_header_offset32 = read_u32(archive.bytes_, pos + 42);
    entry.compressed_size = compressed_size32;
    entry.uncompressed_size = uncompressed_size32;
    entry.local_header_offset = local_header_offset32;

    const std::size_t name_pos = pos + 46;
    if (name_pos + name_len > archive.bytes_.size()) {
      return Status::error("central directory file name is truncated");
    }
    entry.name.assign(
        reinterpret_cast<const char*>(archive.bytes_.data() + name_pos),
        static_cast<std::size_t>(name_len));
    Status zip64_extra_status = apply_zip64_extra(
        archive.bytes_,
        name_pos + name_len,
        extra_len,
        uncompressed_size32 == 0xffffffffu,
        compressed_size32 == 0xffffffffu,
        local_header_offset32 == 0xffffffffu,
        &entry);
    if (!zip64_extra_status.ok_status()) {
      return zip64_extra_status;
    }
    archive.entries_.push_back(std::move(entry));

    pos = name_pos + name_len + extra_len + comment_len;
  }

  Status data_status = compute_data_offsets(archive.bytes_, &archive.entries_);
  if (!data_status.ok_status()) {
    return data_status;
  }

  return archive;
}

const PthEntry* PthArchive::find_entry(const std::string& name) const {
  auto it = std::find_if(entries_.begin(), entries_.end(), [&](const PthEntry& entry) {
    return entry.name == name;
  });
  return it == entries_.end() ? nullptr : &(*it);
}

Result<std::vector<std::uint8_t>> PthArchive::read_stored_entry(const PthEntry& entry) const {
  if (!entry.is_stored()) {
    return Status::error("entry is compressed; only stored entries are supported: " + entry.name);
  }
  if (entry.data_offset + entry.uncompressed_size > bytes_.size()) {
    return Status::error("entry data range is out of file bounds: " + entry.name);
  }
  const auto begin = bytes_.begin() + static_cast<std::ptrdiff_t>(entry.data_offset);
  const auto end = begin + static_cast<std::ptrdiff_t>(entry.uncompressed_size);
  return std::vector<std::uint8_t>(begin, end);
}

Result<PthEntryView> PthArchive::stored_entry_view(const PthEntry& entry) const {
  if (!entry.is_stored()) {
    return Status::error("entry is compressed; only stored entries are supported: " + entry.name);
  }
  if (entry.data_offset + entry.uncompressed_size > bytes_.size()) {
    return Status::error("entry data range is out of file bounds: " + entry.name);
  }
  PthEntryView view;
  view.data = bytes_.data() + static_cast<std::size_t>(entry.data_offset);
  view.size = entry.uncompressed_size;
  return view;
}

}  // namespace llm_infer
