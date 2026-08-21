#include "pth_archive.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

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

Result<std::uint64_t> file_size(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return Status::error("failed to open file: " + path);
  }
  const std::streamoff size = file.tellg();
  if (size < 0) {
    return Status::error("failed to determine file size: " + path);
  }
  return static_cast<std::uint64_t>(size);
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

struct PthArchive::StreamReader {
  explicit StreamReader(const std::string& path) {
    file.rdbuf()->pubsetbuf(nullptr, 0);
    file.open(path, std::ios::binary);
#ifndef _WIN32
    advise_fd = ::open(path.c_str(), O_RDONLY);
    if (advise_fd >= 0) {
      posix_fadvise(advise_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
#endif
  }

  ~StreamReader() {
#ifndef _WIN32
    if (advise_fd >= 0) ::close(advise_fd);
#endif
  }

  Status read(std::uint64_t offset, void* destination, std::size_t size) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
      return Status::error("file range is too large for this platform");
    }
    std::lock_guard<std::mutex> lock(mutex);
    file.clear();
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file) {
      return Status::error("failed to seek model file");
    }
    if (size != 0) {
      file.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(size));
      if (!file) {
        return Status::error("failed to read model file range");
      }
    }
#ifndef _WIN32
    if (advise_fd >= 0 && size != 0) {
      posix_fadvise(
          advise_fd,
          static_cast<off_t>(offset),
          static_cast<off_t>(size),
          POSIX_FADV_DONTNEED);
    }
#endif
    return Status::ok();
  }

  std::ifstream file;
  std::mutex mutex;
#ifndef _WIN32
  int advise_fd = -1;
#endif
};

Result<PthArchive> PthArchive::open(const std::string& path, bool chunk_load) {
  if (chunk_load) {
    auto size_result = file_size(path);
    if (!size_result.ok()) {
      return size_result.status();
    }
    const std::uint64_t total_size = size_result.value();
    if (total_size < 22) {
      return Status::error("file is too small to be a zip archive");
    }

    auto stream_reader = std::make_shared<StreamReader>(path);
    if (!stream_reader->file) {
      return Status::error("failed to open file: " + path);
    }
    auto read_stream_range = [&](std::uint64_t offset, std::uint64_t size)
        -> Result<std::vector<std::uint8_t>> {
      if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error("file range is too large for this platform: " + path);
      }
      std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
      Status status = stream_reader->read(offset, bytes.data(), bytes.size());
      if (!status.ok_status()) {
        return Status::error(status.message() + ": " + path);
      }
      return bytes;
    };

    constexpr std::uint64_t kMaxEocdSearch = 20 + 22 + 65535;
    const std::uint64_t tail_size = std::min(total_size, kMaxEocdSearch);
    const std::uint64_t tail_offset = total_size - tail_size;
    auto tail_result = read_stream_range(tail_offset, tail_size);
    if (!tail_result.ok()) {
      return tail_result.status();
    }
    const auto& tail = tail_result.value();
    auto eocd_result = find_eocd(tail);
    if (!eocd_result.ok()) {
      return eocd_result.status();
    }
    const std::size_t eocd = eocd_result.value();

    const std::uint16_t disk_no = read_u16(tail, eocd + 4);
    const std::uint16_t cd_disk = read_u16(tail, eocd + 6);
    std::uint64_t entry_count = read_u16(tail, eocd + 10);
    std::uint64_t cd_size = read_u32(tail, eocd + 12);
    std::uint64_t cd_offset = read_u32(tail, eocd + 16);
    if (disk_no != 0 || cd_disk != 0) {
      return Status::error("multi-disk zip archives are not supported");
    }
    if (entry_count == 0xffffu || cd_size == 0xffffffffu || cd_offset == 0xffffffffu) {
      if (eocd < 20 || read_u32(tail, eocd - 20) != kZip64EndOfCentralDirectoryLocator) {
        return Status::error("zip64 end of central directory locator not found");
      }
      const std::size_t locator = eocd - 20;
      if (read_u32(tail, locator + 4) != 0 || read_u32(tail, locator + 16) != 1) {
        return Status::error("multi-disk zip64 archives are not supported");
      }
      const std::uint64_t zip64_offset = read_u64(tail, locator + 8);
      auto zip64_result = read_stream_range(zip64_offset, 56);
      if (!zip64_result.ok()) {
        return zip64_result.status();
      }
      const auto& zip64 = zip64_result.value();
      if (read_u32(zip64, 0) != kZip64EndOfCentralDirectory) {
        return Status::error("zip64 end of central directory signature mismatch");
      }
      if (read_u32(zip64, 16) != 0 || read_u32(zip64, 20) != 0) {
        return Status::error("multi-disk zip64 archives are not supported");
      }
      entry_count = read_u64(zip64, 32);
      cd_size = read_u64(zip64, 40);
      cd_offset = read_u64(zip64, 48);
    }
    if (cd_offset > total_size || cd_size > total_size - cd_offset) {
      return Status::error("central directory range is out of file bounds");
    }

    auto cd_result = read_stream_range(cd_offset, cd_size);
    if (!cd_result.ok()) {
      return cd_result.status();
    }
    const auto& cd = cd_result.value();
    PthArchive archive;
    archive.path_ = path;
    archive.file_size_ = total_size;
    archive.chunk_load_ = true;
    archive.entries_.reserve(static_cast<std::size_t>(entry_count));
    std::size_t pos = 0;
    for (std::uint64_t i = 0; i < entry_count; ++i) {
      if (pos + 46 > cd.size()) {
        return Status::error("central directory header is truncated");
      }
      if (read_u32(cd, pos) != kCentralDirectoryFileHeader) {
        return Status::error("central directory signature mismatch at entry " + number(i));
      }
      PthEntry entry;
      entry.compression_method = read_u16(cd, pos + 10);
      entry.crc32 = read_u32(cd, pos + 16);
      const std::uint32_t compressed_size32 = read_u32(cd, pos + 20);
      const std::uint32_t uncompressed_size32 = read_u32(cd, pos + 24);
      const std::uint16_t name_len = read_u16(cd, pos + 28);
      const std::uint16_t extra_len = read_u16(cd, pos + 30);
      const std::uint16_t comment_len = read_u16(cd, pos + 32);
      const std::uint32_t local_header_offset32 = read_u32(cd, pos + 42);
      entry.compressed_size = compressed_size32;
      entry.uncompressed_size = uncompressed_size32;
      entry.local_header_offset = local_header_offset32;
      const std::size_t name_pos = pos + 46;
      if (name_pos + name_len > cd.size()) {
        return Status::error("central directory file name is truncated");
      }
      entry.name.assign(
          reinterpret_cast<const char*>(cd.data() + name_pos),
          static_cast<std::size_t>(name_len));
      Status zip64_extra_status = apply_zip64_extra(
          cd,
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

    for (PthEntry& entry : archive.entries_) {
      auto header_result = read_stream_range(entry.local_header_offset, 30);
      if (!header_result.ok()) {
        return header_result.status();
      }
      const auto& header = header_result.value();
      if (read_u32(header, 0) != kLocalFileHeader) {
        return Status::error("local header signature mismatch for entry: " + entry.name);
      }
      const std::uint16_t name_len = read_u16(header, 26);
      const std::uint16_t extra_len = read_u16(header, 28);
      entry.data_offset = entry.local_header_offset + 30 + name_len + extra_len;
      if (entry.data_offset > total_size || entry.compressed_size > total_size - entry.data_offset) {
        return Status::error("entry data range is out of file bounds: " + entry.name);
      }
    }
    archive.stream_reader_ = std::move(stream_reader);
    return archive;
  }

  auto bytes_result = read_all(path);
  if (!bytes_result.ok()) {
    return bytes_result.status();
  }

  PthArchive archive;
  archive.path_ = path;
  archive.bytes_ = std::move(bytes_result.value());
  archive.file_size_ = archive.bytes_.size();

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
  if (entry.data_offset > file_size_ || entry.uncompressed_size > file_size_ - entry.data_offset) {
    return Status::error("entry data range is out of file bounds: " + entry.name);
  }
  if (chunk_load_) {
    if (entry.uncompressed_size >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return Status::error("entry is too large for this platform: " + entry.name);
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(entry.uncompressed_size));
    Status status = read_stored_entry_range(entry, 0, bytes.data(), bytes.size());
    if (!status.ok_status()) {
      return status;
    }
    return bytes;
  }
  const auto begin = bytes_.begin() + static_cast<std::ptrdiff_t>(entry.data_offset);
  const auto end = begin + static_cast<std::ptrdiff_t>(entry.uncompressed_size);
  return std::vector<std::uint8_t>(begin, end);
}

Status PthArchive::read_stored_entry_range(
    const PthEntry& entry,
    std::uint64_t entry_offset,
    void* destination,
    std::size_t size) const {
  if (!entry.is_stored()) {
    return Status::error("entry is compressed; only stored entries are supported: " + entry.name);
  }
  const std::uint64_t read_size = static_cast<std::uint64_t>(size);
  if (entry_offset > entry.uncompressed_size ||
      read_size > entry.uncompressed_size - entry_offset) {
    return Status::error("requested range is out of entry bounds: " + entry.name);
  }
  if (size != 0 && destination == nullptr) {
    return Status::error("destination is null for entry range: " + entry.name);
  }
  if (entry.data_offset > file_size_ ||
      entry_offset > file_size_ - entry.data_offset ||
      read_size > file_size_ - entry.data_offset - entry_offset) {
    return Status::error("entry data range is out of file bounds: " + entry.name);
  }
  if (chunk_load_) {
    if (!stream_reader_) {
      return Status::error("chunk-load stream is not open: " + path_);
    }
    Status status = stream_reader_->read(entry.data_offset + entry_offset, destination, size);
    if (!status.ok_status()) {
      return Status::error(status.message() + ": " + entry.name);
    }
    return Status::ok();
  }
  if (size != 0) {
    std::memcpy(
        destination,
        bytes_.data() + static_cast<std::size_t>(entry.data_offset + entry_offset),
        size);
  }
  return Status::ok();
}

Result<PthEntryView> PthArchive::stored_entry_view(const PthEntry& entry) const {
  if (!entry.is_stored()) {
    return Status::error("entry is compressed; only stored entries are supported: " + entry.name);
  }
  if (chunk_load_) {
    return Status::error("entry views are unavailable in chunk-load mode: " + entry.name);
  }
  if (entry.data_offset > file_size_ || entry.uncompressed_size > file_size_ - entry.data_offset) {
    return Status::error("entry data range is out of file bounds: " + entry.name);
  }
  PthEntryView view;
  view.data = bytes_.data() + static_cast<std::size_t>(entry.data_offset);
  view.size = entry.uncompressed_size;
  return view;
}

}  // namespace llm_infer
