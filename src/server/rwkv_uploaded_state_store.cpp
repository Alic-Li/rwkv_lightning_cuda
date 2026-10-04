#include "rwkv/server/rwkv_uploaded_state_store.hpp"

#include <algorithm>
#include <array>
#include <ctime>
#include <set>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "rwkv/io/pth_archive.hpp"
#include "rwkv/io/pth_tensor.hpp"

namespace rwkv7_server {
namespace {

bool is_time_state_tensor(const std::string& name) {
  constexpr const char* prefix = "blocks.";
  constexpr const char* suffix = ".att.time_state";
  constexpr std::size_t prefix_size = 7;
  constexpr std::size_t suffix_size = 15;
  return name.rfind(prefix, 0) == 0 && name.size() > prefix_size + suffix_size &&
         name.compare(name.size() - suffix_size, suffix_size, suffix) == 0;
}

void validate_state_pth(const std::filesystem::path& path, UploadedStateInfo& info) {
  auto archive = llm_infer::PthArchive::open(path.string(), true);
  if (!archive.ok()) {
    throw std::runtime_error("invalid state PTH: " + archive.status().message());
  }
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  if (!records.ok()) {
    throw std::runtime_error("invalid state PTH: " + records.status().message());
  }
  std::set<int> layers;
  for (const auto& record : records.value()) {
    if (!is_time_state_tensor(record.name)) continue;
    const auto layer_text = record.name.substr(7, record.name.size() - 7 - 15);
    if (layer_text.empty() || layer_text.find_first_not_of("0123456789") != std::string::npos) {
      throw std::runtime_error("invalid state tensor name: " + record.name);
    }
    const int layer = std::stoi(layer_text);
    if (!layers.insert(layer).second) {
      throw std::runtime_error("duplicate state tensor: " + record.name);
    }
    if ((record.dtype != llm_infer::TensorDType::kBFloat16 &&
         record.dtype != llm_infer::TensorDType::kFloat32) ||
        record.shape.size() != 3 || record.shape[0] <= 0 || record.shape[1] <= 0 ||
        record.shape[0] > INT32_MAX || record.shape[1] > INT32_MAX ||
        record.shape[1] != record.shape[2]) {
      throw std::runtime_error("invalid shape or dtype for state tensor: " + record.name);
    }
    if (!info.heads) {
      info.heads = static_cast<int>(record.shape[0]);
      info.head_size = static_cast<int>(record.shape[1]);
    } else if (info.heads != record.shape[0] || info.head_size != record.shape[1]) {
      throw std::runtime_error("inconsistent state tensor shapes: " + record.name);
    }
    // Bound every dimension/stride before the generic tensor reader walks it.
    const auto dtype_bytes = llm_infer::dtype_size_bytes(record.dtype);
    if (record.stride.size() != 3 || record.storage_size > kMaxUploadedStateBytes / dtype_bytes ||
        record.storage_offset >= record.storage_size) {
      throw std::runtime_error("invalid state tensor storage: " + record.name);
    }
    auto last = record.storage_offset;
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto dim = static_cast<std::uint64_t>(record.shape[i]);
      if (record.stride[i] < 0 || elements > kMaxUploadedStateBytes / dtype_bytes / dim) {
        throw std::runtime_error("invalid state tensor stride or size: " + record.name);
      }
      elements *= dim;
      const auto stride = static_cast<std::uint64_t>(record.stride[i]);
      if (dim > 1 && stride > (record.storage_size - 1 - last) / (dim - 1)) {
        throw std::runtime_error("state tensor range exceeds storage: " + record.name);
      }
      last += (dim - 1) * stride;
    }
    // Scan one tensor at a time, without materializing the complete state.
    auto stats = llm_infer::compute_tensor_stats(archive.value(), record);
    if (!stats.ok()) {
      throw std::runtime_error("invalid state tensor storage: " + stats.status().message());
    }
  }
  if (layers.empty()) {
    throw std::runtime_error("invalid state PTH: no blocks.N.att.time_state tensors found");
  }
  if (*layers.begin() != 0 || *layers.rbegin() != static_cast<int>(layers.size()) - 1) {
    throw std::runtime_error("state PTH layers must be contiguous starting at blocks.0");
  }
  info.tensor_count = info.layers = static_cast<int>(layers.size());
}

std::int64_t now_milliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

// RFC 9562: 48 timestamp bits, version 7, variant 10, 74 random bits.
std::string uuid_v7(std::int64_t timestamp) {
  std::array<unsigned char, 16> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw std::runtime_error("failed to generate state upload UUID");
  }
  for (int i = 5; i >= 0; --i) {
    bytes[i] = static_cast<unsigned char>(timestamp & 0xff);
    timestamp >>= 8;
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x70;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
    result << std::setw(2) << static_cast<unsigned int>(bytes[i]);
  }
  return result.str();
}

bool is_uuid_v7(const std::string& uuid) {
  if (uuid.size() != 36 || uuid[14] != '7' ||
      std::string("89ab").find(uuid[19]) == std::string::npos) return false;
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    const bool separator = i == 8 || i == 13 || i == 18 || i == 23;
    if (separator ? uuid[i] != '-' : std::string("0123456789abcdef").find(uuid[i]) == std::string::npos) {
      return false;
    }
  }
  return true;
}

}  // namespace

struct UploadedStateStore::Entry {
  UploadedStateInfo info;
  std::filesystem::path path;
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};

  ~Entry() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};

UploadedStateStore& UploadedStateStore::instance() {
  static UploadedStateStore store;
  return store;
}

UploadedStateStore::UploadedStateStore() : random_(std::random_device{}()) {}

UploadedStateStore::~UploadedStateStore() {
  shutdown();
}

void UploadedStateStore::ensure_directory_locked() {
  if (!directory_.empty()) {
    return;
  }
  std::ostringstream suffix;
  suffix << std::hex << std::setfill('0') << std::setw(16) << random_();
  directory_ = std::filesystem::temp_directory_path() /
               ("rwkv-lightning-uploaded-states-" + suffix.str());
  std::error_code error;
  if (!std::filesystem::create_directories(directory_, error) && error) {
    throw std::runtime_error(
        "failed to create uploaded state directory: " + error.message());
  }
}

UploadedStateInfo UploadedStateStore::upload(
    const std::string& filename,
    const char* data,
    std::size_t size,
    const std::string& upload_uuid) {
  if (data == nullptr || size == 0) {
    throw std::runtime_error("uploaded state file is empty");
  }
  if (size > kMaxUploadedStateBytes) {
    throw std::runtime_error("uploaded state file exceeds the 512 MiB limit");
  }

  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  if (!SHA256(reinterpret_cast<const unsigned char*>(data), size, digest.data())) {
    throw std::runtime_error("failed to hash uploaded state");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_directory_locked();
  // Use a basename only: multipart clients may include a local path, but that
  // must neither become part of the public ID nor escape the upload directory.
  std::string basename = filename;
  std::replace(basename.begin(), basename.end(), '\\', '/');
  basename = std::filesystem::path(basename).filename().string();
  if (basename.empty() || basename == "." || basename == "..") basename = "state.pth";
  // Keep names portable and leave enough room for the UUID on common filesystems.
  if (basename.size() > 180 || std::any_of(basename.begin(), basename.end(), [](unsigned char c) {
        return c < 32 || std::string("<>:\"|?*").find(c) != std::string::npos;
      })) {
    throw std::runtime_error("invalid uploaded state filename");
  }
  const auto created_ms = now_milliseconds();
  const auto uuid = upload_uuid.empty() ? uuid_v7(created_ms) : upload_uuid;
  if (!is_uuid_v7(uuid)) throw std::runtime_error("upload UUID must be a canonical UUID-v7");
  auto stem = std::filesystem::path(basename).stem().string();
  if (stem.empty()) stem = "state";
  const std::string state_id = stem + "-" + uuid;
  if (const auto existing = states_.find(state_id); existing != states_.end()) {
    // A fan-out retry may follow a lost response. Never rewrite the first copy.
    if (!upload_uuid.empty() && existing->second->info.original_filename == basename &&
        existing->second->info.size_bytes == size && existing->second->digest == digest) {
      return existing->second->info;
    }
    throw std::runtime_error("upload UUID conflicts with existing state content: " + state_id);
  }
  const auto path = directory_ / state_id;
  try {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data, static_cast<std::streamsize>(size));
    output.close();
    if (!output) {
      throw std::runtime_error("failed to save uploaded state file");
    }

    auto entry = std::make_shared<Entry>();
    entry->digest = digest;
    entry->info.state_id = state_id;
    entry->info.filename = state_id;
    entry->info.original_filename = basename;
    entry->info.size_bytes = static_cast<std::uint64_t>(size);
    validate_state_pth(path, entry->info);
    entry->info.created_ms = created_ms;
    entry->info.created = created_ms / 1000;
    const auto seconds = static_cast<std::time_t>(entry->info.created);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    std::ostringstream uploaded_at;
    uploaded_at << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
                << std::setfill('0') << std::setw(3) << created_ms % 1000 << 'Z';
    entry->info.uploaded_at = uploaded_at.str();
    entry->path = path;
    states_.emplace(state_id, entry);
    return entry->info;
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
}

std::optional<UploadedStateHandle> UploadedStateStore::acquire(
    const std::string& state_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = states_.find(state_id);
  if (it == states_.end()) {
    return std::nullopt;
  }
  UploadedStateHandle handle;
  handle.state_id = state_id;
  handle.path = it->second->path;
  handle.keepalive = it->second;
  return handle;
}

bool UploadedStateStore::erase(const std::string& state_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return states_.erase(state_id) > 0;
}

std::vector<UploadedStateInfo> UploadedStateStore::list() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<UploadedStateInfo> result;
  result.reserve(states_.size());
  for (const auto& item : states_) {
    result.push_back(item.second->info);
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.created_ms == rhs.created_ms ? lhs.state_id < rhs.state_id
                                            : lhs.created_ms > rhs.created_ms;
  });
  return result;
}

void UploadedStateStore::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  states_.clear();
  if (!directory_.empty()) {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
    directory_.clear();
  }
}

}  // namespace rwkv7_server
