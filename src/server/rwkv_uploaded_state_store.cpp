#include "rwkv/server/rwkv_uploaded_state_store.hpp"

#include <algorithm>
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

int validate_state_pth(const std::filesystem::path& path) {
  auto archive = llm_infer::PthArchive::open(path.string(), true);
  if (!archive.ok()) {
    throw std::runtime_error("invalid state PTH: " + archive.status().message());
  }
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  if (!records.ok()) {
    throw std::runtime_error("invalid state PTH: " + records.status().message());
  }
  int count = 0;
  for (const auto& record : records.value()) {
    if (is_time_state_tensor(record.name)) {
      ++count;
    }
  }
  if (count == 0) {
    throw std::runtime_error("invalid state PTH: no blocks.N.att.time_state tensors found");
  }
  return count;
}

std::int64_t now_seconds() {
  return static_cast<std::int64_t>(
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
}

}  // namespace

struct UploadedStateStore::Entry {
  UploadedStateInfo info;
  std::filesystem::path path;

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
    std::size_t size) {
  if (data == nullptr || size == 0) {
    throw std::runtime_error("uploaded state file is empty");
  }
  if (size > kMaxUploadedStateBytes) {
    throw std::runtime_error("uploaded state file exceeds the 512 MiB limit");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  ensure_directory_locked();
  // Use a basename only: multipart clients may include a local path, but that
  // must neither become part of the public ID nor escape the upload directory.
  std::string state_id = std::filesystem::path(filename).filename().string();
  if (state_id.empty() || state_id == "." || state_id == "..") {
    state_id = "state.pth";
  }
  if (states_.find(state_id) != states_.end()) {
    throw std::runtime_error("uploaded state already exists: " + state_id);
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
    entry->info.state_id = state_id;
    entry->info.filename = state_id;
    entry->info.size_bytes = static_cast<std::uint64_t>(size);
    entry->info.tensor_count = validate_state_pth(path);
    entry->info.created = now_seconds();
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
    return lhs.created == rhs.created ? lhs.state_id < rhs.state_id
                                      : lhs.created < rhs.created;
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
