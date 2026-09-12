#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace rwkv7_server {

constexpr std::size_t kMaxUploadedStateBytes = 512ull * 1024ull * 1024ull;

struct UploadedStateInfo {
  std::string state_id;
  std::string filename;
  std::uint64_t size_bytes = 0;
  int tensor_count = 0;
  std::int64_t created = 0;
};

struct UploadedStateHandle {
  std::string state_id;
  std::filesystem::path path;
  std::shared_ptr<const void> keepalive;
};

class UploadedStateStore {
 public:
  static UploadedStateStore& instance();

  UploadedStateInfo upload(
      const std::string& filename,
      const char* data,
      std::size_t size);
  std::optional<UploadedStateHandle> acquire(const std::string& state_id) const;
  bool erase(const std::string& state_id);
  std::vector<UploadedStateInfo> list() const;
  void shutdown();

 private:
  struct Entry;

  UploadedStateStore();
  ~UploadedStateStore();
  UploadedStateStore(const UploadedStateStore&) = delete;
  UploadedStateStore& operator=(const UploadedStateStore&) = delete;

  void ensure_directory_locked();

  mutable std::mutex mutex_;
  std::filesystem::path directory_;
  std::unordered_map<std::string, std::shared_ptr<Entry>> states_;
  std::mt19937_64 random_;
};

}  // namespace rwkv7_server
