#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

#include "rwkv_server_backend.hpp"

namespace rwkv7_server {

class StateCacheManager {
 public:
  struct StateSummary {
    std::vector<std::string> l1_cache;
    std::vector<std::string> l2_cache;
    std::vector<std::string> database;
  };

  static StateCacheManager& instance();

  void initialize(int l1_capacity = 16, int l2_capacity = 32, const std::string& db_path = "rwkv_sessions.db");
  void shutdown();

  void put_state(const std::string& session_id, const GenerationState& state);
  std::optional<GenerationState> get_state(const std::string& session_id);
  bool delete_state_from_any_level(const std::string& session_id);
  StateSummary list_all_states();
  std::optional<double> get_db_timestamp(const std::string& session_id);

 private:
  StateCacheManager() = default;
  ~StateCacheManager() = default;
  StateCacheManager(const StateCacheManager&) = delete;
  StateCacheManager& operator=(const StateCacheManager&) = delete;

  struct HostState {
    int batch_size = 0;
    std::vector<uint16_t> shift;
    std::vector<uint16_t> wkv_state;
    std::vector<int> elapsed;
  };

  struct L1Entry {
    std::shared_ptr<GenerationState> state;
    std::list<std::string>::iterator it;
  };

  struct L2Entry {
    HostState state;
    std::list<std::string>::iterator it;
  };

  void init_db();
  HostState copy_to_host(const GenerationState& state) const;
  GenerationState copy_to_device(const HostState& state) const;
  std::shared_ptr<GenerationState> clone_device_state(const GenerationState& state) const;
  void touch_l1(const std::string& key);
  void touch_l2(const std::string& key);
  void evict_l1_if_needed_locked();
  void evict_l2_if_needed_locked();
  void persist_state_locked(const std::string& session_id, const HostState& state);
  std::optional<HostState> load_state_locked(const std::string& session_id);

  bool initialized_ = false;
  int l1_capacity_ = 16;
  int l2_capacity_ = 32;
  std::string db_path_ = "rwkv_sessions.db";
  sqlite3* db_ = nullptr;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, L1Entry> l1_cache_;
  std::unordered_map<std::string, L2Entry> l2_cache_;
  std::list<std::string> l1_order_;
  std::list<std::string> l2_order_;
};

}  // namespace rwkv7_server
