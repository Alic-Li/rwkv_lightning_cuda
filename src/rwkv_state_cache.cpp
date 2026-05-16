#include "rwkv_state_cache.hpp"

#include <ctime>
#include <stdexcept>

#include <cuda_runtime.h>

#include "rwkv7_fast_v4_common.hpp"

namespace rwkv7_server {
namespace {

using rwkv7_fast_v4::check_cuda;

std::vector<uint16_t> copy_half_buffer_to_host(const rwkv7_fast_v4::DeviceBuffer<half>& buffer) {
  std::vector<uint16_t> host(buffer.n);
  if (buffer.n > 0) {
    check_cuda(
        cudaMemcpy(host.data(), buffer.p, buffer.n * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy half buffer to host");
  }
  return host;
}

std::vector<float> copy_float_buffer_to_host(const rwkv7_fast_v4::DeviceBuffer<float>& buffer) {
  std::vector<float> host(buffer.n);
  if (buffer.n > 0) {
    check_cuda(
        cudaMemcpy(host.data(), buffer.p, buffer.n * sizeof(float), cudaMemcpyDeviceToHost),
        "copy float buffer to host");
  }
  return host;
}

void copy_half_buffer_to_device(
    const std::vector<uint16_t>& host,
    rwkv7_fast_v4::DeviceBuffer<half>& buffer,
    const char* label) {
  buffer.resize(host.size(), label);
  if (!host.empty()) {
    check_cuda(
        cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
        label);
  }
}

void copy_float_buffer_to_device(
    const std::vector<float>& host,
    rwkv7_fast_v4::DeviceBuffer<float>& buffer,
    const char* label) {
  buffer.resize(host.size(), label);
  if (!host.empty()) {
    check_cuda(
        cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
        label);
  }
}

}  // namespace

StateCacheManager& StateCacheManager::instance() {
  static StateCacheManager manager;
  return manager;
}

void StateCacheManager::initialize(int l1_capacity, int l2_capacity, const std::string& db_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;
  }
  l1_capacity_ = l1_capacity;
  l2_capacity_ = l2_capacity;
  db_path_ = db_path;
  init_db();
  initialized_ = true;
}

void StateCacheManager::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return;
  }

  for (const auto& [session_id, entry] : l1_cache_) {
    persist_state_locked(session_id, copy_to_host(*entry.state));
  }
  for (const auto& [session_id, entry] : l2_cache_) {
    persist_state_locked(session_id, entry.state);
  }

  l1_cache_.clear();
  l2_cache_.clear();
  l1_order_.clear();
  l2_order_.clear();
  if (db_ != nullptr) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
  initialized_ = false;
}

void StateCacheManager::init_db() {
  if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
    throw std::runtime_error("failed to open sqlite db: " + db_path_);
  }
  const char* sql =
      "CREATE TABLE IF NOT EXISTS sessions ("
      "session_id TEXT PRIMARY KEY,"
      "batch_size INTEGER NOT NULL,"
      "shift_blob BLOB NOT NULL,"
      "wkv_blob BLOB NOT NULL,"
      "wkv32 INTEGER NOT NULL DEFAULT 0,"
      "elapsed_blob BLOB NOT NULL,"
      "last_updated REAL NOT NULL"
      ")";
  char* err = nullptr;
  if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    const std::string msg = err ? err : "unknown sqlite error";
    sqlite3_free(err);
    throw std::runtime_error("failed to initialize sqlite schema: " + msg);
  }
  err = nullptr;
  if (sqlite3_exec(db_, "ALTER TABLE sessions ADD COLUMN wkv32 INTEGER NOT NULL DEFAULT 0", nullptr, nullptr, &err) != SQLITE_OK) {
    const std::string msg = err ? err : "";
    sqlite3_free(err);
    if (msg.find("duplicate column name") == std::string::npos) {
      throw std::runtime_error("failed to migrate sqlite schema: " + msg);
    }
  }
}

StateCacheManager::HostState StateCacheManager::copy_to_host(const GenerationState& state) const {
  HostState host;
  host.batch_size = state.batch_size;
  host.wkv32 = state.wkv32;
  host.shift = copy_half_buffer_to_host(state.shift);
  if (state.wkv32) {
    host.wkv_state32 = copy_float_buffer_to_host(state.wkv_state32);
  } else {
    host.wkv_state16 = copy_half_buffer_to_host(state.wkv_state);
  }
  host.elapsed.resize(state.elapsed.n);
  if (state.elapsed.n > 0) {
    check_cuda(
        cudaMemcpy(
            host.elapsed.data(),
            state.elapsed.p,
            state.elapsed.n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy elapsed to host");
  }
  return host;
}

GenerationState StateCacheManager::copy_to_device(const HostState& state) const {
  GenerationState device;
  device.batch_size = state.batch_size;
  device.wkv32 = state.wkv32;
  copy_half_buffer_to_device(state.shift, device.shift, "copy shift to device");
  if (state.wkv32) {
    copy_float_buffer_to_device(state.wkv_state32, device.wkv_state32, "copy wkv32 state to device");
  } else {
    copy_half_buffer_to_device(state.wkv_state16, device.wkv_state, "copy wkv state to device");
  }
  device.elapsed.resize(state.elapsed.size(), "alloc elapsed from host");
  if (!state.elapsed.empty()) {
    check_cuda(
        cudaMemcpy(
            device.elapsed.p,
            state.elapsed.data(),
            state.elapsed.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        "copy elapsed to device");
  }
  return device;
}

std::shared_ptr<GenerationState> StateCacheManager::clone_device_state(const GenerationState& state) const {
  auto clone = std::make_shared<GenerationState>();
  clone->batch_size = state.batch_size;
  clone->wkv32 = state.wkv32;
  clone->shift.resize(state.shift.n, "clone shift");
  if (state.wkv32) {
    clone->wkv_state32.resize(state.wkv_state32.n, "clone wkv32");
  } else {
    clone->wkv_state.resize(state.wkv_state.n, "clone wkv");
  }
  clone->elapsed.resize(state.elapsed.n, "clone elapsed");
  if (state.shift.n > 0) {
    check_cuda(
        cudaMemcpy(clone->shift.p, state.shift.p, state.shift.n * sizeof(half), cudaMemcpyDeviceToDevice),
        "clone shift memcpy");
  }
  if (state.wkv32) {
    if (state.wkv_state32.n > 0) {
      check_cuda(
          cudaMemcpy(
              clone->wkv_state32.p,
              state.wkv_state32.p,
              state.wkv_state32.n * sizeof(float),
              cudaMemcpyDeviceToDevice),
          "clone wkv32 memcpy");
    }
  } else if (state.wkv_state.n > 0) {
    check_cuda(
        cudaMemcpy(
            clone->wkv_state.p,
            state.wkv_state.p,
            state.wkv_state.n * sizeof(half),
            cudaMemcpyDeviceToDevice),
        "clone wkv memcpy");
  }
  if (state.elapsed.n > 0) {
    check_cuda(
        cudaMemcpy(
            clone->elapsed.p,
            state.elapsed.p,
            state.elapsed.n * sizeof(int),
            cudaMemcpyDeviceToDevice),
        "clone elapsed memcpy");
  }
  return clone;
}

void StateCacheManager::touch_l1(const std::string& key) {
  auto it = l1_cache_.find(key);
  if (it == l1_cache_.end()) {
    return;
  }
  l1_order_.erase(it->second.it);
  l1_order_.push_back(key);
  it->second.it = std::prev(l1_order_.end());
}

void StateCacheManager::touch_l2(const std::string& key) {
  auto it = l2_cache_.find(key);
  if (it == l2_cache_.end()) {
    return;
  }
  l2_order_.erase(it->second.it);
  l2_order_.push_back(key);
  it->second.it = std::prev(l2_order_.end());
}

void StateCacheManager::evict_l1_if_needed_locked() {
  while (static_cast<int>(l1_cache_.size()) > l1_capacity_) {
    const std::string victim = l1_order_.front();
    auto it = l1_cache_.find(victim);
    HostState host = copy_to_host(*it->second.state);
    l1_order_.pop_front();
    l1_cache_.erase(it);
    l2_order_.push_back(victim);
    l2_cache_[victim] = {std::move(host), std::prev(l2_order_.end())};
  }
  evict_l2_if_needed_locked();
}

void StateCacheManager::evict_l2_if_needed_locked() {
  while (static_cast<int>(l2_cache_.size()) > l2_capacity_) {
    const std::string victim = l2_order_.front();
    auto it = l2_cache_.find(victim);
    persist_state_locked(victim, it->second.state);
    l2_order_.pop_front();
    l2_cache_.erase(it);
  }
}

void StateCacheManager::persist_state_locked(const std::string& session_id, const HostState& state) {
  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "INSERT OR REPLACE INTO sessions "
      "(session_id, batch_size, shift_blob, wkv_blob, wkv32, elapsed_blob, last_updated) "
      "VALUES (?, ?, ?, ?, ?, ?, ?)";
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, state.batch_size);
  sqlite3_bind_blob(
      stmt, 3, state.shift.data(), static_cast<int>(state.shift.size() * sizeof(uint16_t)), SQLITE_TRANSIENT);
  if (state.wkv32) {
    sqlite3_bind_blob(
        stmt, 4, state.wkv_state32.data(), static_cast<int>(state.wkv_state32.size() * sizeof(float)), SQLITE_TRANSIENT);
  } else {
    sqlite3_bind_blob(
        stmt, 4, state.wkv_state16.data(), static_cast<int>(state.wkv_state16.size() * sizeof(uint16_t)), SQLITE_TRANSIENT);
  }
  sqlite3_bind_int(stmt, 5, state.wkv32 ? 1 : 0);
  sqlite3_bind_blob(
      stmt, 6, state.elapsed.data(), static_cast<int>(state.elapsed.size() * sizeof(int)), SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, 7, static_cast<double>(time(nullptr)));
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

std::optional<StateCacheManager::HostState> StateCacheManager::load_state_locked(const std::string& session_id) {
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(
      db_,
      "SELECT batch_size, shift_blob, wkv_blob, wkv32, elapsed_blob FROM sessions WHERE session_id = ?",
      -1,
      &stmt,
      nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  HostState state;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    state.batch_size = sqlite3_column_int(stmt, 0);
    const auto* shift_blob = static_cast<const uint16_t*>(sqlite3_column_blob(stmt, 1));
    const void* wkv_blob = sqlite3_column_blob(stmt, 2);
    state.wkv32 = sqlite3_column_int(stmt, 3) != 0;
    const auto* elapsed_blob = static_cast<const int*>(sqlite3_column_blob(stmt, 4));
    const int shift_bytes = sqlite3_column_bytes(stmt, 1);
    const int wkv_bytes = sqlite3_column_bytes(stmt, 2);
    const int elapsed_bytes = sqlite3_column_bytes(stmt, 4);
    if (shift_blob != nullptr && shift_bytes > 0) {
      state.shift.assign(shift_blob, shift_blob + shift_bytes / static_cast<int>(sizeof(uint16_t)));
    }
    if (wkv_blob != nullptr && wkv_bytes > 0) {
      if (state.wkv32) {
        const auto* wkv32_blob = static_cast<const float*>(wkv_blob);
        state.wkv_state32.assign(wkv32_blob, wkv32_blob + wkv_bytes / static_cast<int>(sizeof(float)));
      } else {
        const auto* wkv16_blob = static_cast<const uint16_t*>(wkv_blob);
        state.wkv_state16.assign(wkv16_blob, wkv16_blob + wkv_bytes / static_cast<int>(sizeof(uint16_t)));
      }
    }
    if (elapsed_blob != nullptr && elapsed_bytes > 0) {
      state.elapsed.assign(elapsed_blob, elapsed_blob + elapsed_bytes / static_cast<int>(sizeof(int)));
    }
    sqlite3_finalize(stmt);
    return state;
  }
  sqlite3_finalize(stmt);
  return std::nullopt;
}

void StateCacheManager::put_state(const std::string& session_id, const GenerationState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || session_id.empty()) {
    return;
  }

  if (auto it = l1_cache_.find(session_id); it != l1_cache_.end()) {
    l1_order_.erase(it->second.it);
    l1_cache_.erase(it);
  }
  if (auto it = l2_cache_.find(session_id); it != l2_cache_.end()) {
    l2_order_.erase(it->second.it);
    l2_cache_.erase(it);
  }

  l1_order_.push_back(session_id);
  l1_cache_[session_id] = {clone_device_state(state), std::prev(l1_order_.end())};
  evict_l1_if_needed_locked();
}

std::optional<GenerationState> StateCacheManager::get_state(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || session_id.empty()) {
    return std::nullopt;
  }

  if (auto it = l1_cache_.find(session_id); it != l1_cache_.end()) {
    touch_l1(session_id);
    auto cloned = clone_device_state(*it->second.state);
    return GenerationState(std::move(*cloned));
  }

  if (auto it = l2_cache_.find(session_id); it != l2_cache_.end()) {
    touch_l2(session_id);
    auto state = copy_to_device(it->second.state);
    l2_order_.erase(it->second.it);
    l2_cache_.erase(it);
    l1_order_.push_back(session_id);
    l1_cache_[session_id] = {clone_device_state(state), std::prev(l1_order_.end())};
    evict_l1_if_needed_locked();
    return state;
  }

  if (auto state = load_state_locked(session_id); state.has_value()) {
    auto device = copy_to_device(*state);
    l1_order_.push_back(session_id);
    l1_cache_[session_id] = {clone_device_state(device), std::prev(l1_order_.end())};
    evict_l1_if_needed_locked();
    return device;
  }

  return std::nullopt;
}

bool StateCacheManager::delete_state_from_any_level(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool found = false;

  if (auto it = l1_cache_.find(session_id); it != l1_cache_.end()) {
    l1_order_.erase(it->second.it);
    l1_cache_.erase(it);
    found = true;
  }
  if (auto it = l2_cache_.find(session_id); it != l2_cache_.end()) {
    l2_order_.erase(it->second.it);
    l2_cache_.erase(it);
    found = true;
  }

  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(db_, "DELETE FROM sessions WHERE session_id = ?", -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  if (sqlite3_step(stmt) == SQLITE_DONE && sqlite3_changes(db_) > 0) {
    found = true;
  }
  sqlite3_finalize(stmt);
  return found;
}

StateCacheManager::StateSummary StateCacheManager::list_all_states() {
  std::lock_guard<std::mutex> lock(mutex_);
  StateSummary summary;
  summary.l1_cache.reserve(l1_cache_.size());
  summary.l2_cache.reserve(l2_cache_.size());
  for (const auto& [key, _] : l1_cache_) {
    summary.l1_cache.push_back(key);
  }
  for (const auto& [key, _] : l2_cache_) {
    summary.l2_cache.push_back(key);
  }

  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(db_, "SELECT session_id FROM sessions ORDER BY last_updated DESC", -1, &stmt, nullptr);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const auto* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    if (text != nullptr) {
      summary.database.emplace_back(text);
    }
  }
  sqlite3_finalize(stmt);
  return summary;
}

std::optional<double> StateCacheManager::get_db_timestamp(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(
      db_,
      "SELECT last_updated FROM sessions WHERE session_id = ?",
      -1,
      &stmt,
      nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    const double ts = sqlite3_column_double(stmt, 0);
    sqlite3_finalize(stmt);
    return ts;
  }
  sqlite3_finalize(stmt);
  return std::nullopt;
}

}  // namespace rwkv7_server
