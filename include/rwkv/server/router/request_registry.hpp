#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rwkv/inference/rwkv_inference_engine.hpp"
#include <json/json.h>

namespace rwkv7_server::router {

class RequestRegistry {
 public:
  struct ActiveRequest {
    std::string id;
    std::string endpoint;
    std::string model;
    std::string state_key;
    Json::Int64 created = 0;
    int prompt_tokens = 0;
    int max_tokens = 0;
    std::atomic<int> generated_tokens{0};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> pause_requested{false};
    std::atomic<int> prefilled_tokens{0};
    std::atomic<double> prefill_progress{0.0};
    std::atomic<double> prefill_speed{0.0};
    std::atomic<double> decode_speed{0.0};
  };

  struct RequestSnapshot {
    std::string id;
    std::string endpoint;
    std::string model;
    std::string state_key;
    Json::Int64 created = 0;
    Json::Int64 finished = 0;
    int prompt_tokens = 0;
    int prefilled_tokens = 0;
    int generated_tokens = 0;
    int max_tokens = 0;
    bool stop_requested = false;
    bool pause_requested = false;
    double prefill_progress = 0.0;
    double prefill_speed = 0.0;
    double decode_speed = 0.0;
  };

  struct PausedRequest {
    std::string id;
    std::string endpoint;
    std::string model;
    Json::Int64 created = 0;
    int generated_tokens = 0;
    int chunk_size = 1;
    GenerateOptions options;
    std::shared_ptr<GenerationState> state;
    std::shared_ptr<DeviceLogits> logits;
  };

  static RequestRegistry& instance() {
    static RequestRegistry registry;
    return registry;
  }

  std::shared_ptr<ActiveRequest> start(
      std::string endpoint,
      std::string model,
      int max_tokens,
      int prompt_tokens = 0,
      std::string state_key = {},
      std::string request_id = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto active = std::make_shared<ActiveRequest>();
    active->id = request_id.empty() ? make_id_locked() : std::move(request_id);
    active->endpoint = std::move(endpoint);
    active->model = std::move(model);
    active->state_key = std::move(state_key);
    active->created = now_seconds();
    active->prompt_tokens = prompt_tokens;
    active->max_tokens = max_tokens;
    active_.push_back(active);
    return active;
  }

  std::shared_ptr<ActiveRequest> active(const std::string& request_id = {}) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_id.empty()) {
      return active_.empty() ? nullptr : active_.back();
    }
    const auto it = std::find_if(active_.rbegin(), active_.rend(), [&](const auto& item) {
      return item->id == request_id;
    });
    return it == active_.rend() ? nullptr : *it;
  }

  bool stop_active(const std::string& request_id = {}) {
    if (!request_id.empty()) {
      const auto target = active(request_id);
      if (!target) {
        return false;
      }
      target->stop_requested.store(true);
      return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : active_) {
      item->stop_requested.store(true);
    }
    return !active_.empty();
  }

  std::optional<std::string> pause_active(const std::string& request_id = {}) {
    const auto active = this->active(request_id);
    if (!active) {
      return std::nullopt;
    }
    active->pause_requested.store(true);
    return active->id;
  }

  void finish(const std::shared_ptr<ActiveRequest>& active) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find(active_.begin(), active_.end(), active);
    if (it != active_.end()) {
      auto snapshot = snapshot_request(**it);
      snapshot.finished = now_seconds();
      last_ = std::move(snapshot);
      active_.erase(it);
    }
  }

  void put_paused(PausedRequest paused) {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_[paused.id] = std::move(paused);
  }

  std::optional<PausedRequest> take_paused(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = paused_.find(id);
    if (it == paused_.end()) {
      return std::nullopt;
    }
    auto paused = std::move(it->second);
    paused_.erase(it);
    return paused;
  }

  Json::Value active_json() const {
    const auto active = this->active();
    if (!active) {
      return Json::Value(Json::nullValue);
    }
    return active_request_json(*active);
  }

  Json::Value active_requests_json() const {
    std::vector<std::shared_ptr<ActiveRequest>> active;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active = active_;
    }
    Json::Value out = Json::arrayValue;
    for (const auto& item : active) {
      out.append(active_request_json(*item));
    }
    return out;
  }

  Json::Value last_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_.has_value()) {
      return Json::Value(Json::nullValue);
    }
    return snapshot_json(*last_);
  }

  Json::Value paused_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Json::Value out = Json::arrayValue;
    for (const auto& [id, paused] : paused_) {
      Json::Value item;
      item["id"] = id;
      item["endpoint"] = paused.endpoint;
      item["model"] = paused.model;
      item["created"] = paused.created;
      item["generated_tokens"] = paused.generated_tokens;
      out.append(item);
    }
    return out;
  }

 private:
  static Json::Int64 now_seconds() {
    return static_cast<Json::Int64>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  }

  std::string make_id_locked() {
    std::ostringstream oss;
    oss << "req-" << now_seconds() << "-" << ++next_id_;
    return oss.str();
  }

  static RequestSnapshot snapshot_request(const ActiveRequest& active) {
    RequestSnapshot snapshot;
    snapshot.id = active.id;
    snapshot.endpoint = active.endpoint;
    snapshot.model = active.model;
    snapshot.state_key = active.state_key;
    snapshot.created = active.created;
    snapshot.prompt_tokens = active.prompt_tokens;
    snapshot.prefilled_tokens = active.prefilled_tokens.load();
    snapshot.generated_tokens = active.generated_tokens.load();
    snapshot.max_tokens = active.max_tokens;
    snapshot.stop_requested = active.stop_requested.load();
    snapshot.pause_requested = active.pause_requested.load();
    snapshot.prefill_progress = active.prefill_progress.load();
    snapshot.prefill_speed = active.prefill_speed.load();
    snapshot.decode_speed = active.decode_speed.load();
    return snapshot;
  }

  static Json::Value snapshot_json(const RequestSnapshot& snapshot) {
    Json::Value out;
    out["id"] = snapshot.id;
    out["endpoint"] = snapshot.endpoint;
    out["model"] = snapshot.model;
    out["created"] = snapshot.created;
    if (snapshot.finished > 0) {
      out["finished"] = snapshot.finished;
    }
    out["prompt_tokens"] = snapshot.prompt_tokens;
    out["prefilled_tokens"] = snapshot.prefilled_tokens;
    out["generated_tokens"] = snapshot.generated_tokens;
    out["max_tokens"] = snapshot.max_tokens;
    out["stop_requested"] = snapshot.stop_requested;
    out["pause_requested"] = snapshot.pause_requested;
    out["prefill_progress"] = snapshot.prefill_progress;
    out["prefill_speed"] = snapshot.prefill_speed;
    out["decode_speed"] = snapshot.decode_speed;
    if (!snapshot.state_key.empty()) {
      out["state_key"] = snapshot.state_key;
    }
    return out;
  }

  static Json::Value active_request_json(const ActiveRequest& active) {
    Json::Value out;
    out["id"] = active.id;
    out["endpoint"] = active.endpoint;
    out["model"] = active.model;
    out["created"] = active.created;
    out["prompt_tokens"] = active.prompt_tokens;
    out["prefilled_tokens"] = active.prefilled_tokens.load();
    out["generated_tokens"] = active.generated_tokens.load();
    out["max_tokens"] = active.max_tokens;
    out["stop_requested"] = active.stop_requested.load();
    out["pause_requested"] = active.pause_requested.load();
    out["prefill_progress"] = active.prefill_progress.load();
    out["prefill_speed"] = active.prefill_speed.load();
    out["decode_speed"] = active.decode_speed.load();
    if (!active.state_key.empty()) {
      out["state_key"] = active.state_key;
    }
    return out;
  }

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<ActiveRequest>> active_;
  std::optional<RequestSnapshot> last_;
  std::unordered_map<std::string, PausedRequest> paused_;
  Json::UInt64 next_id_ = 0;
};

}  // namespace rwkv7_server::router
