#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include "rwkv_server_backend.hpp"

namespace rwkv7_server {

class PrefillBatchLimitExceeded : public std::runtime_error {
 public:
  PrefillBatchLimitExceeded(int request_batch_size, int max_batch_size);

  int request_batch_size() const { return request_batch_size_; }
  int max_batch_size() const { return max_batch_size_; }

 private:
  int request_batch_size_ = 0;
  int max_batch_size_ = 0;
};

class PrefillAdmissionController {
 public:
  using CapacityProvider = std::function<PrefillCapacity()>;
  using CancelCallback = std::function<bool()>;

  struct Snapshot {
    int hard_max_batch_size = 0;
    int dynamic_max_batch_size = 0;
    int reserved_batch_size = 0;
    std::size_t queued_requests = 0;
    PrefillCapacity capacity;
  };

  class Permit {
   public:
    Permit() = default;
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    Permit(Permit&& other) noexcept;
    Permit& operator=(Permit&& other) noexcept;
    ~Permit();

    explicit operator bool() const { return controller_ != nullptr; }
    int batch_size() const { return batch_size_; }
    std::uint64_t ticket() const { return ticket_; }
    void release();

   private:
    friend class PrefillAdmissionController;
    Permit(PrefillAdmissionController* controller, int batch_size, std::uint64_t ticket);

    PrefillAdmissionController* controller_ = nullptr;
    int batch_size_ = 0;
    std::uint64_t ticket_ = 0;
  };

  explicit PrefillAdmissionController(CapacityProvider capacity_provider);

  std::optional<Permit> acquire(
      int request_batch_size,
      std::string request_label,
      const CancelCallback& should_cancel = {});
  Snapshot snapshot(bool refresh_capacity = true);

 private:
  struct QueueEntry {
    std::uint64_t ticket = 0;
    int batch_size = 0;
    std::string label;
  };

  void release(int batch_size, std::uint64_t ticket);
  void refresh_capacity_locked();
  void remove_ticket_locked(std::uint64_t ticket);

  CapacityProvider capacity_provider_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<QueueEntry> queue_;
  PrefillCapacity capacity_;
  int hard_max_batch_size_ = 0;
  bool hard_limit_initialized_ = false;
  int dynamic_max_batch_size_ = 0;
  int reserved_batch_size_ = 0;
  std::uint64_t next_ticket_ = 0;
};

}  // namespace rwkv7_server
