#include "rwkv_prefill_admission.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace rwkv7_server {

PrefillBatchLimitExceeded::PrefillBatchLimitExceeded(
    int request_batch_size,
    int max_batch_size)
    : std::runtime_error(
          "bsz overflow, Max bsz=" + std::to_string(max_batch_size)),
      request_batch_size_(request_batch_size),
      max_batch_size_(max_batch_size) {}

PrefillAdmissionController::Permit::Permit(
    PrefillAdmissionController* controller,
    int batch_size,
    std::uint64_t ticket)
    : controller_(controller), batch_size_(batch_size), ticket_(ticket) {}

PrefillAdmissionController::Permit::Permit(Permit&& other) noexcept
    : controller_(other.controller_),
      batch_size_(other.batch_size_),
      ticket_(other.ticket_) {
  other.controller_ = nullptr;
}

PrefillAdmissionController::Permit& PrefillAdmissionController::Permit::operator=(
    Permit&& other) noexcept {
  if (this != &other) {
    release();
    controller_ = other.controller_;
    batch_size_ = other.batch_size_;
    ticket_ = other.ticket_;
    other.controller_ = nullptr;
  }
  return *this;
}

PrefillAdmissionController::Permit::~Permit() {
  release();
}

void PrefillAdmissionController::Permit::release() {
  if (controller_ != nullptr) {
    controller_->release(batch_size_, ticket_);
    controller_ = nullptr;
  }
}

PrefillAdmissionController::PrefillAdmissionController(
    CapacityProvider capacity_provider)
    : capacity_provider_(std::move(capacity_provider)) {
  if (!capacity_provider_) {
    throw std::runtime_error("prefill capacity provider must not be empty");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  refresh_capacity_locked();
  hard_max_batch_size_ = dynamic_max_batch_size_;
  hard_limit_initialized_ = true;
  std::printf(
      "[PrefillQueue] initialized max_prefill_bsz=%d bytes_per_bsz=%zu free_vram=%zu\n",
      hard_max_batch_size_,
      capacity_.bytes_per_batch,
      capacity_.free_vram_bytes);
}

std::optional<PrefillAdmissionController::Permit> PrefillAdmissionController::acquire(
    int request_batch_size,
    std::string request_label,
    const CancelCallback& should_cancel) {
  request_batch_size = std::max(1, request_batch_size);
  std::unique_lock<std::mutex> lock(mutex_);
  if (request_batch_size > hard_max_batch_size_) {
    throw PrefillBatchLimitExceeded(request_batch_size, hard_max_batch_size_);
  }

  const std::uint64_t ticket = next_ticket_++;
  queue_.push_back({ticket, request_batch_size, request_label});
  bool queued_logged = false;

  try {
    while (true) {
      if (should_cancel && should_cancel()) {
        remove_ticket_locked(ticket);
        condition_.notify_all();
        return std::nullopt;
      }

      const bool is_turn = !queue_.empty() && queue_.front().ticket == ticket;
      if (is_turn) {
        refresh_capacity_locked();
        const int available = std::max(0, dynamic_max_batch_size_ - reserved_batch_size_);
        if (request_batch_size <= available) {
          reserved_batch_size_ += request_batch_size;
          queue_.pop_front();
          condition_.notify_all();
          std::printf(
              "[PrefillQueue] admitted ticket=%llu path=%s request_bsz=%d reserved_bsz=%d max_prefill_bsz=%d\n",
              static_cast<unsigned long long>(ticket),
              request_label.c_str(),
              request_batch_size,
              reserved_batch_size_,
              dynamic_max_batch_size_);
          return Permit(this, request_batch_size, ticket);
        }
      }

      if (!queued_logged) {
        std::printf(
            "[PrefillQueue] queued ticket=%llu path=%s request_bsz=%d requests_ahead=%zu reserved_bsz=%d max_prefill_bsz=%d\n",
            static_cast<unsigned long long>(ticket),
            request_label.c_str(),
            request_batch_size,
            queue_.empty() ? 0 : queue_.size() - 1,
            reserved_batch_size_,
            dynamic_max_batch_size_);
        queued_logged = true;
      }
      condition_.wait_for(lock, std::chrono::milliseconds(100));
    }
  } catch (...) {
    remove_ticket_locked(ticket);
    condition_.notify_all();
    throw;
  }
}

PrefillAdmissionController::Snapshot PrefillAdmissionController::snapshot(
    bool refresh_capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (refresh_capacity) {
    refresh_capacity_locked();
  }
  Snapshot out;
  out.hard_max_batch_size = hard_max_batch_size_;
  out.dynamic_max_batch_size = dynamic_max_batch_size_;
  out.reserved_batch_size = reserved_batch_size_;
  out.queued_requests = queue_.size();
  out.capacity = capacity_;
  return out;
}

void PrefillAdmissionController::release(int batch_size, std::uint64_t ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  reserved_batch_size_ = std::max(0, reserved_batch_size_ - batch_size);
  refresh_capacity_locked();
  std::printf(
      "[PrefillQueue] released ticket=%llu request_bsz=%d reserved_bsz=%d max_prefill_bsz=%d\n",
      static_cast<unsigned long long>(ticket),
      batch_size,
      reserved_batch_size_,
      dynamic_max_batch_size_);
  condition_.notify_all();
}

void PrefillAdmissionController::refresh_capacity_locked() {
  capacity_ = capacity_provider_();
  dynamic_max_batch_size_ = std::max(0, capacity_.max_batch_size);
  if (hard_limit_initialized_) {
    dynamic_max_batch_size_ = std::min(dynamic_max_batch_size_, hard_max_batch_size_);
  }
}

void PrefillAdmissionController::remove_ticket_locked(std::uint64_t ticket) {
  const auto it = std::find_if(queue_.begin(), queue_.end(), [&](const QueueEntry& entry) {
    return entry.ticket == ticket;
  });
  if (it != queue_.end()) {
    queue_.erase(it);
  }
}

}  // namespace rwkv7_server
