#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include "rwkv/server/rwkv_prefill_admission.hpp"
#include "test_common.hpp"

int main() {
  try {
    std::atomic<int> dynamic_limit{4};
    std::atomic<int> refresh_count{0};
    rwkv7_server::PrefillAdmissionController controller([&]() {
      ++refresh_count;
      rwkv7_server::PrefillCapacity capacity;
      capacity.max_batch_size = dynamic_limit.load();
      capacity.free_vram_bytes = static_cast<std::size_t>(capacity.max_batch_size) * 1024;
      capacity.bytes_per_batch = 1024;
      return capacity;
    });

    auto first = controller.acquire(3, "first");
    TEST_CHECK(first.has_value());
    TEST_EQ(controller.snapshot(false).reserved_batch_size, 3);

    dynamic_limit.store(3);
    auto waiter = std::async(std::launch::async, [&]() {
      return controller.acquire(1, "waiter");
    });
    TEST_CHECK(waiter.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout);

    first.reset();
    auto second = waiter.get();
    TEST_CHECK(second.has_value());
    TEST_EQ(controller.snapshot(false).reserved_batch_size, 1);
    second.reset();

    bool overflow = false;
    try {
      controller.acquire(5, "overflow");
    } catch (const rwkv7_server::PrefillBatchLimitExceeded& error) {
      overflow = true;
      TEST_EQ(error.request_batch_size(), 5);
      TEST_EQ(error.max_batch_size(), 4);
    }
    TEST_CHECK(overflow);

    std::atomic<bool> cancelled{false};
    dynamic_limit.store(0);
    auto cancelled_waiter = std::async(std::launch::async, [&]() {
      return controller.acquire(1, "cancelled", [&]() { return cancelled.load(); });
    });
    TEST_CHECK(
        cancelled_waiter.wait_for(std::chrono::milliseconds(150)) == std::future_status::timeout);
    cancelled.store(true);
    TEST_CHECK(!cancelled_waiter.get().has_value());
    TEST_EQ(controller.snapshot(false).queued_requests, static_cast<std::size_t>(0));
    TEST_CHECK(refresh_count.load() >= 4);

    // Free VRAM shrinking after startup must not strand a request that passed
    // the hard limit: with nothing reserved, waiting cannot help.
    std::atomic<int> shrinking_limit{4};
    rwkv7_server::PrefillAdmissionController stalled(
        [&]() {
          rwkv7_server::PrefillCapacity capacity;
          capacity.max_batch_size = shrinking_limit.load();
          capacity.bytes_per_batch = 1024;
          return capacity;
        },
        std::chrono::milliseconds(100));
    shrinking_limit.store(2);
    const auto stall_start = std::chrono::steady_clock::now();
    bool stalled_rejected = false;
    try {
      stalled.acquire(3, "stalled");
    } catch (const rwkv7_server::PrefillBatchLimitExceeded& error) {
      stalled_rejected = true;
      TEST_EQ(error.request_batch_size(), 3);
      TEST_EQ(error.max_batch_size(), 2);
    }
    TEST_CHECK(stalled_rejected);
    TEST_CHECK(std::chrono::steady_clock::now() - stall_start < std::chrono::seconds(2));
    TEST_EQ(stalled.snapshot(false).queued_requests, static_cast<std::size_t>(0));

    // While another prefill holds a reservation the head keeps waiting, past
    // the idle timeout, and is admitted once that reservation is released.
    auto holder = stalled.acquire(2, "holder");
    TEST_CHECK(holder.has_value());
    auto patient = std::async(std::launch::async, [&]() {
      return stalled.acquire(2, "patient");
    });
    TEST_CHECK(patient.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout);
    holder.reset();
    TEST_CHECK(patient.get().has_value());

    std::cout << "rwkv_prefill_admission_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rwkv_prefill_admission_test failed: " << error.what() << "\n";
    return 1;
  }
}
