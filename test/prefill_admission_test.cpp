#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include "rwkv_prefill_admission.hpp"
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

    std::cout << "rwkv_prefill_admission_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rwkv_prefill_admission_test failed: " << error.what() << "\n";
    return 1;
  }
}
