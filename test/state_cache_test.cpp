#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "rwkv_state_cache.hpp"
#include "test_common.hpp"

namespace {

rwkv7_server::GenerationState make_state(
    int batch_size,
    const std::vector<float>& shift_values,
    const std::vector<float>& wkv_values,
    const std::vector<int>& elapsed_values) {
  rwkv7_server::GenerationState state;
  state.batch_size = batch_size;

  const auto shift_bits = rwkv_test::to_half_bits(shift_values);
  state.shift.resize(shift_bits.size(), "alloc test shift");
  if (!shift_bits.empty()) {
    rwkv_test::require_cuda(
        cudaMemcpy(state.shift.p, shift_bits.data(), shift_bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
        "copy test shift");
  }

  const auto wkv_bits = rwkv_test::to_half_bits(wkv_values);
  state.wkv_state.resize(wkv_bits.size(), "alloc test wkv");
  if (!wkv_bits.empty()) {
    rwkv_test::require_cuda(
        cudaMemcpy(state.wkv_state.p, wkv_bits.data(), wkv_bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
        "copy test wkv");
  }

  rwkv_test::copy_host_to_device(elapsed_values, state.elapsed, "alloc test elapsed", "copy test elapsed");
  return state;
}

std::vector<std::uint16_t> copy_half_bits(const rwkv7_fast_v4::DeviceBuffer<half>& buffer, const char* label) {
  std::vector<std::uint16_t> host(buffer.n);
  if (buffer.n > 0) {
    rwkv_test::require_cuda(
        cudaMemcpy(host.data(), buffer.p, buffer.n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
        label);
  }
  return host;
}

void expect_state_eq(const rwkv7_server::GenerationState& state, const rwkv7_server::GenerationState& expected) {
  TEST_EQ(state.batch_size, expected.batch_size);
  TEST_EQ(copy_half_bits(state.shift, "copy state shift"), copy_half_bits(expected.shift, "copy expected shift"));
  TEST_EQ(copy_half_bits(state.wkv_state, "copy state wkv"), copy_half_bits(expected.wkv_state, "copy expected wkv"));
  TEST_EQ(rwkv_test::copy_device_buffer(state.elapsed, "copy state elapsed"),
          rwkv_test::copy_device_buffer(expected.elapsed, "copy expected elapsed"));
}

}  // namespace

int main() {
  std::filesystem::path db_path = rwkv_test::unique_temp_path("rwkv_state_cache_test");
  try {
    if (!rwkv_test::cuda_device_available()) {
      std::cout << "rwkv_state_cache_test skipped: no CUDA device available\n";
      return 0;
    }

    auto& manager = rwkv7_server::StateCacheManager::instance();
    manager.initialize(1, 1, db_path.string());

    auto state1 = make_state(1, {1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}, {7});
    auto state2 = make_state(1, {6.0f}, {7.0f, 8.0f}, {9});
    auto state3 = make_state(1, {10.0f, 11.0f}, {12.0f}, {13});

    manager.put_state("session-1", state1);
    auto loaded1 = manager.get_state("session-1");
    TEST_CHECK(loaded1.has_value());
    expect_state_eq(*loaded1, state1);

    manager.put_state("session-2", state2);
    auto summary = manager.list_all_states();
    TEST_CHECK(rwkv_test::contains(summary.l1_cache, std::string("session-2")));
    TEST_CHECK(rwkv_test::contains(summary.l2_cache, std::string("session-1")));

    manager.put_state("session-3", state3);
    summary = manager.list_all_states();
    TEST_CHECK(rwkv_test::contains(summary.l1_cache, std::string("session-3")));
    TEST_CHECK(rwkv_test::contains(summary.l2_cache, std::string("session-2")));
    TEST_CHECK(rwkv_test::contains(summary.database, std::string("session-1")));

    auto loaded_from_db = manager.get_state("session-1");
    TEST_CHECK(loaded_from_db.has_value());
    expect_state_eq(*loaded_from_db, state1);
    TEST_CHECK(manager.get_db_timestamp("session-1").has_value());

    TEST_CHECK(manager.delete_state_from_any_level("session-2"));
    TEST_CHECK(!manager.get_state("session-2").has_value());

    manager.shutdown();
    std::filesystem::remove(db_path);
    std::cout << "rwkv_state_cache_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    rwkv7_server::StateCacheManager::instance().shutdown();
    std::filesystem::remove(db_path);
    std::cerr << "rwkv_state_cache_test failed: " << e.what() << "\n";
    return 1;
  }
}
