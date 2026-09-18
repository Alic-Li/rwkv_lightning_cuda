#include "rwkv/io/pth_writer.hpp"
#include "rwkv/runtime/rwkv_server_backend.hpp"
#include "rwkv/server/rwkv_state_cache.hpp"
#include "test_common.hpp"
#include <future>
#include <iostream>
using namespace rwkv7_server;
using namespace rwkv7_miss;
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: miss_model_test model adapter output.pth\n";
    return 77;
  }
  try {
    ModelBackend model(
        argv[1], true, true, "off",
        (std::filesystem::path(argv[3]).parent_path() / "miss-test.tune")
            .string());
    auto &cache = adapter_cache();
    cache.register_adapter("trained", argv[2]);
    auto h = cache.resolve("trained");
    float zero = 0;
    auto z = cache.resolve("trained", {}, &zero);
    auto run = [&](std::shared_ptr<const AdapterHandle> a, bool chunk) {
      auto state = model.create_state(1);
      bind_adapter(state, a);
      DeviceLogits logits;
      if (chunk) {
        model.forward_prefill({{98, 99}}, state, logits);
        model.forward_prefill({{100, 98, 99, 100}}, state, logits);
      } else
        model.forward_prefill({{98, 99, 100, 98, 99, 100}}, state, logits);
      auto prefill = rwkv_test::copy_device_buffer(logits.values, "prefill");
      for (int t = 0; t < 3; ++t)
        model.forward_decode({98 + t}, state, logits);
      auto out = rwkv_test::copy_device_buffer(logits.values, "decode");
      prefill.insert(prefill.end(), out.begin(), out.end());
      return prefill;
    };
    const auto base = run({}, false), zero_out = run(z, false);
    TEST_EQ(base.size(), zero_out.size());
    for (size_t i = 0; i < base.size(); ++i)
      TEST_EQ(base[i], zero_out[i]);
    const auto expected = run(h, false), chunked = run(h, true);
    for (size_t i = 0; i < expected.size(); ++i)
      TEST_CHECK(std::abs(expected[i] - chunked[i]) < .015f);
    const auto variant_path = rwkv_test::unique_temp_path("miss_variant");
    auto variant_data = h->package->data;
    for (auto &bits : variant_data)
      bits ^= 0x8000;
    save_package(variant_path.string(), h->package->manifest,
                 h->package->tensors, variant_data);
    cache.register_adapter("variant", variant_path.string());
    auto variant = cache.resolve("variant");
    const auto expected_variant = run(variant, false);
    TEST_CHECK(expected_variant != expected);
    const auto uploads = cache.stats().uploads;
    auto one = std::async(std::launch::async, [&] { return run(h, false); });
    auto two =
        std::async(std::launch::async, [&] { return run(variant, false); });
    auto actual = one.get(), other = two.get();
    for (size_t i = 0; i < base.size(); ++i) {
      TEST_EQ(actual[i], expected[i]);
      TEST_EQ(other[i], expected_variant[i]);
    }
    TEST_EQ(cache.stats().uploads, uploads);
    // Release/reload the working set and preserve an immutable handle.
    cache.trim();
    const auto reloaded = run(h, false);
    for (size_t i = 0; i < expected.size(); ++i)
      TEST_EQ(reloaded[i], expected[i]);
    TEST_EQ(cache.stats().uploads, uploads + 1);
    // Persist identity through the SQLite level; pause/reload keeps the adjoint
    // state.
    const auto db = rwkv_test::unique_temp_path("miss_state_db");
    auto &states = StateCacheManager::instance();
    states.initialize(0, 0, db.string());
    auto state = model.create_state(1);
    bind_adapter(state, h);
    DeviceLogits before;
    model.forward_prefill({{98, 99, 100}}, state, before);
    states.put_state("request", state);
    auto restored = states.get_state("request");
    TEST_CHECK(restored.has_value());
    bind_adapter(*restored, h);
    state.adapter_gpu.reset();
    restored->adapter_gpu.reset();
    cache.trim();
    DeviceLogits after;
    model.forward_decode({98}, state, before);
    model.forward_decode({98}, *restored, after);
    TEST_CHECK(rwkv_test::copy_device_buffer(before.values, "original") ==
               rwkv_test::copy_device_buffer(after.values, "restored"));
    bool rejected = false;
    try {
      bind_adapter(*restored, variant);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    TEST_CHECK(rejected);
    states.shutdown();
    std::filesystem::remove(db);
    std::filesystem::remove_all(variant_path);
    llm_infer::WriteTensor t{
        "logits", {2, int(expected.size() / 2)}, false, {}};
    t.data.resize(expected.size() * 4);
    std::memcpy(t.data.data(), expected.data(), t.data.size());
    llm_infer::write_pth(argv[3], {t});
    std::cout << "MiSS model prefill/decode/concurrency/reload passed; uploads="
              << cache.stats().uploads << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
