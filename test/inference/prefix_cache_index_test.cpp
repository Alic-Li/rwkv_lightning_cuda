#include "rwkv/inference/prefix_cache.hpp"
#include "rwkv/inference/prefix_cache_pool.hpp"
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>
using rwkv7_server::PrefixCache;
void check(bool ok) {
  if (!ok)
    throw std::runtime_error("radix prefix cache mismatch");
}
void pool_test() {
  using rwkv7_server::PrefixCachePool;
  using Value = std::array<int, 4>;
  size_t allocations = 0;
  auto make = [&] {
    ++allocations;
    return std::make_shared<Value>();
  };
  // The byte limit wins over the entry limit; all namespaces share three slots.
  PrefixCachePool<Value> pool(20, 3 * sizeof(Value) + 1, sizeof(Value), make);
  auto put = [&](const std::string &ns, std::vector<int64_t> ids, int value) {
    return pool.put(ns, ids, [&](Value &v) { v.fill(value); });
  };
  check(pool.capacity() == 3 && pool.allocated_bytes() == 0);
  check(put("ref", {1, 2}, 12));
  check(put("v1", {1, 2}, 112));
  check(put("v1", {3}, 3));
  check((*pool.find("ref", {1, 2, 3}).value)[0] == 12); // touch global LRU
  check(put("v2", {4}, 4)); // evicts v1/12, not ref/12
  check(!pool.find("v1", {1, 2}).value);
  check(pool.size("ref") == 1 && pool.size("v1") == 1 && pool.size("v2") == 1);
  check(pool.evictions() == 1 && allocations == 3);
  check(put("v2", {4}, 44)); // exact replacement reuses its existing slot
  check(allocations == 3 && pool.evictions() == 1);
  check((*pool.find("v2", {4}).value)[0] == 44);
  pool.erase_identity("ref");
  check(!pool.find("ref", {1, 2}).value && pool.retired() == 1);
  check(pool.size() == 2 && pool.used_bytes() == 2 * sizeof(Value));
  check(pool.allocated_bytes() == 3 * sizeof(Value)); // free slot stays in pool
  check(put("v3", {5}, 5));
  check(allocations == 3 && pool.evictions() == 1);
  // Repeated publications and LRU churn must never allocate new state buffers.
  for (int i = 0; i < 5000; ++i) {
    auto ns = "version-" + std::to_string(i);
    for (int j = 0; j < 5; ++j)
      check(put(ns, {1, 2, j, i}, i));
    check(pool.size() == 3 && pool.allocated_slots() == 3);
    check(pool.allocated_bytes() <= pool.memory_limit());
    pool.erase_identity(ns);
    check(pool.size() == 0);
  }
  check(allocations == 3);
  // A live Match must keep its old value, even with a single-slot pool.
  PrefixCachePool<Value> pinned(1, sizeof(Value), sizeof(Value), make);
  pinned.put("v", {1}, [](Value &v) { v.fill(1); });
  auto reader = pinned.find("v", {1});
  check(!pinned.put("v", {1}, [](Value &v) { v.fill(2); }));
  check((*reader.value)[0] == 1 && pinned.skipped() == 1);
  check(pinned.allocated_slots() == 1);
  reader.value.reset();
  check(pinned.put("v", {2}, [](Value &v) { v.fill(2); }));
  check(pinned.allocated_slots() == 1 && (*pinned.find("v", {2}).value)[0] == 2);
  reader = pinned.find("v", {2});
  check(!pinned.put("another-version", {3}, [](Value &v) { v.fill(3); }));
  check((*reader.value)[0] == 2 && pinned.allocated_slots() == 1);
  reader.value.reset();
  // A failed transfer must not publish a partially filled snapshot or lose a slot.
  try {
    pinned.put("v", {3}, [](Value &) { throw std::runtime_error("transfer"); });
    check(false);
  } catch (const std::runtime_error &e) {
    check(std::string(e.what()) == "transfer");
  }
  check(!pinned.find("v", {3}).value);
  check(pinned.put("v", {4}, [](Value &v) { v.fill(4); }));
  check(pinned.allocated_slots() == 1);
  for (auto limits : {std::pair<size_t, size_t>{0, 100}, {20, 0}, {20, sizeof(Value) - 1}}) {
    PrefixCachePool<Value> disabled(limits.first, limits.second, sizeof(Value), make);
    check(disabled.capacity() == 0);
    check(!disabled.put("v", {1}, [](Value &) { check(false); }));
    check(disabled.allocated_slots() == 0);
  }
}
int main() {
  PrefixCache<int> cache(3);
  auto put = [&](std::vector<int64_t> ids, int value) {
    cache.put("v1", ids, std::make_shared<int>(value));
  };
  put({1, 2}, 2);
  put({1, 2, 3, 4}, 4);
  put({1, 2, 3, 4, 5, 6}, 6);
  check(cache.find("v1", {1, 2, 3, 4, 5, 6, 7, 8}).length == 6);
  check(cache.find("v2", {1, 2, 3, 4}).length == 0);
  check(cache.find("v1", {1, 2, 3, 9}).length == 2);
  put({1, 2, 3, 4, 7}, 7); // evicts ABCD after AB was touched
  check(cache.find("v1", {1, 2, 3, 4}).length == 2);
  put({1, 9}, 9);
  put({1, 2, 3, 4}, 40);
  check(*cache.find("v1", {1, 2, 3, 4}).value == 40);
  check(cache.size() == 3);
  PrefixCache<int> disabled(0);
  disabled.put("v", {1}, std::make_shared<int>(1));
  check(!disabled.find("v", {1}).value);
  // Differential randomized LRU test, including splits, replacements, pruning,
  // namespaces and requests ending INSIDE compressed edges.
  struct E {
    std::string ns;
    std::vector<int64_t> ids;
    int value;
  };
  std::list<E> reference;
  PrefixCache<int> random_cache(7);
  rwkv7_server::PrefixCachePool<int> random_pool(
      7, 7 * sizeof(int), sizeof(int), [] { return std::make_shared<int>(); });
  std::mt19937 rng(55);
  for (int i = 0; i < 20000; ++i) {
    std::string ns = std::to_string(rng() % 3);
    std::vector<int64_t> ids(1 + rng() % 12);
    for (auto &id : ids)
      id = rng() % 4;
    if (rng() % 2) {
      random_cache.put(ns, ids, std::make_shared<int>(i));
      check(random_pool.put(ns, ids, [&](int &value) { value = i; }));
      reference.remove_if(
          [&](const E &e) { return e.ns == ns && e.ids == ids; });
      reference.push_front({ns, ids, i});
      if (reference.size() > 7)
        reference.pop_back();
    } else {
      auto best = reference.end();
      size_t length = 0;
      for (auto it = reference.begin(); it != reference.end(); ++it)
        if (it->ns == ns && it->ids.size() > length &&
            it->ids.size() <= ids.size() &&
            std::equal(it->ids.begin(), it->ids.end(), ids.begin())) {
          best = it;
          length = it->ids.size();
        }
      auto actual = random_cache.find(ns, ids);
      auto pooled = random_pool.find(ns, ids);
      check(actual.length == length);
      check(pooled.length == length);
      if (best != reference.end()) {
        check(*actual.value == best->value);
        check(*pooled.value == best->value);
        reference.splice(reference.begin(), reference, best);
      }
    }
    check(random_pool.size() == reference.size());
    check(random_pool.allocated_slots() <= 7);
  }
  // Exact extraction must not accidentally remove a shorter prefix match.
  PrefixCache<int> extract(3);
  extract.put("a", {1, 2}, std::make_shared<int>(2));
  extract.put("a", {1, 2, 3, 4}, std::make_shared<int>(4));
  extract.put("b", {1, 2}, std::make_shared<int>(20));
  check(!extract.extract_exact("a", {1, 2, 3}));
  check(!extract.extract_exact("a", {1, 2, 3, 4, 5}));
  check(*extract.extract_exact("a", {1, 2}) == 2);
  check(extract.find("a", {1, 2, 3, 4}).length == 4);
  check(*extract.extract_lru() == 20);
  check(*extract.extract_lru() == 4);
  check(!extract.extract_lru() && extract.size() == 0);
  pool_test();
  std::cout << "Radix prefix cache/global LRU/byte budget/buffer reuse/version retirement passed\n";
}
