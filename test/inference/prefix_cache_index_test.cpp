#include "rwkv/inference/prefix_cache_pool.hpp"
#include "test_common.hpp"
#include <list>
#include <random>

using namespace rwkv7_server;
int main() {
  const bool have_gpu = rwkv_test::cuda_device_available();
  TEST_EQ(parse_prefix_cache_capacity("vram:4G,ram:2M").vram, size_t{4} << 30);
  TEST_THROW(parse_prefix_cache_capacity("vram:-1G,ram:2G"));
  TEST_THROW(parse_prefix_cache_capacity("vram:0G,ram:99999999999999999999G"));
  TEST_THROW(PrefixCachePool({1, 0}, sizeof(int)));
  for (size_t hot : {0, 1, 3, 7}) {
    if (hot && !have_gpu)
      continue;
    PrefixCachePool pool({hot * sizeof(int), (7 - hot) * sizeof(int)},
                         sizeof(int));
    struct Item {
      std::string ns;
      std::vector<int64_t> key;
      int value;
    };
    std::list<Item> lru;
    std::mt19937 rng(55);
    for (int i = 0; i < 10000; ++i) {
      std::string ns = std::to_string(rng() % 3);
      std::vector<int64_t> key(1 + rng() % 8);
      for (auto &token : key)
        token = rng() % 3;
      if (rng() % 2) {
        pool.put(ns, key, [&](unsigned char *slot, bool gpu) {
          copy_cache_bytes(slot, gpu, &i, false, sizeof(i));
        });
        auto exact = std::find_if(lru.begin(), lru.end(), [&](const Item &e) {
          return e.ns == ns && e.key == key;
        });
        if (exact == lru.end()) {
          lru.push_front({ns, key, i});
          if (lru.size() > 7)
            lru.pop_back();
        }
      } else {
        auto hit = pool.find(ns, key);
        auto best = lru.end();
        for (auto it = lru.begin(); it != lru.end(); ++it)
          if (it->ns == ns && it->key.size() <= key.size() &&
              (best == lru.end() || it->key.size() > best->key.size()) &&
              std::equal(it->key.begin(), it->key.end(), key.begin()))
            best = it;
        TEST_EQ(pool.length(hit), best == lru.end() ? 0 : best->key.size());
        if (best != lru.end()) {
          int value;
          copy_cache_bytes(&value, false, pool.data(hit), pool.gpu(hit),
                           sizeof(value));
          TEST_EQ(value, best->value);
          pool.promote(hit, [&](unsigned char *dst) {
            copy_cache_bytes(dst, true, &value, false, sizeof(value));
          });
          lru.splice(lru.begin(), lru, best);
        }
      }
      TEST_EQ(pool.size(), lru.size());
    }
    pool.erase_identity("0");
    TEST_EQ(pool.size("0"), size_t(0));
  }
}
