#include "rwkv/inference/prefix_cache.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
using rwkv7_server::PrefixCache;
void check(bool ok) {
  if (!ok)
    throw std::runtime_error("radix prefix cache mismatch");
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
  std::mt19937 rng(55);
  for (int i = 0; i < 20000; ++i) {
    std::string ns = std::to_string(rng() % 3);
    std::vector<int64_t> ids(1 + rng() % 12);
    for (auto &id : ids)
      id = rng() % 4;
    if (rng() % 2) {
      random_cache.put(ns, ids, std::make_shared<int>(i));
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
      check(actual.length == length);
      if (best != reference.end()) {
        check(*actual.value == best->value);
        reference.splice(reference.begin(), reference, best);
      }
    }
  }
  std::cout << "Radix prefix cache/LRU/version isolation passed\n";
}
