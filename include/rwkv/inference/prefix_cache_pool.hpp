#pragma once
#include "rwkv/inference/prefix_cache_capacity.hpp"
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <openssl/sha.h>
#include <string>
#include <vector>

namespace rwkv7_server {

inline void check_cache_copy(cudaError_t err) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("prefix cache: ") +
                             cudaGetErrorString(err));
}

inline void copy_cache_bytes(void *dst, bool dst_gpu, const void *src,
                             bool src_gpu, size_t bytes) {
  if (!bytes)
    return;
  if (!src_gpu && !dst_gpu) {
    std::memcpy(dst, src, bytes);
    return;
  }
  check_cache_copy(cudaMemcpy(
      dst, src, bytes,
      src_gpu ? (dst_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost)
              : cudaMemcpyHostToDevice));
}

// Serialized by the caller. Slot locations never change. A match is valid until
// the next mutating operation. The sorted index hashes each requested prefix
// incrementally at occupied lengths; even long prompts leave no resident key
// copies.
class PrefixCachePool {
  using Digest = std::array<unsigned char, SHA256_DIGEST_LENGTH>;
  struct Key {
    Digest identity{}, tokens{};
    size_t length = 0;
    uint64_t touched = 0;
  };
  struct DeviceFree {
    void operator()(unsigned char *p) const {
      rwkv_gpu::check_cleanup(cudaFree(p), "free prefix cache VRAM");
    }
  };
  size_t bytes_, hot_;
  std::unique_ptr<unsigned char[]> ram_;
  std::unique_ptr<unsigned char, DeviceFree> vram_;
  std::vector<Key> keys_;
  std::vector<size_t> by_length_;
  uint64_t clock_ = 0, hits_ = 0, misses_ = 0, evictions_ = 0, retired_ = 0;
  static Digest digest(const void *p, size_t n) {
    SHA256_CTX hash;
    Digest out;
    SHA256_Init(&hash);
    SHA256_Update(&hash, p, n);
    SHA256_Final(out.data(), &hash);
    return out;
  }
  size_t oldest(size_t first, size_t end) const {
    return std::min_element(keys_.begin() + first, keys_.begin() + end,
                            [](const Key &a, const Key &b) {
                              return a.touched < b.touched;
                            }) -
           keys_.begin();
  }
  void unlink(size_t i) {
    if (!keys_[i].length)
      return;
    by_length_.erase(std::find(by_length_.begin(), by_length_.end(), i));
    keys_[i] = {};
  }
  void link(size_t i, const Key &key) {
    keys_[i] = key;
    auto it = std::lower_bound(by_length_.begin(), by_length_.end(), key.length,
                               [&](size_t slot, size_t length) {
                                 return keys_[slot].length < length;
                               });
    by_length_.insert(it, i);
  }
  void move(size_t dst, size_t src) {
    const Key key = keys_[src];
    unlink(dst);
    copy_cache_bytes(data(dst), gpu(dst), data(src), gpu(src), bytes_);
    unlink(src);
    link(dst, key);
  }

public:
  static constexpr size_t none = size_t(-1);
  PrefixCachePool(PrefixCacheCapacity c, size_t bytes)
      : bytes_(bytes), hot_(bytes ? c.vram / bytes : 0) {
    if (!bytes || (c.vram && !hot_) || (c.ram && c.ram < bytes))
      throw std::invalid_argument(
          "prefix cache tier too small for one snapshot of " +
          std::to_string(bytes) + " bytes");
    if (c.ram > std::numeric_limits<size_t>::max() - c.vram)
      throw std::invalid_argument("combined prefix cache capacity overflow");
    if (c.vram) {
      unsigned char *p = nullptr;
      check_cache_copy(cudaMalloc(reinterpret_cast<void **>(&p), c.vram));
      vram_.reset(p);
    }
    if (c.ram) {
      ram_.reset(new unsigned char[c.ram]);
      // Touch each page now, including on overcommitting hosts.
      volatile unsigned char *pages = ram_.get();
      for (size_t i = 0; i <= (c.ram - 1) / 4096; ++i)
        pages[i * 4096] = 0;
      pages[c.ram - 1] = 0;
    }
    keys_.resize(hot_ + c.ram / bytes);
    by_length_.resize(keys_.size());
    by_length_.clear();
  }
  size_t capacity() const { return keys_.size(); }
  size_t entry_bytes() const { return bytes_; }
  size_t size() const { return by_length_.size(); }
  size_t size(const std::string &id) const {
    auto d = digest(id.data(), id.size());
    return std::count_if(keys_.begin(), keys_.end(), [&](const Key &k) {
      return k.length && k.identity == d;
    });
  }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t evictions() const { return evictions_; }
  uint64_t retired() const { return retired_; }
  bool gpu(size_t i) const { return i < hot_; }
  unsigned char *data(size_t i) const {
    return gpu(i) ? vram_.get() + i * bytes_ : ram_.get() + (i - hot_) * bytes_;
  }
  size_t length(size_t i) const { return i == none ? 0 : keys_[i].length; }
  // Read before promotion. The caller can refill the hot slot from the restored
  // request row, so even a full pool needs no extra snapshot to swap tiers.
  size_t find(const std::string &id, const std::vector<int64_t> &tokens,
              bool touch = true) {
    const auto identity = digest(id.data(), id.size());
    size_t hashed = 0, best = none;
    SHA256_CTX rolling;
    SHA256_Init(&rolling);
    Digest prefix{};
    for (auto i : by_length_) {
      const auto &k = keys_[i];
      if (k.length > tokens.size())
        break;
      if (k.identity != identity)
        continue;
      if (k.length > hashed) {
        SHA256_Update(&rolling, tokens.data() + hashed,
                      (k.length - hashed) * sizeof(int64_t));
        hashed = k.length;
        auto copy = rolling;
        SHA256_Final(prefix.data(), &copy);
      }
      if (k.tokens == prefix)
        best = i;
    }
    if (!touch)
      return best;
    if (best == none)
      ++misses_;
    else {
      ++hits_;
      keys_[best].touched = ++clock_;
    }
    return best;
  }
  bool needs_promotion(size_t i) const { return hot_ && i != none && !gpu(i); }
  template <class Fill> void promote(size_t cold, Fill refill) {
    if (!needs_promotion(cold))
      return;
    size_t hot = oldest(0, hot_);
    const Key hit = keys_[cold];
    unlink(cold);
    if (keys_[hot].length)
      move(cold, hot);
    refill(data(hot));
    link(hot, hit);
  }
  template <class Fill>
  void put(const std::string &id, const std::vector<int64_t> &tokens,
           Fill fill) {
    if (!capacity() || tokens.empty())
      return;
    Key key{digest(id.data(), id.size()),
            digest(tokens.data(), tokens.size() * sizeof(int64_t)),
            tokens.size(), ++clock_};
    for (auto i : by_length_)
      if (keys_[i].length == key.length && keys_[i].identity == key.identity &&
          keys_[i].tokens == key.tokens)
        return;
    size_t i = oldest(0, hot_ ? hot_ : capacity());
    if (keys_[i].length && hot_ && hot_ < capacity()) {
      size_t cold = oldest(hot_, capacity());
      if (keys_[cold].length)
        ++evictions_;
      move(cold, i);
    } else if (keys_[i].length) {
      ++evictions_;
      unlink(i);
    }
    fill(data(i), gpu(i));
    link(i, key);
  }
  void erase_identity(const std::string &id) {
    auto d = digest(id.data(), id.size());
    for (size_t i = 0; i < capacity(); ++i)
      if (keys_[i].length && keys_[i].identity == d) {
        unlink(i);
        ++retired_;
      }
  }
};
} // namespace rwkv7_server
