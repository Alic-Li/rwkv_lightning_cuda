#pragma once
#include "rwkv/inference/prefix_cache.hpp"
#include <functional>
#include <stdexcept>

namespace rwkv7_server {
// One LRU across policy identities, backed by a fixed maximum number of CPU
// snapshot buffers. Buffers are allocated lazily and reused after eviction or
// retirement. The byte budget covers Value's fixed tensor payload; trie/token
// metadata is additional and bounded by capacity and the caller's context limit.
// All access must be serialized. A Match pins its buffer: never overwrite a
// snapshot still held by a reader, even if that costs a skipped insertion.
template <class Value> class PrefixCachePool {
  using Pointer = std::shared_ptr<Value>;
  const size_t capacity_, memory_limit_, entry_bytes_;
  PrefixCache<Value> index_;
  std::function<Pointer()> make_;
  std::vector<Pointer> free_;
  size_t allocated_ = 0;
  uint64_t hits_ = 0, misses_ = 0, evictions_ = 0, retired_ = 0, skipped_ = 0;

  Pointer acquire() {
    for (size_t i = 0; i < free_.size(); ++i) {
      if (free_[i].use_count() != 1)
        continue;
      auto value = std::move(free_[i]);
      if (i + 1 != free_.size())
        free_[i] = std::move(free_.back());
      free_.pop_back();
      return value;
    }
    if (allocated_ < capacity_) {
      auto value = make_();
      if (!value)
        throw std::runtime_error("null prefix cache buffer");
      ++allocated_;
      return value;
    }
    while (auto value = index_.extract_lru()) {
      ++evictions_;
      if (value.use_count() == 1)
        return value;
      free_.push_back(std::move(value));
    }
    return {};
  }

public:
  using Match = typename PrefixCache<Value>::Match;
  PrefixCachePool(size_t capacity, size_t memory_limit, size_t entry_bytes,
                  std::function<Pointer()> make)
      : capacity_(entry_bytes ? std::min(capacity, memory_limit / entry_bytes) : 0),
        memory_limit_(memory_limit), entry_bytes_(entry_bytes),
        index_(capacity_), make_(std::move(make)) {
    if (!entry_bytes)
      throw std::invalid_argument("zero prefix cache entry size");
  }
  PrefixCachePool(const PrefixCachePool &) = delete;
  PrefixCachePool &operator=(const PrefixCachePool &) = delete;
  size_t capacity() const { return capacity_; }
  size_t memory_limit() const { return memory_limit_; }
  size_t entry_bytes() const { return entry_bytes_; }
  size_t size() const { return index_.size(); }
  size_t size(const std::string &identity) const { return index_.size(identity); }
  size_t used_bytes() const { return size() * entry_bytes_; }
  size_t allocated_slots() const { return allocated_; }
  size_t allocated_bytes() const { return allocated_ * entry_bytes_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t evictions() const { return evictions_; }
  uint64_t retired() const { return retired_; }
  uint64_t skipped() const { return skipped_; }
  Match find(const std::string &identity, const std::vector<int64_t> &tokens) {
    auto match = index_.find(identity, tokens);
    if (match.value)
      ++hits_;
    else
      ++misses_;
    return match;
  }
  // make() must allocate exactly entry_bytes of tensor storage per Value.
  // fill() writes into that storage without resizing/replacing its buffers.
  template <class Fill>
  bool put(const std::string &identity, const std::vector<int64_t> &tokens,
           Fill fill) {
    if (!capacity_ || tokens.empty())
      return false;
    auto value = index_.extract_exact(identity, tokens);
    if (value && value.use_count() != 1)
      free_.push_back(std::move(value));
    if (!value)
      value = acquire();
    if (!value) {
      ++skipped_;
      return false;
    }
    try {
      fill(*value);
      index_.put(identity, tokens, value);
    } catch (...) {
      free_.push_back(std::move(value));
      throw;
    }
    return true;
  }
  void erase_identity(const std::string &identity) {
    index_.erase_identity(identity, [&](Pointer value) {
      ++retired_;
      free_.push_back(std::move(value));
    });
  }
};
} // namespace rwkv7_server
