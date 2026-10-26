#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rwkv7_server {
// Compressed token radix trie with a bounded LRU of CPU snapshots. The caller
// supplies the full policy/state identity and serializes access. Internal trie
// nodes never imply that a recurrent-state snapshot exists at that boundary.
template <class Value> class PrefixCache {
  struct Entry;
  using Order = std::list<std::shared_ptr<Entry>>;
  struct Entry {
    std::string identity;
    std::vector<int64_t> tokens;
    std::shared_ptr<Value> value;
    typename Order::iterator position;
  };
  struct Node {
    std::vector<int64_t> edge;
    std::map<int64_t, std::unique_ptr<Node>> children;
    std::shared_ptr<Entry> entry;
  };
  size_t capacity_;
  std::map<std::string, Node> roots_;
  Order order_;

  static Node &insert(Node &node, const std::vector<int64_t> &tokens,
                      size_t off) {
    if (off == tokens.size())
      return node;
    auto &child = node.children[tokens[off]];
    if (!child) {
      child = std::make_unique<Node>();
      child->edge.assign(tokens.begin() + off, tokens.end());
      return *child;
    }
    size_t common = 0;
    while (common < child->edge.size() && off + common < tokens.size() &&
           child->edge[common] == tokens[off + common])
      ++common;
    if (common < child->edge.size()) {
      auto tail = std::move(child);
      child = std::make_unique<Node>();
      child->edge.assign(tail->edge.begin(), tail->edge.begin() + common);
      tail->edge.erase(tail->edge.begin(), tail->edge.begin() + common);
      const auto first = tail->edge.front();
      child->children.emplace(first, std::move(tail));
    }
    return insert(*child, tokens, off + common);
  }
  static void erase(Node &node, const std::vector<int64_t> &tokens,
                    size_t off) {
    if (off == tokens.size()) {
      node.entry.reset();
      return;
    }
    auto it = node.children.find(tokens[off]);
    if (it == node.children.end())
      return;
    auto &child = *it->second;
    erase(child, tokens, off + child.edge.size());
    if (!child.entry && child.children.empty())
      node.children.erase(it);
    else if (!child.entry && child.children.size() == 1) {
      auto next = std::move(child.children.begin()->second);
      next->edge.insert(next->edge.begin(), child.edge.begin(),
                        child.edge.end());
      it->second = std::move(next);
    }
  }
  std::shared_ptr<Value> extract(typename Order::iterator position) {
    auto entry = *position;
    auto root = roots_.find(entry->identity);
    erase(root->second, entry->tokens, 0);
    if (root->second.children.empty())
      roots_.erase(root);
    order_.erase(position);
    return std::move(entry->value);
  }

public:
  struct Match {
    std::shared_ptr<Value> value;
    size_t length = 0;
  };
  explicit PrefixCache(size_t capacity) : capacity_(capacity) {}
  size_t size() const { return order_.size(); }
  size_t size(const std::string &identity) const {
    return std::count_if(order_.begin(), order_.end(), [&](const auto &entry) {
      return entry->identity == identity;
    });
  }
  // Removing an entry returns its snapshot so a bounded pool can reuse it.
  std::shared_ptr<Value> extract_lru() {
    return order_.empty() ? nullptr : extract(std::prev(order_.end()));
  }
  std::shared_ptr<Value> extract_exact(const std::string &identity,
                                      const std::vector<int64_t> &tokens) {
    auto root = roots_.find(identity);
    if (root == roots_.end())
      return {};
    auto *node = &root->second;
    size_t offset = 0;
    while (offset < tokens.size()) {
      auto it = node->children.find(tokens[offset]);
      if (it == node->children.end())
        return {};
      node = it->second.get();
      if (offset + node->edge.size() > tokens.size() ||
          !std::equal(node->edge.begin(), node->edge.end(),
                      tokens.begin() + offset))
        return {};
      offset += node->edge.size();
    }
    return node->entry ? extract(node->entry->position) : nullptr;
  }
  template <class Recycle>
  void erase_identity(const std::string &identity, Recycle recycle) {
    for (auto it = order_.begin(); it != order_.end();) {
      auto current = it++;
      if ((*current)->identity == identity)
        recycle(extract(current));
    }
  }
  void clear() {
    roots_.clear();
    order_.clear();
  }
  Match find(const std::string &identity, const std::vector<int64_t> &tokens) {
    auto root = roots_.find(identity);
    if (root == roots_.end())
      return {};
    auto *node = &root->second;
    std::shared_ptr<Entry> best;
    size_t offset = 0;
    while (offset < tokens.size()) {
      auto it = node->children.find(tokens[offset]);
      if (it == node->children.end())
        break;
      node = it->second.get();
      if (offset + node->edge.size() > tokens.size() ||
          !std::equal(node->edge.begin(), node->edge.end(),
                      tokens.begin() + offset))
        break;
      offset += node->edge.size();
      if (node->entry)
        best = node->entry;
    }
    if (!best)
      return {};
    order_.splice(order_.begin(), order_, best->position);
    return {best->value, best->tokens.size()};
  }
  void put(const std::string &identity, const std::vector<int64_t> &tokens,
           std::shared_ptr<Value> value) {
    if (!capacity_ || tokens.empty())
      return;
    auto &node = insert(roots_[identity], tokens, 0);
    if (node.entry) {
      node.entry->value = std::move(value);
      order_.splice(order_.begin(), order_, node.entry->position);
      return;
    }
    auto entry = std::make_shared<Entry>();
    entry->identity = identity;
    entry->tokens = tokens;
    entry->value = std::move(value);
    order_.push_front(entry);
    entry->position = order_.begin();
    node.entry = entry;
    while (order_.size() > capacity_)
      extract_lru();
  }
};
} // namespace rwkv7_server
