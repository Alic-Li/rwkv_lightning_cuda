#pragma once

#include <string>
#include <string_view>
#include <vector>

class trie_tokenizer;

namespace rwkv7_server {

inline constexpr int kTokenizerSuccess = 0;
inline constexpr int kTokenizerError = -1;

class TrieTokenizer {
 public:
  TrieTokenizer();
  ~TrieTokenizer();

  TrieTokenizer(const TrieTokenizer&) = delete;
  TrieTokenizer& operator=(const TrieTokenizer&) = delete;

  int load(const std::string& vocab_file);
  std::vector<int> encode(std::string_view text) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string decode(int id) const;

 private:
  trie_tokenizer* impl_ = nullptr;
};

}  // namespace rwkv7_server
