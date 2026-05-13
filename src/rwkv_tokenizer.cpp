#include "rwkv_tokenizer.hpp"

#include "rwkv_trie.hpp"

namespace rwkv7_server {

TrieTokenizer::TrieTokenizer() : TokenizerBase(0, 0, 0) {}

TrieTokenizer::~TrieTokenizer() {
  delete tokenizer_;
  tokenizer_ = nullptr;
}

int TrieTokenizer::load(const std::string& vocab_file) {
  delete tokenizer_;
  tokenizer_ = new OptimizedTrieTokenizer(vocab_file);
  return tokenizer_->inited() ? kTokenizerSuccess : kTokenizerError;
}

std::vector<int> TrieTokenizer::encode(std::string_view text) const {
  return tokenizer_ ? tokenizer_->encode(std::string(text)) : std::vector<int>{};
}

std::string TrieTokenizer::decode(const std::vector<int>& ids) const {
  return tokenizer_ ? tokenizer_->decode(ids) : std::string{};
}

std::string TrieTokenizer::decode(int id) const {
  return decode(std::vector<int>{id});
}

}  // namespace rwkv7_server
