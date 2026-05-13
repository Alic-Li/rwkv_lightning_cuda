#include "rwkv_tokenizer.hpp"

#include "utils/tokenizer.h"

namespace rwkv7_server {

TrieTokenizer::TrieTokenizer() = default;

TrieTokenizer::~TrieTokenizer() {
  delete impl_;
  impl_ = nullptr;
}

int TrieTokenizer::load(const std::string& vocab_file) {
  delete impl_;
  impl_ = new trie_tokenizer();
  const int status = impl_->load(vocab_file);
  if (status != RWKV_SUCCESS) {
    delete impl_;
    impl_ = nullptr;
    return kTokenizerError;
  }
  return kTokenizerSuccess;
}

std::vector<int> TrieTokenizer::encode(std::string_view text) const {
  return impl_ ? impl_->encode(text) : std::vector<int>{};
}

std::string TrieTokenizer::decode(const std::vector<int>& ids) const {
  return impl_ ? impl_->decode(ids) : std::string{};
}

std::string TrieTokenizer::decode(int id) const {
  return decode(std::vector<int>{id});
}

}  // namespace rwkv7_server
