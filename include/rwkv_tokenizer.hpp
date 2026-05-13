#pragma once

#include "utils/tokenizer.h"

namespace rwkv7_server {

using TokenizerBase = ::tokenizer_base;
using TrieTokenizer = ::trie_tokenizer;

inline constexpr int kTokenizerSuccess = RWKV_SUCCESS;
inline constexpr int kTokenizerError = RWKV_ERROR_TOKENIZER;

}  // namespace rwkv7_server
