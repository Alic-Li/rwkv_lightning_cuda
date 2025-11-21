// Molly 老师的Tokenizer实现 我搬过来用上了 十分感谢 Molly 老师
#include "tokenizer.cuh"
#include "trie.cuh"

int trie_tokenizer::load(const std::string vocab_file) {
    _tokenizer = new OptimizedTrieTokenizer(vocab_file);
    if (!_tokenizer->inited())
        return RWKV_ERROR_TOKENIZER;
    return RWKV_SUCCESS;
}

trie_tokenizer::~trie_tokenizer() {
    if (_tokenizer != nullptr) {
        delete _tokenizer;
    }
}

std::vector<int> trie_tokenizer::encode(std::string_view str) const {
    auto ids = _tokenizer->encode(std::string(str));
    return ids;
}

std::string trie_tokenizer::decode(int id) const {
    return _tokenizer->decode(std::vector<int>{id});
}

std::string trie_tokenizer::decode(const std::vector<int> &ids) const {
    return _tokenizer->decode(ids);
}

