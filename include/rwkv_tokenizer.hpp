#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rwkv7_server {

constexpr int kTokenizerSuccess = 0;
constexpr int kTokenizerError = -1;

class OptimizedTrieTokenizer;

class TokenizerBase {
 public:
  TokenizerBase(int pad_token_id, int bos_token_id, int eos_token_id)
      : pad_token_id(pad_token_id),
        bos_token_id(bos_token_id),
        eos_token_id(eos_token_id) {}
  virtual ~TokenizerBase() = default;

  virtual int load(const std::string& vocab_file) = 0;
  virtual std::vector<int> encode(std::string_view text) const = 0;
  virtual std::string decode(const std::vector<int>& ids) const = 0;
  virtual std::string decode(int id) const = 0;

  const int pad_token_id;
  const int bos_token_id;
  const int eos_token_id;
};

class TrieTokenizer : public TokenizerBase {
 public:
  TrieTokenizer();
  ~TrieTokenizer() override;

  int load(const std::string& vocab_file) override;
  std::vector<int> encode(std::string_view text) const override;
  std::string decode(const std::vector<int>& ids) const override;
  std::string decode(int id) const override;

 private:
  OptimizedTrieTokenizer* tokenizer_ = nullptr;
};

}  // namespace rwkv7_server
