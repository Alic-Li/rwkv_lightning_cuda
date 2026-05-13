#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rwkv7_server {

inline std::string process_vocab_format(const std::string& input) {
  std::string final;
  if (!input.empty() && (input[0] == '\'' || input[0] == '"')) {
    final = input.substr(1, input.length() - 3);
  } else if (input.length() > 1 && input[0] == 'b' && (input[1] == '\'' || input[1] == '"')) {
    final = input.substr(2, input.length() - 4);
  } else {
    final = input;
  }
  return final;
}

inline std::vector<uint8_t> process_escapes(
    const std::string& input,
    bool utf8_string = false,
    int utf8_byte_length = -1) {
  if (utf8_string && utf8_byte_length > 0 && input.length() > 1 &&
      input[0] == '\\' && (input[1] == 'u' || input[1] == 'x')) {
    std::vector<uint8_t> result;
    std::istringstream stream(input);
    char ch;
    while (stream.get(ch)) {
      if (ch == '\\' && (stream.peek() == 'u' || stream.peek() == 'x')) {
        std::string hex_code;
        stream.get(ch);
        for (int i = 0; i < 4 && stream.get(ch); ++i) {
          hex_code += ch;
        }
        std::istringstream hex_stream(hex_code);
        uint32_t code_point = 0;
        hex_stream >> std::hex >> code_point;
        if (code_point <= 0x7F) {
          result.push_back(static_cast<uint8_t>(code_point));
        } else if (code_point <= 0x7FF) {
          result.push_back(static_cast<uint8_t>(192 + (code_point >> 6)));
          result.push_back(static_cast<uint8_t>(128 + (code_point & 0x3F)));
        } else if (code_point <= 0xFFFF) {
          result.push_back(static_cast<uint8_t>(224 + (code_point >> 12)));
          result.push_back(static_cast<uint8_t>(128 + ((code_point >> 6) & 0x3F)));
          result.push_back(static_cast<uint8_t>(128 + (code_point & 0x3F)));
        } else if (code_point <= 0x10FFFF) {
          result.push_back(static_cast<uint8_t>(240 + (code_point >> 18)));
          result.push_back(static_cast<uint8_t>(128 + ((code_point >> 12) & 0x3F)));
          result.push_back(static_cast<uint8_t>(128 + ((code_point >> 6) & 0x3F)));
          result.push_back(static_cast<uint8_t>(128 + (code_point & 0x3F)));
        }
      } else {
        result.push_back(static_cast<uint8_t>(ch));
      }
    }
    while (static_cast<int>(result.size()) < utf8_byte_length) {
      result.insert(result.begin(), 0);
    }
    return result;
  }

  std::vector<uint8_t> result;
  bool escape = false;
  for (size_t i = 0; i < input.length(); ++i) {
    const char c = input[i];
    if (escape) {
      switch (c) {
        case 'x': {
          std::string hex_digits;
          for (int j = 0; j < 2 && i + 1 < input.length(); ++j) {
            hex_digits += input[++i];
          }
          if (hex_digits.length() == 2) {
            result.push_back(static_cast<uint8_t>(std::stoul(hex_digits, nullptr, 16)));
          }
          break;
        }
        case 'n':
          result.push_back('\n');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '\'':
          result.push_back('\'');
          break;
        case '"':
          result.push_back('"');
          break;
        default:
          result.push_back('\\');
          result.push_back(static_cast<uint8_t>(c));
          break;
      }
      escape = false;
      continue;
    }
    if (c == '\\') {
      escape = true;
    } else {
      result.push_back(static_cast<uint8_t>(c));
    }
  }
  return result;
}

struct VectorEqual {
  bool operator()(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) const noexcept {
    return a == b;
  }
};

struct VectorHash {
  size_t operator()(const std::vector<uint8_t>& vec) const noexcept {
    size_t hash = 0;
    for (uint8_t byte : vec) {
      hash = hash * 31 + byte;
    }
    return hash;
  }
};

class OptimizedTrie {
 public:
  OptimizedTrie(OptimizedTrie* parent = nullptr, uint8_t ch = 0)
      : parent_(parent), ch_(ch) {}

  OptimizedTrie* add(const std::vector<uint8_t>& key, size_t idx = 0, int val = -1) {
    if (idx == key.size()) {
      if (val != -1) {
        values_.insert(val);
      }
      return this;
    }

    const uint8_t uchar = key[idx];
    auto& child = children_[uchar];
    if (!child) {
      child = std::make_unique<OptimizedTrie>(this, uchar);
    }
    return child->add(key, idx + 1, val);
  }

  std::tuple<size_t, int> find_longest_fast(const std::vector<uint8_t>& key, size_t idx = 0) {
    auto* node = this;
    std::tuple<size_t, int> ret;
    while (idx < key.size()) {
      const uint8_t uchar = key[idx];
      const auto it = node->children_.find(uchar);
      if (it == node->children_.end()) {
        break;
      }
      node = it->second.get();
      ++idx;
      if (!node->values_.empty()) {
        ret = std::make_tuple(idx, *node->values_.begin());
      }
    }
    return ret;
  }

 private:
  uint8_t ch_ = 0;
  std::unordered_map<uint8_t, std::unique_ptr<OptimizedTrie>> children_;
  std::unordered_set<int> values_;
  OptimizedTrie* parent_ = nullptr;
};

class TokenMapping {
 public:
  void add_token(int token_id, const std::vector<uint8_t>& data) {
    const auto it = data_to_idx_.find(data);
    if (it != data_to_idx_.end()) {
      idx_to_data_idx_[token_id] = it->second;
      return;
    }
    const size_t data_idx = token_data_.size();
    token_data_.push_back(data);
    idx_to_data_idx_[token_id] = data_idx;
    data_to_idx_[data] = token_id;
  }

  const std::vector<uint8_t>& get_token_data(int token_id) const {
    const auto it = idx_to_data_idx_.find(token_id);
    if (it != idx_to_data_idx_.end()) {
      return token_data_[it->second];
    }
    static const std::vector<uint8_t> empty;
    return empty;
  }

 private:
  std::vector<std::vector<uint8_t>> token_data_;
  std::unordered_map<int, size_t> idx_to_data_idx_;
  std::unordered_map<std::vector<uint8_t>, int, VectorHash, VectorEqual> data_to_idx_;
};

class OptimizedTrieTokenizer {
 public:
  explicit OptimizedTrieTokenizer(const std::string& file_name) {
    root_ = std::make_unique<OptimizedTrie>();
    std::ifstream file(file_name);
    if (!file.is_open()) {
      return;
    }

    std::string line;
    while (getline(file, line)) {
      const size_t first_space = line.find(' ');
      const size_t last_space = line.rfind(' ');
      const int idx = std::stoi(line.substr(0, first_space));
      const int utf8_byte_length = std::stoi(line.substr(last_space + 1));
      const bool utf8_string = line[first_space + 1] != 'b';

      std::vector<uint8_t> bytes = process_escapes(
          process_vocab_format(line.substr(first_space + 1, last_space - first_space)),
          utf8_string,
          utf8_byte_length);
      token_mapping_.add_token(idx, bytes);
      root_->add(bytes, 0, idx);
    }

    const std::vector<uint8_t> eod_data{'<', 'E', 'O', 'D', '>'};
    token_mapping_.add_token(0, eod_data);
    root_->add(eod_data, 0, 0);
    inited_ = true;
  }

  std::vector<int> encode(const std::string& src) {
    const std::vector<uint8_t> bytes(src.begin(), src.end());
    std::vector<int> tokens;
    tokens.reserve(bytes.size());
    size_t idx = 0;
    while (idx < bytes.size()) {
      const size_t old_idx = idx;
      int token = -1;
      std::tie(idx, token) = root_->find_longest_fast(bytes, idx);
      if (idx > old_idx && token != -1) {
        tokens.push_back(token);
      } else {
        break;
      }
    }
    return tokens;
  }

  std::string decode(const std::vector<int>& tokens) {
    std::vector<uint8_t> bytes;
    for (int token : tokens) {
      const auto& token_bytes = token_mapping_.get_token_data(token);
      bytes.insert(bytes.end(), token_bytes.begin(), token_bytes.end());
    }
    return std::string(bytes.begin(), bytes.end());
  }

  bool inited() const {
    return inited_;
  }

 private:
  TokenMapping token_mapping_;
  std::unique_ptr<OptimizedTrie> root_;
  bool inited_ = false;
};

}  // namespace rwkv7_server
