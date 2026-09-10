#pragma once

#include <cstddef>
#include <fstream>
#include <string>

namespace rwkv7_state_tuning {

class JsonlTextReader {
public:
  explicit JsonlTextReader(const std::string &path);
  bool next(std::string &text);
  std::size_t line_number() const { return line_number_; }

private:
  std::ifstream input_;
  std::string path_;
  std::size_t line_number_ = 0;
};

std::size_t count_jsonl_rows(const std::string &path);

} // namespace rwkv7_state_tuning
