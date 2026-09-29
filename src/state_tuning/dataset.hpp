#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace rwkv7_state_tuning {

// One tokenization unit of a sample. Tokens produced from a segment with
// train == false are context only: they are fed forward but never used as
// loss targets.
struct TextSegment {
  std::string text;
  bool train = true;
};

class JsonlTextReader {
public:
  explicit JsonlTextReader(const std::string &path);
  bool next(std::vector<TextSegment> &segments);
  // Plain {"text": ...} rows only; throws on rows with masked segments.
  bool next(std::string &text);
  std::size_t line_number() const { return line_number_; }

private:
  std::ifstream input_;
  std::string path_;
  std::size_t line_number_ = 0;
};

std::size_t count_jsonl_rows(const std::string &path);

} // namespace rwkv7_state_tuning
