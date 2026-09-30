#include "dataset.hpp"

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace rwkv7_state_tuning {
namespace {

void append_utf8(std::string &out, std::uint32_t cp) {
  if (cp <= 0x7f) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

int hex(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

std::uint32_t unicode_escape(const std::string &line, std::size_t &at) {
  if (at + 4 > line.size())
    throw std::runtime_error("truncated unicode escape");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    const int digit = hex(line[at++]);
    if (digit < 0)
      throw std::runtime_error("invalid unicode escape");
    value = (value << 4) | static_cast<std::uint32_t>(digit);
  }
  return value;
}

std::string json_string(const std::string &line, std::size_t &at) {
  if (at >= line.size() || line[at++] != '"')
    throw std::runtime_error("expected JSON string");
  std::string out;
  while (at < line.size()) {
    const unsigned char c = static_cast<unsigned char>(line[at++]);
    if (c == '"')
      return out;
    if (c < 0x20)
      throw std::runtime_error("control byte in JSON string");
    if (c != '\\') {
      out.push_back(static_cast<char>(c));
      continue;
    }
    if (at >= line.size())
      throw std::runtime_error("truncated JSON escape");
    const char escaped = line[at++];
    switch (escaped) {
    case '"':
    case '\\':
    case '/':
      out.push_back(escaped);
      break;
    case 'b':
      out.push_back('\b');
      break;
    case 'f':
      out.push_back('\f');
      break;
    case 'n':
      out.push_back('\n');
      break;
    case 'r':
      out.push_back('\r');
      break;
    case 't':
      out.push_back('\t');
      break;
    case 'u': {
      std::uint32_t cp = unicode_escape(line, at);
      if (cp >= 0xd800 && cp <= 0xdbff) {
        if (at + 2 > line.size() || line[at] != '\\' || line[at + 1] != 'u')
          throw std::runtime_error("missing low surrogate");
        at += 2;
        const std::uint32_t low = unicode_escape(line, at);
        if (low < 0xdc00 || low > 0xdfff)
          throw std::runtime_error("invalid low surrogate");
        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
      } else if (cp >= 0xdc00 && cp <= 0xdfff) {
        throw std::runtime_error("unexpected low surrogate");
      }
      append_utf8(out, cp);
      break;
    }
    default:
      throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

void whitespace(const std::string &line, std::size_t &at) {
  while (at < line.size() && std::isspace(static_cast<unsigned char>(line[at])))
    ++at;
}

void expect(const std::string &line, std::size_t &at, char c,
            const char *error) {
  whitespace(line, at);
  if (at >= line.size() || line[at] != c)
    throw std::runtime_error(error);
  ++at;
  whitespace(line, at);
}

bool json_bool(const std::string &line, std::size_t &at) {
  if (line.compare(at, 4, "true") == 0) {
    at += 4;
    return true;
  }
  if (line.compare(at, 5, "false") == 0) {
    at += 5;
    return false;
  }
  throw std::runtime_error("segment train must be true or false");
}

TextSegment parse_segment(const std::string &line, std::size_t &at) {
  expect(line, at, '{', "segment must be an object");
  bool has_text = false;
  bool has_train = false;
  TextSegment segment;
  while (true) {
    const std::string key = json_string(line, at);
    expect(line, at, ':', "missing ':' after segment field");
    if (key == "text" && !has_text) {
      segment.text = json_string(line, at);
      has_text = true;
    } else if (key == "train" && !has_train) {
      segment.train = json_bool(line, at);
      has_train = true;
    } else {
      throw std::runtime_error(
          "segment must contain exactly one text and one train field");
    }
    whitespace(line, at);
    if (at < line.size() && line[at] == ',') {
      expect(line, at, ',', "missing ','");
      continue;
    }
    expect(line, at, '}', "missing closing segment brace");
    break;
  }
  if (!has_text || !has_train)
    throw std::runtime_error("segment requires both text and train");
  return segment;
}

// A row is either {"text": "..."} (every token trained) or
// {"segments": [{"text": "...", "train": bool}, ...]}.
std::vector<TextSegment> parse_row(const std::string &line) {
  std::size_t at = 0;
  expect(line, at, '{', "JSONL row must be an object");
  const std::string key = json_string(line, at);
  expect(line, at, ':', "missing ':' after row field");
  std::vector<TextSegment> segments;
  if (key == "text") {
    segments.push_back({json_string(line, at), true});
  } else if (key == "segments") {
    expect(line, at, '[', "segments must be an array");
    if (at < line.size() && line[at] == ']')
      throw std::runtime_error("segments must not be empty");
    while (true) {
      segments.push_back(parse_segment(line, at));
      whitespace(line, at);
      if (at < line.size() && line[at] == ',') {
        expect(line, at, ',', "missing ','");
        continue;
      }
      expect(line, at, ']', "missing closing segments bracket");
      break;
    }
  } else {
    throw std::runtime_error("JSONL row must contain a text or segments field");
  }
  whitespace(line, at);
  if (at < line.size() && line[at] == ',')
    throw std::runtime_error("JSONL row must contain exactly one field");
  expect(line, at, '}', "missing closing object brace");
  if (at != line.size())
    throw std::runtime_error("trailing JSON data");
  return segments;
}

} // namespace

JsonlTextReader::JsonlTextReader(const std::string &path)
    : input_(path), path_(path) {
  if (!input_)
    throw std::runtime_error("failed to open dataset: " + path);
}

bool JsonlTextReader::next(std::vector<TextSegment> &segments) {
  std::string line;
  while (std::getline(input_, line)) {
    ++line_number_;
    if (line.empty())
      continue;
    try {
      segments = parse_row(line);
      return true;
    } catch (const std::exception &error) {
      throw std::runtime_error(path_ + ":" + std::to_string(line_number_) +
                               ": " + error.what());
    }
  }
  return false;
}

bool JsonlTextReader::next(std::string &text) {
  std::vector<TextSegment> segments;
  if (!next(segments))
    return false;
  if (segments.size() != 1 || !segments[0].train)
    throw std::runtime_error(path_ + ":" + std::to_string(line_number_) +
                             ": expected a plain text row");
  text = std::move(segments[0].text);
  return true;
}

std::size_t count_jsonl_rows(const std::string &path) {
  JsonlTextReader reader(path);
  std::vector<TextSegment> segments;
  std::size_t rows = 0;
  while (reader.next(segments))
    ++rows;
  return rows;
}

} // namespace rwkv7_state_tuning
