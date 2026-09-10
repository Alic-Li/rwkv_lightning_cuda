#include "dataset.hpp"

#include <cctype>
#include <cstdint>
#include <stdexcept>

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

std::string parse_text(const std::string &line) {
  std::size_t at = 0;
  whitespace(line, at);
  if (at >= line.size() || line[at++] != '{')
    throw std::runtime_error("JSONL row must be an object");
  whitespace(line, at);
  const std::string key = json_string(line, at);
  whitespace(line, at);
  if (at >= line.size() || line[at++] != ':')
    throw std::runtime_error("missing ':' after text");
  whitespace(line, at);
  const std::string value = json_string(line, at);
  whitespace(line, at);
  if (key != "text")
    throw std::runtime_error("JSONL row must contain only the text field");
  if (at < line.size() && line[at] == ',')
    throw std::runtime_error("only the text field is supported");
  if (at >= line.size() || line[at++] != '}')
    throw std::runtime_error("missing closing object brace");
  whitespace(line, at);
  if (at != line.size())
    throw std::runtime_error("trailing JSON data");
  return value;
}

} // namespace

JsonlTextReader::JsonlTextReader(const std::string &path)
    : input_(path), path_(path) {
  if (!input_)
    throw std::runtime_error("failed to open dataset: " + path);
}

bool JsonlTextReader::next(std::string &text) {
  std::string line;
  while (std::getline(input_, line)) {
    ++line_number_;
    if (line.empty())
      continue;
    try {
      text = parse_text(line);
      return true;
    } catch (const std::exception &error) {
      throw std::runtime_error(path_ + ":" + std::to_string(line_number_) +
                               ": " + error.what());
    }
  }
  return false;
}

std::size_t count_jsonl_rows(const std::string &path) {
  JsonlTextReader reader(path);
  std::string text;
  std::size_t rows = 0;
  while (reader.next(text))
    ++rows;
  return rows;
}

} // namespace rwkv7_state_tuning
