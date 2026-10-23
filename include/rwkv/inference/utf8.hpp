#pragma once
#include <string>

namespace rwkv7_server {
inline bool is_utf8_continuation(unsigned char byte) {
  return (byte & 0xC0u) == 0x80u;
}

inline int utf8_sequence_length(unsigned char lead) {
  if (lead <= 0x7Fu) {
    return 1;
  }
  if (lead >= 0xC2u && lead <= 0xDFu) {
    return 2;
  }
  if (lead >= 0xE0u && lead <= 0xEFu) {
    return 3;
  }
  if (lead >= 0xF0u && lead <= 0xF4u) {
    return 4;
  }
  return 0;
}

inline bool is_valid_utf8_sequence(const std::string &text, size_t pos,
                                   int len) {
  const auto b0 = static_cast<unsigned char>(text[pos]);
  switch (len) {
  case 1:
    return true;
  case 2:
    return is_utf8_continuation(static_cast<unsigned char>(text[pos + 1]));
  case 3: {
    const auto b1 = static_cast<unsigned char>(text[pos + 1]);
    const auto b2 = static_cast<unsigned char>(text[pos + 2]);
    if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2)) {
      return false;
    }
    if (b0 == 0xE0u && b1 < 0xA0u) {
      return false;
    }
    if (b0 == 0xEDu && b1 >= 0xA0u) {
      return false;
    }
    return true;
  }
  case 4: {
    const auto b1 = static_cast<unsigned char>(text[pos + 1]);
    const auto b2 = static_cast<unsigned char>(text[pos + 2]);
    const auto b3 = static_cast<unsigned char>(text[pos + 3]);
    if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2) ||
        !is_utf8_continuation(b3)) {
      return false;
    }
    if (b0 == 0xF0u && b1 < 0x90u) {
      return false;
    }
    if (b0 == 0xF4u && b1 > 0x8Fu) {
      return false;
    }
    return true;
  }
  default:
    return false;
  }
}

inline std::string take_complete_utf8(std::string &pending, bool flush_all) {
  std::string out;
  size_t i = 0;
  size_t last_copy = 0;
  while (i < pending.size()) {
    const auto lead = static_cast<unsigned char>(pending[i]);
    const int len = utf8_sequence_length(lead);
    if (len == 0) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    if (i + static_cast<size_t>(len) > pending.size()) {
      break;
    }
    if (!is_valid_utf8_sequence(pending, i, len)) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    i += static_cast<size_t>(len);
  }
  out.append(pending, last_copy, i - last_copy);
  pending.erase(0, i);
  if (flush_all && !pending.empty()) {
    pending.clear();
  }
  return out;
}

} // namespace rwkv7_server
