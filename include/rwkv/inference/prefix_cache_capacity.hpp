#pragma once
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace rwkv7_server {
struct PrefixCacheCapacity {
  size_t vram = 0;
  size_t ram = size_t{2} << 30;
};

inline PrefixCacheCapacity parse_prefix_cache_capacity(std::string_view spec) {
  auto parse = [](std::string_view field, std::string_view prefix) -> size_t {
    if (field.size() <= prefix.size() ||
        field.substr(0, prefix.size()) != prefix)
      throw std::invalid_argument(
          "expected --prefix-cache-capacity vram:XG,ram:XG");
    field.remove_prefix(prefix.size());
    size_t scale = 1;
    switch (field.back()) {
    case 'G':
      scale = size_t{1} << 30;
      break;
    case 'M':
      scale = size_t{1} << 20;
      break;
    case 'K':
      scale = 1024;
      break;
    case 'B':
      break;
    default:
      throw std::invalid_argument("prefix cache size requires G, M, K or B");
    }
    field.remove_suffix(1);
    size_t n = 0;
    auto end = field.data() + field.size();
    auto parsed = std::from_chars(field.data(), end, n);
    if (parsed.ec != std::errc{} || parsed.ptr != end ||
        n > std::numeric_limits<size_t>::max() / scale)
      throw std::invalid_argument("invalid prefix cache size");
    return n * scale;
  };
  const auto comma = spec.find(',');
  if (comma == spec.npos)
    throw std::invalid_argument("expected VRAM and RAM capacities");
  return {parse(spec.substr(0, comma), "vram:"),
          parse(spec.substr(comma + 1), "ram:")};
}
} // namespace rwkv7_server
