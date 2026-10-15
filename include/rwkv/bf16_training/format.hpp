#pragma once
#include "rwkv/io/fingerprint.hpp"
#include <array>
#include <cstdint>
#include <json/json.h>
#include <vector>
namespace rwkv_bf16_miss {
using rwkv7_miss::fingerprint_file;
inline constexpr std::array<const char *, 6> target_names = {
    "att.receptance.weight", "att.key.weight", "att.value.weight",
    "att.output.weight",     "ffn.key.weight", "ffn.value.weight"};
struct HostTensor {
  int layer, target, in, out, rank;
  size_t offset;
};
Json::Value adapter_manifest(Json::Value, const std::vector<HostTensor> &,
                             const std::vector<uint16_t> &);
void save_adapter_pth(const std::string &, Json::Value,
                      const std::vector<HostTensor> &,
                      const std::vector<uint16_t> &);
} // namespace rwkv_bf16_miss
