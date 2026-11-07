#pragma once
#include "rwkv/io/fingerprint.hpp"
#include "rwkv/runtime/rwkv_miss.hpp"
#include <json/json.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace rwkv7_miss {
extern const std::array<const char *, TargetCount> target_names;
struct HostTensor {
  int layer = 0, target = 0, in = 0, out = 0, rank = 0;
  std::size_t offset = 0, a_offset = 0;
};
struct Package {
  Json::Value manifest;
  std::vector<HostTensor> tensors;
  std::vector<std::uint16_t> data;
  std::string version, base;
  float scale = 1;
};
std::string content_version(const Json::Value &,
                            const std::vector<std::uint16_t> &);
Json::Value adapter_manifest(Json::Value, const std::vector<HostTensor> &,
                            const std::vector<std::uint16_t> &);
void save_adapter_pth(const std::string &, Json::Value,
                      const std::vector<HostTensor> &,
                      const std::vector<std::uint16_t> &);
std::shared_ptr<const Package> load_package(const std::string &directory,
                                            std::size_t max_bytes = 1ULL << 30);
void save_package(const std::string &directory, Json::Value manifest,
                  const std::vector<HostTensor> &,
                  const std::vector<std::uint16_t> &);
} // namespace rwkv7_miss
