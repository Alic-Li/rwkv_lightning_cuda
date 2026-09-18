#pragma once
#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"
#include "rwkv/runtime/rwkv_miss.hpp"
#include <json/json.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace rwkv7_miss {
extern const std::array<const char *, TargetCount> target_names;
struct HostTensor {
  int layer = 0, target = 0, in = 0, out = 0, rank = 0;
  std::size_t offset = 0;
};
struct Package {
  Json::Value manifest;
  std::vector<HostTensor> tensors;
  std::vector<std::uint16_t> data;
  std::string version, base;
  float scale = 1;
};
std::string fingerprint_file(const std::string &);
std::string new_runtime_identity();
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
// Handle is immutable and independent of GPU residency. Keep it with a request
// or paused session, and release its GPU lease when that request is paused.
struct AdapterHandle {
  const std::shared_ptr<const Package> package;
  const float scale;
  const std::string identity;
  AdapterHandle(std::shared_ptr<const Package>, float);
};
struct GpuLease {
  rwkv7_fast_v4::DeviceBuffer<half> data;
  std::vector<Block> blocks;
  std::string version;
  double h2d_ms = 0;
};
struct CacheStats {
  std::uint64_t ram_hits = 0, gpu_hits = 0, gpu_misses = 0, uploads = 0,
                evictions = 0;
  std::size_t ram_bytes = 0, gpu_bytes = 0, gpu_peak_bytes = 0;
  double h2d_ms = 0;
};
class AdapterCache {
public:
  AdapterCache(std::size_t ram = 1ULL << 30, std::size_t gpu = 512ULL << 20,
               std::size_t staging = 128ULL << 20);
  std::string register_adapter(const std::string &id,
                               const std::string &directory);
  std::shared_ptr<const AdapterHandle> resolve(const std::string &id,
                                               const std::string &version = {},
                                               const float *scale = nullptr);
  std::shared_ptr<const GpuLease> acquire(const AdapterHandle &,
                                          bool *cold = nullptr);
  void erase(const std::string &id, const std::string &version = {});
  void trim();
  Json::Value list() const;
  CacheStats stats() const;

private:
  mutable std::mutex mutex_;
  std::size_t ram_budget_, gpu_budget_, staging_budget_;
  std::uint64_t clock_ = 0;
  struct Resident {
    std::shared_ptr<GpuLease> lease;
    std::uint64_t used = 0;
  };
  std::map<std::string, std::map<std::string, std::shared_ptr<const Package>>>
      registry_;
  std::map<std::string, std::string> latest_;
  std::map<std::string, std::vector<std::string>> registration_order_;
  std::vector<std::weak_ptr<const Package>> host_lifetimes_;
  std::map<std::string, Resident> gpu_;
  CacheStats stats_;
  void account_ram();
  void admit_gpu(std::size_t bytes);
};
AdapterCache &adapter_cache();
std::string effective_model_key(const std::string &runtime,
                                const AdapterHandle *adapter,
                                const std::string &initial_state, bool fp32);
} // namespace rwkv7_miss
