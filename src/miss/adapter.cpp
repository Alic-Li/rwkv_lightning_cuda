#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
namespace rwkv7_miss {
const std::array<const char *, TargetCount> target_names = {
    "att.receptance.weight", "att.key.weight", "att.value.weight",
    "att.output.weight",     "ffn.key.weight", "ffn.value.weight"};
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Hash {
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  Hash() {
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("SHA256 initialization");
  }
  ~Hash() { EVP_MD_CTX_free(ctx); }
  void add(const void *p, size_t n) {
    if (EVP_DigestUpdate(ctx, p, n) != 1)
      throw std::runtime_error("SHA256 update");
  }
  std::string finish() {
    unsigned char d[32];
    unsigned int n = 0;
    EVP_DigestFinal_ex(ctx, d, &n);
    std::ostringstream o;
    for (auto x : d)
      o << std::hex << std::setw(2) << std::setfill('0') << int(x);
    return o.str();
  }
};
std::string json(const Json::Value &v) {
  Json::StreamWriterBuilder w;
  w["indentation"] = "";
  return Json::writeString(w, v);
}
std::string tensor_name(const HostTensor &t) {
  return "blocks." + std::to_string(t.layer) + "." + target_names[t.target] +
         ".D";
}
} // namespace
std::string new_runtime_identity() {
  static const std::string process = [] {
    std::random_device random;
    std::ostringstream out;
    for (int i = 0; i < 4; ++i)
      out << std::hex << std::setw(8) << std::setfill('0') << random();
    return out.str();
  }();
  static std::atomic<std::uint64_t> next{0};
  return process + ":" + std::to_string(++next);
}
std::string content_version(const Json::Value &manifest,
                            const std::vector<std::uint16_t> &data) {
  auto m = manifest;
  m.removeMember("content_version");
  const auto s = json(m);
  Hash h;
  h.add(s.data(), s.size());
  h.add(data.data(), data.size() * 2);
  return h.finish();
}
Json::Value adapter_manifest(Json::Value m, const std::vector<HostTensor> &ts,
                             const std::vector<std::uint16_t> &data) {
  m["format_version"] = 1;
  m["method"] = "miss";
  if (m["dtype"] != "bfloat16") m["dtype"] = "float16";
  m["layout"] = "modulo_rank_zero_pad";
  m["kind"] = "inference_adapter";
  m["targets"] = Json::Value(Json::arrayValue);
  for (const auto &t : ts) {
    Json::Value v;
    v["name"] = tensor_name(t);
    v["layer"] = t.layer;
    v["target"] = target_names[t.target];
    v["in_features"] = t.in;
    v["shape"].append(t.out);
    v["shape"].append(t.rank);
    m["targets"].append(v);
  }
  m["content_version"] = content_version(m, data);
  return m;
}
void save_adapter_pth(const std::string &path, Json::Value m,
                      const std::vector<HostTensor> &ts,
                      const std::vector<std::uint16_t> &data) {
  if (std::filesystem::exists(path))
    throw std::runtime_error("adapter destination already exists: " + path);
  std::vector<llm_infer::WriteTensor> write;
  for (const auto &t : ts) {
    const size_t n = size_t(t.out) * t.rank;
    if (t.offset + n > data.size())
      throw std::runtime_error("adapter data bounds");
    llm_infer::WriteTensor w{tensor_name(t), {t.out, t.rank}, m["dtype"] != "bfloat16", {}};
    w.bf16 = m["dtype"] == "bfloat16";
    w.data.resize(n * 2);
    std::memcpy(w.data.data(), data.data() + t.offset, n * 2);
    write.push_back(std::move(w));
  }
  llm_infer::write_pth(path, write, json(adapter_manifest(m, ts, data)));
}
void save_package(const std::string &dir, Json::Value m,
                  const std::vector<HostTensor> &ts,
                  const std::vector<std::uint16_t> &data) {
  if (std::filesystem::exists(dir))
    throw std::runtime_error("adapter destination already exists: " + dir);
  auto temp = dir + ".tmp";
  if (std::filesystem::exists(temp))
    throw std::runtime_error("adapter staging directory already exists");
  std::filesystem::create_directories(temp);
  m["format_version"] = 1;
  m["method"] = "miss";
  if (m["dtype"] != "bfloat16") m["dtype"] = "float16";
  m["layout"] = "modulo_rank_zero_pad";
  m["kind"] = "inference_adapter";
  m["targets"] = Json::Value(Json::arrayValue);
  std::vector<llm_infer::WriteTensor> write;
  for (const auto &t : ts) {
    Json::Value v;
    v["name"] = tensor_name(t);
    v["layer"] = t.layer;
    v["target"] = target_names[t.target];
    v["in_features"] = t.in;
    v["shape"].append(t.out);
    v["shape"].append(t.rank);
    m["targets"].append(v);
    const auto n = size_t(t.out) * t.rank;
    if (t.offset + n > data.size())
      throw std::runtime_error("adapter data bounds");
    llm_infer::WriteTensor w{tensor_name(t), {t.out, t.rank}, m["dtype"] != "bfloat16", {}};
    w.bf16 = m["dtype"] == "bfloat16";
    w.data.resize(n * 2);
    std::memcpy(w.data.data(), data.data() + t.offset, n * 2);
    write.push_back(std::move(w));
  }
  m["content_version"] = content_version(m, data);
  llm_infer::write_pth(temp + "/adapter.pth", write);
  {
    std::ofstream f(temp + "/adapter.json");
    f << json(m) << '\n';
    if (!f)
      throw std::runtime_error("manifest write failed");
  }
  load_package(temp);
  std::filesystem::rename(temp, dir);
}
std::shared_ptr<const Package> load_package(const std::string &dir,
                                            size_t max_bytes) {
  const bool single = std::filesystem::is_regular_file(dir);
  const auto path = single ? dir : dir + "/adapter.pth";
  // Stream the archive: optimizer tensors must never be loaded into the RAM cache.
  auto archive = llm_infer::PthArchive::open(path, true);
  if (!archive.ok())
    throw std::runtime_error(archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  if (!records.ok())
    throw std::runtime_error(records.status().message());
  std::set<std::string> record_names;
  for (const auto &record : records.value())
    if (!record_names.insert(record.name).second)
      throw std::runtime_error("duplicate adapter/checkpoint tensor");
  auto p = std::make_shared<Package>();
  Json::CharReaderBuilder reader;
  std::string errors;
  bool training = false, legacy = false;
  if (single && archive.value().find_entry("archive/miss.json")) {
    const auto *entry = archive.value().find_entry("archive/miss.json");
    if (entry->uncompressed_size > (8ULL << 20))
      throw std::runtime_error("adapter manifest too large");
    auto bytes = archive.value().read_stored_entry(*entry);
    if (!bytes.ok()) throw std::runtime_error(bytes.status().message());
    std::istringstream f(std::string(bytes.value().begin(), bytes.value().end()));
    if (!Json::parseFromStream(reader, f, &p->manifest, &errors))
      throw std::runtime_error("invalid embedded MiSS manifest");
    const auto source = p->manifest.get("tensor_source", "D").asString();
    if (source != "D" && source != "master")
      throw std::runtime_error("unsupported adapter tensor source");
    training = source == "master";
    p->manifest.removeMember("tensor_source");
  } else if (single && std::filesystem::exists(
                          std::filesystem::path(dir).parent_path() / "checkpoint.json")) {
    // Legacy checkpoint metadata lives beside training.pth.
    const auto meta = std::filesystem::path(dir).parent_path() / "checkpoint.json";
    if (std::filesystem::file_size(meta) > (8ULL << 20))
      throw std::runtime_error("checkpoint metadata too large");
    std::ifstream f(meta);
    Json::Value checkpoint;
    if (!Json::parseFromStream(reader, f, &checkpoint, &errors) ||
        checkpoint["kind"] != "miss_training_checkpoint" ||
        checkpoint["tensor_digest"].asString() != fingerprint_file(dir))
      throw std::runtime_error("invalid legacy MiSS checkpoint");
    const auto &config = checkpoint["config"];
    Json::Value m;
    m["rank"] = config["rank"];
    m["alpha"] = config["alpha"];
    if (m["rank"].asInt() <= 0) throw std::runtime_error("invalid rank");
    m["scale"] = config["alpha"].asFloat() / m["rank"].asInt();
    m["base_fingerprint"] = config["base_fingerprint"];
    std::vector<HostTensor> ts;
    int channels = 0, hidden = 0;
    for (const auto &r : records.value()) {
      if (r.name.size() < 9 || r.name.substr(r.name.size()-9) != ".D.master") continue;
      if (r.shape.size() != 2) throw std::runtime_error("invalid master D shape");
      for (int i=0; i<TargetCount; ++i) {
        const std::string suffix = std::string(".") + target_names[i] + ".D.master";
        if (r.name.size() <= suffix.size() + 7 || r.name.compare(0,7,"blocks.") ||
            r.name.substr(r.name.size()-suffix.size()) != suffix) continue;
        auto layer = r.name.substr(7,r.name.size()-suffix.size()-7);
        if (layer.find_first_not_of("0123456789") != std::string::npos)
          throw std::runtime_error("invalid layer name");
        HostTensor t{std::stoi(layer),i,0,int(r.shape[0]),int(r.shape[1]),0};
        if (i == FfnKey) hidden = t.out; else channels = t.out;
        ts.push_back(t);
      }
    }
    for (auto &t : ts) {
      t.in = t.target == FfnValue ? hidden : channels;
      if (!t.in) throw std::runtime_error("legacy checkpoint lacks input widths; re-export with embedded metadata");
    }
    p->manifest = adapter_manifest(m,ts,{});
    training = legacy = true;
  } else {
    const auto meta = single ? std::filesystem::path(dir).parent_path() / "adapter.json"
                             : std::filesystem::path(dir) / "adapter.json";
    if (!std::filesystem::exists(meta))
      throw std::runtime_error("PTH has no embedded MiSS metadata; legacy checkpoints require sibling checkpoint.json (or upload it as a second file)");
    if (std::filesystem::file_size(meta) > (8ULL << 20))
      throw std::runtime_error("adapter manifest too large");
    std::ifstream f(meta);
    if (!Json::parseFromStream(reader,f,&p->manifest,&errors))
      throw std::runtime_error("invalid adapter manifest: " + errors);
  }
  const auto &m = p->manifest;
  if (m["format_version"] != 1 || m["method"] != "miss" ||
      (m["dtype"] != "float16" && m["dtype"] != "bfloat16") || m["layout"] != "modulo_rank_zero_pad" ||
      m["kind"] != "inference_adapter")
    throw std::runtime_error("unsupported adapter format");
  const bool bf16_storage = m["dtype"] == "bfloat16";
  int rank = m["rank"].asInt();
  p->scale = m["scale"].asFloat();
  p->base = m["base_fingerprint"].asString();
  p->version = m["content_version"].asString();
  if (rank <= 0 || rank > 1024 || !m["scale"].isNumeric() ||
      !std::isfinite(p->scale) || !m["alpha"].isNumeric() ||
      !std::isfinite(m["alpha"].asFloat()) || p->base.size() != 64 ||
      p->version.size() != 64 || !m["targets"].isArray() ||
      m["targets"].empty())
    throw std::runtime_error("invalid adapter metadata");
  if (records.value().size() != m["targets"].size() * (training ? 4 : 1) + (training ? 1 : 0))
    throw std::runtime_error("adapter tensor count mismatch");
  std::set<std::pair<int, int>> seen;
  for (const auto &v : m["targets"]) {
    HostTensor t;
    t.layer = v["layer"].asInt();
    t.in = v["in_features"].asInt();
    t.out = v["shape"][0].asInt();
    t.rank = v["shape"][1].asInt();
    t.offset = p->data.size();
    auto it = std::find(target_names.begin(), target_names.end(),
                        v["target"].asString());
    t.target = int(it - target_names.begin());
    if (t.target == TargetCount || t.layer < 0 || t.layer >= 1024 ||
        t.in <= 0 || t.in > (1 << 20) || t.out <= 0 || t.out > (1 << 20) ||
        t.rank != rank || !seen.emplace(t.layer, t.target).second)
      throw std::runtime_error("invalid/duplicate adapter target");
    if ((p->data.size() + size_t(t.out) * t.rank) * 2 > max_bytes)
      throw std::runtime_error("adapter RAM budget exceeded");
    const auto name = tensor_name(t);
    if (v["name"].asString() != name)
      throw std::runtime_error("adapter target name mismatch");
    auto r = std::find_if(records.value().begin(), records.value().end(),
                          [&](const auto &r) { return r.name == name + (training ? ".master" : ""); });
    if (r == records.value().end() ||
        r->dtype != (training ? llm_infer::TensorDType::kFloat32 : (bf16_storage ? llm_infer::TensorDType::kBFloat16 : llm_infer::TensorDType::kFloat16)) ||
        r->shape != std::vector<int64_t>{t.out, t.rank})
      throw std::runtime_error("adapter tensor shape/dtype mismatch");
    auto data =
        llm_infer::load_tensor_select(archive.value(), *r, bf16_storage, !bf16_storage, true);
    if (!data.ok())
      throw std::runtime_error(data.status().message());
    for (float x : data.value().values)
      if (!std::isfinite(x))
        throw std::runtime_error("nonfinite adapter weight");
    for (auto x : data.value().values_f16)
      if (!std::isfinite(llm_infer::f16_bits_to_float(x)))
        throw std::runtime_error("adapter weight overflows FP16");
    const auto &native = bf16_storage ? data.value().values_bf16 : data.value().values_f16;
    for (auto bits : native) {
      const float x = bf16_storage ? llm_infer::bf16_bits_to_float(bits) : llm_infer::f16_bits_to_float(bits);
      if (!std::isfinite(x) || std::abs(x) > 65504.0f)
        throw std::runtime_error("adapter cannot be represented by the FP16 inference runtime");
    }
    p->data.insert(p->data.end(), native.begin(), native.end());
    p->tensors.push_back(t);
  }
  if (legacy) {
    p->version = content_version(m, p->data);
    p->manifest["content_version"] = p->version;
  }
  if (content_version(m, p->data) != p->version)
    throw std::runtime_error("adapter content digest mismatch");
  return p;
}
AdapterHandle::AdapterHandle(std::shared_ptr<const Package> p, float s)
    : package(std::move(p)), scale(s), identity([&] {
        if (!package || !std::isfinite(s))
          throw std::invalid_argument("invalid adapter handle");
        if (s == 0)
          s = 0;
        std::ostringstream o;
        o << package->version << ':' << std::hexfloat << s;
        return o.str();
      }()) {}
AdapterCache::AdapterCache(size_t r, size_t g, size_t s)
    : ram_budget_(r), gpu_budget_(g), staging_budget_(s) {}
void AdapterCache::account_ram() {
  stats_.ram_bytes = 0;
  for (auto i = host_lifetimes_.begin(); i != host_lifetimes_.end();) {
    if (auto p = i->lock()) {
      stats_.ram_bytes += p->data.size() * 2;
      ++i;
    } else
      i = host_lifetimes_.erase(i);
  }
}
std::string AdapterCache::register_adapter(const std::string &id,
                                           const std::string &dir) {
  if (id.empty())
    throw std::invalid_argument("empty adapter id");
  std::lock_guard<std::mutex> lock(mutex_);
  account_ram();
  // Reserve payload capacity before reading disk; simultaneous registrations
  // must not temporarily allocate several complete adapters beyond the budget.
  auto p =
      load_package(dir, ram_budget_ - std::min(ram_budget_, stats_.ram_bytes));
  auto set_latest = [&] {
    auto &order = registration_order_[id];
    order.erase(std::remove(order.begin(), order.end(), p->version),
                order.end());
    order.push_back(p->version);
    latest_[id] = p->version;
  };
  auto existing = registry_.find(id);
  if (existing != registry_.end() && existing->second.count(p->version)) {
    ++stats_.ram_hits;
    set_latest();
    return p->version;
  }
  for (const auto &weak : host_lifetimes_)
    if (auto shared = weak.lock())
      if (shared->version == p->version) {
        registry_[id][p->version] = shared;
        set_latest();
        ++stats_.ram_hits;
        return p->version;
      }
  account_ram();
  if (p->data.size() * 2 >
      ram_budget_ - std::min(ram_budget_, stats_.ram_bytes))
    throw std::runtime_error("adapter RAM admission denied");
  registry_[id][p->version] = p;
  set_latest();
  host_lifetimes_.push_back(p);
  account_ram();
  return p->version;
}
std::shared_ptr<const AdapterHandle>
AdapterCache::resolve(const std::string &id, const std::string &version,
                      const float *scale) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto i = registry_.find(id);
  if (i == registry_.end())
    throw std::runtime_error("unknown adapter id");
  auto j = i->second.find(version.empty() ? latest_.at(id) : version);
  if (j == i->second.end())
    throw std::runtime_error("unknown adapter version");
  ++stats_.ram_hits;
  return std::make_shared<const AdapterHandle>(
      j->second, scale ? *scale : j->second->scale);
}
void AdapterCache::admit_gpu(size_t bytes) {
  if (bytes > gpu_budget_)
    throw std::runtime_error("adapter exceeds GPU budget");
  while (stats_.gpu_bytes > gpu_budget_ - bytes) {
    auto best = gpu_.end();
    for (auto i = gpu_.begin(); i != gpu_.end(); ++i)
      if (i->second.lease.use_count() == 1 &&
          (best == gpu_.end() || i->second.used < best->second.used))
        best = i;
    if (best == gpu_.end())
      throw std::runtime_error("adapter GPU admission denied: active leases");
    stats_.gpu_bytes -= best->second.lease->data.n * 2;
    gpu_.erase(best);
    ++stats_.evictions;
  }
}
std::shared_ptr<const GpuLease> AdapterCache::acquire(const AdapterHandle &h,
                                                      bool *cold) {
  // Serial admission/upload is also a single-flight barrier: only a fully
  // event-complete lease is published, concurrent misses never duplicate H2D.
  std::lock_guard<std::mutex> lock(mutex_);
  auto i = gpu_.find(h.package->version);
  if (i != gpu_.end()) {
    if (cold)
      *cold = false;
    ++stats_.gpu_hits;
    i->second.used = ++clock_;
    return i->second.lease;
  }
  if (cold)
    *cold = true;
  ++stats_.gpu_misses;
  const size_t bytes = h.package->data.size() * 2;
  if (bytes > staging_budget_)
    throw std::runtime_error("adapter exceeds pinned staging budget");
  admit_gpu(bytes);
  auto lease = std::make_shared<GpuLease>();
  lease->version = h.package->version;
  size_t free_bytes = 0, total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (bytes > free_bytes)
    throw std::runtime_error("adapter GPU OOM admission denied");
  half *allocation = nullptr;
  check(cudaMalloc(&allocation, bytes));
  lease->data.p = allocation;
  lease->data.n = h.package->data.size();
  void *pinned = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  try {
    check(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
    if (h.package->manifest["dtype"] == "bfloat16") {
      auto *target = static_cast<uint16_t *>(pinned);
      for (size_t i = 0; i < h.package->data.size(); ++i)
        target[i] = llm_infer::float_to_f16_bits(llm_infer::bf16_bits_to_float(h.package->data[i]));
    } else
      std::memcpy(pinned, h.package->data.data(), bytes);
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    check(cudaEventRecord(begin, stream));
    check(cudaMemcpyAsync(lease->data.p, pinned, bytes, cudaMemcpyHostToDevice,
                          stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms, begin, end));
    stats_.h2d_ms += ms;
    lease->h2d_ms = ms;
    for (const auto &t : h.package->tensors) {
      if (lease->blocks.size() <= size_t(t.layer))
        lease->blocks.resize(t.layer + 1);
      lease->blocks[t.layer][t.target] = {
          t.in, t.out, t.rank, 1, lease->data.p + t.offset, nullptr};
    }
  } catch (...) {
    if (stream)
      cudaStreamSynchronize(stream);
    if (end)
      cudaEventDestroy(end);
    if (begin)
      cudaEventDestroy(begin);
    if (stream)
      cudaStreamDestroy(stream);
    if (pinned)
      cudaFreeHost(pinned);
    throw;
  }
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cudaStreamDestroy(stream);
  cudaFreeHost(pinned);
  gpu_[lease->version] = {lease, ++clock_};
  stats_.gpu_bytes += bytes;
  stats_.gpu_peak_bytes = std::max(stats_.gpu_peak_bytes, stats_.gpu_bytes);
  ++stats_.uploads;
  return lease;
}
void AdapterCache::erase(const std::string &id, const std::string &version) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto i = registry_.find(id);
  if (i == registry_.end())
    return;
  if (version.empty()) {
    registry_.erase(i);
    latest_.erase(id);
    registration_order_.erase(id);
  } else {
    i->second.erase(version);
    auto &order = registration_order_[id];
    order.erase(std::remove(order.begin(), order.end(), version), order.end());
    if (i->second.empty()) {
      registry_.erase(i);
      latest_.erase(id);
      registration_order_.erase(id);
    } else if (latest_[id] == version)
      latest_[id] = order.back();
  }
  account_ram();
}
void AdapterCache::trim() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto i = gpu_.begin(); i != gpu_.end();)
    if (i->second.lease.use_count() == 1) {
      stats_.gpu_bytes -= i->second.lease->data.n * 2;
      i = gpu_.erase(i);
      ++stats_.evictions;
    } else
      ++i;
  account_ram();
}
CacheStats AdapterCache::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = stats_;
  result.ram_bytes = 0;
  for (const auto &weak : host_lifetimes_)
    if (auto p = weak.lock())
      result.ram_bytes += p->data.size() * 2;
  return result;
}
Json::Value AdapterCache::list() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Json::Value out(Json::arrayValue);
  for (const auto &i : registry_)
    for (const auto &j : i.second) {
      Json::Value v;
      v["id"] = i.first;
      v["version"] = j.first;
      v["manifest"] = j.second->manifest;
      out.append(v);
    }
  return out;
}
AdapterCache &adapter_cache() {
  auto budget = [](const char *name, size_t fallback) -> size_t {
    const char *value = std::getenv(name);
    if (!value)
      return fallback;
    std::string s(value);
    size_t used = 0;
    auto n = std::stoull(s, &used);
    if (s.empty() || s[0] == '-' || used != s.size() ||
        n > SIZE_MAX / (1ULL << 20))
      throw std::runtime_error(std::string("invalid budget: ") + name);
    return size_t(n) * (1ULL << 20);
  };
  static AdapterCache c(budget("RWKV_ADAPTER_RAM_MIB", 1ULL << 30),
                        budget("RWKV_ADAPTER_GPU_MIB", 512ULL << 20),
                        budget("RWKV_ADAPTER_STAGING_MIB", 128ULL << 20));
  return c;
}
std::string effective_model_key(const std::string &r, const AdapterHandle *a,
                                const std::string &s, bool fp32) {
  Json::Value v;
  v.append(r);
  v.append(a ? a->identity : "");
  v.append(s);
  v.append(fp32);
  auto b = json(v);
  Hash h;
  h.add(b.data(), b.size());
  return h.finish();
}
} // namespace rwkv7_miss
