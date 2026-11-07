#include "rwkv/io/adapter_layout.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
namespace rwkv7_miss {
const std::array<const char *, TargetCount> target_names = {
    "att.receptance.weight", "att.key.weight", "att.value.weight",
    "att.output.weight",     "ffn.key.weight", "ffn.value.weight"};
namespace {
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
std::string tensor_name(const HostTensor &t, const char *factor = "D") {
  return "blocks." + std::to_string(t.layer) + "." + target_names[t.target] +
         "." + factor;
}
} // namespace
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
  const bool nora = m["method"] == "nora";
  m["method"] = nora ? "nora" : "miss";
  if (m["dtype"] != "bfloat16") m["dtype"] = "float16";
  m["layout"] = nora ? "low_rank_ba" : "modulo_rank_zero_pad";
  if (nora)
    m["initialization"] = "nora_init";
  m["kind"] = "inference_adapter";
  m["targets"] = Json::Value(Json::arrayValue);
  for (const auto &t : ts) {
    Json::Value v;
    v["name"] = tensor_name(t, nora ? "B" : "D");
    if (nora) {
      v["a_name"] = tensor_name(t, "A");
      v["a_shape"].append(t.rank);
      v["a_shape"].append(t.in);
    }
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
  const bool nora = m["method"] == "nora";
  for (const auto &t : ts)
    rwkv_adapter_detail::components(
        t, nora, target_names,
        [&](const std::string &name, int rows, int cols, size_t offset) {
          size_t n = size_t(rows) * cols;
          if (offset > data.size() || n > data.size() - offset)
            throw std::runtime_error("adapter data bounds");
          llm_infer::WriteTensor w{
              name, {rows, cols}, m["dtype"] != "bfloat16", {}};
          w.bf16 = m["dtype"] == "bfloat16";
          w.data.resize(n * 2);
          std::memcpy(w.data.data(), data.data() + offset, n * 2);
          write.push_back(std::move(w));
        });
  llm_infer::write_pth(path, write, json(adapter_manifest(m, ts, data)),
                       nora ? "archive/nora.json" : "archive/miss.json");
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
  m = adapter_manifest(m, ts, data);
  save_adapter_pth(temp + "/adapter.pth", m, ts, data);
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
  const auto *nora_entry = archive.value().find_entry("archive/nora.json");
  const auto *miss_entry = archive.value().find_entry("archive/miss.json");
  if (nora_entry && miss_entry)
    throw std::runtime_error("ambiguous adapter manifests");
  if (single && (nora_entry || miss_entry)) {
    const auto *entry = nora_entry ? nora_entry : miss_entry;
    if (entry->uncompressed_size > (8ULL << 20))
      throw std::runtime_error("adapter manifest too large");
    auto bytes = archive.value().read_stored_entry(*entry);
    if (!bytes.ok()) throw std::runtime_error(bytes.status().message());
    std::istringstream f(std::string(bytes.value().begin(), bytes.value().end()));
    if (!Json::parseFromStream(reader, f, &p->manifest, &errors))
      throw std::runtime_error("invalid embedded adapter manifest");
    if ((nora_entry && p->manifest["method"] != "nora") ||
        (miss_entry && p->manifest["method"] != "miss"))
      throw std::runtime_error("adapter entry/method mismatch");
    const auto source =
        p->manifest.get("tensor_source", nora_entry ? "BA" : "D").asString();
    if (source != (nora_entry ? "BA" : "D") && source != "master")
      throw std::runtime_error("unsupported adapter tensor source");
    training = source == "master";
    p->manifest.removeMember("tensor_source");
  } else if (single &&
             std::filesystem::exists(std::filesystem::path(dir).parent_path() /
                                     "checkpoint.json")) {
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
    // Pre-BF16 checkpoints did not declare a training dtype.
    m["dtype"] = config.get("training_dtype", "float16");
    if (m["dtype"] != "float16" && m["dtype"] != "bfloat16")
      throw std::runtime_error("unsupported legacy MiSS training dtype");
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
      throw std::runtime_error(
          "PTH has no embedded adapter metadata; legacy checkpoints require "
          "sibling checkpoint.json (or upload it as a second file)");
    if (std::filesystem::file_size(meta) > (8ULL << 20))
      throw std::runtime_error("adapter manifest too large");
    std::ifstream f(meta);
    if (!Json::parseFromStream(reader,f,&p->manifest,&errors))
      throw std::runtime_error("invalid adapter manifest: " + errors);
  }
  const auto &m = p->manifest;
  const bool nora = m["method"] == "nora";
  if (m["format_version"] != 1 || (!nora && m["method"] != "miss") ||
      (m["dtype"] != "float16" && m["dtype"] != "bfloat16") ||
      m["layout"] != (nora ? "low_rank_ba" : "modulo_rank_zero_pad") ||
      m["kind"] != "inference_adapter")
    throw std::runtime_error("unsupported adapter format");
  if (nora && (m["initialization"] != "nora_init" ||
               m["alpha"].asFloat() != m["rank"].asInt() ||
               m["scale"].asFloat() != 1.0f))
    throw std::runtime_error("NoRA requires nora_init and alpha/rank = 1");
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
  if (records.value().size() !=
      m["targets"].size() * (nora ? 2 : 1) * (training ? 4 : 1) +
          (training ? 1 : 0))
    throw std::runtime_error("adapter tensor count mismatch");
  std::set<std::pair<int, int>> seen;
  for (const auto &v : m["targets"]) {
    HostTensor t;
    t.layer = v["layer"].asInt();
    t.in = v["in_features"].asInt();
    if (!v["shape"].isArray() || v["shape"].size() != 2)
      throw std::runtime_error("invalid adapter shape");
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
    t.a_offset = t.offset + size_t(t.out) * t.rank;
    const size_t count =
        size_t(t.out) * t.rank + (nora ? size_t(t.in) * t.rank : 0);
    if (p->data.size() > max_bytes / 2 ||
        count > max_bytes / 2 - p->data.size())
      throw std::runtime_error("adapter RAM budget exceeded");
    rwkv_adapter_detail::components(
        t, nora, target_names,
        [&](const std::string &name, int rows, int cols, size_t) {
          bool is_a = nora && name.back() == 'A';
          if (v[is_a ? "a_name" : "name"].asString() != name ||
              (is_a && (!v["a_shape"].isArray() || v["a_shape"].size() != 2 ||
                        v["a_shape"][0].asInt() != rows ||
                        v["a_shape"][1].asInt() != cols)))
            throw std::runtime_error("adapter target name/shape mismatch");
          auto r = std::find_if(records.value().begin(), records.value().end(),
                                [&](const auto &r) {
                                  return r.name ==
                                         name + (training ? ".master" : "");
                                });
          if (r == records.value().end() ||
              r->dtype !=
                  (training
                       ? llm_infer::TensorDType::kFloat32
                       : (bf16_storage ? llm_infer::TensorDType::kBFloat16
                                       : llm_infer::TensorDType::kFloat16)) ||
              r->shape != std::vector<int64_t>{rows, cols})
            throw std::runtime_error("adapter tensor shape/dtype mismatch");
          auto data = llm_infer::load_tensor_select(
              archive.value(), *r, bf16_storage, !bf16_storage, true);
          if (!data.ok())
            throw std::runtime_error(data.status().message());
          for (float x : data.value().values)
            if (!std::isfinite(x))
              throw std::runtime_error("nonfinite adapter weight");
          for (auto x : data.value().values_f16)
            if (!std::isfinite(llm_infer::f16_bits_to_float(x)))
              throw std::runtime_error("adapter weight overflows FP16");
          const auto &native =
              bf16_storage ? data.value().values_bf16 : data.value().values_f16;
          for (auto bits : native) {
            const float x = bf16_storage ? llm_infer::bf16_bits_to_float(bits)
                                         : llm_infer::f16_bits_to_float(bits);
            if (!std::isfinite(x) || std::abs(x) > 65504.0f)
              throw std::runtime_error("adapter cannot be represented by the "
                                       "FP16 inference runtime");
          }
          p->data.insert(p->data.end(), native.begin(), native.end());
        });
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
} // namespace rwkv7_miss
