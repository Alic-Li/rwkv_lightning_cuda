#include "format.hpp"
#include "rwkv/io/pth_writer.hpp"
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>
namespace rwkv_bf16_miss {
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
std::string tensor_name(const HostTensor &t) {
  return "blocks." + std::to_string(t.layer) + "." + target_names[t.target] +
         ".D";
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
  m["method"] = "miss";
  m["dtype"] = "bfloat16";
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
    llm_infer::WriteTensor w{tensor_name(t), {t.out, t.rank}, false, {}};
    w.bf16 = true;
    w.data.resize(n * 2);
    std::memcpy(w.data.data(), data.data() + t.offset, n * 2);
    write.push_back(std::move(w));
  }
  llm_infer::write_pth(path, write, json(adapter_manifest(m, ts, data)));
}
} // namespace rwkv_bf16_miss
