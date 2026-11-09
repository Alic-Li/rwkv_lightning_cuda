#include "rwkv/io/fingerprint.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <json/json.h>
#include <memory>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace rwkv7_miss {
std::string fingerprint_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot fingerprint " + path);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA256 init");
  std::vector<char> b(4 << 20);
  while (f) {
    f.read(b.data(), b.size());
    if (EVP_DigestUpdate(ctx.get(), b.data(), f.gcount()) != 1)
      throw std::runtime_error("SHA256 update");
  }
  if (!f.eof())
    throw std::runtime_error("fingerprint read error");
  unsigned char d[32];
  unsigned int n = 0;
  if (EVP_DigestFinal_ex(ctx.get(), d, &n) != 1 || n != 32)
    throw std::runtime_error("SHA256 finalize");
  std::ostringstream o;
  for (auto x : d)
    o << std::hex << std::setw(2) << std::setfill('0') << int(x);
  return o.str();
}
void write_source_fingerprint(const std::string &source,
                              const std::string &output) {
  Json::Value m;
  m["format_version"] = 1;
  m["source_sha256"] = fingerprint_file(source);
  m["runtime_sha256"] = fingerprint_file(output);
  const auto path = output + ".source.json";
  {
    std::ofstream f(path + ".tmp");
    f << m;
    if (!f)
      throw std::runtime_error("source manifest write failed");
  }
  std::filesystem::rename(path + ".tmp", path);
}
std::string source_fingerprint(const std::string &path) {
  auto digest = fingerprint_file(path);
  std::ifstream f(path + ".source.json");
  if (!f)
    return digest;
  Json::Value m;
  f >> m;
  if (m["format_version"] != 1 || m["runtime_sha256"].asString() != digest ||
      m["source_sha256"].asString().size() != 64)
    throw std::runtime_error("invalid quantized source identity");
  return m["source_sha256"].asString();
}
} // namespace rwkv7_miss
