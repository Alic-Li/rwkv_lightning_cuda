// CPU-only MiSS / NoRA merger. Reuses only the host PTH reader and SHA256
// helper. The ZIP64 writer here streams unchanged storages instead of
// materializing a second full model. No runtime, trainer, CUDA/HIP header or
// GPU library is used.
#include "rwkv/io/fingerprint.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/version.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <json/json.h>
#include <limits>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <zlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace llm_infer;
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
template <class T> T take(Result<T> result) {
  require(result.ok(), result.status().message());
  return std::move(result.value());
}
struct Hash {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free};
  Hash() {
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1,
            "SHA256 init failed");
  }
  void add(const void *p, size_t n) {
    require(EVP_DigestUpdate(ctx.get(), p, n) == 1, "SHA256 update failed");
  }
  std::string finish() {
    unsigned char bytes[32];
    unsigned n = 0;
    require(EVP_DigestFinal_ex(ctx.get(), bytes, &n) == 1 && n == 32,
            "SHA256 finalize failed");
    std::ostringstream out;
    for (auto b : bytes)
      out << std::hex << std::setw(2) << std::setfill('0') << int(b);
    return out.str();
  }
};
std::string prefix(const PthArchive &archive) {
  std::string result;
  std::set<std::string> names;
  for (const auto &e : archive.entries()) {
    require(names.insert(e.name).second, "duplicate ZIP entry: " + e.name);
    require(e.is_stored(), "compressed PTH entries are unsupported: " + e.name);
    const std::string suffix = "/data.pkl";
    if (e.name.size() > suffix.size() &&
        e.name.compare(e.name.size() - suffix.size(), suffix.size(), suffix) ==
            0) {
      require(result.empty(), "ambiguous PTH root");
      result = e.name.substr(0, e.name.size() - suffix.size());
    }
  }
  require(!result.empty(), "PTH data.pkl not found");
  if (const auto *entry = archive.find_entry(result + "/byteorder")) {
    auto data = take(archive.read_stored_entry(*entry));
    require(std::string(data.begin(), data.end()) == "little",
            "only little-endian PTH is supported");
  }
  return result;
}
std::map<std::string, TensorRecord> records(const PthArchive &archive) {
  std::map<std::string, TensorRecord> result;
  for (auto &t : take(parse_pth_tensor_records(archive)))
    require(result.emplace(t.name, t).second, "duplicate tensor: " + t.name);
  return result;
}
struct Target {
  std::string name;
  int in, out, rank;
  std::vector<float> b, a;
};
struct Adapter {
  std::string base;
  float scale;
  std::vector<Target> targets;
};
Adapter load_adapter(fs::path path, const std::string &method) {
  if (fs::is_directory(path))
    path /= fs::exists(path / "adapter.pth") ? "adapter.pth" : "training.pth";
  auto archive = take(PthArchive::open(path.string(), true));
  const auto root = prefix(archive);
  const auto tensors = records(archive);
  const auto *entry = archive.find_entry(root + "/" + method + ".json");
  require(
      !archive.find_entry(root +
                          (method == "miss" ? "/nora.json" : "/miss.json")),
      "Adapter method does not match --method (or manifests are ambiguous)");
  std::string metadata;
  if (entry) {
    require(entry->uncompressed_size <= (8ULL << 20),
            "Adapter metadata too large");
    auto data = take(archive.read_stored_entry(*entry));
    metadata.assign(data.begin(), data.end());
  } else {
    const auto sidecar = path.parent_path() / "adapter.json";
    require(fs::exists(sidecar) && fs::file_size(sidecar) <= (8ULL << 20),
            "Adapter requires embedded metadata or sibling adapter.json; "
            "re-export legacy checkpoints first");
    std::ifstream f(sidecar);
    metadata.assign(std::istreambuf_iterator<char>(f), {});
  }
  Json::Value m;
  Json::CharReaderBuilder reader;
  reader["rejectDupKeys"] = true;
  std::string errors;
  std::istringstream stream(metadata);
  require(Json::parseFromStream(reader, stream, &m, &errors),
          "invalid Adapter JSON: " + errors);
  const bool nora = method == "nora", bf16 = m["dtype"] == "bfloat16";
  require(m["format_version"] == 1 && m["method"] == method &&
              m["kind"] == "inference_adapter" &&
              (bf16 || m["dtype"] == "float16") &&
              m["layout"] == (nora ? "low_rank_ba" : "modulo_rank_zero_pad"),
          "unsupported Adapter format/method/layout/dtype");
  require(m["rank"].isInt() && m["rank"].asInt() > 0 &&
              m["rank"].asInt() <= 1024 && m["scale"].isNumeric() &&
              std::isfinite(m["scale"].asFloat()) && m["alpha"].isNumeric() &&
              std::isfinite(m["alpha"].asFloat()),
          "invalid Adapter rank/scale/alpha");
  const int rank = m["rank"].asInt();
  require(!nora || (m["initialization"] == "nora_init" &&
                    m["alpha"].asFloat() == rank && m["scale"].asFloat() == 1),
          "invalid NoRA initialization/scale");
  const std::string source =
      m.get("tensor_source", nora ? "BA" : "D").asString();
  const bool master = source == "master";
  require(master || source == (nora ? "BA" : "D"),
          "unsupported Adapter tensor_source");
  const auto version = m["content_version"].asString();
  Adapter result{m["base_fingerprint"].asString(), m["scale"].asFloat(), {}};
  require(version.size() == 64 && result.base.size() == 64 &&
              m["targets"].isArray() && !m["targets"].empty(),
          "invalid Adapter identity/targets");
  require(tensors.size() ==
              m["targets"].size() * (nora ? 2 : 1) * (master ? 4 : 1) +
                  (master ? 1 : 0),
          "Adapter tensor count mismatch");
  m.removeMember("tensor_source");
  auto hashed = m;
  hashed.removeMember("content_version");
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  const auto canonical = Json::writeString(writer, hashed);
  Hash hash;
  hash.add(canonical.data(), canonical.size());
  std::set<std::string> seen;
  const std::set<std::string> allowed{
      "att.receptance.weight", "att.key.weight", "att.value.weight",
      "att.output.weight",     "ffn.key.weight", "ffn.value.weight"};
  for (const auto &v : m["targets"]) {
    require(v["layer"].isInt() && v["layer"].asInt() >= 0 &&
                v["layer"].asInt() < 1024 &&
                allowed.count(v["target"].asString()) &&
                v["in_features"].isInt() && v["shape"].isArray() &&
                v["shape"].size() == 2 && v["shape"][0].isInt() &&
                v["shape"][1] == rank,
            "invalid Adapter target");
    Target t{"blocks." + std::to_string(v["layer"].asInt()) + "." +
                 v["target"].asString(),
             v["in_features"].asInt(),
             v["shape"][0].asInt(),
             rank,
             {},
             {}};
    require(t.in > 0 && t.in <= (1 << 20) && t.out > 0 && t.out <= (1 << 20) &&
                seen.insert(t.name).second,
            "invalid/duplicate target: " + t.name);
    auto load = [&](const char *factor, int rows, int cols) {
      const auto name = t.name + "." + factor;
      const bool a = factor[0] == 'A';
      require(v[a ? "a_name" : "name"] == name, "Adapter factor name mismatch");
      if (a)
        require(v["a_shape"].isArray() && v["a_shape"].size() == 2 &&
                    v["a_shape"][0] == rows && v["a_shape"][1] == cols,
                "NoRA A shape mismatch");
      const auto it = tensors.find(name + (master ? ".master" : ""));
      require(it != tensors.end() &&
                  it->second.shape == std::vector<int64_t>{rows, cols} &&
                  it->second.dtype == (master ? TensorDType::kFloat32
                                              : (bf16 ? TensorDType::kBFloat16
                                                      : TensorDType::kFloat16)),
              "missing or mismatched factor: " + name);
      auto data =
          take(load_tensor_select(archive, it->second, bf16, !bf16, true));
      // Checkpoint masters are rounded to the declared forward dtype, exactly
      // as adapter-final.pth export; optimizer and initial State are not
      // merged.
      const auto &bits = bf16 ? data.values_bf16 : data.values_f16;
      hash.add(bits.data(), bits.size() * 2);
      for (size_t i = 0; i < bits.size(); ++i) {
        require(std::isfinite(data.values[i]),
                "nonfinite Adapter factor: " + name);
        data.values[i] =
            bf16 ? bf16_bits_to_float(bits[i]) : f16_bits_to_float(bits[i]);
        require(std::isfinite(data.values[i]),
                "Adapter factor overflow: " + name);
      }
      return std::move(data.values);
    };
    t.b = load(nora ? "B" : "D", t.out, rank);
    if (nora)
      t.a = load("A", rank, t.in);
    result.targets.push_back(std::move(t));
  }
  require(hash.finish() == version, "Adapter content digest mismatch");
  return result;
}

// Stored ZIP64, including valid CRCs, readable by torch.load and unzip. Each
// entry is streamed, with its local CRC patched before publishing the archive.
class ZipWriter {
  struct Entry {
    std::string name;
    uint64_t offset, size;
    uint32_t crc;
  };
  std::ofstream file;
  std::vector<Entry> entries;
  void integer(uint64_t value, unsigned n) {
    for (unsigned i = 0; i < n; ++i)
      file.put(char(value >> (8 * i)));
  }

public:
  explicit ZipWriter(const fs::path &path) : file(path, std::ios::binary) {
    require(bool(file), "cannot create output staging file");
    file.exceptions(std::ios::badbit | std::ios::failbit);
  }
  template <class Producer> void add(const PthEntry &source, Producer produce) {
    require(source.name.size() <= 65535, "ZIP entry name too long");
    Entry e{source.name, uint64_t(file.tellp()), source.uncompressed_size, 0};
    integer(0x04034b50, 4);
    integer(45, 2);
    integer(0x800, 2);
    integer(0, 2);
    integer(0, 4);
    integer(0, 4);
    integer(0xffffffff, 4);
    integer(0xffffffff, 4);
    integer(e.name.size(), 2);
    integer(20, 2);
    file.write(e.name.data(), e.name.size());
    integer(1, 2);
    integer(16, 2);
    integer(e.size, 8);
    integer(e.size, 8);
    uint64_t written = 0;
    produce([&](const uint8_t *data, size_t size) {
      require(size <= e.size - written, "ZIP payload exceeds declared size");
      written += size;
      while (size) {
        const size_t n = std::min<size_t>(size, 4 << 20);
        e.crc = uint32_t(crc32(e.crc, data, uInt(n)));
        file.write(reinterpret_cast<const char *>(data), n);
        data += n;
        size -= n;
      }
    });
    require(written == e.size, "ZIP payload shorter than declared size");
    auto end = file.tellp();
    file.seekp(e.offset + 14);
    integer(e.crc, 4);
    file.seekp(end);
    entries.push_back(std::move(e));
  }
  void finish() {
    const uint64_t start = uint64_t(file.tellp());
    for (const auto &e : entries) {
      integer(0x02014b50, 4);
      integer(45, 2);
      integer(45, 2);
      integer(0x800, 2);
      integer(0, 2);
      integer(0, 4);
      integer(e.crc, 4);
      integer(0xffffffff, 4);
      integer(0xffffffff, 4);
      integer(e.name.size(), 2);
      integer(28, 2);
      integer(0, 2);
      integer(0, 2);
      integer(0, 2);
      integer(0, 4);
      integer(0xffffffff, 4);
      file.write(e.name.data(), e.name.size());
      integer(1, 2);
      integer(24, 2);
      integer(e.size, 8);
      integer(e.size, 8);
      integer(e.offset, 8);
    }
    const uint64_t end = uint64_t(file.tellp());
    integer(0x06064b50, 4);
    integer(44, 8);
    integer(45, 2);
    integer(45, 2);
    integer(0, 4);
    integer(0, 4);
    integer(entries.size(), 8);
    integer(entries.size(), 8);
    integer(end - start, 8);
    integer(start, 8);
    integer(0x07064b50, 4);
    integer(0, 4);
    integer(end, 8);
    integer(1, 4);
    integer(0x06054b50, 4);
    integer(0, 4);
    integer(0xffffffff, 4);
    integer(0xffffffff, 4);
    integer(0xffffffff, 4);
    integer(0, 2);
    file.close();
  }
};
struct Staging {
  fs::path dir;
  explicit Staging(const fs::path &output) {
    const auto parent = output.parent_path();
    require(parent.empty() || fs::is_directory(parent),
            "output parent directory does not exist: " + parent.string());
    std::random_device random;
    std::error_code last_error;
    for (int i = 0; i < 100; ++i) {
      auto candidate = output;
      candidate += ".merge-" + std::to_string(random());
      if (fs::create_directory(candidate, last_error)) {
        dir = candidate;
        return;
      }
    }
    throw std::runtime_error(
        "cannot reserve output staging directory" +
        (last_error ? ": " + last_error.message() : ""));
  }
  ~Staging() {
    std::error_code error;
    fs::remove_all(dir, error);
  }
};

std::pair<uint64_t, uint64_t> byte_span(const TensorRecord &w) {
  require(w.shape.size() == w.stride.size(), "invalid shape/stride: " + w.name);
  uint64_t last = w.storage_offset;
  for (size_t i = 0; i < w.shape.size(); ++i) {
    require(w.shape[i] >= 0 && w.stride[i] >= 0,
            "invalid tensor view: " + w.name);
    if (!w.shape[i])
      return {0, 0};
    const uint64_t dim = uint64_t(w.shape[i] - 1),
                   stride = uint64_t(w.stride[i]);
    require(!dim || stride <= (UINT64_MAX - last) / dim,
            "tensor view overflow: " + w.name);
    last += dim * stride;
  }
  const auto width = dtype_size_bytes(w.dtype);
  require(last < w.storage_size && w.storage_size <= UINT64_MAX / width,
          "tensor view exceeds storage: " + w.name);
  return {w.storage_offset * width, (last + 1) * width};
}
struct Patch {
  const Target *target;
  uint64_t begin, end;
};

float read_value(const uint8_t *p, TensorDType dtype) {
  if (dtype == TensorDType::kFloat32) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
  }
  uint16_t bits;
  std::memcpy(&bits, p, 2);
  return dtype == TensorDType::kBFloat16 ? bf16_bits_to_float(bits)
                                         : f16_bits_to_float(bits);
}
// IEEE round-to-nearest, ties-to-even, including FP16 subnormals. The legacy
// reader's float_to_f16_bits rounds ties away and is unsuitable for new
// weights.
uint16_t half_rne(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint32_t sign = (bits >> 16) & 0x8000, magnitude = bits & 0x7fffffff;
  const unsigned exponent = magnitude >> 23;
  if (exponent >= 143)
    return uint16_t(sign | 0x7c00);
  if (exponent >= 113)
    return uint16_t(
        sign |
        ((magnitude + 0xfff + ((magnitude >> 13) & 1) - 0x38000000) >> 13));
  if (exponent < 102)
    return uint16_t(sign);
  const unsigned shift = 126 - exponent;
  const uint32_t mantissa = (magnitude & 0x7fffff) | 0x800000;
  uint32_t rounded = mantissa >> shift;
  const uint32_t remainder = mantissa & ((1u << shift) - 1),
                 midpoint = 1u << (shift - 1);
  rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1));
  return uint16_t(sign | rounded);
}
bool write_value(uint8_t *p, TensorDType dtype, float value) {
  if (!std::isfinite(value))
    return false;
  if (dtype == TensorDType::kFloat32) {
    std::memcpy(p, &value, 4);
    return true;
  }
  const auto bits = dtype == TensorDType::kBFloat16 ? float_to_bf16_bits(value)
                                                    : half_rne(value);
  std::memcpy(p, &bits, 2);
  return std::isfinite(read_value(p, dtype));
}
void merge(std::vector<uint8_t> &storage, const TensorRecord &weight,
           const Target &t, float scale) {
  if (scale == 0)
    return; // Preserve original bits for an explicitly disabled delta.
  const auto width = dtype_size_bytes(weight.dtype);
  auto *base = storage.data() + weight.storage_offset * width;
  int bad = 0;
#pragma omp parallel for reduction(| : bad) schedule(static)
  for (int row = 0; row < t.out; ++row) {
    // No full-size FP32 base/delta copy: one row per CPU thread.
    std::vector<float> delta(t.in, 0);
    if (t.a.empty()) {
      for (int col = 0; col < t.in; ++col)
        delta[col] = t.b[size_t(row) * t.rank + col % t.rank];
    } else {
      for (int k = 0; k < t.rank; ++k) {
        const float b = t.b[size_t(row) * t.rank + k];
        const float *a = t.a.data() + size_t(k) * t.in;
        // SIMD requires OpenMP 4.0; MSVC's default /openmp supports 2.0.
#if defined(_OPENMP) && _OPENMP >= 201307
#pragma omp simd
#endif
        for (int col = 0; col < t.in; ++col)
          delta[col] += b * a[col];
      }
    }
    for (int col = 0; col < t.in; ++col) {
      auto *p = base + (size_t(row) * t.in + col) * width;
      const float value = read_value(p, weight.dtype) + scale * delta[col];
      if (!write_value(p, weight.dtype, value))
        bad = 1;
    }
  }
  require(!bad, "merged weight is nonfinite or overflows its original dtype: " +
                    t.name);
}
void help() {
  std::cout
      << "Usage: rwkv_peft_weight_merge --method miss|nora --model BASE.pth\n"
         "       --adapter ADAPTER.pth --output MERGED.pth [--scale FLOAT] "
         "[--threads N]\n"
         "CPU only. --method is case-insensitive. Scale defaults to Adapter "
         "metadata;\n"
         "--scale overrides that effective scale (0 preserves original "
         "weights).\n"
         "MiSS: W[o,i] += scale * D[o,i % rank]. NoRA: W += scale * B @ A.\n"
         "Accepts exported PTH, package directory, or current embedded "
         "training.pth.\n"
         "Checkpoint masters use the declared export dtype. Preserves base "
         "dtypes;\n"
         "requires matching base SHA256. Existing output is never "
         "overwritten.\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && rwkv::is_version_flag(argv[1]))
    return rwkv::print_version("rwkv_peft_weight_merge");
  try {
    uint16_t endian = 1;
    require(*reinterpret_cast<uint8_t *>(&endian) == 1,
            "little-endian host required");
    std::map<std::string, std::string> args;
    const std::set<std::string> options{"--method", "--model", "--adapter",
                                        "--output", "--scale", "--threads"};
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help" || key == "-h") {
        help();
        return 0;
      }
      require(options.count(key) && i + 1 < argc,
              "unknown option or missing value: " + key);
      require(args.emplace(key, argv[++i]).second, "duplicate option: " + key);
    }
    for (auto key : {"--method", "--model", "--adapter", "--output"})
      require(args.count(key) && !args[key].empty(),
              std::string("required: ") + key);
    std::string method = args.at("--method");
    for (auto &c : method)
      if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';
    require(method == "miss" || method == "nora",
            "--method must be miss or nora");
    if (args.count("--threads")) {
      size_t end = 0;
      int n = 0;
      try {
        n = std::stoi(args.at("--threads"), &end);
      } catch (const std::exception &) {
        throw std::runtime_error("invalid --threads");
      }
      require(end == args.at("--threads").size() && n > 0 && n <= 1024,
              "invalid --threads");
#ifdef _OPENMP
      omp_set_num_threads(n);
#else
      require(n == 1, "this build has no OpenMP; --threads must be 1");
#endif
    }
    const fs::path output = args.at("--output");
    require(!fs::exists(fs::symlink_status(output)),
            "output already exists: " + output.string());
    require(output.extension() == ".pth", "output must end in .pth");
    auto adapter = load_adapter(args.at("--adapter"), method);
    float scale = adapter.scale;
    if (args.count("--scale")) {
      size_t end = 0;
      try {
        scale = std::stof(args.at("--scale"), &end);
      } catch (const std::exception &) {
        throw std::runtime_error("invalid --scale");
      }
      require(end == args.at("--scale").size() && std::isfinite(scale),
              "invalid --scale");
    }
    std::cout << "Checking base SHA256..." << std::endl;
    require(
        rwkv7_miss::fingerprint_file(args.at("--model")) == adapter.base,
        "base fingerprint mismatch: use the exact unquantized training base");
    auto base = take(PthArchive::open(args.at("--model"), true));
    const auto root = prefix(base);
    const auto weights = records(base);
    std::map<std::string, std::vector<Patch>> replacements;
    for (const auto &t : adapter.targets) {
      const auto it = weights.find(t.name);
      require(it != weights.end(), "base target missing: " + t.name);
      const auto &w = it->second;
      require(w.shape == std::vector<int64_t>{t.out, t.in} &&
                  w.stride == std::vector<int64_t>{t.in, 1},
              "base target must be contiguous [out,in]: " + t.name);
      const uint64_t count = uint64_t(t.out) * t.in,
                     width = dtype_size_bytes(w.dtype);
      require(w.storage_offset <= w.storage_size &&
                  count <= w.storage_size - w.storage_offset &&
                  w.storage_size <= std::numeric_limits<size_t>::max() / width,
              "invalid base storage bounds: " + t.name);
      const auto span = byte_span(w);
      // Packed bases commonly put disjoint tensor views in one storage.
      // Reject overlapping aliases rather than silently changing another
      // weight.
      for (const auto &[name, other] : weights) {
        if (name == t.name || other.storage_key != w.storage_key)
          continue;
        const auto other_span = byte_span(other);
        require(span.second <= other_span.first ||
                    other_span.second <= span.first,
                "target overlaps another tensor: " + t.name + " / " + name);
      }
      const auto storage = root + "/data/" + w.storage_key;
      const auto *e = base.find_entry(storage);
      require(e && e->uncompressed_size == w.storage_size * width,
              "base storage size mismatch: " + t.name);
      replacements[storage].push_back({&t, span.first, span.second});
    }
    for (auto &[name, patches] : replacements)
      std::sort(
          patches.begin(), patches.end(),
          [](const Patch &a, const Patch &b) { return a.begin < b.begin; });
    Staging staging(output);
    const auto temp = staging.dir / "model.pth";
    ZipWriter zip(temp);
    size_t merged = 0;
    std::cout << "Merging " << method << ", scale=" << scale
              << ", targets=" << adapter.targets.size() << std::endl;
    std::vector<uint8_t> buffer(4 << 20);
    for (const auto &entry : base.entries()) {
      zip.add(entry, [&](auto emit) {
        uint64_t at = 0;
        auto copy_until = [&](uint64_t end) {
          while (at < end) {
            const auto n = size_t(std::min<uint64_t>(buffer.size(), end - at));
            const auto status =
                base.read_stored_entry_range(entry, at, buffer.data(), n);
            require(status.ok_status(), status.message());
            emit(buffer.data(), n);
            at += n;
          }
        };
        auto it = replacements.find(entry.name);
        if (it != replacements.end())
          for (const auto &patch : it->second) {
            copy_until(patch.begin);
            const auto &t = *patch.target;
            std::vector<uint8_t> data(size_t(patch.end - patch.begin));
            const auto status = base.read_stored_entry_range(
                entry, patch.begin, data.data(), data.size());
            require(status.ok_status(), status.message());
            auto local = weights.at(t.name);
            local.storage_offset = 0;
            merge(data, local, t, scale);
            emit(data.data(), data.size());
            at = patch.end;
            std::cout << "[" << ++merged << "/" << adapter.targets.size()
                      << "] " << t.name << std::endl;
          }
        copy_until(entry.uncompressed_size);
      });
    }
    zip.finish();
    auto check = take(PthArchive::open(temp.string(), true));
    require(records(check).size() == weights.size(),
            "output PTH tensor self-check failed");
    // Same filesystem, atomic no-replace publication; a concurrent creator is
    // never overwritten. The staging hard link is removed by the RAII guard.
    fs::create_hard_link(temp, output);
    std::cout << "Saved: " << output.string()
              << " (CPU only, original dtypes, ZIP64)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rwkv_peft_weight_merge: " << error.what() << '\n';
    return 1;
  }
}
