#include "rwkv/io/pth_writer.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace llm_infer {
namespace {
void u16(std::vector<std::uint8_t> &out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void bytes(std::vector<std::uint8_t> &out, const void *data, std::size_t size) {
  const auto *p = static_cast<const std::uint8_t *>(data);
  out.insert(out.end(), p, p + size);
}

void string_bytes(std::vector<std::uint8_t> &out, const std::string &value) {
  bytes(out, value.data(), value.size());
}

std::uint32_t crc32(const std::vector<std::uint8_t> &data) {
  std::uint32_t crc = 0xffffffffu;
  for (std::uint8_t byte : data) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

void pickle_string(std::vector<std::uint8_t> &out, const std::string &value) {
  out.push_back('X');
  u32(out, static_cast<std::uint32_t>(value.size()));
  string_bytes(out, value);
}

void pickle_int(std::vector<std::uint8_t> &out, std::int32_t value) {
  if (value >= 0 && value <= 255) {
    out.push_back('K');
    out.push_back(static_cast<std::uint8_t>(value));
  } else if (value >= 0 && value <= 65535) {
    out.push_back('M');
    u16(out, static_cast<std::uint16_t>(value));
  } else {
    out.push_back('J');
    u32(out, static_cast<std::uint32_t>(value));
  }
}

void pickle_global(std::vector<std::uint8_t> &out, const char *module,
                   const char *name) {
  out.push_back('c');
  string_bytes(out, module);
  out.push_back('\n');
  string_bytes(out, name);
  out.push_back('\n');
}

void pickle_tuple3(std::vector<std::uint8_t> &out, int a, int b, int c) {
  out.push_back('(');
  pickle_int(out, a);
  pickle_int(out, b);
  pickle_int(out, c);
  out.push_back('t');
}

struct Entry {
  std::string name;
  std::vector<std::uint8_t> data;
  std::uint32_t crc = 0;
  std::uint32_t offset = 0;
};

void write_zip(const std::filesystem::path &path, std::vector<Entry> entries) {
  if (entries.size() > std::numeric_limits<std::uint16_t>::max())
    throw std::runtime_error("too many checkpoint entries");
  std::vector<std::uint8_t> out;
  for (auto &entry : entries) {
    if (entry.data.size() > std::numeric_limits<std::uint32_t>::max() ||
        out.size() > std::numeric_limits<std::uint32_t>::max() ||
        entry.name.size() > std::numeric_limits<std::uint16_t>::max())
      throw std::runtime_error("checkpoint exceeds non-ZIP64 limits");
    entry.crc = crc32(entry.data);
    entry.offset = static_cast<std::uint32_t>(out.size());
    u32(out, 0x04034b50u);
    u16(out, 20);
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u32(out, entry.crc);
    u32(out, static_cast<std::uint32_t>(entry.data.size()));
    u32(out, static_cast<std::uint32_t>(entry.data.size()));
    u16(out, static_cast<std::uint16_t>(entry.name.size()));
    u16(out, 0);
    string_bytes(out, entry.name);
    bytes(out, entry.data.data(), entry.data.size());
  }
  const auto central_offset = static_cast<std::uint32_t>(out.size());
  for (const auto &entry : entries) {
    u32(out, 0x02014b50u);
    u16(out, 20);
    u16(out, 20);
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u32(out, entry.crc);
    u32(out, static_cast<std::uint32_t>(entry.data.size()));
    u32(out, static_cast<std::uint32_t>(entry.data.size()));
    u16(out, static_cast<std::uint16_t>(entry.name.size()));
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u16(out, 0);
    u32(out, 0);
    u32(out, entry.offset);
    string_bytes(out, entry.name);
  }
  const auto central_size =
      static_cast<std::uint32_t>(out.size()) - central_offset;
  u32(out, 0x06054b50u);
  u16(out, 0);
  u16(out, 0);
  u16(out, static_cast<std::uint16_t>(entries.size()));
  u16(out, static_cast<std::uint16_t>(entries.size()));
  u32(out, central_size);
  u32(out, central_offset);
  u16(out, 0);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    throw std::runtime_error("failed to create " + path.string());
  file.write(reinterpret_cast<const char *>(out.data()),
             static_cast<std::streamsize>(out.size()));
  if (!file)
    throw std::runtime_error("failed to write " + path.string());
}

std::vector<std::uint8_t> data_of(const std::string &value) {
  return {value.begin(), value.end()};
}

} // namespace
void write_pth(const std::string &path,
               const std::vector<WriteTensor> &tensors,
               const std::string &miss_manifest) {
  std::vector<std::uint8_t> p{0x80, 2, '}', '('};
  std::vector<Entry> entries;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &t = tensors[i];
    size_t n = 1;
    for (int d : t.shape) {
      if (d <= 0 || n > size_t(INT32_MAX) / d)
        throw std::invalid_argument("invalid PTH shape");
      n *= d;
    }
    if (t.data.size() != n * (t.fp16 ? 2 : 4))
      throw std::invalid_argument("PTH data size mismatch");
    pickle_string(p, t.name);
    pickle_global(p, "torch._utils", "_rebuild_tensor_v2");
    p.push_back('(');
    p.push_back('(');
    pickle_string(p, "storage");
    pickle_global(p, "torch", t.fp16 ? "HalfStorage" : "FloatStorage");
    pickle_string(p, std::to_string(i));
    pickle_string(p, "cpu");
    pickle_int(p, n);
    p.push_back('t');
    p.push_back('Q');
    pickle_int(p, 0);
    p.push_back('(');
    for (int d : t.shape)
      pickle_int(p, d);
    p.push_back('t');
    std::vector<int> stride(t.shape.size());
    int step = 1;
    for (int j = int(stride.size()) - 1; j >= 0; --j) {
      stride[j] = step;
      step *= t.shape[j];
    }
    p.push_back('(');
    for (int d : stride)
      pickle_int(p, d);
    p.push_back('t');
    p.push_back(0x89);
    p.push_back('}');
    p.push_back('t');
    p.push_back('R');
    entries.push_back({"archive/data/" + std::to_string(i), t.data});
  }
  p.push_back('u');
  p.push_back('.');
  entries.push_back({"archive/data.pkl", p});
  entries.push_back({"archive/byteorder", data_of("little")});
  entries.push_back({"archive/version", data_of("3\n")});
  if (!miss_manifest.empty())
    entries.push_back({"archive/miss.json", data_of(miss_manifest)});
  const auto temp = path + ".tmp";
  write_zip(temp, std::move(entries));
  auto archive = PthArchive::open(temp);
  if (!archive.ok())
    throw std::runtime_error("PTH self-check failed");
  auto records = parse_pth_tensor_records(archive.value());
  if (!records.ok() || records.value().size() != tensors.size())
    throw std::runtime_error("PTH self-check count");
  std::filesystem::rename(temp, path);
}
} // namespace llm_infer
