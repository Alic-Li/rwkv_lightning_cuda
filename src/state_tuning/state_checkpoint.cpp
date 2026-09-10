#include "rwkv_state_tuning.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pth_archive.hpp"
#include "pth_tensor.hpp"

namespace rwkv7_state_tuning {
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

std::vector<std::uint8_t> make_pickle(int layers, int heads, int n) {
  const std::int64_t lane64 = static_cast<std::int64_t>(heads) * n * n;
  if (lane64 > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("state tensor is too large");
  const int lane = static_cast<int>(lane64);
  std::vector<std::uint8_t> out{0x80, 2, '}', '('};
  for (int layer = 0; layer < layers; ++layer) {
    pickle_string(out, "blocks." + std::to_string(layer) + ".att.time_state");
    pickle_global(out, "torch._utils", "_rebuild_tensor_v2");
    out.push_back('(');
    out.push_back('(');
    pickle_string(out, "storage");
    pickle_global(out, "torch", "FloatStorage");
    pickle_string(out, std::to_string(layer));
    pickle_string(out, "cpu");
    pickle_int(out, lane);
    out.push_back('t');
    out.push_back('Q');
    pickle_int(out, 0);
    pickle_tuple3(out, heads, n, n);
    pickle_tuple3(out, n * n, n, 1);
    out.push_back(0x89);
    out.push_back('}');
    out.push_back('t');
    out.push_back('R');
  }
  out.push_back('u');
  out.push_back('.');
  return out;
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

void verify(const std::filesystem::path &path, int layers, int heads, int n) {
  auto archive = llm_infer::PthArchive::open(path.string());
  if (!archive.ok())
    throw std::runtime_error("checkpoint self-check: " +
                             archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  if (!records.ok() || static_cast<int>(records.value().size()) != layers)
    throw std::runtime_error("checkpoint self-check: tensor count");
  for (int layer = 0; layer < layers; ++layer) {
    const auto &record = records.value()[static_cast<std::size_t>(layer)];
    if (record.name != "blocks." + std::to_string(layer) + ".att.time_state" ||
        record.dtype != llm_infer::TensorDType::kFloat32 ||
        record.shape != std::vector<std::int64_t>{heads, n, n})
      throw std::runtime_error("checkpoint self-check: tensor metadata");
  }
}

} // namespace

void save_state_checkpoint_pth_host(const std::string &path, int layers,
                                    int heads, int n,
                                    const std::vector<float> &runtime) {
  if (path.empty() || layers <= 0 || heads <= 0 || n <= 0)
    throw std::invalid_argument("invalid checkpoint arguments");
  const std::size_t lane = static_cast<std::size_t>(heads) * n * n;
  if (runtime.size() != static_cast<std::size_t>(layers) * lane)
    throw std::invalid_argument("checkpoint element count mismatch");
  std::vector<Entry> entries;
  entries.push_back({"archive/data.pkl", make_pickle(layers, heads, n)});
  entries.push_back({"archive/byteorder", data_of("little")});
  for (int layer = 0; layer < layers; ++layer) {
    std::vector<float> pth(lane);
    const float *source =
        runtime.data() + static_cast<std::size_t>(layer) * lane;
    for (int h = 0; h < heads; ++h)
      for (int k = 0; k < n; ++k)
        for (int v = 0; v < n; ++v)
          pth[(static_cast<std::size_t>(h) * n + v) * n + k] =
              source[(static_cast<std::size_t>(h) * n + k) * n + v];
    std::vector<std::uint8_t> raw(pth.size() * sizeof(float));
    std::memcpy(raw.data(), pth.data(), raw.size());
    entries.push_back(
        {"archive/data/" + std::to_string(layer), std::move(raw)});
  }
  entries.push_back({"archive/version", data_of("3\n")});
  const std::filesystem::path target(path);
  const std::filesystem::path temporary = target.string() + ".tmp";
  write_zip(temporary, std::move(entries));
  verify(temporary, layers, heads, n);
  std::error_code error;
  std::filesystem::rename(temporary, target, error);
  if (error) {
    std::filesystem::remove(target, error);
    error.clear();
    std::filesystem::rename(temporary, target, error);
  }
  if (error)
    throw std::runtime_error("failed to install checkpoint: " +
                             error.message());
}

void save_state_checkpoint_pth(const std::string &path, cudaStream_t stream,
                               int layers, int heads, int n,
                               const float *runtime_time_state) {
  if (!runtime_time_state)
    throw std::invalid_argument("null device time_state");
  const std::size_t count = static_cast<std::size_t>(layers) * heads * n * n;
  std::vector<float> host(count);
  auto error =
      cudaMemcpyAsync(host.data(), runtime_time_state, count * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  save_state_checkpoint_pth_host(path, layers, heads, n, host);
}

} // namespace rwkv7_state_tuning
