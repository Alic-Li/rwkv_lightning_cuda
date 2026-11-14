#include "rwkv_quantized.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include "rwkv/io/pth_tensor.hpp"
#include <limits>
#include <unordered_set>

namespace llm_infer {
namespace {

constexpr std::array<char, 8> kMagic{{'R', 'W', 'K', 'V', 'Q', '\0', '\1', '\0'}};
constexpr std::uint32_t kVersion = 1;

template <typename T>
bool read_value(std::ifstream& file, T* value) {
  file.read(reinterpret_cast<char*>(value), sizeof(T));
  return static_cast<bool>(file);
}

template <typename T>
void write_value(std::ofstream& file, T value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

bool valid_dtype(std::uint8_t value) {
  return value <= 2 || value == 4 || value == 6;
}

std::uint64_t checked_numel(const std::vector<std::int64_t>& shape) {
  std::uint64_t result = 1;
  for (std::int64_t dim : shape) {
    if (dim < 0 || (dim != 0 && result > std::numeric_limits<std::uint64_t>::max() /
                                      static_cast<std::uint64_t>(dim))) {
      return 0;
    }
    result *= static_cast<std::uint64_t>(dim);
  }
  return result;
}

std::uint64_t data_bytes(const QuantizedTensorRecord& record) {
  if (asym_bits(record.dtype)) {
    if (record.shape.size() != 2) return UINT64_MAX;
    return asym_weight_bytes(record.shape[0], record.shape[1], asym_bits(record.dtype));
  }
  if (record.dtype == QuantizedDType::kInt8) return record.numel;

  return record.numel * 2;
}

bool is_integer_quantized(QuantizedDType dtype) {
  return dtype == QuantizedDType::kInt8 || asym_bits(dtype);
}

}  // namespace

std::uint64_t asym_weight_bytes(std::uint64_t rows, std::uint64_t cols, int bits) {
  if (bits != 4 && bits != 6) return UINT64_MAX;
  const auto groups = cols / 32 + (cols % 32 != 0);
  const std::uint64_t bytes = 16 + 4 * (bits - 4);
  if (groups > UINT64_MAX / bytes || (groups && rows > UINT64_MAX / (groups * bytes)))
    return UINT64_MAX;
  return rows * groups * bytes;
}

Result<std::vector<std::uint16_t>> expand_asym_metadata(
    const QuantizedTensorRecord& record, const std::vector<std::uint16_t>& metadata) {
  if (!asym_bits(record.dtype) || record.shape.size() != 2 || record.quant_group_size != 32 ||
      record.shape[0] <= 0 || record.shape[1] <= 0)
    return Status::error("invalid SB6 metadata shape");
  const std::uint64_t rows = record.shape[0], cols = record.shape[1];
  const auto groups = cols / 32 + (cols % 32 != 0), superblocks = cols / 256 + (cols % 256 != 0);
  if (superblocks > UINT64_MAX / 8 || rows > UINT64_MAX / (superblocks * 8) ||
      metadata.size() != rows * superblocks * 8 || rows > SIZE_MAX / groups / 2)
    return Status::error("invalid SB6 metadata length");
  std::vector<std::uint16_t> expanded(rows * groups * 2);
  for (std::uint64_t n = 0; n < rows; ++n) for (std::uint64_t sb = 0; sb < superblocks; ++sb) {
    const auto* m = metadata.data() + (n * superblocks + sb) * 8;
    const float d = f16_bits_to_float(m[0]), dm = f16_bits_to_float(m[1]);
    if (!std::isfinite(d) || d < 0 || !std::isfinite(dm)) return Status::error("invalid SB6 base");
    std::uint64_t qs = 0, qm = 0;
    for (int j = 0; j < 3; ++j) {
      qs |= std::uint64_t(m[2 + j]) << (16 * j);
      qm |= std::uint64_t(m[5 + j]) << (16 * j);
    }
    for (int g = 0; g < 8 && sb * 8 + g < groups; ++g) {
      // Multiplication of FP16 by a six-bit integer is exact in FP32.
      // Round once to FP16, ties-to-even, for the runtime half2(s,m).
      const auto round_half = [](float x) {
        auto h = float_to_f16_bits(x);
        auto mag = static_cast<std::uint16_t>(h & 0x7fff);
        const float a = std::fabs(x);
        if (!mag && a > 0x1p-25f) mag = 1;
        if ((mag & 1) && mag < 0x7c00 && a ==
            (f16_bits_to_float(mag) + f16_bits_to_float(mag - 1)) * .5f) --mag;
        return static_cast<std::uint16_t>((h & 0x8000) | mag);
      };
      const auto s = round_half(d * ((qs >> (6 * g)) & 63));
      const auto b = round_half(dm * ((qm >> (6 * g)) & 63));
      if (!std::isfinite(f16_bits_to_float(s)) || !std::isfinite(f16_bits_to_float(b)))
        return Status::error("expanded SB6 parameter overflow");
      const auto i = (n * groups + sb * 8 + g) * 2;
      expanded[i] = s; expanded[i + 1] = b;
    }
  }
  return expanded;
}

bool is_quantized_archive(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::array<char, 8> magic{};
  return static_cast<bool>(file.read(magic.data(), magic.size())) && magic == kMagic;
}

Result<QuantizedArchive> QuantizedArchive::open(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return Status::error("failed to open quantized model: " + path);
  const std::streamoff end = file.tellg();
  if (end < 0) return Status::error("failed to determine quantized model size: " + path);
  const std::uint64_t size = static_cast<std::uint64_t>(end);
  file.seekg(0, std::ios::beg);

  std::array<char, 8> magic{};
  std::uint32_t version = 0;
  std::uint32_t count = 0;
  if (!file.read(magic.data(), magic.size()) || !read_value(file, &version) || !read_value(file, &count)) {
    return Status::error("truncated quantized model header");
  }
  if (magic != kMagic || version != kVersion) {
    return Status::error("unsupported quantized model format");
  }

  QuantizedArchive archive;
  archive.path_ = path;
  archive.file_size_ = size;
  archive.records_.reserve(count);
  std::unordered_set<std::string> names;
  for (std::uint32_t tensor_index = 0; tensor_index < count; ++tensor_index) {
    std::uint32_t name_len = 0;
    std::uint8_t dtype = 0;
    std::uint8_t rank = 0;
    std::uint16_t reserved = 0;
    std::uint64_t numel = 0;
    if (!read_value(file, &name_len) || !read_value(file, &dtype) || !read_value(file, &rank) ||
        !read_value(file, &reserved) || !read_value(file, &numel)) {
      return Status::error("truncated quantized tensor header");
    }
    if (dtype == 3 || dtype == 5)
      return Status::error("retired quantized dtype; re-export as W8A16, SB6 W4 or SB6 W6");
    if (!valid_dtype(dtype) || rank > 16 || name_len == 0 || name_len > (1u << 20) ||
        (dtype >= static_cast<std::uint8_t>(QuantizedDType::kInt8) && rank == 0)) {
      return Status::error("invalid quantized tensor metadata");
    }
    QuantizedTensorRecord record;
    record.name.resize(name_len);
    file.read(record.name.data(), static_cast<std::streamsize>(name_len));
    if (!file) return Status::error("truncated quantized tensor name");
    record.dtype = static_cast<QuantizedDType>(dtype);
    record.quant_group_size = asym_bits(record.dtype) ? reserved : 0;
    if (!names.insert(record.name).second) {
      return Status::error("duplicate quantized tensor: " + record.name);
    }
    record.numel = numel;
    record.shape.resize(rank);
    for (std::uint8_t dimension_index = 0; dimension_index < rank; ++dimension_index) {
      if (!read_value(file, &record.shape[dimension_index]) || record.shape[dimension_index] < 0) {
        return Status::error("invalid quantized tensor shape: " + record.name);
      }
    }
    if (checked_numel(record.shape) != record.numel) {
      return Status::error("quantized tensor element count mismatch: " + record.name);
    }
    if (!is_integer_quantized(record.dtype) &&
        record.numel > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint16_t)) {
      return Status::error("quantized tensor is too large: " + record.name);
    }
    record.data_offset = static_cast<std::uint64_t>(file.tellg());
    const std::uint64_t bytes = data_bytes(record);
    if (bytes > size - std::min<std::uint64_t>(size, record.data_offset) ||
        record.data_offset > size) {
      return Status::error("quantized tensor data is out of bounds: " + record.name);
    }
    file.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!file) return Status::error("truncated quantized tensor data: " + record.name);
    if (is_integer_quantized(record.dtype)) {
      std::uint64_t scale_count = 0;
      std::uint64_t expected_scales = static_cast<std::uint64_t>(record.shape.front());

      if (asym_bits(record.dtype)) {
        if (record.shape.size() != 2 || record.shape[0] <= 0 || record.shape[1] <= 0 || reserved != 32)
          return Status::error("invalid SB6 tensor metadata: " + record.name);
        const std::uint64_t sb = record.shape[1] / 256 + (record.shape[1] % 256 != 0);
        if (sb > UINT64_MAX / 8 || expected_scales > UINT64_MAX / (sb * 8))
          return Status::error("SB6 metadata overflow");
        expected_scales *= sb * 8;
      }
      if (!read_value(file, &scale_count) || scale_count != expected_scales) {
        return Status::error("invalid quantized scale count: " + record.name);
      }
      record.scale_count = scale_count;
      if (scale_count > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint16_t)) {
        return Status::error("quantized scale tensor is too large: " + record.name);
      }
      record.scale_offset = static_cast<std::uint64_t>(file.tellg());
      const std::uint64_t scale_bytes = scale_count * sizeof(std::uint16_t);
      if (record.scale_offset > size || scale_bytes > size - record.scale_offset) {
        return Status::error("quantized scale data is out of bounds: " + record.name);
      }
      file.seekg(static_cast<std::streamoff>(scale_bytes), std::ios::cur);
      if (!file) return Status::error("truncated quantized scales: " + record.name);
    }
    archive.records_.push_back(std::move(record));
  }
  return archive;
}

const QuantizedTensorRecord* QuantizedArchive::find(const std::string& name) const {
  for (const auto& record : records_) {
    if (record.name == name) return &record;
  }
  return nullptr;
}

Status QuantizedArchive::read_data(const QuantizedTensorRecord& record, std::vector<std::uint8_t>* out) const {
  if (!out || record.data_offset > file_size_ || data_bytes(record) > file_size_ - record.data_offset) {
    return Status::error("quantized tensor data range is invalid: " + record.name);
  }
  std::ifstream file(path_, std::ios::binary);
  if (!file) return Status::error("failed to open quantized model: " + path_);
  file.seekg(static_cast<std::streamoff>(record.data_offset), std::ios::beg);
  out->resize(static_cast<std::size_t>(data_bytes(record)));
  if (!out->empty()) file.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(out->size()));
  return file ? Status::ok() : Status::error("failed to read quantized tensor: " + record.name);
}

Status QuantizedArchive::read_scales(const QuantizedTensorRecord& record, std::vector<std::uint16_t>* out) const {
  if (!is_integer_quantized(record.dtype) || record.scale_offset > file_size_ ||
      record.scale_count * sizeof(std::uint16_t) > file_size_ - record.scale_offset) {
    return Status::error("quantized scale range is invalid: " + record.name);
  }
  std::ifstream file(path_, std::ios::binary);
  if (!file) return Status::error("failed to open quantized model: " + path_);
  file.seekg(static_cast<std::streamoff>(record.scale_offset), std::ios::beg);
  out->resize(static_cast<std::size_t>(record.scale_count));
  if (!out->empty()) file.read(reinterpret_cast<char*>(out->data()),
                               static_cast<std::streamsize>(out->size() * sizeof(std::uint16_t)));
  return file ? Status::ok() : Status::error("failed to read quantized scales: " + record.name);
}

QuantizedWriter::QuantizedWriter(const std::string& path, std::uint32_t tensor_count)
    : file_(path, std::ios::binary), expected_count_(tensor_count) {
  if (!file_) return;
  file_.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_value(file_, kVersion);
  write_value(file_, tensor_count);
}

QuantizedWriter::~QuantizedWriter() {
  if (!closed_) close();
}

Status QuantizedWriter::append(
    const std::string& name, QuantizedDType dtype, const std::vector<std::int64_t>& shape,
    const std::vector<std::uint8_t>& data, const std::vector<std::uint16_t>& scales,
    std::uint16_t quant_group_size) {
  if (asym_bits(dtype)) {
    QuantizedTensorRecord rec;
    rec.dtype = dtype; rec.shape = shape; rec.quant_group_size = quant_group_size;
    auto expanded = expand_asym_metadata(rec, scales);
    if (!file_ || closed_ || count_ >= expected_count_ || name.empty() || name.size() > (1u << 20) ||
        !expanded.ok() || data.size() != asym_weight_bytes(shape[0], shape[1], asym_bits(dtype)))
      return Status::error("invalid SB6 tensor for output: " + name);
    write_value(file_, static_cast<std::uint32_t>(name.size()));
    write_value(file_, static_cast<std::uint8_t>(dtype));
    write_value(file_, static_cast<std::uint8_t>(2));
    write_value(file_, static_cast<std::uint16_t>(32));
    write_value(file_, checked_numel(shape));
    file_.write(name.data(), name.size());
    for (auto dim : shape) write_value(file_, dim);
    file_.write(reinterpret_cast<const char*>(data.data()), data.size());
    write_value(file_, static_cast<std::uint64_t>(scales.size()));
    file_.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 2);
    if (!file_) return Status::error("failed to write SB6 tensor: " + name);
    ++count_;
    return Status::ok();
  }
  const std::uint64_t logical_numel = checked_numel(shape);
  const bool shape_nonnegative =
      std::all_of(shape.begin(), shape.end(), [](std::int64_t dim) { return dim >= 0; });
  if (!file_) return Status::error("failed to open quantized output");
  if (closed_ || count_ >= expected_count_ || name.empty() || name.size() > std::numeric_limits<std::uint32_t>::max() ||
      !valid_dtype(static_cast<std::uint8_t>(dtype)) ||
      !shape_nonnegative ||
      (is_integer_quantized(dtype) && shape.empty()) || shape.size() > 16 ||
      quant_group_size != 0 ||
      data.size() != (dtype == QuantizedDType::kInt8 ? logical_numel : logical_numel * 2) ||
      (is_integer_quantized(dtype) && scales.size() != static_cast<std::uint64_t>(shape.front())) ||
      (!is_integer_quantized(dtype) && !scales.empty())) {
    return Status::error("invalid quantized tensor for output: " + name);
  }
  write_value(file_, static_cast<std::uint32_t>(name.size()));
  write_value(file_, static_cast<std::uint8_t>(dtype));
  write_value(file_, static_cast<std::uint8_t>(shape.size()));
  write_value(file_, static_cast<std::uint16_t>(0));
  write_value(file_, logical_numel);
  file_.write(name.data(), static_cast<std::streamsize>(name.size()));
  for (std::int64_t dim : shape) write_value(file_, dim);
  if (!data.empty()) file_.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (is_integer_quantized(dtype)) {
    write_value(file_, static_cast<std::uint64_t>(scales.size()));
    file_.write(reinterpret_cast<const char*>(scales.data()),
                static_cast<std::streamsize>(scales.size() * sizeof(std::uint16_t)));
  }
  if (!file_) return Status::error("failed to write quantized tensor: " + name);
  ++count_;
  return Status::ok();
}

Status QuantizedWriter::close() {
  if (closed_) return Status::ok();
  closed_ = true;
  if (!file_) return Status::error("quantized output is not writable");
  if (count_ != expected_count_) return Status::error("quantized output tensor count mismatch");
  file_.flush();
  return file_ ? Status::ok() : Status::error("failed to finalize quantized output");
}

}  // namespace llm_infer
