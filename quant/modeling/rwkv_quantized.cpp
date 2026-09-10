#include "rwkv_quantized.hpp"

#include <algorithm>
#include <array>
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
  return value <= static_cast<std::uint8_t>(QuantizedDType::kInt4);
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
  if (record.dtype == QuantizedDType::kInt8) return record.numel;
  if (record.dtype == QuantizedDType::kInt4 && record.shape.size() == 2) {
    const std::uint64_t rows = static_cast<std::uint64_t>(record.shape[0]);
    const std::uint64_t cols = static_cast<std::uint64_t>(record.shape[1]);
    const std::uint64_t row_bytes = cols / 2 + cols % 2;
    if (row_bytes != 0 && rows > std::numeric_limits<std::uint64_t>::max() / row_bytes) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    return rows * row_bytes;
  }
  return record.numel * 2;
}

bool is_integer_quantized(QuantizedDType dtype) {
  return dtype == QuantizedDType::kInt8 || dtype == QuantizedDType::kInt4;
}

}  // namespace

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
    if (!valid_dtype(dtype) || rank > 16 || name_len == 0 || name_len > (1u << 20) ||
        (dtype >= static_cast<std::uint8_t>(QuantizedDType::kInt8) && rank == 0)) {
      return Status::error("invalid quantized tensor metadata");
    }
    QuantizedTensorRecord record;
    record.name.resize(name_len);
    file.read(record.name.data(), static_cast<std::streamsize>(name_len));
    if (!file) return Status::error("truncated quantized tensor name");
    record.dtype = static_cast<QuantizedDType>(dtype);
    record.quant_group_size = record.dtype == QuantizedDType::kInt4 ? reserved : 0;
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
    if (record.dtype != QuantizedDType::kInt8 && record.dtype != QuantizedDType::kInt4 &&
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
      if (record.dtype == QuantizedDType::kInt4) {
        if (record.shape.size() != 2 ||
            (record.quant_group_size != 32 && record.quant_group_size != 128)) {
          return Status::error("invalid INT4 metadata: " + record.name);
        }
        const std::uint64_t cols = static_cast<std::uint64_t>(record.shape[1]);
        const std::uint64_t groups = cols / record.quant_group_size +
                                     (cols % record.quant_group_size != 0);
        if (groups != 0 && expected_scales > std::numeric_limits<std::uint64_t>::max() / groups) {
          return Status::error("INT4 scale count overflow: " + record.name);
        }
        expected_scales *= groups;
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
  const std::uint64_t logical_numel = checked_numel(shape);
  const bool shape_nonnegative =
      std::all_of(shape.begin(), shape.end(), [](std::int64_t dim) { return dim >= 0; });
  const bool int4 = dtype == QuantizedDType::kInt4;
  const std::uint64_t rows = shape.empty() ? 0 : static_cast<std::uint64_t>(shape[0]);
  const std::uint64_t cols = shape.size() == 2 ? static_cast<std::uint64_t>(shape[1]) : 0;
  const std::uint64_t int4_row_bytes = cols / 2 + cols % 2;
  const bool int4_size_overflow = int4_row_bytes != 0 &&
                                  rows > std::numeric_limits<std::uint64_t>::max() / int4_row_bytes;
  const std::uint64_t int4_data_bytes = int4_size_overflow ? 0 : rows * int4_row_bytes;
  const std::uint64_t groups = int4 && quant_group_size != 0
                                   ? cols / quant_group_size + (cols % quant_group_size != 0)
                                   : 0;
  const bool scale_count_overflow = groups != 0 &&
                                    rows > std::numeric_limits<std::uint64_t>::max() / groups;
  const std::uint64_t expected_scales = int4 ? rows * groups : rows;
  if (!file_) return Status::error("failed to open quantized output");
  if (closed_ || count_ >= expected_count_ || name.empty() || name.size() > std::numeric_limits<std::uint32_t>::max() ||
      static_cast<std::uint8_t>(dtype) > static_cast<std::uint8_t>(QuantizedDType::kInt4) ||
      !shape_nonnegative ||
      (is_integer_quantized(dtype) && shape.empty()) || shape.size() > 16 ||
      (int4 && (shape.size() != 2 || (quant_group_size != 32 && quant_group_size != 128) ||
                int4_size_overflow || scale_count_overflow)) ||
      (!int4 && quant_group_size != 0) ||
      data.size() != (int4 ? int4_data_bytes
                           : dtype == QuantizedDType::kInt8 ? logical_numel : logical_numel * 2) ||
      (is_integer_quantized(dtype) && scales.size() != expected_scales) ||
      (!is_integer_quantized(dtype) && !scales.empty())) {
    return Status::error("invalid quantized tensor for output: " + name);
  }
  write_value(file_, static_cast<std::uint32_t>(name.size()));
  write_value(file_, static_cast<std::uint8_t>(dtype));
  write_value(file_, static_cast<std::uint8_t>(shape.size()));
  write_value(file_, int4 ? quant_group_size : static_cast<std::uint16_t>(0));
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
