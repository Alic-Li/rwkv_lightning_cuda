#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pth_archive.hpp"
#include "status.hpp"

namespace llm_infer {

enum class TensorDType {
  kBFloat16,
};

struct TensorRecord {
  std::string name;
  TensorDType dtype = TensorDType::kBFloat16;
  std::string storage_key;
  std::uint64_t storage_size = 0;
  std::uint64_t storage_offset = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
};

struct TensorStats {
  double min = 0.0;
  double max = 0.0;
  bool has_value = false;
};

struct TensorData {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<std::uint16_t> values_bf16;
  std::vector<std::uint16_t> values_bf16_t;
  std::vector<std::uint16_t> values_f16;
  std::vector<std::uint16_t> values_f16_t;
  std::vector<float> values;
  std::vector<float> values_t;
};

const char* dtype_name(TensorDType dtype);
std::uint64_t dtype_size_bytes(TensorDType dtype);
std::string shape_string(const std::vector<std::int64_t>& dims);
float bf16_bits_to_float(std::uint16_t bits);
float f16_bits_to_float(std::uint16_t bits);
std::uint16_t float_to_bf16_bits(float value);
std::uint16_t float_to_f16_bits(float value);

Result<std::vector<TensorRecord>> parse_pth_tensor_records(const PthArchive& archive);
Result<TensorStats> compute_tensor_stats(const PthArchive& archive, const TensorRecord& record);
Result<TensorData> load_bf16_tensor_select(
    const PthArchive& archive,
    const TensorRecord& record,
    bool need_bf16,
    bool need_f16,
    bool need_float);
Result<TensorData> load_bf16_tensor_as_float(const PthArchive& archive, const TensorRecord& record);

}  // namespace llm_infer
