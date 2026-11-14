#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "rwkv/common/status.hpp"

namespace llm_infer {

enum class QuantizedDType : std::uint8_t {
  kBFloat16 = 0,
  kFloat16 = 1,
  kInt8 = 2,
  // Values 3 (legacy signed INT4) and 5 (SB6 W5) are retired.
  kInt4Asym = 4,
  kInt6Asym = 6,
};

struct QuantizedTensorRecord {
  std::string name;
  QuantizedDType dtype = QuantizedDType::kBFloat16;
  std::vector<std::int64_t> shape;
  std::uint64_t numel = 0;
  std::uint64_t data_offset = 0;
  std::uint64_t scale_offset = 0;
  std::uint64_t scale_count = 0;
  std::uint16_t quant_group_size = 0;
};

class QuantizedArchive {
 public:
  static Result<QuantizedArchive> open(const std::string& path);

  const std::string& path() const { return path_; }
  const std::vector<QuantizedTensorRecord>& records() const { return records_; }
  const QuantizedTensorRecord* find(const std::string& name) const;

  Status read_data(const QuantizedTensorRecord& record, std::vector<std::uint8_t>* out) const;
  Status read_scales(const QuantizedTensorRecord& record, std::vector<std::uint16_t>* out) const;

 private:
  std::string path_;
  std::uint64_t file_size_ = 0;
  std::vector<QuantizedTensorRecord> records_;
};

class QuantizedWriter {
 public:
  QuantizedWriter(const std::string& path, std::uint32_t tensor_count);
  ~QuantizedWriter();

  QuantizedWriter(const QuantizedWriter&) = delete;
  QuantizedWriter& operator=(const QuantizedWriter&) = delete;

  Status append(
      const std::string& name,
      QuantizedDType dtype,
      const std::vector<std::int64_t>& shape,
      const std::vector<std::uint8_t>& data,
      const std::vector<std::uint16_t>& scales = {},
      std::uint16_t quant_group_size = 0);
  Status close();

 private:
  std::ofstream file_;
  std::uint32_t expected_count_ = 0;
  std::uint32_t count_ = 0;
  bool closed_ = false;
};

// SB6: each padded group of 32 stores q4[16], then uint32 bit-4 and
// bit-5 planes for 6 bits. Each superblock of 8 groups stores 16 bytes:
// FP16 d, signed FP16 dm, qs[8] packed into 6 bytes, qm[8] into 6 bytes.
// read_scales returns these 8 raw uint16 words per superblock for asym types.
inline int asym_bits(QuantizedDType dtype) {
  const int v = static_cast<int>(dtype);
  return v == 4 || v == 6 ? v : 0;
}
std::uint64_t asym_weight_bytes(std::uint64_t rows, std::uint64_t cols, int bits);
Result<std::vector<std::uint16_t>> expand_asym_metadata(
    const QuantizedTensorRecord& record, const std::vector<std::uint16_t>& metadata);

bool is_quantized_archive(const std::string& path);

}  // namespace llm_infer
