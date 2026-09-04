#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "status.hpp"

namespace llm_infer {

enum class QuantizedDType : std::uint8_t {
  kBFloat16 = 0,
  kFloat16 = 1,
  kInt8 = 2,
};

struct QuantizedTensorRecord {
  std::string name;
  QuantizedDType dtype = QuantizedDType::kBFloat16;
  std::vector<std::int64_t> shape;
  std::uint64_t numel = 0;
  std::uint64_t data_offset = 0;
  std::uint64_t scale_offset = 0;
  std::uint64_t scale_count = 0;
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
      const std::vector<std::uint16_t>& scales = {});
  Status close();

 private:
  std::ofstream file_;
  std::uint32_t expected_count_ = 0;
  std::uint32_t count_ = 0;
  bool closed_ = false;
};

bool is_quantized_archive(const std::string& path);

}  // namespace llm_infer
