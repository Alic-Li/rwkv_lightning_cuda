#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "pth_archive.hpp"
#include "pth_tensor.hpp"
#include "rwkv_quantized.hpp"

namespace {

bool quantize_weight_name(const std::string& name) {
  const auto ends_with = [&](const char* suffix) {
    const std::string suffix_text(suffix);
    return name.size() >= suffix_text.size() &&
           name.compare(name.size() - suffix_text.size(), suffix_text.size(), suffix_text) == 0;
  };
  return ends_with("att.receptance.weight") || ends_with("att.key.weight") ||
         ends_with("att.value.weight") || ends_with("att.output.weight") ||
         ends_with("ffn.key.weight") || ends_with("ffn.value.weight") ||
         name == "head.weight";
}

std::uint64_t tensor_numel(const std::vector<std::int64_t>& shape) {
  std::uint64_t element_count = 1;
  for (std::int64_t dim : shape) {
    if (dim < 0) return 0;
    element_count *= static_cast<std::uint64_t>(dim);
  }
  return element_count;
}

bool is_contiguous(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& stride) {
  if (shape.size() != stride.size()) return false;
  std::int64_t expected = 1;
  for (std::size_t rank_index = shape.size(); rank_index > 0; --rank_index) {
    const std::size_t index = rank_index - 1;
    if (shape[index] == 0) return true;
    if (shape[index] != 1 && stride[index] != expected) return false;
    expected *= shape[index];
  }
  return true;
}

std::string archive_prefix(const llm_infer::PthArchive& archive) {
  for (const auto& entry : archive.entries()) {
    constexpr const char* suffix = "/data.pkl";
    if (entry.name.size() >= 9 && entry.name.compare(entry.name.size() - 9, 9, suffix) == 0) {
      return entry.name.substr(0, entry.name.size() - 9);
    }
  }
  return {};
}

llm_infer::Result<std::vector<float>> load_values(
    const llm_infer::PthArchive& archive, const llm_infer::TensorRecord& record) {
  if (record.dtype != llm_infer::TensorDType::kBFloat16) {
    auto loaded = llm_infer::load_tensor_select(
        archive, record, false, false, true);
    if (!loaded.ok()) return loaded.status();
    return loaded.value().values;
  }
  const std::uint64_t element_count = tensor_numel(record.shape);
  if (!is_contiguous(record.shape, record.stride)) {
    auto loaded = llm_infer::load_bf16_tensor_select(archive, record, false, false, true);
    if (!loaded.ok()) return loaded.status();
    return loaded.value().values;
  }
  const std::string prefix = archive_prefix(archive);
  const auto* entry = archive.find_entry(prefix + "/data/" + record.storage_key);
  if (!entry || !entry->is_stored() || record.storage_offset + element_count > record.storage_size ||
      record.storage_size * sizeof(std::uint16_t) != entry->uncompressed_size) {
    return llm_infer::Status::error("invalid storage for tensor: " + record.name);
  }
  std::vector<std::uint16_t> bits(static_cast<std::size_t>(element_count));
  const auto status = archive.read_stored_entry_range(
      *entry, record.storage_offset * sizeof(std::uint16_t), bits.data(), bits.size() * sizeof(std::uint16_t));
  if (!status.ok_status()) return status;
  std::vector<float> values;
  values.reserve(bits.size());
  for (std::uint16_t value : bits) values.push_back(llm_infer::bf16_bits_to_float(value));
  return values;
}

void append_u16(std::vector<std::uint8_t>* out, std::uint16_t value) {
  out->push_back(static_cast<std::uint8_t>(value));
  out->push_back(static_cast<std::uint8_t>(value >> 8));
}

enum class QuantMode { W8A16, W4A16 };

int run(const std::string& input, const std::string& output, QuantMode mode, int group_size) {
  if (std::filesystem::absolute(input) == std::filesystem::absolute(output)) {
    std::cerr << "error: input and output paths must differ\n";
    return 1;
  }
  auto archive = llm_infer::PthArchive::open(input, true);
  if (!archive.ok()) {
    std::cerr << "error: " << archive.status().message() << "\n";
    return 1;
  }
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  if (!records.ok()) {
    std::cerr << "error: " << records.status().message() << "\n";
    return 1;
  }
  llm_infer::QuantizedWriter writer(output, static_cast<std::uint32_t>(records.value().size()));
  for (const auto& record : records.value()) {
    auto loaded = load_values(archive.value(), record);
    if (!loaded.ok()) {
      std::cerr << "error loading " << record.name << ": " << loaded.status().message() << "\n";
      return 1;
    }
    const auto& values = loaded.value();
    const std::uint64_t numel = tensor_numel(record.shape);
    if (numel != values.size()) {
      std::cerr << "error: element count mismatch for " << record.name << "\n";
      return 1;
    }
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
      std::cerr << "error: non-finite value in " << record.name << "\n";
      return 1;
    }
    std::vector<std::uint8_t> data;
    std::vector<std::uint16_t> scales;
    llm_infer::QuantizedDType dtype = llm_infer::QuantizedDType::kBFloat16;
    if (quantize_weight_name(record.name) && record.shape.size() == 2 && record.shape[0] > 0) {
      const std::size_t rows = static_cast<std::size_t>(record.shape[0]);
      const std::size_t cols = static_cast<std::size_t>(record.shape[1]);
      if (mode == QuantMode::W4A16) {
        const std::size_t groups = (cols + group_size - 1) / group_size;
        const std::size_t packed_row_bytes = (cols + 1) / 2;
        data.assign(rows * packed_row_bytes, 0);
        scales.resize(rows * groups);
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t begin = group * group_size;
            const std::size_t end = std::min(cols, begin + static_cast<std::size_t>(group_size));
            float max_abs = 0.0f;
            for (std::size_t col = begin; col < end; ++col) {
              max_abs = std::max(max_abs, std::fabs(values[row * cols + col]));
            }
            float scale_f16 = 1.0f;
            if (max_abs != 0.0f) {
              const float scale = std::max(max_abs / 7.0f, 5.9604644775390625e-8f);
              scale_f16 = std::max(llm_infer::f16_bits_to_float(llm_infer::float_to_f16_bits(scale)),
                                   5.9604644775390625e-8f);
              if (!std::isfinite(scale_f16)) {
                std::cerr << "error: FP16 scale overflow in " << record.name << " row=" << row
                          << " group=" << group << "\n";
                return 1;
              }
            }
            scales[row * groups + group] = llm_infer::float_to_f16_bits(scale_f16);
            for (std::size_t col = begin; col < end; ++col) {
              const long rounded = static_cast<long>(std::nearbyint(values[row * cols + col] / scale_f16));
              const std::uint8_t nibble =
                  static_cast<std::uint8_t>(std::max(-7L, std::min(7L, rounded))) & 0x0f;
              const std::size_t packed_index = row * packed_row_bytes + col / 2;
              if (col & 1) data[packed_index] |= static_cast<std::uint8_t>(nibble << 4);
              else data[packed_index] |= nibble;
            }
          }
        }
        dtype = llm_infer::QuantizedDType::kInt4;
      } else {
        data.resize(values.size());
        scales.resize(rows);
        for (std::size_t row = 0; row < rows; ++row) {
          float max_abs = 0.0f;
          for (std::size_t col = 0; col < cols; ++col) {
            max_abs = std::max(max_abs, std::fabs(values[row * cols + col]));
          }
          const float scale = std::max(max_abs / 127.0f, std::numeric_limits<float>::min());
          const float rounded_scale = llm_infer::f16_bits_to_float(llm_infer::float_to_f16_bits(scale));
          const float scale_f16 = std::max(rounded_scale, 5.9604644775390625e-8f);
          if (!std::isfinite(scale_f16)) {
            std::cerr << "error: FP16 scale overflow in " << record.name << " row=" << row << "\n";
            return 1;
          }
          scales[row] = llm_infer::float_to_f16_bits(scale_f16);
          for (std::size_t col = 0; col < cols; ++col) {
            const long rounded = static_cast<long>(std::nearbyint(values[row * cols + col] / scale_f16));
            const long clamped = std::max(-127L, std::min(127L, rounded));
            data[row * cols + col] = static_cast<std::uint8_t>(static_cast<std::int8_t>(clamped));
          }
        }
        dtype = llm_infer::QuantizedDType::kInt8;
      }
    } else {
      data.reserve(values.size() * sizeof(std::uint16_t));
      for (float value : values) append_u16(&data, llm_infer::float_to_bf16_bits(value));
    }
    auto status = writer.append(record.name, dtype, record.shape, data, scales,
                                dtype == llm_infer::QuantizedDType::kInt4 ? group_size : 0);
    if (!status.ok_status()) {
      std::cerr << "error writing " << record.name << ": " << status.message() << "\n";
      return 1;
    }
    const char* dtype_name = dtype == llm_infer::QuantizedDType::kInt4
                                 ? "int4"
                                 : dtype == llm_infer::QuantizedDType::kInt8 ? "int8" : "bf16";
    std::cout << record.name << " " << dtype_name
              << " numel=" << numel << "\n";
  }
  auto status = writer.close();
  if (!status.ok_status()) {
    std::cerr << "error: " << status.message() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const bool help_requested = argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
  if (help_requested) {
    std::cerr << "usage: rwkv_quantize [--format w8a16|w4a16] [--group-size 32|128] "
                 "<input.pth> <output.rwkvq>\n";
    return 0;
  }
  QuantMode mode = QuantMode::W8A16;
  int group_size = 128;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--format" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "w8a16") mode = QuantMode::W8A16;
      else if (value == "w4a16") mode = QuantMode::W4A16;
      else {
        std::cerr << "error: --format must be w8a16 or w4a16\n";
        return 2;
      }
    } else if (arg == "--group-size" && i + 1 < argc) {
      try {
        std::size_t parsed = 0;
        const std::string value = argv[++i];
        group_size = std::stoi(value, &parsed);
        if (parsed != value.size()) group_size = 0;
      } catch (...) {
        group_size = 0;
      }
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "error: unknown option: " << arg << "\n";
      return 2;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2 || (group_size != 32 && group_size != 128)) {
    std::cerr << "usage: rwkv_quantize [--format w8a16|w4a16] [--group-size 32|128] "
                 "<input.pth> <output.rwkvq>\n";
    return 2;
  }
  if (mode == QuantMode::W8A16 && group_size != 128) {
    std::cerr << "error: --group-size is only valid with --format w4a16\n";
    return 2;
  }
  return run(positional[0], positional[1], mode, group_size);
}
