#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <json/json.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "rwkv/io/fingerprint.hpp"
#include "rwkv/io/pth_archive.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv_quantized.hpp"
#include "rwkv/version.hpp"

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

// AW grids and explicitly requested FP16 tensors use IEEE ties-to-even, as
// torch.float16 does. Keep the legacy W8 conversion below unchanged.
std::uint16_t f16_nearest_even(float value) {
  const float magnitude = std::fabs(value);
  std::uint16_t bits = llm_infer::float_to_f16_bits(magnitude);
  if (bits == 0 && magnitude > 2.98023223876953125e-8f) bits = 1;
  if (bits > 0 && bits < 0x7c00u && (bits & 1u)) {
    const float midpoint = (llm_infer::f16_bits_to_float(bits - 1) +
                            llm_infer::f16_bits_to_float(bits)) * 0.5f;
    if (magnitude == midpoint) --bits;
  }
  return bits | (std::signbit(value) ? 0x8000u : 0u);
}

enum class QuantMode { W8A16, W4Asym, W6Asym };

struct Options {
  std::string policy, calibrated;
};

Json::Value read_json(const std::string& path) {
  if (path.empty()) return Json::Value(Json::objectValue);
  std::ifstream file(path);
  Json::Value value;
  Json::CharReaderBuilder builder;
  builder["rejectDupKeys"] = true;
  std::string errors;
  if (!file || !Json::parseFromStream(builder, file, &value, &errors) || !value.isObject())
    throw std::runtime_error("invalid JSON object: " + path + " " + errors);
  return value;
}

bool valid_type(const std::string& type) {
  return type == "bf16" || type == "fp16" || type == "w8a16" ||
         type == "w4a16_asym" || type == "w6a16_asym";
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path, std::size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != static_cast<std::streamoff>(count))
    throw std::runtime_error("calibrated payload size mismatch: " + path.string());
  file.seekg(0);
  std::vector<std::uint8_t> bytes(count);
  if (count && !file.read(reinterpret_cast<char*>(bytes.data()), count))
    throw std::runtime_error("cannot read calibrated payload: " + path.string());
  return bytes;
}

int run(const std::string& input, const std::string& output, QuantMode mode, const Options& options) {
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
  const auto policy = read_json(options.policy);
  const auto calibrated = read_json(options.calibrated);
  std::map<std::string, const llm_infer::TensorRecord*> by_name;
  for (const auto& record : records.value()) by_name.emplace(record.name, &record);
  for (const auto& name : policy.getMemberNames()) {
    if (!by_name.count(name) || !policy[name].isString() || !valid_type(policy[name].asString()))
      throw std::runtime_error("unknown tensor or invalid policy type: " + name);
    if (!quantize_weight_name(name) && policy[name] != "bf16" && policy[name] != "fp16")
      throw std::runtime_error("embedding/LoRA/non-linear tensor cannot be quantized: " + name);
  }
  for (const auto& name : calibrated.getMemberNames()) {
    if (!by_name.count(name) || !quantize_weight_name(name) || !calibrated[name].isObject())
      throw std::runtime_error("invalid calibrated tensor: " + name);
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
    std::string type = record.dtype == llm_infer::TensorDType::kFloat16 ? "fp16" : "bf16";
    if (quantize_weight_name(record.name)) {
      type = "w8a16";
      if (mode == QuantMode::W4Asym) type = "w4a16_asym";
      if (mode == QuantMode::W6Asym) type = "w6a16_asym";
      if (mode == QuantMode::W4Asym && record.name == "blocks.0.att.output.weight") type = "bf16";
    }
    if (policy.isMember(record.name)) type = policy[record.name].asString();
    const bool asym = type == "w4a16_asym" || type == "w6a16_asym";
    const int bits = asym ? type[1] - '0' : 0;
    if ((asym || type == "w8a16") &&
        (record.shape.size() != 2 || record.shape[0] <= 0 || record.shape[1] <= 0))
      throw std::runtime_error("quantization requires a nonempty matrix: " + record.name);
    if (calibrated.isMember(record.name)) {
      const auto& spec = calibrated[record.name];
      if ((!asym && type != "w8a16") || spec["type"].asString() != type ||
          !spec["shape"].isArray() || spec["shape"].size() != 2 ||
          spec["shape"][0].asInt64() != record.shape[0] ||
          spec["shape"][1].asInt64() != record.shape[1] ||
          !spec["data"].isString() || !spec["scales"].isString())
        throw std::runtime_error("calibrated metadata/policy mismatch: " + record.name);
      const std::size_t rows = record.shape[0], cols = record.shape[1];
      const auto base = std::filesystem::path(options.calibrated).parent_path();
      data = read_bytes(base / spec["data"].asString(), asym ? llm_infer::asym_weight_bytes(rows, cols, bits) : rows * cols);
      const auto raw = read_bytes(base / spec["scales"].asString(),
                                 asym ? rows * ((cols + 255) / 256) * 16 : 2 * rows);
      for (std::size_t j = 0; j < raw.size(); j += 2) {
        const std::uint16_t bits = raw[j] | (static_cast<std::uint16_t>(raw[j + 1]) << 8);
        const float scale = llm_infer::f16_bits_to_float(bits);
        if (!asym && (!std::isfinite(scale) || scale <= 0))
          throw std::runtime_error("invalid calibrated FP16 scale: " + record.name);
        scales.push_back(bits);
      }
      dtype = asym ? static_cast<llm_infer::QuantizedDType>(bits) : llm_infer::QuantizedDType::kInt8;
    } else if (asym) {
      throw std::runtime_error("SB6 requires --calibrated payloads; use tools/calibration/export_calibrated.py: " + record.name);
    } else if (type == "w8a16") {
      const std::size_t rows = static_cast<std::size_t>(record.shape[0]);
      const std::size_t cols = static_cast<std::size_t>(record.shape[1]);
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
    } else {
      data.reserve(values.size() * sizeof(std::uint16_t));
      dtype = type == "fp16" ? llm_infer::QuantizedDType::kFloat16 : llm_infer::QuantizedDType::kBFloat16;
      for (float value : values) {
        const auto bits = type == "fp16" ? f16_nearest_even(value) : llm_infer::float_to_bf16_bits(value);
        if (type == "fp16" && !std::isfinite(llm_infer::f16_bits_to_float(bits)))
          throw std::runtime_error("FP16 weight overflow: " + record.name);
        append_u16(&data, bits);
      }
    }
    auto status = writer.append(record.name, dtype, record.shape, data, scales,
                                asym ? 32 : 0);
    if (!status.ok_status()) {
      std::cerr << "error writing " << record.name << ": " << status.message() << "\n";
      return 1;
    }
    std::cout << record.name << " " << type << " numel=" << numel << "\n";
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
    std::cerr << "usage: rwkv_quantize [--format w8a16|w4a16_asym|w6a16_asym] "
                 "[--policy policy.json] [--calibrated manifest.json] "
                 "<input.pth> <output.rwkvq>\n"
                 "       rwkv_quantize --version\n";
    return 0;
  }
  if (argc >= 2 && rwkv::is_version_flag(argv[1])) {
    return rwkv::print_version("rwkv_quantize");
  }
  Options options;
  QuantMode mode = QuantMode::W8A16;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--policy" || arg == "--calibrated") && i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--policy") options.policy = value;
      else options.calibrated = value;
    } else if (arg == "--format" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "w8a16") mode = QuantMode::W8A16;
      else if (value == "w4a16_asym") mode = QuantMode::W4Asym;
      else if (value == "w6a16_asym") mode = QuantMode::W6Asym;
      else {
        std::cerr << "error: --format must be w8a16, w4a16_asym or w6a16_asym\n";
        return 2;
      }
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "error: unknown option: " << arg << "\n";
      return 2;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) {
    std::cerr << "usage: rwkv_quantize [--format w8a16|w4a16_asym|w6a16_asym] "
                 "[--policy policy.json] [--calibrated manifest.json] "
                 "<input.pth> <output.rwkvq>\n";
    return 2;
  }

  const std::string temporary = positional[1] + ".tmp." + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  int result = 1;
  try {
    if (std::filesystem::weakly_canonical(positional[0]) ==
        std::filesystem::weakly_canonical(positional[1]) ||
        (std::filesystem::exists(positional[1]) && std::filesystem::equivalent(positional[0], positional[1])))
      throw std::runtime_error("input and output paths must differ");
    result = run(positional[0], temporary, mode, options);
    if (!result) {
      rwkv7_miss::write_source_fingerprint(positional[0], temporary);
      std::filesystem::rename(temporary, positional[1]);
      std::filesystem::rename(temporary + ".source.json", positional[1] + ".source.json");
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    result = 1;
  }
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  std::filesystem::remove(temporary + ".source.json", ignored);
  std::filesystem::remove(temporary + ".source.json.tmp", ignored);
  return result;
}
