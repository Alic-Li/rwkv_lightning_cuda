#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "pth_archive.hpp"
#include "pth_tensor.hpp"

namespace rwkv7_fast_v4 {

constexpr int kLayers = 32;
constexpr int kChannels = 4096;
constexpr int kHeads = 64;
constexpr int kHeadSize = 64;
constexpr int kVocab = 65536;
constexpr int kFfn = 16384;
constexpr int kLowrankMax = 512;
constexpr float kLnEps = 1e-5f;
constexpr int kLowrankInRowsT = 7;
constexpr int kLowrankOutRowsT = 4;
constexpr int kCmixNoFcMaxRows = 19;
constexpr int kCmixNoFcRow20MaxT = 5;

enum class CmixMode {
  NoFcOne,
  NoFcRows2,
  Dense,
};

struct Case {
  int B = 1;
  int T = 1;
  bool all_logits = false;
  bool list_weights = false;
  bool weight_stats = false;
  bool model_forward = false;
  bool eval_b1t1 = false;
  bool model_memory_plan = false;
  bool graph_bench = false;
  bool profile_range = false;
  int warmup = 3;
  int iters = 10;
  std::string model_path;
  std::string eval_json;
  std::string cases;
};

struct PathConfig {
  int rows = 1;
  bool use_batched_rkv = false;
  CmixMode cmix = CmixMode::NoFcOne;
};

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

template <typename T>
struct DeviceBuffer {
  T* p = nullptr;
  std::size_t n = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), n(other.n) {
    other.p = nullptr;
    other.n = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (p) {
      cudaFree(p);
    }
    p = other.p;
    n = other.n;
    other.p = nullptr;
    other.n = 0;
    return *this;
  }

  ~DeviceBuffer() {
    if (p) cudaFree(p);
  }

  void resize(std::size_t count, const char* what) {
    if (count <= n) {
      return;
    }
    if (p) {
      check_cuda(cudaFree(p), what);
    }
    n = count;
    check_cuda(cudaMalloc(&p, n * sizeof(T)), what);
  }

  void zero(const char* what) {
    if (p && n) {
      check_cuda(cudaMemset(p, 0, n * sizeof(T)), what);
    }
  }
};

struct HalfArena {
  DeviceBuffer<half> storage;
  std::size_t off = 0;

  void allocate(std::size_t elems) {
    storage.resize(elems, "alloc half arena");
    off = 0;
  }

  half* take(std::size_t elems, const char* name) {
    if (off + elems > storage.n) {
      std::fprintf(stderr, "half arena overflow while allocating %s\n", name);
      std::exit(1);
    }
    half* ptr = storage.p + off;
    off += elems;
    return ptr;
  }
};

struct GpuTensor {
  std::string name;
  std::vector<std::int64_t> shape;
  DeviceBuffer<std::uint16_t> f16;

  std::size_t bytes() const {
    return f16.n * sizeof(std::uint16_t);
  }
};

inline const half* hp(const GpuTensor* tensor) {
  return reinterpret_cast<const half*>(tensor->f16.p);
}

struct LayerWeights {
  const GpuTensor* ln0_w = nullptr;
  const GpuTensor* ln0_b = nullptr;
  const GpuTensor* ln1_w = nullptr;
  const GpuTensor* ln1_b = nullptr;
  const GpuTensor* ln2_w = nullptr;
  const GpuTensor* ln2_b = nullptr;
  const GpuTensor* att_x_r = nullptr;
  const GpuTensor* att_x_w = nullptr;
  const GpuTensor* att_x_k = nullptr;
  const GpuTensor* att_x_v = nullptr;
  const GpuTensor* att_x_a = nullptr;
  const GpuTensor* att_x_g = nullptr;
  const GpuTensor* att_receptance_w = nullptr;
  const GpuTensor* att_key_w = nullptr;
  const GpuTensor* att_value_w = nullptr;
  const GpuTensor* att_output_w = nullptr;
  const GpuTensor* att_w0 = nullptr;
  const GpuTensor* att_w1 = nullptr;
  const GpuTensor* att_w1_t = nullptr;
  const GpuTensor* att_w2 = nullptr;
  const GpuTensor* att_w2_t = nullptr;
  const GpuTensor* att_a0 = nullptr;
  const GpuTensor* att_a1 = nullptr;
  const GpuTensor* att_a1_t = nullptr;
  const GpuTensor* att_a2 = nullptr;
  const GpuTensor* att_a2_t = nullptr;
  const GpuTensor* att_g1 = nullptr;
  const GpuTensor* att_g1_t = nullptr;
  const GpuTensor* att_g2 = nullptr;
  const GpuTensor* att_g2_t = nullptr;
  const GpuTensor* att_k_k = nullptr;
  const GpuTensor* att_k_a = nullptr;
  const GpuTensor* att_r_k = nullptr;
  const GpuTensor* att_ln_x_w = nullptr;
  const GpuTensor* att_ln_x_b = nullptr;
  const GpuTensor* att_v0 = nullptr;
  const GpuTensor* att_v1 = nullptr;
  const GpuTensor* att_v1_t = nullptr;
  const GpuTensor* att_v2 = nullptr;
  const GpuTensor* att_v2_t = nullptr;
  const GpuTensor* ffn_x_k = nullptr;
  const GpuTensor* ffn_key_w = nullptr;
  const GpuTensor* ffn_value_w = nullptr;
};

inline PathConfig select_path(const Case& c) {
  PathConfig path;
  path.rows = c.B * c.T;
  path.use_batched_rkv = false;
  if (path.rows == 1) {
    path.cmix = CmixMode::NoFcOne;
  } else if (path.rows <= kCmixNoFcMaxRows || (path.rows == 20 && c.T <= kCmixNoFcRow20MaxT)) {
    path.cmix = CmixMode::NoFcRows2;
  } else {
    path.cmix = CmixMode::Dense;
  }
  return path;
}

inline std::size_t mib(std::size_t bytes) {
  return (bytes + ((1u << 20) - 1)) >> 20;
}

inline bool is_contiguous_shape(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& stride) {
  if (shape.size() != stride.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t idx = shape.size(); idx > 0; --idx) {
    const std::size_t i = idx - 1;
    if (shape[i] == 0) {
      return true;
    }
    if (shape[i] != 1 && stride[i] != expected) {
      return false;
    }
    expected *= shape[i];
  }
  return true;
}

inline std::string archive_prefix(const llm_infer::PthArchive& archive) {
  for (const auto& entry : archive.entries()) {
    const std::string suffix = "/data.pkl";
    if (entry.name.size() >= suffix.size() &&
        entry.name.compare(entry.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return entry.name.substr(0, entry.name.size() - suffix.size());
    }
  }
  return {};
}

inline void print_cuda_mem(const char* label) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  std::printf("gpu_mem %s free=%zuMiB used=%zuMiB total=%zuMiB\n",
              label, mib(free_bytes), mib(total_bytes - free_bytes), mib(total_bytes));
}

inline std::size_t numel(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (std::int64_t d : shape) {
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

inline void require_result(bool ok, const std::string& message) {
  if (!ok) {
    std::cerr << "error: " << message << "\n";
    std::exit(1);
  }
}

inline bool ends_with(const std::string& s, const char* suffix) {
  const std::size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

inline bool is_att_c2c_weight(const std::string& name) {
  return name.find(".att.") != std::string::npos &&
         (ends_with(name, "receptance.weight") || ends_with(name, "key.weight") ||
          ends_with(name, "value.weight") || ends_with(name, "output.weight"));
}

inline bool is_ffn_key_weight(const std::string& name) {
  return name.find(".ffn.key.weight") != std::string::npos;
}

inline bool is_head_weight(const std::string& name) {
  return name == "head.weight";
}

inline bool should_keep_orig_layout_like_v3a(const std::string& name) {
  return is_att_c2c_weight(name) || is_ffn_key_weight(name) || is_head_weight(name);
}

inline bool should_transpose_like_v3a(const std::string& name) {
  if (should_keep_orig_layout_like_v3a(name)) {
    return false;
  }
  return name.find("key.weight") != std::string::npos ||
         name.find("value.weight") != std::string::npos ||
         name.find("receptance.weight") != std::string::npos ||
         name.find("output.weight") != std::string::npos;
}

inline bool unused_weight_like_v3a(const std::string& name) {
  return name == "emb.weight" ||
         name == "blocks.0.att.v0" ||
         name == "blocks.0.att.v1" ||
         name == "blocks.0.att.v2";
}

inline bool lowrank_needs_t_copy_like_v3a(const std::string& name) {
  if (unused_weight_like_v3a(name)) {
    return false;
  }
  return ends_with(name, "att.w1") || ends_with(name, "att.w2") ||
         ends_with(name, "att.a1") || ends_with(name, "att.a2") ||
         ends_with(name, "att.g1") || ends_with(name, "att.g2") ||
         ends_with(name, "att.v1") || ends_with(name, "att.v2");
}

inline Case parse_case(int argc, char** argv) {
  Case c;
  for (int i = 1; i < argc; ++i) {
    auto need_value = [&](const char* name) -> char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--B") == 0) {
      c.B = std::atoi(need_value("--B"));
    } else if (std::strcmp(argv[i], "--T") == 0) {
      c.T = std::atoi(need_value("--T"));
    } else if (std::strcmp(argv[i], "--all-logits") == 0) {
      c.all_logits = true;
    } else if (std::strcmp(argv[i], "--list-weights") == 0) {
      c.list_weights = true;
    } else if (std::strcmp(argv[i], "--model-forward") == 0) {
      c.model_forward = true;
    } else if (std::strcmp(argv[i], "--eval-b1t1") == 0) {
      c.eval_b1t1 = true;
      c.model_forward = true;
    } else if (std::strcmp(argv[i], "--model-memory-plan") == 0) {
      c.model_memory_plan = true;
    } else if (std::strcmp(argv[i], "--graph-bench") == 0) {
      c.graph_bench = true;
    } else if (std::strcmp(argv[i], "--profile-range") == 0) {
      c.profile_range = true;
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      c.warmup = std::atoi(need_value("--warmup"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      c.iters = std::atoi(need_value("--iters"));
    } else if (std::strcmp(argv[i], "--weight-stats") == 0) {
      c.weight_stats = true;
    } else if (std::strcmp(argv[i], "--model") == 0) {
      c.model_path = need_value("--model");
    } else if (std::strcmp(argv[i], "--eval-json") == 0) {
      c.eval_json = need_value("--eval-json");
    } else if (std::strcmp(argv[i], "--cases") == 0) {
      c.cases = need_value("--cases");
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("usage: benchmark [--B n] [--T n] [--cases list] [--all-logits] [--eval-b1t1] [--graph-bench --profile-range --warmup n --iters n] [--model path --list-weights|--model-memory-plan|--model-forward [--eval-json path] [--weight-stats]]\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (c.B <= 0 || c.T <= 0) {
    std::fprintf(stderr, "B and T must be positive\n");
    std::exit(2);
  }
  if (c.warmup < 0 || c.iters <= 0) {
    std::fprintf(stderr, "warmup must be >=0 and iters must be >0\n");
    std::exit(2);
  }
  return c;
}

inline void list_weights(const Case& c) {
  if (c.model_path.empty()) {
    std::cerr << "error: --list-weights requires --model path\n";
    std::exit(2);
  }
  auto archive = llm_infer::PthArchive::open(c.model_path);
  require_result(archive.ok(), archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  require_result(records.ok(), records.status().message());

  int layers = 0;
  int channels = 0;
  int heads = 0;
  int head_size = 0;
  int vocab = 0;
  std::unordered_map<std::string, const llm_infer::TensorRecord*> by_name;
  for (const auto& rec : records.value()) {
    by_name.emplace(rec.name, &rec);
    if (rec.name == "emb.weight" && rec.shape.size() == 2) {
      vocab = static_cast<int>(rec.shape[0]);
      channels = static_cast<int>(rec.shape[1]);
    } else if (rec.name == "blocks.0.att.r_k" && rec.shape.size() == 2) {
      heads = static_cast<int>(rec.shape[0]);
      head_size = static_cast<int>(rec.shape[1]);
    }
    if (rec.name.rfind("blocks.", 0) == 0) {
      const char* s = rec.name.c_str() + 7;
      char* end = nullptr;
      long layer = std::strtol(s, &end, 10);
      if (end && *end == '.' && layer >= 0) {
        layers = std::max(layers, static_cast<int>(layer) + 1);
      }
    }
  }
  std::cout << "weights path=" << c.model_path
            << " tensors=" << records.value().size()
            << " L=" << layers
            << " C=" << channels
            << " H=" << heads
            << " N=" << head_size
            << " V=" << vocab
            << "\n";

  const char* keys[] = {
      "emb.weight",
      "blocks.0.ln0.weight",
      "blocks.0.att.x_r",
      "blocks.0.att.receptance.weight",
      "blocks.0.att.w1",
      "blocks.0.att.w2",
      "blocks.0.att.r_k",
      "blocks.0.ffn.key.weight",
      "blocks.0.ffn.value.weight",
      "ln_out.weight",
      "head.weight",
  };
  std::cout << "name\tdtype\tshape";
  if (c.weight_stats) {
    std::cout << "\tmin\tmax";
  }
  std::cout << "\n";
  for (const char* key : keys) {
    auto it = by_name.find(key);
    if (it == by_name.end()) {
      std::cout << key << "\tmissing\t";
      if (c.weight_stats) std::cout << "\t\t";
      std::cout << "\n";
      continue;
    }
    const auto& rec = *it->second;
    std::cout << rec.name << "\t"
              << llm_infer::dtype_name(rec.dtype) << "\t"
              << llm_infer::shape_string(rec.shape);
    if (c.weight_stats) {
      auto stats = llm_infer::compute_tensor_stats(archive.value(), rec);
      require_result(stats.ok(), stats.status().message());
      if (stats.value().has_value) {
        std::cout << "\t" << stats.value().min << "\t" << stats.value().max;
      } else {
        std::cout << "\t\t";
      }
    }
    std::cout << "\n";
  }
}

inline std::vector<int> read_eval_tokens(const std::string& path, int count) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "error: failed to open eval json: " << path << "\n";
    std::exit(1);
  }
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::size_t key = text.find("\"tokens\"");
  const std::size_t lbr = key == std::string::npos ? std::string::npos : text.find('[', key);
  if (lbr == std::string::npos) {
    std::cerr << "error: eval json has no tokens array: " << path << "\n";
    std::exit(1);
  }
  std::vector<int> out;
  for (std::size_t i = lbr + 1; i < text.size() && static_cast<int>(out.size()) < count;) {
    while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])) && text[i] != ']') ++i;
    if (i >= text.size() || text[i] == ']') break;
    int value = 0;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
      value = value * 10 + (text[i++] - '0');
    }
    out.push_back(value);
  }
  if (static_cast<int>(out.size()) < count) {
    std::cerr << "error: eval json has fewer than " << count << " tokens\n";
    std::exit(1);
  }
  return out;
}

inline std::vector<int> read_eval_tokens_all(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "error: failed to open eval json: " << path << "\n";
    std::exit(1);
  }
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::size_t key = text.find("\"tokens\"");
  const std::size_t lbr = key == std::string::npos ? std::string::npos : text.find('[', key);
  if (lbr == std::string::npos) {
    std::cerr << "error: eval json has no tokens array: " << path << "\n";
    std::exit(1);
  }
  std::vector<int> out;
  for (std::size_t i = lbr + 1; i < text.size();) {
    while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])) && text[i] != ']') ++i;
    if (i >= text.size() || text[i] == ']') break;
    int value = 0;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
      value = value * 10 + (text[i++] - '0');
    }
    out.push_back(value);
  }
  return out;
}

inline std::vector<std::pair<int, int>> model_forward_cases(const Case& c) {
  std::vector<std::pair<int, int>> out;
  if (c.cases.empty()) {
    out.emplace_back(c.B, c.T);
    return out;
  }
  for (std::size_t i = 0; i < c.cases.size();) {
    while (i < c.cases.size() && (c.cases[i] == ',' || std::isspace(static_cast<unsigned char>(c.cases[i])))) ++i;
    if (i >= c.cases.size()) break;
    int b = 0;
    while (i < c.cases.size() && std::isdigit(static_cast<unsigned char>(c.cases[i]))) {
      b = b * 10 + (c.cases[i++] - '0');
    }
    if (i >= c.cases.size() || (c.cases[i] != 'x' && c.cases[i] != 'X')) {
      std::cerr << "error: invalid --cases item, expected BxT: " << c.cases << "\n";
      std::exit(2);
    }
    ++i;
    int t = 0;
    while (i < c.cases.size() && std::isdigit(static_cast<unsigned char>(c.cases[i]))) {
      t = t * 10 + (c.cases[i++] - '0');
    }
    if (b <= 0 || t <= 0) {
      std::cerr << "error: --cases requires positive B and T\n";
      std::exit(2);
    }
    out.emplace_back(b, t);
  }
  if (out.empty()) {
    std::cerr << "error: empty --cases\n";
    std::exit(2);
  }
  return out;
}

inline void model_memory_plan(const Case& c) {
  if (c.model_path.empty()) {
    std::cerr << "error: --model-memory-plan requires --model path\n";
    std::exit(2);
  }
  auto archive = llm_infer::PthArchive::open(c.model_path);
  require_result(archive.ok(), archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  require_result(records.ok(), records.status().message());

  std::size_t gpu_base = 0;
  std::size_t gpu_lowrank_t = 0;
  std::size_t skipped_cpu = 0;
  int t_copies = 0;
  for (const auto& rec : records.value()) {
    const std::size_t bytes = numel(rec.shape) * sizeof(std::uint16_t);
    if (unused_weight_like_v3a(rec.name)) {
      skipped_cpu += bytes;
      continue;
    }
    gpu_base += bytes;
    if (lowrank_needs_t_copy_like_v3a(rec.name)) {
      gpu_lowrank_t += bytes;
      ++t_copies;
    }
  }
  std::cout << "model_memory_plan path=" << c.model_path
            << " gpu_base_mib=" << mib(gpu_base)
            << " gpu_extra_t_mib=" << mib(gpu_lowrank_t)
            << " gpu_total_mib=" << mib(gpu_base + gpu_lowrank_t)
            << " cpu_skipped_mib=" << mib(skipped_cpu)
            << " t_copies=" << t_copies
            << "\n";
}

}  // namespace rwkv7_fast_v4
