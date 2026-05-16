#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kLowrankMax = 512;
constexpr float kLnEps = 1e-5f;
constexpr int kLowrankInRowsT = 7;
constexpr int kLowrankOutRowsT = 4;
constexpr int kLowrankFusedMinC = 1024;

struct ModelDims {
  int layers = 0;
  int channels = 0;
  int heads = 0;
  int head_size = 0;
  int vocab = 0;
  int ffn = 0;
};

enum class CmixMode {
  NoFcOne,
  NoFcRows2,
  Dense,
};

struct Case {
  int B = 1;
  int T = 1;
  bool wkv32 = false;
  std::string cmix_sparse = "no-fc";
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

inline int default_cmix_nofc_max_rows(int channels) {
  if (channels <= 1024) return 4;
  if (channels == 2048) return 8;
  return 12;
}

inline PathConfig select_path(const Case& c, int channels) {
  PathConfig path;
  path.rows = c.B * c.T;
  path.use_batched_rkv = false;
  if (c.cmix_sparse == "off") {
    path.cmix = CmixMode::Dense;
  } else if (path.rows == 1) {
    path.cmix = CmixMode::NoFcOne;
  } else if (path.rows <= default_cmix_nofc_max_rows(channels)) {
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

inline ModelDims infer_model_dims(const std::vector<llm_infer::TensorRecord>& records) {
  ModelDims d;
  for (const auto& rec : records) {
    if (rec.name == "emb.weight" && rec.shape.size() == 2) {
      d.vocab = static_cast<int>(rec.shape[0]);
      d.channels = static_cast<int>(rec.shape[1]);
    } else if (rec.name == "blocks.0.att.r_k" && rec.shape.size() == 2) {
      d.heads = static_cast<int>(rec.shape[0]);
      d.head_size = static_cast<int>(rec.shape[1]);
    } else if (rec.name == "blocks.0.ffn.key.weight" && rec.shape.size() == 2) {
      d.ffn = static_cast<int>(rec.shape[0]);
    }
    if (rec.name.rfind("blocks.", 0) == 0) {
      const char* s = rec.name.c_str() + 7;
      char* end = nullptr;
      long layer = std::strtol(s, &end, 10);
      if (end && *end == '.' && layer >= 0) {
        d.layers = std::max(d.layers, static_cast<int>(layer) + 1);
      }
    }
  }
  require_result(d.layers > 0 && d.channels > 0 && d.heads > 0 && d.head_size > 0 && d.vocab > 0 && d.ffn > 0,
                 "could not infer model dimensions");
  require_result(d.channels == d.heads * d.head_size, "C must equal H*N");
  require_result(d.head_size == 64, "current kernels require head size 64");
  return d;
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

}  // namespace rwkv7_fast_v4
