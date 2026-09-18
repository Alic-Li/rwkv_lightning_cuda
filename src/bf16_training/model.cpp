#include "model.hpp"
#include "ops.hpp"
#include "rwkv/io/fingerprint.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
namespace rwkv_bf16_training {
namespace {
using namespace llm_infer;
size_t elements(const TensorRecord &r) {
  if (r.stride.size() != r.shape.size())
    throw std::runtime_error("invalid tensor strides: " + r.name);
  size_t n = 1;
  for (int i = int(r.shape.size()) - 1; i >= 0; --i) {
    if (r.shape[i] <= 0 || size_t(r.shape[i]) > SIZE_MAX / n)
      throw std::runtime_error("invalid shape: " + r.name);
    if (r.shape[i] > 1 && r.stride[i] != int64_t(n))
      throw std::runtime_error("noncontiguous base tensor: " + r.name);
    n *= r.shape[i];
  }
  return n;
}
// Two bounded pinned buffers. Neither a full-file copy nor an FP32 model copy.
struct Staging {
  static constexpr size_t bytes = 8 * 1024 * 1024;
  void *raw = nullptr;
  uint16_t *converted = nullptr;
  Staging() {
    gpu_check(cudaHostAlloc(&raw, bytes, cudaHostAllocDefault), "pinned input");
    try {
      gpu_check(cudaHostAlloc(reinterpret_cast<void **>(&converted), bytes,
                              cudaHostAllocDefault),
                "pinned BF16 staging");
    } catch (...) {
      cudaFreeHost(raw);
      throw;
    }
  }
  ~Staging() {
    if (raw)
      cudaFreeHost(raw);
    if (converted)
      cudaFreeHost(converted);
  }
  void load(const PthArchive &a, const TensorRecord &r, bf16 *dst) {
    if (r.dtype != TensorDType::kBFloat16 && r.dtype != TensorDType::kFloat32 &&
        r.dtype != TensorDType::kFloat16)
      throw std::runtime_error("unsupported base tensor dtype: " + r.name);
    if (r.storage_offset < 0)
      throw std::runtime_error("negative storage offset: " + r.name);
    auto *entry = a.find_entry("archive/data/" + r.storage_key);
    if (!entry) {
      for (auto &e : a.entries())
        if (e.name.size() >= r.storage_key.size() + 6 &&
            e.name.substr(e.name.size() - r.storage_key.size() - 6) ==
                "/data/" + r.storage_key) {
          entry = &e;
          break;
        }
    }
    if (!entry || !entry->is_stored())
      throw std::runtime_error("missing/uncompressed tensor storage: " +
                               r.name);
    size_t n = elements(r), width = dtype_size_bytes(r.dtype),
           step = bytes / width;
    for (size_t off = 0; off < n; off += step) {
      size_t count = std::min(step, n - off);
      auto status = a.read_stored_entry_range(
          *entry, (r.storage_offset + off) * width, raw, count * width);
      if (!status.ok_status())
        throw std::runtime_error(status.message());
      const uint16_t *source = static_cast<uint16_t *>(raw);
      if (r.dtype == TensorDType::kBFloat16) {
        for (size_t i = 0; i < count; ++i)
          if ((source[i] & 0x7f80) == 0x7f80)
            throw std::runtime_error("nonfinite base: " + r.name);
      } else {
        for (size_t i = 0; i < count; ++i) {
          float x = r.dtype == TensorDType::kFloat32
                        ? static_cast<float *>(raw)[i]
                        : f16_bits_to_float(source[i]);
          if (!std::isfinite(x))
            throw std::runtime_error("nonfinite base: " + r.name);
          converted[i] = float_to_bf16_bits(x);
          if ((converted[i] & 0x7f80) == 0x7f80)
            throw std::runtime_error("BF16 conversion overflow: " + r.name);
        }
        source = converted;
      }
      gpu_check(cudaMemcpyAsync(dst + off, source, count * 2,
                                cudaMemcpyHostToDevice, nullptr),
                "stream BF16 weight");
      gpu_check(cudaStreamSynchronize(nullptr), "staging ready");
    }
  }
};
} // namespace
Model::Model(const std::string &path) {
  auto opened = llm_infer::PthArchive::open(path, true);
  if (!opened.ok())
    throw std::runtime_error(opened.status().message());
  auto &a = opened.value();
  auto parsed = llm_infer::parse_pth_tensor_records(a);
  if (!parsed.ok())
    throw std::runtime_error(parsed.status().message());
  std::map<std::string, llm_infer::TensorRecord> records;
  for (auto &r : parsed.value())
    if (!records.emplace(r.name, r).second)
      throw std::runtime_error("duplicate base tensor");
  auto rec = [&](const std::string &n) -> const llm_infer::TensorRecord & {
    auto it = records.find(n);
    if (it == records.end())
      throw std::runtime_error("missing base tensor " + n);
    return it->second;
  };
  const auto &e = rec("emb.weight");
  if (e.shape.size() != 2)
    throw std::runtime_error("embedding shape");
  if (e.shape[0] > std::numeric_limits<int>::max() ||
      e.shape[1] > std::numeric_limits<int>::max())
    throw std::runtime_error("model dimensions exceed supported range");
  int C = e.shape[1], V = e.shape[0];
  if (C <= 0 || C % 64 || V <= 0)
    throw std::runtime_error("invalid BF16 model dimensions");
  view_.channels = C;
  view_.heads = C / 64;
  view_.vocab = V;
  Staging staging;
  auto load = [&](const std::string &name, std::vector<int64_t> shape,
                  bool transpose = false) -> const bf16 * {
    auto &r = rec(name);
    std::vector<int64_t> squeezed;
    for (auto d : r.shape)
      if (d != 1)
        squeezed.push_back(d);
    std::vector<int64_t> wanted;
    for (auto d : shape)
      if (d != 1)
        wanted.push_back(d);
    if (squeezed != wanted)
      throw std::runtime_error("base tensor shape mismatch: " + name);
    auto &b = weights_[name];
    b.resize(elements(r), name.c_str());
    staging.load(a, r, b.p);
    if (transpose) {
      DeviceBuffer<bf16> result;
      result.resize(b.n, "transpose workspace");
      transpose_bf16(shape[0], shape[1], b.p, result.p);
      b = std::move(result);
    }
    return b.p;
  };
  auto ln0w = load("blocks.0.ln0.weight", {C}),
       ln0b = load("blocks.0.ln0.bias", {C});
  // Embedding is normalized in BF16 on device once; only the BF16 lookup lives
  // in RAM.
  auto emb = load("emb.weight", {V, C});
  DeviceBuffer<bf16> normalized;
  normalized.resize(size_t(256) * C, "embedding tile");
  embeddings_.resize(size_t(V) * C);
  for (int row = 0; row < V; row += 256) {
    int rows = std::min(256, V - row);
    rwkv7_v3a_layer_norm_bf16_launch(nullptr, rows, C, emb + size_t(row) * C,
                                     ln0w, ln0b, normalized.p, 1e-5f);
    gpu_check(cudaMemcpy(embeddings_.data() + size_t(row) * C, normalized.p,
                         size_t(rows) * C * 2, cudaMemcpyDeviceToHost),
              "embedding normalization");
  }
  weights_.erase("emb.weight");
  weights_.erase("blocks.0.ln0.weight");
  weights_.erase("blocks.0.ln0.bias");
  view_.cpu_emb_ln0_bf16 = embeddings_.data();
  view_.cpu_emb_ln0_elements = embeddings_.size();
  view_.ln_out_weight = load("ln_out.weight", {C});
  view_.ln_out_bias = load("ln_out.bias", {C});
  view_.head_weight_orig = load("head.weight", {V, C});
  for (int l = 0; records.count("blocks." + std::to_string(l) + ".ln1.weight");
       ++l) {
    std::string p = "blocks." + std::to_string(l) + ".";
    FrozenBlockWeights w;
    w.channels = C;
    w.heads = C / 64;
    auto &r = rec(p + "ffn.key.weight");
    if (r.shape.size() != 2)
      throw std::runtime_error("FFN shape");
    w.ffn = r.shape[0];
    auto vec = [&](const char *n) { return load(p + n, {C}); };
    auto mat = [&](const char *n) { return load(p + n, {C, C}); };
    w.ln1_weight = vec("ln1.weight");
    w.ln1_bias = vec("ln1.bias");
    w.ln2_weight = vec("ln2.weight");
    w.ln2_bias = vec("ln2.bias");
    w.mix_r = vec("att.x_r");
    w.mix_w = vec("att.x_w");
    w.mix_k = vec("att.x_k");
    w.mix_v = vec("att.x_v");
    w.mix_a = vec("att.x_a");
    w.mix_g = vec("att.x_g");
    w.w0 = vec("att.w0");
    w.a0 = vec("att.a0");
    w.k_k = vec("att.k_k");
    w.k_a = vec("att.k_a");
    w.att_group_norm_weight = vec("att.ln_x.weight");
    w.att_group_norm_bias = vec("att.ln_x.bias");
    w.ffn_mix = vec("ffn.x_k");
    w.r_k = load(p + "att.r_k", {C / 64, 64});
    w.receptance = mat("att.receptance.weight");
    w.key = mat("att.key.weight");
    w.value = mat("att.value.weight");
    w.output = mat("att.output.weight");
    w.rank_w = rec(p + "att.w1").shape.at(1);
    w.w1 = load(p + "att.w1", {C, w.rank_w});
    w.w2 = load(p + "att.w2", {w.rank_w, C});
    w.rank_a = rec(p + "att.a1").shape.at(1);
    w.a1 = load(p + "att.a1", {C, w.rank_a});
    w.a2 = load(p + "att.a2", {w.rank_a, C});
    w.rank_g = rec(p + "att.g1").shape.at(1);
    w.g1 = load(p + "att.g1", {C, w.rank_g});
    w.g2 = load(p + "att.g2", {w.rank_g, C});
    if (l) {
      w.rank_v = rec(p + "att.v1").shape.at(1);
      w.v0 = vec("att.v0");
      w.v1 = load(p + "att.v1", {C, w.rank_v});
      w.v2 = load(p + "att.v2", {w.rank_v, C});
    }
    w.ffn_key = load(p + "ffn.key.weight", {w.ffn, C});
    w.ffn_value = load(p + "ffn.value.weight", {C, w.ffn}, true);
    view_.blocks.push_back(w);
    std::cout << "BF16 load layer=" << l << " staging_mib=16" << std::endl;
  }
  view_.layers = view_.blocks.size();
  if (!view_.layers)
    throw std::runtime_error("base has no layers");
  fingerprint_ = rwkv7_miss::fingerprint_file(path);
}
DeviceBuffer<float> Model::load_state(const std::string &path) const {
  auto a = llm_infer::PthArchive::open(path, true);
  if (!a.ok())
    throw std::runtime_error(a.status().message());
  auto records = llm_infer::parse_pth_tensor_records(a.value());
  if (!records.ok())
    throw std::runtime_error(records.status().message());
  size_t lane = size_t(view_.heads) * 4096;
  DeviceBuffer<float> out;
  out.resize(view_.layers * lane, "initial state");
  for (int l = 0; l < view_.layers; ++l) {
    std::string name = "blocks." + std::to_string(l) + ".att.time_state";
    const llm_infer::TensorRecord *r = nullptr;
    for (auto &record : records.value())
      if (record.name == name) {
        if (r)
          throw std::runtime_error("duplicate state");
        r = &record;
      }
    if (!r || r->shape != std::vector<int64_t>{view_.heads, 64, 64})
      throw std::runtime_error("state shape: " + name);
    auto data =
        llm_infer::load_tensor_select(a.value(), *r, false, false, true);
    if (!data.ok())
      throw std::runtime_error(data.status().message());
    std::vector<float> transposed(lane);
    for (size_t h = 0; h < size_t(view_.heads); ++h)
      for (int k = 0; k < 64; ++k)
        for (int v = 0; v < 64; ++v) {
          float x = data.value().values[h * 4096 + v * 64 + k];
          if (!std::isfinite(x))
            throw std::runtime_error("nonfinite initial state");
          transposed[h * 4096 + k * 64 + v] = x;
        }
    gpu_check(cudaMemcpy(out.p + l * lane, transposed.data(), lane * 4,
                         cudaMemcpyHostToDevice),
              "initial state");
  }
  return out;
}
void save_state_checkpoint_pth_host(const std::string &path, int L, int H,
                                    int N, const std::vector<float> &runtime) {
  size_t lane = size_t(H) * N * N;
  if (L <= 0 || H <= 0 || N != 64 || runtime.size() != size_t(L) * lane)
    throw std::runtime_error("state export dimensions");
  std::vector<llm_infer::WriteTensor> ts;
  for (int l = 0; l < L; ++l) {
    llm_infer::WriteTensor t{"blocks." + std::to_string(l) + ".att.time_state",
                             {H, N, N},
                             false,
                             {}};
    t.bf16 = true;
    t.data.resize(lane * 2);
    for (int h = 0; h < H; ++h)
      for (int k = 0; k < N; ++k)
        for (int v = 0; v < N; ++v) {
          float x = runtime[size_t(l) * lane + (size_t(h) * N + k) * N + v];
          if (!std::isfinite(x))
            throw std::runtime_error("nonfinite state export");
          uint16_t bits = llm_infer::float_to_bf16_bits(x);
          if ((bits & 0x7f80) == 0x7f80)
            throw std::runtime_error("BF16 state export overflow");
          std::memcpy(t.data.data() + ((size_t(h) * N + v) * N + k) * 2, &bits,
                      2);
        }
    ts.push_back(std::move(t));
  }
  llm_infer::write_pth(path, ts);
}
void save_state_checkpoint_pth(const std::string &path, cudaStream_t stream,
                               int L, int H, int N, const float *p) {
  std::vector<float> host(size_t(L) * H * N * N);
  gpu_check(cudaMemcpyAsync(host.data(), p, host.size() * 4,
                            cudaMemcpyDeviceToHost, stream),
            "state export");
  gpu_check(cudaStreamSynchronize(stream), "state export ready");
  save_state_checkpoint_pth_host(path, L, H, N, host);
}
} // namespace rwkv_bf16_training
