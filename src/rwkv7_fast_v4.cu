#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "pth_archive.hpp"
#include "pth_tensor.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv7_fast_v4_common.hpp"
#include "rwkv7_fast_v4_kernels.cuh"

namespace {

using namespace rwkv7_fast_v4;

struct RawBf16TensorView {
  const std::uint16_t* data = nullptr;
  std::uint64_t elems = 0;
};

RawBf16TensorView raw_bf16_tensor_view(const llm_infer::PthArchive& archive, const llm_infer::TensorRecord& rec) {
  if (!is_contiguous_shape(rec.shape, rec.stride)) {
    std::cerr << "error: v4 GPU loader currently requires contiguous tensor: " << rec.name << "\n";
    std::exit(1);
  }
  const std::string prefix = archive_prefix(archive);
  if (prefix.empty()) {
    std::cerr << "error: archive prefix not found\n";
    std::exit(1);
  }
  const auto* entry = archive.find_entry(prefix + "/data/" + rec.storage_key);
  if (!entry) {
    std::cerr << "error: storage entry not found for tensor: " << rec.name << "\n";
    std::exit(1);
  }
  auto view = archive.stored_entry_view(*entry);
  require_result(view.ok(), view.status().message());
  if (rec.storage_size * sizeof(std::uint16_t) != view.value().size) {
    std::cerr << "error: storage byte size mismatch for tensor: " << rec.name << "\n";
    std::exit(1);
  }
  const std::uint64_t n = numel(rec.shape);
  if (rec.storage_offset + n > rec.storage_size) {
    std::cerr << "error: tensor data range exceeds storage: " << rec.name << "\n";
    std::exit(1);
  }
  RawBf16TensorView out;
  out.data = reinterpret_cast<const std::uint16_t*>(view.value().data + rec.storage_offset * sizeof(std::uint16_t));
  out.elems = n;
  return out;
}

struct WeightLoadPipeline {
  struct Slot {
    DeviceBuffer<std::uint16_t> staging;
    cudaStream_t copy = nullptr;
    cudaStream_t compute = nullptr;
    cudaEvent_t copied = nullptr;
    cudaEvent_t done = nullptr;
    bool in_flight = false;
  };

  Slot slots[2];
  int next = 0;

  WeightLoadPipeline() {
    for (auto& s : slots) {
      check_cuda(cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking), "create weight copy stream");
      check_cuda(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking), "create weight compute stream");
      check_cuda(cudaEventCreateWithFlags(&s.copied, cudaEventDisableTiming), "create weight copied event");
      check_cuda(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "create weight done event");
    }
  }

  ~WeightLoadPipeline() {
    sync();
    for (auto& s : slots) {
      cudaEventDestroy(s.copied);
      cudaEventDestroy(s.done);
      cudaStreamDestroy(s.copy);
      cudaStreamDestroy(s.compute);
    }
  }

  Slot& acquire(std::size_t elems) {
    Slot& s = slots[next++ & 1];
    if (s.in_flight) {
      check_cuda(cudaEventSynchronize(s.done), "wait weight slot");
      s.in_flight = false;
    }
    s.staging.resize(elems, "alloc bf16 staging");
    return s;
  }

  void sync() {
    for (auto& s : slots) {
      if (s.in_flight) {
        check_cuda(cudaEventSynchronize(s.done), "sync weight load");
        s.in_flight = false;
      }
    }
  }
};

std::unique_ptr<GpuTensor> load_tensor_f16_like_v3a(
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
    const std::string& key,
    bool required,
    WeightLoadPipeline& pipeline) {
  auto it = by_name.find(key);
  if (it == by_name.end()) {
    if (required) {
      std::cerr << "error: missing tensor: " << key << "\n";
      std::exit(1);
    }
    return nullptr;
  }
  const auto& rec = *it->second;
  const RawBf16TensorView raw = raw_bf16_tensor_view(archive, rec);
  std::vector<std::int64_t> runtime_shape = rec.shape;
  const bool transpose = should_transpose_like_v3a(rec.name);
  if (transpose) {
    if (rec.shape.size() != 2) {
      std::cerr << "error: v3a transpose rule requires 2D tensor: " << rec.name << "\n";
      std::exit(1);
    }
    runtime_shape = {rec.shape[1], rec.shape[0]};
  }
  auto tensor = std::make_unique<GpuTensor>();
  tensor->name = rec.name;
  tensor->shape = std::move(runtime_shape);
  tensor->f16.resize(static_cast<std::size_t>(raw.elems), "alloc weight tensor");
  auto& slot = pipeline.acquire(static_cast<std::size_t>(raw.elems));
  check_cuda(cudaMemcpyAsync(slot.staging.p, raw.data, raw.elems * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, slot.copy), "copy raw bf16 weight");
  check_cuda(cudaEventRecord(slot.copied, slot.copy), "record raw bf16 copied");
  check_cuda(cudaStreamWaitEvent(slot.compute, slot.copied, 0), "wait raw bf16 copied");
  if (transpose) {
    rwkv7_v4_bf16_to_f16_transpose_launch(
        slot.compute, slot.staging.p, tensor->f16.p,
        static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
  } else {
    rwkv7_v4_bf16_to_f16_launch(slot.compute, slot.staging.p, tensor->f16.p, raw.elems);
  }
  check_cuda(cudaGetLastError(), "launch bf16 weight preprocess");
  check_cuda(cudaEventRecord(slot.done, slot.compute), "record weight preprocess done");
  slot.in_flight = true;
  return tensor;
}

std::string block_key(int layer, const char* suffix) {
  return "blocks." + std::to_string(layer) + "." + suffix;
}

struct CudaWeights {
  std::unordered_map<std::string, std::unique_ptr<GpuTensor>> tensors;
  std::vector<LayerWeights> layers;
  const GpuTensor* ln_out_w = nullptr;
  const GpuTensor* ln_out_b = nullptr;
  const GpuTensor* head_w = nullptr;
  int optional_loaded = 0;
  int t_copy_count = 0;
  std::size_t cpu_emb_bytes = 0;
  std::vector<std::uint16_t> cpu_emb_ln0_f16;

  const GpuTensor* optional(const std::string& key) const {
    auto it = tensors.find(key);
    return it == tensors.end() ? nullptr : it->second.get();
  }

  const GpuTensor* require(const std::string& key) const {
    const GpuTensor* t = optional(key);
    if (!t) {
      std::cerr << "error: tensor view missing: " << key << "\n";
      std::exit(1);
    }
    return t;
  }

  void load(
      const llm_infer::PthArchive& archive,
      const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
      const std::string& key,
      bool required,
      WeightLoadPipeline& pipeline) {
    auto tensor = load_tensor_f16_like_v3a(archive, by_name, key, required, pipeline);
    if (!tensor) {
      return;
    }
    tensors.emplace(key, std::move(tensor));
    if (!required) {
      ++optional_loaded;
    }
  }

  void add_t_copy(const std::string& key, WeightLoadPipeline& pipeline) {
    pipeline.sync();
    const GpuTensor* src = require(key);
    if (src->shape.size() != 2) {
      std::cerr << "error: .t copy requires 2D tensor: " << key << "\n";
      std::exit(1);
    }
    const int rows = static_cast<int>(src->shape[0]);
    const int cols = static_cast<int>(src->shape[1]);
    auto tensor = std::make_unique<GpuTensor>();
    tensor->name = key + ".t";
    tensor->shape = {cols, rows};
    tensor->f16.resize(static_cast<std::size_t>(rows) * cols, "alloc .t tensor");
    rwkv7_v4_f16_transpose_launch(nullptr, src->f16.p, tensor->f16.p, rows, cols);
    check_cuda(cudaGetLastError(), "launch .t transpose");
    tensors.emplace(tensor->name, std::move(tensor));
    ++t_copy_count;
  }

  std::size_t bytes() const {
    std::size_t total = 0;
    for (const auto& kv : tensors) {
      total += kv.second->bytes();
    }
    return total;
  }

  LayerWeights layer_view(int layer) const {
    LayerWeights w;
    auto req = [&](const char* suffix) { return require(block_key(layer, suffix)); };
    auto opt = [&](const char* suffix) { return optional(block_key(layer, suffix)); };
    w.ln0_w = opt("ln0.weight"); w.ln0_b = opt("ln0.bias");
    w.ln1_w = req("ln1.weight"); w.ln1_b = req("ln1.bias");
    w.ln2_w = req("ln2.weight"); w.ln2_b = req("ln2.bias");
    w.att_x_r = req("att.x_r"); w.att_x_w = req("att.x_w"); w.att_x_k = req("att.x_k");
    w.att_x_v = req("att.x_v"); w.att_x_a = req("att.x_a"); w.att_x_g = req("att.x_g");
    w.att_receptance_w = req("att.receptance.weight");
    w.att_key_w = req("att.key.weight");
    w.att_value_w = req("att.value.weight");
    w.att_output_w = req("att.output.weight");
    w.att_w0 = req("att.w0"); w.att_w1 = req("att.w1"); w.att_w2 = req("att.w2");
    w.att_w1_t = req("att.w1.t"); w.att_w2_t = req("att.w2.t");
    w.att_a0 = req("att.a0"); w.att_a1 = req("att.a1"); w.att_a2 = req("att.a2");
    w.att_a1_t = req("att.a1.t"); w.att_a2_t = req("att.a2.t");
    w.att_g1 = req("att.g1"); w.att_g2 = req("att.g2");
    w.att_g1_t = req("att.g1.t"); w.att_g2_t = req("att.g2.t");
    w.att_k_k = req("att.k_k"); w.att_k_a = req("att.k_a"); w.att_r_k = req("att.r_k");
    w.att_ln_x_w = req("att.ln_x.weight"); w.att_ln_x_b = req("att.ln_x.bias");
    w.att_v0 = opt("att.v0"); w.att_v1 = opt("att.v1"); w.att_v2 = opt("att.v2");
    w.att_v1_t = opt("att.v1.t"); w.att_v2_t = opt("att.v2.t");
    w.ffn_x_k = req("ffn.x_k");
    w.ffn_key_w = req("ffn.key.weight");
    w.ffn_value_w = req("ffn.value.weight");
    return w;
  }

  void build_global_view() {
    ln_out_w = require("ln_out.weight");
    ln_out_b = require("ln_out.bias");
    head_w = require("head.weight");
  }
};

void build_cpu_emb_ln0_f16(
    CudaWeights& weights,
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name) {
  if (weights.layers.empty() || !weights.layers[0].ln0_w || !weights.layers[0].ln0_b) {
    std::cerr << "error: layer0 ln0 weights are required before emb+ln0 preprocessing\n";
    std::exit(1);
  }
  auto raw = [&](const std::string& key) -> RawBf16TensorView {
    auto it = by_name.find(key);
    if (it == by_name.end()) {
      std::cerr << "error: missing tensor for emb+ln0 preprocessing: " << key << "\n";
      std::exit(1);
    }
    return raw_bf16_tensor_view(archive, *it->second);
  };
  const RawBf16TensorView emb = raw("emb.weight");
  const RawBf16TensorView ln0_w = raw("blocks.0.ln0.weight");
  const RawBf16TensorView ln0_b = raw("blocks.0.ln0.bias");
  const std::size_t elems = static_cast<std::size_t>(kVocab) * kChannels;
  if (emb.elems != elems || ln0_w.elems != kChannels || ln0_b.elems != kChannels) {
    std::cerr << "error: emb/ln0 shape mismatch for emb+ln0 preprocessing\n";
    std::exit(1);
  }
  DeviceBuffer<std::uint16_t> gpu_emb;
  DeviceBuffer<std::uint16_t> gpu_ln0_w;
  DeviceBuffer<std::uint16_t> gpu_ln0_b;
  DeviceBuffer<std::uint16_t> gpu_out;
  gpu_emb.resize(emb.elems, "alloc raw bf16 emb");
  gpu_ln0_w.resize(ln0_w.elems, "alloc raw bf16 ln0 weight");
  gpu_ln0_b.resize(ln0_b.elems, "alloc raw bf16 ln0 bias");
  gpu_out.resize(elems, "alloc emb+ln0 gpu output");
  check_cuda(cudaMemcpy(gpu_emb.p, emb.data, emb.elems * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
             "copy raw bf16 emb");
  check_cuda(cudaMemcpy(gpu_ln0_w.p, ln0_w.data, ln0_w.elems * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
             "copy raw bf16 ln0 weight");
  check_cuda(cudaMemcpy(gpu_ln0_b.p, ln0_b.data, ln0_b.elems * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
             "copy raw bf16 ln0 bias");
  rwkv7_v4_emb_ln0_bf16_to_f16_launch(
      nullptr, kVocab, kChannels, gpu_emb.p, gpu_ln0_w.p, gpu_ln0_b.p, gpu_out.p, kLnEps);
  check_cuda(cudaGetLastError(), "launch emb+ln0 preprocess");
  weights.cpu_emb_ln0_f16.resize(elems);
  check_cuda(cudaMemcpy(weights.cpu_emb_ln0_f16.data(), gpu_out.p, elems * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost), "copy emb+ln0 to CPU");
  check_cuda(cudaDeviceSynchronize(), "sync emb+ln0 preprocess");
  weights.cpu_emb_bytes = weights.cpu_emb_ln0_f16.size() * sizeof(std::uint16_t);
}

void load_layer_into(
    CudaWeights& weights,
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
    int layer,
    WeightLoadPipeline& pipeline) {
  const char* required[] = {
      "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
      "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g",
      "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight",
      "att.w0", "att.w1", "att.w2", "att.a0", "att.a1", "att.a2", "att.g1", "att.g2",
      "att.k_k", "att.k_a", "att.r_k", "att.ln_x.weight", "att.ln_x.bias",
      "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
  };
  if (layer == 0) {
    weights.load(archive, by_name, block_key(layer, "ln0.weight"), true, pipeline);
    weights.load(archive, by_name, block_key(layer, "ln0.bias"), true, pipeline);
  }
  for (const char* suffix : required) {
    weights.load(archive, by_name, block_key(layer, suffix), true, pipeline);
  }
  if (layer > 0) {
    weights.load(archive, by_name, block_key(layer, "att.v0"), true, pipeline);
    weights.load(archive, by_name, block_key(layer, "att.v1"), true, pipeline);
    weights.load(archive, by_name, block_key(layer, "att.v2"), true, pipeline);
  }
  const char* lowrank_t[] = {"att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2"};
  for (const char* suffix : lowrank_t) {
    weights.add_t_copy(block_key(layer, suffix), pipeline);
  }
  if (layer > 0) {
    weights.add_t_copy(block_key(layer, "att.v1"), pipeline);
    weights.add_t_copy(block_key(layer, "att.v2"), pipeline);
  }
  weights.layers.push_back(weights.layer_view(layer));
}

CudaWeights load_model_weights(
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name) {
  CudaWeights weights;
  auto emb = by_name.find("emb.weight");
  if (emb != by_name.end()) {
    weights.cpu_emb_bytes = numel(emb->second->shape) * sizeof(std::uint16_t);
  }
  WeightLoadPipeline pipeline;
  weights.load(archive, by_name, "ln_out.weight", true, pipeline);
  weights.load(archive, by_name, "ln_out.bias", true, pipeline);
  weights.load(archive, by_name, "head.weight", true, pipeline);
  std::cout << "load_model global done gpu_mib=" << mib(weights.bytes())
            << " cpu_emb_mib=" << mib(weights.cpu_emb_bytes) << "\n";
  for (int layer = 0; layer < kLayers; ++layer) {
    load_layer_into(weights, archive, by_name, layer, pipeline);
    std::cout << "load_model layer=" << layer
              << " done layers=" << weights.layers.size()
              << " tensors=" << weights.tensors.size()
              << " t_copies=" << weights.t_copy_count
              << " gpu_mib=" << mib(weights.bytes()) << "\n";
  }
  pipeline.sync();
  check_cuda(cudaDeviceSynchronize(), "sync model weight load");
  weights.build_global_view();
  build_cpu_emb_ln0_f16(weights, archive, by_name);
  std::cout << "load_model emb+ln0 done cpu_emb_mib=" << mib(weights.cpu_emb_bytes)
            << " entries=" << weights.cpu_emb_ln0_f16.size() << "\n";
  return weights;
}

enum class LinearGroup {
  AttC2C,
  FfnKey,
  Head,
};

void linear_orig_layout_launch(
    cudaStream_t stream,
    const PathConfig& path,
    LinearGroup group,
    int M,
    int K,
    int N,
    const half* x,
    const half* weight_orig,
    void* workspace,
    std::size_t workspace_bytes,
    half* y) {
  if (path.rows == 1) {
    const int out_tile = group == LinearGroup::AttC2C ? 4 : 2;
    rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 1, out_tile, y);
    return;
  }
  if (path.rows == 2) {
    if (group == LinearGroup::AttC2C) {
      rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, 4, y);
    } else {
      rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 2, 2, y);
    }
    return;
  }
  if (path.rows == 3) {
    if (group == LinearGroup::Head) {
      rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 3, 2, y);
    } else if (group == LinearGroup::FfnKey) {
      rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 32, 3, 4, y);
    } else {
      rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 64, 3, 4, y);
    }
    return;
  }
  auto lt = [&](int workspace_mb, int algo) {
    const std::size_t bytes = static_cast<std::size_t>(workspace_mb) << 20;
    if (bytes > workspace_bytes) {
      std::cerr << "error: cublasLt workspace too small\n";
      std::exit(1);
    }
    rwkv7_v3a_linear_f16_orig_lt_cfg_launch(stream, M, K, N, x, weight_orig, workspace, bytes, algo, y);
  };
  if (group == LinearGroup::Head) {
    if (path.rows >= 1024) return lt(128, 0);
    if (path.rows >= 512) return lt(0, 2);
    if (path.rows >= 384) return lt(128, 2);
    if (path.rows >= 256) return lt(0, 1);
    if (path.rows >= 192) return lt(128, 0);
    if (path.rows >= 160) return lt(32, 0);
    if (path.rows >= 128) return lt(128, 0);
    if (path.rows >= 112) return lt(32, 0);
    if (path.rows >= 96) return lt(32, 1);
    if (path.rows >= 80) return lt(32, 2);
    if (path.rows >= 72) return lt(128, 2);
  } else if (group == LinearGroup::AttC2C) {
    if (path.rows >= 1024) return lt(32, 4);
    if (path.rows >= 768) return lt(32, 0);
    if (path.rows >= 512) return lt(32, 1);
    if (path.rows >= 384) return lt(128, 2);
    if (path.rows >= 256) return lt(128, 0);
    if (path.rows >= 192) return lt(32, 2);
    if (path.rows >= 160) return lt(128, 1);
    if (path.rows >= 128) return lt(128, 0);
    if (path.rows >= 112) {
      rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
      return;
    }
    if (path.rows >= 72) return lt(32, 0);
    if (path.rows >= 64) return lt(128, 2);
    if (path.rows == 4) {
      rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, 4, y);
      return;
    }
  } else {
    if (path.rows >= 1024) return lt(0, 0);
    if (path.rows >= 768) return lt(32, 1);
    if (path.rows >= 512) return lt(128, 3);
    if (path.rows >= 384) return lt(32, 0);
    if (path.rows >= 256) return lt(0, 0);
    if (path.rows >= 192) return lt(0, 3);
    if (path.rows >= 160) return lt(0, 2);
    if (path.rows >= 128) return lt(32, 0);
    if (path.rows >= 112) return lt(32, 3);
    if (path.rows >= 96) return lt(32, 1);
    if (path.rows >= 72) return lt(128, 1);
    if (path.rows >= 64) return lt(0, 0);
  }
  rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
}

void linear_rank_in_launch(
    cudaStream_t stream,
    int rows,
    int K,
    int N,
    const half* x,
    const half* weight,
    const half* weight_t,
    half* y) {
  if (rows <= kLowrankInRowsT) {
    rwkv7_v3a_linear_t_f16_launch(stream, rows, K, N, x, weight_t, y);
  } else {
    rwkv7_v3a_linear_f16_launch(stream, rows, K, N, x, weight, y);
  }
}

void linear_rank_out_launch(
    cudaStream_t stream,
    int rows,
    int K,
    int N,
    const half* x,
    const half* weight,
    const half* weight_t,
    half* y) {
  if (rows <= kLowrankOutRowsT) {
    rwkv7_v3a_linear_t_f16_launch(stream, rows, K, N, x, weight_t, y);
  } else {
    rwkv7_v3a_linear_f16_launch(stream, rows, K, N, x, weight, y);
  }
}

void linear_rank_out_act_launch(
    cudaStream_t stream,
    int rows,
    int K,
    int N,
    const half* x,
    const half* weight,
    const half* weight_t,
    int act,
    half* act_scratch,
    half* y) {
  if (rows <= kLowrankOutRowsT) {
    rwkv7_v3a_linear_t_act_f16_launch(stream, rows, K, N, x, weight_t, act, y);
    return;
  }
  if (act == 1) {
    rwkv7_act_tanh_launch(stream, x, act_scratch, static_cast<long long>(rows) * K);
  } else {
    rwkv7_act_sigmoid_launch(stream, x, act_scratch, static_cast<long long>(rows) * K);
  }
  rwkv7_v3a_linear_f16_launch(stream, rows, K, N, act_scratch, weight, y);
}

}  // namespace

namespace rwkv7_server {
namespace {

std::string basename_without_extension(const std::string& path) {
  const std::filesystem::path fs_path(path);
  return fs_path.stem().string();
}

CudaWeights load_backend_weights(const std::string& model_path) {
  auto archive = llm_infer::PthArchive::open(model_path);
  require_result(archive.ok(), archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  require_result(records.ok(), records.status().message());
  std::unordered_map<std::string, const llm_infer::TensorRecord*> by_name;
  for (const auto& rec : records.value()) {
    by_name.emplace(rec.name, &rec);
  }
  return load_model_weights(archive.value(), by_name);
}

std::vector<float> run_backend_forward(
    const CudaWeights& weights,
    const std::vector<std::vector<int64_t>>& token_batches,
    GenerationState& state) {
  const int B = static_cast<int>(token_batches.size());
  if (B <= 0) {
    throw std::runtime_error("token_batches must not be empty");
  }
  if (state.batch_size != B) {
    throw std::runtime_error("state batch size mismatch");
  }

  const int T = static_cast<int>(token_batches.front().size());
  if (T <= 0) {
    throw std::runtime_error("token batch length must be positive");
  }
  for (const auto& batch : token_batches) {
    if (static_cast<int>(batch.size()) != T) {
      throw std::runtime_error("all token batches must share the same length");
    }
  }

  Case run;
  run.B = B;
  run.T = T;
  const PathConfig path = select_path(run);
  const int rows = B * T;
  const int output_rows = B;
  constexpr int C = kChannels;
  constexpr int H = kHeads;
  constexpr int N = kHeadSize;

  HalfArena arena;
  arena.allocate(static_cast<std::size_t>(rows) * C * 31 + static_cast<std::size_t>(output_rows) * C +
                 static_cast<std::size_t>(rows) * kFfn +
                 static_cast<std::size_t>(rows) * kLowrankMax * 4 + static_cast<std::size_t>(output_rows) * kVocab);

  DeviceBuffer<unsigned char> lt_workspace;
  lt_workspace.resize(static_cast<std::size_t>(128) << 20, "alloc backend cublasLt workspace");

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create backend stream");

  if (weights.cpu_emb_ln0_f16.size() != static_cast<std::size_t>(kVocab) * C) {
    check_cuda(cudaStreamDestroy(stream), "destroy backend stream");
    throw std::runtime_error("cpu emb+ln0 table is not ready");
  }

  std::vector<std::uint16_t> host_x(static_cast<std::size_t>(rows) * C);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const int token_id = static_cast<int>(token_batches[static_cast<size_t>(b)][static_cast<size_t>(t)]);
      if (token_id < 0 || token_id >= kVocab) {
        check_cuda(cudaStreamDestroy(stream), "destroy backend stream");
        throw std::runtime_error("token id out of range");
      }
      const std::size_t row = static_cast<std::size_t>(b) * T + t;
      const std::uint16_t* src = weights.cpu_emb_ln0_f16.data() + static_cast<std::size_t>(token_id) * C;
      std::copy(src, src + C, host_x.data() + row * C);
    }
  }

  const std::size_t row_elems = static_cast<std::size_t>(rows) * C;
  half* x0 = arena.take(row_elems, "x0");
  half* x1 = arena.take(row_elems, "x1");
  half* xx0 = arena.take(row_elems, "xx0");
  half* xx1 = arena.take(row_elems, "xx1");
  half* xr = arena.take(row_elems, "xr");
  half* xw = arena.take(row_elems, "xw");
  half* xk = arena.take(row_elems, "xk");
  half* xv = arena.take(row_elems, "xv");
  half* xa = arena.take(row_elems, "xa");
  half* xg = arena.take(row_elems, "xg");
  half* r = arena.take(row_elems, "r");
  half* k = arena.take(row_elems, "k");
  half* v_base = arena.take(row_elems, "v_base");
  half* v_first = arena.take(row_elems, "v_first");
  half* v_out = arena.take(row_elems, "v_out");
  half* w1 = arena.take(static_cast<std::size_t>(rows) * kLowrankMax, "w1");
  half* a1 = arena.take(static_cast<std::size_t>(rows) * kLowrankMax, "a1");
  half* g1 = arena.take(static_cast<std::size_t>(rows) * kLowrankMax, "g1");
  half* v1 = arena.take(static_cast<std::size_t>(rows) * kLowrankMax, "v1");
  half* w12 = arena.take(row_elems, "w12");
  half* a12 = arena.take(row_elems, "a12");
  half* g = arena.take(row_elems, "g");
  half* k2 = arena.take(row_elems, "k2");
  half* neg_kk = arena.take(row_elems, "neg_kk");
  half* kka = arena.take(row_elems, "kka");
  half* w_raw = arena.take(row_elems, "w_raw");
  half* y = arena.take(row_elems, "wkv_y");
  half* y2 = arena.take(row_elems, "tmix_out");
  half* att_out = arena.take(row_elems, "att_out");
  half* x_after_att = arena.take(row_elems, "x_after_att");
  half* ln2_out = arena.take(row_elems, "ln2_out");
  half* mixed = arena.take(row_elems, "cmix_mixed");
  half* hid = arena.take(static_cast<std::size_t>(rows) * kFfn, "ffn_hid");
  half* cmix_out = arena.take(row_elems, "cmix_out");
  half* final_x = arena.take(static_cast<std::size_t>(output_rows) * C, "final_x");
  half* logits = arena.take(static_cast<std::size_t>(output_rows) * kVocab, "logits");

  check_cuda(cudaMemcpyAsync(x0, host_x.data(), host_x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream),
             "copy backend emb rows");

  rwkv7_v3a_layer_norm_f16_launch(
      stream, rows, C, x0, hp(weights.layers[0].ln1_w), hp(weights.layers[0].ln1_b), xx0, 1.0e-5f);
  half* x_cur = x0;
  half* xx_cur = xx0;
  half* x_next = x1;
  half* xx_next = xx1;
  bool pre_mix_ready = false;

  for (int layer = 0; layer < kLayers; ++layer) {
    const LayerWeights& w = weights.layers[layer];
    const int Rw = static_cast<int>(w.att_w1_t->shape[0]);
    const int Ra = static_cast<int>(w.att_a1_t->shape[0]);
    const int Rg = static_cast<int>(w.att_g1_t->shape[0]);
    const int Rv = (layer == 0) ? 0 : static_cast<int>(w.att_v1_t->shape[0]);
    if (Rw > kLowrankMax || Ra > kLowrankMax || Rg > kLowrankMax || Rv > kLowrankMax) {
      check_cuda(cudaStreamDestroy(stream), "destroy backend stream");
      throw std::runtime_error("lowrank exceeds arena max");
    }

    half* shift0 = state.shift.p + static_cast<std::size_t>(layer) * 2 * B * C;
    half* shift1 = shift0 + static_cast<std::size_t>(B) * C;
    half* layer_state = state.wkv_state.p + static_cast<std::size_t>(layer) * B * H * N * N;

    if (!pre_mix_ready) {
      rwkv7_tmix_mix6_launch(
          stream,
          B,
          T,
          C,
          xx_cur,
          shift0,
          hp(w.att_x_r),
          hp(w.att_x_w),
          hp(w.att_x_k),
          hp(w.att_x_v),
          hp(w.att_x_a),
          hp(w.att_x_g),
          xr,
          xw,
          xk,
          xv,
          xa,
          xg);
    } else {
      pre_mix_ready = false;
    }

    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xr, hp(w.att_receptance_w), lt_workspace.p, lt_workspace.n, r);
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xk, hp(w.att_key_w), lt_workspace.p, lt_workspace.n, k);
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xv, hp(w.att_value_w), lt_workspace.p, lt_workspace.n, v_base);
    half* v_use = v_base;
    bool v_done = false;

    if (rows <= kLowrankInRowsT && rows <= kLowrankOutRowsT && layer != 0) {
      rwkv7_v3a_linear_wagv_rank_in_f16_launch(
          stream, rows, C, Rw, Ra, Rg, Rv, xw, xa, xg, xv,
          hp(w.att_w1_t), hp(w.att_a1_t), hp(w.att_g1_t), hp(w.att_v1_t), w1, a1, g1, v1);
    } else if (rows <= kLowrankInRowsT) {
      rwkv7_v3a_linear_wag_rank_in_f16_launch(
          stream, rows, C, Rw, Ra, Rg, xw, xa, xg, hp(w.att_w1_t), hp(w.att_a1_t), hp(w.att_g1_t), w1, a1, g1);
    } else {
      linear_rank_in_launch(stream, rows, C, Rw, xw, hp(w.att_w1), hp(w.att_w1_t), w1);
      linear_rank_in_launch(stream, rows, C, Ra, xa, hp(w.att_a1), hp(w.att_a1_t), a1);
      linear_rank_in_launch(stream, rows, C, Rg, xg, hp(w.att_g1), hp(w.att_g1_t), g1);
    }

    if (rows <= kLowrankOutRowsT && layer != 0 && rows <= kLowrankInRowsT) {
      rwkv7_v3a_linear_wagv_rank_out_f16_launch(
          stream, rows, C, Rw, Ra, Rg, Rv, w1, a1, g1, v1,
          hp(w.att_w2_t), hp(w.att_a2_t), hp(w.att_g2_t), hp(w.att_v2_t),
          v_base, v_first, hp(w.att_v0), w12, a12, g, v_out);
      v_use = v_out;
      v_done = true;
    } else if (rows <= kLowrankOutRowsT) {
      rwkv7_v3a_linear_wag_rank_out_f16_launch(
          stream, rows, C, Rw, Ra, Rg, w1, a1, g1, hp(w.att_w2_t), hp(w.att_a2_t), hp(w.att_g2_t), w12, a12, g);
    } else {
      linear_rank_out_act_launch(stream, rows, Rw, C, w1, hp(w.att_w2), hp(w.att_w2_t), 1, w_raw, w12);
      linear_rank_out_launch(stream, rows, Ra, C, a1, hp(w.att_a2), hp(w.att_a2_t), a12);
      linear_rank_out_act_launch(stream, rows, Rg, C, g1, hp(w.att_g2), hp(w.att_g2_t), 2, w_raw, g);
    }

    if (layer == 0) {
      check_cuda(cudaMemcpyAsync(v_first, v_base, row_elems * sizeof(half), cudaMemcpyDeviceToDevice, stream), "copy backend v_first");
    } else if (!v_done) {
      if (rows <= kLowrankOutRowsT) {
        if (rows > kLowrankInRowsT) {
          linear_rank_in_launch(stream, rows, C, Rv, xv, hp(w.att_v1), hp(w.att_v1_t), v1);
        }
        rwkv7_v3a_linear_t_vres_f16_launch(stream, rows, Rv, C, v1, hp(w.att_v2_t), v_base, v_first, hp(w.att_v0), v_out);
      } else {
        linear_rank_in_launch(stream, rows, C, Rv, xv, hp(w.att_v1), hp(w.att_v1_t), v1);
        linear_rank_out_launch(stream, rows, Rv, C, v1, hp(w.att_v2), hp(w.att_v2_t), w_raw);
        rwkv7_tmix_vres_gate_launch(stream, B, T, C, v_base, v_first, hp(w.att_v0), w_raw, v_out);
      }
      v_use = v_out;
    }

    rwkv7_tmix_kk_a_gate_launch(stream, B, T, C, H, k, hp(w.att_k_k), hp(w.att_a0), a12, hp(w.att_k_a), k2, neg_kk, kka);
    if (T <= 16) {
      rwkv7_wkv_fp16_seq_w0_launch(stream, B, T, C, H, layer_state, r, w12, hp(w.att_w0), k2, v_use, neg_kk, kka, y, state.elapsed.p);
    } else {
      rwkv7_add_vec_launch(stream, C, w12, hp(w.att_w0), w_raw, row_elems);
      rwkv7_wkv_fp16_seq_launch(stream, B, T, C, H, layer_state, r, w_raw, k2, v_use, neg_kk, kka, y, state.elapsed.p);
    }

    rwkv7_tmix_lnx_rkvres_xg_launch(stream, B, T, C, H, y, r, k2, v_use, hp(w.att_r_k), hp(w.att_ln_x_w), hp(w.att_ln_x_b), g, y2);
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, y2, hp(w.att_output_w), lt_workspace.p, lt_workspace.n, att_out);
    if (T == 1) {
      rwkv7_v3a_add_layer_norm_cmix_mix_f16_launch(
          stream, rows, x_cur, att_out, shift1, hp(w.ln2_w), hp(w.ln2_b), hp(w.ffn_x_k), x_after_att, mixed, 1.0e-5f);
    } else {
      rwkv7_v3a_add_layer_norm_f16_launch(
          stream, rows, C, x_cur, att_out, hp(w.ln2_w), hp(w.ln2_b), x_after_att, ln2_out, 1.0e-5f);
      rwkv7_cmix_mix_launch(stream, B, T, C, ln2_out, shift1, hp(w.ffn_x_k), mixed);
    }

    linear_orig_layout_launch(stream, path, LinearGroup::FfnKey, rows, C, kFfn, mixed, hp(w.ffn_key_w), lt_workspace.p, lt_workspace.n, hid);
    if (path.cmix == CmixMode::NoFcOne) {
      rwkv7_cmix_sparse_down_relu_one_launch(stream, C, kFfn, hid, hp(w.ffn_value_w), cmix_out);
    } else if (path.cmix == CmixMode::NoFcRows2) {
      if (rows >= 8) {
        rwkv7_cmix_sparse_down_relu_rows_t512_launch(stream, B, T, C, kFfn, hid, hp(w.ffn_value_w), cmix_out);
      } else {
        rwkv7_cmix_sparse_down_relu_rows_launch(stream, B, T, C, kFfn, hid, hp(w.ffn_value_w), cmix_out);
      }
    } else {
      rwkv7_relu_square_launch(stream, hid, hid, static_cast<long long>(rows) * kFfn);
      rwkv7_v3a_linear_f16_launch(stream, rows, kFfn, C, hid, hp(w.ffn_value_w), cmix_out);
    }

    if (layer + 1 < kLayers) {
      const LayerWeights& next = weights.layers[layer + 1];
      if (B == 1 && T == 1) {
        half* next_shift0 = state.shift.p + static_cast<std::size_t>(layer + 1) * 2 * B * C;
        rwkv7_v3a_add_layer_norm_tmix_mix6_f16_launch(
            stream, rows, x_after_att, cmix_out, next_shift0, hp(next.ln1_w), hp(next.ln1_b),
            hp(next.att_x_r), hp(next.att_x_w), hp(next.att_x_k), hp(next.att_x_v), hp(next.att_x_a), hp(next.att_x_g),
            x_next, xr, xw, xk, xv, xa, xg, 1.0e-5f);
        xx_next = x_next;
        pre_mix_ready = true;
      } else {
        rwkv7_v3a_add_layer_norm_f16_launch(stream, rows, C, x_after_att, cmix_out, hp(next.ln1_w), hp(next.ln1_b), x_next, xx_next, 1.0e-5f);
      }
      std::swap(x_cur, x_next);
      std::swap(xx_cur, xx_next);
    } else {
      rwkv7_v3a_add_last_layer_norm_f16_launch(
          stream, B, T, x_after_att, cmix_out, hp(weights.ln_out_w), hp(weights.ln_out_b), final_x, 1.0e-5f);
    }
    check_cuda(cudaGetLastError(), "launch backend layer");
  }

  rwkv7_v3a_advance_i32_launch(stream, state.elapsed.p, T, B);
  PathConfig head_path;
  head_path.rows = output_rows;
  head_path.use_batched_rkv = false;
  head_path.cmix = CmixMode::Dense;
  linear_orig_layout_launch(stream, head_path, LinearGroup::Head, output_rows, C, kVocab, final_x, hp(weights.head_w), lt_workspace.p, lt_workspace.n, logits);
  check_cuda(cudaGetLastError(), "launch backend head");
  check_cuda(cudaStreamSynchronize(stream), "sync backend stream");

  std::vector<half> host_logits(static_cast<std::size_t>(output_rows) * kVocab);
  check_cuda(
      cudaMemcpy(host_logits.data(), logits, host_logits.size() * sizeof(half), cudaMemcpyDeviceToHost),
      "copy backend logits");
  check_cuda(cudaStreamDestroy(stream), "destroy backend stream");

  std::vector<float> out(host_logits.size());
  for (size_t i = 0; i < host_logits.size(); ++i) {
    out[i] = __half2float(host_logits[i]);
  }
  return out;
}

}  // namespace

struct ModelBackend::Impl {
  explicit Impl(std::string path)
      : model_path(std::move(path)),
        model_name(basename_without_extension(model_path)),
        weights(load_backend_weights(model_path)) {}

  std::string model_path;
  std::string model_name;
  CudaWeights weights;
};

ModelBackend::ModelBackend(std::string model_path) : impl_(std::make_unique<Impl>(std::move(model_path))) {}

ModelBackend::~ModelBackend() = default;

ModelBackend::ModelBackend(ModelBackend&&) noexcept = default;

ModelBackend& ModelBackend::operator=(ModelBackend&&) noexcept = default;

GenerationState ModelBackend::create_state(int batch_size) const {
  if (batch_size <= 0) {
    throw std::runtime_error("batch_size must be positive");
  }
  GenerationState state;
  state.batch_size = batch_size;
  const std::size_t shift_elems = static_cast<std::size_t>(kLayers) * 2 * batch_size * kChannels;
  const std::size_t state_elems =
      static_cast<std::size_t>(kLayers) * batch_size * kHeads * kHeadSize * kHeadSize;
  state.shift.resize(shift_elems, "alloc backend shift state");
  state.wkv_state.resize(state_elems, "alloc backend wkv state");
  state.elapsed.resize(batch_size, "alloc backend elapsed");
  state.shift.zero("zero backend shift");
  state.wkv_state.zero("zero backend wkv");
  check_cuda(cudaMemset(state.elapsed.p, 0, static_cast<std::size_t>(batch_size) * sizeof(int)), "zero backend elapsed");
  return state;
}

std::vector<float> ModelBackend::forward_prefill(
    const std::vector<std::vector<int64_t>>& token_batches,
    GenerationState& state) const {
  return run_backend_forward(impl_->weights, token_batches, state);
}

std::vector<float> ModelBackend::forward_decode(
    const std::vector<int64_t>& token_batch,
    GenerationState& state) const {
  std::vector<std::vector<int64_t>> batch;
  batch.reserve(token_batch.size());
  for (int64_t token : token_batch) {
    batch.push_back({token});
  }
  return run_backend_forward(impl_->weights, batch, state);
}

int ModelBackend::vocab_size() const {
  return kVocab;
}

const std::string& ModelBackend::model_path() const {
  return impl_->model_path;
}

const std::string& ModelBackend::model_name() const {
  return impl_->model_name;
}

}  // namespace rwkv7_server
