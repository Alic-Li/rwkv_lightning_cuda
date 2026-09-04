#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "pth_archive.hpp"
#include "pth_tensor.hpp"
#include "rwkv_quantized.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv7_fast_v4_common.hpp"
#include "rwkv7_fast_v4_kernels.cuh"

namespace {

using namespace rwkv7_fast_v4;

constexpr std::size_t kWeightLoadChunkBytes = 32u << 20;
constexpr int kChunkLoadLayerBatch = 4;
constexpr int kDefaultQuantizedCmixSparseMaxRows = 32;

struct TensorStorageSource {
  const llm_infer::PthEntry* entry = nullptr;
  std::uint64_t byte_offset = 0;
  std::uint64_t elems = 0;
};

TensorStorageSource tensor_storage_source(
    const llm_infer::PthArchive& archive,
    const llm_infer::TensorRecord& rec) {
  if (!is_contiguous_shape(rec.shape, rec.stride)) {
    std::cerr << "error: v4 GPU loader currently requires contiguous tensor: " << rec.name << std::endl;
    std::exit(1);
  }
  const std::string prefix = archive_prefix(archive);
  if (prefix.empty()) {
    std::cerr << "error: archive prefix not found\n";
    std::exit(1);
  }
  const auto* entry = archive.find_entry(prefix + "/data/" + rec.storage_key);
  if (!entry) {
    std::cerr << "error: storage entry not found for tensor: " << rec.name << std::endl;
    std::exit(1);
  }
  if (!entry->is_stored()) {
    std::cerr << "error: compressed tensor storage is unsupported: " << rec.name << std::endl;
    std::exit(1);
  }
  if (rec.storage_size * sizeof(std::uint16_t) != entry->uncompressed_size) {
    std::cerr << "error: storage byte size mismatch for tensor: " << rec.name << std::endl;
    std::exit(1);
  }
  const std::uint64_t n = numel(rec.shape);
  if (rec.storage_offset + n > rec.storage_size) {
    std::cerr << "error: tensor data range exceeds storage: " << rec.name << std::endl;
    std::exit(1);
  }
  return {entry, rec.storage_offset * sizeof(std::uint16_t), n};
}

struct RawBf16TensorView {
  const std::uint16_t* data = nullptr;
  std::uint64_t elems = 0;
};

RawBf16TensorView raw_bf16_tensor_view(const llm_infer::PthArchive& archive, const llm_infer::TensorRecord& rec) {
  const TensorStorageSource source = tensor_storage_source(archive, rec);
  auto view = archive.stored_entry_view(*source.entry);
  require_result(view.ok(), view.status().message());
  return {
      reinterpret_cast<const std::uint16_t*>(view.value().data + source.byte_offset),
      source.elems};
}

struct PinnedHostBuffer {
  std::uint8_t* p = nullptr;
  std::size_t n = 0;

  PinnedHostBuffer() = default;
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  ~PinnedHostBuffer() {
    release();
  }

  void release() {
    if (p) cudaFreeHost(p);
    p = nullptr;
    n = 0;
  }

  void resize(std::size_t bytes) {
    if (bytes <= n) return;
    if (p) check_cuda(cudaFreeHost(p), "free chunk-load host buffer");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocPortable),
               "alloc chunk-load host buffer");
    n = bytes;
  }
};

struct WeightLoadPipeline {
  struct Slot {
    DeviceBuffer<std::uint16_t> staging;
    cudaStream_t copy = nullptr;
    cudaStream_t compute = nullptr;
    cudaEvent_t copied = nullptr;
    cudaEvent_t done = nullptr;
    bool in_flight = false;
    PinnedHostBuffer host;
  };

  Slot slots[2];
  int next = 0;

  explicit WeightLoadPipeline(bool chunk_load) {
    for (auto& s : slots) {
      check_cuda(cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking), "create weight copy stream");
      check_cuda(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking), "create weight compute stream");
      check_cuda(cudaEventCreateWithFlags(&s.copied, cudaEventDisableTiming), "create weight copied event");
      check_cuda(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "create weight done event");
      if (chunk_load) {
        s.host.resize(kWeightLoadChunkBytes);
        s.staging.resize(
            kWeightLoadChunkBytes / sizeof(std::uint16_t),
            "alloc chunk-load device staging");
      }
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

  Slot& acquire_chunk(std::size_t bytes) {
    Slot& s = acquire((bytes + sizeof(std::uint16_t) - 1) / sizeof(std::uint16_t));
    s.host.resize(bytes);
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

  void release_buffers() {
    sync();
    for (auto& s : slots) {
      s.host.release();
      s.staging = DeviceBuffer<std::uint16_t>{};
    }
  }
};

std::unique_ptr<GpuTensor> allocate_tensor_f16_like_v3a(
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
    const std::string& key,
    bool required) {
  auto it = by_name.find(key);
  if (it == by_name.end()) {
    if (required) {
      std::cerr << "error: missing tensor: " << key << std::endl;
      std::exit(1);
    }
    return nullptr;
  }
  const auto& rec = *it->second;
  std::vector<std::int64_t> runtime_shape = rec.shape;
  const bool transpose = should_transpose_like_v3a(rec.name);
  if (transpose) {
    if (rec.shape.size() != 2) {
      std::cerr << "error: v3a transpose rule requires 2D tensor: " << rec.name << std::endl;
      std::exit(1);
    }
    runtime_shape = {rec.shape[1], rec.shape[0]};
  }
  auto tensor = std::make_unique<GpuTensor>();
  tensor->name = rec.name;
  tensor->shape = std::move(runtime_shape);
  tensor->f16.resize(static_cast<std::size_t>(numel(rec.shape)), "alloc weight tensor");
  return tensor;
}

void upload_tensor_f16_like_v3a(
    const llm_infer::PthArchive& archive,
    const llm_infer::TensorRecord& rec,
    GpuTensor& tensor,
    WeightLoadPipeline& pipeline) {
  const TensorStorageSource source = tensor_storage_source(archive, rec);
  const bool transpose = should_transpose_like_v3a(rec.name);
  if (archive.chunk_load()) {
    if (transpose) {
      const int rows = static_cast<int>(rec.shape[0]);
      const int cols = static_cast<int>(rec.shape[1]);
      const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(std::uint16_t);
      const int rows_per_chunk = static_cast<int>(
          std::max<std::size_t>(1, kWeightLoadChunkBytes / row_bytes));
      for (int row_offset = 0; row_offset < rows; row_offset += rows_per_chunk) {
        const int chunk_rows = std::min(rows_per_chunk, rows - row_offset);
        const std::size_t chunk_bytes = static_cast<std::size_t>(chunk_rows) * row_bytes;
        auto& slot = pipeline.acquire_chunk(chunk_bytes);
        const auto status = archive.read_stored_entry_range(
            *source.entry,
            source.byte_offset + static_cast<std::uint64_t>(row_offset) * row_bytes,
            slot.host.p,
            chunk_bytes);
        require_result(status.ok_status(), status.message());
        check_cuda(cudaMemcpyAsync(slot.staging.p, slot.host.p, chunk_bytes,
                                   cudaMemcpyHostToDevice, slot.copy), "copy bf16 weight chunk");
        check_cuda(cudaEventRecord(slot.copied, slot.copy), "record bf16 chunk copied");
        check_cuda(cudaStreamWaitEvent(slot.compute, slot.copied, 0), "wait bf16 chunk copied");
        rwkv7_v4_bf16_to_f16_transpose_rows_launch(
            slot.compute, slot.staging.p, tensor.f16.p,
            chunk_rows, cols, rows, row_offset);
        check_cuda(cudaGetLastError(), "launch chunked bf16 transpose");
        check_cuda(cudaEventRecord(slot.done, slot.compute), "record chunk preprocess done");
        slot.in_flight = true;
      }
    } else {
      const std::size_t max_chunk_elems = kWeightLoadChunkBytes / sizeof(std::uint16_t);
      for (std::uint64_t elem_offset = 0; elem_offset < source.elems;) {
        const std::size_t chunk_elems = static_cast<std::size_t>(
            std::min<std::uint64_t>(max_chunk_elems, source.elems - elem_offset));
        const std::size_t chunk_bytes = chunk_elems * sizeof(std::uint16_t);
        auto& slot = pipeline.acquire_chunk(chunk_bytes);
        const auto status = archive.read_stored_entry_range(
            *source.entry,
            source.byte_offset + elem_offset * sizeof(std::uint16_t),
            slot.host.p,
            chunk_bytes);
        require_result(status.ok_status(), status.message());
        check_cuda(cudaMemcpyAsync(slot.staging.p, slot.host.p, chunk_bytes,
                                   cudaMemcpyHostToDevice, slot.copy), "copy bf16 weight chunk");
        check_cuda(cudaEventRecord(slot.copied, slot.copy), "record bf16 chunk copied");
        check_cuda(cudaStreamWaitEvent(slot.compute, slot.copied, 0), "wait bf16 chunk copied");
        rwkv7_v4_bf16_to_f16_launch(
            slot.compute, slot.staging.p, tensor.f16.p + elem_offset, chunk_elems);
        check_cuda(cudaGetLastError(), "launch chunked bf16 preprocess");
        check_cuda(cudaEventRecord(slot.done, slot.compute), "record chunk preprocess done");
        slot.in_flight = true;
        elem_offset += chunk_elems;
      }
    }
  } else {
    const RawBf16TensorView raw = raw_bf16_tensor_view(archive, rec);
    auto& slot = pipeline.acquire(static_cast<std::size_t>(raw.elems));
    check_cuda(cudaMemcpyAsync(slot.staging.p, raw.data, raw.elems * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, slot.copy), "copy raw bf16 weight");
    check_cuda(cudaEventRecord(slot.copied, slot.copy), "record raw bf16 copied");
    check_cuda(cudaStreamWaitEvent(slot.compute, slot.copied, 0), "wait raw bf16 copied");
    if (transpose) {
      rwkv7_v4_bf16_to_f16_transpose_launch(
          slot.compute, slot.staging.p, tensor.f16.p,
          static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
    } else {
      rwkv7_v4_bf16_to_f16_launch(slot.compute, slot.staging.p, tensor.f16.p, raw.elems);
    }
    check_cuda(cudaGetLastError(), "launch bf16 weight preprocess");
    check_cuda(cudaEventRecord(slot.done, slot.compute), "record weight preprocess done");
    slot.in_flight = true;
  }
}

std::string block_key(int layer, const char* suffix) {
  return "blocks." + std::to_string(layer) + "." + suffix;
}

struct CudaWeights {
  struct CmixStatsBuffers {
    bool enabled = false;
    DeviceBuffer<unsigned long long> nonzero;
    DeviceBuffer<unsigned long long> total;
    DeviceBuffer<unsigned int> max_bits;

    void initialize(int layers) {
      enabled = true;
      nonzero.resize(static_cast<std::size_t>(layers), "alloc cmix stats nonzero");
      total.resize(static_cast<std::size_t>(layers), "alloc cmix stats total");
      max_bits.resize(static_cast<std::size_t>(layers), "alloc cmix stats max");
      nonzero.zero("zero cmix stats nonzero");
      total.zero("zero cmix stats total");
      max_bits.zero("zero cmix stats max");
      check_cuda(cudaDeviceSynchronize(), "sync cmix stats init");
    }
  };

  ModelDims dims;
  std::unordered_map<std::string, std::unique_ptr<GpuTensor>> tensors;
  std::vector<LayerWeights> layers;
  const GpuTensor* ln_out_w = nullptr;
  const GpuTensor* ln_out_b = nullptr;
  const GpuTensor* head_w = nullptr;
  int optional_loaded = 0;
  int t_copy_count = 0;
  std::size_t cpu_emb_bytes = 0;
  std::vector<std::uint16_t> cpu_emb_ln0_f16;
  CmixStatsBuffers cmix_stats;
  int cmix_sparse_max_rows = kDefaultQuantizedCmixSparseMaxRows;
  std::vector<int> cmix_sparse_max_rows_by_layer;

  const GpuTensor* optional(const std::string& key) const {
    auto it = tensors.find(key);
    return it == tensors.end() ? nullptr : it->second.get();
  }

  const GpuTensor* require(const std::string& key) const {
    const GpuTensor* t = optional(key);
    if (!t) {
      std::cerr << "error: tensor view missing: " << key << std::endl;
      std::exit(1);
    }
    return t;
  }

  GpuTensor* require_mutable(const std::string& key) {
    auto it = tensors.find(key);
    if (it == tensors.end()) {
      std::cerr << "error: mutable tensor view missing: " << key << std::endl;
      std::exit(1);
    }
    return it->second.get();
  }

  bool allocate(
      const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
      const std::string& key,
      bool required) {
    auto tensor = allocate_tensor_f16_like_v3a(by_name, key, required);
    if (!tensor) return false;
    tensors.emplace(key, std::move(tensor));
    if (!required) ++optional_loaded;
    return true;
  }

  void upload(
      const llm_infer::PthArchive& archive,
      const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
      const std::string& key,
      WeightLoadPipeline& pipeline) {
    auto it = by_name.find(key);
    if (it == by_name.end()) {
      std::cerr << "error: tensor record missing during upload: " << key << std::endl;
      std::exit(1);
    }
    upload_tensor_f16_like_v3a(archive, *it->second, *require_mutable(key), pipeline);
  }

  void load(
      const llm_infer::PthArchive& archive,
      const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
      const std::string& key,
      bool required,
      WeightLoadPipeline& pipeline) {
    if (!allocate(by_name, key, required)) return;
    upload(archive, by_name, key, pipeline);
  }

  void allocate_t_copy(const std::string& key) {
    const GpuTensor* src = require(key);
    if (src->shape.size() != 2) {
      std::cerr << "error: .t copy requires 2D tensor: " << key << std::endl;
      std::exit(1);
    }
    const int rows = static_cast<int>(src->shape[0]);
    const int cols = static_cast<int>(src->shape[1]);
    auto tensor = std::make_unique<GpuTensor>();
    tensor->name = key + ".t";
    tensor->shape = {cols, rows};
    tensor->f16.resize(static_cast<std::size_t>(rows) * cols, "alloc .t tensor");
    const std::string tensor_name = tensor->name;
    tensors.emplace(tensor_name, std::move(tensor));
    ++t_copy_count;
  }

  void launch_t_copy(const std::string& key) {
    const GpuTensor* src = require(key);
    const GpuTensor* dst = require(key + ".t");
    const int rows = static_cast<int>(src->shape[0]);
    const int cols = static_cast<int>(src->shape[1]);
    rwkv7_v4_f16_transpose_launch(nullptr, src->f16.p, dst->f16.p, rows, cols);
    check_cuda(cudaGetLastError(), "launch .t transpose");
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
    if (head_w->shape.size() != 2 ||
        static_cast<int>(head_w->shape[0]) != dims.vocab ||
        static_cast<int>(head_w->shape[1]) != dims.channels) {
      std::cerr << "error: head.weight shape mismatch"
                << " expected=[" << dims.vocab << "," << dims.channels << "]"
                << " actual=[";
      if (!head_w->shape.empty()) {
        std::cerr << head_w->shape[0];
        if (head_w->shape.size() > 1) {
          std::cerr << "," << head_w->shape[1];
        }
      }
      std::cerr << "]\n";
      std::exit(1);
    }
  }
};

struct BackendProfile {
  struct Event {
    std::string name;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
  };

  explicit BackendProfile(bool enabled_) : enabled(enabled_) {}

  int begin(cudaStream_t stream, const char* name) {
    if (!enabled) return -1;
    Event event;
    event.name = name;
    check_cuda(cudaEventCreate(&event.start), "create profile start event");
    check_cuda(cudaEventCreate(&event.stop), "create profile stop event");
    check_cuda(cudaEventRecord(event.start, stream), "record profile start event");
    events.push_back(std::move(event));
    return static_cast<int>(events.size() - 1);
  }

  void end(cudaStream_t stream, int index) {
    if (index < 0) return;
    check_cuda(cudaEventRecord(events[static_cast<std::size_t>(index)].stop, stream), "record profile stop event");
  }

  void report() {
    if (!enabled) return;
    std::unordered_map<std::string, double> totals;
    for (auto& event : events) {
      float elapsed = 0.0f;
      check_cuda(cudaEventElapsedTime(&elapsed, event.start, event.stop), "profile event elapsed");
      totals[event.name] += elapsed;
      cudaEventDestroy(event.start);
      cudaEventDestroy(event.stop);
    }
    for (const auto& item : totals) {
      std::cout << "profile kernel=" << item.first << " ms=" << item.second << "\n";
    }
    events.clear();
  }

  bool enabled = false;
  std::vector<Event> events;
};

struct ForwardResources {
  HalfArena arena;
  DeviceBuffer<unsigned char> lt_workspace;
  cudaStream_t stream = nullptr;

  ~ForwardResources() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  void ensure_stream() {
    if (stream == nullptr) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create backend stream");
    }
  }
};

void build_cpu_emb_ln0_f16(
    CudaWeights& weights,
    const ModelDims& dims,
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name) {
  if (weights.layers.empty() || !weights.layers[0].ln0_w || !weights.layers[0].ln0_b) {
    std::cerr << "error: layer0 ln0 weights are required before emb+ln0 preprocessing\n";
    std::exit(1);
  }
  auto record = [&](const std::string& key) -> const llm_infer::TensorRecord& {
    auto it = by_name.find(key);
    if (it == by_name.end()) {
      std::cerr << "error: missing tensor for emb+ln0 preprocessing: " << key << std::endl;
      std::exit(1);
    }
    return *it->second;
  };
  const auto& emb_rec = record("emb.weight");
  const auto& ln0_w_rec = record("blocks.0.ln0.weight");
  const auto& ln0_b_rec = record("blocks.0.ln0.bias");
  const std::size_t elems = static_cast<std::size_t>(dims.vocab) * dims.channels;

  if (archive.chunk_load()) {
    const TensorStorageSource emb = tensor_storage_source(archive, emb_rec);
    const TensorStorageSource ln0_w = tensor_storage_source(archive, ln0_w_rec);
    const TensorStorageSource ln0_b = tensor_storage_source(archive, ln0_b_rec);
    if (emb.elems != elems || ln0_w.elems != static_cast<std::size_t>(dims.channels) ||
        ln0_b.elems != static_cast<std::size_t>(dims.channels)) {
      std::cerr << "error: emb/ln0 shape mismatch for chunked emb+ln0 preprocessing\n";
      std::exit(1);
    }

    PinnedHostBuffer host;
    host.resize(kWeightLoadChunkBytes);
    DeviceBuffer<std::uint16_t> gpu_emb;
    DeviceBuffer<std::uint16_t> gpu_ln0_w;
    DeviceBuffer<std::uint16_t> gpu_ln0_b;
    DeviceBuffer<std::uint16_t> gpu_out;
    gpu_ln0_w.resize(ln0_w.elems, "alloc raw bf16 ln0 weight");
    gpu_ln0_b.resize(ln0_b.elems, "alloc raw bf16 ln0 bias");

    auto load_small = [&](const TensorStorageSource& source, std::uint16_t* destination,
                          const char* copy_label) {
      const std::size_t bytes = static_cast<std::size_t>(source.elems) * sizeof(std::uint16_t);
      host.resize(bytes);
      const auto status = archive.read_stored_entry_range(
          *source.entry, source.byte_offset, host.p, bytes);
      require_result(status.ok_status(), status.message());
      check_cuda(cudaMemcpy(destination, host.p, bytes, cudaMemcpyHostToDevice), copy_label);
    };
    load_small(ln0_w, gpu_ln0_w.p, "copy raw bf16 ln0 weight");
    load_small(ln0_b, gpu_ln0_b.p, "copy raw bf16 ln0 bias");

    const std::size_t row_bytes = static_cast<std::size_t>(dims.channels) * sizeof(std::uint16_t);
    const int rows_per_chunk = static_cast<int>(
        std::max<std::size_t>(1, kWeightLoadChunkBytes / row_bytes));
    const std::size_t max_chunk_elems =
        static_cast<std::size_t>(std::min(rows_per_chunk, dims.vocab)) * dims.channels;
    gpu_emb.resize(max_chunk_elems, "alloc chunked raw bf16 emb");
    gpu_out.resize(max_chunk_elems, "alloc chunked emb+ln0 output");
    weights.cpu_emb_ln0_f16.resize(elems);
    for (int row_offset = 0; row_offset < dims.vocab; row_offset += rows_per_chunk) {
      const int chunk_rows = std::min(rows_per_chunk, dims.vocab - row_offset);
      const std::size_t chunk_elems = static_cast<std::size_t>(chunk_rows) * dims.channels;
      const std::size_t chunk_bytes = chunk_elems * sizeof(std::uint16_t);
      host.resize(chunk_bytes);
      const auto status = archive.read_stored_entry_range(
          *emb.entry,
          emb.byte_offset + static_cast<std::uint64_t>(row_offset) * row_bytes,
          host.p,
          chunk_bytes);
      require_result(status.ok_status(), status.message());
      check_cuda(cudaMemcpy(gpu_emb.p, host.p, chunk_bytes, cudaMemcpyHostToDevice),
                 "copy chunked raw bf16 emb");
      rwkv7_v4_emb_ln0_bf16_to_f16_launch(
          nullptr, chunk_rows, dims.channels,
          gpu_emb.p, gpu_ln0_w.p, gpu_ln0_b.p, gpu_out.p, kLnEps);
      check_cuda(cudaGetLastError(), "launch chunked emb+ln0 preprocess");
      check_cuda(cudaMemcpy(
          weights.cpu_emb_ln0_f16.data() + static_cast<std::size_t>(row_offset) * dims.channels,
          gpu_out.p,
          chunk_bytes,
          cudaMemcpyDeviceToHost),
          "copy chunked emb+ln0 to CPU");
    }
    weights.cpu_emb_bytes = weights.cpu_emb_ln0_f16.size() * sizeof(std::uint16_t);
    return;
  }

  auto raw = [&](const std::string& key) -> RawBf16TensorView {
    return raw_bf16_tensor_view(archive, record(key));
  };
  RawBf16TensorView emb = raw("emb.weight");
  RawBf16TensorView ln0_w = raw("blocks.0.ln0.weight");
  RawBf16TensorView ln0_b = raw("blocks.0.ln0.bias");
  if (emb.elems != elems || ln0_w.elems != static_cast<std::size_t>(dims.channels) ||
      ln0_b.elems != static_cast<std::size_t>(dims.channels)) {
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
      nullptr, dims.vocab, dims.channels, gpu_emb.p, gpu_ln0_w.p, gpu_ln0_b.p, gpu_out.p, kLnEps);
  check_cuda(cudaGetLastError(), "launch emb+ln0 preprocess");
  weights.cpu_emb_ln0_f16.resize(elems);
  check_cuda(cudaMemcpy(weights.cpu_emb_ln0_f16.data(), gpu_out.p, elems * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost), "copy emb+ln0 to CPU");
  check_cuda(cudaDeviceSynchronize(), "sync emb+ln0 preprocess");
  weights.cpu_emb_bytes = weights.cpu_emb_ln0_f16.size() * sizeof(std::uint16_t);
}

void build_cpu_emb_ln0_f16_quantized(
    CudaWeights& weights,
    const ModelDims& dims,
    const llm_infer::QuantizedArchive& archive) {
  const auto* emb = archive.find("emb.weight");
  const auto* ln0_w = archive.find("blocks.0.ln0.weight");
  const auto* ln0_b = archive.find("blocks.0.ln0.bias");
  if (!emb || !ln0_w || !ln0_b) {
    std::cerr << "error: quantized model is missing emb/ln0 tensors\n";
    std::exit(1);
  }
  if (emb->dtype == llm_infer::QuantizedDType::kInt8 ||
      ln0_w->dtype == llm_infer::QuantizedDType::kInt8 ||
      ln0_b->dtype == llm_infer::QuantizedDType::kInt8) {
    std::cerr << "error: emb/ln0 tensors must remain floating point in quantized model\n";
    std::exit(1);
  }
  const std::size_t elems = static_cast<std::size_t>(dims.vocab) * dims.channels;
  std::vector<std::uint8_t> emb_data, w_data, b_data;
  require_result(archive.read_data(*emb, &emb_data).ok_status(), "read quantized emb");
  require_result(archive.read_data(*ln0_w, &w_data).ok_status(), "read quantized ln0 weight");
  require_result(archive.read_data(*ln0_b, &b_data).ok_status(), "read quantized ln0 bias");
  if (emb_data.size() != elems * 2 || w_data.size() != static_cast<std::size_t>(dims.channels) * 2 ||
      b_data.size() != static_cast<std::size_t>(dims.channels) * 2) {
    std::cerr << "error: quantized emb/ln0 shape mismatch\n";
    std::exit(1);
  }
  DeviceBuffer<std::uint16_t> gpu_emb, gpu_w, gpu_b, gpu_out;
  gpu_emb.resize(elems, "alloc quantized emb");
  gpu_w.resize(dims.channels, "alloc quantized ln0 weight");
  gpu_b.resize(dims.channels, "alloc quantized ln0 bias");
  gpu_out.resize(elems, "alloc quantized emb output");
  check_cuda(cudaMemcpy(gpu_emb.p, emb_data.data(), emb_data.size(), cudaMemcpyHostToDevice), "copy quantized emb");
  check_cuda(cudaMemcpy(gpu_w.p, w_data.data(), w_data.size(), cudaMemcpyHostToDevice), "copy quantized ln0 weight");
  check_cuda(cudaMemcpy(gpu_b.p, b_data.data(), b_data.size(), cudaMemcpyHostToDevice), "copy quantized ln0 bias");
  if (emb->dtype == llm_infer::QuantizedDType::kBFloat16) {
    rwkv7_v4_emb_ln0_bf16_to_f16_launch(nullptr, dims.vocab, dims.channels,
                                         gpu_emb.p, gpu_w.p, gpu_b.p, gpu_out.p, kLnEps);
  } else {
    std::cerr << "error: float16 emb preprocessing is not supported\n";
    std::exit(1);
  }
  check_cuda(cudaGetLastError(), "launch quantized emb+ln0 preprocess");
  weights.cpu_emb_ln0_f16.resize(elems);
  check_cuda(cudaMemcpy(weights.cpu_emb_ln0_f16.data(), gpu_out.p, elems * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost), "copy quantized emb+ln0");
  check_cuda(cudaDeviceSynchronize(), "sync quantized emb+ln0 preprocess");
  weights.cpu_emb_bytes = weights.cpu_emb_ln0_f16.size() * sizeof(std::uint16_t);
}

template <typename Fn>
void for_each_layer_source_tensor(int layer, Fn&& fn) {
  const char* required[] = {
      "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
      "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g",
      "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight",
      "att.w0", "att.w1", "att.w2", "att.a0", "att.a1", "att.a2", "att.g1", "att.g2",
      "att.k_k", "att.k_a", "att.r_k", "att.ln_x.weight", "att.ln_x.bias",
      "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
  };
  if (layer == 0) {
    fn(block_key(layer, "ln0.weight"));
    fn(block_key(layer, "ln0.bias"));
  }
  for (const char* suffix : required) {
    fn(block_key(layer, suffix));
  }
  if (layer > 0) {
    fn(block_key(layer, "att.v0"));
    fn(block_key(layer, "att.v1"));
    fn(block_key(layer, "att.v2"));
  }
}

void allocate_layer_tensors(
    CudaWeights& weights,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
    int layer) {
  for_each_layer_source_tensor(layer, [&](const std::string& key) {
    weights.allocate(by_name, key, true);
  });
}

void upload_layer_tensors(
    CudaWeights& weights,
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name,
    int layer,
    WeightLoadPipeline& pipeline) {
  for_each_layer_source_tensor(layer, [&](const std::string& key) {
    weights.upload(archive, by_name, key, pipeline);
  });
}

template <typename Fn>
void for_each_layer_t_copy(int layer, Fn&& fn) {
  const char* lowrank_t[] = {"att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2"};
  for (const char* suffix : lowrank_t) {
    fn(block_key(layer, suffix));
  }
  if (layer > 0) {
    fn(block_key(layer, "att.v1"));
    fn(block_key(layer, "att.v2"));
  }
}

CudaWeights load_model_weights(
    const ModelDims& dims,
    const llm_infer::PthArchive& archive,
    const std::unordered_map<std::string, const llm_infer::TensorRecord*>& by_name) {
  CudaWeights weights;
  weights.dims = dims;
  weights.layers.reserve(static_cast<std::size_t>(dims.layers));
  auto emb = by_name.find("emb.weight");
  if (emb != by_name.end()) {
    weights.cpu_emb_bytes = numel(emb->second->shape) * sizeof(std::uint16_t);
  }
  WeightLoadPipeline pipeline(archive.chunk_load());
  auto load_globals = [&]() {
    const char* globals[] = {"ln_out.weight", "ln_out.bias", "head.weight"};
    if (archive.chunk_load()) {
      for (const char* key : globals) weights.allocate(by_name, key, true);
      for (const char* key : globals) weights.upload(archive, by_name, key, pipeline);
    } else {
      for (const char* key : globals) weights.load(archive, by_name, key, true, pipeline);
    }
    std::cout << "load_model global done gpu_mib=" << mib(weights.bytes())
              << " cpu_emb_mib=" << mib(weights.cpu_emb_bytes) << std::endl;
  };
  if (!archive.chunk_load()) {
    load_globals();
  }
  const int layers_per_batch = archive.chunk_load() ? kChunkLoadLayerBatch : 1;
  for (int first_layer = 0; first_layer < dims.layers; first_layer += layers_per_batch) {
    const int last_layer = std::min(first_layer + layers_per_batch, dims.layers);
    for (int layer = first_layer; layer < last_layer; ++layer) {
      allocate_layer_tensors(weights, by_name, layer);
    }
    for (int layer = first_layer; layer < last_layer; ++layer) {
      upload_layer_tensors(weights, archive, by_name, layer, pipeline);
    }

    // All source weights for this batch must be resident before allocating and
    // launching the derived transpose copies. Allocate every destination first
    // so cudaMalloc cannot serialize the transpose kernels one by one.
    pipeline.sync();
    for (int layer = first_layer; layer < last_layer; ++layer) {
      for_each_layer_t_copy(layer, [&](const std::string& key) {
        weights.allocate_t_copy(key);
      });
    }
    for (int layer = first_layer; layer < last_layer; ++layer) {
      for_each_layer_t_copy(layer, [&](const std::string& key) {
        weights.launch_t_copy(key);
      });
    }
    check_cuda(cudaDeviceSynchronize(), "sync layer load batch");
    for (int layer = first_layer; layer < last_layer; ++layer) {
      weights.layers.push_back(weights.layer_view(layer));
    }
    std::cout << "load_model layer_batch=" << first_layer << "-" << (last_layer - 1)
              << " done layers=" << weights.layers.size()
              << " tensors=" << weights.tensors.size()
              << " t_copies=" << weights.t_copy_count
              << " gpu_mib=" << mib(weights.bytes()) << std::endl;
  }
  if (archive.chunk_load()) {
    load_globals();
  }
  pipeline.sync();
  pipeline.release_buffers();
  check_cuda(cudaDeviceSynchronize(), "sync model weight load");
  weights.build_global_view();
  build_cpu_emb_ln0_f16(weights, dims, archive, by_name);
  std::cout << "load_model emb+ln0 done cpu_emb_mib=" << mib(weights.cpu_emb_bytes)
            << " entries=" << weights.cpu_emb_ln0_f16.size() << std::endl;
  if (std::getenv("RWKV_CMIX_STATS") != nullptr) weights.cmix_stats.initialize(dims.layers);
  return weights;
}

ModelDims infer_quantized_model_dims(const llm_infer::QuantizedArchive& archive) {
  ModelDims dimensions;
  for (const auto& rec : archive.records()) {
    if (rec.name == "emb.weight" && rec.shape.size() == 2) {
      dimensions.vocab = static_cast<int>(rec.shape[0]);
      dimensions.channels = static_cast<int>(rec.shape[1]);
    } else if (rec.name == "blocks.0.att.r_k" && rec.shape.size() == 2) {
      dimensions.heads = static_cast<int>(rec.shape[0]);
      dimensions.head_size = static_cast<int>(rec.shape[1]);
    } else if (rec.name == "blocks.0.ffn.key.weight" && rec.shape.size() == 2) {
      dimensions.ffn = static_cast<int>(rec.shape[0]);
    }
    if (rec.name.rfind("blocks.", 0) == 0) {
      const char* begin = rec.name.c_str() + 7;
      char* end = nullptr;
      const long layer = std::strtol(begin, &end, 10);
      if (end && *end == '.' && layer >= 0) {
        dimensions.layers = std::max(dimensions.layers, static_cast<int>(layer) + 1);
      }
    }
  }
  require_result(dimensions.layers > 0 && dimensions.channels > 0 && dimensions.heads > 0 &&
                     dimensions.head_size > 0 && dimensions.vocab > 0 && dimensions.ffn > 0,
                 "could not infer model dimensions from quantized model");
  require_result(dimensions.channels == dimensions.heads * dimensions.head_size, "C must equal H*N");
  require_result(dimensions.head_size == 64, "current kernels require head size 64");
  return dimensions;
}

void load_quantized_tensor(
    const llm_infer::QuantizedArchive& archive,
    const llm_infer::QuantizedTensorRecord& rec,
    GpuTensor* tensor) {
  std::vector<std::uint8_t> data;
  require_result(archive.read_data(rec, &data).ok_status(), "read quantized tensor " + rec.name);
  if (rec.dtype == llm_infer::QuantizedDType::kInt8) {
    tensor->dtype = GpuTensor::DType::I8;
    tensor->i8.resize(static_cast<std::size_t>(rec.numel), "alloc int8 weight");
    tensor->scale.resize(static_cast<std::size_t>(rec.scale_count), "alloc int8 scales");
    std::vector<std::uint16_t> scales;
    require_result(archive.read_scales(rec, &scales).ok_status(), "read quantized scales");
    const bool transpose = should_transpose_like_v3a(rec.name);
    if (transpose) {
      DeviceBuffer<std::int8_t> staging;
      staging.resize(static_cast<std::size_t>(rec.numel), "alloc quantized int8 staging");
      check_cuda(cudaMemcpy(staging.p, data.data(), data.size(), cudaMemcpyHostToDevice), "copy int8 weight staging");
      rwkv7_v4_i8_transpose_launch(nullptr, staging.p, tensor->i8.p,
                                    static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
      check_cuda(cudaGetLastError(), "launch quantized int8 transpose");
      check_cuda(cudaDeviceSynchronize(), "sync quantized int8 transpose");
      tensor->i8_transposed = true;
    } else {
      DeviceBuffer<std::int8_t> staging;
      staging.resize(static_cast<std::size_t>(rec.numel), "alloc quantized int8 staging");
      check_cuda(cudaMemcpy(staging.p, data.data(), data.size(), cudaMemcpyHostToDevice), "copy int8 weight staging");
      require_result(rec.shape.size() == 2, "packed int8 weight must be rank-2: " + rec.name);
      const int compute_major = rwkv7_w8a16_device_info().compute_major;
      if (compute_major >= 8) {
        rwkv7_v4_i8_pack_launch(nullptr, staging.p, tensor->i8.p,
                                static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
        check_cuda(cudaGetLastError(), "launch quantized int8 pack");
        check_cuda(cudaDeviceSynchronize(), "sync quantized int8 pack");
        tensor->i8_packed = true;
      } else {
        check_cuda(cudaMemcpy(tensor->i8.p, staging.p, data.size(), cudaMemcpyDeviceToDevice),
                   "copy raw int8 weight");
      }
    }
    check_cuda(cudaMemcpy(tensor->scale.p, scales.data(), scales.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
               "copy int8 scales");
    return;
  }
  tensor->dtype = GpuTensor::DType::F16;
  tensor->f16.resize(static_cast<std::size_t>(rec.numel), "alloc floating weight");
  DeviceBuffer<std::uint16_t> staging;
  staging.resize(static_cast<std::size_t>(rec.numel), "alloc quantized staging");
  check_cuda(cudaMemcpy(staging.p, data.data(), data.size(), cudaMemcpyHostToDevice), "copy floating weight");
  const bool transpose = should_transpose_like_v3a(rec.name);
  if (rec.dtype == llm_infer::QuantizedDType::kFloat16 && transpose) {
    rwkv7_v4_f16_transpose_launch(nullptr, staging.p, tensor->f16.p,
                                   static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
  } else if (rec.dtype == llm_infer::QuantizedDType::kFloat16) {
    check_cuda(cudaMemcpy(tensor->f16.p, staging.p, data.size(), cudaMemcpyDeviceToDevice),
               "copy quantized f16 weight");
  } else if (transpose) {
    rwkv7_v4_bf16_to_f16_transpose_launch(nullptr, staging.p, tensor->f16.p,
                                           static_cast<int>(rec.shape[0]), static_cast<int>(rec.shape[1]));
  } else {
    rwkv7_v4_bf16_to_f16_launch(nullptr, staging.p, tensor->f16.p, static_cast<long long>(rec.numel));
  }
  check_cuda(cudaGetLastError(), "preprocess quantized floating weight");
  check_cuda(cudaDeviceSynchronize(), "sync quantized floating weight");
}

CudaWeights load_quantized_model_weights(
    const ModelDims& dims,
    const llm_infer::QuantizedArchive& archive) {
  CudaWeights weights;
  weights.dims = dims;
  std::unordered_map<std::string, const llm_infer::QuantizedTensorRecord*> by_name;
  for (const auto& rec : archive.records()) by_name.emplace(rec.name, &rec);
  auto allocate = [&](const std::string& key, bool required) {
    const auto it = by_name.find(key);
    if (it == by_name.end()) {
      if (required) {
        std::cerr << "error: quantized model missing tensor: " << key << "\n";
        std::exit(1);
      }
      return;
    }
    const auto& rec = *it->second;
    auto tensor = std::make_unique<GpuTensor>();
    tensor->name = rec.name;
    tensor->shape = rec.shape;
    if (should_transpose_like_v3a(rec.name)) {
      tensor->shape = {rec.shape[1], rec.shape[0]};
    }
    tensor->dtype = rec.dtype == llm_infer::QuantizedDType::kInt8 ? GpuTensor::DType::I8 : GpuTensor::DType::F16;
    if (tensor->is_int8()) {
      tensor->i8.resize(static_cast<std::size_t>(rec.numel), "alloc int8 tensor");
      tensor->scale.resize(static_cast<std::size_t>(rec.scale_count), "alloc int8 tensor scales");
    } else {
      tensor->f16.resize(static_cast<std::size_t>(rec.numel), "alloc quantized f16 tensor");
    }
    weights.tensors.emplace(key, std::move(tensor));
  };
  auto upload = [&](const std::string& key) {
    const auto it = by_name.find(key);
    if (it == by_name.end()) {
      std::cerr << "error: quantized model missing tensor during upload: " << key << "\n";
      std::exit(1);
    }
    load_quantized_tensor(archive, *it->second, weights.require_mutable(key));
  };
  for (const char* key : {"ln_out.weight", "ln_out.bias", "head.weight"}) allocate(key, true);
  for (const char* key : {"ln_out.weight", "ln_out.bias", "head.weight"}) upload(key);
  for (int layer = 0; layer < dims.layers; ++layer) {
    for_each_layer_source_tensor(layer, [&](const std::string& key) { allocate(key, true); });
    for_each_layer_source_tensor(layer, upload);
    for_each_layer_t_copy(layer, [&](const std::string& key) { weights.allocate_t_copy(key); });
    for_each_layer_t_copy(layer, [&](const std::string& key) { weights.launch_t_copy(key); });
    check_cuda(cudaDeviceSynchronize(), "sync quantized layer load");
    weights.layers.push_back(weights.layer_view(layer));
  }
  weights.build_global_view();
  build_cpu_emb_ln0_f16_quantized(weights, dims, archive);
  std::cout << "load_model quantized done tensors=" << weights.tensors.size()
            << " gpu_mib=" << mib(weights.bytes()) << " cpu_emb_mib=" << mib(weights.cpu_emb_bytes) << "\n";
  if (std::getenv("RWKV_CMIX_STATS") != nullptr) weights.cmix_stats.initialize(dims.layers);
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
    const GpuTensor* weight_tensor,
    void* workspace,
    std::size_t workspace_bytes,
    half* y) {
  if (weight_tensor == nullptr) {
    std::cerr << "error: null linear weight\n";
    std::exit(1);
  }
  if (weight_tensor->is_int8()) {
    rwkv7_w8a16_linear_launch(
        stream, M, K, N, x, weight_tensor->i8.p,
        reinterpret_cast<const half*>(weight_tensor->scale.p),
        weight_tensor->i8_packed ? W8BLayout::PackedNK
                                 : (weight_tensor->i8_transposed ? W8BLayout::KN : W8BLayout::NK),
        y, workspace, workspace_bytes);
    return;
  }
  const half* weight_orig = hp(weight_tensor);
  if (path.rows == 1) {
    if (group == LinearGroup::FfnKey) {
      if (K == 2560) {
        rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, true, y);
        return;
      }
      rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, K <= 1024, y);
    } else {
      rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, group != LinearGroup::AttC2C || K < 2048, y);
    }
    return;
  }
  if (path.rows == 2) {
    if (group == LinearGroup::AttC2C) {
      rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, true, y);
    } else if (group == LinearGroup::FfnKey) {
      if (K == 2560) {
        rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, false, y);
        return;
      }
      if (K < 4096) {
        rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, true, y);
      } else {
        rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, false, y);
      }
    } else if (group == LinearGroup::Head && K == 2560) {
      rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 128, 2, false, y);
    } else {
      rwkv7_v3a_linear_orig_rows_exact_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, true, y);
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
  if (path.rows == 3) {
    if (group == LinearGroup::Head) {
      if (K <= 2048) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      if (K == 2560) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 3, 2, y);
    } else if (group == LinearGroup::FfnKey) {
      if (K <= 1024) {
        rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 64, 3, 4, y);
        return;
      }
      if (K == 2048) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      if (K == 2560) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      lt(0, 0);
    } else {
      if (K == 768) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 1, 2, y);
        return;
      }
      if (K == 1024) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 2, 2, y);
        return;
      }
      if (K == 2048) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 3, 4, y);
        return;
      }
      if (K == 2560) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 3, 2, y);
        return;
      }
      lt(0, 2);
    }
    return;
  }
  if (path.rows == 4) {
    if (group == LinearGroup::FfnKey) {
      if (K <= 1024) {
        rwkv7_v3a_linear_orig_rows_cfg_f16_launch(stream, M, K, N, x, weight_orig, 64, 2, 4, y);
        return;
      }
      if (K == 2048) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      if (K == 2560) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      return lt(0, 0);
    }
    if (group == LinearGroup::AttC2C) {
      if (K <= 1024) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 2, 2, y);
        return;
      }
      if (K == 2048) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 4, 2, y);
        return;
      }
      if (K == 2560) {
        rwkv7_v3a_linear_orig_rows_f16_launch(stream, M, K, N, x, weight_orig, 4, 2, y);
        return;
      }
      return lt(0, 2);
    }
  }
  if (group == LinearGroup::Head) {
    if (K == 768) {
      if (path.rows >= 192 && path.rows < 256) return lt(128, 3);
      if (path.rows >= 96 && path.rows < 160) return lt(0, 1);
    }
    if (K == 1024) {
      if (path.rows >= 256 && path.rows < 384) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      if (path.rows >= 192 && path.rows < 256) return lt(0, 2);
      if (path.rows >= 96 && path.rows < 160) return lt(32, 1);
    }
    if (K == 2048) {
      if (path.rows >= 256 && path.rows < 384) return lt(32, 0);
      if (path.rows >= 192 && path.rows < 256) return lt(32, 6);
      if (path.rows >= 128 && path.rows < 160) return lt(0, 1);
      if (path.rows >= 96 && path.rows < 112) return lt(0, 0);
    }
    if (K == 2560) {
      if (path.rows >= 128 && path.rows < 160) return lt(0, 1);
      if (path.rows >= 80) return lt(0, 0);
      if (path.rows >= 72) return lt(32, 1);
    }
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
    if (K == 768) {
      if (path.rows >= 256 && path.rows < 384) return lt(128, 1);
      if (path.rows >= 96 && path.rows < 112) return lt(32, 3);
    }
    if (K == 1024) {
      if (path.rows >= 256 && path.rows < 384) return lt(128, 0);
      if (path.rows >= 96 && path.rows < 112) return lt(32, 6);
    }
    if (K == 2048) {
      if (path.rows >= 256 && path.rows < 384) return lt(32, 3);
      if (path.rows >= 192 && path.rows < 256) return lt(128, 0);
      if (path.rows >= 96 && path.rows < 112) return lt(32, 4);
    }
    if (K == 2560) {
      if (path.rows >= 128 && path.rows < 160) return lt(128, 2);
      if (path.rows >= 112) return lt(128, 3);
      if (path.rows >= 72) return lt(128, 2);
      if (path.rows >= 5) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
    }
    if (path.rows >= 1024) return lt(32, 4);
    if (path.rows >= 768) return lt(32, 0);
    if (path.rows >= 512) return lt(32, 1);
    if (path.rows >= 384) return lt(128, 2);
    if (path.rows >= 256) return lt(128, 0);
    if (path.rows >= 192) return lt(0, 0);
    if (path.rows >= 160) return lt(128, 1);
    if (path.rows >= 128) return lt(128, 0);
    if (path.rows >= 112) {
      rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
      return;
    }
    if (path.rows >= 96) return lt(0, 5);
    if (path.rows >= 72) return lt(32, 0);
    if (path.rows >= 48) return lt(32, 6);
    if (path.rows >= 32) return lt(0, 0);
    if (path.rows >= 24) return lt(0, 6);
    if (path.rows >= 12) return lt(0, 0);
    if (path.rows >= 5) return lt(0, 2);
  } else {
    if (K == 768) {
      if (path.rows >= 256 && path.rows < 384) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
      if (path.rows >= 96 && path.rows < 112) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
    }
    if (K == 1024) {
      if (path.rows >= 256 && path.rows < 384) return lt(32, 2);
      if (path.rows >= 192 && path.rows < 256) return lt(0, 0);
      if (path.rows >= 96 && path.rows < 160) return lt(32, 2);
    }
    if (K == 2048 && path.rows >= 128 && path.rows < 160) return lt(0, 3);
    if (K == 2560) {
      if (path.rows >= 128 && path.rows < 160) return lt(32, 5);
      if (path.rows >= 112) return lt(128, 4);
      if (path.rows >= 80) return lt(0, 3);
      if (path.rows >= 72) return lt(32, 4);
      if (path.rows >= 3) {
        rwkv7_v3a_linear_f16_orig_launch(stream, M, K, N, x, weight_orig, y);
        return;
      }
    }
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
    if (path.rows >= 12) return lt(0, 0);
    if (path.rows == 5 || path.rows == 6) return lt(0, 1);
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
  if (rows <= kLowrankOutRowsT && N >= kLowrankFusedMinC) {
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
  if (rows <= kLowrankOutRowsT && N >= kLowrankFusedMinC) {
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

CudaWeights load_backend_weights(const std::string& model_path, bool chunk_load) {
  if (llm_infer::is_quantized_archive(model_path)) {
    auto archive = llm_infer::QuantizedArchive::open(model_path);
    require_result(archive.ok(), archive.status().message());
    const ModelDims dims = infer_quantized_model_dims(archive.value());
    return load_quantized_model_weights(dims, archive.value());
  }
  std::cout << "load_model mode=" << (chunk_load ? "chunked" : "whole-file");
  if (chunk_load) {
    std::cout << " chunk_mib=" << mib(kWeightLoadChunkBytes)
              << " host_buffers=2"
              << " layer_batch=" << kChunkLoadLayerBatch;
  }
  std::cout << std::endl;
  auto archive = llm_infer::PthArchive::open(model_path, chunk_load);
  require_result(archive.ok(), archive.status().message());
  auto records = llm_infer::parse_pth_tensor_records(archive.value());
  require_result(records.ok(), records.status().message());
  std::unordered_map<std::string, const llm_infer::TensorRecord*> by_name;
  for (const auto& rec : records.value()) {
    by_name.emplace(rec.name, &rec);
  }
  const ModelDims dims = infer_model_dims(records.value());
  return load_model_weights(dims, archive.value(), by_name);
}

struct W8A16TuneShape {
  std::string name;
  int K = 0;
  int N = 0;
  W8BLayout layout = W8BLayout::NK;
  const GpuTensor* tensor = nullptr;
};

struct W8A16TuneRecord {
  int K = 0;
  int N = 0;
  W8BLayout layout = W8BLayout::NK;
  int m_bucket = 0;
  int split_k = 0;
};

struct W8A16TuneCacheData {
  std::string gpu_name;
  std::string model_name;
  ModelDims dims;
  int sparse_max_rows = 0;
  std::vector<int> sparse_max_rows_by_layer;
  std::vector<W8A16TuneRecord> records;
};

std::vector<int> measured_cmix_sparse_rows_by_layer(const ModelDims& dims) {
  std::vector<int> rows(static_cast<std::size_t>(dims.layers), kDefaultQuantizedCmixSparseMaxRows);
  if (dims.layers == 32 && dims.channels == 4096 && dims.ffn == 16384) {
    static constexpr int measured[] = {
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 16,
        16, 16, 16, 16, 16, 16, 16, 32, 16, 64, 64, 64, 64, 32, 32, 16};
    rows.assign(std::begin(measured), std::end(measured));
  }
  return rows;
}

W8BLayout w8a16_layout(const GpuTensor* tensor) {
  return tensor->i8_packed ? W8BLayout::PackedNK
                           : (tensor->i8_transposed ? W8BLayout::KN : W8BLayout::NK);
}

const char* w8a16_layout_name(W8BLayout layout) {
  switch (layout) {
    case W8BLayout::NK: return "NK";
    case W8BLayout::KN: return "KN";
    case W8BLayout::PackedNK: return "PackedNK";
  }
  return "unknown";
}

void add_w8a16_tune_shape(
    std::vector<W8A16TuneShape>& shapes,
    const std::string& name,
    int K,
    int N,
    const GpuTensor* tensor) {
  if (tensor == nullptr || !tensor->is_int8()) return;
  const W8BLayout layout = w8a16_layout(tensor);
  for (const auto& shape : shapes) {
    if (shape.K == K && shape.N == N && shape.layout == layout) return;
  }
  shapes.push_back({name, K, N, layout, tensor});
}

std::vector<W8A16TuneShape> collect_w8a16_tune_shapes(const CudaWeights& weights) {
  std::vector<W8A16TuneShape> shapes;
  for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
    const LayerWeights& w = weights.layers[layer];
    add_w8a16_tune_shape(shapes, "att_c2c", weights.dims.channels, weights.dims.channels,
                         w.att_receptance_w);
    add_w8a16_tune_shape(shapes, "ffn_key", weights.dims.channels, weights.dims.ffn,
                         w.ffn_key_w);
    add_w8a16_tune_shape(shapes, "ffn_value_rawkn", weights.dims.ffn, weights.dims.channels,
                         w.ffn_value_w);
  }
  add_w8a16_tune_shape(shapes, "head", weights.dims.channels, weights.dims.vocab, weights.head_w);
  return shapes;
}

std::filesystem::path default_w8a16_tune_cache(
    const std::string& model_path,
    const std::string& gpu_name,
    const std::string& cache_directory) {
  std::string gpu_file = gpu_name;
  for (char& value : gpu_file) {
    if (!(std::isalnum(static_cast<unsigned char>(value)) || value == '-' || value == '_')) value = '_';
  }
  if (gpu_file.empty()) gpu_file = "gpu";
  const std::filesystem::path path(model_path);
  const std::filesystem::path directory = cache_directory.empty()
      ? std::filesystem::current_path()
      : std::filesystem::path(cache_directory);
  return directory / (path.stem().string() + "." + gpu_file + ".w8a16.tune");
}

bool same_model_dims(const ModelDims& lhs, const ModelDims& rhs) {
  return lhs.layers == rhs.layers && lhs.channels == rhs.channels && lhs.heads == rhs.heads &&
         lhs.head_size == rhs.head_size && lhs.vocab == rhs.vocab && lhs.ffn == rhs.ffn;
}

bool load_w8a16_tune_cache(
    const std::filesystem::path& path,
    const std::string& model_name,
    const ModelDims& dims,
    const std::string& gpu_name,
    const std::vector<W8A16TuneShape>& shapes,
    W8A16TuneCacheData* cache) {
  std::ifstream input(path);
  if (!input) return false;
  std::string magic;
  input >> magic;
  if (magic != "rwkv_w8a16_tune_v1") return false;
  int tune_version = -1;
  std::string key;
  while (input >> key) {
    if (key == "tune_version") {
      input >> tune_version;
    } else if (key == "gpu") {
      input >> std::quoted(cache->gpu_name);
    } else if (key == "model") {
      input >> std::quoted(cache->model_name);
    } else if (key == "dims") {
      input >> cache->dims.layers >> cache->dims.channels >> cache->dims.heads >>
          cache->dims.head_size >> cache->dims.vocab >> cache->dims.ffn;
    } else if (key == "sparse_max_rows") {
      input >> cache->sparse_max_rows;
    } else if (key == "sparse_layer_rows") {
      int layer = -1;
      int rows = -1;
      input >> layer >> rows;
      if (layer < 0 || layer >= dims.layers || rows < 0 || rows > 64) return false;
      if (cache->sparse_max_rows_by_layer.empty()) {
        cache->sparse_max_rows_by_layer.assign(static_cast<std::size_t>(dims.layers), -1);
      }
      cache->sparse_max_rows_by_layer[static_cast<std::size_t>(layer)] = rows;
    } else if (key == "split") {
      int layout = 0;
      W8A16TuneRecord record;
      input >> record.K >> record.N >> layout >> record.m_bucket >> record.split_k;
      record.layout = static_cast<W8BLayout>(layout);
      cache->records.push_back(record);
    } else {
      std::string ignored;
      std::getline(input, ignored);
    }
  }
  if (!input.good() && !input.eof()) return false;
  if (tune_version != kTuneVersion || cache->gpu_name != gpu_name || cache->model_name != model_name ||
      !same_model_dims(cache->dims, dims) || cache->sparse_max_rows < 0 ||
      cache->sparse_max_rows > 64) return false;
  if (!cache->sparse_max_rows_by_layer.empty()) {
    if (cache->sparse_max_rows_by_layer.size() != static_cast<std::size_t>(dims.layers)) return false;
    for (int rows : cache->sparse_max_rows_by_layer) {
      if (rows < 0 || rows > 64) return false;
    }
  }
  for (const auto& shape : shapes) {
    for (int m : {8, 16, 32, 64, 128}) {
      bool found = false;
      for (const auto& record : cache->records) {
        if (record.K == shape.K && record.N == shape.N && record.layout == shape.layout &&
            record.m_bucket == m && record.split_k >= 1 && record.split_k <= 4) {
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
  }
  return true;
}

void save_w8a16_tune_cache(const std::filesystem::path& path, const W8A16TuneCacheData& cache) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  const std::filesystem::path temp_path = path.string() + ".tmp";
  std::ofstream output(temp_path, std::ios::trunc);
  if (!output) {
    std::cerr << "warning: cannot write W8A16 tune cache: " << temp_path << "\n";
    return;
  }
  output << "rwkv_w8a16_tune_v1\n"
         << "tune_version " << kTuneVersion << "\n"
         << "gpu " << std::quoted(cache.gpu_name) << "\n"
         << "model " << std::quoted(cache.model_name) << "\n"
         << "dims " << cache.dims.layers << ' ' << cache.dims.channels << ' '
         << cache.dims.heads << ' ' << cache.dims.head_size << ' '
         << cache.dims.vocab << ' ' << cache.dims.ffn << "\n"
         << "sparse_max_rows " << cache.sparse_max_rows << "\n";
  for (std::size_t layer = 0; layer < cache.sparse_max_rows_by_layer.size(); ++layer) {
    output << "sparse_layer_rows " << layer << ' ' << cache.sparse_max_rows_by_layer[layer] << "\n";
  }
  for (const auto& record : cache.records) {
    output << "split " << record.K << ' ' << record.N << ' '
           << static_cast<int>(record.layout) << ' ' << record.m_bucket << ' '
           << record.split_k << "\n";
  }
  output.flush();
  if (!output) {
    std::cerr << "warning: cannot flush W8A16 tune cache: " << temp_path << "\n";
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
    return;
  }
  output.close();
  if (!output) {
    std::cerr << "warning: cannot close W8A16 tune cache: " << temp_path << "\n";
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
    return;
  }
  std::error_code rename_error;
  std::filesystem::rename(temp_path, path, rename_error);
  if (rename_error) {
    std::cerr << "warning: cannot replace W8A16 tune cache " << path << ": "
              << rename_error.message() << "\n";
    std::error_code remove_error;
    std::filesystem::remove(temp_path, remove_error);
  }
}

double tune_linear_once(
    cudaStream_t stream,
    int M,
    const W8A16TuneShape& shape,
    const half* x,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int split_k) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "create W8A16 tuning start event");
  check_cuda(cudaEventCreate(&stop), "create W8A16 tuning stop event");
  check_cuda(cudaEventRecord(start, stream), "record W8A16 tuning start");
  rwkv7_w8a16_linear_launch(stream, M, shape.K, shape.N, x, shape.tensor->i8.p,
                            reinterpret_cast<const half*>(shape.tensor->scale.p), shape.layout,
                            y, workspace, workspace_bytes, split_k);
  check_cuda(cudaGetLastError(), "launch W8A16 tuning timed linear");
  check_cuda(cudaEventRecord(stop, stream), "record W8A16 tuning stop");
  check_cuda(cudaEventSynchronize(stop), "sync W8A16 tuning stop");
  float elapsed = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "elapsed W8A16 tuning linear");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed;
}

double tune_linear_median(
    cudaStream_t stream,
    int M,
    const W8A16TuneShape& shape,
    const half* x,
    half* y,
    void* workspace,
    std::size_t workspace_bytes,
    int split_k) {
  for (int warmup = 0; warmup < 1; ++warmup) {
    rwkv7_w8a16_linear_launch(stream, M, shape.K, shape.N, x, shape.tensor->i8.p,
                              reinterpret_cast<const half*>(shape.tensor->scale.p), shape.layout,
                              y, workspace, workspace_bytes, split_k);
  }
  check_cuda(cudaStreamSynchronize(stream), "sync W8A16 tuning warmup");
  std::array<double, 5> samples{};
  for (double& sample : samples) {
    sample = tune_linear_once(stream, M, shape, x, y, workspace, workspace_bytes, split_k);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

W8A16TuneCacheData tune_w8a16_model(
    const CudaWeights& weights,
    const std::vector<W8A16TuneShape>& shapes,
    const std::string& model_name,
    const std::string& gpu_name) {
  W8A16TuneCacheData cache;
  cache.gpu_name = gpu_name;
  cache.model_name = model_name;
  cache.dims = weights.dims;
  rwkv7_w8a16_tuning_reset();
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create W8A16 tuning stream");
  DeviceBuffer<unsigned char> workspace;
  workspace.resize(static_cast<std::size_t>(128) << 20, "alloc W8A16 tuning workspace");
  DeviceBuffer<half> x;
  DeviceBuffer<half> y;
  std::mt19937 rng(1380668983u);
  for (const auto& shape : shapes) {
    for (int M : {8, 16, 32, 64, 128}) {
      x.resize(static_cast<std::size_t>(M) * shape.K, "alloc W8A16 tuning x");
      y.resize(static_cast<std::size_t>(M) * shape.N, "alloc W8A16 tuning y");
      std::vector<half> host_x(static_cast<std::size_t>(M) * shape.K);
      for (half& value : host_x) {
        value = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000.0f);
      }
      check_cuda(cudaMemcpyAsync(x.p, host_x.data(), host_x.size() * sizeof(half),
                                 cudaMemcpyHostToDevice, stream), "copy W8A16 tuning x");
      check_cuda(cudaStreamSynchronize(stream), "sync W8A16 tuning x");
      double best_ms = std::numeric_limits<double>::infinity();
      int best_split = 1;
      for (int split_k : {1, 2, 4}) {
        const double elapsed = tune_linear_median(stream, M, shape, x.p, y.p,
                                                  workspace.p, workspace.n, split_k);
        std::cout << "w8a16_tune linear=" << shape.name << " K=" << shape.K
                  << " N=" << shape.N << " layout=" << w8a16_layout_name(shape.layout)
                  << " M=" << M << " split_k=" << split_k << " ms=" << elapsed << "\n";
        if (elapsed < best_ms) {
          best_ms = elapsed;
          best_split = split_k;
        }
      }
      rwkv7_w8a16_tuning_set(shape.K, shape.N, shape.layout, M, best_split);
      cache.records.push_back({shape.K, shape.N, shape.layout, M, best_split});
      std::cout << "w8a16_tune_selected linear=" << shape.name << " M=" << M
                << " split_k=" << best_split << " ms=" << best_ms << "\n";
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "sync W8A16 tuning linear complete");
  cudaStreamDestroy(stream);
  return cache;
}

double tune_sparse_median(
    cudaStream_t stream,
    int rows,
    int C,
    int F,
    const half* preact,
    half* dense_input,
    half* out,
    const GpuTensor* value,
    void* workspace,
    std::size_t workspace_bytes,
    bool sparse) {
  for (int warmup = 0; warmup < 1; ++warmup) {
    if (sparse) {
      rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(
          stream, rows, 1, C, F, preact, value->i8.p,
          reinterpret_cast<const half*>(value->scale.p), out);
    } else {
      check_cuda(cudaMemcpyAsync(dense_input, preact,
                                 static_cast<std::size_t>(rows) * F * sizeof(half),
                                 cudaMemcpyDeviceToDevice, stream), "copy W8A16 dense tuning input");
      rwkv7_relu_square_launch(stream, dense_input, dense_input,
                               static_cast<long long>(rows) * F);
      rwkv7_w8a16_linear_launch(
          stream, rows, F, C, dense_input, value->i8.p,
          reinterpret_cast<const half*>(value->scale.p), w8a16_layout(value), out,
          workspace, workspace_bytes);
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "sync cmix tuning warmup");
  std::array<double, 5> samples{};
  for (double& sample : samples) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create cmix tuning start event");
    check_cuda(cudaEventCreate(&stop), "create cmix tuning stop event");
    check_cuda(cudaEventRecord(start, stream), "record cmix tuning start");
    if (sparse) {
      rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(
          stream, rows, 1, C, F, preact, value->i8.p,
          reinterpret_cast<const half*>(value->scale.p), out);
    } else {
      check_cuda(cudaMemcpyAsync(dense_input, preact,
                                 static_cast<std::size_t>(rows) * F * sizeof(half),
                                 cudaMemcpyDeviceToDevice, stream), "copy W8A16 dense tuning input");
      rwkv7_relu_square_launch(stream, dense_input, dense_input,
                               static_cast<long long>(rows) * F);
      rwkv7_w8a16_linear_launch(
          stream, rows, F, C, dense_input, value->i8.p,
          reinterpret_cast<const half*>(value->scale.p), w8a16_layout(value), out,
          workspace, workspace_bytes);
    }
    check_cuda(cudaGetLastError(), "launch cmix tuning path");
    check_cuda(cudaEventRecord(stop, stream), "record cmix tuning stop");
    check_cuda(cudaEventSynchronize(stop), "sync cmix tuning stop");
    float elapsed = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "elapsed cmix tuning path");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    sample = elapsed;
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int tune_cmix_sparse_threshold(
    const W8A16TuneShape& value_shape) {
  const int C = value_shape.N;
  const int F = value_shape.K;
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create cmix tuning stream");
  DeviceBuffer<half> preact;
  DeviceBuffer<half> dense_input;
  DeviceBuffer<half> out;
  DeviceBuffer<unsigned char> workspace;
  workspace.resize(static_cast<std::size_t>(128) << 20, "alloc cmix tuning workspace");
  int threshold = 0;
  bool sparse_wins_for_all_smaller_rows = true;
  for (int rows : {16, 32, 64}) {
    preact.resize(static_cast<std::size_t>(rows) * F, "alloc cmix tuning preact");
    dense_input.resize(static_cast<std::size_t>(rows) * F, "alloc cmix tuning dense input");
    out.resize(static_cast<std::size_t>(rows) * C, "alloc cmix tuning output");
    std::vector<half> host(static_cast<std::size_t>(rows) * F);
    for (std::size_t index = 0; index < host.size(); ++index) {
      const bool nonzero = (index % 20) < 3;
      host[index] = __float2half(nonzero ? 0.25f : -0.25f);
    }
    check_cuda(cudaMemcpy(preact.p, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice),
               "copy cmix tuning preact");
    const double sparse_ms = tune_sparse_median(
        stream, rows, C, F, preact.p, dense_input.p, out.p, value_shape.tensor,
        workspace.p, workspace.n, true);
    const double dense_ms = tune_sparse_median(
        stream, rows, C, F, preact.p, dense_input.p, out.p, value_shape.tensor,
        workspace.p, workspace.n, false);
    std::cout << "w8a16_tune cmix rows=" << rows << " sparse_ms=" << sparse_ms
              << " dense_ms=" << dense_ms << "\n";
    if (sparse_wins_for_all_smaller_rows && sparse_ms < dense_ms) threshold = rows;
    else sparse_wins_for_all_smaller_rows = false;
  }
  cudaStreamDestroy(stream);
  return threshold == 0 ? kDefaultQuantizedCmixSparseMaxRows : threshold;
}

void configure_w8a16_tuning(
    CudaWeights& weights,
    const std::string& model_path,
    const std::string& requested_cache,
    const std::string& default_cache_directory,
    bool retune) {
  const std::vector<W8A16TuneShape> shapes = collect_w8a16_tune_shapes(weights);
  if (shapes.empty()) return;
  const W8A16DeviceInfo device = rwkv7_w8a16_device_info();
  const std::string gpu_name(device.name);
  const std::filesystem::path cache_path = requested_cache.empty()
      ? default_w8a16_tune_cache(model_path, gpu_name, default_cache_directory)
      : std::filesystem::path(requested_cache);
  const std::string model_name = basename_without_extension(model_path);
  W8A16TuneCacheData cache;
  if (!retune && load_w8a16_tune_cache(cache_path, model_name, weights.dims, gpu_name, shapes, &cache)) {
    rwkv7_w8a16_tuning_reset();
    for (const auto& record : cache.records) {
      rwkv7_w8a16_tuning_set(record.K, record.N, record.layout, record.m_bucket, record.split_k);
    }
    weights.cmix_sparse_max_rows = cache.sparse_max_rows;
    weights.cmix_sparse_max_rows_by_layer = cache.sparse_max_rows_by_layer.empty()
        ? measured_cmix_sparse_rows_by_layer(weights.dims)
        : cache.sparse_max_rows_by_layer;
    std::cout << "w8a16_tune_cache hit path=" << cache_path
              << " gpu=\"" << gpu_name << "\" sparse_max_rows="
              << weights.cmix_sparse_max_rows << "\n";
    return;
  }
  std::cout << "w8a16_tune start path=" << cache_path << " gpu=\"" << gpu_name << "\"\n";
  cache = tune_w8a16_model(weights, shapes, model_name, gpu_name);
  const auto value_it = std::find_if(shapes.begin(), shapes.end(), [](const W8A16TuneShape& shape) {
    return shape.name == "ffn_value_rawkn";
  });
  cache.sparse_max_rows = value_it == shapes.end()
      ? kDefaultQuantizedCmixSparseMaxRows
      : tune_cmix_sparse_threshold(*value_it);
  cache.sparse_max_rows_by_layer = measured_cmix_sparse_rows_by_layer(weights.dims);
  weights.cmix_sparse_max_rows = cache.sparse_max_rows;
  weights.cmix_sparse_max_rows_by_layer = cache.sparse_max_rows_by_layer;
  save_w8a16_tune_cache(cache_path, cache);
  std::cout << "w8a16_tune complete path=" << cache_path
            << " sparse_max_rows=" << weights.cmix_sparse_max_rows << "\n";
}

void run_backend_forward(
    const CudaWeights& weights,
    const std::vector<std::vector<int64_t>>& token_batches,
    bool use_wkv32,
    const std::string& cmix_sparse,
    GenerationState& state,
    DeviceLogits& out) {
  const auto& dims = weights.dims;
  const int B = static_cast<int>(token_batches.size());
  if (B <= 0) {
    throw std::runtime_error("token_batches must not be empty");
  }
  if (state.batch_size != B) {
    throw std::runtime_error("state batch size mismatch");
  }
  if (state.wkv32 != use_wkv32) {
    throw std::runtime_error("state wkv precision mismatch");
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
  run.wkv32 = use_wkv32;
  run.cmix_sparse = cmix_sparse;
  const int C = dims.channels;
  PathConfig path = select_path(run, C);
  const int rows = B * T;
  const int output_rows = B;
  const int H = dims.heads;
  const int N = dims.head_size;
  const int V = dims.vocab;
  const int F = dims.ffn;

  static thread_local ForwardResources resources;
  resources.arena.allocate(static_cast<std::size_t>(rows) * C * 31 + static_cast<std::size_t>(output_rows) * C +
                           static_cast<std::size_t>(rows) * F +
                           static_cast<std::size_t>(rows) * kLowrankMax * 4 + static_cast<std::size_t>(output_rows) * V);
  resources.lt_workspace.resize(static_cast<std::size_t>(128) << 20, "alloc backend cublasLt workspace");
  resources.ensure_stream();
  HalfArena& arena = resources.arena;
  DeviceBuffer<unsigned char>& lt_workspace = resources.lt_workspace;
  cudaStream_t stream = resources.stream;
  BackendProfile profiler(std::getenv("RWKV_PROFILE") != nullptr);

  if (weights.cpu_emb_ln0_f16.size() != static_cast<std::size_t>(V) * C) {
    throw std::runtime_error("cpu emb+ln0 table is not ready");
  }

  std::vector<std::uint16_t> host_x(static_cast<std::size_t>(rows) * C);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const int token_id = static_cast<int>(token_batches[static_cast<size_t>(b)][static_cast<size_t>(t)]);
      if (token_id < 0 || token_id >= V) {
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
  half* hid = arena.take(static_cast<std::size_t>(rows) * F, "ffn_hid");
  half* cmix_out = arena.take(row_elems, "cmix_out");
  half* final_x = arena.take(static_cast<std::size_t>(output_rows) * C, "final_x");
  half* logits_f16 = arena.take(static_cast<std::size_t>(output_rows) * V, "logits_f16");

  check_cuda(cudaMemcpyAsync(x0, host_x.data(), host_x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream),
             "copy backend emb rows");

  const int profile_ln1 = profiler.begin(stream, "layer_norm_ln1");
  rwkv7_v3a_layer_norm_f16_launch(
      stream, rows, C, x0, hp(weights.layers[0].ln1_w), hp(weights.layers[0].ln1_b), xx0, kLnEps);
  profiler.end(stream, profile_ln1);
  half* x_cur = x0;
  half* xx_cur = xx0;
  half* x_next = x1;
  half* xx_next = xx1;
  bool pre_mix_ready = false;

  for (int layer = 0; layer < dims.layers; ++layer) {
    const LayerWeights& w = weights.layers[layer];
    const int Rw = static_cast<int>(w.att_w1_t->shape[0]);
    const int Ra = static_cast<int>(w.att_a1_t->shape[0]);
    const int Rg = static_cast<int>(w.att_g1_t->shape[0]);
    const int Rv = (layer == 0) ? 0 : static_cast<int>(w.att_v1_t->shape[0]);
    if (Rw > kLowrankMax || Ra > kLowrankMax || Rg > kLowrankMax || Rv > kLowrankMax) {
      throw std::runtime_error("lowrank exceeds arena max");
    }

    half* shift0 = state.shift.p + static_cast<std::size_t>(layer) * 2 * B * C;
    half* shift1 = shift0 + static_cast<std::size_t>(B) * C;
    half* layer_state16 = nullptr;
    float* layer_state32 = nullptr;
    if (use_wkv32) {
      layer_state32 = state.wkv_state32.p + static_cast<std::size_t>(layer) * B * H * N * N;
    } else {
      layer_state16 = state.wkv_state.p + static_cast<std::size_t>(layer) * B * H * N * N;
    }

    if (!pre_mix_ready) {
      const int profile_tmix = profiler.begin(stream, "tmix_mix6");
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
      profiler.end(stream, profile_tmix);
    } else {
      pre_mix_ready = false;
    }

    const int profile_att_receptance = profiler.begin(stream, "att_receptance");
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xr, w.att_receptance_w, lt_workspace.p, lt_workspace.n, r);
    profiler.end(stream, profile_att_receptance);
    const int profile_att_key = profiler.begin(stream, "att_key");
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xk, w.att_key_w, lt_workspace.p, lt_workspace.n, k);
    profiler.end(stream, profile_att_key);
    const int profile_att_value = profiler.begin(stream, "att_value");
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, xv, w.att_value_w, lt_workspace.p, lt_workspace.n, v_base);
    profiler.end(stream, profile_att_value);
    half* v_use = v_base;
    bool v_done = false;

    const int profile_att_lowrank = profiler.begin(stream, "att_lowrank");
    if (C >= kLowrankFusedMinC && rows <= kLowrankInRowsT && rows <= kLowrankOutRowsT && layer != 0) {
      rwkv7_v3a_linear_wagv_rank_in_f16_launch(
          stream, rows, C, Rw, Ra, Rg, Rv, xw, xa, xg, xv,
          hp(w.att_w1_t), hp(w.att_a1_t), hp(w.att_g1_t), hp(w.att_v1_t), w1, a1, g1, v1);
    } else if (C >= kLowrankFusedMinC && rows <= kLowrankInRowsT) {
      rwkv7_v3a_linear_wag_rank_in_f16_launch(
          stream, rows, C, Rw, Ra, Rg, xw, xa, xg, hp(w.att_w1_t), hp(w.att_a1_t), hp(w.att_g1_t), w1, a1, g1);
    } else {
      linear_rank_in_launch(stream, rows, C, Rw, xw, hp(w.att_w1), hp(w.att_w1_t), w1);
      linear_rank_in_launch(stream, rows, C, Ra, xa, hp(w.att_a1), hp(w.att_a1_t), a1);
      linear_rank_in_launch(stream, rows, C, Rg, xg, hp(w.att_g1), hp(w.att_g1_t), g1);
    }
    profiler.end(stream, profile_att_lowrank);

    const int profile_att_lowrank_out = profiler.begin(stream, "att_lowrank_out");
    if (C >= kLowrankFusedMinC && rows <= kLowrankOutRowsT && layer != 0 && rows <= kLowrankInRowsT) {
      rwkv7_v3a_linear_wagv_rank_out_f16_launch(
          stream, rows, C, Rw, Ra, Rg, Rv, w1, a1, g1, v1,
          hp(w.att_w2_t), hp(w.att_a2_t), hp(w.att_g2_t), hp(w.att_v2_t),
          v_base, v_first, hp(w.att_v0), w12, a12, g, v_out);
      v_use = v_out;
      v_done = true;
    } else if (C >= kLowrankFusedMinC && rows <= kLowrankOutRowsT) {
      rwkv7_v3a_linear_wag_rank_out_f16_launch(
          stream, rows, C, Rw, Ra, Rg, w1, a1, g1, hp(w.att_w2_t), hp(w.att_a2_t), hp(w.att_g2_t), w12, a12, g);
    } else {
      linear_rank_out_act_launch(stream, rows, Rw, C, w1, hp(w.att_w2), hp(w.att_w2_t), 1, w_raw, w12);
      linear_rank_out_launch(stream, rows, Ra, C, a1, hp(w.att_a2), hp(w.att_a2_t), a12);
      linear_rank_out_act_launch(stream, rows, Rg, C, g1, hp(w.att_g2), hp(w.att_g2_t), 2, w_raw, g);
    }

    if (layer == 0) {
      check_cuda(cudaMemcpyAsync(v_first, v_base, row_elems * sizeof(half), cudaMemcpyDeviceToDevice, stream), "copy v_first");
    } else if (!v_done) {
      if (C >= kLowrankFusedMinC && rows <= kLowrankOutRowsT) {
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
    profiler.end(stream, profile_att_lowrank_out);

    const int profile_kk_gate = profiler.begin(stream, "tmix_kk_a_gate");
    rwkv7_tmix_kk_a_gate_launch(stream, B, T, C, H, k, hp(w.att_k_k), hp(w.att_a0), a12, hp(w.att_k_a), k2, neg_kk, kka);
    profiler.end(stream, profile_kk_gate);
    const int profile_wkv = profiler.begin(stream, "wkv");
    if (use_wkv32) {
      rwkv7_add_vec_launch(stream, C, w12, hp(w.att_w0), w_raw, row_elems);
      rwkv7_wkv_fp32io16_launch(stream, B, T, C, H, 0, layer_state32, r, w_raw, k2, v_use, neg_kk, kka, y);
    } else if (T <= 16) {
      rwkv7_wkv_fp16_seq_w0_launch(stream, B, T, C, H, layer_state16, r, w12, hp(w.att_w0), k2, v_use, neg_kk, kka, y, state.elapsed.p);
    } else {
      rwkv7_add_vec_launch(stream, C, w12, hp(w.att_w0), w_raw, row_elems);
      rwkv7_wkv_fp16_seq_launch(stream, B, T, C, H, layer_state16, r, w_raw, k2, v_use, neg_kk, kka, y, state.elapsed.p);
    }
    profiler.end(stream, profile_wkv);

    const int profile_att_post = profiler.begin(stream, "tmix_lnx_rkvres_xg");
    rwkv7_tmix_lnx_rkvres_xg_launch(stream, B, T, C, H, y, r, k2, v_use, hp(w.att_r_k), hp(w.att_ln_x_w), hp(w.att_ln_x_b), g, y2);
    profiler.end(stream, profile_att_post);
    const int profile_att_output = profiler.begin(stream, "att_output");
    linear_orig_layout_launch(stream, path, LinearGroup::AttC2C, rows, C, C, y2, w.att_output_w, lt_workspace.p, lt_workspace.n, att_out);
    profiler.end(stream, profile_att_output);
    const int profile_ffn_mix = profiler.begin(stream, "ffn_mix");
    if (T == 1) {
      rwkv7_v3a_add_layer_norm_cmix_mix_f16_launch(
          stream, rows, C, x_cur, att_out, shift1, hp(w.ln2_w), hp(w.ln2_b), hp(w.ffn_x_k), x_after_att, mixed, kLnEps);
    } else {
      rwkv7_v3a_add_layer_norm_f16_launch(
          stream, rows, C, x_cur, att_out, hp(w.ln2_w), hp(w.ln2_b), x_after_att, ln2_out, kLnEps);
      rwkv7_cmix_mix_launch(stream, B, T, C, ln2_out, shift1, hp(w.ffn_x_k), mixed);
    }
    profiler.end(stream, profile_ffn_mix);

    const int profile_ffn_key = profiler.begin(stream, "ffn_key");
    linear_orig_layout_launch(stream, path, LinearGroup::FfnKey, rows, C, F, mixed, w.ffn_key_w, lt_workspace.p, lt_workspace.n, hid);
    profiler.end(stream, profile_ffn_key);
    if (weights.cmix_stats.enabled && T == 1) {
      rwkv7_cmix_stats_relu2_launch(
          stream, rows, F, hid, weights.cmix_stats.nonzero.p + layer,
          weights.cmix_stats.total.p + layer, weights.cmix_stats.max_bits.p + layer);
    }
    const int layer_sparse_max_rows =
        layer < static_cast<int>(weights.cmix_sparse_max_rows_by_layer.size())
            ? weights.cmix_sparse_max_rows_by_layer[static_cast<std::size_t>(layer)]
            : weights.cmix_sparse_max_rows;
    const CmixMode cmix_mode =
        run.cmix_sparse != "off" && path.cmix == CmixMode::Dense && w.ffn_value_w->is_int8() && C >= 4096 &&
                rows <= layer_sparse_max_rows
            ? CmixMode::NoFcRows2
            : path.cmix;
    const int profile_ffn_value = profiler.begin(stream, "ffn_value");
    if (cmix_mode == CmixMode::NoFcOne) {
      if (w.ffn_value_w->is_int8()) {
        rwkv7_cmix_sparse_down_relu_one_i8_launch(
            stream, C, F, hid, w.ffn_value_w->i8.p,
            reinterpret_cast<const half*>(w.ffn_value_w->scale.p), cmix_out);
      } else {
        rwkv7_cmix_sparse_down_relu_one_launch(stream, C, F, hid, hp(w.ffn_value_w), cmix_out);
      }
    } else if (cmix_mode == CmixMode::NoFcRows2) {
      if (rows >= 8) {
        if (w.ffn_value_w->is_int8()) {
          rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(
              stream, B, T, C, F, hid, w.ffn_value_w->i8.p,
              reinterpret_cast<const half*>(w.ffn_value_w->scale.p), cmix_out);
        } else {
          rwkv7_cmix_sparse_down_relu_rows_t512_launch(stream, B, T, C, F, hid, hp(w.ffn_value_w), cmix_out);
        }
      } else {
        if (w.ffn_value_w->is_int8()) {
          rwkv7_cmix_sparse_down_relu_rows_i8_launch(
              stream, B, T, C, F, hid, w.ffn_value_w->i8.p,
              reinterpret_cast<const half*>(w.ffn_value_w->scale.p), cmix_out);
        } else {
          rwkv7_cmix_sparse_down_relu_rows_launch(stream, B, T, C, F, hid, hp(w.ffn_value_w), cmix_out);
        }
      }
    } else {
      rwkv7_relu_square_launch(stream, hid, hid, static_cast<long long>(rows) * F);
      if (w.ffn_value_w->is_int8()) {
        rwkv7_w8a16_linear_launch(
            stream, rows, F, C, hid, w.ffn_value_w->i8.p,
            reinterpret_cast<const half*>(w.ffn_value_w->scale.p),
            w.ffn_value_w->i8_packed ? W8BLayout::PackedNK
                                     : (w.ffn_value_w->i8_transposed ? W8BLayout::KN : W8BLayout::NK),
            cmix_out, lt_workspace.p, lt_workspace.n);
      } else {
        rwkv7_v3a_linear_f16_launch(stream, rows, F, C, hid, hp(w.ffn_value_w), cmix_out);
      }
    }
    profiler.end(stream, profile_ffn_value);

    if (layer + 1 < dims.layers) {
      const LayerWeights& next = weights.layers[layer + 1];
      if (B == 1 && T == 1) {
        half* next_shift0 = state.shift.p + static_cast<std::size_t>(layer + 1) * 2 * B * C;
        const int profile_next_mix = profiler.begin(stream, "layer_norm_tmix_mix6");
        rwkv7_v3a_add_layer_norm_tmix_mix6_f16_launch(
            stream, rows, C, x_after_att, cmix_out, next_shift0, hp(next.ln1_w), hp(next.ln1_b),
            hp(next.att_x_r), hp(next.att_x_w), hp(next.att_x_k), hp(next.att_x_v), hp(next.att_x_a), hp(next.att_x_g),
            x_next, xr, xw, xk, xv, xa, xg, kLnEps);
        profiler.end(stream, profile_next_mix);
        xx_next = x_next;
        pre_mix_ready = true;
      } else {
        const int profile_next_ln = profiler.begin(stream, "layer_norm_next");
        rwkv7_v3a_add_layer_norm_f16_launch(stream, rows, C, x_after_att, cmix_out, hp(next.ln1_w), hp(next.ln1_b), x_next, xx_next, kLnEps);
        profiler.end(stream, profile_next_ln);
      }
      std::swap(x_cur, x_next);
      std::swap(xx_cur, xx_next);
    } else {
      const int profile_last_ln = profiler.begin(stream, "layer_norm_last");
      rwkv7_v3a_add_last_layer_norm_f16_launch(
          stream, B, T, C, x_after_att, cmix_out, hp(weights.ln_out_w), hp(weights.ln_out_b), final_x, kLnEps);
      profiler.end(stream, profile_last_ln);
    }
    check_cuda(cudaGetLastError(), "launch backend layer");
  }

  const int profile_advance = profiler.begin(stream, "advance_elapsed");
  rwkv7_v3a_advance_i32_launch(stream, state.elapsed.p, T, B);
  profiler.end(stream, profile_advance);
  PathConfig head_path;
  head_path.rows = output_rows;
  head_path.use_batched_rkv = false;
  head_path.cmix = CmixMode::Dense;
  const int profile_head = profiler.begin(stream, "head");
  linear_orig_layout_launch(
      stream,
      head_path,
      LinearGroup::Head,
      output_rows,
      C,
      V,
      final_x,
      weights.head_w,
      lt_workspace.p,
      lt_workspace.n,
      logits_f16);
  profiler.end(stream, profile_head);
  check_cuda(cudaGetLastError(), "launch backend head");
  out.rows = output_rows;
  out.vocab_size = V;
  out.values.resize(static_cast<std::size_t>(output_rows) * V, "alloc backend logits f32");
  const int profile_logits = profiler.begin(stream, "f16_to_f32_logits");
  rwkv7_v4_f16_to_f32_launch(stream, logits_f16, out.values.p, out.values.n);
  profiler.end(stream, profile_logits);
  check_cuda(cudaGetLastError(), "launch backend logits f16->f32");
  check_cuda(cudaStreamSynchronize(stream), "sync backend stream");
  profiler.report();
}

}  // namespace

struct ModelBackend::Impl {
  explicit Impl(std::string path, bool use_wkv32_, bool chunk_load_, std::string cmix_sparse_,
                std::string tune_cache_, bool retune_, std::string tune_cache_directory_)
      : model_path(std::move(path)),
        model_name(basename_without_extension(model_path)),
        use_wkv32(use_wkv32_),
        chunk_load(chunk_load_),
        cmix_sparse(std::move(cmix_sparse_)),
        tune_cache(std::move(tune_cache_)),
        retune(retune_),
        tune_cache_directory(std::move(tune_cache_directory_)),
        weights(load_backend_weights(model_path, chunk_load)) {
    configure_w8a16_tuning(weights, model_path, tune_cache, tune_cache_directory, retune);
  }

  ~Impl() {
    if (!weights.cmix_stats.enabled) return;
    std::vector<unsigned long long> nonzero(weights.dims.layers);
    std::vector<unsigned long long> total(weights.dims.layers);
    std::vector<unsigned int> max_bits(weights.dims.layers);
    check_cuda(cudaMemcpy(nonzero.data(), weights.cmix_stats.nonzero.p,
                          nonzero.size() * sizeof(nonzero.front()), cudaMemcpyDeviceToHost),
               "copy cmix stats nonzero");
    check_cuda(cudaMemcpy(total.data(), weights.cmix_stats.total.p,
                          total.size() * sizeof(total.front()), cudaMemcpyDeviceToHost),
               "copy cmix stats total");
    check_cuda(cudaMemcpy(max_bits.data(), weights.cmix_stats.max_bits.p,
                          max_bits.size() * sizeof(max_bits.front()), cudaMemcpyDeviceToHost),
               "copy cmix stats max");
    std::cout << "cmix_stats_begin\n";
    for (int layer = 0; layer < weights.dims.layers; ++layer) {
      const double ratio = total[layer] == 0
                               ? 0.0
                               : static_cast<double>(nonzero[layer]) / static_cast<double>(total[layer]);
      float max_relu2 = 0.0f;
      std::memcpy(&max_relu2, &max_bits[layer], sizeof(max_relu2));
      std::cout << "cmix_stats layer=" << layer
                << " decode_steps=" << (weights.dims.ffn == 0 ? 0 : total[layer] / weights.dims.ffn)
                << " nonzero=" << nonzero[layer]
                << " total=" << total[layer]
                << " ratio=" << ratio
                << " max_relu2=" << max_relu2;
      const GpuTensor* value = weights.layers[static_cast<std::size_t>(layer)].ffn_value_w;
      if (value != nullptr && value->is_int8() && value->scale.n > 0) {
        std::vector<std::uint16_t> scales(value->scale.n);
        check_cuda(cudaMemcpy(scales.data(), value->scale.p,
                              scales.size() * sizeof(scales.front()), cudaMemcpyDeviceToHost),
                   "copy cmix stats scales");
        float min_scale = std::numeric_limits<float>::infinity();
        for (std::uint16_t bits : scales) {
          min_scale = std::min(min_scale, __half2float(*reinterpret_cast<const half*>(&bits)));
        }
        const double average_nnz = ratio * weights.dims.ffn;
        const double overflow_est = min_scale > 0.0f
                                        ? static_cast<double>(max_relu2) * average_nnz * 127.0 / min_scale
                                        : std::numeric_limits<double>::infinity();
        std::cout << " scale_min=" << min_scale << " overflow_est=" << overflow_est;
      }
      std::cout << "\n";
    }
    std::cout << "cmix_stats_end\n";
  }

  std::string model_path;
  std::string model_name;
  bool use_wkv32 = false;
  bool chunk_load = false;
  std::string cmix_sparse = "no-fc";
  std::string tune_cache;
  bool retune = false;
  std::string tune_cache_directory;
  CudaWeights weights;
};

ModelBackend::ModelBackend(std::string model_path, bool use_wkv32, bool chunk_load, std::string cmix_sparse,
                           std::string tune_cache, bool retune, std::string tune_cache_directory)
    : impl_(std::make_unique<Impl>(std::move(model_path), use_wkv32, chunk_load,
                                   std::move(cmix_sparse), std::move(tune_cache), retune,
                                   std::move(tune_cache_directory))) {}

ModelBackend::~ModelBackend() = default;

ModelBackend::ModelBackend(ModelBackend&&) noexcept = default;

ModelBackend& ModelBackend::operator=(ModelBackend&&) noexcept = default;

GenerationState ModelBackend::create_state(int batch_size) const {
  if (batch_size <= 0) {
    throw std::runtime_error("batch_size must be positive");
  }
  GenerationState state;
  state.batch_size = batch_size;
  state.wkv32 = impl_->use_wkv32;
  const auto& dims = impl_->weights.dims;
  const std::size_t shift_elems = static_cast<std::size_t>(dims.layers) * 2 * batch_size * dims.channels;
  const std::size_t state_elems =
      static_cast<std::size_t>(dims.layers) * batch_size * dims.heads * dims.head_size * dims.head_size;
  state.shift.resize(shift_elems, "alloc backend shift state");
  if (impl_->use_wkv32) {
    state.wkv_state32.resize(state_elems, "alloc backend wkv32 state");
  } else {
    state.wkv_state.resize(state_elems, "alloc backend wkv state");
  }
  state.elapsed.resize(batch_size, "alloc backend elapsed");
  state.shift.zero("zero backend shift");
  if (impl_->use_wkv32) {
    check_cuda(cudaMemset(state.wkv_state32.p, 0, state.wkv_state32.n * sizeof(float)), "zero backend wkv32");
  } else {
    state.wkv_state.zero("zero backend wkv");
  }
  check_cuda(cudaMemset(state.elapsed.p, 0, static_cast<std::size_t>(batch_size) * sizeof(int)), "zero backend elapsed");
  check_cuda(cudaDeviceSynchronize(), "sync backend state init");
  return state;
}

void ModelBackend::forward_prefill(
    const std::vector<std::vector<int64_t>>& token_batches,
    GenerationState& state,
    DeviceLogits& logits) const {
  run_backend_forward(impl_->weights, token_batches, impl_->use_wkv32, impl_->cmix_sparse, state, logits);
}

void ModelBackend::forward_decode(
    const std::vector<int64_t>& token_batch,
    GenerationState& state,
    DeviceLogits& logits) const {
  std::vector<std::vector<int64_t>> batch;
  batch.reserve(token_batch.size());
  for (int64_t token : token_batch) {
    batch.push_back({token});
  }
  run_backend_forward(impl_->weights, batch, impl_->use_wkv32, impl_->cmix_sparse, state, logits);
}

void ModelBackend::copy_state_slice(
    const GenerationState& src,
    int src_offset,
    GenerationState& dst,
    int dst_offset,
    int count) const {
  if (src_offset < 0 || dst_offset < 0 || count <= 0) {
    throw std::runtime_error("invalid backend state slice range");
  }
  if (src.batch_size <= 0 || dst.batch_size <= 0) {
    throw std::runtime_error("state slice requires non-empty source and destination");
  }
  if (src.wkv32 != dst.wkv32 || src.wkv32 != impl_->use_wkv32) {
    throw std::runtime_error("state slice wkv precision mismatch");
  }
  if (src_offset + count > src.batch_size || dst_offset + count > dst.batch_size) {
    throw std::runtime_error("state slice range exceeds batch size");
  }

  const auto& dims = impl_->weights.dims;
  const std::size_t shift_lane_elems = static_cast<std::size_t>(dims.channels);
  const std::size_t wkv_lane_elems =
      static_cast<std::size_t>(dims.heads) * dims.head_size * dims.head_size;

  for (int layer = 0; layer < dims.layers; ++layer) {
    half* src_shift0 = src.shift.p + static_cast<std::size_t>(layer) * 2 * src.batch_size * dims.channels +
                       static_cast<std::size_t>(src_offset) * dims.channels;
    half* src_shift1 = src.shift.p + static_cast<std::size_t>(layer) * 2 * src.batch_size * dims.channels +
                       static_cast<std::size_t>(src.batch_size + src_offset) * dims.channels;
    half* dst_shift0 = dst.shift.p + static_cast<std::size_t>(layer) * 2 * dst.batch_size * dims.channels +
                       static_cast<std::size_t>(dst_offset) * dims.channels;
    half* dst_shift1 = dst.shift.p + static_cast<std::size_t>(layer) * 2 * dst.batch_size * dims.channels +
                       static_cast<std::size_t>(dst.batch_size + dst_offset) * dims.channels;

    check_cuda(
        cudaMemcpy(
            dst_shift0,
            src_shift0,
            static_cast<std::size_t>(count) * shift_lane_elems * sizeof(half),
            cudaMemcpyDeviceToDevice),
        "copy backend state slice shift0");
    check_cuda(
        cudaMemcpy(
            dst_shift1,
            src_shift1,
            static_cast<std::size_t>(count) * shift_lane_elems * sizeof(half),
            cudaMemcpyDeviceToDevice),
        "copy backend state slice shift1");

    if (impl_->use_wkv32) {
      float* src_wkv = src.wkv_state32.p + static_cast<std::size_t>(layer) * src.batch_size * wkv_lane_elems +
                       static_cast<std::size_t>(src_offset) * wkv_lane_elems;
      float* dst_wkv = dst.wkv_state32.p + static_cast<std::size_t>(layer) * dst.batch_size * wkv_lane_elems +
                       static_cast<std::size_t>(dst_offset) * wkv_lane_elems;
      check_cuda(
          cudaMemcpy(
              dst_wkv,
              src_wkv,
              static_cast<std::size_t>(count) * wkv_lane_elems * sizeof(float),
              cudaMemcpyDeviceToDevice),
          "copy backend state slice wkv32");
    } else {
      half* src_wkv = src.wkv_state.p + static_cast<std::size_t>(layer) * src.batch_size * wkv_lane_elems +
                      static_cast<std::size_t>(src_offset) * wkv_lane_elems;
      half* dst_wkv = dst.wkv_state.p + static_cast<std::size_t>(layer) * dst.batch_size * wkv_lane_elems +
                      static_cast<std::size_t>(dst_offset) * wkv_lane_elems;
      check_cuda(
          cudaMemcpy(
              dst_wkv,
              src_wkv,
              static_cast<std::size_t>(count) * wkv_lane_elems * sizeof(half),
              cudaMemcpyDeviceToDevice),
          "copy backend state slice wkv16");
    }
  }

  check_cuda(
      cudaMemcpy(
          dst.elapsed.p + dst_offset,
          src.elapsed.p + src_offset,
          static_cast<std::size_t>(count) * sizeof(int),
          cudaMemcpyDeviceToDevice),
      "copy backend state slice elapsed");
}

void ModelBackend::copy_logits_slice(
    const DeviceLogits& src,
    int src_offset,
    DeviceLogits& dst,
    int dst_offset,
    int count) const {
  if (src_offset < 0 || dst_offset < 0 || count <= 0) {
    throw std::runtime_error("invalid backend logits slice range");
  }
  if (src.rows <= 0 || src.vocab_size <= 0 || src.values.p == nullptr) {
    throw std::runtime_error("logits slice requires non-empty source");
  }
  if (dst.rows <= 0 || dst.vocab_size != src.vocab_size || dst.values.p == nullptr) {
    throw std::runtime_error("logits slice destination shape mismatch");
  }
  if (src_offset + count > src.rows || dst_offset + count > dst.rows) {
    throw std::runtime_error("logits slice range exceeds row count");
  }

  check_cuda(
      cudaMemcpy(
          dst.values.p + static_cast<std::size_t>(dst_offset) * dst.vocab_size,
          src.values.p + static_cast<std::size_t>(src_offset) * src.vocab_size,
          static_cast<std::size_t>(count) * src.vocab_size * sizeof(float),
          cudaMemcpyDeviceToDevice),
      "copy backend logits slice");
}

PrefillCapacity ModelBackend::query_prefill_capacity(int prefill_chunk_size) const {
  if (prefill_chunk_size <= 0) {
    throw std::runtime_error("prefill chunk size must be positive");
  }

  PrefillCapacity capacity;
  check_cuda(
      cudaMemGetInfo(&capacity.free_vram_bytes, &capacity.total_vram_bytes),
      "query prefill CUDA memory");

  const auto& dims = impl_->weights.dims;
  const std::size_t layers = static_cast<std::size_t>(dims.layers);
  const std::size_t channels = static_cast<std::size_t>(dims.channels);
  const std::size_t heads = static_cast<std::size_t>(dims.heads);
  const std::size_t head_size = static_cast<std::size_t>(dims.head_size);
  const std::size_t vocab = static_cast<std::size_t>(dims.vocab);
  const std::size_t ffn = static_cast<std::size_t>(dims.ffn);
  const std::size_t tokens = static_cast<std::size_t>(prefill_chunk_size);

  const std::size_t shift_bytes = layers * 2 * channels * sizeof(half);
  const std::size_t wkv_bytes =
      layers * heads * head_size * head_size *
      (impl_->use_wkv32 ? sizeof(float) : sizeof(half));
  const std::size_t state_bytes = shift_bytes + wkv_bytes + sizeof(int);

  // Match the peak allocations in run_backend_forward for one batch lane.
  const std::size_t arena_half_elements =
      tokens * channels * 31 + channels + tokens * ffn +
      tokens * static_cast<std::size_t>(kLowrankMax) * 4 + vocab;
  const std::size_t arena_bytes = arena_half_elements * sizeof(half);
  const std::size_t logits_bytes = vocab * sizeof(float);
  capacity.bytes_per_batch = state_bytes + arena_bytes + logits_bytes;

  constexpr std::size_t kMinimumReserveBytes = static_cast<std::size_t>(512) << 20;
  constexpr std::size_t kCublasLtWorkspaceBytes = static_cast<std::size_t>(128) << 20;
  capacity.reserve_vram_bytes = std::max(kMinimumReserveBytes, capacity.free_vram_bytes / 10);
  const std::size_t unavailable = capacity.reserve_vram_bytes + kCublasLtWorkspaceBytes;
  const std::size_t usable = capacity.free_vram_bytes > unavailable
      ? capacity.free_vram_bytes - unavailable
      : 0;
  const std::size_t max_batch = capacity.bytes_per_batch > 0
      ? usable / capacity.bytes_per_batch
      : 0;
  capacity.max_batch_size = static_cast<int>(
      std::min<std::size_t>(max_batch, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return capacity;
}

int ModelBackend::vocab_size() const {
  return impl_->weights.dims.vocab;
}

const std::string& ModelBackend::model_path() const {
  return impl_->model_path;
}

const std::string& ModelBackend::model_name() const {
  return impl_->model_name;
}

}  // namespace rwkv7_server
