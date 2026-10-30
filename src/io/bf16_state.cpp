#include "rwkv/bf16_training/model.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include <cmath>
#include <cstring>

namespace rwkv_bf16_training {
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
