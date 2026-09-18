#include "trainer.hpp"
#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/io/pth_writer.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
namespace rwkv_bf16_miss {
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
std::string name(const HostTensor &t) {
  return "blocks." + std::to_string(t.layer) + "." + target_names[t.target] +
         ".D";
}
} // namespace
Trainer::Trainer(rwkv_bf16_training::FrozenModelView &model, int rank,
                 float alpha, const std::string &targets, int time,
                 Json::Value config, cudaStream_t stream)
    : stream_(stream), config_(config), rank_(rank), alpha_(alpha) {
  if (rank < 1 || rank > 1024 || !std::isfinite(alpha))
    throw std::invalid_argument("invalid MiSS rank/alpha");
  std::set<int> selected;
  if (targets == "all")
    for (int i = 0; i < TargetCount; ++i)
      selected.insert(i);
  else {
    std::istringstream list(targets);
    std::string t;
    while (std::getline(list, t, ',')) {
      int i = 0;
      for (; i < TargetCount; ++i)
        if (t == target_names[i])
          break;
      if (i == TargetCount || !selected.insert(i).second)
        throw std::invalid_argument("unknown/duplicate MiSS target: " + t);
    }
  }
  if (selected.empty())
    throw std::invalid_argument("empty MiSS targets");
  size_t count = 0;
  reduced_.resize(model.layers);
  for (int l = 0; l < model.layers; ++l)
    for (int i : selected) {
      const auto &w = model.blocks[l];
      HostTensor t{l,
                   i,
                   i == FfnValue ? w.ffn : w.channels,
                   i == FfnKey ? w.ffn : w.channels,
                   rank,
                   count};
      count += size_t(t.out) * rank;
      targets_.push_back(t);
      reduced_[l][i].resize(size_t(time) * rank, "MiSS reduced tape");
    }
  master_.resize(count, "MiSS FP32 master");
  gradient_.resize(count, "MiSS FP32 gradient");
  m_.resize(count, "MiSS Adam m");
  v_.resize(count, "MiSS Adam v");
  forward_.resize(count, "MiSS BF16 D");
  for (auto *p : {master_.p, gradient_.p, m_.p, v_.p})
    check(cudaMemsetAsync(p, 0, count * 4, stream_));
  check(cudaMemsetAsync(forward_.p, 0, count * 2, stream_));
  for (const auto &t : targets_)
    model.blocks[t.layer].miss[t.target] = {t.in,
                                            t.out,
                                            rank,
                                            alpha / rank,
                                            forward_.p + t.offset,
                                            gradient_.p + t.offset};
}
void Trainer::bind_tapes(
    std::vector<rwkv_bf16_training::BlockTapeView> &tapes) {
  for (const auto &t : targets_)
    tapes[t.layer].miss_reduced[t.target] = reduced_[t.layer][t.target].p;
}
void Trainer::update(float lr, uint64_t step) {
  // Check the whole accumulated gradient before touching any optimizer tensor.
  std::vector<float> g(gradient_.n);
  check(cudaMemcpyAsync(g.data(), gradient_.p, g.size() * 4,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  for (float x : g)
    if (!std::isfinite(x) || std::abs(x) > 1e18f)
      throw std::runtime_error(
          "nonfinite/overflow MiSS gradient; optimizer not executed");
  rwkv_bf16_training::AdamConfig adam;
  adam.learning_rate = lr;
  rwkv_bf16_training::adam_update_state_f32(
      stream_, master_.p, gradient_.p, m_.p, v_.p, master_.n, step, adam, true);
  cast_master(stream_, master_.p, forward_.p, master_.n);
}
void Trainer::checkpoint(const std::string &dir, uint64_t step, int epoch,
                         uint64_t row, const float *initial, size_t states) {
  if (std::filesystem::exists(dir) || std::filesystem::exists(dir + ".tmp"))
    throw std::runtime_error("checkpoint destination exists");
  std::filesystem::create_directories(dir + ".tmp");
  std::vector<llm_infer::WriteTensor> ts;
  auto add = [&](const std::string &n, const float *p, std::vector<int> shape,
                 size_t count) {
    llm_infer::WriteTensor t{n, shape, false, {}};
    t.data.resize(count * 4);
    check(cudaMemcpyAsync(t.data.data(), p, count * 4, cudaMemcpyDeviceToHost,
                          stream_));
    check(cudaStreamSynchronize(stream_));
    ts.push_back(std::move(t));
  };
  for (const auto &t : targets_)
    for (const auto &p : std::vector<std::pair<std::string, const float *>>{
             {"master", master_.p},
             {"gradient", gradient_.p},
             {"adam_m", m_.p},
             {"adam_v", v_.p}})
      add(name(t) + "." + p.first, p.second + t.offset, {t.out, t.rank},
          size_t(t.out) * rank_);
  add("initial_state", initial, {int(states)}, states);
  std::vector<uint16_t> forward(forward_.n);
  check(cudaMemcpyAsync(forward.data(), forward_.p, forward.size() * 2,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  Json::Value manifest;
  manifest["dtype"] = "bfloat16";
  manifest["rank"] = rank_;
  manifest["alpha"] = alpha_;
  manifest["scale"] = alpha_ / rank_;
  manifest["base_fingerprint"] = config_["base_fingerprint"];
  manifest = adapter_manifest(manifest, targets_, forward);
  manifest["tensor_source"] = "master";
  llm_infer::write_pth(
      dir + ".tmp/training.pth", ts,
      Json::writeString(Json::StreamWriterBuilder{}, manifest));
  Json::Value m;
  m["kind"] = "miss_training_checkpoint";
  m["format_version"] = 1;
  m["config"] = config_;
  m["step"] = Json::UInt64(step);
  m["epoch"] = epoch;
  m["row"] = Json::UInt64(row);
  m["tensor_digest"] = fingerprint_file(dir + ".tmp/training.pth");
  std::mt19937_64 rng(config_["seed"].asUInt64());
  std::ostringstream rng_state;
  rng_state << rng;
  m["rng"] = rng_state.str();
  m["scheduler"] = config_["planned_steps"];
  {
    std::ofstream f(dir + ".tmp/checkpoint.json");
    f << m;
    if (!f)
      throw std::runtime_error("checkpoint metadata write failed");
  }
  std::filesystem::rename(dir + ".tmp", dir);
}
void Trainer::resume(const std::string &dir, uint64_t &step, int &epoch,
                     uint64_t &row, float *initial, size_t states) {
  Json::Value m;
  std::ifstream f(dir + "/checkpoint.json");
  if (!f)
    throw std::runtime_error("resume requires a training checkpoint directory");
  f >> m;
  if (m["kind"] != "miss_training_checkpoint" || m["format_version"] != 1 ||
      m["config"]["training_dtype"] != "bfloat16" ||
      Json::writeString(Json::StreamWriterBuilder{}, m["config"]) !=
          Json::writeString(Json::StreamWriterBuilder{}, config_))
    throw std::runtime_error(
        "resume requires a matching BF16 training checkpoint/config");
  if (m["tensor_digest"].asString() != fingerprint_file(dir + "/training.pth"))
    throw std::runtime_error("checkpoint digest mismatch");
  std::mt19937_64 rng;
  std::istringstream rs(m["rng"].asString());
  if (!(rs >> rng))
    throw std::runtime_error("invalid RNG state");
  auto a = llm_infer::PthArchive::open(dir + "/training.pth", true);
  if (!a.ok())
    throw std::runtime_error(a.status().message());
  auto records = llm_infer::parse_pth_tensor_records(a.value());
  if (!records.ok())
    throw std::runtime_error(records.status().message());
  if (records.value().size() != targets_.size() * 4 + 1)
    throw std::runtime_error("checkpoint tensor count");
  auto load = [&](const std::string &n, float *p, std::vector<int64_t> shape) {
    const llm_infer::TensorRecord *r = nullptr;
    for (const auto &t : records.value())
      if (t.name == n) {
        if (r)
          throw std::runtime_error("duplicate checkpoint tensor");
        r = &t;
      }
    if (!r || r->shape != shape || r->dtype != llm_infer::TensorDType::kFloat32)
      throw std::runtime_error("checkpoint tensor mismatch: " + n);
    auto d = llm_infer::load_tensor_select(a.value(), *r, false, false, true);
    if (!d.ok())
      throw std::runtime_error(d.status().message());
    for (float x : d.value().values)
      if (!std::isfinite(x))
        throw std::runtime_error("nonfinite checkpoint");
    check(cudaMemcpyAsync(p, d.value().values.data(),
                          d.value().values.size() * 4, cudaMemcpyHostToDevice,
                          stream_));
    check(cudaStreamSynchronize(stream_));
  };
  for (const auto &t : targets_)
    for (const auto &p :
         std::vector<std::pair<std::string, float *>>{{"master", master_.p},
                                                      {"gradient", gradient_.p},
                                                      {"adam_m", m_.p},
                                                      {"adam_v", v_.p}})
      load(name(t) + "." + p.first, p.second + t.offset, {t.out, t.rank});
  load("initial_state", initial, {int64_t(states)});
  cast_master(stream_, master_.p, forward_.p, master_.n);
  step = m["step"].asUInt64();
  epoch = m["epoch"].asInt();
  row = m["row"].asUInt64();
}
void Trainer::export_adapter(const std::string &dir) {
  std::vector<uint16_t> data(forward_.n);
  check(cudaMemcpyAsync(data.data(), forward_.p, data.size() * 2,
                        cudaMemcpyDeviceToHost, stream_));
  check(cudaStreamSynchronize(stream_));
  Json::Value m;
  m["dtype"] = "bfloat16";
  m["rank"] = rank_;
  m["alpha"] = alpha_;
  m["scale"] = alpha_ / rank_;
  m["base_fingerprint"] = config_["base_fingerprint"];
  save_adapter_pth(dir, m, targets_, data);
}
} // namespace rwkv_bf16_miss
