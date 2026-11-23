#include "bf16_model_fixture.hpp"
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/bf16_training/trainer.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

using namespace rwkv_bf16_training;
namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class T> std::vector<T> read(const T *p, size_t n) {
  std::vector<T> host(n);
  gpu_check(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost),
            "read regression output");
  return host;
}
template <class T> void fill(DeviceBuffer<T> &buffer, float value) {
  T converted;
  if constexpr (std::is_same_v<T, bf16>)
    converted = to_bf16(value);
  else
    converted = value;
  std::vector<T> host(buffer.n, converted);
  gpu_check(cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice), "fill regression buffer");
}
float scalar(float x) { return x; }
float scalar(bf16 x) { return to_float(x); }
template <class T>
void equal(const std::vector<T> &a, const std::vector<T> &b, const char *name) {
  require(a.size() == b.size(), "gradient size mismatch");
  for (size_t i = 0; i < a.size(); ++i)
    require(std::isfinite(scalar(a[i])) && scalar(a[i]) == scalar(b[i]), name);
}
void written(const bf16 *p, size_t n) {
  for (bf16 x : read(p, n))
    require(std::isfinite(to_float(x)) && to_float(x) != 7.f,
            "requested gradient was not written");
}
std::vector<float> adapter_gradients(const FrozenModelView &m) {
  std::vector<float> out;
  for (const auto &b : m.blocks)
    for (const auto &a : b.miss)
      for (bool first : {false, true}) {
        const float *p = first ? a.gradient_a : a.gradient;
        if (p) {
          auto h = read(p, size_t(a.rank) * (first ? a.in : a.out));
          out.insert(out.end(), h.begin(), h.end());
        }
      }
  return out;
}
void check(int layers, const std::string &method, bool residual_only,
           cudaStream_t stream) {
  constexpr int T = 8, C = 64;
  Bf16ModelFixture fixture(layers, residual_only);
  auto m = fixture.view;
  std::unique_ptr<rwkv_bf16_miss::Trainer> trainer;
  if (method != "state") {
    Json::Value config;
    if (method == "nora") {
      config["method"] = "nora";
      config["initialization"] = "nora_init";
    }
    trainer = std::make_unique<rwkv_bf16_miss::Trainer>(
        m, 16, 16, "all", T, config, stream);
  }
  Executor executor(m, T, stream);
  if (trainer)
    trainer->bind_tapes(executor.tapes());
  DeviceBuffer<float> initial, terminal, ds;
  for (auto *b : {&initial, &terminal, &ds})
    b->resize(layers * 4096, "state gradient test");
  fill(initial, .01f);
  fill(terminal, residual_only ? 0.f : .02f);
  DeviceBuffer<bf16> go, ping, pong, first, shifts, final_shifts, workspace;
  for (auto *b : {&go, &ping, &pong, &first})
    b->resize(T * C, "model gradient test");
  shifts.resize(layers * 2 * C, "shift gradient test");
  final_shifts.resize(shifts.n, "terminal shift gradient test");
  fill(go, .125f);
  fill(final_shifts, residual_only ? 0.f : .03f);
  workspace.resize(block_backward_workspace_bf16_elements(1, T, m.blocks[0]),
                   "backward workspace test");
  std::vector<BlockBackwardState> gradients(layers);
  for (int l = 0; l < layers; ++l)
    gradients[l] = {terminal.p + l * 4096, ds.p + l * 4096,
                    shifts.p + 2 * l * C, shifts.p + (2 * l + 1) * C,
                    first.p, final_shifts.p + 2 * l * C,
                    final_shifts.p + (2 * l + 1) * C};
  std::vector<int> tokens(T);
  for (int i = 0; i < T; ++i)
    tokens[i] = 33 + i;
  for (int round = 0; round < (trainer ? 2 : 1); ++round) {
    executor.reset(initial.p);
    executor.forward(m, tokens.data(), T, LogitsMode::None);
    auto backward = [&](bool compute_input, bool attention_shift) {
      if (trainer)
        trainer->zero_gradient();
      fill(ping, 7);
      fill(pong, 7);
      fill(shifts, 7);
      gradients[0].att_shift_grad = attention_shift ? shifts.p : nullptr;
      // Omit the flag on the full path to test the public default contract.
      auto result = compute_input
          ? model_backward_state_only(
                stream, layers, m.blocks.data(), executor.tapes().data(),
                gradients.data(), go.p, ping.p, pong.p, workspace.p, workspace.n)
          : model_backward_state_only(
                stream, layers, m.blocks.data(), executor.tapes().data(),
                gradients.data(), go.p, ping.p, pong.p, workspace.p, workspace.n,
                false);
      gpu_check(cudaStreamSynchronize(stream), "backward test sync");
      require(compute_input ? (result == ping.p || result == pong.p)
                            : result == nullptr, "model backward return contract");
      return result;
    };
    const bf16 *result = backward(true, true);
    written(result, go.n);
    written(shifts.p, shifts.n);
    const auto full_input = read(result, go.n);
    const auto full_state = read(ds.p, ds.n);
    const auto full_shift = read(shifts.p, shifts.n);
    const auto full_adapter = adapter_gradients(m);
    if (trainer && round == 1) {
      for (int target = 0; target < 3; ++target) {
        const auto &a = m.blocks[0].miss[target];
        for (bool first_factor : {false, true}) {
          const float *p = first_factor ? a.gradient_a : a.gradient;
          if (!p)
            continue;
          double norm = 0;
          for (float x : read(p, size_t(a.rank) * (first_factor ? a.in : a.out)))
            norm += double(x) * x;
          require(std::isfinite(norm) && norm > 0,
                  "layer-zero adapter gradient was not exercised");
        }
      }
    }
    if (residual_only) {
      // Independent exact oracle: both residual branches have zero output
      // weights, hence the whole block is the identity, dX=dY and dShift=0.
      equal(full_input, read(go.p, go.n), "identity block input gradient");
      equal(full_shift, std::vector<bf16>(shifts.n, to_bf16(0)),
            "identity block shift gradient");
    }
    // Skipping dX must still honor explicitly requested attention shifts.
    backward(false, true);
    equal(full_shift, read(shifts.p, shifts.n), "requested shift with no dX");
    equal(full_state, read(ds.p, ds.n), "state with no dX");
    equal(full_adapter, adapter_gradients(m), "adapter with no dX");
    // The executor's pruning path omits BOTH layer-zero dX and attention shift.
    backward(false, false);
    equal(full_state, read(ds.p, ds.n), "pruned state gradient");
    equal(full_adapter, adapter_gradients(m), "pruned adapter gradient");
    equal(std::vector<bf16>(full_shift.begin() + C, full_shift.end()),
          read(shifts.p + C, shifts.n - C), "pruned live shift gradients");
    equal(read(shifts.p, C), std::vector<bf16>(C, to_bf16(7)),
          "unrequested shift must not be touched");
    if (trainer && round == 0)
      trainer->update(.01f, round + 1);
  }

  // Exercise the block API default output contract directly as well.
  gradients[0].att_shift_grad = shifts.p;
  fill(first, 0);
  fill(pong, 7);
  fill(shifts, 7);
  block_backward_state_only(stream, go.p, pong.p, gradients[0],
                            executor.tapes()[0], m.blocks[0], workspace.p,
                            workspace.n, true);
  gpu_check(cudaStreamSynchronize(stream), "block test sync");
  written(pong.p, pong.n);
  written(shifts.p, C);
  if (residual_only) {
    DeviceBuffer<bf16> dlogits, expected, head_gradient;
    dlogits.resize(T * m.vocab, "head gradient test");
    expected.resize(go.n, "expected head derivative");
    head_gradient.resize(go.n, "head derivative");
    fill(dlogits, .125f);
    const bf16 *output = executor.tapes()[0].x; // identity block output = input
    linear_orig_backward_input_bf16(stream, T, C, m.vocab, dlogits.p,
                                    m.head_weight_orig, head_gradient.p);
    layer_norm_backward_input_bf16(stream, T, C, output, m.ln_out_weight,
                                   head_gradient.p, expected.p, 1e-5f);
    auto result = model_output_backward_state_only(
        stream, layers, 1, T, m.blocks.data(), executor.tapes().data(),
        gradients.data(), output, m.ln_out_weight, m.head_weight_orig, m.vocab,
        dlogits.p, ping.p, pong.p, workspace.p, workspace.n);
    gpu_check(cudaStreamSynchronize(stream), "head test sync");
    require(result == ping.p || result == pong.p, "head backward return contract");
    equal(read(expected.p, expected.n), read(result, expected.n),
          "head backward input derivative");
    require(model_output_backward_state_only(
        stream, layers, 1, T, m.blocks.data(), executor.tapes().data(),
        gradients.data(), output, m.ln_out_weight, m.head_weight_orig, m.vocab,
        dlogits.p, ping.p, pong.p, workspace.p, workspace.n, false) == nullptr,
        "head backward pruning contract");
    gpu_check(cudaStreamSynchronize(stream), "pruned head test sync");
  }
  if (layers > 1) {
    bool rejected = false;
    try {
      block_backward_state_only(stream, go.p, nullptr, gradients[1],
                                executor.tapes()[1], m.blocks[1], workspace.p,
                                workspace.n, false);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "nonzero layer accepted missing input gradient");
  }
  std::cout << "backward contract/pruning: layers=" << layers
            << " method=" << method << " identity=" << residual_only << '\n';
}

void check_tiled_head_and_grad_norm(cudaStream_t stream) {
  set_consistent_forward(true, true);
  Bf16ModelFixture fixture(2, false);
  auto view = fixture.view;
  Json::Value config;
  config["method"] = "nora";
  config["initialization"] = "nora_init";
  constexpr int chunk = 256, lanes = 8;
  rwkv_bf16_miss::Trainer trainer(view, 16, 16, "all", chunk * lanes, config,
                                  stream);
  Executor single(view, chunk, stream), packed(view, chunk, stream, lanes);
  const size_t states = size_t(view.layers) * view.heads * 4096;
  DeviceBuffer<float> initial;
  initial.resize(states, "tiled head test state");
  fill(initial, .01f);

  std::vector<std::vector<int>> batch_tokens(lanes, std::vector<int>(chunk));
  std::vector<std::vector<float>> batch_weights(lanes,
                                                std::vector<float>(chunk, 1.f));
  size_t normalizer = 0;
  for (int b = 0; b < lanes; ++b) {
    batch_weights[b][0] = 0.f;
    for (int t = 0; t < chunk; ++t) {
      batch_tokens[b][t] = 33 + ((b * 31 + t * 7) % (view.vocab - 33));
      if (t > 0)
        normalizer += 1;
    }
  }
  require(normalizer > 1024, "tiled head test must exceed kOutputTileRows");

  trainer.bind_tapes(single.tapes());
  trainer.zero_gradient();
  float expected_loss = 0.f;
  std::vector<float> expected_state(states, 0.f);
  for (int b = 0; b < lanes; ++b) {
    expected_loss += single.backward(view, batch_tokens[b], batch_weights[b],
                                     initial.p, normalizer);
    const auto g = read(single.state_gradient(), states);
    for (size_t j = 0; j < states; ++j)
      expected_state[j] += g[j];
  }
  const auto expected_adapter = adapter_gradients(view);

  trainer.bind_tapes(packed.tapes());
  trainer.zero_gradient();
  const float loss = packed.backward_packed(view, batch_tokens, batch_weights,
                                            initial.p, normalizer);
  require(std::abs(loss - expected_loss) <= 1e-5f * std::abs(expected_loss),
          "tiled cross-entropy loss mismatch");
  const auto got_adapter = adapter_gradients(view);
  double err = 0.0, norm = 0.0;
  for (size_t i = 0; i < expected_adapter.size(); ++i) {
    const double d = double(expected_adapter[i]) - got_adapter[i];
    err += d * d;
    norm += double(expected_adapter[i]) * expected_adapter[i];
  }
  require(norm > 0.0 && std::sqrt(err / norm) <= 1e-5,
          "tiled cross-entropy adapter gradient mismatch");

  // Exercise device-side gradient_norm_and_check_f32 finite & NaN paths.
  DeviceBuffer<float> gbuf;
  DeviceBuffer<double> d_norm_sq;
  DeviceBuffer<int> d_bad;
  gbuf.resize(5000, "grad norm test");
  d_norm_sq.resize(1, "grad norm accum");
  d_bad.resize(1, "grad bad flag");
  std::vector<float> host_g(gbuf.n);
  double ref_sq = 0.0;
  for (size_t i = 0; i < host_g.size(); ++i) {
    host_g[i] = 0.001f * float(int(i % 97) - 48);
    ref_sq += double(host_g[i]) * host_g[i];
  }
  gpu_check(cudaMemcpyAsync(gbuf.p, host_g.data(), host_g.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "upload grad norm test");
  gradient_norm_and_check_f32(stream, gbuf.p, gbuf.n, d_norm_sq.p, d_bad.p);
  gpu_check(cudaStreamSynchronize(stream), "sync grad norm");
  const double got_sq = read(d_norm_sq.p, 1)[0];
  const int got_bad = read(d_bad.p, 1)[0];
  if (got_bad != 0 || std::abs(got_sq - ref_sq) > 1e-9 * ref_sq) {
    std::cerr << "got_bad=" << got_bad << " got_sq=" << got_sq
              << " ref_sq=" << ref_sq << '\n';
    require(false, "device gradient norm mismatch");
  }
  host_g[3333] = std::numeric_limits<float>::quiet_NaN();
  gpu_check(cudaMemcpyAsync(gbuf.p, host_g.data(), host_g.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "upload nan grad test");
  gradient_norm_and_check_f32(stream, gbuf.p, gbuf.n, d_norm_sq.p, d_bad.p);
  gpu_check(cudaStreamSynchronize(stream), "sync nan grad norm");
  require(read(d_bad.p, 1)[0] != 0, "NaN gradient was not flagged on device");
  std::cout << "tiled head & device grad norm verified (supervised="
            << normalizer << ")\n";
}
} // namespace
int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
    return 77;
  cudaStream_t stream = nullptr;
  try {
    gpu_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "regression stream");
    for (int policy = 0; policy < 3; ++policy) {
      set_consistent_forward(policy != 2, policy == 1);
      check(1, "state", true, stream);
      for (int layers : {1, 2})
        for (const char *method : {"state", "miss", "nora"})
          check(layers, method, false, stream);
    }
    check_tiled_head_and_grad_norm(stream);
    gpu_check(cudaStreamDestroy(stream), "destroy regression stream");
  } catch (const std::exception &e) {
    if (stream)
      cudaStreamDestroy(stream);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
