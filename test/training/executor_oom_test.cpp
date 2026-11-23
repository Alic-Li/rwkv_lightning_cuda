#include "bf16_model_fixture.hpp"
#include "rwkv/bf16_training/executor.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include <iostream>

using namespace rwkv_bf16_training;
// Fail one selected allocation inside the static training library. Ask CUDA
// for an impossible allocation so its per-thread error state is exercised too.
static int until_failure = 0;
extern "C" cudaError_t __real_cudaMalloc(void **, size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void **p, size_t bytes) {
  if (until_failure && !--until_failure)
    return __real_cudaMalloc(p, size_t(1) << 54);
  return __real_cudaMalloc(p, bytes);
}
template <class F> void fail_after(int count, F f, const char *expected) {
  until_failure = count;
  try {
    f();
  } catch (const std::runtime_error &e) {
    std::cout << e.what() << std::endl;
    if (!until_failure &&
        std::string(e.what()).find(expected) != std::string::npos &&
        cudaPeekAtLastError() == cudaSuccess)
      return;
    throw;
  }
  throw std::runtime_error("missing expected OOM");
}
int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
    return 77;
  Bf16ModelFixture fixture(3);
  auto &m = fixture.view;
  set_consistent_forward(true, true);
  Executor e(m, 8, nullptr);
  DeviceBuffer<float> state;
  state.resize(3 * 4096, "initial state");
  state.zero("initial state");
  const std::vector<int> ids{1, 2, 3, 4, 5};
  const std::vector<float> weights{0, 1, 1, 1, 1};
  auto train = [&] { return e.backward(m, ids, weights, state.p, 4); };
  const float reference = train();
  for (int allocation : {1, 3, 6}) {
    fail_after(allocation, [&] { e.reserve_sequence(64); }, "window ");
    if (train() != reference)
      throw std::runtime_error("window recovery loss");
  }
  e.reset(state.p);
  fail_after(4, [&] { e.forward(m, ids.data(), ids.size()); }, "tune tape");
  e.reset(state.p);
  e.forward(m, ids.data(), ids.size());
  const auto logits = e.last_logits();
  Executor fresh(m, 8, nullptr);
  fresh.reset(state.p);
  fresh.forward(m, ids.data(), ids.size());
  if (logits != fresh.last_logits())
    throw std::runtime_error("partial tape recovery");
  std::cout << "window and full-tape recovery passed" << std::endl;
}
