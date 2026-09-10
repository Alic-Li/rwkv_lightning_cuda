#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rwkv_w4a16.cuh"
#include "rwkv_w8a16.cuh"

namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <class T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(std::size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
template <class F>
float measure(cudaStream_t stream, F fn, int iters) {
  for (int i = 0; i < 10; ++i) fn();
  check(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  check(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) fn();
  check(cudaEventRecord(stop, stream));
  check(cudaEventSynchronize(stop));
  float ms;
  check(cudaEventElapsedTime(&ms, start, stop));
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  return ms * 1000 / iters;
}
template <class F>
float measure_concurrent(cudaStream_t timing_stream, F fn, int iters) {
  cudaStream_t streams[4];
  cudaEvent_t done[4], start, stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  for (int s = 0; s < 4; ++s) {
    check(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
    for (int i = 0; i < 10; ++i) fn(s, streams[s]);
  }
  for (auto stream : streams) check(cudaStreamSynchronize(stream));
  check(cudaEventRecord(start, timing_stream));
  for (auto stream : streams) check(cudaStreamWaitEvent(stream, start));
  for (int i = 0; i < iters; ++i)
    for (int s = 0; s < 4; ++s) fn(s, streams[s]);
  for (int s = 0; s < 4; ++s) {
    check(cudaEventRecord(done[s], streams[s]));
    check(cudaStreamWaitEvent(timing_stream, done[s]));
  }
  check(cudaEventRecord(stop, timing_stream));
  check(cudaEventSynchronize(stop));
  float ms;
  check(cudaEventElapsedTime(&ms, start, stop));
  for (int s = 0; s < 4; ++s) {
    check(cudaEventDestroy(done[s]));
    check(cudaStreamDestroy(streams[s]));
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  return ms * 1000 / (iters * 4);
}
void run(int M, int K, int N, int G, int iters, cudaStream_t stream) {
  Buffer<half> w(std::size_t(N) * K), x(std::size_t(M) * K), s4(rwkv7_w4a16_scale_count(N, K, G)), s8(N);
  Buffer<half> y4(std::size_t(M) * N), y8(std::size_t(M) * N);
  Buffer<unsigned char> q4(rwkv7_w4a16_weight_bytes(N, K));
  Buffer<std::int8_t> q8(std::size_t(N) * K), packed8(std::size_t(N) * K);
  const auto wsbytes = rwkv7_w4a16_workspace_bytes(M, N, 32);
  Buffer<float> ws(wsbytes / 4);
  std::vector<half> hostw(std::size_t(N) * K), hostx(std::size_t(M) * K), hosts(N, __float2half(1.0f / 127));
  std::vector<std::int8_t> hostq(std::size_t(N) * K);
  for (std::size_t i = 0; i < hostw.size(); ++i) {
    hostq[i] = int((i * 37 + i / K * 13) % 255) - 127;
    hostw[i] = __float2half(float(hostq[i]) * __half2float(hosts[0]));
  }
  for (std::size_t i = 0; i < hostx.size(); ++i) hostx[i] = __float2half(float(int(i % 31) - 15) / 16);
  check(cudaMemcpy(w.p, hostw.data(), hostw.size() * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(x.p, hostx.data(), hostx.size() * 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(q8.p, hostq.data(), hostq.size(), cudaMemcpyHostToDevice));
  check(cudaMemcpy(s8.p, hosts.data(), hosts.size() * 2, cudaMemcpyHostToDevice));
  rwkv7_w4a16_quantize_launch(stream, w.p, q4.p, s4.p, N, K, G);
  rwkv7_v4_i8_pack_launch(stream, q8.p, packed8.p, N, K);
  check(cudaStreamSynchronize(stream));
  const float t8 = measure(
      stream,
      [&] {
        rwkv7_w8a16_linear_launch(stream, M, K, N, x.p, packed8.p, s8.p, W8BLayout::PackedNK, y8.p, ws.p,
                                  wsbytes);
      },
      iters);
  const float t4 = measure(
      stream, [&] { rwkv7_w4a16_linear_launch(stream, M, K, N, x.p, q4.p, s4.p, y4.p, G, ws.p, wsbytes); },
      iters);
  std::printf("%d,%d,%d,%d,%.3f,%.3f,%.3f", M, K, N, G, t8, t4, t8 / t4);
  for (int split : {1, 2, 4, 8, 16, 32}) {
    const float t = measure(
        stream,
        [&] { rwkv7_w4a16_linear_launch(stream, M, K, N, x.p, q4.p, s4.p, y4.p, G, ws.p, wsbytes, split); },
        iters);
    std::printf(",%.3f", t);
  }
  std::vector<std::unique_ptr<Buffer<half>>> outputs;
  std::vector<std::unique_ptr<Buffer<float>>> workspaces;
  for (int s = 0; s < 4; ++s) {
    outputs.emplace_back(new Buffer<half>(std::size_t(M) * N));
    workspaces.emplace_back(new Buffer<float>(wsbytes / 4));
  }
  const float concurrent = measure_concurrent(
      stream,
      [&](int s, cudaStream_t st) {
        rwkv7_w4a16_linear_launch(st, M, K, N, x.p, q4.p, s4.p, outputs[s]->p, G, workspaces[s]->p, wsbytes);
      },
      iters);
  std::printf(",%.3f\n", concurrent);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const int K = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int N = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 100;
    if (K < 4096 || K % 64 || N <= 0 || N % 64 || iters <= 0)
      throw std::invalid_argument(
          "usage: rwkv_w4a16_bench [K>=4096 multiple of 64] [N multiple of 64] [iterations>0]");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::cerr << prop.name
              << "; CUDA event latency, warm cache, shared W8/W4 input weights; quantization excluded\n";
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::puts(
        "M,K,N,G,W8_us,W4_auto_us,speedup,W4_s1_us,W4_s2_us,W4_s4_us,W4_s8_us,W4_s16_us,W4_s32_us,W4_"
        "4streams_us_per_call");
    for (int G : {128, 32})
      for (int M : {1, 2, 4, 8, 16, 32, 64, 128}) run(M, K, N, G, iters, stream);
    check(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
