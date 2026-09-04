#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

int main() {
  constexpr std::size_t bytes = 512ull << 20;
  void* source = nullptr;
  void* destination = nullptr;
  if (cudaMalloc(&source, bytes) != cudaSuccess || cudaMalloc(&destination, bytes) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }
  for (int i = 0; i < 3; ++i) cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float samples[10]{};
  for (int i = 0; i < 10; ++i) {
    cudaEventRecord(start);
    cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&samples[i], start, stop);
  }
  std::sort(samples, samples + 10);
  const double median_ms = 0.5 * (samples[4] + samples[5]);
  const double bandwidth = (2.0 * bytes) / (median_ms * 1.0e-3) / 1.0e9;
  std::printf("d2d_bandwidth_gbps=%.3f median_ms=%.3f\n", bandwidth, median_ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(source);
  cudaFree(destination);
  return 0;
}
