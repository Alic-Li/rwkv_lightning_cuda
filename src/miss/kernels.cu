#include "rwkv/runtime/rwkv_miss.hpp"
#include "rwkv/runtime/rwkv_nora.hpp"
#include <cmath>
#include <stdexcept>
namespace rwkv7_miss {
namespace {
constexpr int threads = 256;
__global__ void reduce(int rows, int in, int rank, const half *x, float *s) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= rows * rank)
    return;
  float sum = 0;
  for (int k = q % rank; k < in; k += rank)
    sum += __half2float(x[(q / rank) * (size_t)in + k]);
  s[q] = sum;
}
// A CTA shares the reduced input across 256 output channels. Inference needs
// no global scratch, including decode. FP32 accumulation, one FP16 rounding.
__global__ void project(int in, int out, int rank, float scale, const half *x,
                        float *saved, bool read_saved, const half *d, half *y) {
  extern __shared__ float s[];
  const int row = blockIdx.y;
  if (!read_saved && rank <= threads) {
    // Distribute the input reduction across the entire CTA, not just rank
    // lanes. This matters for small-rank decode with wide inputs.
    __shared__ float partial[threads];
    const int groups = threads / rank;
    const int j = threadIdx.x % rank, group = threadIdx.x / rank;
    float sum = 0;
    if (group < groups)
      for (int k = j + group * rank; k < in; k += groups * rank)
        sum += __half2float(x[row * (size_t)in + k]);
    partial[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < rank) {
      sum = 0;
      for (int g = 0; g < groups; ++g)
        sum += partial[g * rank + threadIdx.x];
      s[threadIdx.x] = sum;
      if (saved && blockIdx.x == 0)
        saved[row * (size_t)rank + threadIdx.x] = sum;
    }
  } else {
    for (int j = threadIdx.x; j < rank; j += blockDim.x) {
      float sum = 0;
      if (saved && read_saved)
        sum = saved[row * (size_t)rank + j];
      else
        for (int k = j; k < in; k += rank)
          sum += __half2float(x[row * (size_t)in + k]);
      s[j] = sum;
      if (saved && !read_saved && blockIdx.x == 0)
        saved[row * (size_t)rank + j] = sum;
    }
  }
  __syncthreads();
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out)
    return;
  float delta = 0;
  for (int j = 0; j < rank; ++j)
    delta = fmaf(s[j], __half2float(d[o * (size_t)rank + j]), delta);
  const size_t i = row * (size_t)out + o;
  y[i] = __float2half_rn(fmaf(scale, delta, __half2float(y[i])));
}
void validate(int rows, const Linear &w) {
  if (rows <= 0 || rows > 65535 || w.in <= 0 || w.out <= 0 || w.rank <= 0 ||
      w.rank > 1024 || !std::isfinite(w.scale))
    throw std::invalid_argument("invalid MiSS shape/scale (rank <= 1024)");
}
void checked() {
  auto e = cudaGetLastError();
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace
void forward(cudaStream_t stream, int rows, const Linear &w, const half *x,
             half *y, float *s) {
  if (w.a)
    return rwkv7_nora::forward(stream, rows, w, x, y, s);
  if (!w.d)
    return;
  validate(rows, w);
  if (!x || !y)
    throw std::invalid_argument("null MiSS forward input");
  const bool separate = s && w.out > threads;
  if (separate)
    reduce<<<(rows * w.rank + threads - 1) / threads, threads, 0, stream>>>(
        rows, w.in, w.rank, x, s);
  project<<<dim3((w.out + threads - 1) / threads, rows), threads,
            w.rank * sizeof(float), stream>>>(w.in, w.out, w.rank, w.scale, x,
                                              s, separate, w.d, y);
  checked();
}
} // namespace rwkv7_miss
