// Copied from src/miss/kernels.cu; BF16 tensors, FP32 reduction/accumulation.
#include "miss.hpp"
#include <cmath>
#include <stdexcept>
namespace rwkv_bf16_miss {
namespace {
constexpr int threads = 256;
__global__ void reduce(int rows, int in, int rank, const bf16 *x, float *s) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= rows * rank)
    return;
  float sum = 0;
  for (int k = q % rank; k < in; k += rank)
    sum += to_float(x[(q / rank) * (size_t)in + k]);
  s[q] = sum;
}
// A CTA shares the reduced input across 256 output channels. Inference needs
// no global scratch, including decode. FP32 accumulation, one BF16 rounding.
__global__ void project(int in, int out, int rank, float scale, const bf16 *x,
                        float *saved, bool read_saved, const bf16 *d, bf16 *y) {
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
        sum += to_float(x[row * (size_t)in + k]);
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
          sum += to_float(x[row * (size_t)in + k]);
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
    delta = fmaf(s[j], to_float(d[o * (size_t)rank + j]), delta);
  const size_t i = row * (size_t)out + o;
  y[i] = to_bf16(fmaf(scale, delta, to_float(y[i])));
}
// Short chunks need only one thread per parameter, avoiding thousands of
// mostly idle CTAs when rows is smaller than a warp.
__global__ void grad_d_short(int rows, int out, int rank, float scale,
                             const float *s, const bf16 *g, float *dd) {
  const size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (q >= size_t(out) * rank)
    return;
  float sum = 0;
  for (int t = 0; t < rows; ++t)
    sum = fmaf(to_float(g[t * (size_t)out + q / rank]),
               s[t * (size_t)rank + q % rank], sum);
  dd[q] += scale * sum;
}

// One CTA owns a D element; deterministic reduction, no cross-CTA atomics.
__global__ void grad_d(int rows, int out, int rank, float scale, const float *s,
                       const bf16 *g, float *dd) {
  const int q = blockIdx.x;
  float sum = 0;
  for (int t = threadIdx.x; t < rows; t += blockDim.x)
    sum = fmaf(to_float(g[t * (size_t)out + q / rank]),
               s[t * (size_t)rank + q % rank], sum);
  __shared__ float partial[threads];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int n = threads / 2; n; n /= 2) {
    if (threadIdx.x < n)
      partial[threadIdx.x] += partial[threadIdx.x + n];
    __syncthreads();
  }
  if (!threadIdx.x)
    dd[q] += scale * partial[0];
}
// Shared tiles reuse G across 16 rank slots and S across 16 outputs.
// This avoids one strided CTA reduction per D element for longer chunks.
__global__ void grad_d_tiled(int rows, int out, int rank, float scale,
                             const float *s, const bf16 *g, float *dd) {
  constexpr int tile = 16;
  const int lane = threadIdx.x;
  const int o = blockIdx.x * tile + lane / tile;
  const int j = blockIdx.y * tile + lane % tile;
  __shared__ float sg[tile][tile], ss[tile][tile];
  float sum = 0;
  for (int t0 = 0; t0 < rows; t0 += tile) {
    const int t = t0 + lane / tile;
    const int go = blockIdx.x * tile + lane % tile;
    const int sj = blockIdx.y * tile + lane % tile;
    sg[lane / tile][lane % tile] =
        (t < rows && go < out) ? to_float(g[t * (size_t)out + go]) : 0;
    ss[lane / tile][lane % tile] =
        (t < rows && sj < rank) ? s[t * (size_t)rank + sj] : 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < tile; ++k)
      sum = fmaf(sg[k][lane / tile], ss[k][lane % tile], sum);
    __syncthreads();
  }
  if (o < out && j < rank)
    dd[o * (size_t)rank + j] += scale * sum;
}

// Computes dS once per rank slot and broadcasts it directly to dX.
__global__ void grad_x(int in, int out, int rank, float scale, const bf16 *d,
                       const bf16 *g, bf16 *dx) {
  const int row = blockIdx.x, j = blockIdx.y;
  float sum = 0;
  for (int o = threadIdx.x; o < out; o += blockDim.x)
    sum = fmaf(to_float(g[row * (size_t)out + o]),
               to_float(d[o * (size_t)rank + j]), sum);
  __shared__ float partial[threads];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int n = threads / 2; n; n /= 2) {
    if (threadIdx.x < n)
      partial[threadIdx.x] += partial[threadIdx.x + n];
    __syncthreads();
  }
  for (int k = j + threadIdx.x * rank; k < in; k += threads * rank) {
    const size_t i = row * (size_t)in + k;
    dx[i] = to_bf16(fmaf(scale, partial[0], to_float(dx[i])));
  }
}
__global__ void cast(const float *x, bf16 *y, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = to_bf16(x[i]);
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
void forward(cudaStream_t stream, int rows, const Linear &w, const bf16 *x,
             bf16 *y, float *s) {
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
void backward(cudaStream_t stream, int rows, const Linear &w, const float *s,
              const bf16 *g, bf16 *dx) {
  if (!w.d)
    return;
  validate(rows, w);
  if (!g || !dx || (w.gradient && !s))
    throw std::invalid_argument("null MiSS backward input");
  if (w.gradient) {
    if (rows < 32)
      grad_d_short<<<(w.out * w.rank + threads - 1) / threads, threads, 0,
                     stream>>>(rows, w.out, w.rank, w.scale, s, g, w.gradient);
    else if (w.out >= 256)
      grad_d_tiled<<<dim3((w.out + 15) / 16, (w.rank + 15) / 16), threads, 0,
                     stream>>>(rows, w.out, w.rank, w.scale, s, g, w.gradient);
    else
      grad_d<<<w.out * w.rank, threads, 0, stream>>>(rows, w.out, w.rank,
                                                     w.scale, s, g, w.gradient);
  }
  grad_x<<<dim3(rows, w.rank), threads, 0, stream>>>(w.in, w.out, w.rank,
                                                     w.scale, w.d, g, dx);
  checked();
}
void cast_master(cudaStream_t stream, const float *x, bf16 *y, size_t n) {
  cast<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, y, n);
  checked();
}
} // namespace rwkv_bf16_miss
