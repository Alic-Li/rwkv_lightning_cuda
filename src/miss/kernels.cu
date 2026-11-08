#include "rwkv/runtime/rwkv_miss.hpp"
#include "rwkv/runtime/rwkv_nora.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
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
template <bool Relu2> __device__ __forceinline__ float load(half v) {
  float f = __half2float(v);
  if (Relu2)
    f = f > 0.0f ? f * f : 0.0f;
  return f;
}
// One CTA owns 256 outputs of one row, like project/fused. Base rows and rows
// whose adapter lacks this target exit before any barrier (CTA-uniform).
// Packed decode is split in two so the latent of each row is computed once,
// not once per 256-output tile (a NoRA FFN key would otherwise re-read A for
// all 32 tiles of every row). Stage 1 splits the input into fixed chunks, so
// each row's result is independent of its batch neighbours and deterministic.
// Decode launches only a handful of CTAs, so both stages are latency bound:
// every global load a thread needs is issued before its first use.
constexpr int latent_chunk = 2 * threads;
template <bool Relu2>
__global__ void latent_kernel(int in, int chunks, int max_rank,
                              const int *row_slot, const Linear *table,
                              const half *x, float *latent) {
  __shared__ float xs[latent_chunk];
  // Padded rows keep the column folds below bank-conflict free.
  __shared__ float red[16][threads + 16];
  float *partial = &red[0][0];
  const int row = blockIdx.y, chunk = blockIdx.x;
  const int slot = row_slot ? row_slot[row] : 0;
  if (slot < 0)
    return;
  const Linear w = table[slot];
  if (!w.d || w.scale == 0.0f)
    return;
  const int rank = w.rank;
  const int k0 = chunk * latent_chunk, n = min(in - k0, latent_chunk);
  const half *xr = x + row * (size_t)in + k0;
  float *out = latent + (size_t(row) * chunks + chunk) * max_rank;
  // NoRA: thread t owns inputs 2t and 2t+1 for sixteen coordinates. The first
  // tile of A does not depend on x, so its loads are in flight while the
  // chunk is staged below (one DRAM round trip instead of two).
  const int k = 2 * threadIdx.x;
  const bool nora = w.a || w.at;
  const bool pair = w.a && k + 1 < n && ((in | k0) & 1) == 0 &&
                    (reinterpret_cast<uintptr_t>(w.a) & 3) == 0;
  float2 av[16];
  // A^T rows are contiguous: two inputs x sixteen coordinates are four
  // 16-byte loads per thread and the CTA reads one contiguous span.
  const bool at_vec = w.at && (rank & 7) == 0 &&
                      (reinterpret_cast<uintptr_t>(w.at) & 15) == 0;
  const auto load_a = [&](int start) {
    const int count = min(16, rank - start);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      av[j] = make_float2(0.0f, 0.0f);
    if (at_vec && count == 16) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        if (k + e >= n)
          continue;
        const uint4 *p = reinterpret_cast<const uint4 *>(
            w.at + size_t(k0 + k + e) * rank + start);
        const uint4 v[2] = {p[0], p[1]};
        const __half2 *h = reinterpret_cast<const __half2 *>(v);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const float2 f = __half22float2(h[q]);
          (e ? av[2 * q].y : av[2 * q].x) = f.x;
          (e ? av[2 * q + 1].y : av[2 * q + 1].x) = f.y;
        }
      }
      return;
    }
    if (w.at) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        if (k + e >= n)
          continue;
        const half *row = w.at + size_t(k0 + k + e) * rank + start;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < count)
            (e ? av[j].y : av[j].x) = __half2float(row[j]);
      }
      return;
    }
    const half *a = w.a + size_t(start) * in + k0 + k;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (j < count) {
        if (pair)
          av[j] = __half22float2(
              *reinterpret_cast<const __half2 *>(a + size_t(j) * in));
        else {
          if (k < n)
            av[j].x = __half2float(a[size_t(j) * in]);
          if (k + 1 < n)
            av[j].y = __half2float(a[size_t(j) * in + 1]);
        }
      }
    }
  };
  if (nora)
    load_a(0);
  // One coalesced pass stages the chunk (relu^2 applied once per element).
  for (int i = threadIdx.x; i < n; i += threads)
    xs[i] = load<Relu2>(xr[i]);
  __syncthreads();
  if (nora) {
    const float x0 = k < n ? xs[k] : 0.0f, x1 = k + 1 < n ? xs[k + 1] : 0.0f;
    for (int start = 0; start < rank; start += 16) {
      const int count = min(16, rank - start);
      if (start)
        load_a(start);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        red[j][threadIdx.x] = fmaf(x1, av[j].y, x0 * av[j].x);
      __syncthreads();
      const int j = threadIdx.x / 16, part = threadIdx.x % 16;
      float sum = 0.0f;
#pragma unroll
      for (int i = part; i < threads; i += 16)
        sum += red[j][i];
      __syncthreads();
      partial[threadIdx.x] = sum;
      __syncthreads();
      if (threadIdx.x < count) {
        sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          sum += partial[threadIdx.x * 16 + i];
        out[start + threadIdx.x] = sum;
      }
      __syncthreads();
    }
  } else {
    // MiSS modulo-rank sum restricted to this chunk, read from shared.
    for (int j = threadIdx.x; j < rank; j += threads) {
      float sum = 0.0f;
      for (int k = (j - k0 % rank + rank) % rank; k < n; k += rank)
        sum += xs[k];
      out[j] = sum;
    }
  }
}
// Stage 2: y += scale * D s, one CTA per 256 outputs of one row.
__global__ void expand_kernel(int out, int chunks, int max_rank,
                              const int *row_slot, const Linear *table,
                              const float *latent, half *y) {
  extern __shared__ float s[];
  const int row = blockIdx.y;
  const int slot = row_slot ? row_slot[row] : 0;
  if (slot < 0)
    return;
  const Linear w = table[slot];
  if (!w.d || w.scale == 0.0f)
    return;
  const int rank = w.rank;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t i = row * (size_t)out + o;
  // Issue the output and the first D vectors before waiting on the latent.
  const float base = o < out ? __half2float(y[i]) : 0.0f;
  const float *lat = latent + size_t(row) * chunks * max_rank;
  for (int j = threadIdx.x; j < rank; j += threads) {
    float sum = 0.0f;
#pragma unroll 4
    for (int c = 0; c < chunks; ++c)
      sum += lat[size_t(c) * max_rank + j];
    s[j] = sum;
  }
  __syncthreads();
  if (o >= out)
    return;
  const half *d = w.d + o * (size_t)rank;
  float delta = 0;
  if ((rank & 7) == 0 && (reinterpret_cast<uintptr_t>(d) & 15) == 0) {
    for (int j = 0; j < rank; j += 16) {
      uint4 v[2];
      v[0] = *reinterpret_cast<const uint4 *>(d + j);
      v[1] = j + 8 < rank ? *reinterpret_cast<const uint4 *>(d + j + 8)
                          : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const __half2 *h = reinterpret_cast<const __half2 *>(&v[q]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(h[e]);
          const int jj = j + q * 8 + 2 * e;
          if (jj < rank) {
            delta = fmaf(s[jj], f.x, delta);
            delta = fmaf(s[jj + 1], f.y, delta);
          }
        }
      }
    }
  } else {
    for (int j = 0; j < rank; ++j)
      delta = fmaf(s[j], __half2float(d[j]), delta);
  }
  y[i] = __float2half_rn(fmaf(w.scale, delta, base));
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
std::size_t forward_rows_scratch(int rows, int in, int max_rank) {
  return size_t(rows) * ((in + latent_chunk - 1) / latent_chunk) * max_rank;
}
void forward_rows(cudaStream_t stream, int rows, int in, int out, int max_rank,
                  const int *row_slot, const Linear *table, const half *x,
                  half *y, float *latent, bool relu2_input) {
  if (rows <= 0)
    return;
  if (in <= 0 || out <= 0 || max_rank <= 0 || max_rank > 1024 || !table ||
      !x || !y || !latent)
    throw std::invalid_argument("invalid packed adapter forward arguments");
  const int chunks = (in + latent_chunk - 1) / latent_chunk;
  // grid.y is limited to 65535; large prefill batches run in row slices.
  constexpr int max_grid_rows = 65535;
  for (int r0 = 0; r0 < rows; r0 += max_grid_rows) {
    const int n = std::min(max_grid_rows, rows - r0);
    const int *slots = row_slot ? row_slot + r0 : nullptr;
    const half *xs = x + size_t(r0) * in;
    half *ys = y + size_t(r0) * out;
    float *lat = latent + size_t(r0) * chunks * max_rank;
    if (relu2_input)
      latent_kernel<true><<<dim3(chunks, n), threads, 0, stream>>>(
          in, chunks, max_rank, slots, table, xs, lat);
    else
      latent_kernel<false><<<dim3(chunks, n), threads, 0, stream>>>(
          in, chunks, max_rank, slots, table, xs, lat);
    expand_kernel<<<dim3((out + threads - 1) / threads, n), threads,
                    size_t(max_rank) * sizeof(float), stream>>>(
        out, chunks, max_rank, slots, table, lat, ys);
  }
  checked();
}
void forward(cudaStream_t stream, int rows, const Linear &w, const half *x,
             half *y, float *s) {
  if (w.a)
    return rwkv7_nora::forward(stream, rows, w, x, y, s);
  if (w.at)
    throw std::invalid_argument("transposed-only NoRA requires forward_rows");
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
