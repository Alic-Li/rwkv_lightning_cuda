#include "rwkv/runtime/nora_gemm.cuh"
#include "rwkv/runtime/rwkv_nora.hpp"
#include <cmath>
#include <stdexcept>
namespace rwkv7_nora {
void forward(cudaStream_t stream, int rows, const rwkv7_miss::Linear &w,
             const half *x, half *y, float *s) {
  if (!w.a || !w.d || !x || !y || rows < 1 || rows > 65535 || w.in < 1 ||
      w.out < 1 || w.rank < 1 || w.rank > 1024 || !std::isfinite(w.scale))
    throw std::invalid_argument("invalid NoRA forward arguments");
  if (w.scale == 0)
    return;
  using namespace rwkv_nora_detail;
  if (s) {
    gemm<false, true, false>(stream, rows, w.rank, w.in, x, w.a, s);
    gemm<false, true, true>(stream, rows, w.out, w.rank, s, w.d, y, w.scale);
  } else {
    fused<<<dim3((w.out + 255) / 256, rows), 256, w.rank * sizeof(float),
            stream>>>(w.in, w.out, w.rank, w.scale, w.a, w.d, x, y);
  }
  auto e = cudaGetLastError();
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace rwkv7_nora
