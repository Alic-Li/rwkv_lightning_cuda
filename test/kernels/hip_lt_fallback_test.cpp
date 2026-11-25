#include "rwkv7_fast_v4_kernels.hip.hpp"
#include "test_common.hpp"
#include <hipblaslt/hipblaslt.h>

namespace {
bool return_candidate;
int matmul_calls;
}

extern "C" hipblasStatus_t __wrap_hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
    hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatmulPreference_t, int,
    hipblasLtMatmulHeuristicResult_t* candidates, int* count) {
  *count = return_candidate ? 1 : 0;
  if (return_candidate) candidates[0] = {};
  return HIPBLAS_STATUS_SUCCESS;
}

extern "C" hipblasStatus_t __wrap_hipblasLtMatmul(
    hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void*, const void*, hipblasLtMatrixLayout_t,
    const void*, hipblasLtMatrixLayout_t, const void*, const void*, hipblasLtMatrixLayout_t,
    void*, hipblasLtMatrixLayout_t, const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t) {
  ++matmul_calls;
  return HIPBLAS_STATUS_INTERNAL_ERROR;
}

int main() {
  if (!rwkv_test::cuda_device_available()) return 77;
  constexpr int M = 5, K = 16, N = 32;
  std::vector<half> input(M * K), weights(N * K);
  for (int i = 0; i < M * K; ++i) input[i] = __float2half(float(i % 7 - 3));
  for (int i = 0; i < N * K; ++i) weights[i] = __float2half(float(i % 5 - 2));
  rwkv7_fast_v4::DeviceBuffer<half> x, w, y;
  rwkv_test::copy_host_to_device(input, x, "alloc x", "copy x");
  rwkv_test::copy_host_to_device(weights, w, "alloc w", "copy w");
  y.resize(M * N, "alloc y");
  hipStream_t stream;
  rwkv_test::require_cuda(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
  for (bool candidate : {false, true}) {
    return_candidate = candidate;
    rwkv_test::require_cuda(hipMemsetAsync(y.p, 0xff, M * N * sizeof(half), stream), "poison y");
    rwkv7_v3a_linear_f16_orig_lt_cfg_launch(stream, M, K, N, x.p, w.p, nullptr, 0, 0, y.p);
    rwkv_test::require_cuda(hipStreamSynchronize(stream), "complete GEMM");
    const auto result = rwkv_test::copy_device_buffer(y, "read y");
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float expected = 0;
        for (int k = 0; k < K; ++k)
          expected += __half2float(input[m * K + k]) * __half2float(weights[n * K + k]);
        TEST_EQ(__half2float(result[m * N + n]), expected);
      }
    }
  }
  rwkv_test::require_cuda(hipStreamDestroy(stream), "destroy stream");
  TEST_EQ(matmul_calls, 1);
}
