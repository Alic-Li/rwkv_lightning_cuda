#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "rwkv_w8a16.cuh"

namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void check_blas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << what << " failed with status " << static_cast<int>(status) << "\n";
    std::exit(1);
  }
}

template <typename T>
struct Buffer {
  T* p = nullptr;
  std::size_t n = 0;
  void resize(std::size_t count) {
    n = count;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
  }
  Buffer() = default;
  Buffer(Buffer&& other) noexcept : p(other.p), n(other.n) {
    other.p = nullptr;
    other.n = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (p) cudaFree(p);
      p = other.p;
      n = other.n;
      other.p = nullptr;
      other.n = 0;
    }
    return *this;
  }
  ~Buffer() {
    if (p) cudaFree(p);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

struct Matrix {
  int K;
  int N;
  W8BLayout layout;
  Buffer<std::int8_t> q;
  Buffer<std::int8_t> q_packed;
  Buffer<half> f16;
  Buffer<half> scale;
  Matrix(int k, int n, W8BLayout matrix_layout) : K(k), N(n), layout(matrix_layout) {}
  Matrix(Matrix&&) noexcept = default;
  Matrix& operator=(Matrix&&) noexcept = default;
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
};

std::uint16_t half_bits(float value) {
  const half h = __float2half(value);
  return *reinterpret_cast<const std::uint16_t*>(&h);
}

Matrix make_matrix(int K, int N, W8BLayout layout, int seed) {
  Matrix matrix{K, N, layout};
  std::vector<std::int8_t> q_nk(static_cast<std::size_t>(N) * K);
  std::vector<std::int8_t> q_storage(q_nk.size());
  std::vector<half> scales(N);
  std::vector<half> f16_nk(q_nk.size());
  for (int n = 0; n < N; ++n) {
    const float scale = 0.002f + 0.0008f * static_cast<float>((n * 11 + seed) % 17);
    const std::uint16_t scale_bits = half_bits(scale);
    scales[n] = *reinterpret_cast<const half*>(&scale_bits);
    for (int k = 0; k < K; ++k) {
      const int qv = (n * 37 + k * 17 + seed * 13) % 255 - 127;
      q_nk[static_cast<std::size_t>(n) * K + k] = static_cast<std::int8_t>(qv);
      f16_nk[static_cast<std::size_t>(n) * K + k] = __int2half_rn(qv) * scales[n];
    }
  }
  if (layout == W8BLayout::NK) {
    q_storage = q_nk;
  } else {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) q_storage[static_cast<std::size_t>(k) * N + n] = q_nk[static_cast<std::size_t>(n) * K + k];
    }
  }
  std::vector<half> f16;
  if (layout == W8BLayout::NK) {
    f16 = std::move(f16_nk);
  } else {
    f16.resize(q_nk.size());
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) f16[static_cast<std::size_t>(k) * N + n] = f16_nk[static_cast<std::size_t>(n) * K + k];
    }
  }
  matrix.q.resize(q_storage.size());
  matrix.f16.resize(f16.size());
  matrix.scale.resize(scales.size());
  check(cudaMemcpy(matrix.q.p, q_storage.data(), q_storage.size() * sizeof(std::int8_t), cudaMemcpyHostToDevice), "copy q");
  if (layout == W8BLayout::NK) {
    matrix.q_packed.resize(q_nk.size());
    rwkv7_v4_i8_pack_launch(nullptr, matrix.q.p, matrix.q_packed.p, N, K);
    check(cudaGetLastError(), "launch pack q");
    check(cudaDeviceSynchronize(), "sync pack q");
  }
  check(cudaMemcpy(matrix.f16.p, f16.data(), f16.size() * sizeof(half), cudaMemcpyHostToDevice), "copy f16");
  check(cudaMemcpy(matrix.scale.p, scales.data(), scales.size() * sizeof(half), cudaMemcpyHostToDevice), "copy scale");
  return matrix;
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return static_cast<double>(ms);
}

double bench_w8(
    cudaStream_t stream,
    const std::vector<Matrix*>& matrices,
    int M,
    Buffer<half>& x,
    Buffer<half>& y,
    Buffer<unsigned char>& workspace,
    int force_split_k) {
  for (int warmup = 0; warmup < 3; ++warmup) {
    for (Matrix* matrix : matrices) {
      rwkv7_w8a16_linear_launch(stream, M, matrix->K, matrix->N, x.p,
                                matrix->layout == W8BLayout::NK ? matrix->q_packed.p : matrix->q.p,
                                matrix->scale.p,
                                matrix->layout == W8BLayout::NK ? W8BLayout::PackedNK : matrix->layout,
                                y.p, workspace.p, workspace.n, force_split_k);
    }
  }
  check(cudaStreamSynchronize(stream), "warmup w8");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event start");
  check(cudaEventCreate(&stop), "event stop");
  check(cudaEventRecord(start, stream), "record start");
  for (int rep = 0; rep < 10; ++rep) {
    for (Matrix* matrix : matrices) {
      rwkv7_w8a16_linear_launch(stream, M, matrix->K, matrix->N, x.p,
                                matrix->layout == W8BLayout::NK ? matrix->q_packed.p : matrix->q.p,
                                matrix->scale.p,
                                matrix->layout == W8BLayout::NK ? W8BLayout::PackedNK : matrix->layout,
                                y.p, workspace.p, workspace.n, force_split_k);
    }
  }
  check(cudaEventRecord(stop, stream), "record stop");
  check(cudaEventSynchronize(stop), "sync stop");
  const double result = elapsed_ms(start, stop) / 10.0;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return result;
}

double bench_f16(
    cudaStream_t stream,
    cublasHandle_t handle,
    const std::vector<Matrix*>& matrices,
    int M,
    Buffer<half>& x,
    Buffer<half>& y) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int warmup = 0; warmup < 2; ++warmup) {
    for (Matrix* matrix : matrices) {
      const cublasOperation_t trans_a = matrix->layout == W8BLayout::NK ? CUBLAS_OP_T : CUBLAS_OP_N;
      const int lda = matrix->layout == W8BLayout::NK ? matrix->K : matrix->N;
      check_blas(cublasGemmEx(handle, trans_a, CUBLAS_OP_N, matrix->N, M, matrix->K,
                              &alpha, matrix->f16.p, CUDA_R_16F, lda,
                              x.p, CUDA_R_16F, matrix->K, &beta, y.p, CUDA_R_16F, matrix->N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    }
  }
  check(cudaStreamSynchronize(stream), "warmup f16");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event start");
  check(cudaEventCreate(&stop), "event stop");
  check(cudaEventRecord(start, stream), "record start");
  for (int rep = 0; rep < 5; ++rep) {
    for (Matrix* matrix : matrices) {
      const cublasOperation_t trans_a = matrix->layout == W8BLayout::NK ? CUBLAS_OP_T : CUBLAS_OP_N;
      const int lda = matrix->layout == W8BLayout::NK ? matrix->K : matrix->N;
      check_blas(cublasGemmEx(handle, trans_a, CUBLAS_OP_N, matrix->N, M, matrix->K,
                              &alpha, matrix->f16.p, CUDA_R_16F, lda,
                              x.p, CUDA_R_16F, matrix->K, &beta, y.p, CUDA_R_16F, matrix->N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    }
  }
  check(cudaEventRecord(stop, stream), "record stop");
  check(cudaEventSynchronize(stop), "sync stop");
  const double result = elapsed_ms(start, stop) / 5.0;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  int force_split_k = 0;
  if (argc == 3 && std::string(argv[1]) == "--force-split-k") {
    force_split_k = std::atoi(argv[2]);
    if (force_split_k < 1) {
      std::cerr << "--force-split-k must be positive\n";
      return 2;
    }
  } else if (argc != 1) {
    std::cerr << "usage: rwkv_w8a16_bench [--force-split-k N]\n";
    return 2;
  }
  check(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
  cublasHandle_t handle;
  check_blas(cublasCreate(&handle), "cublasCreate");
  check_blas(cublasSetStream(handle, stream), "cublasSetStream");
  Buffer<unsigned char> workspace;
  workspace.resize(static_cast<std::size_t>(128) << 20);

  std::vector<Matrix> owned;
  owned.emplace_back(make_matrix(4096, 4096, W8BLayout::NK, 1));
  owned.emplace_back(make_matrix(4096, 16384, W8BLayout::NK, 2));
  owned.emplace_back(make_matrix(16384, 4096, W8BLayout::KN, 3));
  owned.emplace_back(make_matrix(4096, 65536, W8BLayout::NK, 4));
  std::vector<Matrix*> matrices;
  for (Matrix& matrix : owned) matrices.push_back(&matrix);

  for (int M : {1, 2, 4, 8, 16, 32, 64, 128}) {
    std::vector<half> host_x(owned.back().K * M, __float2half(0.0f));
    std::mt19937 rng(static_cast<unsigned>(M));
    for (Matrix& matrix : owned) {
      std::vector<Matrix*> one{&matrix};
      host_x.resize(static_cast<std::size_t>(matrix.K) * M);
      for (half& value : host_x) value = __float2half(static_cast<float>(static_cast<int>(rng() % 2000) - 1000) / 1000.0f);
      Buffer<half> x;
      Buffer<half> y;
      x.resize(host_x.size());
      check(cudaMemcpy(x.p, host_x.data(), host_x.size() * sizeof(half), cudaMemcpyHostToDevice), "copy x");
      y.resize(static_cast<std::size_t>(M) * matrix.N);
      const double int8_ms = bench_w8(stream, one, M, x, y, workspace, force_split_k);
      const double fp16_ms = bench_f16(stream, handle, one, M, x, y);
      std::cout << "batch=" << M << " K=" << matrix.K << " N=" << matrix.N
                << " w8a16_ms=" << int8_ms << " fp16_ms=" << fp16_ms
                << " speedup=" << (fp16_ms / int8_ms) << "\n";
    }
  }
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  return 0;
}
