#pragma once
#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <array>
#include <cstddef>

namespace rwkv7_miss {
// Original logical layout D[out,rank]. Ragged last blocks are zero padded:
// S[t,j] = sum_{k < in, k % rank == j} X[t,k]. No fixed A is stored.
enum Target { Receptance, Key, Value, Output, FfnKey, FfnValue, TargetCount };
struct Linear {
  int in = 0, out = 0, rank = 0;
  float scale = 1;
  const half *d = nullptr; // MiSS D, or NoRA B
  const half *a = nullptr; // NoRA A[rank,in]; null selects MiSS
  // NoRA A^T[in,rank]. GPU leases carry only this layout (a == nullptr);
  // such entries are consumed by forward_rows, not forward.
  const half *at = nullptr;
};
using Block = std::array<Linear, TargetCount>;
// Inference uses fused reduction/project/add when reduced is null.
// Optional rows*rank FP32 scratch is retained for forward diagnostics.
void forward(cudaStream_t, int rows, const Linear &, const half *x, half *y,
             float *reduced = nullptr);
// Packed decode rows may each use a different adapter (or none). row_slot[r]
// selects table[slot] (MiSS when a is null, NoRA otherwise); -1 or a null d
// leaves row r untouched. A null row_slot applies table[0] to every row
// (single-adapter prefill). One launch serves every adapter in the batch, so
// mixed traffic shares a single base-model forward. With relu2_input, x is
// the FFN pre-activation and relu(x)^2 is applied on load, letting the sparse
// FFN value kernels keep their fused activation.
// latent: FP32 scratch of forward_rows_scratch(rows, in, max_rank) floats.
std::size_t forward_rows_scratch(int rows, int in, int max_rank);
void forward_rows(cudaStream_t, int rows, int in, int out, int max_rank,
                  const int *row_slot, const Linear *table, const half *x,
                  half *y, float *latent, bool relu2_input = false);
} // namespace rwkv7_miss
