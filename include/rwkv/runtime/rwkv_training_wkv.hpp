#pragma once

#include "rwkv/runtime/rwkv_gpu_runtime.hpp"
#include <cstddef>
#include <cstdint>

namespace rwkv7_state_tuning {

// Shared exact WKV operator used by the standalone BF16 training backbone.
enum class IoType : std::uint8_t { F16, BF16 };

struct WkvShape {
  int batch = 0;
  int time = 0;
  int heads = 0;
  int head_size = 64;
};

// Exact short-context tape: FP32 [B,H,T,K,V]. A later checkpoint/recompute
// implementation can change the storage policy without changing the API.
struct WkvTapeView {
  float *state_before = nullptr;
  std::size_t state_before_elements = 0;
};

std::size_t wkv_tape_elements(const WkvShape &shape);

// r/w/k/v/a/b/y are [B,T,H,N]. w is the raw clamp-w input including w0.
// Initial/final states use the inference physical ABI [B,H,K,V].
// An empty tape disables recording for forward only.
void wkv_forward(cudaStream_t stream, IoType io_type, const WkvShape &shape,
                 const float *initial_state, const void *r, const void *w,
                 const void *k, const void *v, const void *a, const void *b,
                 void *y, float *final_state, WkvTapeView tape);

// Input gradients only. ds_final may be null (zero). ds_initial is FP32;
// sequence gradients have io_type. No frozen-weight gradient is computed.
void wkv_backward_input(cudaStream_t stream, IoType io_type,
                        const WkvShape &shape, const void *r, const void *w,
                        const void *k, const void *v, const void *a,
                        const void *b, const void *dy, const float *ds_final,
                        WkvTapeView tape, float *ds_initial, void *dr, void *dw,
                        void *dk, void *dv, void *da, void *db);

} // namespace rwkv7_state_tuning
