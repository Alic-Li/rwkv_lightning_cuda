# Native BF16 training backbone

`rwkv_state_tune` and `rwkv_miss_tune` now build from this directory and link
`rwkv_bf16_training`, independently of `rwkv7_fast_backend_core`. The existing
FP16 inference backend remains unchanged apart from accepting BF16 adapter
files. The layer forward/reverse implementations were copied from the existing
state-tuning path; the linear, normalization, mixing and activation primitives
implement the same RWKV ordering with BF16 inputs and outputs.

## Precision

- Frozen model weights, embeddings, activations, shifts, dLogits, dX, MiSS
  forward D and intermediate sequence gradients: BF16 throughout training.
- GEMM accumulation, reduction, normalization statistics, loss, WKV recurrent
  state and its adjoints: FP32.
- Trainable master state/D, accumulated parameter gradients and optimizer
  moments: FP32. Base dW is never computed.
- No loss scaling is used or exposed in the CLI. Token-weighted mean loss and
  gradients remain normalized across all samples/chunks. Resume requires the
  current checkpoint configuration exactly; legacy fields are not migrated.

WKV is **not forked here**. CUDA directly compiles and calls
[`cuda/rwkv7_statepassing_clampw.cu`](../../cuda/rwkv7_statepassing_clampw.cu) and
[its shape helper](../../cuda/rwkv7_statepassing_clampw.cpp), selecting
`IoType::BF16`. HIP uses the existing corresponding HIP operator. The shared
operator also supports old callers; the new training execution never selects
its FP16 specialization.

## Loading and memory

The loader always opens the PTH archive in streaming mode. It loads one tensor
at a time, in layer order, using two bounded 8 MiB pinned buffers. Native BF16
storage is copied directly. FP32 or FP16 *input archives* are converted directly
to BF16 before H2D; there is no FP16 model/activation copy in the training path
and no full-model FP32 host allocation. Noncontiguous or malformed base tensors
are rejected explicitly. FFN value weights are transposed on GPU with at most
one tensor of temporary GPU storage. The normalized BF16 embedding lookup table
remains in CPU RAM (512 MiB for the 7.2B model). `--chunk-load` remains accepted;
streaming is now unconditional. Quantized base archives are inference-only.

## Chunk training and artifacts

The CLI retains sequential microbatch accumulation. `--chunk` bounds activation
tapes, and `--wkv_tape` shares the WKV tape across layers through recomputation.
WKV state, attention/FFN shifts and their adjoints pass between chunks without
detaching; v_first gradients propagate between layers. All samples/chunks finish
before the optimizer updates parameters.

State exports (`state-step-*.pth`, `state-final.pth`) contain BF16
`blocks.N.att.time_state` in `[H,V,K]` layout. State master parameters remain
FP32 while training. As before, these state-only PTHs do not include optimizer
state and are not resumable optimizer checkpoints.

MiSS exports `adapter-final.pth` with BF16 `D[out,rank]` and embedded MiSS metadata
(`dtype=bfloat16`). Resumable directories remain
`checkpoint-N/{checkpoint.json,training.pth}`; master D, gradients, Adam moments
and initial state remain FP32 for exact resume. The configuration identifies
`training_dtype=bfloat16`. FP16-backbone checkpoints are rejected for BF16 exact
resume instead of silently changing the numerical model.

The existing server accepts BF16 state exports, BF16 MiSS exports and MiSS
`training.pth` files. Adapter RAM storage and content hashes retain the declared
BF16 bytes. At a GPU cache miss, the FP16 inference runtime converts D in pinned
staging once, then uploads the whole adapter; hot decode performs no repeated
conversion or upload. This conversion is an inference-only boundary.

## Validation

`rwkv_bf16_training_test` checks native WKV finite differences and chunk adjoints,
BF16 normalization backward, and tiny softmax gradients at a 262144-token batch
normalizer without loss scaling. `tools/check_miss.py` checks BF16 full-model
PyTorch gradients, all six targets, shared tapes, chunk/accumulation consistency,
loss decrease, exact BF16 checkpoint resume/export, streamed source conversion,
and FP16/W8A16/W4A16 serving. CUDA is exercised locally; HIP execution requires
an AMD GPU and must not be inferred from CUDA results.
