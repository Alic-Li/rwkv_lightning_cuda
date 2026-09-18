# RWKV-7 state tuning (legacy FP16 implementation)

English | [简体中文](README.zh-CN.md)

The active `rwkv_state_tune` and `rwkv_miss_tune` binaries now use the independent
[native BF16 training backbone](../bf16_training/README.md). It keeps chunk/state
passing and microbatch accumulation, streams base tensors into BF16, and exports
BF16 state/adapter files. WKV directly reuses the standard operator in `cuda/`.

The implementation details below describe the retained FP16 sidecar,
not the current CLI's numerical path. The dataset
reader remains shared; FP16 sidecar tests continue to cover existing consumers.

This directory is intentionally not a general training framework. It adds a
CUDA/HIP reverse path for frozen FP16 runtime weights; BF16 model archives are
still converted by the existing loader to the same FP16 runtime representation.
No function computes frozen-base weight gradients. Optional MiSS views add
FP32 gradients for D only; see [MiSS training and serving](../miss/README.md).

## Short-context execution

1. ModelBackend::state_tuning_model_view exposes read-only views of the
   already-loaded inference weights and rejects INT8 tensors.
2. model_forward_state_tuning_f16 reuses the inference LayerNorm, time-mix,
   linear, activation, ChannelMix and post-WKV kernels. WKV uses the torch-free
   exact state-passing kernel and records FP32 [B,H,T,K,V] state checkpoints.
3. cross_entropy_forward_backward_f16 produces the scalar row-mean loss and
   FP16 dLogits.
4. model_output_backward_state_only applies the frozen head and final
   LayerNorm input gradients, then visits blocks in L-1 ... 0 order.
5. model_backward_state_only uses only two persistent [B,T,C] gradient buffers.
   Each block returns dx and a per-batch FP32 dState.
6. reduce_state_gradient_f32 reduces the batch dimension and
   adam_update_state_f32 updates only time_state, dState, m, and v.

The runtime state ABI is [H,K,V]. PyTorch state files use [H,V,K], so loading
and checkpoint serialization transpose the last two dimensions. The inference
loader already performs this conversion on load.

## Tape ownership

BlockTapeView is non-owning. The caller allocates its fields for one
short-context forward/backward. WKV tape sizing is returned by
wkv_tape_elements. Block temporary workspaces are reusable across layers.

v_first_grad is one model-level buffer shared by every block. Reverse execution
accumulates later-layer value-residual gradients and consumes them in layer 0.

## Execution and limits

- CUDA or ROCm/HIP, head size 64.
- FP16 compute with FP32 WKV state and optimizer tensors.
- No W8A16/INT8 training path.
- Samples are truncated only at `--ctx`. `--chunk` controls activation memory.
  Forward saves WKV and attention/FFN shift states at chunk boundaries.
  Backward recomputes each chunk in reverse order and carries all three state
  gradients across boundaries, without detaching the recurrent graph.
- One chunk activation tape is reused. Boundary checkpoints currently remain
  on GPU, so their memory grows with the number of chunks.
- `--batch-size N` (alias `--batch`) accumulates N independent samples
  sequentially, then runs one optimizer update. This supports variable lengths
  without padding; it does not execute samples in parallel on GPU. Loss and
  gradients are weighted by the total number of valid tokens in the batch.
- Nonfinite loss/dState stops training before the optimizer, preserving the last good
  optimizer state. The progress bar reports updates, loss, tokens/s, and ETA.

## Standalone CLI

Both GPU backends produce `rwkv_state_tune`. It streams JSONL records containing
exactly one `text` field, tokenizes each record, applies causal next-token loss,
and performs one state-only optimizer update per batch:

```bash
./build/rwkv_state_tune \
  --model /path/to/model.pth \
  --data /path/to/train.jsonl \
  --output ./state_output \
  --ctx 2048 \
  --chunk 1024 \
  --batch-size 2 \
  --epochs 1 \
  --lr 1.0 \
  --lr-final 0.01 \
  --warmup-steps 10 \
  --save-every 500
```

Checkpoints contain only FP32 `blocks.N.att.time_state` tensors in PyTorch
`[H,V,K]` layout. The writer validates each archive with the same PTH parser
used by inference before publishing it as `state-step-XXXXXXXX.pth` or
`state-final.pth`.

The GPU regression test checks WKV input/state derivatives against CPU double
finite differences and verifies the chunk boundary adjoint. Full model
PyTorch gradient alignment remains a separate validation task.

For ROCm build commands and a tested training example, see [HIP backend](../../hip/README.md).
Use `--chunk-load` to bound weight-loading staging buffers for large models.

### Optimizer selection

`rwkv_state_tune --optimizer adam` explicitly selects the existing Adam
optimizer; omitting `--optimizer` also uses Adam. Use `--optimizer muon` to
apply Muon independently to each layer/head's 64×64 time-state matrix.
All other model parameters remain frozen.

The CUDA FP32 implementation follows [KellerJordan/Muon](https://github.com/KellerJordan/Muon):
momentum 0.95, Nesterov enabled, five quintic Newton–Schulz iterations,
and no weight decay in the CLI. Unlike the reference's BF16 orthogonalization,
this implementation performs orthogonalization in FP32.
Both optimizers use the existing `--lr`, `--lr-final`, and `--warmup-steps`
schedule, including its unchanged defaults (1.0, 0.01, 10). Muon and Adam
learning rates are not interchangeable; set these explicitly when comparing
runs. For example, append `--optimizer muon --lr 0.02 --lr-final 0.002` to
an existing training command as a starting point for tuning.

### Shared WKV tape

Append `--wkv_tape` to enable layer-shared FP32 WKV tape, independently of
`--optimizer adam|muon`. Without this flag the original per-layer tape path
remains active.

Forward saves each layer's chunk initial WKV state and skips per-token WKV
state recording. Immediately before each block backward, its WKV recurrence
is replayed from the saved state into one shared tape. Linear layers and FFN
are not replayed by this option. All layers execute sequentially on the same
stream; chunk boundary gradients remain connected.

For L layers, H heads and T=min(ctx,chunk), WKV tape-related storage changes
from `4*L*H*T*64*64` bytes to `4*H*64*64*(T+L+1)` bytes (shared tape,
per-layer initial states, and shared final-state scratch). This trades an
additional WKV forward recurrence per backward block for lower memory.
Other activations, weights, and GPU chunk-boundary checkpoints are unchanged;
very small chunks or a single layer may not benefit.

```bash
./build/rwkv_state_tune --model /path/to/model.pth \
  --data /path/to/train.jsonl --output ./state_output \
  --ctx 2048 --chunk 256 --optimizer adam --wkv_tape
```

See the [FP16 gradient precision investigation](../../docs/training-precision-investigation.md)
for real-model failure reproduction and validation.
