# RWKV-7 state tuning

English | [简体中文](README.zh-CN.md)

The active `rwkv_state_tune` and `rwkv_miss_tune` binaries now use the independent
[native BF16 training backbone](../bf16_training/README.md). It keeps chunk/state
passing and microbatch accumulation, streams base tensors into BF16, and exports
BF16 state/adapter files. WKV directly reuses the standard operator in `cuda/`.

This directory retains the shared JSONL dataset reader and state-tuning usage
notes. The legacy FP16 training sidecar and its inference-backend bridge have
been removed. Neither training CLI links the inference core. MiSS training is
implemented in `src/bf16_training/`; FP16 MiSS remains an inference format.

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
- BF16 compute with FP32 WKV state, master parameters and optimizer tensors.
- No W8A16/INT8 training path.
- Samples are truncated only at `--ctx`. `--chunk` controls activation memory.
  Forward saves WKV and attention/FFN shift states at chunk boundaries.
  Backward recomputes each chunk in reverse order and carries all three state
  gradients across boundaries, without detaching the recurrent graph.
- One chunk activation tape is reused. Boundary checkpoints currently remain
  on GPU, so their memory grows with the number of chunks.
- `--batch-size N` (alias `--batch`) accumulates N independent samples, then
  runs one optimizer update. Loss and gradients are weighted by the total
  number of valid tokens in the batch.
- `--micro-bsz N` runs N of those samples together without padding: each chunk
  step packs every sample's next tokens back to back, and finished samples
  take no rows. Activation memory scales with `N * chunk`. The same applies to
  `rwkv_miss_tune`, `rwkv_nora_tune` and the online RL servers.
- `--gemm-mode bf16` (default) pins one GEMM kernel per shape with FP32
  reductions, so a row's result never depends on how many rows share the GEMM:
  `--micro-bsz` and `--chunk` change no bits of the forward pass. `strict`
  uses pedantic FP32 accumulation; `auto` restores the previous per-call
  cuBLAS choice (which may use reduced-precision split-K reductions).
- Nonfinite loss/dState stops training before the optimizer, preserving the last good
  optimizer state. The progress bar reports updates, loss, tokens/s, and ETA.

## Standalone CLI

Both GPU backends produce `rwkv_state_tune`. It streams JSONL records containing
exactly one `text` field (or masked `segments`, below), tokenizes each record, applies causal next-token loss,
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

### Masked loss (train only the assistant reply)

A row may instead list `segments`. Every segment is fed through the model, but only
tokens from segments with `"train": true` are loss targets:

```json
{"segments":[{"text":"User: Hi\n\nAssistant:","train":false},{"text":" Hello! How can I help?\n\n","train":true}]}
```

- Each segment needs exactly one string `text` and one boolean `train`; `{"text":"..."}`
  rows still train every token.
- Segments are tokenized independently and concatenated, so split them where a token
  boundary is natural (for example right after `Assistant:`).
- Loss and gradient are averaged over trained tokens in the batch. Untrained prefixes
  still run forward and backward, because the initial state gets its gradient through
  them; tokens after the last trained token are dropped. Rows with no trained token
  inside `--ctx` are skipped.
- The progress line reports `tokens=` (positions run through the model, used for
  tok/s) and `trained=` (loss targets).
- `rwkv_miss_tune` reads the same format.

State exports contain only BF16 `blocks.N.att.time_state` tensors in PyTorch
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
is replayed from the saved state into one shared tape, which keeps one state
checkpoint per 16 rows; the WKV backward recomputes the rows in between with
the forward's arithmetic, so gradients are bitwise those of a full tape.
Linear layers and FFN are not replayed by this option. All layers execute
sequentially on the same stream; chunk boundary gradients remain connected.

For L layers, H heads and T=min(ctx,chunk), WKV tape-related storage changes
from `4*L*H*T*64*64` bytes to `4*H*64*64*(ceil(T/16)+16+L+1)` bytes (shared
checkpoint tape with its 16-state backward scratch, per-layer initial states,
and shared final-state scratch). This trades an additional WKV forward
recurrence per backward block for lower memory.
Other activations, weights, and GPU chunk-boundary checkpoints are unchanged;
very small chunks or a single layer may not benefit.

```bash
./build/rwkv_state_tune --model /path/to/model.pth \
  --data /path/to/train.jsonl --output ./state_output \
  --ctx 2048 --chunk 256 --optimizer adam --wkv_tape
```

See the [FP16 gradient precision investigation](../../docs/training-precision-investigation.md)
for real-model failure reproduction and validation.
