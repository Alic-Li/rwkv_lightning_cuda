# RWKV-7 state-tuning sidecar

This directory is intentionally not a general training framework. It adds a
CUDA-only reverse path for frozen FP16 runtime weights; BF16 model archives are
still converted by the existing loader to the same FP16 runtime representation.
No function in the sidecar has a weight-gradient output.

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

- CUDA, head size 64.
- FP16 compute with FP32 WKV state and optimizer tensors.
- No W8A16/INT8 training path.
- Samples are truncated only at `--ctx`. `--chunk` controls activation memory.
  Forward saves WKV and attention/FFN shift states at chunk boundaries.
  Backward recomputes each chunk in reverse order and carries all three state
  gradients across boundaries, without detaching the recurrent graph.
- One chunk activation tape is reused. Boundary checkpoints currently remain
  on GPU, so their memory grows with the number of chunks.
- `--batch-size N` (alias `--batch`) accumulates N independent samples
  sequentially, then runs one Adam update. This supports variable lengths
  without padding; it does not execute samples in parallel on GPU. Loss and
  gradients are weighted by the total number of valid tokens in the batch.
- Nonfinite loss/dState stops training before Adam, preserving the last good
  optimizer state. The progress bar reports updates, loss, tokens/s, and ETA.

## Standalone CLI

The CUDA build produces `rwkv_state_tune`. It streams JSONL records containing
exactly one `text` field, tokenizes each record, applies causal next-token loss,
and performs one state-only Adam update per batch:

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

The CUDA regression test checks WKV input/state derivatives against CPU double
finite differences and verifies the chunk boundary adjoint. Full model
PyTorch gradient alignment remains a separate validation task.
