# HIP migration validation

English | [简体中文](VALIDATION.zh-CN.md)

This file records the earlier FP16/state-tuning migration. W4A16/W8A16 were
subsequently migrated and validated separately: [quantized inference report](QUANTIZATION.md).

Validated locally on gfx1100 with ROCm HIP 7.2.53211-9999.

Guidance: AMD [magpie-kernel-evaluator](https://github.com/amd/skills/tree/6916fb371d1cba40757b223cf16b4cd4912b202f/skills/magpie-kernel-evaluator).
The Magpie CLI was not installed locally; validation used the native CMake/CTest
suite and executable CPU numerical references, not a Magpie performance report.

## Build and numerical tests

```
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 8
ctest --test-dir build-hip --output-on-failure
```

Result: 10/10 tests passed. State-tuning checks cover WKV finite differences,
chunk state adjoints, LayerNorm, cross entropy, gradient reduction and Adam.
The inference test's synthetic vocabulary was padded to a multiple of four
to satisfy the existing sampler contract on both backends. Quantized W8 GPU
tests are now correctly restricted to CUDA, as W4 tests already were.

## Real-data smoke runs

All runs used `html_vibe_rwkv_clean.jsonl`, seed 1234 and two optimizer updates.

| Model | Context / chunk | Batch | Learning rates | Losses |
| --- | --- | --- | --- | --- |
| rwkv7-g1d-0.1b-20260129-ctx8192 | 64 / 32 | 2 | 0.1, 0.2 | 3.1931, 7.6466 |
| rwkv7-g1d-0.1b-20260129-ctx8192 | 64 / 64 | 2 | 0.1, 0.2 | 3.1931, 7.6499 |
| rwkv7-g1i-7.2b-20260805-ctx16384 | 64 / 32 | 1 | 0.1, 0.2 | 3.0527, 12.3912 |

The 7.2B run used `--chunk-load`. Each run produced step checkpoints and
`state-final.pth`; the writer validates archive layout with the inference PTH
parser. Outputs remain in `build-hip/state-smoke*`.

These runs verify execution and finite gradients, not convergence. The CLI's
default learning rate is aggressive. Different chunks can select different
FP16 GEMM execution shapes; the two 0.1B final states were not numerically
identical (maximum absolute difference 0.48238, RMS 0.02389 after two Adam
updates). WKV chunk adjoints independently passed the tighter numerical test.
Full-model gradient alignment, BF16 WKV numerical comparison and architectures
other than gfx1100 remain unvalidated. CUDA kernels were not executed locally.

## Extended validation after freeing GPU memory

All extended runs use constant `--lr 0.001 --lr-final 0.001 --warmup-steps 0`
and seed 1234. Logs and the checkpoint inference harness are retained under
`build-hip/validation/`.

- **7.2B, context 1024, chunk 128, batch 2, 10 updates:** completed 20,480
  tokens without the CLI's nonfinite loss/gradient guard firing. Losses were
  0.7898, 0.7837, 0.7840, 0.7284, 0.7556, 0.6936, 0.6643, 0.6037, 0.7202,
  0.6179. Samples differ across updates; this is not held-out evaluation.
- **0.1B, repeated first JSONL record, context 128, 10 updates:** loss changed
  from 3.3215 to 1.8455 with chunk 32 and from 3.3206 to 1.4973 with chunk 128.
  These are training losses on the same repeated sample, not generalization
  metrics. Final-state chunk comparison: max absolute difference 0.0121814,
  RMS 0.00330784. Chunk equivalence remains unproven at the full-model level.
- **Checkpoint reuse:** reloaded both the 0.1B repeated-sample checkpoint and
  the 7.2B context-1024 checkpoint via `ModelBackend::load_state_from_pth`,
  performed four-token prefill and one-token decode using FP32 WKV state,
  and checked all 65,536 output logits for finiteness. Both passed.

7.2B reproduction (change output directory to preserve existing artifacts):

```sh
./build-hip/rwkv_state_tune \
  --model /mnt/SDD_1/rwkv7_model_weights/rwkv7-g1i-7.2b-20260805-ctx16384.pth \
  --data ./html_vibe_rwkv_clean.jsonl \
  --output ./build-hip/state-7b-ctx1024 \
  --chunk-load --ctx 1024 --chunk 128 --batch-size 2 --max-steps 10 \
  --lr 0.001 --lr-final 0.001 --warmup-steps 0 --save-every 5
```

Extended 7.2B context-4096 run: `--ctx 4096 --chunk 256 --batch-size 2 --max-steps 3 --save-every 3`, with the same constant learning rate and dataset, completed 24,576 tokens. Losses: 0.5582, 0.5339, 0.5204. All three updates and checkpoint saves completed without nonfinite loss/gradient errors. An in-run ROCm memory snapshot reported 25,784,668,160 bytes (24.0 GiB) of device-wide VRAM usage; this is a snapshot, not a measured peak. Output: `build-hip/state-7b-ctx4096/state-final.pth`.
