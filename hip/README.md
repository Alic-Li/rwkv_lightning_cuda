# HIP backend

English | [简体中文](README.zh-CN.md)

This directory contains the ROCm/HIP port of the inference backend and its GPU kernels.
Configure it with:

```sh
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```

The CUDA backend remains the default. HIP sources are kept separate so backend-specific
kernel tuning can evolve without changing the CUDA implementation.

The port's warp-oriented kernels are currently validated with the native 32-lane
wavefront mode on `gfx1100`; other AMD architectures require correctness testing.

## State tuning

`RWKV7_STATE_TUNING=ON` (the default) also builds `rwkv_state_tune` on HIP.
The HIP implementations are `rwkv7_statepassing_clampw.hip`,
`backward_kernels.hip`, `block_forward.hip`, `block_backward.hip`,
`model_forward.hip`, and `training_kernels.hip`. They reuse backend-neutral
host code for model backward orchestration, tape sizing and checkpoint writing.
New reduction kernels explicitly use 32-lane shuffle subgroups; the existing
inference kernels still require architecture-specific validation beyond gfx1100.

```sh
./build-hip/rwkv_state_tune \
  --model /mnt/SDD_1/rwkv7_model_weights/rwkv7-g1i-7.2b-20260805-ctx16384.pth \
  --data ./html_vibe_rwkv_clean.jsonl \
  --output ./build-hip/state-smoke-7b \
  --chunk-load --ctx 64 --chunk 32 --max-steps 2 --save-every 1
```

Both training CLIs use the independent [BF16 backbone](../src/bf16_training/README.md).
Weight loading always streams through bounded staging buffers. `--chunk` controls
activation recomputation and carries recurrent gradients across boundaries.
State exports contain BF16 tensors; master parameters and optimizer state are FP32.

`rwkv_bf16_training_test` covers WKV finite differences and chunk adjoints,
normalization backward and cross-entropy gradients. It uses HIP in a HIP build;
CUDA results do not establish HIP correctness. The legacy FP16 training sidecar
and `rwkv_state_tuning_cuda_test` have been removed.

HIP inference now accepts W4A16/W8A16 `.rwkvq` archives as well as BF16 PTH
weights. Quantized projection weights remain compressed on device; gfx1100
uses vectorized GEMV and native RDNA3 WMMA with FP32 accumulation. See
[quantized inference and W7900 validation](QUANTIZATION.md) for formats, tests,
commands and measured limitations. State tuning still requires FP16/BF16 frozen
weights and rejects quantized projections.
