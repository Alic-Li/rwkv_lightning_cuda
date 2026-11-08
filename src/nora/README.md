# NoRA-init tuning and kernels

This implements the **random NoRA-init** variant of
[Normalized Low-Rank Adaptation](https://arxiv.org/abs/2608.31036)
([project and reference implementation](https://spherelab.ai/NoRA/)).

For each selected frozen linear weight, `Y = base(X) + (X @ A.T) @ B.T`:

- `A[rank, in]` starts from seeded uniform random values. Each **column** is
  L2-normalized along the rank dimension, once, in FP32.
- `B[out, rank]` starts at zero. Both A and B receive gradients and Adam updates.
- `alpha = rank` is required, so `alpha/rank = 1`. A different CLI alpha is rejected.
- No constant/block-identity initialization, frozen A, or repeated normalization.
- Base weights and initial recurrent state remain frozen. Forward factors are
  BF16; masters, parameter gradients, Adam moments and latent products are FP32.

The six target names, masked JSONL format, sequential gradient accumulation,
learning-rate schedule and chunk/state passing follow MiSS. Input activations
for dA are reconstructed from the existing chunk tape and saved incoming shifts.
One input/latent-adjoint workspace is shared across layers, avoiding an extra
full activation per layer and projection. No dense base dW or delta-W is created.

Training kernels live in `src/nora/training_kernels.cu`; FP16 inference
kernels live in `src/nora/kernels.cu`. The tiled product primitive is shared in
`include/rwkv/runtime/nora_gemm.cuh`. Adapter views, serialization and the trainer
reuse the MiSS infrastructure; a non-null A selects
NoRA. Kernels use FP32 accumulation, not Tensor Core products. Large-model
throughput has not been benchmarked. HIP shares these sources but has not been
built or validated on AMD hardware.

## Build and training

`RWKV7_STATE_TUNING=ON` builds `rwkv_nora_tune`; with Drogon it also builds
`rwkv_nora_tune_server`. Both are included in runtime/release bundles.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 4

./build/rwkv_nora_tune \
  --model /mnt/nvme0n1/rwkv7-g1k-7.2b-20260930-ctx25600.pth \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --data /mnt/nvme0n1/no22/segments-2-2.jsonl \
  --output train-2-2-g1k/g1k-nora-r16-lr2e-5-e2 \
  --targets all --ctx 16384 --chunk 8192 --wkv_tape \
  --batch-size 8 --warmup-steps 10 --save-every 186 \
  --rank 16 --alpha 16 --lr 0.00002 --lr-final 0.000002 --epochs 2
```

`--alpha` defaults to rank. `--seed` defaults to 1234 and controls random A
initialization. Rank 1–1024 is supported, including nonmultiples of input width.
The example above is a launch configuration, not a measured memory/throughput
result. Training an existing model does not require running a conversion script.

## Checkpoints and inference packages

`checkpoint-N/{checkpoint.json,training.pth}` stores both factors' FP32 masters,
gradients and Adam moments, plus the frozen initial state and exact run progress.
Use the same configuration with `--resume checkpoint-N` and a new output path.
MiSS checkpoints cannot resume NoRA training and vice versa.

Successful training writes `adapter-final.pth` containing BF16 tensors named
`blocks.N.<target>.B` and `.A`, and embedded `archive/nora.json`. The manifest
uses `method=nora`, `layout=low_rank_ba`, `initialization=nora_init`, rank, alpha,
scale, base fingerprint and content digest. A/B shapes and native bytes are both
validated and hashed. No initialization is reapplied when loading trained A.

The native inference server uses only the currently loaded model weights. Request adapters and their HTTP management endpoints have been removed; legacy adapter fields return HTTP 400. Dynamic State upload, selection, caching and sessions remain supported. MiSS/NoRA kernels, package formats and standalone training tools are retained. Online adapter checkout and online weight merging are not implemented in the server; use `rwkv_peft_weight_merge` for offline weight merging.

## Online tuning server

```bash
./build/rwkv_nora_tune_server --model model.pth \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --rank 16 --alpha 16 --targets all --port 8850 \
  --chunk 256 --ctx 4096 --output nora-online-output
```

The default port is 8850. `--adapter adapter-final.pth` cold-loads both A and B;
rank/alpha/targets can be inferred from the package. `--resume` accepts a policy
publication directory saved with `checkpoint:true`. `/health` reports `mode=nora`.
`/v1/completions`, `/v1/train`, `/v1/publish`, signed advantages, actor snapshots,
reference policy, versioned prefix caching and restore follow the
[online RL API](../../docs/online-rl.md). Each train request supplies its learning
rate. The state tuning server can also use a frozen NoRA adapter.

## Validation

```bash
ctest --test-dir build --output-on-failure
python tools/check_nora.py --build build --work /tmp/nora-acceptance
python test/online_rl_http_test.py --modes nora --build build \
  --model /tmp/nora-acceptance/tiny.pth --vocab /tmp/nora-acceptance/vocab.txt \
  --adapter /tmp/nora-acceptance/learn/adapter-final.pth --work /tmp/nora-online-acceptance
```

Use an environment with PyTorch for the independent oracle. Tests cover tiled
and scratch-free forward, dA/dB/dX against a double-precision reference, rank 1
and rank larger than input, random unit-column initialization, first-step zero dA,
second-step nonzero dA, two-layer autograd across all six targets, chunking,
batch accumulation, bitwise resume, loss decrease and the training HTTP lifecycle.

On RTX PRO 6000 Blackwell, the two-layer test loss fell from 5.2132 to 2.5217
in 16 updates. Full-model autograd, exact resume,
and online rollout/train/publish/cache/restore checks passed. The supplied 7.2B
training command has not been launched as part of these tests.
