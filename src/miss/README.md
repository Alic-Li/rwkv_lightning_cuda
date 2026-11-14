# MiSS adapter training and serving

English | [简体中文](README.zh-CN.md)

The BF16 training base remains frozen. Only `D[out, rank]` receives parameter gradients.
This implements the efficient MiSS form from [the paper](https://arxiv.org/pdf/2409.15371)
and [reference project](https://github.com/Joluck/MiSS), with no stored or trained A.

- [Math and target placement](#math-and-target-placement)
- [Build and quick start](#build-and-quick-start)
- [Training options](#training-options)
- [Checkpoints and resume](#checkpoints-and-resume)
- [Inference package](#inference-package)
- [HTTP API and caches](#http-api-and-caches)
- [Profiling](#profiling)
- [Validation and current limits](#validation-and-current-limits)

The measured large-model baseline and staged performance work are tracked in
[OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md).

## Math and target placement

For input width K, `S[t,j] = sum(X[t,k] for k < K if k % rank == j)`.
The final partial block is implicitly zero padded, including when rank > K.
`Y = base_linear(X) + scale * S @ D.T`, with `scale = alpha / rank`.
Ranks 1–1024 and up to 65535 rows per invocation are supported.

Training uses the independent [BF16 backbone](../training/README.md).
D is BF16 for forward and dX; master D, accumulated dD and Adam moments are
FP32. Reduction and projection accumulate in FP32. dD accumulates over every
chunk and sample before one Adam update; no base dW is allocated or computed.
Backward adds `scale * (G @ D)[..., k % rank]` to the frozen base dX.

Targets are comma-separated original weight names, or `all`:

- `att.receptance.weight`, `att.key.weight`, `att.value.weight`, `att.output.weight`
- `ffn.key.weight`, `ffn.value.weight`

Attention key/value deltas precede key gating and value residuals. Layer zero
writes the adapted value into v_first. FFN key delta precedes ReLU²; FFN value
consumes ReLU² output. Inference keeps the sparse FFN value kernels: they read
the pre-activation, so the adapter applies ReLU² while loading its input.

Training reuses the exact state-passing WKV operator and existing attention
shift, FFN shift, v_first and cross-chunk adjoints. Reduced inputs (rows × rank
FP32 values per target) are saved in the chunk tape, then recomputed with the
chunk during reverse traversal. Full input matrices need not persist for MiSS.
`--wkv_tape` also enables the existing shared WKV replay tape.

## Build and quick start

CMake builds `rwkv_miss_tune` when `RWKV7_STATE_TUNING=ON` (default), for CUDA
and HIP. JsonCpp and OpenSSL are needed for manifests and SHA-256 identities.
Python/PyTorch is needed only for the independent validation scripts.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 6
./build/rwkv_miss_tune --model model.pth --data train.jsonl --output miss_output \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --rank 16 --alpha 16 --targets all --ctx 2048 --chunk 256 --batch-size 2 \
  --epochs 1 --lr 0.001 --lr-final 0.0001 --warmup-steps 10 --save-every 100
```

Each JSONL record must be `{"text":"..."}` or a masked
`{"segments":[{"text":"...","train":bool}, ...]}` row (see
[state tuning](../state_tuning/README.md#masked-loss-train-only-the-assistant-reply)). Paths are resolved from the shell's
working directory. A production-sized example is:

```bash
./build/rwkv_miss_tune \
  --model /mnt/nvme0n1/rwkv7-g1j-7.2b-20260831-ctx16384.pth \
  --data /mnt/nvme0n1/html_vibe_datasets/3d/3000_training.jsonl \
  --output nora_output \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --rank 16 --alpha 16 --targets all \
  --ctx 4096 --chunk 1024 --batch-size 8 --epochs 1 \
  --lr 0.0001 --lr-final 0.00001 --warmup-steps 10 \
  --save-every 100 --wkv_tape
```

### Training options

| Option | Meaning |
|---|---|
| `--model`, `--data`, `--output`, `--vocab` | Base checkpoint, JSONL dataset, new output directory, and tokenizer vocabulary |
| `--rank` | MiSS rank; default 16; trailing input columns use implicit zero padding |
| `--alpha` | Adapter alpha; default equals rank, so the default effective scale is 1 |
| `--targets` | `all` or a comma-separated subset of the six supported weight suffixes |
| `--state` | Optional frozen initial WKV state; attention and FFN shifts still start at zero |
| `--lr`, `--lr-final`, `--warmup-steps` | Adam learning-rate schedule |
| `--ctx`, `--chunk` | Maximum tokens per sample and state-passing chunk length |
| `--batch-size` | Number of independent samples accumulated before one optimizer update |
| `--epochs`, `--max-steps` | Dataset passes and optional optimizer-update limit |
| `--save-every`, `--resume` | Periodic resumable checkpoint interval and checkpoint directory |
| `--wkv_tape` | Share the WKV replay tape across layers, trading extra WKV recomputation for lower memory |

Defaults are `epochs=1`, `rank=16`, `alpha=rank`, `ctx=128`, `chunk=64`, and
`batch-size=1`. Steps count optimizer updates, not chunks. The learning-rate
horizon comes from epochs and dataset size; changing only `--max-steps` for an
interrupted run preserves that horizon. Batch size currently processes samples
sequentially and accumulates valid-token-weighted gradients. It is not a
parallel GPU batch, which can limit utilization on large GPUs.

BF16 training uses unscaled backward with no `--loss-scale` option; intermediate
gradients remain BF16 and parameter gradients accumulate in FP32. The backbone
and loader do not reuse FP16 inference weights. Exact resume requires a BF16
training checkpoint with the same configuration; old FP16-backbone checkpoints
remain usable for inference but cannot silently resume under different arithmetic.

### Checkpoints and resume

Checkpoints are directories `checkpoint-N/{checkpoint.json,training.pth}`.
They include FP32 master D, dD, Adam moments, initial state, optimizer step,
scheduler horizon, dataset epoch/row, configuration fingerprints and RNG state.
The data order and zero initialization are deterministic; training currently
uses no stochastic operation after initialization. Checkpoints are published
at update boundaries, where accumulated gradients have been cleared.

Resume with the same training configuration and `--resume checkpoint-N`;
use a new output directory. Model/data/vocabulary/initial-state fingerprints
and schedule settings are checked. Inference always exports to `output/adapter-final.pth`
on successful completion. Existing output packages/checkpoints are not overwritten.

## Inference package

The final single PTH contains BF16 D tensors and an embedded `archive/miss.json`
manifest (not an optimizer tensor). It remains readable with `torch.load`.
New `checkpoint-N/training.pth` files embed the same inference metadata and can
also be registered or uploaded independently. The loader reads only `.D.master`
from a training checkpoint and converts it to the manifest dtype in RAM;
BF16 D is converted to FP16 only at inference GPU admission. Gradient and optimizer
tensors stay out of the RAM/GPU adapter caches. Resume still uses the complete
checkpoint directory and its original FP32 tensors.

Legacy `training.pth` files need their sibling `checkpoint.json` for registration;
when uploading an old checkpoint, include that JSON as a second multipart file.
Legacy checkpoints containing only FFN value targets may lack the input width
needed for inference and require re-export. Original two-file packages remain
supported:

- `adapter.json`: format_version=1, kind=inference_adapter, method=miss,
  dtype=bfloat16 (legacy float16 is also accepted), layout=modulo_rank_zero_pad, source base SHA-256, rank,
  alpha, scale, target names, layer numbers, input widths, D shapes,
  content_version (SHA-256 of canonical manifest without the version and native-dtype D bytes).
- `adapter.pth`: torch-readable BF16 tensors named
  `blocks.N.att.key.weight.D`, etc., in original `[out,rank]` layout.

The loader checks format, shapes, duplicate targets, finite values and content hash.
Source base fingerprints must match. The base fingerprint is captured once
when loading the model, before requests can bind adapters; this adds a
sequential model-file hash read at startup. `rwkv_quantize` writes
`output.rwkvq.source.json`, binding the quantized archive hash to its source PTH;
keep this file beside the quantized model to reuse an adapter trained on its
unquantized source PTH. Older quantized models need to be requantized to obtain
this provenance file. Quantized base **inference** is supported; training still
uses a separate native BF16 backbone with streamed BF16/FP32/FP16 input conversion.

## HTTP API and caches

Registration, listing, deletion, authentication, and generation request examples
are maintained in the [HTTP API guide](../../docs/http-api.zh-CN.md#miss-适配器注册列表与删除)
([English](../../docs/http-api.md)). Registration reads an inference directory on
the server or accepts a multipart PTH upload at `POST /v1/adapters`. Existing request
handles survive deletion, while new lookups of deleted versions fail.

GPU miss uploads the entire contiguous D collection once using pinned staging
and a nonblocking copy stream. NoRA A is staged transposed (A^T[in,rank]) in
place of A, so inference reads it contiguously at no extra memory. A ready event must complete before publication.
Admission/upload is serialized under the cache mutex, which coalesces same-version
concurrent misses; uploads of different adapters are currently serialized too.
GPU hits reuse that allocation across all layers, prefill and decode. Active
leases prevent eviction; unleased allocations use LRU. Pause releases the
adapter GPU lease and trims unleased allocations; resume reloads from RAM.
RAM counts active deleted-version handles as well as registered packages.

Budgets are independent process environment settings in MiB:

```bash
RWKV_ADAPTER_RAM_MIB=1024
RWKV_ADAPTER_GPU_MIB=512
RWKV_ADAPTER_STAGING_MIB=128
```

An adapter must fit the staging budget as a whole. Admission rejects packages
exceeding RAM/staging/GPU budgets, active-lease capacity or available GPU memory.
These limits cover D payloads, not JSON/container overhead or base-model memory.
The runtime currently assumes one GPU device per serving process.

Effective state identity contains a unique loaded-model lifetime, adapter
content version, effective scale, initial-state identity and WKV precision.
State copies and continuation reject mismatches. Session cache keys are
namespaced; persisted states retain their effective identity, and cannot be
reused across a different runtime model lifetime. There is no radix cache in
this checkout; any future radix cache must use the same effective identity.
Shared `LayerWeights` are never modified.

`GET /v1/adapters` reports RAM/GPU hits, misses, uploads, H2D milliseconds,
resident bytes and adapter GPU peak bytes. Generation emits JSON
`miss_request_metrics` records with cold/hot cache status, TTFT and
sample-to-sample TPOT.

### Serving cost

Adapters never split the continuous decode batch. Every row names its own GPU
lease (or none), the base model runs once for the whole batch, and each adapted
projection adds one latent launch plus one expand launch covering every adapter
in the batch (base rows exit immediately). Prefill uses the same kernels with all
rows on one adapter. Leases are acquired on the request thread, registration runs
on the HTTP worker pool, and the base-model fingerprint is hashed in the
background after load, so cold switches do not stall other streams.

RTX 4060 Ti, 1.5B FP16, rank 16, all targets, 256 greedy tokens per request
(`test/adapter_hotswap_bench.py`), aggregate tokens/s:

| Concurrency | Base | MiSS | NoRA | Base + MiSS | Base + MiSS + NoRA |
|---|---:|---:|---:|---:|---:|
| 1 | 116 | 107 | 107 | – | – |
| 8 | 660 | 638 | 633 | 613 | 603 |
| 32 | 1767 | 1652 | 1622 | 1669 | 1644 |

The previous implementation measured 85/77 (MiSS/NoRA) at concurrency 1,
546/287 at 8 and 1602/922 at 32, with 327 (base + MiSS) and 162 (three-way) at 8.
A 1913-token prompt prefills in 0.237 s (base), 0.275 s (MiSS) and 0.289 s (NoRA).

With 500 simultaneous streaming requests of 128 tokens (64 decode slots), five
NoRA adapters with 100 requests each reach 1628 tokens/s against 1761 for base
and 1638 for a single shared NoRA adapter (previously 408 and 1109).
Concurrently used adapters must fit `RWKV_ADAPTER_GPU_MIB` together, because
leased adapters cannot be evicted.

## Profiling

`nvidia-smi` GPU utilization reports whether the device is busy; it is not a
FLOP utilization percentage. Hopper and newer GPUs expose more useful GPM
counters through `dmon`:

```bash
nvidia-smi dmon -i 0 \
  --gpm-metrics 2,3,5,7,10,13,249,250,252,260 \
  --gpm-options d -d 1 --format csv -o DT \
  -f miss-training-gpm.csv
```

The key percentage columns are SM activity, SM occupancy, Tensor/HMMA activity,
DRAM activity, and FP16 activity. Pair them with optimizer-step token throughput,
power, and memory. Use Nsight Systems to find launch gaps and dominant kernels.
Nsight Compute gives kernel-level achieved throughput only when the driver
allows access to GPU performance counters; otherwise it fails with
`ERR_NVGPUCTRPERM`.

## Validation and current limits

```bash
ctest --test-dir build --output-on-failure
python tools/check_miss.py --build build --work /tmp/miss-acceptance
python tools/check_miss_http.py --build build --work /tmp/miss-acceptance
```

The work directory must be new. Tests cover explicit A/autograd dD/dX, finite
differences, ragged blocks, gradient accumulation, two-layer full-model autograd,
full versus chunked gradients, sequential batch accumulation, shared WKV tape,
loss decrease, bitwise checkpoint resume/export,
no-adapter versus zero-scale outputs, concurrent versions/scales, SQLite identity round trips, GPU admission/LRU/reload,
no repeated decode upload, and dynamic FP16/W8A16/SB6 W4/W6 inference.

On RTX PRO 6000 Blackwell, the two-layer test's loss decreased from 5.2124 to
2.9414 in 16 updates. CUDA numerical and runtime tests passed. HIP uses the
same MiSS kernel source and equivalent hooks, but **has not been built or run
on AMD hardware in this environment**.

Decode reduction/projection/add is one kernel with no global workspace. Small
training projections also save S in that kernel; larger training projections
use separate reduction and projection to avoid redundant input reduction.
Backward selects per-parameter threads for short chunks and shared-memory
16×16 G/S tiles for longer, wide projections, with deterministic reductions
and no full A or delta-W. These are correctness-first kernels, not yet Tensor
Core tiled large-prefill/dD kernels.
The inherited sequential microbatch/WKV path also limits utilization.

Nsight Systems training traces were collected. Nsight Compute failed with
`ERR_NVGPUCTRPERM` on this machine; hardware performance counters require an
administrator's driver configuration. No claim of saturated GPU FLOPs or HIP
performance parity is made. Large-model throughput tuning and AMD execution
remain acceptance work.

See [the validation record](VALIDATION.md) for measured errors, microbenchmark
results and the exact remaining acceptance limits.

See the [FP16 gradient precision investigation](../../docs/training-precision-investigation.md)
for real-model failure reproduction and validation.
