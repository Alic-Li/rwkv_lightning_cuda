# W7900 W4A16 / W8A16 migration

English | [简体中文](QUANTIZATION.zh-CN.md)

The HIP backend accepts the same `.rwkvq` archives as CUDA. INT4/INT8 projection
weights remain compressed in VRAM. This migration was validated on a Radeon Pro
W7900, `gfx1100`, native wave32, ROCm HIP 7.2.53211-9999, AMD Clang 22.0.0,
Linux 6.19.6-zen1-1-zen. Base repository commit: `7a3ccf34ac2ac568cb90aa211eeb841ac3d70a60`.

## Build and use

Run from the repository root:

```sh
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 8
ctest --test-dir build-hip --output-on-failure

build-hip/rwkv_quantize --format w8a16 MODEL.pth MODEL.w8.rwkvq
build-hip/rwkv_quantize --format w4a16 --group-size 128 MODEL.pth MODEL.w4.rwkvq
# G32 is also supported; it uses more scales and may improve quantization quality.

build-hip/rwkv_quantized_smoke MODEL.w4.rwkvq
build-hip/rwkv_lighting_cuda --model-path MODEL.w4.rwkvq \
  --vocab-path assets/rwkv_vocab_v20230424.txt
```

The executable retains its existing name. Archive detection selects W4/W8;
no additional quantized-inference flag is required. Existing archives do not
need re-quantization. The PTH path still requires BF16 source weights.

## Implementation

- W4: signed two's-complement nibbles in NK order, even K in the low nibble;
  FP16 scales per output channel and K group, G32/G128. External -8 is accepted.
- W8: signed INT8 NK archive weights and one FP16 scale per output channel.
  The standalone HIP linear interface also supports raw KN. CUDA's in-memory
  PackedNK format, packing helpers and tuning caches are not HIP interfaces.
- Decode (M <= 4): 32-lane reductions, FP32 accumulation, aligned eight-element
  loads, and a bounded scalar path for tails/unaligned inputs/KN.
- Prefill (M > 4): gfx1100 native `v_wmma_f32_16x16x16_f16`, four waves per
  16x64 output tile, coalesced 64-wide K tiles unpacked into LDS. A portable
  scalar tiled implementation is retained and tested independently.
- Shape-based automatic split-K uses at most 16 partitions, caller-owned
  workspace and ordered FP32 reduction. No kernel-level allocations, stream
  synchronization, device queries, global tuning state or atomic accumulation.
- HIP loading keeps every integer projection in NK, including `ffn.value`.
  Quantized FFN applies ReLU-square then dense quantized down-projection; the
  existing FP16 sparse down-projection remains on the floating-point path.
- Attention receptance/key/value/output, FFN key/value and head dispatch on
  tensor dtype. Embedding, norms and low-rank tensors remain floating point.
  Unsupported integer tensor roles are rejected. Quantized state tuning is
  not supported; its existing FP16/BF16 requirement remains enforced.

RDNA3 fragment mapping follows the
[AMD GPUOpen WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/).
Only gfx1100 is hardware-validated here. A source fallback does not establish
support for another GPU, OS or ROCm version.

## Validation

The final HIP CTest suite passed **14/14** tests. New/ported checks include:

- W4 GPU quantization compared against CPU round-to-nearest-even scales and
  nibbles, G32/G128, all-zero groups, subnormals, external -8, odd K and tails;
  linear shapes up to M1024 and K16384, explicit/automatic split-K, insufficient
  workspace rejection, two streams and repeated Graph replay.
- W8 double-precision CPU dot-product references for NK/KN, signed byte range,
  zero/subnormal scales, odd dimensions, split-K, streams and Graph replay.
- A nonzero **two-layer** synthetic model for W4 G32/G128 and W8, compared
  with independently CPU-dequantized FP16 projection weights. B1/B2,
  prefill lengths 1/7/17/33, three decode steps and FP16/FP32 WKV states.
  Every logit must be finite and within `0.015 + 0.015*abs(reference)`;
  a nonzero-reference check prevents a trivial all-zero pass.
- Standalone scalar and WMMA builds both passed the W4/W8 CPU-reference suites.

Kernel comparisons use `0.0006*sum(abs(products)) + 0.0006*abs(reference) + 0.002`
as the per-output tolerance, accounting for FP16 dequantization in WMMA,
FP32 accumulation/reduction and FP16 output rounding.

Real-model validation uses fixed bilingual teacher-forced input, 32-token
prefill followed by 32 decode steps, then a separate 32-token greedy Chinese
generation. It checks every output logit for finiteness. The native tool also
reports logit MSE and top-1 agreement against a reference model:

```sh
build-hip/rwkv_quantized_validate MODEL.w8.rwkvq \
  assets/rwkv_vocab_v20230424.txt MODEL.pth

# Isolate migration error from quantization error: expand the SAME quantized
# weights on the CPU, then run them through the existing FP16 model path.
build-hip/rwkv_dequantize_reference MODEL.w4.rwkvq MODEL.w4-fp16-reference.rwkvq
build-hip/rwkv_quantized_validate MODEL.w4.rwkvq \
  assets/rwkv_vocab_v20230424.txt MODEL.w4-fp16-reference.rwkvq
```

Reference output must not already exist. It is a validation archive, not a
memory-saving inference option.

## Observed model results

Models: `rwkv7-g1d-0.1b-20260129-ctx8192.pth` and
`rwkv7-g1i-7.2b-20260805-ctx16384.pth`. SHA256:

```text
0.1B e10d7b1930c2644c5c6b194444774d6d82ec8212a78763493149de09aac7d83f
7.2B 0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8
```

7.2B, B1, FP32 WKV state, one warmup of each shape, host-clock timing with
synchronized forwards; decode is the median of 32 steps. These are short
local measurements, not a server throughput/load test. CPU work, allocation
and recurrent kernels are included. Weight VRAM is the loader's tensor-byte
accounting; it excludes state, temporary activations, 128 MiB workspace and
allocator overhead. All three keep a separate 512 MiB embedding table on CPU.

| Mode | Weight VRAM | 32-token prefill | Decode | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 PTH / FP16 runtime | 13,634 MiB | 87.48 ms | 19.995 ms | 50.01 |
| W4 G128 | 4,134 MiB | 120.26 ms | 15.005 ms | 66.64 |
| W8 | 7,237 MiB | 149.65 ms | 16.728 ms | 59.78 |

The first untuned W4 implementation measured 1,309 ms prefill and 35.46 ms
decode. Coalesced LDS loading, vectorized decode and automatic split-K reduced
these costs. **Final prefill remains slower than FP16**, despite the decode
and memory improvements. Small 0.1B models also do not establish a decode
speedup; launch/host costs are significant there.

Precision observations (33 teacher-forced positions):

| Comparison | Logit MSE | Top-1 matches |
| --- | ---: | ---: |
| 7.2B W8 vs original BF16 PTH | 0.03519 | 33/33 |
| 7.2B W4 G128 vs original BF16 PTH | 2.689 | 26/33 |
| 0.1B W4 G32 vs original BF16 PTH | 2.820 | 24/33 |
| 0.1B W4 G128 vs its CPU-dequantized FP16 reference | 0.001327 | 32/33 |

The last comparison has relative logit RMSE 0.1665% and identical greedy text;
it measures migration arithmetic differences, not loss from reducing precision.
The initial 0.1B G128-to-original comparison had MSE about 19.57, illustrating
that small models can be sensitive to this quantizer. Successful execution
does not imply FP16-equivalent quality. No long-context perplexity or task
quality qualification is claimed for W4. W8 also remains an approximation.

## AMD skill / Magpie evidence

Used the AMD catalog's
[magpie-kernel-evaluator](https://github.com/amd/skills/tree/main/skills/magpie-kernel-evaluator)
workflow, with Magpie checkout `d4de63dffe8df0229a88d4c364442051387ec00e`.
Its compatibility matrix does not list W7900; the results above are local
hardware validation. Magpie is available from `/tmp/rwkv-magpie-quant-migration`
in this workspace session; it is not a runtime dependency.

```sh
PYTHONPATH=/tmp/rwkv-magpie-quant-migration python -m Magpie analyze \
  -k hip/quantized-magpie.yaml --no-perf \
  --output-dir build-hip/quant-validation/magpie
PYTHONPATH=/tmp/rwkv-magpie-quant-migration python -m Magpie --workers 1 compare \
  -k hip/quantized-compare-magpie.yaml \
  --output-dir build-hip/quant-validation/magpie
```

Analyze passed compilation and the numerical testcase. Compare passed both
scalar and WMMA correctness gates and executed custom HIP-event timing.
Magpie's custom profiler exposes only a completion flag in JSON and omits the
timing stdout; its resulting `winner: 0` is **not a performance ranking**.
Use the per-variant `timings.csv` artifacts. No TraceLens/torch report is claimed
for this native C++ backend.

The standalone HIP-event comparison uses seed 42, three warmups, 20 timed
calls and the same automatic split workspace. Example timings (microseconds):

| Bits | M | K | N | Scalar tiled | WMMA tiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | 4096 | 4096 | 144.80 | 137.87 |
| 4 | 64 | 4096 | 4096 | 671.94 | 513.72 |
| 8 | 16 | 4096 | 4096 | 153.70 | 164.64 |
| 8 | 64 | 4096 | 4096 | 655.74 | 638.17 |

WMMA is not uniformly faster for every W8 shape; further shape-specific tuning
is possible. M1/M4 share the vectorized implementation in both variants.
These kernel times are separate from the full-model timing table.

A separate `rocprofv3 --kernel-trace --stats` run succeeded and saved kernel
trace/statistics CSVs under `build-hip/quant-validation/rocprof/`. In that mixed
shape microbenchmark, W8 GEMM accounted for 49.08% and W4 GEMM 42.84% of traced
kernel time. This identifies matrix multiplication as the remaining main cost
in that workload; it is not an end-to-end or roofline report. Profiled event
timings are not used for the unprofiled comparison above.

Artifacts are retained under `build-hip/quant-validation/`: quantized models,
CPU-dequantized reference, export logs, real-model logs, final CTest log,
standalone validation binaries and Magpie reports. These generated files are
not source-controlled. Reproducible configs, tests and tools are included in
the migration changes.
