# MiSS validation record — 2026-09-18

English | [简体中文](VALIDATION.zh-CN.md)

Environment: RTX PRO 6000 Blackwell Workstation Edition, CUDA architecture
sm_120, NVIDIA driver 610.57.04. Release CMake build. Python oracle: local
PyTorch 2.14.0+cu130. ROCm SDK and AMD device were unavailable.

## Correctness

- CTest: 15 passing GPU/host tests. The fixture-dependent model executable
  skips its argument-free CTest invocation. Running it explicitly with the
  generated fixture passed FP16 prefill/decode, concurrency, and reload.
- `tools/check_miss.py`: explicit fixed-A PyTorch/autograd reference, CPU
  finite differences, divisible and ragged layouts, dD accumulation, two-layer
  full RWKV autograd, chunk=30 versus chunk=7, batch-size=2 accumulation,
  shared WKV replay tape, frozen initial-state serialization.
- Full-model adapter gradient relative L2 error: maximum approximately
  0.115%; acceptance threshold 0.5%. Loss/reference: 5.2124 / 5.212386.
- 16 optimizer updates: loss 5.2124 → 2.9414 on a repeated ASCII fixture.
- Continuous four updates versus two updates plus resume: bitwise identical
  FP32 master D, moments and FP16 inference exports.
- Dynamic FP16, W8A16 and W4A16: prefill/decode, full/chunked prefill,
  concurrent different content versions and scales, zero-scale equivalence,
  incompatible-state rejection, SQLite identity round trip, eviction/reload.
- RAM, GPU and staging admission rejection; single upload for eight concurrent
  same-version misses; active leases prevent eviction; deleting a registration
  preserves existing handles; latest-version fallback follows registration order.
- HTTP registration/list/delete, cold/hot generation, session isolation by
  scale and deletion by logical session ID. Registration uploads=0; first
  generation uploads=1; repeated generation uploads remains 1.

The model is synthetic and small (2 layers, C=64, F=128, vocab=256). This
validates the chain and serialization; it is not a quality benchmark on a
pretrained model or a large-model training stability result.

Production-binary smoke tests also passed on this machine: one state-tuning
update with shared WKV tape and checkpoint export; fresh W8A16 and W4A16
quantization plus MiSS inference on both archives; HTTP adapter registration,
cold/hot generation, listing, and deletion; and an unquantized 7.2B server
request. The 7.2B prompt `The capital of France is` produced a coherent greedy
continuation and exercised server status and token-count endpoints.

## Kernel measurements

`tools/bench_miss.cpp`, K=N=4096, rank=16, 10 warmup iterations and 100 timed
iterations, CUDA events. No base Linear or H2D is included. Forward is fused
for decode, with saved S for training shapes. Backward includes dD and adapter
dX accumulation. Both baseline and optimized standalone binaries used
`nvcc -O3 -std=c++17 -arch=sm_120`.

| Rows | Initial forward µs | Optimized forward µs | Initial backward µs | Optimized backward µs |
|---:|---:|---:|---:|---:|
| 1 | 38.478 | 2.545 | 68.727 | 5.329 |
| 64 | 14.680 | 14.742 | 90.938 | 11.902 |
| 256 | 30.122 | 22.605 | 127.624 | 34.771 |
| 1024 | 61.403 | 50.715 | 817.094 | 126.511 |

The initial kernels used one lane per reduction slot and one CTA per dD
entry. The optimized version distributes reduction across a CTA, uses one
thread per dD entry for short chunks, and shares G/S tiles for longer chunks.
These timings are a single local microbenchmark, subject to clock and desktop
workload variation; they do not establish end-to-end training speedup.

Build and run the maintained benchmark with `./build/rwkv_miss_bench`.

## Profiler status and outstanding acceptance

Nsight Systems 2026.3.2 produced a training CUDA trace. In the initial tiny
training trace, WKV backward consumed 13.5% of kernel time; MiSS projection
averaged about 1.02 µs, dD 1.76 µs, and dX 1.08 µs. Small-projection reduction
and projection were subsequently fused; these initial timings are not claims
about the final fused implementation.

The 7.2B training command documented in `README.md` completed 26 updates before
the profiling run was intentionally stopped. It sustained about 2.14k token/s
and used about 22.0 GiB VRAM. Across 238 stable one-second Blackwell GPM samples,
GPU busy averaged 99.8%, while SM activity averaged 52.0%, occupancy 12.9%,
Tensor/HMMA activity 15.4%, and DRAM activity 20.9%. Average power was 401.6 W.
This confirms that ordinary `nvidia-smi` GPU utilization substantially
overstates arithmetic utilization for the current sequential microbatch path.

Nsight Compute 2026.3 could attach, but reported `ERR_NVGPUCTRPERM` when
profiling kernels. Hardware counter access needs administrator configuration,
so an exact achieved-FLOPS percentage is still unavailable. See
[`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) for the measurement gate,
shape-specific MiSS work, real microbatch plan, and acceptance targets.

HIP shares the portable MiSS kernel source and has corresponding forward and
backward hooks, but HIP compilation and AMD numerical/performance execution
remain unverified. Large-prefill Tensor Core projection/dD, parallel training
microbatches and large pretrained-model throughput tuning also remain future
performance work.
