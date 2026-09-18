# MiSS training optimization plan

## Baseline

The first large-model run used:

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

On an RTX PRO 6000 Blackwell Workstation Edition, 26 optimizer updates completed
without OOM or non-finite loss. After model loading, the run held about 22.0 GiB
of device memory and sustained approximately 2.14k token/s. A 238-second stable
GPM sample reported:

| Metric | Mean | Observed range |
|---|---:|---:|
| GPU busy | 99.8% | 92–100% |
| Power | 401.6 W | 362–427 W |
| SM activity | 52.0% | 49–55% |
| SM occupancy | 12.9% | 12–14% |
| Tensor/HMMA activity | 15.4% | 12–18% |
| DRAM activity | 20.9% | 19–24% |

The ordinary GPU-busy value is therefore misleading: the queue stays busy, but
the device spends relatively few cycles issuing Tensor Core work. A rough lower
bound for frozen 7.2B Linear forward plus input-gradient work is
`4 * 7.2B * 2.14k = 61.6 TFLOP/s`; it excludes WKV, the head, MiSS, elementwise
work, and padding, so it must not be presented as an exact achieved-FLOPS result.
Nsight Compute counters are currently blocked by `ERR_NVGPUCTRPERM`.

## Measurement gate

Before changing kernels, capture one optimizer update after warmup under Nsight
Systems with CUDA, NVTX, cuBLAS, and OS runtime tracing. Add NVTX ranges for
sample, chunk, layer, base Linear forward, base dX, WKV forward/backward, MiSS
reduction, MiSS projection, dD, scatter-dX, and optimizer update. Export these
tables from the same trace:

1. CUDA GPU kernel summary and cuBLAS call summary.
2. Kernel launch gaps and CPU launch-thread utilization.
3. Per-range wall time for forward, backward, replay, and optimizer update.
4. GEMM dimensions, data types, and algorithms for every repeated shape.

Keep the command, commit, driver, clocks, data rows, warmup count, token count,
and raw GPM CSV with each result. Compare median step time over at least ten
post-warmup updates. Do not combine test processes with the measurement window.

## Optimization stages

### 1. Form a real parallel microbatch

`--batch-size 8` currently runs eight samples sequentially and only accumulates
their gradients. This is the largest structural utilization limit. Add a batch
dimension to state, attention/FFN shifts, `v_first`, WKV forward/backward, and
the chunk tape. Pack same-length chunks from several samples into one launch and
one larger Linear GEMM while keeping each sample's recurrent state independent.

Use length buckets and a validity mask for the final ragged chunk. Accumulate
loss and dD with the existing valid-token weighting. Start with a fixed batch of
two, then scale to 4 and 8 after full-sequence versus chunked gradient tests pass.
This should turn repeated `M≈1024` GEMMs into fewer `M≈B*1024` GEMMs and reduce
CPU launch overhead without changing optimizer-step semantics.

### 2. Split MiSS paths by shape

Keep the fused reduction, projection, and add kernel for decode and very short
rows. For training/prefill, reduce `X` to contiguous FP16/BF16 `S` with FP32
accumulation, then use cuBLASLt for `S @ D.T` and `G.T @ S`. Rank 16 is a native
Tensor Core shape; the current custom projection leaves most Tensor capacity
idle. Benchmark the crossover instead of applying one path to every row count.

For dX, fuse `dS = G @ D` with modulo-rank scatter when this wins, but retain a
cuBLASLt-plus-scatter option for large row counts. Accumulate dD into FP32 master
gradients at the end of each microbatch, preserving deterministic accumulation
within the documented tolerance.

### 3. Reduce replay and launch overhead

Use the Nsight ranges to decide whether storing reduced `S`, selected layer
inputs, or WKV intermediates is cheaper than recomputing them. Provide an
explicit memory/performance mode rather than silently increasing the tape.
Capture static full-chunk forward and backward launch sequences in CUDA Graphs
after tensor addresses and adapter targets are stable. Combine FP32 master-to-
FP16 refresh, Adam updates, and gradient clearing with multi-tensor kernels.

### 4. Tune the dominant base and WKV kernels

Do this only after stages 1–3 change the shapes and the new trace identifies the
remaining dominant kernels. Tune cuBLASLt algorithms with a persistent cache for
the actual `(M,N,K,dtype)` set. For WKV, test block size, register pressure,
shared-memory use, state layout, and vectorized loads on SM120. Preserve the
existing state-passing boundary exactly and compare full versus chunked state and
gradient tensors after every layout change.

### 5. HIP parity

Port stable shape dispatch and batching semantics to HIP. Select rocBLAS/hipBLASLt
algorithms independently; do not copy CUDA launch parameters blindly. Run the
same reference, finite-difference, checkpoint, chunk, and state-passing tests on
real AMD hardware before claiming parity.

## Acceptance gates

Each stage must keep all existing correctness checks green and add no base dW.
For the 7.2B command above, record tokens/s, median step time, peak VRAM, power,
SM activity, occupancy, Tensor activity, DRAM activity, and the top ten kernels.
The first practical target is at least 1.5x token/s with unchanged loss and
checkpoint semantics, then at least 70% SM activity and 35% Tensor activity.
Targets should be revised from the first clean Nsight Systems trace if WKV or
other non-Tensor work sets a lower roof.
