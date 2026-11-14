# SB6 activation-aware calibration

Offline AW and weighted least squares target this project's native archive.
Supported quantized tiers: `w8a16`, `w4a16_asym`, `w6a16_asym`.
The mathematical reference is [RafaelUI/rwkv-quant](https://github.com/RafaelUI/rwkv-quant),
reviewed at `d22874666d5131868579149fcbcda58ff2808ef1`.
Inference has no dependency on that repository or Python.

Requirements: Python 3.10+, PyTorch, NumPy and a built `rwkv_quantize`.
The calibration forward uses FP32 weights/activations and resets recurrent state
per window. A 7.2B model needs roughly 4 bytes per parameter plus temporaries on
the calibration device; the validated machine has 96 GB VRAM. CPU is supported
for small fixtures. Runtime inference instead uses FP16 activations.

## Collect and export

```bash
python tools/calibration/collect_activations.py \
  --model model.pth --corpus calibration.txt --windows 16 --seq-len 256 \
  --batch-size 4 --seed 42 --output artifacts/quantization/data/moments.pth
python tools/calibration/export_calibrated.py \
  --model model.pth --activation-stats artifacts/quantization/data/moments.pth \
  --default w4a16_asym --output-dir artifacts/quantization/export/w4 \
  --export model.sb6_w4.rwkvq
python tools/calibration/export_calibrated.py \
  --model model.pth --activation-stats artifacts/quantization/data/moments.pth \
  --default w6a16_asym --output-dir artifacts/quantization/export/w6 \
  --export model.sb6_w6.rwkvq
# Original per-row W8 needs no activation statistics:
build/rwkv_quantize model.pth model.w8a16.rwkvq
```

Use representative calibration text separate from evaluation. The sampler picks
non-overlapping windows without replacement; insufficient data fails. `--tokens
windows.json` accepts `{"tokens": [[token IDs], ...]}` in explicit order.
Statistics are a flat tensor dictionary `name -> E[x²] [K]`, measured at actual
linear inputs: after mixing, WKV/GroupNorm/gating, ReLU squared or final LayerNorm
as appropriate. Every SB6 target requires matching statistics when supplied.
Without statistics, SB6 runs the same endpoint/LS search with uniform importance.
W8 retains the original per-row grid and halfway-away FP16 rounding.

SB6 searches endpoint candidates, fits scale/offset with weighted least squares,
projects to shared 6-bit metadata and FP16, and scores the resulting codes.
See [the exact layout and search](../../quant/SB6.md). W4 protects
`blocks.0.att.output.weight` as BF16 by default; W6 quantizes all target projections.

## Payloads and explicit policies

`--policy policy.json` optionally overrides exact tensor names with `bf16`,
`fp16`, `w8a16`, `w4a16_asym` or `w6a16_asym`. Unlisted tensors use `--default`.
Embedding/LoRA integer policies and unknown names/types are errors. The exporter
writes complete `policy.json`, `manifest.json`, payload files and `run.json`.
`manifest.json` maps each quantized tensor to its type, shape and relative `data`
and `scales` files. SB6's scales file contains the 16-byte superblock metadata,
not a plain FP16 scale array. Payloads bypass requantization in the C++ writer.

```bash
build/rwkv_quantize --format w4a16_asym \
  --policy artifacts/quantization/export/w4/policy.json \
  --calibrated artifacts/quantization/export/w4/manifest.json \
  model.pth model.sb6_w4.rwkvq
```

For Launcher export, choose the matching format and provide the manifest path.
The Launcher invokes the CPU archive writer; prepare AW payloads separately.
Keep source identity sidecars with the final archives. Keep statistics and resolved
policy for reproducibility; intermediate payloads may be deleted after validation
unless the manifest is needed for another export. Do not mix checkpoints or corpora.

## Accuracy and performance

```bash
python test/kernels/calibration_test.py
python test/kernels/sb6_calibration_test.py
# Optional override for another build: RWKV_QUANTIZER=/path/to/rwkv_quantize

build/rwkv_logit_mse_curve model.pth model.pth assets/rwkv_vocab_v20230424.txt \
  --text heldout.txt --ppl --ppl-seq-len 4096 --ppl-stride 128 \
  --output-prefix artifacts/quantization/validation/english \
  --candidate w8a16=model.w8a16.rwkvq \
  --candidate sb6_w4=model.sb6_w4.rwkvq --candidate sb6_w6=model.sb6_w6.rwkvq
```

All candidates use the identical FP16 logit buffer. The positional source model
runs again to quantify repeat noise. CSV columns retain historical `fp16/int8`
names; the second column is whichever candidate was supplied. PPL counts exactly
`--ppl-seq-len` next-token predictions. Use 16384 explicitly for 16K evaluation;
`--ppl-stride` groups reporting intervals without resetting recurrent state.
RMSE/top-1 summaries should use the same prediction count, excluding the extra
target token's logits. Aggregate chunk PPL by token-weighted mean log probability.

`evaluate.py --model model.pth --archive model.sb6_w4.rwkvq --corpus heldout.txt
--windows 16 --output result.json` is a separate FP32/dequantized reference check,
not a measurement of CUDA inference. Native results are authoritative for runtime.

For the complete sequential three-round benchmark plus 4K/16K comparison, use
`tools/validate_quantization.py`; configuration and final measured results are in
[the unified report](../../quant/VALIDATION.zh-CN.md). Evaluation on two fixed
texts is a regression check, not a general downstream-task accuracy benchmark.
