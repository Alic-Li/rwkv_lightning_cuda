# W8A16 and SB6 W4/W6A16

English | [简体中文](README.zh-CN.md)

The supported quantized formats are original per-row **W8A16**, asymmetric
**SB6 W4** and **SB6 W6**. FP16 remains the unquantized inference baseline.
See the [unified performance and accuracy report](VALIDATION.zh-CN.md).

| Format / CLI name | CUDA | HIP | Aligned quantized weights: disk / device bpw |
|---|---|---|---|
| W8A16 / `w8a16` | Yes | Yes | `8 + 16/K` / `8 + 16/K` |
| SB6 W4 / `w4a16_asym` | Yes | No | 4.5 / 5 |
| SB6 W6 / `w6a16_asym` | Yes | No | 6.5 / 7 |

These bit counts exclude floating tensors, headers and padding. SB6 expands its
compact metadata to one FP16 `(scale, offset)` pair per 32 weights on load.
Embedding, norms and low-rank tensors remain floating. SB6 W4 also protects
`blocks.0.att.output.weight` as BF16 by default; SB6 W6 quantizes this projection.

## Export and inference

```bash
# Original W8 algorithm and rounding, no calibration required:
build/rwkv_quantize model.pth model.w8a16.rwkvq

# SB6: collect activation importance, then search and pack parameters.
python tools/calibration/collect_activations.py \
  --model model.pth --corpus calibration.txt --windows 16 --seq-len 256 \
  --output artifacts/quantization/data/moments.pth
python tools/calibration/export_calibrated.py \
  --model model.pth --activation-stats artifacts/quantization/data/moments.pth \
  --default w4a16_asym --output-dir artifacts/quantization/export/w4 \
  --export model.sb6_w4.rwkvq
# For SB6 W6 use --default w6a16_asym and separate output paths.

build/rwkv_lighting_cuda --model-path model.sb6_w4.rwkvq \
  --vocab-path assets/rwkv_vocab_v20230424.txt
```

The backend detects the archive format automatically and retains packed weights
on device. CUDA W8 uses its existing tuning cache; SB6 uses deterministic shape
selection. HIP accepts W8 only; see [HIP support](../hip/QUANTIZATION.md).
Keep the generated `.rwkvq.source.json` beside each model for adapter identity.
The Launcher exposes the three formats; SB6 export requires an already prepared
`manifest.json`. Offline calibration is described in the
[calibration guide](../tools/calibration/README.md).

## Format and compatibility

The native container magic remains `RWKVQ\0\1\0`. Dtypes 0/1 are BF16/FP16,
2 is W8, 4 is SB6 W4, and 6 is SB6 W6. Retired dtype IDs 3 and 5 are explicitly
rejected and never reused. Export old archives again from the source checkpoint.
The legacy signed W4 G32/G128, SB6 W5, Autopick and GPTQ workflows have been
removed. Existing W8 archives and their quantization grid remain unchanged.

[SB6 layout and kernels](SB6.md) describes bit planes, metadata and rounding.
This project uses its own archive format; external RWKVQ files are not compatible.

## Validation and local artifacts

```bash
ctest --test-dir build -R 'rwkv_(quantized_archive|quantized_model|sb6_kernels|w8a16_kernels)_test' --output-on-failure
python test/kernels/calibration_test.py
python test/kernels/sb6_calibration_test.py
python tools/validate_quantization.py config.json --output artifacts/quantization/validation
```

The validation config contains file paths for `source`, `w8a16`, `sb6_w4`,
`sb6_w6`, `english` and `multilingual`. It compares all candidates against a
single FP16 logit buffer, including a second FP16 pass to measure numerical
noise, and runs three sequential performance rounds. Raw logs and data stay in
the ignored `artifacts/quantization/` directory. Its `models/` contains only the
three final archives plus identity/policy files; `data/` holds reusable AW
statistics and separate evaluation corpora. Published metrics are in
[validation/results.json](validation/results.json); the report is the single
maintained measurement summary.
