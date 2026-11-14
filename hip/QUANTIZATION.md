# HIP W8A16 quantization support

English | [简体中文](QUANTIZATION.zh-CN.md)

HIP retains original W8A16 `.rwkvq` and floating PTH inference. SB6 W4/W6 is
CUDA-only and explicitly rejected by the HIP loader. Legacy symmetric W4 has
been removed. The current [unified report](../quant/VALIDATION.zh-CN.md) measures
NVIDIA CUDA only and makes no AMD performance claim.

W8 keeps raw NK/KN bytes and per-output-channel FP16 scales, vectorized GEMV,
gfx1100 RDNA3 WMMA and FP32 accumulation. Quantized FFN value remains NK and
uses dense projection after ReLU squared. CUDA PackedNK and tuning caches are
not used. State tuning continues to require the floating training backbone.

```bash
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
build-hip/rwkv_quantize model.pth model.w8a16.rwkvq
build-hip/rwkv_quantized_validate model.w8a16.rwkvq assets/rwkv_vocab_v20230424.txt
ctest --test-dir build-hip -R 'w8a16|quantized_archive|quantized_model' --output-on-failure

# Standalone scalar and WMMA reference checks:
bash tools/check_quantized_hip.sh scalar build
bash tools/check_quantized_hip.sh scalar test
bash tools/check_quantized_hip.sh wmma build
bash tools/check_quantized_hip.sh wmma test
```

The cleanup environment has no ROCm compiler or AMD GPU. These are reproduction
commands; a post-cleanup HIP build/run has not been verified.
