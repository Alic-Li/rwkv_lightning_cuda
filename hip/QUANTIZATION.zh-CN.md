# HIP W8A16 量化支持

[English](QUANTIZATION.md) | 简体中文

HIP 保留原始 W8A16 `.rwkvq` 与浮点 PTH 推理。SB6 W4/W6 当前仅支持 CUDA；
HIP 在加载时明确拒绝。旧对称 W4 已移除。当前统一测量仅在 NVIDIA CUDA 上运行，
见[验证报告](../quant/VALIDATION.zh-CN.md)，不作为 AMD 性能结论。

W8 权重使用原始 NK/KN 字节和逐输出通道 FP16 scale，gfx1100 使用向量化 GEMV
与 RDNA3 WMMA，FP32 累加。量化 FFN value 保持 NK，ReLU² 后使用稠密投影。
没有 CUDA PackedNK 或 CUDA 调优缓存。状态微调继续使用浮点训练 backbone。

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

本轮环境没有 ROCm 编译器或 AMD GPU，以上是复验命令，未宣称清理后的 HIP 构建已验证。
