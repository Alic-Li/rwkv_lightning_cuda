# W8A16 与 SB6 W4/W6A16 量化

[English](README.md) | 简体中文

当前只保留原始 **W8A16**、**SB6 W4**、**SB6 W6** 三种量化格式，FP16 用作未量化推理基线。
统一速度和精度结果见[验证报告](VALIDATION.zh-CN.md)。

| 格式 / 命令参数 | CUDA | HIP | 对齐量化矩阵磁盘 / 显存 bpw |
|---|---|---|---|
| W8A16 / `w8a16` | 支持 | 支持 | `8 + 16/K` / `8 + 16/K` |
| SB6 W4 / `w4a16_asym` | 支持 | 不支持 | 4.5 / 5 |
| SB6 W6 / `w6a16_asym` | 支持 | 不支持 | 6.5 / 7 |

bpw 不包含浮点张量、文件头和尾部补齐。SB6 加载时将紧凑元数据展开为每 32 个权重一个
FP16 `(scale, offset)` 对。Embedding、归一化、低秩参数保持浮点；SB6 W4 默认额外将
`blocks.0.att.output.weight` 保留为 BF16，SB6 W6 默认量化该矩阵。

## 导出与运行

```bash
# 原始 W8：量化网格和舍入方式不变，无需校准。
build/rwkv_quantize model.pth model.w8a16.rwkvq

# SB6：先收集激活重要性，再搜索参数并打包。
python tools/calibration/collect_activations.py \
  --model model.pth --corpus calibration.txt --windows 16 --seq-len 256 \
  --output artifacts/quantization/data/moments.pth
python tools/calibration/export_calibrated.py \
  --model model.pth --activation-stats artifacts/quantization/data/moments.pth \
  --default w4a16_asym --output-dir artifacts/quantization/export/w4 \
  --export model.sb6_w4.rwkvq
# W6 改用 --default w6a16_asym，并使用独立输出路径。

build/rwkv_lighting_cuda --model-path model.sb6_w4.rwkvq \
  --vocab-path assets/rwkv_vocab_v20230424.txt
```

运行时自动识别格式，权重保持压缩。W8 保留原有调优缓存；SB6 按形状选择内核。
`.rwkvq.source.json` 应与模型放在一起，以便复用来源模型的适配器。
Launcher 同样只提供这三种格式，SB6 需要填写离线生成的 `manifest.json` 路径。
详细流程见[校准工具说明](../tools/calibration/README.md)，编码见 [SB6 格式](SB6.md)。

## 兼容性与目录

本项目容器魔数保持 `RWKVQ\0\1\0`，dtype 0/1 为 BF16/FP16，2 为 W8，4/6 为 SB6 W4/W6。
旧 dtype 3、5 已退役，读取时明确拒绝，编号不复用。旧模型需从来源 PTH 重新导出。
旧对称 W4 G32/G128、SB6 W5、Autopick、GPTQ 入口均已删除。已有 W8 文件继续兼容。
HIP 仅保留 W8 量化推理，参见 [HIP 支持说明](../hip/QUANTIZATION.zh-CN.md)。

`artifacts/quantization/` 不进入 Git：`models/` 只保存三个最终模型及身份/策略文件，
`data/` 保存可复用的 AW 统计与独立评测语料，`validation/` 保存本轮原始测量。
旧实验模型、中间 payload、日志及重复报告已清理。

```bash
ctest --test-dir build -R 'rwkv_(quantized_archive|quantized_model|sb6_kernels|w8a16_kernels)_test' --output-on-failure
python test/kernels/calibration_test.py
python test/kernels/sb6_calibration_test.py
python tools/validate_quantization.py config.json --output artifacts/quantization/validation
```

配置 JSON 包含六个文件路径：`source`、`w8a16`、`sb6_w4`、`sb6_w6`、`english`、`multilingual`。
验证工具顺序执行三轮性能测试，以及共享同一 FP16 logits 的 4K/16K 精度对比；
第二遍 FP16 衡量运行波动。最终指标保存在 [results.json](validation/results.json)，
唯一维护的测量总结为[验证报告](VALIDATION.zh-CN.md)。
