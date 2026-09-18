# HIP 移植验证

[English](VALIDATION.md) | 简体中文

本文记录早期 FP16/状态微调移植。W4A16/W8A16 随后单独完成移植和验证，详见
[量化推理报告](QUANTIZATION.zh-CN.md)。

本地验证环境为 gfx1100、ROCm HIP 7.2.53211-9999。

参考流程：AMD [magpie-kernel-evaluator](https://github.com/amd/skills/tree/6916fb371d1cba40757b223cf16b4cd4912b202f/skills/magpie-kernel-evaluator)。
当时本地未安装 Magpie CLI；验证使用原生 CMake/CTest 测试套件及可执行 CPU 数值参考，
并非 Magpie 性能报告。

## 构建与数值测试

```
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 8
ctest --test-dir build-hip --output-on-failure
```

结果：10/10 测试通过。状态微调检查覆盖 WKV 有限差分、分块状态伴随梯度、LayerNorm、
交叉熵、梯度归约和 Adam。推理测试的合成词表补齐至四的倍数，以满足两个后端现有采样器
的约定。量化 W8 GPU 测试已像 W4 测试一样，正确限制为仅 CUDA 执行。

## 真实数据冒烟测试

所有运行使用 `html_vibe_rwkv_clean.jsonl`，随机种子 1234，执行两次优化器更新。

| 模型 | 上下文 / 分块 | 批大小 | 学习率 | 损失 |
| --- | --- | --- | --- | --- |
| rwkv7-g1d-0.1b-20260129-ctx8192 | 64 / 32 | 2 | 0.1, 0.2 | 3.1931, 7.6466 |
| rwkv7-g1d-0.1b-20260129-ctx8192 | 64 / 64 | 2 | 0.1, 0.2 | 3.1931, 7.6499 |
| rwkv7-g1i-7.2b-20260805-ctx16384 | 64 / 32 | 1 | 0.1, 0.2 | 3.0527, 12.3912 |

7.2B 运行使用 `--chunk-load`。每次运行均生成阶段检查点和 `state-final.pth`，
写入器使用推理 PTH 解析器校验文件布局。输出保留在 `build-hip/state-smoke*`。

这些运行验证执行流程和梯度有限性，不验证收敛。CLI 默认学习率较激进。
不同分块可能选择不同的 FP16 GEMM 执行形状；两次 0.1B 运行的最终状态数值不完全相同
（两次 Adam 更新后最大绝对差 0.48238，RMS 0.02389）。WKV 分块伴随梯度单独通过了
更严格的数值测试。完整模型梯度对齐、BF16 WKV 数值比较，以及 gfx1100 之外的架构
仍未验证。本地未执行 CUDA 算子。

## 释放显存后的扩展验证

所有扩展运行使用恒定学习率 `--lr 0.001 --lr-final 0.001 --warmup-steps 0` 和种子 1234。
日志和检查点推理测试工具保留在 `build-hip/validation/`。

- **7.2B，上下文 1024，分块 128，批大小 2，更新 10 次：** 完成 20,480 个 token，未触发 CLI 的非有限损失/梯度保护。损失依次为 0.7898、0.7837、0.7840、0.7284、0.7556、0.6936、0.6643、0.6037、0.7202、0.6179。各次更新使用不同样本，不是留出集评估。
- **0.1B，重复第一条 JSONL 记录，上下文 128，更新 10 次：** 分块 32 时损失从 3.3215 降至 1.8455；分块 128 时从 3.3206 降至 1.4973。这是重复样本的训练损失，不是泛化指标。最终状态的分块比较：最大绝对差 0.0121814，RMS 0.00330784。完整模型层面的分块等价性仍未证明。
- **检查点复用：** 通过 `ModelBackend::load_state_from_pth` 重新加载 0.1B 重复样本检查点和 7.2B 上下文 1024 检查点，以 FP32 WKV 状态执行四 token 预填充和单 token 解码，检查全部 65,536 个输出 logit 的有限性，两者均通过。

7.2B 复现命令（更换输出目录以保留已有产物）：

```sh
./build-hip/rwkv_state_tune \
  --model /mnt/SDD_1/rwkv7_model_weights/rwkv7-g1i-7.2b-20260805-ctx16384.pth \
  --data ./html_vibe_rwkv_clean.jsonl \
  --output ./build-hip/state-7b-ctx1024 \
  --chunk-load --ctx 1024 --chunk 128 --batch-size 2 --max-steps 10 \
  --lr 0.001 --lr-final 0.001 --warmup-steps 0 --save-every 5
```

扩展的 7.2B 上下文 4096 运行使用 `--ctx 4096 --chunk 256 --batch-size 2 --max-steps 3 --save-every 3`，
以及相同的恒定学习率和数据集，完成 24,576 个 token。损失为 0.5582、0.5339、0.5204。
三次更新及检查点保存均未出现非有限损失/梯度错误。运行中的 ROCm 内存快照显示设备总显存
占用 25,784,668,160 字节（24.0 GiB）；这是快照，不是测得的峰值。
输出为 `build-hip/state-7b-ctx4096/state-final.pth`。
