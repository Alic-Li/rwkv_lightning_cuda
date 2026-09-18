# HIP 后端

[English](README.md) | 简体中文

本目录包含推理后端及 GPU 算子的 ROCm/HIP 移植实现。配置方式如下：

```sh
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```

默认后端仍为 CUDA。HIP 源码独立存放，便于针对后端调优算子，而无需修改 CUDA 实现。

移植中的 warp 算子目前已在 `gfx1100` 原生 32 通道 wavefront 模式下验证；
其他 AMD 架构仍需正确性测试。

## 状态微调

`RWKV7_STATE_TUNING=ON`（默认）也会在 HIP 下构建 `rwkv_state_tune`。
HIP 实现包括 `rwkv7_statepassing_clampw.hip`、`backward_kernels.hip`、
`block_forward.hip`、`block_backward.hip`、`model_forward.hip` 和 `training_kernels.hip`。
它们复用与后端无关的主机端代码，完成模型反向编排、计算记录大小计算和检查点写入。
新增归约算子显式使用 32 通道 shuffle 子组；现有推理算子在 gfx1100 之外仍需逐架构验证。

```sh
./build-hip/rwkv_state_tune \
  --model /mnt/SDD_1/rwkv7_model_weights/rwkv7-g1i-7.2b-20260805-ctx16384.pth \
  --data ./html_vibe_rwkv_clean.jsonl \
  --output ./build-hip/state-smoke-7b \
  --chunk-load --ctx 64 --chunk 32 --max-steps 2 --save-every 1
```

`--chunk-load` 限制加载权重时的临时缓冲区，运行时权重仍全部驻留 GPU。
`--chunk` 则控制激活重算，并跨分块边界传递循环梯度。
检查点包含 FP32 状态张量，使用与推理兼容的 PyTorch 布局。

验证包括 WKV 导数的 CPU 有限差分、分块边界伴随梯度、LayerNorm 和交叉熵梯度、
批次梯度归约，以及 Adam 动量/更新。测试沿用历史名称 `rwkv_state_tuning_cuda_test`，
但在 HIP 构建中执行 HIP 算子。
短训练成功不代表完整模型的 PyTorch 梯度对齐或长期收敛已得到验证。
BF16 WKV 输入/输出已编译，数值回归使用 FP16 输入/输出，这也是模型运行时训练格式。

HIP 推理目前支持 W4A16/W8A16 `.rwkvq` 文件和 BF16 PTH 权重。
量化投影权重在设备上保持压缩；gfx1100 使用向量化 GEMV 和原生 RDNA3 WMMA，
以 FP32 累加。格式、测试、命令和实测限制见 [量化推理与 W7900 验证](QUANTIZATION.zh-CN.md)。
状态微调仍要求冻结的 FP16/BF16 权重，不接受量化投影。
