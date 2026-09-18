# RWKV-7 状态微调模块

[English](README.md) | 简体中文

本目录不是通用训练框架，而是为冻结的 FP16 运行时权重提供 CUDA/HIP 反向计算路径。
BF16 模型文件仍由现有加载器转换为相同的 FP16 运行时表示。所有函数均不计算冻结基模
权重的梯度。可选的 MiSS 视图只为 D 增加 FP32 参数梯度，详见 [MiSS 训练与服务](../miss/README.zh-CN.md)。

## 短上下文执行流程

1. `ModelBackend::state_tuning_model_view` 暴露已加载推理权重的只读视图，并拒绝 INT8 张量。
2. `model_forward_state_tuning_f16` 复用推理用的 LayerNorm、time-mix、线性层、激活、ChannelMix 和 WKV 后处理算子。WKV 使用不依赖 PyTorch 的精确状态传递算子，记录 FP32 `[B,H,T,K,V]` 状态检查点。
3. `cross_entropy_forward_backward_f16` 生成按行平均的标量损失和 FP16 `dLogits`。
4. `model_output_backward_state_only` 计算冻结输出头及最终 LayerNorm 的输入梯度，然后按 `L-1 ... 0` 的顺序遍历各层。
5. `model_backward_state_only` 只使用两个持久化的 `[B,T,C]` 梯度缓冲区。每个块返回 `dx` 和各批次的 FP32 `dState`。
6. `reduce_state_gradient_f32` 归约批次维度，`adam_update_state_f32` 只更新 `time_state`、`dState`、`m` 和 `v`。

运行时状态 ABI 为 `[H,K,V]`；PyTorch 状态文件使用 `[H,V,K]`，因此加载和检查点序列化时
需要转置最后两个维度。推理加载器已在加载时完成此转换。

## 反向计算记录的所有权

`BlockTapeView` 不拥有内存。调用方为一次短上下文前向/反向计算分配其字段。
`wkv_tape_elements` 返回 WKV 计算记录所需的元素数。块级临时工作区可跨层复用。

`v_first_grad` 是各块共享的模型级缓冲区。反向计算累加后续层的值残差梯度，并在第 0 层使用。

## 执行条件与限制

- 支持 CUDA 或 ROCm/HIP，注意力头大小为 64。
- 使用 FP16 计算，WKV 状态和优化器张量为 FP32。
- 不支持 W8A16/INT8 训练路径。
- 样本只按 `--ctx` 截断；`--chunk` 控制激活内存。前向计算在分块边界保存 WKV、注意力移位和 FFN 移位状态；反向按逆序重算各分块，并跨边界传递这三类状态梯度，不截断循环计算图。
- 复用一份分块激活记录。边界检查点目前保留在 GPU 上，显存占用随分块数量增长。
- `--batch-size N`（别名 `--batch`）按顺序累积 N 个独立样本后执行一次优化器更新。支持无需填充的变长样本，但不会在 GPU 上并行执行样本。损失和梯度按批次中有效 token 的总数加权。
- 若损失或 `dState` 出现非有限值，训练会在优化器执行前停止，保留最后一次有效的优化器状态。进度条显示更新次数、损失、tokens/s 和预计剩余时间。

## 独立命令行工具

两种 GPU 后端都会生成 `rwkv_state_tune`。它以流式方式读取仅包含一个 `text` 字段的
JSONL 记录，分词后计算因果下一 token 损失，并在每个批次后执行一次仅更新状态的优化：

```bash
./build/rwkv_state_tune \
  --model /path/to/model.pth \
  --data /path/to/train.jsonl \
  --output ./state_output \
  --ctx 2048 \
  --chunk 1024 \
  --batch-size 2 \
  --epochs 1 \
  --lr 1.0 \
  --lr-final 0.01 \
  --warmup-steps 10 \
  --save-every 500
```

检查点只包含 FP32 `blocks.N.att.time_state` 张量，采用 PyTorch `[H,V,K]` 布局。
写入器使用与推理相同的 PTH 解析器校验文件，然后将其发布为
`state-step-XXXXXXXX.pth` 或 `state-final.pth`。

GPU 回归测试将 WKV 输入/状态导数与 CPU 双精度有限差分对比，并验证分块边界的伴随梯度。
完整模型的 PyTorch 梯度对齐仍需单独验证。

ROCm 构建命令和已测试的训练示例见 [HIP 后端](../../hip/README.zh-CN.md)。
大模型可使用 `--chunk-load` 限制加载权重时的暂存缓冲区大小。
训练结果的 HTTP 上传与使用见 [初始状态文件接口](../../docs/http-api.zh-CN.md#初始状态文件上传列表与删除)。

### 优化器选择

`rwkv_state_tune --optimizer adam` 显式选择现有 Adam 优化器；省略 `--optimizer` 时也使用 Adam。
`--optimizer muon` 对每层、每个注意力头的 64×64 时间状态矩阵分别应用 Muon，其他模型参数保持冻结。

CUDA FP32 实现遵循 [KellerJordan/Muon](https://github.com/KellerJordan/Muon)：
动量为 0.95，启用 Nesterov，执行五次五次多项式 Newton–Schulz 迭代，CLI 不使用权重衰减。
参考实现以 BF16 正交化，本实现则使用 FP32。
两种优化器均使用现有的 `--lr`、`--lr-final` 和 `--warmup-steps` 调度及默认值（1.0、0.01、10）。
Muon 和 Adam 的学习率不能直接互换，比较时应显式设置。例如，可在已有训练命令后添加
`--optimizer muon --lr 0.02 --lr-final 0.002` 作为调参起点。

### 共享 WKV 计算记录

添加 `--wkv_tape` 可启用各层共享的 FP32 WKV 计算记录，与 `--optimizer adam|muon` 独立。
不指定该选项时，沿用原来的逐层记录路径。

前向保存每层的分块初始 WKV 状态，跳过逐 token 状态记录。在每个块反向计算前，从保存的
状态重放 WKV 递推，写入同一份共享记录。该选项不会重放线性层和 FFN。所有层在同一流上
顺序执行，分块边界的梯度保持连通。

设层数为 L、头数为 H、`T=min(ctx,chunk)`，WKV 记录相关存储由
`4*L*H*T*64*64` 字节变为 `4*H*64*64*(T+L+1)` 字节，包含共享记录、每层初始状态和
共享最终状态暂存区。每个块反向前增加一次 WKV 前向递推，以降低显存占用。
其他激活、权重和 GPU 分块边界检查点不变；分块很小或只有一层时可能没有收益。

```bash
./build/rwkv_state_tune --model /path/to/model.pth \
  --data /path/to/train.jsonl --output ./state_output \
  --ctx 2048 --chunk 256 --optimizer adam --wkv_tape
```
