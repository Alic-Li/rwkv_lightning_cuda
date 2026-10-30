# RWKV-7 状态微调

[English](README.md) | 简体中文

`rwkv_state_tune` 和 `rwkv_miss_tune` 使用独立的
[BF16 训练 backbone](../training/README.md)，不链接推理核心。
本目录只保留共享 JSONL 数据读取器和状态微调使用说明。
旧 FP16 训练 sidecar 及其推理 backend 桥接接口已移除。
MiSS 的 FP16 表示仍用于在线推理，训练使用 BF16；主参数与优化器张量使用 FP32。

运行时状态 ABI 为 `[H,K,V]`；PyTorch 状态文件使用 `[H,V,K]`，因此加载和检查点序列化时
需要转置最后两个维度。推理加载器已在加载时完成此转换。

## 反向计算记录的所有权

`BlockTapeView` 不拥有内存。调用方为一次短上下文前向/反向计算分配其字段。
`wkv_tape_elements` 返回 WKV 计算记录所需的元素数。块级临时工作区可跨层复用。

`v_first_grad` 是各块共享的模型级缓冲区。反向计算累加后续层的值残差梯度，并在第 0 层使用。

## 执行条件与限制

- 支持 CUDA 或 ROCm/HIP，注意力头大小为 64。
- 使用 BF16 计算，WKV 状态、主参数和优化器张量为 FP32。
- 不支持 W8A16/INT8 训练路径。
- 样本只按 `--ctx` 截断；`--chunk` 控制激活内存。前向计算在分块边界保存 WKV、注意力移位和 FFN 移位状态；反向按逆序重算各分块，并跨边界传递这三类状态梯度，不截断循环计算图。
- 复用一份分块激活记录。边界检查点目前保留在 GPU 上，显存占用随分块数量增长。
- `--batch-size N`（别名 `--batch`）累积 N 个独立样本后执行一次优化器更新。损失和梯度按批次中有效 token 的总数加权。
- `--micro-bsz N` 让其中 N 个样本不加 padding 一起计算：每个分块步把各样本接下来的 token 首尾相接地拼在一起，已经结束的样本不占行。激活内存随 `N * chunk` 增长。`rwkv_miss_tune`、`rwkv_nora_tune` 和在线 RL 服务器同样支持。
- `--gemm-mode bf16`（默认）为每种 GEMM 形状固定一个 kernel，并使用 FP32 归约，每一行的结果与同一次 GEMM 里有多少行无关：`--micro-bsz` 和 `--chunk` 不会改变前向的任何一位。`strict` 使用 pedantic FP32 累加；`auto` 恢复以前由 cuBLAS 每次自选 kernel 的行为（可能使用降精度的 split-K 归约）。
- 若损失或 `dState` 出现非有限值，训练会在优化器执行前停止，保留最后一次有效的优化器状态。进度条显示更新次数、损失、tokens/s 和预计剩余时间。

## 独立命令行工具

两种 GPU 后端都会生成 `rwkv_state_tune`。它以流式方式读取仅包含一个 `text` 字段（或下文的
掩码 `segments`）的 JSONL 记录，分词后计算因果下一 token 损失，并在每个批次后执行一次仅更新状态的优化：

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

### 掩码训练（只训练 AI 回复）

数据行也可以写成 `segments`。所有片段都会送入模型，但只有 `"train": true` 片段中的
token 作为 loss 目标：

```json
{"segments":[{"text":"User: 你好\n\nAssistant:","train":false},{"text":" 你好！有什么可以帮你？\n\n","train":true}]}
```

- 每个片段必须恰好包含一个字符串 `text` 和一个布尔 `train`；`{"text":"..."}` 行仍然
  训练全部 token。
- 各片段分别分词后拼接，因此应在自然的 token 边界切分，例如紧跟在 `Assistant:` 之后。
- loss 和梯度按批次内参与训练的 token 数取平均。不训练的前缀仍会做前向和反向，因为
  初始 state 的梯度要经过它们回传；最后一个训练 token 之后的部分会被丢弃。`--ctx`
  范围内没有训练 token 的行会被跳过。
- 进度行中 `tokens=` 是送入模型的位置数（用于计算 tok/s），`trained=` 是 loss 目标数。
- `rwkv_miss_tune` 读取相同格式。

状态导出只包含 BF16 `blocks.N.att.time_state` 张量，采用 PyTorch `[H,V,K]` 布局。
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
状态重放 WKV 递推，写入同一份共享记录。共享记录每 16 行只保存一个状态检查点，WKV
反向用与前向相同的运算重算中间各行，梯度与完整记录逐比特一致。该选项不会重放线性层和
FFN。所有层在同一流上顺序执行，分块边界的梯度保持连通。

设层数为 L、头数为 H、`T=min(ctx,chunk)`，WKV 记录相关存储由
`4*L*H*T*64*64` 字节变为 `4*H*64*64*(ceil(T/16)+16+L+1)` 字节，包含共享检查点记录
及其 16 个状态的反向暂存区、每层初始状态和共享最终状态暂存区。每个块反向前增加一次 WKV 前向递推，以降低显存占用。
其他激活、权重和 GPU 分块边界检查点不变；分块很小或只有一层时可能没有收益。

```bash
./build/rwkv_state_tune --model /path/to/model.pth \
  --data /path/to/train.jsonl --output ./state_output \
  --ctx 2048 --chunk 256 --optimizer adam --wkv_tape
```
