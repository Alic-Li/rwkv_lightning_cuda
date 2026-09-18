# MiSS 适配器训练与推理服务

[English](README.md) | 简体中文

FP16 基模保持冻结，只有 `D[out, rank]` 接收参数梯度。本实现采用
[论文](https://arxiv.org/pdf/2409.15371) 和 [参考项目](https://github.com/Joluck/MiSS)
中的高效 MiSS 形式，不存储或训练 A。

- [数学形式与注入位置](#数学形式与注入位置)
- [构建与快速开始](#构建与快速开始)
- [训练选项](#训练选项)
- [检查点与续训](#检查点与续训)
- [推理包](#推理包)
- [HTTP API 与缓存](#http-api-与缓存)
- [性能分析](#性能分析)
- [验证与当前限制](#验证与当前限制)

大模型实测基线和分阶段性能优化工作记录在 [优化计划](OPTIMIZATION_PLAN.zh-CN.md) 中。

## 数学形式与注入位置

输入宽度为 K 时，`S[t,j] = sum(X[t,k] for k < K if k % rank == j)`。
最后一个不完整块隐式补零，rank > K 时也如此。
`Y = base_linear(X) + scale * S @ D.T`，其中 `scale = alpha / rank`。
支持秩 1–1024，每次调用最多 65535 行。

前向和 dX 使用 FP16 D；D 的主副本、累积 dD 和 Adam 动量使用 FP32。
归约与投影均以 FP32 累加。在一次 Adam 更新前，dD 累积所有分块和样本的梯度；
不分配或计算基模 dW。反向向冻结基模的 dX 加上 `scale * (G @ D)[..., k % rank]`。

目标使用逗号分隔的原始权重名称，也可填写 `all`：

- `att.receptance.weight`、`att.key.weight`、`att.value.weight`、`att.output.weight`
- `ffn.key.weight`、`ffn.value.weight`

注意力 key/value 增量在 key 门控和值残差之前加入；第 0 层将适配后的 value 写入 `v_first`。
FFN key 增量在 ReLU² 前加入，FFN value 使用 ReLU² 的输出。推理时，仅当 FFN value 投影
带有适配器时才强制走稠密路径，以便显式取得其输入。

训练复用精确的状态传递 WKV 算子及现有注意力移位、FFN 移位、`v_first` 和跨分块伴随梯度。
归约后的输入（每个目标有 行数 × rank 个 FP32 值）保存在分块计算记录中，反向遍历时随
分块重算。MiSS 无需持久保存完整输入矩阵。`--wkv_tape` 还可启用现有的共享 WKV 重放记录。

## 构建与快速开始

当 `RWKV7_STATE_TUNING=ON`（默认）时，CMake 为 CUDA 和 HIP 构建 `rwkv_miss_tune`。
清单与 SHA-256 身份标识需要 JsonCpp 和 OpenSSL。只有独立验证脚本需要 Python/PyTorch。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 6
./build/rwkv_miss_tune --model model.pth --data train.jsonl --output miss_output \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --rank 16 --alpha 16 --targets all --ctx 2048 --chunk 256 --batch-size 2 \
  --epochs 1 --lr 0.001 --lr-final 0.0001 --warmup-steps 10 --save-every 100
```

每条 JSONL 记录必须为 `{"text":"..."}`。路径相对于 shell 当前工作目录解析。
以下是接近生产规模的示例：

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

### 训练选项

| 选项 | 含义 |
|---|---|
| `--model`、`--data`、`--output`、`--vocab` | 基模检查点、JSONL 数据集、新输出目录、分词器词表 |
| `--rank` | MiSS 秩，默认 16；尾部输入列采用隐式补零 |
| `--alpha` | 适配器 alpha，默认等于 rank，因此默认有效 scale 为 1 |
| `--targets` | `all`，或六种支持的权重后缀中以逗号分隔的子集 |
| `--state` | 可选的冻结初始 WKV 状态；注意力和 FFN 移位状态仍从零开始 |
| `--lr`、`--lr-final`、`--warmup-steps` | Adam 学习率调度 |
| `--ctx`、`--chunk` | 每个样本的最大 token 数、状态传递分块长度 |
| `--batch-size` | 一次优化器更新前累积的独立样本数 |
| `--epochs`、`--max-steps` | 数据集遍历轮数、可选的优化器更新次数上限 |
| `--save-every`、`--resume` | 可续训检查点的保存间隔、续训检查点目录 |
| `--wkv_tape` | 各层共享 WKV 重放记录，以额外 WKV 重算换取更低显存占用 |

默认值为 `epochs=1`、`rank=16`、`alpha=rank`、`ctx=128`、`chunk=64`、`batch-size=1`。
步数指优化器更新次数，不是分块数。学习率调度总步数由轮数和数据集大小确定；
中断运行时若只修改 `--max-steps`，该调度总步数保持不变。
目前批次内的样本按顺序执行，按有效 token 数加权累积梯度，并非 GPU 并行批次，
因此可能限制大型 GPU 的利用率。

### 检查点与续训

检查点目录为 `checkpoint-N/{checkpoint.json,training.pth}`，包含 FP32 D 主副本、dD、
Adam 动量、初始状态、优化器步数、调度总步数、数据集轮次/行号、配置指纹和 RNG 状态。
数据顺序和零初始化具有确定性；当前训练在初始化后不使用随机操作。
检查点在已清空累积梯度的更新边界发布。

续训时保持相同训练配置，添加 `--resume checkpoint-N` 并使用新的输出目录。
程序会检查模型、数据、词表、初始状态的指纹以及调度设置。
成功完成后导出单文件 `output/adapter-final.pth`，不会覆盖已有文件或检查点。
路径相对于启动命令时的工作目录解析。

## 推理包

最终 PTH 包含 FP16 D 和文件内部的 `archive/miss.json` 元数据，仍可用
`torch.load` 读取。新 checkpoint 的 `training.pth` 也内嵌推理所需元数据，
可以独立上传或注册；程序只提取 `.D.master` 并转为 FP16 缓存，
梯度和 Adam 张量不进入 adapter RAM/GPU 缓存。续训仍读取完整 checkpoint 目录。

旧 `training.pth` 注册时读取同目录的 `checkpoint.json`；上传旧文件时需同时
上传该 JSON。旧文件若仅训练 FFN value，可能缺少输入宽度信息，需要重新导出。
原来的双文件推理目录仍兼容：

- `adapter.json`：`format_version=1`、`kind=inference_adapter`、`method=miss`、`dtype=float16`、`layout=modulo_rank_zero_pad`、来源基模 SHA-256、rank、alpha、scale、目标名称、层号、输入宽度、D 形状，以及 `content_version`（不含版本字段的规范化清单与 FP16 D 的 SHA-256）。
- `adapter.pth`：PyTorch 可读取的 FP16 张量，名称如 `blocks.N.att.key.weight.D`，使用原始 `[out,rank]` 布局。

加载器检查格式、形状、重复目标、有限值和内容哈希，
来源基模指纹必须匹配。加载模型时、请求可以绑定适配器之前，会一次性计算基模指纹，
因此启动时增加一次顺序读取模型文件计算哈希的过程。
`rwkv_quantize` 写入 `output.rwkvq.source.json`，将量化文件哈希绑定到来源 PTH；
将此文件与量化模型放在一起，即可复用基于其 FP16 运行时来源训练的适配器。
旧量化模型需重新量化以生成此来源文件。支持量化基模**推理**，训练仍使用 BF16 文件
加载后得到的未量化 FP16 运行时表示。

## HTTP API 与缓存

注册、列表、删除、鉴权和生成请求示例统一维护在
[HTTP API 中文文档](../../docs/http-api.zh-CN.md#miss-适配器注册列表与删除)。
注册支持服务端 PTH 路径、旧推理目录，或在 `POST /v1/adapters` 直接 multipart 上传。
删除注册后，已有请求句柄仍有效，新请求无法查找已删除版本。

GPU 缓存未命中时，使用锁页暂存内存和非阻塞复制流，一次性上传整段连续的 D 数据。
就绪事件完成后才发布可用对象。准入和上传在缓存互斥锁下串行执行，合并同版本的并发
未命中；目前不同适配器的上传也串行执行。GPU 命中后，各层、预填充和解码复用同一分配。
活跃租约阻止驱逐，无租约的分配使用 LRU。暂停释放适配器 GPU 租约并清理无租约分配，
恢复时从 RAM 重新加载。RAM 统计包含已注册包和仍被活跃句柄持有的已删除版本。

以下独立的进程环境变量设置预算，单位为 MiB：

```bash
RWKV_ADAPTER_RAM_MIB=1024
RWKV_ADAPTER_GPU_MIB=512
RWKV_ADAPTER_STAGING_MIB=128
```

适配器整体必须能放入暂存预算。超过 RAM、暂存、GPU 预算、活跃租约容量或 GPU 可用
内存的包会被拒绝。这些限制只计算 D 数据，不包含 JSON/容器开销和基模内存。
当前运行时假设每个服务进程只使用一个 GPU 设备。

有效状态身份包含唯一的已加载模型运行实例、适配器内容版本、有效 scale、初始状态身份
和 WKV 精度。状态复制与续推会拒绝不匹配的身份。会话缓存键按命名空间隔离，持久化
状态保留有效身份，不能跨不同模型运行实例复用。本仓库目前没有 radix cache；将来若
添加，也必须使用相同的有效身份。共享 `LayerWeights` 始终不被修改。

`GET /v1/adapters` 返回 RAM/GPU 命中、未命中、上传次数、H2D 毫秒数、驻留字节数和
适配器 GPU 峰值字节数。生成过程输出 JSON `miss_request_metrics` 记录，包含冷/热缓存
状态、首 token 延迟（TTFT）、相邻采样间的每 token 延迟（TPOT）以及采样得到的设备显存
峰值。设备显存包含其他进程，占用峰值是采样高水位，并非精确的逐请求分配器峰值。

## 性能分析

`nvidia-smi` 的 GPU 利用率反映设备是否忙碌，不是 FLOP 利用率百分比。
Hopper 及更新 GPU 可通过 `dmon` 提供更有参考价值的 GPM 计数器：

```bash
nvidia-smi dmon -i 0 \
  --gpm-metrics 2,3,5,7,10,13,249,250,252,260 \
  --gpm-options d -d 1 --format csv -o DT \
  -f miss-training-gpm.csv
```

关键百分比列为 SM 活跃度、SM 占用率、Tensor/HMMA 活跃度、DRAM 活跃度和 FP16 活跃度。
应同时观察每次优化器更新的 token 吞吐量、功耗和显存。用 Nsight Systems 查找启动间隙
和主要耗时算子。只有驱动允许访问 GPU 性能计数器时，Nsight Compute 才能测出算子级
实际吞吐量，否则会报 `ERR_NVGPUCTRPERM`。

## 验证与当前限制

```bash
ctest --test-dir build --output-on-failure
python tools/check_miss.py --build build --work /tmp/miss-acceptance
python tools/check_miss_http.py --build build --work /tmp/miss-acceptance
```

工作目录必须是新目录。测试覆盖显式 A/autograd dD/dX、有限差分、不完整分块、梯度累积、
两层完整模型 autograd、完整序列与分块梯度、顺序批次累积、共享 WKV 记录、损失下降、
逐位一致的检查点续训/导出、无适配器与零 scale 输出、并发版本/scale、SQLite 身份往返、
GPU 准入/LRU/重载、解码不重复上传，以及动态 FP16/W8A16/W4A16 推理。

在 RTX PRO 6000 Blackwell 上，两层测试的损失在 16 次更新后由 5.2124 降至 2.9414。
CUDA 数值和运行时测试通过。HIP 使用同一 MiSS 算子源码和等效钩子，但
**尚未在本验证环境中构建或在 AMD 硬件上运行**。

解码的归约、投影和加法融合为一个无全局工作区的算子。小规模训练投影也在该算子中
保存 S；较大的训练投影将归约与投影分开，避免重复归约输入。
反向在短分块时使用每参数线程，较长、较宽的投影使用共享内存 16×16 G/S 分块，
进行确定性归约，不生成完整 A 或 delta-W。这些算子以正确性为先，尚未实现大规模
预填充/dD 的 Tensor Core 分块。沿用的顺序微批次/WKV 路径也限制了利用率。

已采集 Nsight Systems 训练追踪。本机 Nsight Compute 报 `ERR_NVGPUCTRPERM`，
硬件性能计数器访问需管理员配置驱动。目前不声称 GPU FLOP 利用率已饱和或 HIP 性能
已对齐，大模型吞吐优化和 AMD 执行仍待验收。

实测误差、微基准结果和尚未完成的验收范围详见 [验证记录](VALIDATION.zh-CN.md)。
