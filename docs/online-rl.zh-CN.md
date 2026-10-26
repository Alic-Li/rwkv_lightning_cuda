# 常驻 BF16 在线 RL Server

[English](online-rl.md) | 简体中文

`rwkv_miss_tune_server` 更新 MiSS `D`；`rwkv_state_tune_server` 更新初始 `time_state`。
两个 Server 均支持 `--adapter`、`--state` 同时冷启动。State 模式下 Adapter 冻结；MiSS 模式下初始 State 冻结。
原来的 `rwkv_miss_tune`、`rwkv_state_tune` 和 `rwkv_lighting_cuda` 入口保留。

## 构建与启动

依赖沿用项目的 CUDA、Drogon、JsonCpp 等；C++ 服务不依赖 Python/PyTorch。

```bash
cmake -S . -B build -DRWKV7_STATE_TUNING=ON
cmake --build build -j8 --target rwkv_miss_tune_server rwkv_state_tune_server
```

启动 MiSS 服务：

```bash
./build/rwkv_miss_tune_server \
  --model /mnt/nvme0n1/no22/rwkv7-g1k-7.2b-20260930-ctx25600.pth \
  --adapter /mnt/nvme0n1/no22/rwkv_lighting_cuda/train-2-2-g1k/g1k-r16-lr2e-5-e2/adapter-final.pth \
  --state /home/alic-li/rwkv_lightning_cuda/rwkv-state-agentic.pth \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 --port 8848 --chunk 64 --ctx 4096 \
  --max-batch 8 --cache-capacity 8 --keep-versions 2 \
  --output ./online-rl-output/miss
```

训练 State 时使用相同模型、Adapter、State 路径，替换二进制及学习率：

```bash
./build/rwkv_state_tune_server \
  --model /mnt/nvme0n1/no22/rwkv7-g1k-7.2b-20260930-ctx25600.pth \
  --adapter /mnt/nvme0n1/no22/rwkv_lighting_cuda/train-2-2-g1k/g1k-r16-lr2e-5-e2/adapter-final.pth \
  --state /home/alic-li/rwkv_lightning_cuda/rwkv-state-agentic.pth \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 --port 8849 --chunk 64 --ctx 4096 \
  --max-batch 8 --cache-capacity 8 --keep-versions 2 \
  --output ./online-rl-output/state
```

以上命令在源码仓库根目录执行。GitHub Actions 的 Linux/Windows CUDA 12.9、13.2
安装包均包含两个 Server、运行库、词表、`tools/online_rl.py` 和在线 RL 文档。
解压后进入 `rwkv_lighting_cuda/` 目录，将命令中的 `./build/二进制` 改为 `./二进制`，
并将词表改为 `--vocab ./rwkv_vocab_v20230424.txt`；默认词表路径指向构建机器，不能用于发布包。
模型、冷启动 Adapter/State 和数据集需要自行提供，示例中的绝对路径应替换为本机路径。

Windows PowerShell 在解压目录执行下面对应模式的一条命令：

```powershell
.\rwkv_miss_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8848 --chunk 64 --ctx 4096 --max-batch 8 --output .\online-rl-output\miss
.\rwkv_state_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8849 --chunk 64 --ctx 4096 --max-batch 8 --output .\online-rl-output\state
```

选择一种模式启动即可；`--help` 无需加载模型，可检查可执行文件及依赖是否可用。
等待日志出现 `ready`，在另一个终端运行 Python 闭环，关闭客户端不会关闭后端。

不指定 Adapter 时 MiSS 从零 `D` 开始；不指定 State 时使用零初始 State。
冷 Adapter 自动提供 rank、alpha、targets，并校验基模 fingerprint 和维度。
两个 Server 均不再接受启动参数 `--lr`，每次 `POST /v1/train` 必须显式传入 `learning_rate`。
学习率由客户端逐步控制，服务端不提供默认值，也不沿用上一次请求的学习率。
每个 Server 进程仅加载一份 Backbone；同时启动两个进程会各自加载一份。

| 参数 | 默认值与含义 |
|---|---|
| `--host` | `127.0.0.1` |
| `--port` | MiSS `8848`，State `8849` |
| `--chunk` | `64`；计算与反向重算块长，不截断梯度 |
| `--ctx` | `4096`；完整历史 Token 上限，超长请求直接拒绝；对显存的影响见下文 |
| `--max-tokens` | `1024`；每轮生成的防复读硬上限，命中停止串会提前结束；请求可以覆盖 |
| `--stop` | JSON 字符串数组；默认 `["</tool_call>", "</answer>", "\n\nUser:"]`；请求可以覆盖 |
| `--max-batch` | `32`；一次生成或训练的最大样本数；同时决定合批解码的最大行数 |
| `--cache-capacity` | `16`；整个进程共享的 CPU 前缀快照池条数上限，`0` 关闭。所有策略版本合计受限 |
| `--cache-memory-mb` | `4096`；CPU 前缀快照池的张量缓冲区预算，单位 MiB，`0` 关闭。与条数上限取更小值 |
| `--keep-versions` | `2`；当前及旧 Actor 快照总数 |
| `--rank / --alpha / --targets` | 无冷 Adapter 时 `16 / rank / all`；冷启动时推断，显式冲突报错 |
| `--vocab` | 项目的 `assets/rwkv_vocab_v20230424.txt` |
| `--output` | `online-rl-output`；显式发布保存的目录 |
| `--resume` | 无；指向 `checkpoint:true` 发布的目录，恢复优化器状态和 step（见下文） |

### 前缀缓存与系统内存

前缀缓存保存处理完一段 Token 后的完整 WKV State、shift 和下一 Token 的 logits。
后续请求命中同一策略版本的最长前缀时，从该状态继续，只计算新增的 Token。
它不是训练参数或梯度；淘汰后会重新 prefill，不会丢弃训练样本或缩短上下文。

**缓存现在是整个进程共享的固定容量池，不再为每个策略版本单独分配配额。**
`--cache-capacity` 的含义因此从“每版本条数”改为“全局条数”。
条目按命中/写入更新 LRU 顺序，池满时先淘汰最久未使用的前缀，再复用它的张量缓冲区，
不会先额外分配一份完整 State。池按需分配，已腾出的槽位留在池内复用，避免反复分配大块内存。
版本退出 `--keep-versions` 列表时，立即移除该版本的缓存；即使是初始 reference，
也只保留参考权重和初始 State 用于 KL，之后对它做 rollout 不会重新填充缓存。
`--keep-versions` 不再倍增缓存池大小，保留版本仍有各自的 Adapter/初始 State 权重开销。

每个槽位的张量字节数为：

```text
L × H × 64 × 64 × 4 + L × 2 × C × 2 + vocab × 4
有效槽位数 = min(cache-capacity, floor(cache-memory-mb × 1048576 / 每槽字节数))
```

2.9B（32 层、40 头、C=2560、词表 65536）每槽约 20.56 MiB。
例如每批 16 题 × 8 条轨迹，可从以下配置开始，保留约两倍轨迹数的缓存槽位供多轮复用：

```bash
--cache-capacity 256 --cache-memory-mb 6144 --keep-versions 2
```

此时所有版本共享最多 256 槽，张量缓冲区约 5.14 GiB；若预算改为 4096 MiB，
有效槽位数为 199。小预算可能降低命中率、增加 prefill 重算，应结合吞吐调整。
**字节预算限制缓存张量缓冲区，不是进程 RSS 上限**：前缀索引/Token 元数据另外占用少量内存
（受槽位数和 `--ctx` 限制），模型、HTTP 请求、分配器和训练工作区也有各自开销。

`GET /health` 的 `cache` 对象报告 `scope: "global"`、`eviction_policy: "lru"`、
`capacity`、`effective_capacity`、`memory_limit_bytes`、`entry_bytes`、`cached_prefixes`、
`used_bytes`、`allocated_slots`、`allocated_bytes`、`free_slots`、`hits`、`misses`、
`evictions`、`retired_prefixes` 和 `skipped_inserts`。
其中 `used_bytes` 是已索引快照的张量字节数，`allocated_bytes` 包含池内待复用的空闲槽位，
所以清空某个版本后后者不一定下降，但不会超过预算。
`policies[]` 各自提供 `cached_prefixes`、`cache_bytes`、`cache_enabled`；顶层
`reference_cached_prefixes`、`reference_cache_bytes`、`reference_cache_enabled` 单独报告 reference。
reference 仍在普通版本列表时，这些是同一份缓存的统计，不能再次相加。
退出列表后 reference 缓存恒为零，KL 打分仍然可用。

### 显存估算

下表按分配代码估算，**未实测**，只用于选参数；实际值请以 `nvidia-smi` 为准，并预留约 1 GiB 给 CUDA 上下文和 cuBLAS。
2.9B 按 32 层、C=2560（40 头）计算，7.2B 按 32 层、C=4096（64 头）计算。

| 组成 | 公式 | 2.9B | 7.2B |
|---|---|---|---|
| BF16 Backbone | 约 2 字节/参数 | ≈5.8 GB | ≈14.4 GB |
| 训练 Chunk 边界检查点 | `ceil((最长样本-1)/chunk) × (L·H·16 KiB + L·2·C·2 B)` | 每块 ≈20.3 MiB | 每块 ≈32.5 MiB |
| 　ctx 4096、chunk 64（64 块） | | ≈1.3 GiB | ≈2.0 GiB |
| 　ctx 8192、chunk 64（128 块） | | ≈2.5 GiB | ≈4.1 GiB |
| 　ctx 8192、chunk 128（64 块） | | ≈1.3 GiB | ≈2.0 GiB |
| Learner + Actor 两个执行器的激活 | `2 × chunk × L × ≈42·C 字节` | chunk 64 ≈0.45 GB | chunk 64 ≈0.7 GB |
| 合批解码（每请求 B 条 prompt） | `B × L·H·16 KiB` 状态 + `B × L × ≈42·C 字节` 激活 | B=32 ≈0.8 GB | B=32 ≈1.2 GB |
| MiSS rank 16 全部 targets | FP32 master/梯度/m/v + BF16，约 18 字节/参数 | ≈0.2 GB | ≈0.35 GB |

检查点数量取决于**已训练过的最长样本**，不是 `--ctx` 本身；缓冲区按出现过的最大长度分配后保留。
长上下文时增大 `--chunk` 可减少检查点（与 chunk 成反比），代价是执行器激活（与 chunk 成正比）；ctx 8192 时 `--chunk 128` 比 64 更省显存，反向重算也更快。
例：7.2B、ctx 8192、chunk 128、MiSS r16、`--max-batch 32` 约 14.4 + 2.0 + 1.4 + 1.2 + 0.35 ≈ 19.4 GB，另加 CUDA 上下文。
合批解码缓冲区按请求的 prompt 数分配并保留到下一次 prompt 数变化；State 模式的 Adam 状态约为 3 个 State 大小。
Prefix Cache 在 CPU 上，每条包含完整 State、BF16 shifts 和末尾 logits。

## HTTP API

所有 POST 请求使用 `Content-Type: application/json`。`GET /health` 返回当前
`policy_version`、Learner `step`、`published_step`、模型 dtype、保留版本、缓存条数及输入限制。
版本是进程内唯一的 opaque string，应原样传回，不依赖其格式。

`POST /v1/tokenize {"text":"..."}` 返回 `tokens`；
`POST /v1/detokenize {"tokens":[98,99]}` 返回 `text`。
文本显示复用原推理服务的 UTF-8 边界处理；多轮上下文始终使用原始 Token IDs，避免重新编码改变 Token 边界。

### Batch completions

```json
{
  "policy_version": "从 health 获得",
  "prompts": [[98,99], [98,99,100,101]],
  "max_tokens": 1024,
  "temperature": 1.0,
  "seed": 1234,
  "stop": ["</tool_call>", "</answer>", "\n\nUser:"],
  "use_cache": true
}
```

返回 `policy_version`、`step`、`temperature`、`seconds`、统计信息，以及按输入顺序排列的 `completions`：

```json
{
  "tokens": [100,101],
  "logprobs": [-3.25,-2.125],
  "text": "cd",
  "finish_reason": "stop",
  "truncated": false,
  "matched_stop": "</answer>",
  "cached_prompt_tokens": 2,
  "prefill_tokens": 0
}
```

`tokens` 仅包含生成的 Token；`logprobs` 与其一一对应，是 **temperature 后采样策略**的 FP32 log probability。
`max_tokens: 0` 只做 Prefill/缓存。`temperature` 必须为 `[0.01,100]`。
`max_tokens` 是防止复读的硬上限，默认继承 Server 的 `--max-tokens 1024`，不是目标生成长度。
每条轨迹在命中字符串后立即结束；没有命中时，最多生成 `min(max_tokens, ctx - prompt长度)` 个 Token。
达到上限返回 `finish_reason: "length"`、`truncated: true`，`truncation_reason` 为 `max_tokens`
或 `context_length`。同批各行独立计算剩余容量，不截断已有 Prompt。命中停止串与上限同一步发生时优先返回 `stop`。
`max_tokens: 0` 的纯 Prefill 返回 `truncated: false`。

**合批解码**：同一请求的所有 prompt 先逐条 Prefill（复用前缀缓存），再放进同一个批次按行并行解码，
每步一次前向，所有 prompt 共用策略版本和 temperature。Rollout 与训练默认统一使用 BF16 Tensor Core
矩阵乘法和 FP32 累加。保留固定 32 行分块以减少数值漂移，但不再使用慢速 pedantic 累加。
不同 batch/chunk 形状仍允许 BF16 舍入差异，
`rwkv_online_rl_forward_test ... bf16` 检查 logits、分布 KL 和教师强制 logprob 偏差。
提前结束的行留在批次里空转，直到整批结束或达到 `max_tokens`；不做跨请求的 continuous batching。
`seed` 决定每条 prompt 的随机流：第 `i` 条使用 `(seed, i)`；不提供 `seed` 时随机选择。
BF16 模式下改变 batch/chunk 形状可能因舍入差异改变采样结果；固定 seed 不代表跨形状逐位一致。

**停止条件**（只匹配生成部分，不匹配 prompt）：

| 字段 | 含义 |
|---|---|
| `stop_token_ids` | 显式选择 Token ID 停止模式；不再默认启用 Token 0 |
| `stop_token_sequences` | Token 序列，生成部分以该序列结尾时停止；最多 16 条，每条 1–64 个 Token |
| `stop` | 默认继承 Server 的停止串；最多 16 条，每条 1–256 字节；`[]` 显式关闭停止条件 |

命中后在**完成匹配的那个 Token** 处停止：该 Token 计入 `tokens`、`logprobs` 和缓存，之后不再采样，
因此返回的 Token 与训练时提交的 Token 完全一致。多轮拼接请以 `tokens` 为准。
在线 RL 直接复用推理引擎的 `TextStopMatcher`：每个 Token 后只检查生成字节串的**末尾**，
只保留最长停止串长度的字节尾部，支持跨 Token 和跨 UTF-8 字节；区分大小写，不匹配 Prompt，
不因 Token 中间包含子串就停止。多个后缀同时命中时按配置顺序选择。
`finish_reason: "stop"` 时 `matched_stop` 给出命中的字符串、Token ID 或 Token 序列。
字符串模式与 `stop_token_ids` / `stop_token_sequences` 互斥，同时提供（包括空数组）返回 400；
显式 Token 规则会替换默认字符串规则。Token 模式中单 ID 优先于序列。

CPU 以数值稳定的普通 `softmax(logits / temperature)` 和标准离散分布采样，多行并行，不增加 Sampler CUDA Kernel。
不提供 greedy、Top-K、Top-P 或 Penalty：截断采样会改变行为策略，而 `/v1/train` 的 CE 按完整的 tempered softmax 计算，
两者不一致时梯度不再对应采样分布。若以后加入截断采样，需要训练端同时按相同截断计算 CE 和 logprob。

默认 `--gemm-mode bf16` 使用现有 cuBLAS BF16 Tensor Core 路径，并禁止低精度中间归约；
输入、权重、激活和 WKV 的序列梯度都是 BF16，WKV State/状态梯度和 Adam 参数仍为 FP32。
前向、重算和反向直接复用 `cuda/rwkv7_statepassing_clampw.cu`，没有另建 RL WKV Kernel。
需要旧版逐位一致对照时使用 `--gemm-mode strict`，恢复固定 32 行与 pedantic 累加，速度会较慢。
`/health` 返回 `gemm_mode`、`gemm_tile_rows`、`wkv_kernel`、`wkv_io_dtype` 和 `wkv_state_dtype`；导出的 `policy.json` 记录 GEMM 模式。
这些配置局限于在线 worker，原 CLI 默认数值策略不变。CUDA 路径已验证，HIP 不保证相同的精度及性能。

Prefill 只在最后一个 Token 计算输出层；训练保存边界的前向扫描不计算输出层。
没有非零 Advantage 的 Chunk 跳过 CE 和输出层反传，仍完整反传 WKV 和 shift 的跨块梯度。
最后一个非零监督目标之后的无损失后缀不参与计算，零 Advantage 样本直接返回零梯度。
State 模式在 GPU 累加各样本梯度，再统一检查并更新。客户端指标增加 `rollout_seconds`、
`train_seconds`、`publish_seconds` 和 `train_tokens_per_second`，便于区分生成、训练和保存耗时。
`rollout_loss` 根据实际采样 logprob 与本批 Advantage 独立计算，`loss_abs_error` 是它与训练 Loss 的
绝对差，可持续观察 BF16 推训数值漂移；这些指标不增加 GPU 前向次数。

缓存复用最长的已保存 Token 前缀，同时恢复 WKV、attention/FFN shifts 和 next-token logits。
缓存保存完整 Prompt 和完整输出历史（包含最后一个生成 Token）：
`AB → ABCD → ABCDEF → ABCDEFGH` 仅执行每步的新增后缀。
每个 Policy 快照持有自己的 Trie，训练不会访问 Prefix Cache。

### 教师强制 logprob 与参考策略

```json
POST /v1/logprobs
{"policy_version": "可省略，默认当前版本", "temperature": 1.0, "sequences": [[98,99,100,101]]}
```

返回 `logprobs`：每条序列一个数组，长度为 `len(tokens)-1`，第 `i` 项是 `log softmax(logits/temperature)[tokens[i+1]]`。
它与训练使用同一前向和同一 CE 归约，不使用、也不写入前缀缓存。

服务启动时的策略（冷启动 Adapter/State，或 `--resume` 恢复的策略）作为**参考策略**永久保留，不受 `--keep-versions` 淘汰；
`/health` 的 `reference_version` 给出其版本，可用于 `/v1/logprobs` 或 `/v1/completions`。
服务端没有内置 KL 项；需要对 SFT 起点加 KL 约束时，可在客户端把它折进逐 Token Advantage，例如：

```python
full = prompt + out["tokens"]                     # out 来自 /v1/completions
ref = call("/v1/logprobs", {"policy_version": health["reference_version"],
                            "temperature": T, "sequences": [full]})["logprobs"][0]
ref_gen = ref[len(prompt) - 1:]                   # 与 out["tokens"] 逐个对齐
advantages = [A - beta * (lp - rp) for lp, rp in zip(out["logprobs"], ref_gen)]
turn = {"tokens": out["tokens"], "loss_mask": [1] * len(out["tokens"]),
        "advantages": advantages}
```

这是常见的“KL 进奖励”写法（k1 估计）；`beta` 及估计方式由客户端决定。
多轮样本对每个生成 turn 分别这样计算，观察 turn 的 `loss_mask` 为 0，不需要参考 logprob。

### 用外部推理服务采样

`/v1/train` 只校验 `policy_version` 是否为当前版本，不校验 Token 是否由本服务生成，所以技术上可以用
`rwkv_lighting_cuda` 采样，再把 Token 交回训练。需要清楚的是：

- 训练是纯 on-policy REINFORCE，没有重要性采样修正。推理服务使用不同的前向 Kernel（以及可能的量化、稀疏 cmix），
  logprob 与本服务不逐位一致，采样分布有小偏差，相当于轻微 off-policy。偏差是否可接受取决于任务，服务端无法保证；
  可以用 `/v1/logprobs` 对外部轨迹打分，与推理服务返回的 logprob 比较来监控。
- 每步需要 `publish` 时 `save:true`（或 `checkpoint:true`），再把导出的 `adapter.pth` 注册到推理服务：
  `POST /v1/adapters`，同机可传 `{"adapter_id":"rl","path":".../adapter.pth"}`，跨机可用 multipart 上传文件。
  同一 `adapter_id` 每次注册得到新的内容版本，无需重启推理服务；生成时用 `adapter_version` 固定版本，旧版本可 `DELETE`。
  目前没有不经磁盘直接推送权重的接口。State 模式训练的 State 不走这条路径。
- 采样请求本身已经合批，通常应优先用本服务采样，省掉导出、注册和数值偏差。

### RL train step

每个请求的 `learning_rate` **必填**，须为有限数值，范围 `1e-12..10`。
缺失、非数值或越界返回 HTTP 400，不执行优化器更新。每次请求独立生效，
可以由客户端按当前 `step` 计算 warmup、衰减等调度；Adam m/v 和更新步数仍跨请求保留。
启动和 `--resume` 都不设置学习率，恢复后的第一步也必须传入该字段。

```json
{
  "policy_version": "rollout 返回的版本",
  "temperature": 1.0,
  "learning_rate": 0.00002,
  "samples": [
    {
      "tokens": [98,99,100,101,102,103],
      "loss_mask": [0,0,1,1,0,1],
      "advantages": 1.25
    },
    {
      "turns": [
        {"tokens": [98,99], "loss_mask": [0,0]},
        {"tokens": [100,101], "loss_mask": [1,1]},
        {"tokens": [102], "loss_mask": [0]},
        {"tokens": [103,104], "loss_mask": [1,1]}
      ],
      "advantages": -0.75
    }
  ]
}
```

`tokens` 和 `turns` 二选一；每段必须有等长 `loss_mask`（布尔或 0/1）。
`advantages` 可以是样本级 signed scalar，也可以是与该段 Tokens 等长的数组；多轮样本可在各 turn 内覆盖。
样本级 `temperature` 可覆盖请求值，应与 Rollout 一致。第一颗 Token 没有前驱，始终不计算 Loss；后续轮次的首 Token 正常由上一轮末尾预测。
完整 Tokens 先拼接，再按 Chunk 前向与反向；Masked Prompt/工具观察仍参与梯度传播。

目标为：

```text
N = 全 batch 中 loss_mask=1 的可预测 Token 数
loss = - sum(mask[t] * advantage[t] * log softmax(logits[t-1]/temperature)[token[t]]) / N
```

正 Advantage 增大动作概率，负 Advantage 减小动作概率。分母不使用有符号权重之和。
复用原 CE Kernel，加入逐行 signed weight 和 temperature 的链式导数；没有 PPO clipping、KL 项或重要性修正。
GRPO 的组内 Reward 归一化在 Python 中完成。服务每次执行一次 Adam 更新，跨请求保持 FP32 master、梯度和 Adam m/v。

返回 `loss`、`grad_norm`（FP32 梯度的 L2）、本次实际使用的 `learning_rate`、`step`、`trained_tokens`、`total_tokens`、`seconds`、`updated` 和 `publish_required`。
全零 Advantage 返回 `updated:false`，不推进 Adam。全零 Loss Mask 返回 400。

服务要求 `policy_version` 是当前发布版本且其 `published_step == learner.step`。
更新后 Actor 继续保持原策略，第二个 train 会返回 409，直到发布并重新采样。
将一轮的所有组、样本放进同一个 train 请求；这保证此实现的单步 REINFORCE 使用匹配策略，避免把旧轨迹当作 on-policy 数据。

### 发布 Adapter / State 版本

```json
POST /v1/publish
{"expected_step": 1, "save": true}
```

成功返回新的 `policy_version`、`step`，保存时还返回 `directory`。
先复制已更新的可训练参数为新 Actor 快照，完成 CUDA 同步和可选文件写入，再切换当前版本。
Backbone 指针始终共享。显式指定旧版本的请求仍使用其旧权重与旧缓存，超过 `--keep-versions` 后旧版本返回 409。
版本切换不会重建 Learner 或清空 Adam。

`save:true` 输出 `state.pth`、有 Adapter 时的 `adapter.pth`、`policy.json`。
State 和 Adapter 按现有格式导出 BF16，可由原推理服务或两个新 Server 冷启动加载；用它们冷启动会从新 Adam 状态和 step=0 开始。

`checkpoint:true`（隐含 `save:true`）额外写入 `training/` 子目录，即完整训练检查点：

- MiSS：沿用 `rwkv_miss_tune` 的检查点格式（FP32 master、梯度、Adam m/v、初始 State 和 step）；
- State：FP32 可训练 State、Adam m/v 和 step。

返回值中的 `checkpoint` 是该子目录。重启时传 `--resume <目录>`：未显式指定时，`adapter.pth`/`state.pth` 也从该目录读取；
恢复后 `/health` 的 `step` 和 `published_step` 都等于检查点 step，Adam 的偏差校正从该 step 继续。
MiSS 恢复时会校验 rank、alpha、targets 和 Backbone fingerprint 与检查点一致。
7.2B r16 的 MiSS 检查点约为 Adapter 参数量 × 16 字节（约 0.3 GB），State 检查点约为 3 个 State 大小（约 100 MB），建议隔几步再存一次。

输入错误返回 400；版本过期/发布 step 不符返回 409；工作队列满返回 503；执行错误返回 500。
当前采用有界单 GPU worker（最多 32 个排队请求），按请求串行调度。
采样请求内的 prompt 合批解码；训练请求内的样本仍逐条前向/反向、累积整批梯度后更新一次，与现有 CLI 一致。
采样和训练共用同一个 worker，一步的耗时是两者之和；暂未把 Actor 和 Learner 拆到并行的 worker 或 GPU 上。

## Python 闭环示例

仅依赖 `tqdm`，HTTP 使用 Python 标准库。读取提供的 `segments` 数据，每个 `train:true` 段作为待采样的 Assistant 轮次。

```bash
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 \
  --data /mnt/nvme0n1/no22/segments-2-2.jsonl \
  --steps 10 --group-size 4 --max-turns 2 --max-tokens 1024 \
  --temperature 1.0 --lr 2e-5 --seed 1234 --timeout 3600 \
  --save --metrics ./miss-online-metrics.jsonl
```

运行客户端前可用 `curl http://127.0.0.1:8848/health` 检查状态。
数据集每行是一个 JSON 对象，例如：

```json
{"segments":[{"text":"User: 问题\nAssistant:","train":false},{"text":"参考答案","train":true}]}
```

`--steps` 是本次处理的批数，不是 Adam 更新次数；`--group-size` 不能超过服务的 `--max-batch`。
客户端默认继承 Server 的字符串停止配置和生成上限；`--max-tokens`（兼容旧名称 `--max-new-tokens`）
只设置每轮硬上限。可用 `--stop '["</tool_call>", "</answer>", "\n\nUser:"]'` 覆盖停止串，或用 `--stop '[]'` 显式关闭。
不再按“每轮一定生成上限数量”预先跳过多轮数据，而是按各轨迹的实际历史长度判断剩余容量。
超限轨迹保留已生成 Token 及其 logprob，结束本条轨迹，不再追加后续观察；奖励仍交给 Judge 计算。
指标记录 `finish_reasons`、`matched_stops`、`truncated_samples`、`truncation_reasons` 和 `completed_turns`。
进度条的 `trunc` 显示本批截断样本数。首个 Prompt 已占满 `ctx` 的数据行会跳过。
换成 `:8849`、设置客户端 `--lr 0.01` 并使用独立指标文件可测 State。
示例客户端的 `--lr` 必填，它在每个训练请求中发送 `learning_rate`；自定义循环可逐步改变此值。
进度条与 JSONL 指标包含 Reward 均值/标准差、Loss、梯度范数、实际学习率、更新步数、吞吐和缓存命中率。
默认 Judge 是生成 Token 与参考答案的 unigram F1，仅供演示。实际任务可用
`--judge your_module:reward` 替换；函数接收 `(generated_token_ids, reference_token_ids)`，返回标量奖励。
组内 Reward 无方差时 Advantage 全零，会跳过更新。

数据集中的工具结果作为离线观察回放；示例不执行模型生成的工具调用。真实 Agent RL 应由 Python 环境执行动作并提供新的观察和 Judge。

### 真实工具环境的多轮示例

下面是一组 GRPO 轨迹的骨架（`call` 为 POST JSON 的辅助函数，`run_tool`、`judge` 由你的环境提供）。要点：

- 上下文始终是 **Token ID 列表**。工具观察单独用 `/v1/tokenize` 编码后拼到历史末尾，不要把整段文本重新编码，否则 Token 边界会变化，训练的 Token 就和采样时不一致。
- 生成的 Token（包括命中 stop 的最后一个 Token）原样进入历史和训练样本；`text` 用于解析动作。
- `turns` 按时间顺序排列：首个 prompt 和每段观察 `loss_mask` 全 0；每段生成 `loss_mask` 全 1。首 Token 没有前驱，不计入 Loss。
- 同一组的所有活跃轨迹放进同一次 `/v1/completions`，享受合批解码；结束的轨迹从下一轮请求中去掉。历史前缀会命中前缀缓存，每轮只 Prefill 新增的观察。
- Prompt 不能超过 `--ctx`；生成按每行剩余容量和 `max_tokens` 中的较小值封顶，并明确标记截断原因。

```python
health = call("/health")
version = health["policy_version"]
prompt = call("/v1/tokenize", {"text": f"User: {question}\n\nAssistant:"})["tokens"]
episodes = [{"history": list(prompt), "done": False,
             "turns": [{"tokens": list(prompt), "loss_mask": [0] * len(prompt)}]}
            for _ in range(group_size)]
for turn in range(max_turns):
    active = [e for e in episodes if not e["done"]]
    if not active:
        break
    outs = call("/v1/completions", {
        "policy_version": version, "prompts": [e["history"] for e in active],
        "max_tokens": 1024, "temperature": 1.0, "seed": seed * 100 + turn,
        "stop": ["</tool_call>", "</answer>", "\n\nUser:"],
    })["completions"]
    for e, out in zip(active, outs):
        e["turns"].append({"tokens": out["tokens"], "loss_mask": [1] * len(out["tokens"])})
        e["history"] += out["tokens"]
        if out.get("matched_stop") != "</tool_call>":
            e["done"] = True          # 回答结束、截断或其它 stop
            continue
        observation = run_tool(out["text"])
        obs = call("/v1/tokenize", {"text": f"\n<tool_response>{observation}</tool_response>\n"})["tokens"]
        e["turns"].append({"tokens": obs, "loss_mask": [0] * len(obs)})
        e["history"] += obs
rewards = [judge(e) for e in episodes]
mean = sum(rewards) / len(rewards)
std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 or 1.0
samples = [{"turns": e["turns"], "advantages": (r - mean) / std} for e, r in zip(episodes, rewards)]
lr = learning_rate_for_step(health["step"] + 1)  # 由客户端的调度函数决定本次 Adam 更新的 LR
train = call("/v1/train", {"policy_version": version, "temperature": 1.0,
                         "learning_rate": lr, "samples": samples})
if train["updated"]:
    call("/v1/publish", {"expected_step": train["step"]})
```

多组时把各组的样本放进同一个 `/v1/train`（总数不超过 `--max-batch`），每步只训练、发布一次。

快速检查管线时显式使用：

```bash
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 \
  --data /mnt/nvme0n1/no22/segments-2-2.jsonl \
  --steps 2 --group-size 2 --max-turns 1 --max-new-tokens 4 \
  --lr 2e-5 --smoke --save --metrics /tmp/miss-smoke.jsonl
```

`--smoke` 只保留初始上下文末尾 64 Tokens，Reward 相同时使用人工正负 Advantage，并在指标中标记 `synthetic_advantages`。
它验证闭环连通性，不用于衡量训练效果。正常模式保留完整历史，超过服务 `ctx` 的数据行会明确跳过。

## 长请求、保存与断线恢复

Server 禁用了 Drogon 默认的 60 秒连接空闲超时，因为训练期间
HTTP 连接没有数据传输。客户端 `--timeout` 默认 3600 秒，可按训练长度调大。
升级 Server 二进制后需要重启进程才会生效。

遇到断线时不要直接重发 `/v1/train`：GPU 更新可能已经完成。先请求 `/health`；
若 `step > published_step`，用该 `step` 调用 `/v1/publish`，并设置 `save: true`。
客户端启动时会检查这个状态，避免用旧策略继续训练。
重启后可用导出的 `adapter.pth` 和 `state.pth` 冷启动，但 Adam 状态不在这两个文件中，会重新初始化；
若发布时使用了 `checkpoint:true`，改用 `--resume <目录>` 即可连同 Adam 状态和 step 一起恢复。
`--start-row 4` 从数据集第 4 行继续，`--steps` 指本次处理的批数。
指标文件现在追加写入，不会覆盖已有记录；断线丢失的训练指标不作推测或补造。

例如 `/health` 显示 `step=3`、`published_step=2`，可执行：

```bash
curl -X POST http://127.0.0.1:8848/v1/publish \
  -H 'Content-Type: application/json' \
  -d '{"expected_step":3,"checkpoint":true}'
```

务必将 `3` 换成实际查询到的步数。保存返回 `directory` 后，如需重启，启动命令加 `--resume <directory>`
（或只用 `save:true` 时，把 `--adapter`、`--state` 指向该目录下的对应文件，Adam 会重置）。
确认已完成的数据行后，在原 Python 命令中加 `--start-row 4 --steps 7` 可从第 4 行再处理 7 批。
这需要手动确认数据位置，不是自动恢复训练检查点。

## 实现与测试

Actions 会编译两个 Server、打包依赖并在打包目录执行 `--help`；也运行纯 CPU 前缀索引测试。
托管 Runner 没有 GPU，不运行真实 Rollout/训练；GPU 闭环需另行验证。

HTTP 验收脚本可加 `--check-idle-timeout`，对每种 Server 检查同一条连接静默 75 秒后
仍能得到响应，然后执行 rollout/train/publish/冷启动闭环。

- `src/miss_tune_server/`、`src/state_tune_server/`：独立 main 和 CMake Target。
- `src/online_rl/`：公共 Drogon 路由、常驻 Learner、版本快照及调度。
- `bf16_training::Executor`：从原 CLI 提取的共享 Chunk 执行器；CLI 和 Server 都调用它，保留 WKV 与 attention/FFN shift 的跨 Chunk 反向 adjoint。
- `inference/prefix_cache.hpp`：原 Batch Cache 的 CPU 快照管理改用共享 Radix Trie + LRU；原状态格式及 cache identity 隔离保留。原实现的查找是哈希加 LRU 全表扫描，现改为压缩 Token 前缀索引。

```bash
cmake --build build -j8 --target rwkv_bf16_training_test rwkv_prefix_cache_index_test \
  rwkv_prompt_cache_test rwkv_online_rl_training_test rwkv_online_rl_forward_test
ctest --test-dir build -R 'rwkv_(bf16_training|prefix_cache_index|prompt_cache|state_tuning_api)_test' --output-on-failure

# 用已有 check_miss.py 生成 tiny fixture 并回归原 CLI（需要 PyTorch）
uv run --with torch --with numpy python tools/check_miss.py \
  --build build --work /tmp/miss-acceptance
build/test/rwkv_online_rl_training_test /tmp/miss-acceptance/tiny.pth
build/test/rwkv_online_rl_forward_test /tmp/miss-acceptance/tiny.pth \
  /tmp/miss-acceptance/learn/adapter-final.pth /tmp/miss-acceptance/initial.pth
python3 test/online_rl_http_test.py --build build \
  --model /tmp/miss-acceptance/tiny.pth --vocab /tmp/miss-acceptance/vocab.txt \
  --adapter /tmp/miss-acceptance/learn/adapter-final.pth \
  --state /tmp/miss-acceptance/initial.pth --work /tmp/online-http
```

测试覆盖 Signed CE/temperature 与 FP64 softmax 对照、全长与跨 Chunk 梯度、早期 Masked 上下文梯度、正负梯度反转、Radix 分叉/LRU 随机对照、原 Batch Cache、合批解码与 batch=1 逐位一致、教师强制 logprob，以及 HTTP 多轮、并发、合批采样、字符串/Token 序列 stop、冻结 Actor、旧版本隔离、参考策略、连续更新、导出重载和 `--resume` 恢复。
CUDA 实测记录见 [验证记录](online-rl-validation.zh-CN.md)。HIP 路径保留同一套源代码和构建依赖，此次未在 AMD 硬件验证。
