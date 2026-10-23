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
  --max-batch 8 --cache-capacity 8 --keep-versions 2 --lr 2e-5 \
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
  --max-batch 8 --cache-capacity 8 --keep-versions 2 --lr 0.01 \
  --output ./online-rl-output/state
```

以上命令在源码仓库根目录执行。GitHub Actions 的 Linux/Windows CUDA 12.9、13.2
安装包均包含两个 Server、运行库、词表、`tools/online_rl.py` 和在线 RL 文档。
解压后进入 `rwkv_lighting_cuda/` 目录，将命令中的 `./build/二进制` 改为 `./二进制`，
并将词表改为 `--vocab ./rwkv_vocab_v20230424.txt`；默认词表路径指向构建机器，不能用于发布包。
模型、冷启动 Adapter/State 和数据集需要自行提供，示例中的绝对路径应替换为本机路径。

Windows PowerShell 在解压目录执行下面对应模式的一条命令：

```powershell
.\rwkv_miss_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8848 --chunk 64 --ctx 4096 --max-batch 8 --lr 2e-5 --output .\online-rl-output\miss
.\rwkv_state_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8849 --chunk 64 --ctx 4096 --max-batch 8 --lr 0.01 --output .\online-rl-output\state
```

选择一种模式启动即可；`--help` 无需加载模型，可检查可执行文件及依赖是否可用。
等待日志出现 `ready`，在另一个终端运行 Python 闭环，关闭客户端不会关闭后端。

不指定 Adapter 时 MiSS 从零 `D` 开始；不指定 State 时使用零初始 State。
冷 Adapter 自动提供 rank、alpha、targets，并校验基模 fingerprint 和维度。
每个 Server 进程仅加载一份 Backbone；同时启动两个进程会各自加载一份。

| 参数 | 默认值与含义 |
|---|---|
| `--host` | `127.0.0.1` |
| `--port` | MiSS `8848`，State `8849` |
| `--chunk` | `64`；计算与反向重算块长，不截断梯度 |
| `--ctx` | `4096`；完整历史 Token 上限，超长请求直接拒绝 |
| `--max-batch` | `32`；一次生成或训练的最大样本数 |
| `--cache-capacity` | `16`；每个保留版本的 CPU 前缀快照条数，`0` 关闭 |
| `--keep-versions` | `2`；当前及旧 Actor 快照总数 |
| `--lr` | MiSS `2e-5`，State `0.01` |
| `--rank / --alpha / --targets` | 无冷 Adapter 时 `16 / rank / all`；冷启动时推断，显式冲突报错 |
| `--vocab` | 项目的 `assets/rwkv_vocab_v20230424.txt` |
| `--output` | `online-rl-output`；显式发布保存的目录 |

7.2B 的一个完整 FP32 WKV State 约 32 MiB。Prefix Cache 在 CPU 上，还包含 BF16 shifts 和末尾 logits；训练在 GPU 上保留每个 Chunk 边界的 State/shift，因此增大 `ctx/chunk` 也会增加显存需求。

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
  "max_tokens": 16,
  "temperature": 1.0,
  "seed": 1234,
  "stop_token_ids": [0],
  "use_cache": true
}
```

返回 `policy_version`、`step`、`temperature`、`seconds`、统计信息，以及按输入顺序排列的 `completions`：

```json
{
  "tokens": [100,101],
  "logprobs": [-3.25,-2.125],
  "text": "cd",
  "finish_reason": "length",
  "cached_prompt_tokens": 2,
  "prefill_tokens": 0
}
```

`tokens` 仅包含生成的 Token；`logprobs` 与其一一对应，是 **temperature 后采样策略**的 FP32 log probability。
停止 Token 也计入返回值、logprob 和缓存。`stop_token_ids: []` 关闭停止 Token；默认 `[0]`。
`max_tokens: 0` 只做 Prefill/缓存。`temperature` 必须为 `[0.01,100]`，不提供 greedy、Top-K、Top-P 或 Penalty。
CPU 以数值稳定的普通 `softmax(logits / temperature)` 和标准离散分布采样，不增加 Sampler CUDA Kernel。

在线服务的现有 BF16 GEMM 使用固定 32 行计算块及严格 FP32 累加，避免 cuBLAS 在 Prefill、Decode 和不同 Chunk 形状间引入策略数值差异；输入、权重及前向输出仍为 BF16。仅矩阵乘法补齐，WKV 处理真实 Token 数。该设置局限于在线 GPU worker，原 CLI 默认方式不变；小块补齐和严格累加会增加计算开销。

缓存复用最长的已保存 Token 前缀，同时恢复 WKV、attention/FFN shifts 和 next-token logits。
缓存保存完整 Prompt 和完整输出历史（包含最后一个生成 Token）：
`AB → ABCD → ABCDEF → ABCDEFGH` 仅执行每步的新增后缀。
每个 Policy 快照持有自己的 Trie，训练不会访问 Prefix Cache。

### RL train step

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

返回 `loss`、`grad_norm`（FP32 梯度的 L2）、`step`、`trained_tokens`、`total_tokens`、`seconds`、`updated` 和 `publish_required`。
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
State 和 Adapter 按现有格式导出 BF16，可由原推理服务或两个新 Server 冷启动加载。
导出不是完整在线训练检查点：重新启动会从新 Adam 状态和 step=0 开始；本功能保证进程内优化器持续驻留。

输入错误返回 400；版本过期/发布 step 不符返回 409；工作队列满返回 503；执行错误返回 500。
当前采用有界单 GPU worker（最多 32 个排队请求），按请求串行调度；Batch 内样本顺序执行，训练累积整批梯度后更新一次。
这与现有 CLI 的顺序累积一致，暂未增加并行 Actor/Learner 或 GPU continuous batching。

## Python 闭环示例

仅依赖 `tqdm`，HTTP 使用 Python 标准库。读取提供的 `segments` 数据，每个 `train:true` 段作为待采样的 Assistant 轮次。

```bash
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 \
  --data /mnt/nvme0n1/no22/segments-2-2.jsonl \
  --steps 10 --group-size 4 --max-turns 2 --max-new-tokens 64 \
  --temperature 1.0 --seed 1234 --timeout 3600 \
  --save --metrics ./miss-online-metrics.jsonl
```

运行客户端前可用 `curl http://127.0.0.1:8848/health` 检查状态。
数据集每行是一个 JSON 对象，例如：

```json
{"segments":[{"text":"User: 问题\nAssistant:","train":false},{"text":"参考答案","train":true}]}
```

`--steps` 是本次处理的批数，不是 Adam 更新次数；`--group-size` 不能超过服务的 `--max-batch`。
换成 `:8849` 并使用独立指标文件可测 State。进度条与 JSONL 指标包含 Reward 均值/标准差、Loss、梯度范数、更新步数、吞吐和缓存命中率。
默认 Judge 是生成 Token 与参考答案的 unigram F1，仅供演示。实际任务可用
`--judge your_module:reward` 替换；函数接收 `(generated_token_ids, reference_token_ids)`，返回标量奖励。
组内 Reward 无方差时 Advantage 全零，会跳过更新。

数据集中的工具结果作为离线观察回放；示例不执行模型生成的工具调用。真实 Agent RL 应由 Python 环境执行动作并提供新的观察和 Judge。

快速检查管线时显式使用：

```bash
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 \
  --data /mnt/nvme0n1/no22/segments-2-2.jsonl \
  --steps 2 --group-size 2 --max-turns 1 --max-new-tokens 4 \
  --smoke --save --metrics /tmp/miss-smoke.jsonl
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
重启后可用导出的 `adapter.pth` 和 `state.pth` 冷启动；Adam 状态不在这两个文件中，
重启会重新初始化 Adam。`--start-row 4` 从数据集第 4 行继续，`--steps` 指本次处理的批数。
指标文件现在追加写入，不会覆盖已有记录；断线丢失的训练指标不作推测或补造。

例如 `/health` 显示 `step=3`、`published_step=2`，可执行：

```bash
curl -X POST http://127.0.0.1:8848/v1/publish \
  -H 'Content-Type: application/json' \
  -d '{"expected_step":3,"save":true}'
```

务必将 `3` 换成实际查询到的步数。保存返回 `directory` 后，如需重启，将启动命令的
`--adapter`、`--state` 指向该目录下的对应文件；没有导出 Adapter 则不传 `--adapter`。
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

测试覆盖 Signed CE/temperature 与 FP64 softmax 对照、全长与跨 Chunk 梯度、早期 Masked 上下文梯度、正负梯度反转、Radix 分叉/LRU 随机对照、原 Batch Cache，以及 HTTP 多轮、并发、冻结 Actor、旧版本隔离、连续更新和导出重载。
CUDA 实测记录见 [验证记录](online-rl-validation.zh-CN.md)。HIP 路径保留同一套源代码和构建依赖，此次未在 AMD 硬件验证。
