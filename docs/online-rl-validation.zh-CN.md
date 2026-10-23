# 在线 RL CUDA 验证记录

日期：2026-10-22。GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition（96 GB）。
测试期间同卡另有原推理服务运行；下述用时是闭环验证记录，不作为独占 GPU 吞吐基准。

## 构建与原功能回归

成功构建：

- `rwkv_miss_tune_server`、`rwkv_state_tune_server`。
- 原 `rwkv_miss_tune`、`rwkv_state_tune`、`rwkv_lighting_cuda`。
- 新增 Forward、Training、Radix Trie 测试 Target。

通过的 CTest：`rwkv_inference_engine_test`、`rwkv_prompt_cache_test`、
`rwkv_state_tuning_api_test`、`rwkv_bf16_training_test`、`rwkv_prefix_cache_index_test`。

`tools/check_miss.py` 原 CLI 回归通过：全模型 Autograd 对照 Loss `5.2131605`、
MiSS 训练 Loss `5.2132 → 2.9426`、Checkpoint Resume、FP16/W8A16/W4A16 Adapter 推理、BF16 数值范围和模型转换。
额外运行 State CLI 两步 Adam：全长与 7-token Chunk 导出的 8,192 个参数相对 L2 差为 `0.0021142`。
这是 BF16 反向归约和 Adam 放大近零梯度差异后的数值比较，不要求不同训练分块导出逐位相同。

## 梯度与缓存

| 测试 | 结果 |
|---|---|
| Signed CE、零权重 Mask、temperature 导数与独立 FP64 softmax | 通过 |
| WKV 输入/初始 State 有限差分、终端 adjoint 跨 Chunk | 通过 |
| 31 个训练位置，前 21 个目标 Mask 掉，后续含正负 Advantage；全长 vs Chunk=7 | MiSS 梯度相对 L2 `0.000126899`，初始 State 梯度 `0.000177266` |
| 全部 Advantage 反号 | MiSS、State 梯度精确反号，Loss 精确反号 |
| Radix Trie 分叉、短查询落在压缩边中、替换、LRU 淘汰、版本隔离 | 20,000 次随机操作与朴素 LRU 参考一致 |
| `AB → ABCD → ABCDEF → ABCDEFGH` | 命中 `0 / 2 / 4 / 6` Tokens，每步 Prefill `2` Tokens |

前缀缓存只保存 CPU WKV State、BF16 shifts 和末尾 logits。训练使用独立执行器和 Chunk 边界检查点，未连接推理缓存计算图。

## 7.2B 推训前向一致性

使用以下真实资源共同冷启动：

```text
/mnt/nvme0n1/no22/rwkv7-g1k-7.2b-20260930-ctx25600.pth
/mnt/nvme0n1/no22/rwkv_lighting_cuda/train-2-2-g1k/g1k-r16-lr2e-5-e2/adapter-final.pth
/home/alic-li/rwkv_lightning_cuda/rwkv-state-agentic.pth
```

验证暴露了原 cuBLAS 默认 GEMM 的形状相关舍入差异：相同权重在不同行数下会使用不同归约方式，BF16 舍入可在深层累积。
在线 worker 为现有 BF16 GEMM 固定 32 行块，并采用严格 FP32 累加；只在 GEMM 内补齐，WKV 仍处理实际 Token 数。原 CLI 默认路径保持不变。

`rwkv_online_rl_forward_test` 在 40 Tokens 的 7.2B 测试中验证：

- Actor 与 Learner 前向 logits **逐位相同**。
- CPU Cache State/shift 往返后 logits **逐位相同**。
- Chunk `40 / 7 / 2 / 1` 的最终 logits **逐位相同**；最大绝对差与 RMS 均为 `0`。

这是本机、本模型和测试序列的结果，不声明跨 GPU 架构或 CUDA 库版本的逐位确定性。
严格累加和固定块补齐增加计算开销；当前实现优先保证在线策略一致性。

## 7.2B HTTP 闭环

`test/online_rl_http_test.py` 使用同一套 7.2B + 冷 Adapter + 冷 State，在两个 Server 上分别通过：

- 批量生成、返回 Token IDs / FP32 logprobs、缓存与无缓存比较。
- 并发 HTTP 请求进入单 GPU 队列，固定 seed 的结果一致。
- 两轮生成、轮间 Masked 观察、Signed Advantage 训练。
- 非法 Token、非法 temperature、空监督、旧 Policy、错误发布 step 的拒绝。
- 零 Advantage 不推进优化器；非零训练成功推进步骤。
- 发布前 Actor 完全不变；发布后新缓存不命中旧版本；旧版本显式请求结果保持不变。
- 旧版超出保留数后返回 409；重复发布相同步骤是幂等操作。
- 每种模式连续两个 Adam Step 和发布；导出 Adapter/State 后重启冷加载成功。

| 模式 | Train Loss | Rollout logprobs 独立计算 Loss | 第一步梯度 L2 | 第二步梯度 L2 |
|---|---:|---:|---:|---:|
| MiSS | 0.2803220153 | 0.2803221910 | 1963.2714 | 2508.3994 |
| State | 0.2803220153 | 0.2803221910 | 47.2233 | 103.2163 |

两种模式的推训 Loss 差均约 `1.76e-7`。

复现：

```bash
python3 test/online_rl_http_test.py --build build \
  --model /mnt/nvme0n1/no22/rwkv7-g1k-7.2b-20260930-ctx25600.pth \
  --vocab assets/rwkv_vocab_v20230424.txt \
  --adapter /mnt/nvme0n1/no22/rwkv_lighting_cuda/train-2-2-g1k/g1k-r16-lr2e-5-e2/adapter-final.pth \
  --state rwkv-state-agentic.pth --chunk 7 --work /tmp/online-http-7b
```

## 数据集与 tqdm 示例

使用 `/mnt/nvme0n1/no22/segments-2-2.jsonl`，最终二进制运行 `tools/online_rl.py`，
`group-size=2`、`max-turns=2`、`max-new-tokens=16`、`chunk=64`、`ctx=4096`，没有使用 `--smoke` 截短或人工 Advantage。
此验收进程使用 MiSS `lr=0.001`、State `lr=0.01`；日常示例的 MiSS 默认学习率仍为 `2e-5`。
数据集前两行实际各只有一个待生成轮次；多轮路径由上述 HTTP 测试覆盖。

首行每批完整计算 1,984 个训练位置，监督 32 个生成 Tokens：

| 模式 | Reward 均值 / 标准差 | Loss | 梯度 L2 | 完整闭环用时 | Prefix 命中率 |
|---|---|---:|---:|---:|---:|
| MiSS | 0.234848 / 0.007576 | -0.01133154 | 162.6837 | 54.74 秒 | 50% |
| State | 0.234848 / 0.007576 | -0.01133154 | 0.822601 | 54.98 秒 | 50% |

首行两种模式均训练及发布成功。第二行 MiSS 的组内 Reward 全零，按预期跳过更新；
State 的 Reward 均值/标准差为 `0.346154 / 0.038462`，完成第二次更新及发布（Loss `-0.1804123`，梯度 L2 `0.872234`，2,412 个训练位置，用时 64.88 秒）。
这验证真实文件、Tokenizer、Rollout、Judge、Train、发布和指标输出的连通性，不代表奖励质量或收敛结论。
默认 Token F1 Judge 是示例，可替换为真实任务 Judge。

原始本机日志与指标：

```text
/tmp/rwkv-online-http-7b-verified.log
/tmp/rwkv-online-7b-forward-pedantic.log
/tmp/rwkv-online-gradient-verified.log
/tmp/rwkv-online-cli-final.log
/tmp/rwkv-online-7b-final-clients/miss-metrics.jsonl
/tmp/rwkv-online-7b-final-clients/state-metrics.jsonl
```

测试自行启动的进程均在结束后停止；未停止原有推理服务。当前验证范围是 CUDA，未验证 HIP 硬件。
