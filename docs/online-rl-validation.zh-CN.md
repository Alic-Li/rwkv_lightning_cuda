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

## 2026-10-24：在线 BF16 Tensor Core 优化

本节更新上面的旧版 strict 性能及精度结论。WKV 没有替换：两个 Server 的前向、重算、反向
仍直接调用 `cuda/rwkv7_statepassing_clampw.cu`，选择 `IoType::BF16`；State 和状态梯度为 FP32。

默认 `--gemm-mode bf16` 在 rollout 和训练中统一启用现有 cuBLAS BF16 Tensor Core，保留 FP32
累加并禁止低精度中间归约。保留 32 行分块，但移除 `CUBLAS_COMPUTE_32F_PEDANTIC`；
`--gemm-mode strict` 可恢复原数值路径。测试中完全取消分块的方案有较大的单 Token logprob
误差，因此没有作为默认配置。

另有以下优化：Prefill 只计算末尾 Token 的输出层；保存分块边界时不计算输出层；纯 Mask
Chunk 跳过 CE/输出层反传，继续完整传播 WKV、shift 和 MiSS 梯度；跳过零权重样本及最后
非零监督目标后的无损失后缀；State 梯度在 GPU 累加，统一传回检查。没有截断 BPTT 或冻结
本应训练的参数。

### 实际 7.2B 数据首批测速

下面的 64-token 上限用于当时的固定负载测速，不能当作 Agentic RL 的动作/回答结束条件。
当前默认行为已更新为字符串停止 + 防复读硬上限，见本页末尾的字符串停止验证。

使用前述同一套模型、Adapter、初始 State，以及 `segments-2-2.jsonl` 首行。
Server 为 `chunk=64, ctx=4096, max-batch=8, cache-capacity=8`；客户端为
`steps=1, group-size=4, max-turns=2, max-new-tokens=64, temperature=1, seed=1234, --save`。
每批生成 256 Tokens、处理 4,160 个训练位置，缓存命中率 75%。MiSS LR 为 `2e-5`，State LR 为 `0.01`。
每次从相同冷参数启动独立测试 Server，并先做一次 2-token 生成预热。

| 模式 | 阶段 | 修改前 | BF16 优化后 | 加速 |
|---|---|---:|---:|---:|
| MiSS | Rollout | 33.229 秒 | 1.591 秒 | 20.89× |
| MiSS | Train | 39.419 秒 | 6.133 秒 | 6.43× |
| MiSS | 完整一轮，含导出 | 73.051 秒 | 8.121 秒 | 9.00× |
| State | Rollout | 33.262 秒 | 1.570 秒 | 21.18× |
| State | Train | 39.480 秒 | 6.019 秒 | 6.56× |
| State | 完整一轮，含导出 | 73.135 秒 | 7.976 秒 | 9.17× |

这是每种模式各一次首批的实测，不是长期平均或独占 GPU 基准；同卡原有两个服务保持运行。
BF16 舍入改变了部分采样 Tokens，但两侧生成数、训练位置数和输入 Prompt 相同。
奖励变化不能视为训练质量提升。

### 数值检查与适用范围

- 真实首批两种模式的训练 Loss 均为 `-0.14116040`，由 rollout logprob/Advantage 独立计算
  的 Loss 为 `-0.13978841`，绝对差 `0.00137199`。新客户端每批记录 `rollout_loss` 和
  `loss_abs_error`，无需额外 GPU 计算。
- 40-token 前向检查中，同形状 Actor/Learner 和 CPU cache 往返仍逐位相同；不同 Chunk
  `40/2/1/7` 的 logits 不再保证逐位一致，分布 KL 约 `0.001284`。
- 三行、四步批量解码对照逐 Token 单行解码：平均 KL `0.001108`，最大 KL `0.009222`。
  temperature `0.7` 的 39 个教师强制目标 logprob 最大绝对差为 `0.186072`；完全不分块
  的实验方案该值为 `0.514900`，因此保留 32 行分块。
- BF16 测试采用有限精度回归阈值（单次 KL ≤ `0.02`、平均 decode KL ≤ `0.005`、目标
  logprob 最大差 ≤ `0.25`）；strict 测试继续要求跨形状逐位相同。这些是测试覆盖范围，
  不是跨模型/GPU 的误差保证，也没有验证长期 RL 收敛。
- Tiny fixture 的 signed/masked 跨 Chunk 梯度对照、正负方向、零权重样本和 Mask 后缀
  检查通过；全长/Chunk=7 的 MiSS 梯度相对 L2 为 `0.000126899`，State 为 `0.000177266`。
- 两个 Server 的真实 7.2B HTTP 验收通过：缓存、合批与多轮、两次 Adam 更新和发布、版本隔离、
  文件冷加载及 Adam checkpoint 恢复。该短序列用例的 rollout/训练 Loss 绝对差约 `2.64e-7`。
  `rwkv_online_rl_forward_test ... bf16` 的上述 7.2B 数值检查全部通过。
- 最终客户端再跑真实数据前两批，均完成训练、发布及保存：用时 `8.128 / 9.421` 秒，
  新增的 `loss_abs_error` 分别为 `0.00137199 / 0.00044590`。strict 模式的 Tiny + 冷 Adapter
  HTTP 回归，以及 `rwkv_bf16_training_test`、`rwkv_state_tuning_api_test` 也通过。

测试日志与原始测量保存在本机：

```text
/tmp/rwkv-rl-speed/before-miss/
/tmp/rwkv-rl-speed/before-state/
/tmp/rwkv-rl-speed/after-tiled-miss/
/tmp/rwkv-rl-speed/after-tiled-state/
/tmp/rwkv-rl-speed/final-client-miss/
/tmp/rwkv-rl-speed-forward-final.log
/tmp/rwkv-rl-speed-http-7b.log
```

构建后的两个 Server 可直接沿用原启动命令，默认启用 BF16 优化。已运行的进程需要重启
才能加载新二进制；本次没有重启用户原有服务。`/health` 可核对 `gemm_mode`、`gemm_tile_rows`
和 WKV 精度。需要沿用 Adam 状态时仍应使用 `/v1/publish` 的 `checkpoint:true` 与 `--resume`，
并保持 `policy.json` 中记录的 GEMM 模式。

## 2026-10-24：Agentic RL 默认字符串停止

在线 Server 现在直接复用推理引擎的 `include/rwkv/inference/text_stop_matcher.hpp`。
默认停止串为 `</tool_call>`、`</answer>`、`\n\nUser:`，每个 Token 后检查原始生成字节的后缀，
支持跨 Token、跨 UTF-8 字节，命中的 Token 保留在输出、logprob、训练样本和缓存 State 中。
停止串在 Token 中间出现但不是后缀时不会停止；不再使用旧在线路径的任意位置子串查找。

Server 默认 `--max-tokens 1024`，客户端默认从 `/health` 继承该上限和停止串。
上限用于截断复读，不要求每轮生成满额；`--stop` 的 JSON 数组可覆盖默认字符串。
每行还有独立的剩余 context 上限，`finish_reason: length` / `truncated: true` 明确区分
`max_tokens` 和 `context_length`。被截断的轨迹不再追加下一轮观察，已有采样 Token 保持不变。

同一套 7.2B + Adapter + State、真实数据首行、`group-size=4`、seed `1234`：

| 轨迹 | 实际生成 Tokens | 结束原因 | 匹配字符串 | 是否截断 |
|---|---:|---|---|---|
| 0 | 116 | stop | `</tool_call>` | 否 |
| 1 | 112 | stop | `</tool_call>` | 否 |
| 2 | 72 | stop | `</tool_call>` | 否 |
| 3 | 92 | stop | `</tool_call>` | 否 |

共生成 392 Tokens，完整训练、发布、保存成功，一轮用时 9.115 秒（rollout 2.487 秒、
训练 6.244 秒）。Loss 为 `0.07394703`，与 rollout 独立计算值相差 `0.00015761`。
这次按动作边界结束，不能与上面固定生成上限的表直接比较加速比。

推理引擎 matcher 回归、客户端 CPU 多轮/截断/实际上下文预算回归，以及两个模式的 Tiny
和真实 7.2B HTTP 回归通过，覆盖默认停止配置、覆盖/关闭停止、停止与长度上限同时命中、完整停止 Token
的缓存恢复、各行独立 context 截断和 Adam 恢复。

客户端 CPU 回归可用 `python3 test/online_rl_client_test.py` 复现。

```text
/tmp/rwkv-rl-speed/string-stop-e2e/metrics.jsonl
/tmp/rwkv-rl-string-stop-e2e.log
/tmp/rwkv-rl-string-stop-defaults-tiny.log
/tmp/rwkv-rl-string-stop-http-7b.log
```
