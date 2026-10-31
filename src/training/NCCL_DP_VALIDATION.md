# NCCL DP 验证记录（2026-10-31）

工作分支：`feature/nccl-dp`。远端代码和二进制位于 `no22@100.64.0.4:/home/no22/rwkv_lightning_cuda`。

## 正确性与工程验证

本机 RTX PRO 6000 Blackwell、远端 4 × RTX 4090 均完成构建。远端 CUDA 架构为 `89`，本机为 `120`。`RWKV_NCCL_DP=OFF` 配置仍保留原单卡目标，不需要 NCCL。

最终代码的 `tools/check_nccl_dp.py` 在本机单 GPU 和远端四 GPU 通过：

- State / MiSS / NoRA 的单卡与 DP loss、参数、Adam moments 对照通过；跨 rank 求和使用容差，而非承诺跨 world size 逐 bit 相等。
- 不等长样本、不同 train mask、完全屏蔽/过短样本过滤、batch=7 和不足 batch 的最后两个样本，验证全局有效 token 归一化和空 rank 归约。
- State 的 Adam、Muon 均通过。
- MiSS / NoRA 从 checkpoint-1 恢复到 step 2，所有 checkpoint tensor 与连续训练**逐 bit 一致**。
- 不匹配的恢复配置、重复 GPU、某一 rank 使用非法设备、仅 rank 0 检查点路径冲突，均返回失败并回收其余 worker。
- SIGTERM 取消后，launcher 返回 143，所有直接 worker 均被回收。
- `rwkv_bf16_training_test`、`rwkv_packed_batch_test` 在本机及远端通过。

远端最终验收目录：`/home/no22/dp-qa-final`。

## 真实模型

- 远端 **2.9B / 四卡 / NoRA**：用户参数 `ctx=16384, chunk=512, batch=32, micro-bsz=6, rank=alpha=16, lr=2e-5, lr-final=2e-6`；默认 BF16 模式、auto 模式分别完成 10 步并成功导出。
- 远端 **2.9B / 四卡 / State**：相同 ctx/chunk/batch/micro-bsz，完成 2 步，成功写入两个 state checkpoint 和最终 state。此轮使用原 State 默认学习率，验证执行/导出，不用于判断收敛。
- 远端 **2.9B / 四卡 / MiSS**：相同 adapter 参数，完成 2 步，成功写入两个可恢复 checkpoint 和最终 adapter。
- 本机 **7.2B / 单卡 DP / NoRA**：用户参考参数 `ctx=16384, chunk=2048, batch=16, micro-bsz=4, rank=alpha=16`，完成 1 步并成功导出；30,204 个处理位置、loss=0.5556、约 4,049 tokens/s、CUDA 已用显存约 81,368 MiB。

上述为短程工程验证，没有完成全量 epoch，也不据此判断模型最终质量。

## 四卡性能与负载

同一 2.9B 模型、同一数据前 10 个全局 batch、相同训练参数；吞吐为 CLI 统计的**所有处理 token 位置**（包括 masked prompt），不是仅监督 target 的速度。不计模型加载时间，包含数据处理、同步、更新和首步热身。

| 分配方式 / GEMM | 累计吞吐（tokens/s） | 各 rank 的平均归约及等待时间（秒/步） |
|---|---:|---:|
| 初始实现：限制每卡样本数 / bf16 | 10,238.3 | 1.369 |
| 仅均衡 token 数 / bf16 | 10,858.3 | 0.997 |
| 最终实现：token + packed chunk sweep 成本 / bf16 | 12,191.4 | 0.301 |
| 最终实现 / auto | 13,029.7 | 0.180 |

最终默认模式较初始实现提高约 **19.1%**；显式 `--gemm-mode auto` 比最终默认模式再提高约 **6.9%**。每步最晚 rank 的归约通常约 0.03 秒，说明主要改善来自减少卡间负载不均。此结果不是单卡到四卡的 scaling 对比。

`auto` 允许 cuBLAS 按实际形状选择算法，会改变部分 BF16 舍入，因此默认仍为原有 `bf16` 模式。追求这台服务器上实测更高吞吐时，在用户原命令末尾添加 `--gemm-mode auto`；续训保持相同模式。

每秒一次 `nvidia-smi` 采样，最终 auto 运行的 54 个显存稳定采样点（含步骤间隙和边缘采样）：

| GPU | 平均利用率 | 平均功耗 | 峰值功耗 | 显存 min–max |
|---|---:|---:|---:|---:|
| 0 | 96.50% | 306.22 W | 412.27 W | 23,485–23,485 MiB |
| 1 | 99.91% | 334.61 W | 406.78 W | 23,485–23,485 MiB |
| 2 | 96.65% | 368.23 W | 422.52 W | 23,485–23,485 MiB |
| 3 | 95.39% | 365.79 W | 417.63 W | 23,485–23,485 MiB |

最终默认 `bf16` 运行的 58 个稳定采样点中，四卡显存均为 **23,477–23,477 MiB**；CUDA API 的已用值为 23,476 MiB。`auto` 多出的约 8 MiB 为首次 GEMM 后常驻的工作空间。模型加载时显存逐渐增加，进程退出时释放，未纳入稳定窗口。

实际计算的算术强度和长序列尾部 active lanes 数会影响功耗。没有修改 450W 功率上限，也没有通过空转制造功耗。

## 日志与复现

远端 `/home/no22/dp-validation/` 保留：

- `nora-bf16-profile.log`：初始按样本数限制的基准。
- `nora-token-balance.log`：仅 token 均衡对照。
- `nora-packed-balance.log`：最终默认模式。
- `nora-auto-profile.log`：最终 auto 模式。
- `nora-2.9b-b32-m6-gpu.csv`：从首次试跑开始连续采样，包含上述运行及运行之间的空闲时间；不能把整份 CSV 的均值作为单次训练均值。
- `state-2.9b.log`、`miss-2.9b.log`、各自导出/检查点目录。
- `acceptance-final.log`、`packed-final.log`：最终回归记录。

完整构建、启动、参数和验证命令见 [NCCL_DP.md](NCCL_DP.md)。
