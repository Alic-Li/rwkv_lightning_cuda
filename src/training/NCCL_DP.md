# NCCL 单机多 GPU 训练

`feature/nccl-dp` 在共享 BF16 训练器上增加三个独立程序：

- `rwkv_state_tune_dp`：initial state，支持 Adam / Muon。
- `rwkv_miss_tune_dp`：MiSS D 参数，支持 Adam。
- `rwkv_nora_tune_dp`：NoRA A/B 参数，支持 Adam。

原来的 `rwkv_*_tune` 保留；NCCL 只链接到新增的 DP 目标。当前实现限 Linux + CUDA 单机，不需要 torchrun、MPI 或 PyTorch。

## 编译

安装 CUDA、NCCL 开发头文件/动态库和项目原有依赖后：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DRWKV_NCCL_DP=ON
cmake --build build --target \
  rwkv_state_tune_dp rwkv_miss_tune_dp rwkv_nora_tune_dp -j 12
```

`89` 适用于 RTX 4090。其他 GPU 按实际架构设置。非系统安装的 NCCL 可以传 `-DNCCL_ROOT=/path/to/nccl`，或显式设置 `NCCL_INCLUDE_DIR`、`NCCL_LIBRARY`。

`RWKV_NCCL_DP=AUTO` 为默认值：有 NCCL 就构建 DP，否则继续构建原有程序；`ON` 在依赖缺失时明确报错；`OFF` 不构建 DP。HIP 原有构建不依赖 NCCL。

## 四卡启动

```bash
cd /home/no22/rwkv_lightning_cuda
./build/rwkv_nora_tune_dp \
  --model /home/no22/rwkv-models/rwkv7-g1k-2.9b-20260930-ctx25600.pth \
  --vocab ./assets/rwkv_vocab_v20230424.txt \
  --data /home/no22/segments-2-3.jsonl \
  --output train-2-3-g1k/g1k-nora-r16-lr2e-5-e2 \
  --targets all --ctx 16384 --chunk 512 --wkv_tape \
  --batch-size 32 --micro-bsz 6 --devices 0,1,2,3 \
  --warmup-steps 10 --save-every 186 \
  --rank 16 --alpha 16 --lr 0.00002 --lr-final 0.000002
```

原命令没有 `--epochs 2`，因此仍使用默认 **1 epoch**；输出目录名中的 `e2` 不会改变训练轮数。如需两轮训练，显式添加 `--epochs 2`。

`--devices` 是当前 CUDA 可见设备的索引；设置 `CUDA_VISIBLE_DEVICES` 后按映射后的编号填写。默认仅使用 `0`。GPU 不可重复，非法设备会让整个作业退出。

`--batch-size` 是**全局**样本数，`--micro-bsz` 是**每卡**最多并行的 packed lanes。32 样本在四卡上合计仍是一次 32 样本更新，而不是 128。根据样本长度和 micro-batch 的 chunk sweep 开销，各卡分到的样本数可以不同；每个样本只由一张卡计算，原有全局 batch 顺序、有效 token 数、学习率步数保持不变。没有 padding 样本、重复补样本或丢弃尾批次。特别小的全局 batch 会自动限制分配的 lanes，空闲 rank 以零梯度参与归约。

MiSS 使用对应 `_dp` 程序和相同 adapter 参数。State 使用 `rwkv_state_tune_dp`，不接受 `--rank/--alpha/--targets`；学习率沿用 State 训练自己的设置。

## 梯度、检查点和错误处理

每个 worker 独占一张 GPU，持有完整 frozen backbone、可训练参数、FP32 optimizer state 和独立执行器。父进程在任何 CUDA/NCCL 初始化之前创建 worker，通过匿名管道分发 NCCL unique ID，并监控所有 worker。

所有 rank 使用相同文件、seed 和配置构建同一全局 batch。Backward 使用该 batch 的**全局有效目标 token 数**作为分母，随后对 FP32 可训练梯度执行 `ncclAllReduce(..., ncclSum)`。不会再除以 GPU 数。所有 rank 运行相同 optimizer update；全局 loss 也做求和。State 的每 lane 梯度直接在 GPU 上累加，归约后检查有限性；adapter 在全局归约后由共享 Trainer 检查梯度再更新。

只有 rank 0 写检查点和最终导出。写完后 collective barrier，防止其他 rank 提前开始下一步。MiSS/NoRA 的 `--resume checkpoint-N` 会在所有 rank 上恢复同一参数、Adam moments、global step 和数据行游标。DP 检查点额外记录 world size、实际 micro-bsz、分配策略和 GEMM 模式；改变这些设置会被拒绝，防止配置变化改变续训语义；默认 bf16 模式已经过逐 bit 续训验证。State 沿用现有 state-only 导出，不包含 optimizer resume。

任一 worker 非零退出时，父进程终止并回收其余 worker，返回失败状态；SIGINT/SIGTERM 同样清理整个作业。父进程被强制终止时，Linux parent-death signal 会终止 worker。失败时不会无限留着其他 rank 等待 NCCL。

NCCL communicator / collective 接口遵循 NVIDIA 官方文档：[communicators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)、[collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html)。

## 显存与负载

原执行器会按每个 micro-batch 的 chunk 数缩放 checkpoint vector，短序列释放显存，长序列再分配，产生显存波动和同步开销。现在 checkpoint buffer 保留历史最大容量；DP 启动时进一步按 `ctx` 一次预分配边界 state/shift checkpoints。短序列和不足 lanes 的尾批次复用这些地址，训练循环不再分配/释放边界 checkpoint。

预分配使显存需求在启动阶段暴露，不会因后续更长样本突然增长。总显存仍取决于模型、`chunk * micro-bsz` 激活，以及 `ceil(ctx/chunk) * micro-bsz` 个边界状态。特别长的 ctx 会增大预分配；显存不足时降低 micro-bsz，或结合激活与边界状态的开销调整 chunk。模型加载和进程退出期间显存升降属于正常生命周期。

可加 `--dp-profile` 查看各 rank 每步的样本数、token 数、`compute_s`、`reduce_wait_s` 和显存。`reduce_wait_s` 包括等候较慢 rank 的时间，不等于纯 NCCL 带宽时间。

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1
```

GPU 利用率、实际 tokens/s 和同步等待应一起看；collective 等待也可能显示 100% 利用率。450W 是功率上限，训练不保证每张卡持续达到该值。当前实现通过并发计算、packed micro-batch 和工作量分配提高有效负载，没有添加空转或修改硬件功率限制。

## 验证

```bash
# 测试驱动需要 PyTorch；训练程序不需要。
python tools/check_nccl_dp.py --build build \
  --work /tmp/rwkv-dp-qa-new --devices 0,1,2,3
```

测试覆盖三种训练、Adam/Muon、单卡/四卡 loss 和参数/Adam moments 对比、不同长度与 train mask、过滤样本、尾批次空 rank、MiSS/NoRA 精确续训、错误配置、重复/非法设备、rank 0 检查点 I/O 失败和取消作业后的子进程回收。工作目录必须不存在，避免覆盖已有检查点。

可以在有 PyTorch 的本机驱动远端测试，远端只需 Python 标准库和已编译程序：

```bash
python tools/check_nccl_dp.py \
  --build /home/no22/rwkv_lightning_cuda/build \
  --work /tmp/rwkv-dp-qa-remote-new --devices 0,1,2,3 \
  --host no22@100.64.0.4 --remote-work /home/no22/dp-qa-new \
  --ssh-socket /tmp/rwkv-nccl-ssh
```

已有 `rwkv_bf16_training_test` 和 `rwkv_packed_batch_test MODEL.pth` 继续验证 BF16 算子、chunk 梯度传递和 packed 执行器。

本次机器上的测试结果、吞吐、功耗和显存采样见 [验证记录](NCCL_DP_VALIDATION.md)。
