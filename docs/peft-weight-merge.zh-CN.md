# CPU MiSS / NoRA 权重合并

`rwkv_peft_weight_merge` 是独立的离线 CPU 工具。合并逻辑和流式 ZIP64 写入代码位于
`src/rwkv_peft_weight_merge.cpp`，复用仓库的 CPU PTH 读取与指纹函数，不链接推理 Server、
训练库、CUDA/HIP 或 libtorch。一次将一个 MiSS 或 NoRA Adapter 合入其训练基模，输出可直接
加载推理的新 `.pth`。不会修改输入文件，也不会为 Server 增加动态 Adapter 功能。

## 构建

普通工程构建包含该目标：

```bash
cmake --build build --target rwkv_peft_weight_merge -j
```

也可以在没有 GPU SDK 的机器上单独构建，只需 C++17、CMake、JsonCpp、OpenSSL Crypto、
zlib；OpenMP 可选，有则启用多线程，没有则使用单线程。

```bash
cmake -S . -B build-peft-cpu -DRWKV_PEFT_MERGE_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-peft-cpu -j
```

## CI/CD 与发布

CI 的 Linux/Windows × CUDA 12.9/13.2 发布矩阵默认编译并打包该 CPU 工具，
包含它的运行库与本说明，并执行打包后的 `--help`、`--version`、`-v` 检查。
版本号来自仓库 `VERSION`，与其他二进制一致。

手动运行 GitHub Actions 的 **CI and Release** 时，可通过默认勾选的
`build_peft_merge` 开关控制是否编译和打包。PR 与 main push 默认启用。
本地运行发布脚本时，对应环境变量为 `RWKV_BUILD_PEFT_MERGE=ON/OFF`。
完整 CMake 工程可用 `-DRWKV_BUILD_PEFT_MERGE=OFF` 关闭该目标；
默认启用，并提供 `bundle_rwkv_peft_weight_merge` 打包目标。
`RWKV_PEFT_MERGE_ONLY=ON` 则始终只构建该工具，不受完整工程开关影响。

## 使用

```bash
# MiSS
./build-peft-cpu/rwkv_peft_weight_merge \
  --method miss \
  --model /path/to/base.pth \
  --adapter /path/to/miss/adapter-final.pth \
  --output /path/to/merged-miss.pth \
  --threads 16

# NoRA
./build-peft-cpu/rwkv_peft_weight_merge \
  --method nora \
  --model /path/to/base.pth \
  --adapter /path/to/nora/adapter-final.pth \
  --output /path/to/merged-nora.pth \
  --threads 16
```

`--method` 必填，大小写不敏感，接受 `MiSS` / `NoRA`。
`--scale` 可选，默认使用 Adapter metadata 中的有效 scale；指定时是**替换**该值，
不是再乘一次。支持零与负数，零保留原始权重位模式。NoRA 导出的默认 scale 为 1。
`--threads` 必须为正整数；未指定时使用 OpenMP 默认线程数。

输入支持本仓库导出的 `adapter-final.pth`、包含 `adapter.pth` / `adapter.json` 的包目录，
以及带内嵌 metadata 的 `training.pth` 或其 checkpoint 目录。训练 checkpoint 的 FP32 master
先舍入到 metadata 声明的 forward dtype，使合并使用的因子与导出 Adapter 一致；不合并
optimizer、梯度或初始 State。没有内嵌 metadata 的旧 checkpoint 应先通过训练工具导出。
不支持 `.rwkvq` 等量化基模；需要时先合并，再运行量化工具。

## 计算与文件行为

基模 Linear 权重布局为 `[out, in]`，FP32 累加后舍入回该权重原来的 dtype：

- MiSS：`W[o, i] += scale * D[o, i % rank]`，支持输入宽度不整除 rank。
- NoRA：`W += scale * (B @ A)`，不对已训练 A 再做归一化。

保留基模 FP32 / FP16 / BF16 dtype、张量名称、形状和未修改 storage 区间。
目标权重需为连续的二维矩阵；支持同一个 storage 中不重叠的多个权重和非零 storage offset。
目标与其他张量重叠时拒绝合并，以免隐式修改不应修改的权重。

工具校验 Adapter 内容摘要、基模完整 SHA256、method、target、形状、dtype 与有限数值。
基模必须是训练时使用的原文件，不提供绕过指纹校验的选项。已有输出不会覆盖；合并失败不发布
半成品。成功后通过同目录临时文件的硬链接原子发布，因此输出文件系统需支持硬链接。
支持 PyTorch 默认的未压缩 little-endian ZIP PTH，使用 ZIP64 输出，支持超过 4 GiB 的模型。

内存使用主要为 Adapter 因子、一份当前目标权重、每个 CPU 线程的一行 FP32 临时结果，
以及 4 MiB 流式复制缓冲；不读入整份基模，也不生成整份 FP32 基模或 delta 副本。
合并后的一次 Linear 与原本分开执行的 Base/Adapter Linear 舍入位置不同，因而不要求
逐 token 完全一致，尤其是稀疏 FFN 推理。

## 验证

CPU 数值与文件回归使用 PyTorch 作为独立参照（仅测试需要 Python/PyTorch）：

```bash
python test/peft_weight_merge_test.py --binary build-peft-cpu/rwkv_peft_weight_merge -v
```

覆盖两种算法、三种基模 dtype、两种 Adapter dtype、导出文件/训练 checkpoint、共享 storage、
非零 offset、ragged rank、rank 大于输入宽度、默认/覆盖/零/负 scale、ZIP CRC、错误指纹、
错误 method、内容损坏、溢出与输出保护。

2026-11-07 本机验收：

- 无 GPU SDK 构建路径与常规工程目标均构建成功，`ldd` 无 CUDA/HIP/libtorch 依赖。
- 10 项 CPU 回归通过，数值矩阵额外包含 24 种 method/dtype/checkpoint 组合。
- 正式 `rwkv7-g1l-7.2b-20261031-ctx34304.pth` 分别合并 MiSS 和 NoRA，均输出
  14,400,007,704 字节文件。每种结果的 192 个目标张量均抽查首/中/末行，与 PyTorch
  FP32 公式舍入结果完全一致；870 个非目标张量全量逐值相等。ZIP CRC 和 `torch.load` 通过。
- 两种合并模型均通过 10/10 原生 HTTP 冒烟检查：Chat/Batch、普通/SSE、动态 State、
  混合连续批处理及清理。未比较稀疏 FFN 的逐 token 一致性。
- 详细报告及测试产物位于 `build/peft-merge-validation/`，包括 `unit.log`、
  `real-model-report.json`、`miss-http/report.json`、`nora-http/report.json`。
