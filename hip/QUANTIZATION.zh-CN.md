# W7900 W4A16 / W8A16 移植

[English](QUANTIZATION.md) | 简体中文

HIP 后端接受与 CUDA 相同的 `.rwkvq` 文件，INT4/INT8 投影权重在显存中保持压缩。
本次移植在 Radeon Pro W7900、`gfx1100`、原生 wave32、ROCm HIP 7.2.53211-9999、
AMD Clang 22.0.0、Linux 6.19.6-zen1-1-zen 上验证。
基线仓库提交：`7a3ccf34ac2ac568cb90aa211eeb841ac3d70a60`。

## 构建与使用

在仓库根目录执行：

```sh
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 8
ctest --test-dir build-hip --output-on-failure

build-hip/rwkv_quantize --format w8a16 MODEL.pth MODEL.w8.rwkvq
build-hip/rwkv_quantize --format w4a16 --group-size 128 MODEL.pth MODEL.w4.rwkvq
# 也支持 G32；它使用更多 scale，可能改善量化质量。

build-hip/rwkv_quantized_smoke MODEL.w4.rwkvq
build-hip/rwkv_lighting_cuda --model-path MODEL.w4.rwkvq \
  --vocab-path assets/rwkv_vocab_v20230424.txt
```

可执行文件沿用现有名称，通过文件格式检测选择 W4/W8，无需额外量化推理参数。
已有文件不需要重新量化。PTH 路径仍要求 BF16 来源权重。

## 实现

- W4：NK 顺序的有符号二补码半字节，偶数 K 放在低半字节；每个输出通道、每个 K 分组使用 FP16 scale，支持 G32/G128，接受外部数据中的 -8。
- W8：有符号 INT8 NK 文件权重，每个输出通道一个 FP16 scale。独立 HIP 线性接口也支持原始 KN。CUDA 内存中的 PackedNK 格式、打包辅助函数和调优缓存不属于 HIP 接口。
- 解码（M <= 4）：32 通道归约、FP32 累加、对齐的八元素加载；尾部、非对齐输入和 KN 使用边界受控的标量路径。
- 预填充（M > 4）：gfx1100 原生 `v_wmma_f32_16x16x16_f16`，每个 16x64 输出块使用四个 wave，将合并读取的宽度 64 的 K 块解包至 LDS。保留可移植标量分块实现，并单独测试。
- 基于形状的自动 split-K 最多使用 16 个分区、调用方拥有的工作区和有序 FP32 归约。算子内部不进行内存分配、流同步、设备查询，不使用全局调优状态或原子累加。
- HIP 加载时所有整数投影均保持 NK，包括 `ffn.value`。量化 FFN 先计算 ReLU²，再执行稠密量化降维投影；原有 FP16 稀疏降维投影仍用于浮点路径。
- 注意力 receptance/key/value/output、FFN key/value 和输出头根据张量类型分派。嵌入、归一化和低秩张量仍为浮点。不支持的整数张量角色会被拒绝。不支持量化状态微调，继续强制要求 FP16/BF16 权重。

RDNA3 片段映射遵循 [AMD GPUOpen WMMA 指南](https://gpuopen.com/learn/wmma_on_rdna3/)。
此处只有 gfx1100 经过硬件验证。存在源码回退路径并不能证明支持其他 GPU、操作系统或 ROCm 版本。

## 验证

最终 HIP CTest 套件 **14/14** 通过。新增或移植的检查包括：

- W4 GPU 量化与 CPU 最近偶数舍入的 scale/半字节结果对比，覆盖 G32/G128、全零组、次正规数、外部 -8、奇数 K 和尾部；线性形状最大到 M1024、K16384，覆盖显式/自动 split-K、工作区不足拒绝、双流和重复 Graph 重放。
- W8 的 NK/KN CPU 双精度点积参考，覆盖有符号字节范围、零/次正规 scale、奇数维度、split-K、流和 Graph 重放。
- 非零的**两层**合成模型，覆盖 W4 G32/G128 和 W8，与 CPU 独立反量化的 FP16 投影权重对比。使用 B1/B2、预填充长度 1/7/17/33、三步解码和 FP16/FP32 WKV 状态。每个 logit 必须有限，误差不超过 `0.015 + 0.015*abs(reference)`；非零参考检查避免全零结果误通过。
- 独立标量和 WMMA 构建均通过 W4/W8 CPU 参考测试。

算子对比对每个输出使用容差 `0.0006*sum(abs(products)) + 0.0006*abs(reference) + 0.002`，
考虑 WMMA 中的 FP16 反量化、FP32 累加/归约和 FP16 输出舍入。

真实模型验证使用固定双语输入进行教师强制计算，先 32 token 预填充，再 32 步解码，
随后单独进行 32 token 贪心中文生成，并检查全部输出 logit 的有限性。
原生工具还报告相对参考模型的 logit MSE 和 top-1 一致率：

```sh
build-hip/rwkv_quantized_validate MODEL.w8.rwkvq \
  assets/rwkv_vocab_v20230424.txt MODEL.pth

# 区分移植误差和量化误差：在 CPU 上展开同一份量化权重，
# 然后通过现有 FP16 模型路径执行。
build-hip/rwkv_dequantize_reference MODEL.w4.rwkvq MODEL.w4-fp16-reference.rwkvq
build-hip/rwkv_quantized_validate MODEL.w4.rwkvq \
  assets/rwkv_vocab_v20230424.txt MODEL.w4-fp16-reference.rwkvq
```

参考输出文件必须尚不存在。它用于验证，不是节省显存的推理选项。

## 模型实测结果

模型为 `rwkv7-g1d-0.1b-20260129-ctx8192.pth` 和
`rwkv7-g1i-7.2b-20260805-ctx16384.pth`，SHA256 如下：

```text
0.1B e10d7b1930c2644c5c6b194444774d6d82ec8212a78763493149de09aac7d83f
7.2B 0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8
```

7.2B、B1、FP32 WKV 状态，每种形状预热一次，前向同步后用主机时钟计时，解码取 32 步的
中位数。这些是短时本地测量，不是服务吞吐/负载测试，包含 CPU 工作、分配和循环算子。
权重显存来自加载器的张量字节统计，不包含状态、临时激活、128 MiB 工作区和分配器开销。
三种模式均在 CPU 上额外保留 512 MiB 嵌入表。

| 模式 | 权重显存 | 32-token 预填充 | 解码 | 解码 token/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 PTH / FP16 运行时 | 13,634 MiB | 87.48 ms | 19.995 ms | 50.01 |
| W4 G128 | 4,134 MiB | 120.26 ms | 15.005 ms | 66.64 |
| W8 | 7,237 MiB | 149.65 ms | 16.728 ms | 59.78 |

最初未调优的 W4 实现预填充耗时 1,309 ms、解码 35.46 ms。合并 LDS 加载、向量化解码
和自动 split-K 降低了耗时。虽然解码与显存得到改善，**最终预填充仍比 FP16 慢**。
0.1B 小模型也未证明解码加速，算子启动和主机端开销影响较大。

精度观察（33 个教师强制位置）：

| 对比 | Logit MSE | Top-1 匹配 |
| --- | ---: | ---: |
| 7.2B W8 与原始 BF16 PTH | 0.03519 | 33/33 |
| 7.2B W4 G128 与原始 BF16 PTH | 2.689 | 26/33 |
| 0.1B W4 G32 与原始 BF16 PTH | 2.820 | 24/33 |
| 0.1B W4 G128 与其 CPU 反量化 FP16 参考 | 0.001327 | 32/33 |

最后一组相对 logit RMSE 为 0.1665%，贪心文本相同；它衡量移植的算术差异，而非降低精度
造成的损失。最初 0.1B G128 与原始模型对比的 MSE 约 19.57，说明小模型可能对该量化器
敏感。成功执行不代表质量等同 FP16。此处不声称 W4 已通过长上下文困惑度或任务质量
验收；W8 同样是近似计算。

## AMD 技能 / Magpie 验证依据

使用 AMD 目录的 [magpie-kernel-evaluator](https://github.com/amd/skills/tree/main/skills/magpie-kernel-evaluator)
流程，Magpie 提交为 `d4de63dffe8df0229a88d4c364442051387ec00e`。
其兼容矩阵未列出 W7900，以上结果来自本地硬件验证。
本次工作区会话中的 Magpie 位于 `/tmp/rwkv-magpie-quant-migration`，不是运行时依赖。

```sh
PYTHONPATH=/tmp/rwkv-magpie-quant-migration python -m Magpie analyze \
  -k hip/quantized-magpie.yaml --no-perf \
  --output-dir build-hip/quant-validation/magpie
PYTHONPATH=/tmp/rwkv-magpie-quant-migration python -m Magpie --workers 1 compare \
  -k hip/quantized-compare-magpie.yaml \
  --output-dir build-hip/quant-validation/magpie
```

Analyze 通过编译和数值测试。Compare 通过标量和 WMMA 的正确性检查，并执行自定义
HIP 事件计时。Magpie 自定义分析器只在 JSON 中暴露完成标志，未保留计时标准输出；
得到的 `winner: 0` **不是性能排名**。应使用各变体的 `timings.csv` 产物。
此原生 C++ 后端未提供 TraceLens/torch 报告。

独立 HIP 事件对比使用种子 42、三次预热、20 次计时调用和相同自动 split 工作区。
示例时间（微秒）：

| 位数 | M | K | N | 标量分块 | WMMA 分块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | 4096 | 4096 | 144.80 | 137.87 |
| 4 | 64 | 4096 | 4096 | 671.94 | 513.72 |
| 8 | 16 | 4096 | 4096 | 153.70 | 164.64 |
| 8 | 64 | 4096 | 4096 | 655.74 | 638.17 |

WMMA 并非在所有 W8 形状上更快，仍可逐形状调优。M1/M4 在两个变体中使用相同向量化实现。
这些算子时间与完整模型计时表独立。

另一次 `rocprofv3 --kernel-trace --stats` 运行成功，算子追踪/统计 CSV 保存在
`build-hip/quant-validation/rocprof/`。该混合形状微基准中，W8 GEMM 占被追踪算子时间的
49.08%，W4 GEMM 占 42.84%。这说明矩阵乘法仍是该工作负载的主要开销，不是端到端
或 roofline 报告。上面的无分析器对比不使用开启分析器时的事件计时结果。

产物保留在 `build-hip/quant-validation/`：量化模型、CPU 反量化参考、导出日志、
真实模型日志、最终 CTest 日志、独立验证二进制和 Magpie 报告。
这些生成文件不纳入源码管理；可复现配置、测试和工具包含在移植变更中。
