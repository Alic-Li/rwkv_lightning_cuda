# 原生 SB6 非对称 W4/W6A16

本实现参考 [RafaelUI/rwkv-quant](https://github.com/RafaelUI/rwkv-quant)
的非对称分组量化思路，在本项目的 RWKVQ 容器和 CUDA 后端中独立实现。
它不是外部项目的文件格式适配器；推理不依赖 Python 或参考仓库。
逐行 W8 保持原有格式和算子。

## 磁盘格式与精度

容器魔数仍为 `RWKVQ\0\1\0`，dtype 4、6 分别表示
`kInt4Asym`、`kInt6Asym`。dtype 0/1/2 为 BF16/FP16/W8；旧 dtype 3、5 已退役，
新读取器明确拒绝它们，不复用编号。已有 W8 文件继续兼容。

矩阵保持 NK 排列，`quant_group_size=32`。每行独立补齐至 32 个码值，
每组以如下次序连续存储，所有多字节字段均为小端：

| 偏移 | 内容 | 适用档位 |
|---:|---|---|
| 0–15 | 32 个低四位码，偶数列占低 nibble | W4/W6 |
| 16–19 | uint32，第 j 位为第 j 个码的 bit 4 | W6 |
| 20–23 | uint32，第 j 位为第 j 个码的 bit 5 | W6 |

每行的每 256 个权重组成一个超块（8 组）。最后一个超块独立补齐，
紧凑元数据放在容器原有的 scales 区域：

| 字节偏移 | 内容 |
|---:|---|
| 0–1 | FP16 `d`，有限且非负 |
| 2–3 | FP16 `dm`，有限、可为负 |
| 4–9 | 8 个无符号 6-bit `qs[g]`，第 g 项位于 bit `6*g` |
| 10–15 | 8 个无符号 6-bit `qm[g]`，相同排列 |

该区域的 `scale_count = N * ceil(K/256) * 8`，计数单位为原始
uint16 字，不表示全部字段都是 FP16。读取器检查大小、基准和展开结果。
加载时一次性展开为每组 `half2(s,m)`，其中
`s=round_fp16(d*qs)`、`m=round_fp16(dm*qm)`，采用 ties-to-even。
磁盘元数据不保留在设备上。

**16 bytes / 256 weights = 0.50 bits/weight**，因此对齐矩阵的磁盘
量化部分为 4.5/6.5 bpw，显存量化部分为 5/7 bpw。
完整文件还包含浮点张量、头部和尾部补齐，不能用上述值代替完整模型大小。
每个超块共享一个带符号 `dm`，所以各组偏移不能同时取相反符号；编码器
采用最大绝对偏移的符号，将另一侧投影到零，并按实际投影误差评分。

## AW 与加权最小二乘

`tools/calibration/sb6.py` 搜索 41 个初始端点组合：基线 `(1,1)`，以及
单独改变下端点或上端点的 20 个系数，范围 0.70–1.20、步长 0.025，
去掉 1.0。固定初始整数码后，求加权最小二乘的 scale 和 offset。
原始候选与 LS 候选均投影到 SB6 和 FP16 参数网格，重新计算无符号码，
按投影后的加权误差选择整个超块。不会将不同候选的共享基准混用。

激活重要性为真实线性输入的 `E[x²]`。共享参数的评分保留八组之间的
相对激活能量，只对整个超块统一归一化；零能量组使用均匀权重。
尾部补齐不参与端点和误差统计。退化 LS 使用初始候选；正 scale
至少为 FP16 最小次正规数。展开溢出会报错。

## CUDA 执行

`quant/gemmv/rwkv_asym_a16.cu` 用模板实现两个位宽。
M>4 且 K/N 对齐时直接在寄存器中生成 MMA B 操作数：低四位与高位平面
注入 `0x64006400` 的 FP16 尾数，一次 `__hsub2` 和一次 `__hfma2`
完成反量化。SM75 使用 m16n8k8，SM80+ 使用 m16n8k16。
不规则尺寸使用带边界掩码的 WMMA 路径，不分配完整反量化权重。

M≤4 的 GEMV 按组计算 `s*sum(q*x)+m*sum(x)`，在 FP32 中累加。
通用路径将 X 和分组和放入 shared memory；K 为 256 的倍数且工作区
有额外空间时，一次预计算激活和供全部输出通道复用。无工作区时可以
执行单 split；强制多 split 而工作区不足时抛出错误。工作区查询返回
split partials 的最低容量；额外的 `M*ceil(K/32)*sizeof(float)` 可启用
上述缓存优化。调用方应为并发 stream 分配独立工作区。

所有 kernel 在调用方 stream 上提交，无设备分配、同步或运行时调优，
支持 CUDA Graph。FFN value 保持 NK 排列，在 ReLU² 之后进入该分发。
MMA 将反量化值舍入至 FP16；GEMV 使用 FP32 分组恒等式，两者有正常的
舍入差异。Python 校准使用展开参数的 FP32 权重，最终质量应以原生 CUDA
评测为准。

## 导出与验证

见[校准工具说明](../tools/calibration/README.md)和[统一验证报告](VALIDATION.zh-CN.md)。
Python 负责 AW/LS 参数搜索，C++ `rwkv_quantize --format w4a16_asym` 或
`--format w6a16_asym` 必须提供 `--calibrated manifest.json`，验证后事务性写入。
没有第二套 C++ 参数搜索器。默认 W4 保护 layer 0 的 attention output 为 BF16，
W6 默认量化全部目标投影；Embedding 和 LoRA 保持浮点。

`rwkv_sb6_kernels_test` 覆盖位平面、元数据、GEMV、寄存器 MMA、不规则尺寸、split-K、
双流和 Graph；`rwkv_quantized_model_test` 使用非零两层模型，将实际压缩推理与
独立展开的 FP16 模型比较。Python 测试覆盖加权最小二乘、投影后误差、尾部、
原生归档往返及退役格式拒绝。HIP 尚未实现 SB6，并在加载时明确报错。
