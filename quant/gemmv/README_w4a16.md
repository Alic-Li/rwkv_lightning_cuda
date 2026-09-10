# W4A16 CUDA 算子

入口：`rwkv_w4a16.cuh` / `rwkv_w4a16.cu`。计算 `Y[M,N] = X[M,K] × W[N,K]^T`，输入、输出及 scale 为 FP16，累加为 FP32。

## 量化格式

- 沿每个输出通道的 K 维分组，支持 group size 32 和 128，默认 G128。
- 每组 `scale = FP16(max(abs(w)) / 7)`，最小非零 scale 为 FP16 最小正子正规数；全零组使用 1。
- 使用存储后的 FP16 scale 做 round-to-nearest-even，限制到 `[-7,7]`。量化除法显式使用精确舍入，避免 `--use_fast_math` 改变半整数临界值。
- 每个字节存两个二补码 int4，偶数 k 在低 nibble，奇数 k 在高 nibble。算子也接受外部打包的 -8。
- `qweight` 为 `[N, ceil(K/2)]`，奇数 K 的最后一个高 nibble 补 0；`scale` 为 `[N, ceil(K/G)]`。
- 包含 FP16 scale，G128 约占 0.515625 字节/权重，G32 约占 0.5625 字节/权重；相对原逐行 W8 接近减半。

参考 [Palm-Infra quantize_weight.cpp](https://github.com/TencentYoutuResearch/Palm-Infra/blob/eacc4ac6cffe4dd87ecb3d88b7c660f38702343c/tools/quantize_weight.cpp) 的分组对称量化和双 nibble 思路，以及 [Metal kernel](https://github.com/TencentYoutuResearch/Palm-Infra/blob/eacc4ac6cffe4dd87ecb3d88b7c660f38702343c/kernels/metal/mollm.metal) 的即时解包思路。CUDA 实现沿用本项目 W8A16 的 stream 接口和 MMA/共享内存 swizzle 方式。这里的 NK + 独立 FP16 scale 布局与 Palm 的 BG32/BG128 文件布局不同，不能直接读取其打包文件。

## 调用示例

### 导出并加载完整模型

```bash
cmake --build build --target rwkv_quantize rwkv_quantized_smoke -j

build/rwkv_quantize \
  --format w4a16 --group-size 128 \
  /path/to/model.pth /path/to/model.w4a16.rwkvq

# 直接验证 archive 加载、prefill 和 decode
build/rwkv_quantized_smoke /path/to/model.w4a16.rwkvq

# 服务端与 W8 使用同一个加载入口
build/rwkv_lighting_cuda \
  --model-path /path/to/model.w4a16.rwkvq \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt
```

不写 `--format` 时保持原来的 W8A16 导出行为。W4 archive 的线性权重保存为
INT4，group size 随张量写入；embedding、layer norm、低秩矩阵和其他非线性
权重仍保存为 BF16。加载器按 archive dtype 自动选择 W4A16，无需额外推理参数。
`ffn.value` 的 W4 权重保留 NK 布局并走 dense W4A16；现有稀疏 FFN 融合核只用于
W8。G128 占用更低，建议作为默认；G32 有更多 scale，通常精度更好。

### 直接调用算子

以下指针均为已分配的 device buffer，分配和量化在模型准备阶段完成：

```cpp
#include "rwkv_w4a16.cuh"

const int G = 128;
const auto q_bytes = rwkv7_w4a16_weight_bytes(N, K);
const auto scales_count = rwkv7_w4a16_scale_count(N, K, G);
// d_q: q_bytes 字节；d_scale: scales_count 个 half。
rwkv7_w4a16_quantize_launch(stream, d_weight_nk, d_q, d_scale, N, K, G);

// 每个并发 stream 独立分配，cudaMalloc 的对齐满足快路径。
const auto ws_bytes = rwkv7_w4a16_workspace_bytes(M, N, 16);
// d_workspace: ws_bytes 字节；d_y: M*N 个 half。
rwkv7_w4a16_linear_launch(stream, M, K, N,
                         d_x, d_q, d_scale, d_y, G,
                         d_workspace, ws_bytes); // 自动选择 split-K
```

`force_split_k=1` 关闭 split-K；显式 `>1` 要求足够的 workspace，否则抛出 `std::invalid_argument`。自动模式最多使用 16 个 split（M>64 时最多 8 个），workspace 不足时退回单 split。单 split 不需要 workspace。形状非正数时不执行；group size 只能是 32 或 128。量化输入要求有限 FP16 数值。

decode 快路径一次读取 8 个权重和 8 个激活。对齐的多请求路径使用 16×64 / 64×64 输出 tile，在寄存器中解包并使用 Tensor Core；SM75 使用两次 m16n8k8，SM80+ 使用 m16n8k16。非对齐及尾部维度有带边界检查的 GEMV/WMMA 路径。Tensor Core 解包后的权重会舍入为 FP16，与 FP32 解包的 GEMV 存在正常浮点误差。

kernel 不分配临时内存，不同步 stream，不查询设备，不使用全局 tuning 状态或 atomicAdd。split-K 写各自 FP32 分片后固定顺序归约。共享权重可供多个 stream 同时读取，输出和 workspace 必须独立；量化完成后再通过 stream 顺序或 event 建立读取依赖。输入、权重、scale、输出和 workspace 不应相互重叠。可在预先分配好 buffer 后捕获 CUDA Graph。

## 构建和验证

```bash
cmake -S . -B build
cmake --build build --target rwkv_w4a16_kernels_test rwkv_w4a16_model_test rwkv_w4a16_bench -j 4
ctest --test-dir build -R '^rwkv_w4a16_(kernels|model)_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/test/rwkv_w4a16_kernels_test
compute-sanitizer --tool racecheck --error-exitcode 1 build/test/rwkv_w4a16_kernels_test
build/rwkv_w4a16_bench 4096 4096 300
build/rwkv_w4a16_bench 16384 4096 300
```

测试覆盖 G32/G128、逐 nibble/scale 对照、零组、子正规数、-8、奇数 K、N/M 尾部、M=1…128、对齐快路径、无 workspace、显式/自动 split、不足 workspace、双 stream 和 Graph 重放。benchmark 使用相同原始权重与激活，对照现有 PackedNK W8A16，报告 CUDA event 延迟及 split 1/2/4/8/16/32；额外报告四条 stream 的总耗时除以总调用次数，用于观察吞吐，不是单请求响应延迟。准备、量化和分配不计入计时。

`.rwkvq` archive、`rwkv_quantize` CLI、模型加载与服务端已接入 W4。测试包含
一个运行时生成的完整最小 W4 archive，并通过真实 `ModelBackend` 执行 prefill
和 decode。当前仍未在实际大模型上进行 perplexity / logits / 任务质量评估；
正式模型的 W4 精度需要另行校准验证。

## 本机实测（2026-09-10）

RTX PRO 6000 Blackwell Workstation Edition，CUDA 13.3.73，driver 610.57.04，Release；每项预热 10 次、计时 300 次。使用现有 W8 默认分派（未载入模型 tuning 文件），W4 使用自动 split。下表是该次 event 平均值，桌面显示负载会带来波动，不代表所有显卡或模型的速度。

| K | N | M | G | W8 μs | W4 μs | W8/W4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 4096 | 1 | 128 | 6.944 | 5.457 | 1.273 |
| 4096 | 4096 | 8 | 128 | 6.129 | 5.904 | 1.038 |
| 4096 | 4096 | 32 | 128 | 12.726 | 10.961 | 1.161 |
| 4096 | 4096 | 128 | 128 | 36.162 | 23.883 | 1.514 |
| 16384 | 4096 | 1 | 128 | 12.914 | 16.495 | 0.783 |
| 16384 | 4096 | 8 | 128 | 15.647 | 19.154 | 0.817 |
| 16384 | 4096 | 32 | 128 | 33.959 | 34.031 | 0.998 |
| 16384 | 4096 | 128 | 128 | 79.299 | 79.714 | 0.995 |

4096 方阵的大 batch 有收益，但长 K（16384）的若干场景仍落后于 W8；不要将 int4 的存储压缩比例理解为吞吐提升比例。四 stream 的数据用于同权重并发读取场景；不同模型竞争显存带宽可能不同。完整数据：[K4096](../../docs/benchmarks/w4a16_rtx_pro_6000_4096.csv)、[K16384](../../docs/benchmarks/w4a16_rtx_pro_6000_16384.csv)。

当前 CMake 的 SM75/80/86/87/89/90/100/120 编译通过；本机测试、memcheck（0 errors）、racecheck（0 hazards）通过。SM75 的 k8 兼容分支还以 compute_75 PTX 在本机 JIT 执行通过正确性测试；没有在每一种真实 GPU 上做性能验证。
