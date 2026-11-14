# Kernel 审计历史记录（2026-09-12）

审计起点：`f70b568dbc1f690efc01bda672ea22d257e235f0`。
环境：RTX PRO 6000 Blackwell Workstation Edition，CUDA 13.3.73，driver 610.57.04。

旧对称 W4 优化章节与测量产物已随该格式退役删除；当前量化验证见
[统一报告](../quant/VALIDATION.zh-CN.md)。以下保留与该格式无关的历史审计记录。

## 已修复问题

| 位置 | 问题与修复 | 验证方式 |
|---|---|---|
| `cuda/rwkv7_v3a_ops.cu`、HIP 对应文件 | 连续 LayerNorm 归约在所有线程读完结果前重写 shared scratch；在复用前加 block barrier | 修改前 racecheck 复现跨 warp 竞态和数值错误；修改后 0 hazards |
| `cuda/rwkv7_wkv_fp32_v2.cu`、HIP 对应文件 | mode 3 的 warp 内 shared broadcast 存在复用竞态，不能依赖隐式 warp 同步 | 新增多 B/T/mode 回归；racecheck 从 5 warnings 降为 0 |
| `src/utils/sampling.cu`、`hip/sampling.hip` | 三个采样 kernel 对 `[B,V]` scratch 使用 `[B,T,V]` 偏移，T>1 越界 | 三条路径均测试 B=3、T=5、V=4100，只采样最后一步；memcheck |
| FP16 线性、激活融合、低秩投影 | 奇数 K 使奇数行的 half2 读取未对齐，尽管 kernel 已有尾元素逻辑 | 对齐时保留 half2，否则组合两个标量 half；K=65 对照 BLAS，memcheck |
| 激活与向量加法 | 奇数元素数漏写最后一个值；奇数 C 的广播加法索引错误 | 奇数 extent/C 使用标量 fallback，65 元素数值对照 |
| 稀疏 FFN | 512 tile 路径对 C%512 或 F%512 不满足的输入截断网格/漏输出 | 回退 128×256 路径；测试 C=256/768、F=512/384，以及 1024 tile |
| W8 host 调度 | `static bool shared_limit_set` 存在 host 数据竞态，设备信息和 tuning 原来不区分 GPU | 线程局部、按设备记录属性配置，检查 CUDA API 返回值，tuning key 包含 device；四 host 线程回归 |
| BLAS host 调用 | 全局 handle 的 `SetStream` 可被其他 host 线程覆盖 | 每个 host 线程、每个 device 独立持有 BLAS/Lt handle；多线程不同 stream 数值测试 |
| WKV/采样/INT8 packing 地址 | 若干乘法先以 int 计算，长序列或大权重时可能溢出 | 在乘法前提升到 64 位；常规尺寸回归。未申请超大张量验证全部极限 |
| 部分 launch 参数校验 | Release 下 assert 消失，错误 tile/维度可能静默不输出或越界 | 为 W8、WKV、稀疏 FFN、exact linear 和要求偶数 C 的融合路径增加运行时校验 |
| state-tuning model forward | 校验 head 数之前计算 C/H，H=0 可导致 host 除零 | 先检查 H/C，再计算 head size |

输入、输出、scale、workspace 的容量和非重叠关系仍由调用者保证；并发请求不得共享
可写 state、输出或 scratch。上述校验不意味着 API 能验证任意裸指针的实际分配容量。

## 审计与执行覆盖

静态检查覆盖仓库的 CUDA kernel 源文件和对应 HIP 文件，重点检查 shared-memory
生产/消费、异步 copy 等待、warp mask、全局写入归属、向量对齐、尾维度和 launch
分派。`src/backend/rwkv7_fast_v4.cu` 是模型和调度集成，未定义额外 `__global__` kernel；
`tools/calibration/bandwidth.cu` 使用 CUDA memcpy，没有自定义计算 kernel。

| 源码类别 | 执行覆盖 |
|---|---|
| W8 GEMV/WMMA/MMA、packing、稀疏 INT8 | NK/KN/PackedNK、M=1..1024、128/192 输出列、320 K、split 1/3/5/auto；已有随机权重完整测试与缩小形状的 sanitizer 模式 |
| V3a norm/linear/转换 | 非 4096 C 参考值；4096 C 的 1/512/1024 行分派、in-place residual 输出；奇数 K、tile 尾部、BLAS 多线程 |
| fast ops、稀疏 FFN | 奇数激活、稀疏 fallback、已有稀疏数值测试，以及模型 smoke 与 state-tuning 测试触发的门控/混合路径 |
| FP16/FP32 WKV | B=1/3/4/65/129、T=1/3/4/8；FP16 sequence 对逐步 decode，FP32 mode 0/1/2/3 交叉对照 |
| statepassing、backward/training/block/model kernels | 已有 WKV 有限差分、chunk adjoint、LN/CE、state-tuning 测试；模型 forward 的 head 校验修复 |
| HIP | 对应通用修复已同步；没有 ROCm 编译器或 AMD GPU，未做 HIP 编译与运行验证 |

这不是“所有输入与硬件上不存在任何错误”的形式化证明。未执行每一种模板实例、
所有 GPU 架构、所有调度组合或任意错误别名输入。真实多 GPU 切换和正式大模型
端到端吞吐/精度尚未实测；旧单层量化性能数据已移除。

## 验证结果与复现

- CUDA SM75/80/86/87/89/90/100/120 Release 编译通过。PTXAS 对已有稀疏 t1024
  kernel 的过高 `minnctapersm` 给出警告并忽略该 hint，没有编译失败。
- `rwkv_kernel_safety_test`、`rwkv_cuda_non4096_kernels_test`、
  `rwkv_state_tuning_cuda_test`、
  `rwkv_w8a16_kernels_test --quick` 的 memcheck/racecheck/synccheck/initcheck
  通过：0 errors，racecheck 0 warnings。
- 最终重新运行六个 kernel/模型相关 CTest，6/6 通过。
- 全量 CTest 为 13/14 通过。唯一失败 `rwkv_inference_engine_test:67` 在修改前
  已复现：期望 `<think>\n</think`，实际 `<think></think`，不属于 kernel 变更。

```bash
cmake -S . -B build-audit -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build-audit -j 8
ctest --test-dir build-audit --output-on-failure
bash tools/check_kernel_safety.sh build-audit
```

脚本显式使用 `--num-cuda-barriers 256`，因为本机 CUDA 13.3/cuBLAS 的 synccheck
自动 barrier 容量检测溢出；增加工具容量后检查通过。W8 的 `--quick` 缩小 CPU
参考矩阵与 sanitizer 工作量，完整 CTest 仍使用原来的大矩阵。
