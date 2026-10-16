# 连续批处理

服务默认启用连续批处理。每个服务实例在当前 CUDA/HIP 设备上，为已加载模型启动一个常驻 GPU 调度线程。动态切换模型时，现有请求的 model lease 先结束，再释放旧调度器和模型。多卡部署沿用每卡独立后端进程的方式。

```sh
./build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path assets/rwkv_vocab_v20230424.txt \
  --chunk-size 128 \
  --decode-batch-size 64
```

`--decode-batch-size` 限制所有 adapter 合计的活动解码行数，默认 64，必须大于零。超过上限的行排队等待槽位；一个大请求可以分多批进入解码。原有预填充显存准入仍然生效。该选项同时适用于动态加载模式。

调度行为：

- 每轮接纳就绪的新行，最多执行一个预填充块，然后推进活动行的解码。长提示词的多个预填充块不会独占整个解码周期。
- 同一 adapter 身份（包含版本和 scale）的行合并前向；不同 adapter 分组执行。不同初始 state 可以进入同一个 adapter 分组，各自的 state 身份在导出时恢复。
- 每行保留自己的温度、top-k、top-p、presence/frequency/decay、随机数状态、强制思考步数、停止条件和输出缓冲。HTTP 参数格式不变，同一 HTTP 请求中的行继承该请求的参数；不同请求的参数在合并 batch 后仍相互独立。
- state 使用紧凑行槽位和双缓冲。只有成员变化时搬移 state、logits、惩罚和 RNG；稳定 batch 的每步不复制完整 state。已结束行立即退出，空槽接纳等待行。
- EOS 不写回 state；已输出的末尾 token（包括文本停止串和达到长度上限的 token）推进 state 后再导出，供会话续写或暂停恢复使用。
- 网络回调在请求工作线程执行。非流式请求也通过独立工作池处理，避免占住 Drogon 的 4 个 I/O 线程。输出队列达到 128 个事件时，该请求暂时交还槽位，保留 state 和采样历史；队列降到阈值以下后继续。

解码槽位上限不是整个服务的显存硬上限。等待请求的初始/预填充 state、提示词缓存、模型权重、adapter 和预填充 workspace 另占内存。前向 scratch 在调度线程内复用，避免每个 HTTP 请求常驻一套大 workspace。

控制回调在调度线程的步边界执行，应保持快速且不阻塞。流式回调返回 false 或抛异常后，调度器会先移除请求的所有行，再让调用线程返回或抛出异常。正常文本停止和原有 HTTP `finish_reason`、`matched_stop` 语义保持兼容。

## 验证与性能比较

```sh
ctest --test-dir build \
  -R 'rwkv_(sampler|inference_engine|prompt_cache|continuous_batching)_test' \
  --output-on-failure

python test/string_stop_http_test.py --url http://127.0.0.1:8000

python tools/bench_continuous_http.py \
  --urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
  --tokens 256 --repeats 3 --output build/continuous-benchmark.json
```

基准脚本对比 1/8/32 个独立单行请求与单个 64 行请求；关闭提前停止，每行生成固定数量的 token。预热后报告三次端到端吞吐的中位数，包含排队、预填充和 HTTP 耗时。比较时两个服务应使用同一模型及相同参数，测量期间不要运行其他 GPU 工作。

逐行贪心采样在最大 logits 并列时固定选择较小的 token ID。默认 `no-fc` 稀疏 Cmix 路径的浮点归约可能有数值波动；要求逐字比较的真实模型回归可使用 `--cmix-sparse off`。不同 batch 大小也可能产生浮点舍入差异，不能把所有文本差异直接视为 state 串扰。

## 本机验证记录（2026-10-16）

RTX PRO 6000 Blackwell，`rwkv7-g1k-7.2b-20260930-ctx25600.pth`，FP16 state，`--cmix-sparse off --chunk-size 128`，新版本使用默认 64 槽位。基线为提交 `d09ea70`；每行 256 token，三轮端到端吞吐中位数：

| 请求方式 | 改动前 tok/s | 连续批处理 tok/s | 倍率 |
| --- | ---: | ---: | ---: |
| 1 个单行请求 | 106.3 | 106.2 | 1.00× |
| 8 个单行请求并发 | 120.2 | 783.5 | 6.52× |
| 32 个单行请求并发 | 122.5 | 2883.4 | 23.53× |
| 1 个 64 行请求 | 5435.3 | 5370.6 | 0.99× |

验证结果：

- 8 个相关 CTest 目标通过，涵盖逐行采样、RNG/惩罚搬移、持续接纳、预填充交错、adapter/初始 state 隔离、取消、慢消费者恢复，以及分配/复制/前向异常后的恢复。
- 0.1B 模型 dense Cmix 的 HTTP 文本停止回归：187 项通过、0 项失败。
- 7.2B 模型配合仓库的 `latest-state.pth` 和 `adapter-final.pth`：完整 HTTP 回归 147 项通过、4 项失败；新增混合 adapter/scale/state/采样参数并发用例通过。4 项失败均为 `SSE Connection: close` 的三个端点及其遗留 active-request 检查，在未修改的基线上也复现（基线 146 项通过、同样 4 项失败），本次未修改该协议问题。
- CUDA 已编译并在真实 GPU 上验证；HIP 采样实现已同步修改，但本机没有进行 HIP 编译或运行验证。

以上数字只代表此模型和本机配置；不同量化方式、adapter 分组数量、prompt 长度和显存条件需要重新测量。
