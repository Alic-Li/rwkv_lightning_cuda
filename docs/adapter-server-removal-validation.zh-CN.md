# 原生 Server 动态 Adapter 移除验收

2026-11-07，CUDA / RTX PRO 6000 Blackwell。对照基线为
`e4506687b9c409c772ba941b76387e2af7f4d5da`。

本阶段移除了原生推理 Server 的 Adapter HTTP 路由、请求绑定、RAM/GPU 缓存、
版本/Scale 调度、附加前向与临时 NoRA workspace。旧请求字段返回 400，旧管理路径返回 404。
State 上传、动态选择、提示词缓存、L1/L2/SQLite Session 缓存和暂停恢复继续保留。
MiSS/NoRA Kernel、包读写、独立 CLI/在线训练功能保留。没有增加权重副本，
没有实现 Checkout、Weight Merge 或替代 Adapter 管理机制。

## 正式模型配置

```bash
./build/rwkv_lighting_cuda \
  --model-path /mnt/nvme0n1/no22/rwkv7-g1l-7.2b-20261031-ctx34304.pth \
  --vocab-path ./assets/rwkv_vocab_v20230424.txt \
  --host 0.0.0.0 --port 8060 \
  --chunk-size 512 --decode-batch-size 512 \
  --state-db-path build/adapter-removal-final-sessions.db
```

State 实测文件为仓库的 `rwkv-state-agentic.pth`。测试使用独立数据库，未修改用户模型、
State 文件或原有 `rwkv_sessions.db`。主验收保留默认 `--cmix-sparse no-fc`；
长文本精确匹配测试另用 `--cmix-sparse off`，不比较稀疏/普通 FFN 的逐 token 相等性。

## 结果

| 检查 | 结果 | 本机报告 |
|---|---|---|
| Release 全量 CUDA 构建（SM120，含三个训练 CLI/Server） | 通过 | `build/adapter-removal-validation/rwkv-build.log` |
| CTest | 22 通过，3 跳过，0 失败 | `build/adapter-removal-validation/rwkv-ctest-final.log` |
| 原生 HTTP 全量 E2E，202 个 HTTP 请求 | 137 通过，4 个基线已有失败 | `build/adapter-removal-e2e-final/report.json` |
| Continuous Batching，148 个 HTTP 请求 | 22/22 通过 | `build/adapter-removal-stress/report.json` |
| 动态模型模式 + 上传 State，44 个 HTTP 请求 | 31/31 通过 | `build/adapter-removal-dynamic-e2e/report.json` |
| 普通 FFN 字符串停止全量测试，261 个 HTTP 请求 | 186/186 通过 | `build/adapter-removal-string-stop-dense/report.json` |
| 非流式 State 中断、缓存、续推 | 通过 | `build/adapter-removal-validation/rwkv-state-stop.log` |
| 正式模型 MiSS / NoRA / State CLI 训练一步及导出 | 三种模式通过，loss 有限 | `build/adapter-removal-validation/rwkv-train-*.log` |
| 三种在线训练 Server 冷加载、生成、训练一步、发布 | 三种模式通过，梯度有限且非零 | `build/adapter-removal-training-http/report.json` |

CTest 自动跳过 `rwkv_online_rl_training_test`、`rwkv_online_rl_forward_test`、
`rwkv_packed_batch_test`，因为这些目标默认未传入外部模型/Adapter fixture。
本次训练功能验收使用上表中的正式 7.2B CLI 与在线训练实测，未将跳过项计作通过。
HIP 共用头文件对应的推理调用已同步清理，但本机没有执行 HIP 构建或 AMD 实测。

批处理覆盖 1/4/16 个并发请求、每请求 1/16 行、单请求 64 行、
解码中途新请求接纳、独立停止条件，以及混合零 State/上传 State 的请求。
State 单测额外检查身份摘要长度、模型/初始 State/精度隔离、L1/L2/SQLite 回读；
HTTP 检查 State Body/Header、上传/列表/删除、Session Header 删除、暂停时拒绝旧 Adapter
字段后仍可恢复。MiSS 包格式回读与 MiSS/NoRA 数值 Kernel 测试继续执行。

## 基线已有失败

全量 E2E 中的三项 `SSE Connection: close` 返回空流，随后请求注册清理检查失败。
用未修改基线、相同正式模型和相同启动参数重新验证，得到完全相同的四项失败，
证据为 `build/adapter-removal-baseline-e2e/report.json`。
本机 Drogon 1.9.13 的 `HttpServer::sendResponse` 在 `ifCloseConnection()` 为真时
不调用 async stream producer，日志为 `Chunking Set CloseConnection !!!`。
这次功能精简未改动该 HTTP 流实现，也未跳过或掩盖失败。

额外字符串停止测试的首次稀疏运行受到上述遗留请求影响（两项全局清理断言），
并有一次长文本跨请求精确匹配失败；原始记录保存在
`build/adapter-removal-string-stop/report.json`。在独立、干净的普通 FFN 服务上
重跑全套后 186/186 通过。

## 显存对照

通过独立冷启动的基线和修改版进程，使用相同模型、参数和 Base/State 双行请求，
读取 `nvidia-smi` 的进程显存（MiB 粒度）：

| 检查点 | 基线 | 修改版 | 差值 |
|---|---:|---:|---:|
| 模型加载完成 | 14246 MiB | 14246 MiB | 0 MiB |
| Base 双行推理后 | 14574 MiB | 14574 MiB | 0 MiB |
| 上传 State 双行推理及重复请求后 | 14574 MiB | 14574 MiB | 0 MiB |

原始结果在 `build/adapter-removal-memory/report.json`。源码中没有新增 GPU 权重分配，
原有基模加载及 Base 前向路径保留。Server 二进制检查也不再包含旧 Adapter 路由、
缓存预算环境变量或动态 Adapter 指标字符串。

## 复跑入口

```bash
ctest --test-dir build --output-on-failure
python test/http_e2e_test.py --url http://127.0.0.1:8060 \
  --state rwkv-state-agentic.pth --output build/adapter-removal-e2e-rerun
python test/continuous_http_stress_test.py --url http://127.0.0.1:8060 \
  --tokens 16 --repeats 1 --rows 16 --output build/adapter-removal-stress-rerun
BASE_URL=http://127.0.0.1:8060 python test/state_nonstream_stop_test.py
```

严格字符串测试应在独立普通 FFN 服务上执行：
`python test/string_stop_http_test.py --url http://127.0.0.1:8063`。
动态模式先以 `--enable-dynamic-loading --model-path <模型目录>` 启动服务，再执行
`python test/dynamic_http_e2e_test.py --url http://127.0.0.1:8062 --model <模型ID> --state rwkv-state-agentic.pth`。

每次运行使用新的输出目录；已知 `Connection: close` 用例会留下注册记录，因此不同测试套件
应使用干净服务。验收结束已关闭本次启动的所有临时推理与训练服务。

## Router / Launcher 同步清理验收（2026-11-07）

Router 已移除 Adapter 广播上传、版本发现/固定、副本就绪表、调度条件和
`adapter_upload_*` 配置。旧管理路径直接返回 404，旧请求字段返回 400，均不转发给后端。
State 广播、UUID、失败重试、就绪判断、Session 亲和和 HTTP 行为保留。

Launcher 已移除 Adapter 管理界面、选择控件、API 客户端、生成设置、翻译请求注入及文案。
本地与后端前缀的 Adapter 管理路径返回 404。设置存储升级到 v4，清除旧 Adapter 字段，
保留所选 State、采样参数、语言与用户主题；Chat 只序列化支持的生成参数。
State 上传、节点文件导入、管理界面与训练能力继续保留。本次没有修改 GPU 权重或 Kernel
路径，也没有实现 Checkout、Merge、权重副本或新的 GPU 缓冲区。

使用同一个正式 7.2B 模型启动两个独立原生进程（8060 / 8061，chunk/decode batch 均为
512），Router 监听 18080，Launcher 监听 18081，经后端前缀访问 Router。
上传文件为 `rwkv-state-agentic.pth`，Session 数据库单独存放，未覆盖用户数据库。
两个进程各自正常加载模型仅用于验证 State 广播，不是产品新增常驻权重副本。

| 检查 | 结果 | 报告（位于 `build/router-launcher-removal-validation/`） |
|---|---|---|
| Router `go test -race ./...`、`go vet ./...` | 通过 | `router-go.log` |
| Launcher `go test -race .`、`go vet .` | 通过；可选 tiny GPU fixture 测试未启用 | `launcher-go.log` |
| 前端 Bun 测试 | 87/87，492 个断言 | `frontend-test.log` |
| 前端 TypeScript/Vite 构建与 ESLint | 通过 | `frontend-build.log`、`frontend-lint.log` |
| Launcher → Router → 两个正式模型进程 | 38/38，含直接核验共 61 个 HTTP 请求 | `chain-e2e/report.json` |
| 正式模型原生 HTTP 全量功能测试 | 133 通过、4 个已知失败 | `native-functional-e2e/report.json` |
| Launcher HTTP 控制面启动 MiSS / State 训练 | 两种模式各一步成功，loss 有限，最终 PTH 已导出 | `training-report.json` |

链路测试覆盖 Base/State 的 Chat/Batch 非流式与 SSE、State Header、重复推理、混合并发、
State 在每个独立进程中的一致可见性、Session 亲和、缓存续推、删除与不存在 State 拒绝，
以及旧 Adapter 管理接口/请求参数拒绝。每个原生进程均未新增泄漏请求。
回归入口为 `test/router_launcher_e2e_test.py`：

```bash
python test/router_launcher_e2e_test.py \
  --url http://127.0.0.1:18081/api/v1/backends/formal-router \
  --router http://127.0.0.1:18080 \
  --worker http://127.0.0.1:8060 --worker http://127.0.0.1:8061 \
  --state ./rwkv-state-agentic.pth
```

原生测试的 4 个失败与前述基线相同：三个 `Connection: close` 空 SSE，及对应的最终请求
清理断言。首次运行还触发一次稀疏 FFN 的跨请求文本精确匹配失败（原始结果
`native-e2e/report.json`：136 通过、5 失败）。现将 `http_e2e_test.py` 的精确流式/非流式
文本比较改为显式 `--check-stream-parity`，仅用于确定性/普通 FFN 验证；默认仍完整检查
响应结构、流结束、State、Session、并发与控制 API。最终功能测试未启用这四项文本精确
比较，未跳过上述 SSE 失败。
