# RWKV Lightning Launcher

本地 Go Launcher + Bun / TypeScript / React 静态 WebUI。包含 Chat、Parallel Translate、State Tuning、Runtime、Settings，默认深色 Aero 风格，支持 Light / System。

## 构建与运行

在本目录执行：

```bash
bun install
bun run build
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o rwkv_launcher main.go
```

Windows PowerShell：

```powershell
bun install
bun run build
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o rwkv_launcher.exe main.go
```

将 Launcher 放进原生后端的运行目录，保留原生 bundle 的依赖库：

```text
runtime-directory/
├── rwkv_launcher[.exe]
├── rwkv_lighting_cuda[.exe]
├── rwkv_state_tune[.exe]       # 可选；CUDA 构建的训练程序
├── rwkv_vocab_v20230424.txt
└── lib/                      # 保留现有 bundle 的动态库布局
```

Windows release 中位于可执行文件旁的 DLL 也应保留；旧 Launcher 对 `lib/` 的 PATH 补充逻辑继续生效。Linux 使用原生 bundle 自带的库解析规则。CUDA/HIP 在原生程序编译时选择，前端没有虚构 CPU 或设备切换参数。

运行 `./rwkv_launcher`（Windows 为 `./rwkv_launcher.exe`），访问 **http://127.0.0.1:8088**。程序默认打开系统浏览器；设置 `RWKV_LAUNCHER_NO_BROWSER=1` 可禁用自动打开。

`dist/` 是纯静态输出，使用 `//go:embed dist/*` 编入 Go 二进制。生产环境不需要 Bun 或 Node.js。仓库中保留生成的 `dist/`，因此现有 release 脚本的 `go build ... main.go` 仍可使用；修改前端后必须重新执行 `bun run build`，将源码、锁文件和更新后的 `dist/` 一起提交。

路由使用 `/#/chat`、`/#/translate`、`/#/state-tuning`、`/#/runtime`、`/#/settings`，无需服务端 SPA fallback。不要用 `file://` 打开 `dist/index.html`。

### 开发

先运行已构建的 Go Launcher，再在本目录执行：

```bash
bun run dev
```

Vite 将 `/api`、`/v1` 和 `/logs` 转发到 `127.0.0.1:8088`。生产流量直接走 Go，没有 Node 中间层。`go run main.go` 的临时可执行目录不包含原生二进制，不适合验证本地进程启动。

## 使用

- **Runtime**：输入真实模型和词表路径，可调用宿主机原生文件选择器。最近模型保存于浏览器。Start 使用当前表单；Restart 使用实际运行配置。配置修改在下次 Stop → Start 后生效。
- 动态加载时，模型路径必须为目录。启动服务后，在 Available models 中选择并 Load 模型，再开始 Chat / Translate。Ready 表示 HTTP 服务已就绪，模型是否加载另行判断。
- **Chat**：真实 SSE 增量输出，支持 Markdown、代码高亮、表格、Stop、Regenerate / Retry、历史搜索、重命名和删除。停止或网络中断保留部分输出。
- **Parallel Translate**：按段落/句末标点切片，极长无标点文本回退到空白边界和 Unicode 字符边界；默认目标 800 字符、8 个 worker。块结果始终按 ID 合并，支持停止、恢复 pending、单块重试和失败重试。Inspector 每页 50 块，已完成预览使用独立 memo 组件与 content-visibility；流式 UI 更新合并到最多约 10 次/秒。
- 语言名允许自定义。Auto 是 prompt 中的字面名称，不运行语言检测器。Copy / TXT / Markdown 可导出结果；未完成块导出为 `···` 占位。Check request 查看下一次任务首块的真实 body，Inspector 查看已有任务的原始 prompt。
- **State Tuning**：通过真实 CLI 训练，JSONL 每行必须恰好为一个字符串 `text` 字段。验证在 Go 宿主机执行，拒绝额外/重复字段。显示 stdout / stderr、真实 step / epoch / loss / LR / tokens/s / ETA、loss 曲线和实际保存的 checkpoint 路径。
- **Appearance**：顶部栏可随时切换 Dark / Light / System；Settings 中也提供三态选择。System 会跟随操作系统并在系统外观变化时即时更新，选择会保存在本地，页面加载前即应用以避免主题闪烁。
- **Settings**：API Base URL / Key、默认语言/并发/切片、采样参数、清除本地数据。默认 Base URL 留空，Go 自动使用实际 runtime 端口和密码。自定义地址不带 `/v1`，用于直接连接原生 Chat；平行翻译需要同源 Launcher 适配器。

快捷键：`Ctrl/Cmd+K` 命令面板，`Ctrl/Cmd+N` 新会话，`Ctrl/Cmd+Enter` 开始翻译/提交训练，`Esc` 关闭弹窗。Chat 使用 Enter 发送、Shift+Enter 换行，并兼容输入法组合输入。

会话、当前翻译任务和设置保存在本浏览器 localStorage；翻译中的页面切换不会停止调度，刷新则恢复已保存结果并将中断块标为 pending。API Key 和 runtime password 只保留在内存中，不写入浏览器持久化数据，密码也不会出现在启动日志或 status 中。存储容量不足时显示错误，请导出重要结果。

## 与当前原生代码的兼容说明

事实来源为仓库 `README.md`、`rwkv_lightning_api_doc.md`、`src/server/rwkv_api_service.cpp`、`src/app/rwkv_fast_server.cpp`、`src/state_tuning/rwkv_state_tune_main.cpp` 和 `dataset.cpp`。

### 原始翻译续写

需求中的 `English: Hello\n\nChinese:` 是 **raw continuation**。当前 CUDA `/v1/chat/completions` 会给一般的 `contents` 加上 User / Assistant 模板；原生 Python 后端同名接口的行为不同。因此只给 CUDA Chat 发送 `contents` 不能得到需求所要求的原始续写。

本实现保留 C++ 源码和 CLI，通过 Go 做最小路由适配：

```text
Chat:      browser /v1/chat/completions + messages
        → native  /v1/chat/completions

Translate: browser /v1/chat/completions + contents (每次一个 prompt)
        → native  /v1/batch/completions
```

仅当 body 有 `contents` 且没有 `messages` 时适配，body 不变。多个单 prompt 请求由前端 worker pool 调度，**不调用任何专用 Translation API**。检查请求窗口明确显示这个映射。直接连接未经适配的远程 CUDA Chat 地址时，翻译页面会拒绝发送，避免默默改变 prompt 语义。

翻译 sampler 采用当前兼容翻译实现中的参数：`max_tokens=2048`、`temperature=1`、`top_k=1`、`top_p=0`、presence/frequency penalty=0、`stop_tokens=[0]`；流式 `chunk_size=8`。普通 Chat 默认采样字段来自 CUDA API 文档，停止 token 为整数数组。

当前原生 SSE 的收尾 `finish_reason` 通常统一为 `stop`，不能区分 EOS、长度上限和管理性停止；UI 保存原值，不虚构原因。当前 SSE 不返回标准 usage，因此翻译统计使用真实字符数和耗时，不伪造 token 数。

### Runtime

旧 Go 代码只用进程存在判断 `running`。现在同时探测 `/v1/server/status` 的 HTTP 200 和 `status: running`，可区分 offline / starting / ready / stopping / error。新接口不会把 `cmd.Start()` 成功当成 Ready。

保留原有同目录程序解析、`exec.CommandContext` 参数数组、原生文件选择器及 Windows 环境逻辑；将日志/进程管理共享给训练，补充等待退出、stderr 排空、密码脱敏和退出时清理子进程。启动参数增加真实支持的 `--chunk-size`、`--state-db-path`、`--tune-cache`；固定监听本地 `127.0.0.1`。

### State Tuning

实际 CLI 默认值（不同于 README 中的示例值）：

| 字段 | CLI | 默认值 |
|---|---|---:|
| Context length | `--ctx` | 128 |
| Recompute chunk | `--chunk` | 64 |
| Epochs | `--epochs` | 1 |
| Samples per update | `--batch-size` | 1 |
| Max updates | `--max-steps` | 0，无上限 |
| Learning rate | `--lr` | 1.0 |
| Final learning rate | `--lr-final` | 0.01 |
| Warmup | `--warmup-steps` | 10 |
| Save interval | `--save-every` | 0，不定期保存 |
| Seed | `--seed` | 1234 |

训练只接受 BF16 `.pth` 基础模型，CUDA 专用；不会将 state 文件误称为基础模型。最终是否具有正确 tensor 结构由原生加载器验证。Checkpoint 只有 FP32 state tensors，不含 optimizer；**实际 CLI 没有 resume 参数**，因此界面没有伪造恢复功能。日志中的 epoch 值为 CLI 原值，不伪造小数 epoch。

两个原生程序各自申请 GPU，没有跨进程资源协调；仓库没有规定必须互斥。本 Launcher 为避免默认启动两份模型而选择**串行资源策略**：训练与本 Launcher 管理的推理互斥，训练前可确认 Stop & Start；Go 侧也强制检查。不会停止 Launcher 之外的 GPU 进程。Stop 会终止训练，保留此前实际写出的 checkpoint，不声称已保存尚未落盘的更新。

## Launcher HTTP 接口

新增的控制接口在 `main.go` 实现，不是对原生 API 的假设。所有 POST 都发送 JSON，失败返回实际 `{"error":"..."}` 与 HTTP 错误码。

| Method | Path | 行为 |
|---|---|---|
| GET | `/api/status` | 进程状态、真实 backend status、脱敏配置、最近 2000 行日志 |
| POST | `/api/start` | RuntimeConfig；验证路径/端口并启动 |
| POST | `/api/stop` | 等待运行进程退出 |
| POST | `/api/restart` | 使用上次实际启动配置停止并重启 |
| POST | `/api/pick-file` | 原生宿主机文件选择；无图形环境时明确报错，可手动输入路径 |
| GET | `/logs` | 保留旧 Runtime SSE 日志入口 |
| GET | `/api/tuning/status` | 训练状态、可执行文件是否存在、日志、进度、loss 数据、checkpoint |
| POST | `/api/tuning/validate` | `{"path":"..."}`，返回有效样本数或准确行号错误 |
| POST | `/api/tuning/start` | TuningConfig，启动真实 `rwkv_state_tune` |
| POST | `/api/tuning/stop` | 停止训练进程 |
| POST | `/api/tuning/open-folder` | 打开最近实际保存 checkpoint 所在文件夹 |
| * | `/v1/*` | 转发到本 Launcher 管理的原生 backend，SSE 即时 flush |

完整 TypeScript payload 见 `src/lib/api/launcher.ts`。推理接口沿用项目文档，没有新增原生 CLI flag。静态和控制服务仅绑定 loopback，并验证 Host / Origin；不允许从外站操作本地进程。Markdown 不解析原始 HTML。

## 验证

```bash
bun run test
bun run lint
bun run build
go test -race .
go vet .
```

测试覆盖 SSE 任意拆包、UTF-8、CRLF、畸形事件/错误/断流/Abort；切片内容保持、Unicode；worker 并发硬上限、取消、乱序、失败重试；会话与翻译持久化、敏感字段不落盘；Go 参数验证、真实子进程启停、互斥、训练回车日志、脱敏、Ready 探测、原始续写路由、静态托管和跨源拒绝。

本次环境中通过了 Linux 构建和 Windows amd64 交叉构建，并使用实际 `rwkv_lighting_cuda` 验证动态模式服务的 Start → Ready → Restart → Ready → Stop、模型枚举、错误日志和 JSONL 校验。该服务验证没有加载有效基础模型，不代表真实模型 Chat / 翻译质量或训练数值已验证。

当前环境无可连接的浏览器会话，尚未完成交互式视觉验收。后续验收建议用有效基础模型在 1280 / 1024 像素宽度检查五个页面、流式输出、文件选择、训练及 checkpoint，并验证宿主机对应的 CUDA/HIP 和 Windows DLL 环境。
