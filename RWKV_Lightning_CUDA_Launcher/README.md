# RWKV Lightning Launcher

Go Launcher + React / TypeScript 静态 WebUI，使用 Vite 构建、Tailwind CSS 4 设计令牌、Bun 管理依赖与测试。多后端控制台包含节点总览、Runtime 运维、Chat、并行翻译、State / MiSS 训练、量化与设置，默认跟随系统外观，可显式选择深色 / 浅色，并支持中英界面切换。

## 文档入口

- [前端与第三方联调指南](docs/integration-guide.md)：接入方式、鉴权、curl / SDK 示例、错误与 SSE。
- [控制面 API 参考](docs/control-plane-api.md)：完整路由、请求体、返回值与兼容约定。
- [前端开发指南](docs/frontend-development.md)：工程结构、Tailwind 接入、多后端架构、Bun 与测试。
- [未决事项](docs/open-items.md)：已知但未修的问题、需要先做决定的兼容性取舍、尚未实机验收的范围。
- [原生推理 API](../rwkv_lightning_api_doc.md)：C++ server 的生成、模型、state、adapter 接口。

前端框架是 React，样式为 Tailwind CSS 4（CSS-first `@theme` 令牌 + shadcn 风格自建组件 + Radix primitives），不使用 HeroUI；Bun 用于依赖管理、脚本执行和测试。

## 构建与运行

安装 Bun（CI 使用 1.3.9），然后在本目录执行：

```bash
bun install --frozen-lockfile
bun run build
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o rwkv_launcher .
```

Windows PowerShell：

```powershell
bun install --frozen-lockfile
bun run build
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o rwkv_launcher.exe .
```

Linux 上要启用 GPU 指标（NVML 绑定）需以 `CGO_ENABLED=1 go build .` 构建（只需 gcc 与 libdl，不需要 CUDA toolkit）；`CGO_ENABLED=0` 构建仍然可用，metrics 端点会返回 `available:false` 加原因。macOS/Windows 构建不受影响。

将 Launcher 放进原生后端的运行目录，保留原生 bundle 的依赖库：

```text
runtime-directory/
├── rwkv_launcher[.exe]
├── rwkv_lighting_cuda[.exe]
├── rwkv_state_tune[.exe]       # 可选；CUDA 构建的训练程序
├── rwkv_miss_tune[.exe]        # 可选；MiSS adapter 训练程序
├── rwkv_vocab_v20230424.txt
└── lib/                      # 保留现有 bundle 的动态库布局
```

Windows release 中位于可执行文件旁的 DLL 也应保留；旧 Launcher 对 `lib/` 的 PATH 补充逻辑继续生效。Linux 使用原生 bundle 自带的库解析规则。CUDA/HIP 在原生程序编译时选择，前端没有虚构 CPU 或设备切换参数。

运行 `./rwkv_launcher`（Windows 为 `./rwkv_launcher.exe`），访问 **http://127.0.0.1:10721**。程序默认打开系统浏览器；设置 `RWKV_LAUNCHER_NO_BROWSER=1` 可禁用自动打开。

### 形态与启动参数（多后端）

同一个二进制靠 flag 决定跑成什么角色，详见 [docs/control-plane-api.md](docs/control-plane-api.md)：

```bash
rwkv_launcher                                  # 本机全套：Client + Agent 同机（现有行为）
rwkv_launcher --listen 0.0.0.0:18766 --token t # 服务器节点：仅 Agent；非 loopback 必须带 token，否则拒绝启动
rwkv_launcher --client                         # 控制台：仅 Client（本机无 runtime 二进制时自动进入，合法状态）
```

| Flag | 默认 | 说明 |
| --- | --- | --- |
| `--listen` | `127.0.0.1:10721` | 监听地址；非 loopback 必须配 `--token` |
| `--token` | 空 | 控制/推理全部路径的 Bearer token（含 `/v1` 反代） |
| `--client` | false | 纯 Client：WebUI + 后端注册表，不管理本地进程 |
| `--config` | `~/.rwkv_launcher/launcher.json` | 后端注册表落盘位置（JSON，0600） |
| `--card` | 空 | 本 Agent 默认选卡，注入 `CUDA_VISIBLE_DEVICES`（AMD 为 `HIP/ROCR_VISIBLE_DEVICES`） |

`visible_devices`（字符串，如 `0`、`0,1`、空串=显式不注入）可出现在 Runtime / Tuning / Quantization 三个请求体里，优先级：请求体 > `--card` > 继承的 `CUDA_VISIBLE_DEVICES` > 不注入。训练与推理的互斥相应从全局改为按卡：显式选卡且设备集无交集时可并行，不可判定时保守拦截。

`dist/` 是纯静态输出，使用 `//go:embed dist/*` 编入 Go 二进制。生产环境不需要 Bun 或 Node.js。

`dist/` **不入库**，它是构建产物：CI 会使用锁文件重新安装依赖、跑 lint/test、执行 `bun run build`，然后才编译 Go，所以仓库里放一份只会在每次前端改动时产生无意义的二进制冲突。代价是 `go build` 之前必须先 `bun run build`——忘了的话 `//go:embed dist/*` 会直接编译失败（`pattern dist/*: no matching files found`），不会静默产出一个空壳二进制。

字体同理：Geist 与 JetBrains Mono 以 `@fontsource-variable/*` 依赖形式安装，构建时产出 latin subset 的 variable woff2 到 `dist/assets/`，源码树与仓库里都不存放字体文件。二者自托管而非引 CDN，内网与离线机器上的观感才和设计稿一致。

路由使用 `/#/chat`、`/#/translate`、`/#/state-tuning`、`/#/runtime`、`/#/settings`，无需服务端 SPA fallback。不要用 `file://` 打开 `dist/index.html`。

### 开发

先运行已构建的 Go Launcher，再在本目录执行：

```bash
bun run dev
```

Vite 将 `/api` 和 `/v1` 转发到 `127.0.0.1:10721`。生产流量直接走 Go，没有 Node 中间层。`go run .` 的临时可执行目录不包含原生二进制，不适合验证本地进程启动。

## 使用

- **节点总览**：注册表里的每个后端的可达性、runtime 状态、已加载模型与运行中任务；可逐个或全部重新探测、切换当前节点、移除节点（`local` 保留）。切换节点后所有页面跟随该节点。
- **Runtime**：输入真实模型和词表路径，可浏览远端白名单目录，宿主机对话框仅在节点宣告 `host_dialog` 时出现。Start 使用当前表单；Restart 使用实际运行配置。配置修改在下次 Stop → Start 后生效。
- 选卡是三态字符串：缺省沿用 Agent `--card` / 继承环境、空串显式不注入、显式值如 `0,1`；Runtime / 训练 / 量化共用同一控件。
- 动态加载时，模型路径必须为目录。启动服务后，在模型列表中选择并 Load，再开始 Chat / Translate。Ready 表示 HTTP 服务已就绪，模型是否加载另行判断。
- **Chat**：真实 SSE 增量输出，支持 Markdown、代码高亮、表格、Stop、重新生成、HTML 预览。会话按 backend ID 隔离；停止或网络中断保留部分输出。
- **Parallel Translate**：按非空行生成原始续写任务，直接调用 `/v1/batch/completions`，每批 1–128 行非流式执行并按 `choices[].index` 合并。支持停止、重试失败行与待执行行。结果记录所属节点，切换节点不会串结果。
- 语言名允许自定义，未知值回退默认。Copy / TXT / Markdown 可导出结果；未完成块以 `···` 占位。
- **State Tuning**：通过真实 CLI 训练，JSONL 每行必须恰好为一个字符串 `text` 字段。验证在 Go 宿主机执行，拒绝额外/重复字段。显示 stdout / stderr、真实 step / epoch / loss / LR / tokens/s / ETA、loss 曲线和实际保存的 checkpoint 路径。
- **Appearance**：默认跟随系统外观（System），Settings 中可显式选择 Dark / Light / System，以及中文 / English 切换。System 会跟随操作系统并在系统外观变化时即时更新，选择会保存在本地，页面加载前即应用以避免主题闪烁。
- **Settings**：外观与语言、当前节点与 Agent token、翻译默认语言与 batch size，以及清除本机数据。远端 token 由本机 Client 注入，浏览器不持久化。

快捷键：`Ctrl/Cmd+Enter` 开始翻译，`Esc` 关闭弹窗。Chat 使用 Enter 发送、Shift+Enter 换行，并兼容输入法组合输入。

会话、当前翻译任务、各表单与设置保存在本浏览器 localStorage（`rwkv-*` 键，含版本与 migrate）；翻译中的页面切换不会停止调度，刷新则恢复已保存结果并将中断块标为 pending。Agent token 与 runtime password 只保留在内存中，不写入浏览器持久化数据，密码也不会出现在启动日志或 status 中。存储容量不足时显示错误，请导出重要结果。

## 与当前原生代码的兼容说明

事实来源为仓库 `README.md`、`rwkv_lightning_api_doc.md`、`src/server/rwkv_api_service.cpp`、`src/app/rwkv_fast_server.cpp`、`src/state_tuning/rwkv_state_tune_main.cpp` 和 `dataset.cpp`。

### Chat 初始化 state 与思考

Chat 输入框的 State 按钮可以上传 `.pth`、刷新和查看服务端 state 列表（ID、大小、tensor 数和上传时间）、选择初始 state，以及确认删除。上传成功后自动选中，删除当前选中的 state 后恢复默认初始化。请求携带所选 `state_id`，服务端从该 state 初始化，再处理完整会话历史。state 属于当前服务进程，重启后需要重新上传。

Chat 仍调用 `/v1/chat/completions`，服务端内部执行 batch 生成。默认 `think_type=fast`，输入框 Thinking 开关开启后使用 `free`，实际单轮格式如下（末尾故意不补 `>`）：

```text
快思考：User: {用户输入}\n\nAssistant: <think></think
启用思考：User: {用户输入}\n\nAssistant: <think
```

### 原始翻译续写

需求中的 `English: Hello\n\nChinese:` 是 **raw continuation**。当前 CUDA `/v1/chat/completions` 会给一般的 `contents` 加上 User / Assistant 模板；原生 Python 后端同名接口的行为不同。因此只给 CUDA Chat 发送 `contents` 不能得到需求所要求的原始续写。

本实现保留 C++ 源码和 CLI，通过 Go 做最小路由适配：

```text
Chat:      browser /v1/chat/completions + messages
        → native  /v1/chat/completions

Translate: browser /v1/batch/completions + contents (每批最多 128 个 prompt)
        → native  /v1/batch/completions
```

前端直接调用 `/v1/batch/completions`。Go 侧曾有一段 `contents → batch` 适配（POST `/v1/chat/completions` 且 body 有 `contents`、没有 `messages` 时改写路径），那是给旧 WebUI 的；旧 WebUI 已随本次重做移除，适配也一并删除，`/v1` 现在是完全透明的反向代理。翻译以每个非空输入行为一个 chunk，前端按 1–128 的 batch size 分组，每组作为一个原生 batch 请求执行，**不调用任何专用 Translation API**。

翻译 sampler 采用当前兼容翻译实现中的参数：`max_tokens=2048`、`temperature=1`、`top_k=1`、`top_p=0`、presence/frequency penalty=0、`stop_tokens=[0]`。翻译请求使用 `stream=false`，每行完成后一次显示完整结果；普通 Chat 仍使用流式输出。

当前原生补全响应的 `finish_reason` 通常统一为 `stop`，不能区分 EOS、长度上限和管理性停止；UI 保存原值，不虚构原因。响应不返回标准 usage，因此翻译统计使用真实字符数和耗时，不伪造 token 数。

### Runtime

旧 Go 代码只用进程存在判断 `running`。现在同时探测 `/v1/server/status` 的 HTTP 200 和 `status: running`，可区分 offline / starting / ready / stopping / error。新接口不会把 `cmd.Start()` 成功当成 Ready。

保留原有同目录程序解析、`exec.CommandContext` 参数数组、原生文件选择器及 Windows 环境逻辑；将日志/进程管理共享给训练，补充等待退出、stderr 排空、密码脱敏和退出时清理子进程。启动参数增加真实支持的 `--chunk-size`、`--state-db-path`、`--tune-cache`；固定监听本地 `127.0.0.1`。

### State Tuning

Launcher WebUI 默认训练参数（启动时会完整传给 CLI）：

| 字段                | CLI                      |    默认值 |
| ------------------- | ------------------------ | --------: |
| Context length      | `--ctx`                  |       512 |
| Recompute chunk     | `--chunk`                |       128 |
| Epochs              | `--epochs`               |         1 |
| Samples per update  | `--batch-size`           |        16 |
| Max updates         | `--max-steps`            | 0，无上限 |
| Learning rate       | `--lr`                   |    0.0005 |
| Final learning rate | `--lr-final`             |    0.0001 |
| Warmup              | `--warmup-steps`         |        10 |
| Save interval       | `--save-every`           |       100 |
| Seed                | `--seed`                 |      1234 |
| Optimizer           | `--optimizer adam\|muon` |      adam |
| Shared WKV tape     | `--wkv_tape`             |      关闭 |

State tuning 使用 `.pth` 基础模型，最终张量结构由原生加载器验证。State checkpoint 只有 FP32 state tensors，不含 optimizer，State 模式不提供续训恢复。日志中的 epoch 值为 CLI 原值。

### MiSS 训练与推理

在 **State / MiSS Training** 页面将 Method 切换为 **MiSS adapter**，Launcher 会启动同目录的 `rwkv_miss_tune`。
提供 rank、alpha、六类 Linear targets、初始 state、checkpoint 续训目录，以及原有 ctx/chunk、梯度累积 batch、学习率和 WKV tape 设置。
默认 rank=16、alpha=16、targets=all、lr=0.0001、lr-final=0.00001。基模冻结，只训练 D；`batch_size` 沿用 microbatch 梯度累积。
恢复时选择 `checkpoint-N` 目录，保持原模型、数据与训练配置，并选择新的输出目录。
周期保存仍为 `checkpoint-N/{checkpoint.json,training.pth}`；结束生成单个 `adapter-final.pth`。

在 Chat 的 **MiSS** 按钮、Parallel Translate 的 MiSS 面板或 Settings 的 **MiSS adapters** 面板中：

- 选择最终 PTH 或新 checkpoint 的 `training.pth` 后立即上传并用于新请求；adapter ID 可选，留空时使用去掉 `.pth` 的文件名。旧 checkpoint 可先附带 `checkpoint.json`。
- 也可填写服务器已有文件的绝对路径注册。
- 注册只缓存 D 到服务器 RAM；首次调用才装入 GPU。
- 选择具体内容版本，并按需覆盖 scale；留空使用默认值，0 保留为显式零。
- Chat 和 Parallel Translate 的新请求均携带所选 `adapter_id/adapter_version/adapter_scale`；运行中的请求不受切换影响。
- 可刷新列表、删除指定版本和查看 RAM/GPU 驻留量及 H2D 次数。服务器重启后需重新注册。

管理本地 runtime 时，Go 自动透传其鉴权信息；连接远程服务器时使用 Settings 中的 API key。

两个原生程序各自申请 GPU，没有跨进程资源协调；仓库没有规定必须互斥。本 Launcher 按显卡判断训练与推理互斥：设备集有交集或不可判定时拒绝同时启动，显式选择无交集设备时允许并行。WebUI 已在 Runtime / 训练 / 量化三处接入 `visible_devices`；Go 侧仍执行最终检查。不会停止 Launcher 之外的 GPU 进程。Stop 会终止训练，保留此前实际写出的 checkpoint，不声称已保存尚未落盘的更新。

## WebUI 使用的接口

WebUI 使用 [新控制面 API](docs/control-plane-api.md)：浏览器只与本机 Client 同源通信，节点级请求都带 `/api/v1/backends/{id}` 前缀；`src/lib/api/http.ts` 是唯一出口。接入步骤见 [联调指南](docs/integration-guide.md)。

控制面接口的完整路径表、schema 与安全模型见 [docs/control-plane-api.md](docs/control-plane-api.md)。所有 POST 都发送 JSON，失败返回实际 `{"error":"..."}` 与 HTTP 错误码。

新控制面（`/api/v1`）：`/api/v1/node`、`/api/v1/node/metrics`、`/api/v1/node/fs`、`/api/v1/node/dialog/{file,directory,reveal}`、`/api/v1/runtime`（+ `start/stop/restart/load/state/import/logs`）、`/api/v1/jobs`（+ `/{id}`、`/tuning`、`/tuning/validate`、`/quantization`、`/{id}/stop`、`/{id}/logs`）、`/api/v1/backends`（Client：增删查 + `/{id}/probe` + `/{id}/api/v1/*`、`/{id}/v1/*` 转发）。

**没有旧路径 alias。** `/api/v1` 是唯一的控制面。pre-v1 的 `/api/status`、
`/api/start`、`/api/stop`、`/api/restart`、`/api/tuning/*`、
`/api/quantization/*`、`/api/pick-file`、`/api/pick-directory`、
`/api/tuning/open-folder` 和 `/logs` 全部移除，未知 `/api` 路径一律返回
JSON 404。连带移除的还有"连接旧版 Agent"的转发适配（原 `legacy.go`）：
探测阶梯现在只有 `/api/v1/node` → `/v1/server/status` 两级，只会说 pre-v1
协议的主机不再被识别成 agent。

这意味着 **Client 与 Agent 必须同版本部署**。这是有意的取舍——本次重做从干净
版本起步，不承担混版本兼容的负担。跨版本策略见
[未决事项](docs/open-items.md)。

请求体 TypeScript 类型与默认值见 `src/lib/api/types.ts` 与 `src/lib/api/launcher.ts`；完整字段以 [API 参考](docs/control-plane-api.md) 为准。推理接口沿用项目文档，没有新增原生 CLI flag（选卡走环境变量注入）。loopback 形态验证 Host / Origin；非 loopback 的 Agent 以 `--token` 鉴权并保留 same-origin 浏览器检查。Markdown 不直接解析原始 HTML；助手输出完整 HTML 或闭合的 `html` fence 时会出现新标签页预览按钮，生成内容运行在不带同源权限的 sandbox iframe 中。

## 验证

```bash
bun run test
bun run lint
bun run typecheck
bun run build
go test -race .
go vet .
```

测试覆盖 SSE 任意拆包、UTF-8、CRLF、畸形事件/错误/断流/Abort；切片内容保持、Unicode；worker 并发硬上限、取消、乱序、失败重试；注册表与按节点隔离的 snapshot 轮询、三态选卡、空响应体停止、会话与翻译持久化、敏感字段不落盘；Go 参数验证、真实子进程启停、互斥、训练回车日志、脱敏、Ready 探测、原始续写路由、静态托管和跨源拒绝。

历史验证记录：此前通过了 Linux 构建和 Windows amd64 交叉构建，并使用实际 `rwkv_lighting_cuda` 验证动态模式服务的 Start → Ready → Restart → Ready → Stop、模型枚举、错误日志和 JSONL 校验。该服务验证没有加载有效基础模型，不代表真实模型 Chat / 翻译质量或训练数值已验证。

新控制台已完成无头浏览器渲染验证（七个路由均能挂载并请求控制面），但未完成交互式视觉验收，不能据此认为已通过浏览器验收。后续验收建议用有效基础模型在 1280 / 1024 像素宽度检查全部页面、流式输出、远端目录浏览、训练及 checkpoint，并验证宿主机对应的 CUDA/HIP 和 Windows DLL 环境。
