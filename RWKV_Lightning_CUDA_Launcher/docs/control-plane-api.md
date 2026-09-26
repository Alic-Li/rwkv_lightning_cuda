# Launcher 控制面 API 与多后端形态

本文是 Launcher 控制面当前实现的接口参考。首次接入先读 [前端与第三方联调指南](integration-guide.md)；工程准备见 [前端开发指南](frontend-development.md)。原生推理 API 见 [完整参考](../../rwkv_lightning_api_doc.md) 与 [示例](../../docs/http-api.zh-CN.md)。

Runtime / Tuning / Quantization 请求体集中定义在 `request_types.go`，路由在 `api.go`。§1–§6 介绍架构与兼容，§7–§8 给出请求字段、返回值与边界；旧设计草案不作为接口契约。

## 1. 三种形态、两个角色、一个二进制

| 形态 | 启动方式 | 角色 | 说明 |
| --- | --- | --- | --- |
| 本机全套 | `rwkv_launcher`（无参数） | Client + Agent 同机 | 现有行为，不退化；开浏览器到 127.0.0.1 |
| 服务器节点 | `rwkv_launcher --listen 0.0.0.0:18766 --token <t>` | 仅 Agent | 有卡的机器；内嵌 WebUI 本机访问照常可用 |
| 控制台 | `rwkv_launcher --client`（或本机无 runtime 二进制时自动） | 仅 Client | Mac 上合法状态，不是错误 |

### 启动参数

| Flag | 默认 | 说明 |
| --- | --- | --- |
| `--listen` | `127.0.0.1:10721` | 监听地址，默认 loopback |
| `--token` | 空 | Agent 鉴权 token；**监听非 loopback 时必须提供，否则拒绝启动（退出码非 0）** |
| `--client` | false | 强制纯 Client 形态，不查找本机 runtime 二进制 |
| `--config` | `~/.rwkv_launcher/launcher.json` | Client 后端注册表落盘位置（JSON，权限 0600） |
| `--card` | 空 | 本 Agent 的默认选卡（`CUDA_VISIBLE_DEVICES` 风格，如 `0` 或 `0,1`） |

硬约束（实现于 `validateStartup`）：

- `--listen` 解析出的地址不是 `127.0.0.1`/`::1`/`localhost` 且未提供 `--token` → 直接退出。理由：`/api/*` 能以任意参数 spawn 任意路径的进程，无 token 的公网监听等于无认证 RCE。
- Client 形态（`--client` 或本机无 runtime 二进制）监听非 loopback → 直接退出。Client 只绑 loopback（§2 分层表）。
- `host:port` 省略 host（`:18766`）视为绑定全部网卡，同样要求 token。

历史遗留修正：runtime 端口与 launcher HTTP 端口的冲突校验从硬编码 `8088` 改为对比 `--listen` 实际端口；README 与 `vite.config.ts` 同步改为 10721。runtime 端口 8088 现在是合法值。

### 两层鉴权，不合并

```
浏览器 ──同源──> Client ──Bearer <agent token>──> Agent ──Bearer <runtime password>──> C++ server
                                                        (127.0.0.1，可以不设)
```

- 配置了 `--token` 后，`/api/*`、`/v1/*` 全部要求 `Authorization: Bearer <token>`（含本机 `/v1` 反代——否则 token 化的 Agent 会因 password 自动注入变成开放推理代理）。
- token 不接受 query string 传参；401 响应体只回 `{"error":"unauthorized"}`，不含任何路径或配置信息；比较用常量时间。
- agent token 与 runtime password 是两个信任域，禁止合并。Agent 校验 token 后，向 runtime 转发时删除该 Authorization，再按 `config.Password` 设置凭证；runtime 未设密码时不发送 Authorization。未配置 Agent token 的本机旧模式仍保留调用方提供的 runtime Authorization，缺省时才注入配置密码。

## 2. 路径表

命名判据：**能被 OpenAI SDK 或第三方客户端调的，留 `/v1`；只有我们自己 WebUI 调的，进 `/api/v1`。**

表中最后一列只用于迁移对照：pre-v1 路径已全部删除，没有 alias，请求它们会得到 404。

### 节点（Agent）

| 方法 | 路径 | 作用 | 对应的 pre-v1 路径（均已移除） |
| --- | --- | --- | --- |
| GET | `/api/v1/node` | role / version / capabilities / runtime 概况 | — |
| GET | `/api/v1/node/metrics` | GPU 指标 | —（v1 新增） |
| POST | `/api/v1/node/fs` | 目录浏览（白名单内） | —（v1 新增） |
| POST | `/api/v1/node/dialog/file` | 宿主机原生文件选择器 | `/api/pick-file` |
| POST | `/api/v1/node/dialog/directory` | 宿主机原生目录选择器 | `/api/pick-directory` |
| POST | `/api/v1/node/dialog/reveal` | 在宿主机打开 checkpoint 目录 | `/api/tuning/open-folder` |

三个 `dialog/*` 仅在 Agent 角色 + loopback 来源下可用；Client 形态或远程调用返回 `400 {"error":"unsupported","reason":"host-local only"}`。能力位 `host_dialog` 只在本机全套形态出现。

### 推理服务进程（Agent）

| 方法 | 路径 | 作用 | 对应的 pre-v1 路径（均已移除） |
| --- | --- | --- | --- |
| GET | `/api/v1/runtime` | runtime 状态 + 配置 | `/api/status`（拆分） |
| POST | `/api/v1/runtime/start` | 启动，body = RuntimeConfig | `/api/start` |
| POST | `/api/v1/runtime/stop` | 停止 | `/api/stop` |
| POST | `/api/v1/runtime/restart` | 用实际运行配置重启 | `/api/restart` |
| POST | `/api/v1/runtime/load` | 选卡（重）加载：停止 → 以 `visible_devices` 重启 → 等就绪 →（动态模式）加载模型 | — |
| POST | `/api/v1/runtime/state/import` | 把节点本地 `.pth` 交给推理服务的 multipart `/v1/state/upload` | — |
| GET | `/api/v1/runtime/logs` | SSE 日志流 | `/logs` |

### 任务（Agent）

训练与量化在 Go 里本就是同一个 `*process`、同一份 `ProcessStatus` schema，合并为 jobs 只是把既有事实写进路径。**`{id}` 是固定值 `tuning` / `quantization`，不是自增 id**；语义仍是单例，不引入任务队列。

| 方法 | 路径 | 作用 | 对应的 pre-v1 路径（均已移除） |
| --- | --- | --- | --- |
| GET | `/api/v1/jobs` | `{"jobs":{"tuning":…,"quantization":…}}` | — |
| GET | `/api/v1/jobs/{id}` | 单个任务状态 | `/api/tuning/status` · `/api/quantization/status` |
| POST | `/api/v1/jobs/tuning` | 起训练，body = TuningConfig | `/api/tuning/start` |
| POST | `/api/v1/jobs/tuning/validate` | 校验 JSONL | `/api/tuning/validate` |
| POST | `/api/v1/jobs/quantization` | 起量化，body = QuantizationConfig | `/api/quantization/start` |
| POST | `/api/v1/jobs/{id}/stop` | 停止 | `/api/tuning/stop` · `/api/quantization/stop` |
| GET | `/api/v1/jobs/{id}/logs` | SSE 日志流 | — |

### 后端注册表（Client）

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/v1/backends` | 列出已注册后端（不含 token，只回 `has_token`） |
| POST | `/api/v1/backends` | 添加，body = `{name, base_url, token}`；添加时同步探测一次 |
| PATCH | `/api/v1/backends/{id}` | 编辑，body = `{name?, base_url?, token?}` 的任意子集；编辑后重新探测 |
| DELETE | `/api/v1/backends/{id}` | 移除 |
| POST | `/api/v1/backends/{id}/probe` | 重新探测能力 |

- `base_url` 不带 `/v1`/`/api` 后缀；带后缀的请求被拒绝。
- `id` 由 Client 生成（6 位十六进制随机串），不用 base_url 做 id。`local` 固定保留给本机全套形态的本地 runtime，不可删除。
- 注册表落盘 `{"backends":[{id,name,base_url,token}]}`，文件 0600；**任何响应都不回显 token**。先写入独立临时文件并同步、替换成功后才更新内存；失败返回错误，添加/编辑/删除不会假报成功，原节点和探测状态保留。
- 编辑（PATCH）只改 body 里出现的字段，`id` 不可改，所以编辑不会让前端丢失当前选中的节点。`token` 为空串表示**保留原 token**（浏览器从不拿到 token，未动过的密码框不能表示「清空」）。`base_url` 走与添加相同的校验；改地址或 token 后立即重新探测。`local` 不可编辑。

### 转发层（Client）

```
/api/v1/backends/{id}/api/v1/...   → <base_url>/api/v1/...
/api/v1/backends/{id}/v1/...       → <base_url>/v1/...
```

剥前缀、换 token。Client 入口先执行 Host/Origin/Sec-Fetch-Site 校验，向后端转发时移除 `Origin` 和 `Sec-Fetch-Site`：这已是服务器间请求，目标 Host 与浏览器访问的 Client 不同。其余端到端 header、method、status、body 原样透传（**不 re-marshal**——Agent 侧 `decode()` 开了 `DisallowUnknownFields`，任何字段增删都会让请求 400）。SSE 路径设 `FlushInterval = -1`。转发传输层无整体/响应头超时（`/v1/model/load` 要等活跃推理排空，几十秒是正常的）。

### 能力探测（Client，启动时、每 30 秒、添加后端时与 probe 时各跑一次）

1. `GET <base_url>/api/v1/node` → 200 且 `role=="agent"` → 全功能 Agent，读 `capabilities`。
   若响应带 `role:"client"` 则明确报错（Client 不是 backend，不能当节点用）。
2. 否则 `GET <base_url>/v1/server/status` → 200 → 裸推理节点，能力集 `["inference"]`。
3. 都不通 → 报错，区分连接失败与 401。

只回应 pre-v1 路径的主机在第 1 步和第 2 步都不命中，落到第 3 步报不可达。这是
有意的：它不是这个 Client 能驱动的节点。

`capabilities` 是开放集合：出现未知能力位必须忽略而不是报错。

### 没有跨版本适配

Client 与 Agent **必须同版本部署**。曾经存在的旧协议适配层（`legacy.go`）已移除：
不再把 `/api/v1/*` 映射回 pre-v1 路径，不再把旧 `/api/status` 拼成
`/api/v1/node` 的形状，`backendView` 也不再有 `legacy` 字段。

探测阶梯因此只剩两级：

1. `GET /api/v1/node` → `role:"agent"` 即为可驱动的节点；
2. `GET /v1/server/status` → 裸推理服务，`kind:"inference_only"`。

只回应 pre-v1 路径的主机两级都不命中，会被判为不可达并附带原因，**不会**被当成
半残的 agent。转发白名单只有 `api/v1/` 和 `v1/` 两个前缀。

解码仍然使用 `DisallowUnknownFields`。同版本部署下这是优点（字段拼写错误会立刻
报错而不是被静默忽略）；一旦将来要支持混版本，这一条需要重新评估，见
[未决事项](open-items.md)。

## 3. 关键 schema

### `GET /api/v1/node`（Agent）

在旧 `RuntimeState`（`status/running/error/logs/elapsed/config/base_url/backend`）之上新增：

```json
{
  "role": "agent",
  "version": "1.7.5",
  "capabilities": ["runtime", "tuning_state", "tuning_miss", "quantization", "metrics", "fs", "state_import", "host_dialog"],
  "available": true,
  "visible_devices": "",
  "card": "0"
}
```

- `config.password` 永远是空串（不回显密码，历史行为不变）。
- `available`：本机是否存在 runtime 二进制（Client 形态为 false）。
- `visible_devices`：当前 runtime 进程被钉住的原始选卡串（§选卡），未注入时为 `""`。
- `card`：Agent 启动参数 `--card` 的原值，未指定时为 `""`。它是硬性钉定而不是默认值：显式指定其他选卡（包括空串「不注入」）的启动/加载请求会被 400 拒绝（§选卡）。
- `version` 来自构建时的 `-ldflags "-X main.launcherVersion=…"`；本地构建为 `dev`。

Client 形态下 `/api/v1/node` 返回 404（`{"error":"client_only", …}`），避免探测者把 Client 误判为 Agent。

### `GET /api/v1/backends`（Client）

```json
{
  "backends": [
    {
      "id": "local",
      "name": "本机",
      "base_url": "http://127.0.0.1:10721",
      "has_token": false,
      "kind": "agent",
      "capabilities": ["runtime", "…"],
      "reachable": true,
      "last_probe": 1758240000,
      "probe_error": ""
    }
  ]
}
```

探测成功后 `kind` 为 `agent` / `inference_only`；尚未探测或探测失败时可为空字符串。`token` 字段不出现在任何响应里。

### `GET /api/v1/node/metrics`（Agent）

```json
{
  "available": true,
  "vendor": "nvidia",
  "sampled_at": 1758240015,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4060 Ti",
      "memory_total_bytes": 17179869184,
      "memory_used_bytes": 13421772800,
      "own_memory_bytes": 12884901888,
      "utilization_percent": 94,
      "temperature_c": 71,
      "power_watts": 158
    }
  ]
}
```

- NVIDIA 走 NVML（cgo dlopen `libnvidia-ml.so.1`，运行时解析，不需要构建期 CUDA toolkit）；AMD 走 `rocm-smi --json`（1 秒 TTL 缓存）。都不 fork `nvidia-smi`。
- `index` 是 NVML 物理索引。CUDA_VISIBLE_DEVICES 会重映射子进程内的逻辑索引，两者不混用：逻辑设备号只出现在子进程日志里，不改写。
- `temperature_c` / `power_watts` 是可选字段：取不到就整个字段不出现，不许填 0 或 -1。
- `own_memory_bytes`（可选）是本 Agent 管理的 runtime 在这张卡上占用的显存：NVML 按进程统计显存，Agent 把 runtime 主进程及其全部子孙进程（按 `/proc` 父进程链判断）的占用相加。只有 NVML 路径提供；`rocm-smi` 无法按进程归属，非 Linux-cgo 构建没有 NVML，二者都**省略**该字段；本 Agent 的 runtime 没在运行时同样省略。取值不超过 `memory_used_bytes`。前端据此把显存条画成两段：自己的在底部，其他进程的叠在上面并打斜线；字段缺失时退回按 `visible_devices` 整卡归属。
- 取不到任何采样时返回 `{"available": false, "vendor": "unknown", "reason": "nvml: …; rocm-smi: not found in PATH", "gpus": []}`——不许返回全 0 的 gpus 数组。
- Linux 上以 `CGO_ENABLED=0` 构建的 launcher 没有 NVML 支持（metrics 返回 available=false）；发布包在 Linux 上以 `CGO_ENABLED=1` 构建以启用 NVML。Windows 构建暂无 NVML 路径（后续可经 `syscall.NewLazyDLL` 增加，未列入本轮）。

### 选卡 `visible_devices`（三个请求体的可选字段）

`RuntimeConfig`（`/api/v1/runtime/start`、`/api/v1/runtime/load`）、`TuningConfig`（`/api/v1/jobs/tuning`）、`QuantizationConfig`（`/api/v1/jobs/quantization`）各增加可选字段：

```json
{ "visible_devices": "0,1" }
```

字段为**字符串**。优先级（高覆盖低）：

1. 请求体 `visible_devices`（`""` = 显式不注入，与缺省不同）
2. Agent 启动参数 `--card <spec>`
3. Agent 进程自己继承的 `CUDA_VISIBLE_DEVICES`（子进程原样继承，不注入）
4. 都没有 → **自动选卡**：Agent 采样各卡空闲显存（NVML，Windows 无 NVML 绑定时退回一次
   `nvidia-smi --query-gpu=index,memory.free`），把子进程钉到空闲最大的卡，并标记为显式
   （参与按卡互斥、tune cache 跟卡）；选卡结果以 `auto device placement: GPU <n>` 写入对应
   子进程日志。采样不可用时退回旧行为：不注入（子进程看见全部卡，驱动落 device 0）。
   自动放置按「空闲显存最大」而非「占用率最低」——一张 48G 卡用掉 20% 仍优于一张空的 24G 卡。

**`--card` 是硬性钉定，不是默认值**：Agent 以 `--card <spec>` 启动时，任何显式指定其他选卡的
请求——包括空串「显式不注入」——都会被 400 拒绝（`this launcher is pinned to GPU "…" by --card;
loading on "…" is refused`），runtime/tuning/quantization/`runtime/load` 四条路径一致。与钉定串
同集合的写法（如 `1,0` vs `0,1`）放行；请求缺省 `visible_devices` 时照常落到 `--card`。
`/api/v1/node` 的 `card` 字段回显钉定串，前端据此把选卡控件锁到该卡。`runtime/load` 在
**停止当前 runtime 之前**先做该校验，避免校验失败把已在服务的节点摘掉。

注入方式：spawn 子进程时改写环境变量——NVIDIA 设 `CUDA_VISIBLE_DEVICES`；AMD 设 `HIP_VISIBLE_DEVICES` + `ROCR_VISIBLE_DEVICES`（厂商判断复用 metrics 的 vendor）。三个设备变量先全部移除再注入，避免继承值与注入值叠加。

连带行为：

- **按卡互斥**：runtime ↔ tuning 的互斥从全局改为按卡——只有两边解析出的设备集有交集（或任一边不可判定，包括继承 env、UUID 形式）才拦截；显式 `0` 与显式 `1` 并行放行。误拦截的代价是多点一次，误放行的代价是同卡 OOM，所以不可判定一律按有交集处理。quantization 维持现状（本就不参与互斥），本轮不新增拦截。
- **tune cache 跟卡**：显式选卡且未指定 `tune_cache` 且非动态加载时，Agent 自动传 `--tune-cache <state_db 目录>/<模型名>.dev-<tag>.w8a16.tune`（tag 为 `visible_devices` 的路径安全形式）。动态加载模式保持 C++ 侧「模型名 + GPU 名」的默认命名（同型号双卡的复用风险由 `visible_devices` 回显暴露，可用显式 `tune_cache` 消除）。C++ 侧 `load_w8a16_tune_cache` 本就校验 GPU 名与模型名，跨型号复用会自动 miss 重调。
- **逻辑/物理索引不混**：metrics 回物理索引，`/api/v1/node` 与 `/api/v1/runtime` 回原始 `visible_devices` 字符串，由前端负责对应；子进程日志中的逻辑设备号不改写。

### 远程文件浏览 `POST /api/v1/node/fs`

请求（省略 path 则返回白名单根列表）：

```json
{ "path": "/data/models" }
```

目录响应：

```json
{ "path": "/data/models", "parent": "", "entries": [ { "name": "g1i-7b.pth", "is_dir": false, "size": 14700000000 } ] }
```

- 白名单 = `appDir()` + 用户主目录（`~`，模型目录常在 `~/models` 一类位置）+ 注册表式的「本 launcher 实际用过的目录」（runtime/tuning/quantization 配置引用过的目录在成功启动/校验时记录，持久化在 `<appDir>/launcher_fs_roots.json`，上限 64 条）。
- 根列表响应额外带 `"default"`：WebUI 浏览器在表单为空时默认展开的目录（`appDir()`，退化到主目录）。
- 请求 path 指向白名单内的**文件**（如从表单带进来的模型路径）时，返回其所在目录的列表（`path` 为父目录）；白名单外的文件仍是 403。
- 越界（含 `..` 穿越清洗后落在白名单外、指向白名单外的符号链接）返回 403，**响应体不回显被拒绝的路径**；不存在的路径同样 403，避免构成文件存在性 oracle。
- 目录内指向白名单外的符号链接仍会列出名字（它确实在该目录里），但不显示类型/大小等目标元信息。
- `parent` 为空串表示当前已在白名单根（无法再向上）。

## 4. 兼容与不变量

- 老 `/api/*` 与 `/logs` 全部保留为 alias，内部转调新 handler，无第二份实现。
- `/v1/*`（OpenAI 兼容面）冻结：路径、语义、`translate contents→batch` 的 body 改写（仅存量 dist 依赖，等前端改造同批摘除）都保持原样。
- 角色不变量：Client 不 spawn 进程（I1）；Agent 不知道集群（I2，无 peers/cluster/nodes 字段）；Client 不基于多 Agent 状态做二次判断（I3，探测仅用于展示）；浏览器只与同源 Client 说话（I4，无 CORS、无跨源 fetch）。
- `/logs` 保持 ticker 轮询 + 上一行比对的 SSE 模型，不换 WebSocket。
- 单例进程模型不变：一个 Agent 管一个 runtime，多卡多 Agent。

## 5. M0 待填项决议（2026-09-19）

| 项 | 决议 |
| --- | --- |
| T1 metrics 字段表 | schema 见上节（§6.3 已定稿）。NVML 字段全量可取（index/name/mem/util/temp/power，可选字段缺失即省略）。**rocm-smi 的实际 key 集合仍需在 W7900 目标机上确认**：解析按 key 后缀匹配（`VRAM Total Memory (B)`、`GPU use (%)`、`Temperature (Sensor junction|edge) (C)`、`Average Graphics Package Power (W)`、`Card series`），上机后如 label 有出入只需调 `metrics.go` 的后缀表 |
| T2 capabilities 初始清单 | `runtime` / `tuning_state` / `tuning_miss` / `quantization` / `metrics` / `fs` / `state_import` / `host_dialog`（仅本机全套）。量化**不**按 w8a16/w4a16 拆位：一个 `rwkv_quantize` 二进制同时支持两种 format，format 是任务参数不是部署属性 |
| T3 token 生成与分发 | 由操作者经 `--token` 显式提供，不自动生成（自动生成的 token 会进启动日志）。存储位置：只存在于启动参数/进程内，不落盘。轮换：用新 token 重启 Agent，再在各 Client 上删除再注册对应后端（当前没有更新接口，ID 会变化） |
| T4 Client 配置文件位置 | `~/.rwkv_launcher/launcher.json`（macOS/Linux `$HOME`，Windows `%USERPROFILE%`），权限 0600，`--config` 可覆盖。理由：注册表是用户级状态，必须活过 launcher 二进制升级与「每卡一个文件夹」的整目录替换；appDir 会随构建/部署位置漂移 |

## 6. 构建注意

- 包已拆为多文件（`main.go` / `request_types.go` / `api.go` / `backends.go` / `fsbrowse.go` / `metrics*.go` / `devices.go`），构建/运行命令用包路径：`go build .` / `go run .`，不能再 `go build main.go`。
- Linux 启用 NVML 需要 `CGO_ENABLED=1 go build .`（只需 gcc 与 libdl，不需要 CUDA toolkit）；`CGO_ENABLED=0` 构建可用，metrics 走 available=false 分支。macOS/Windows 构建不受影响（NVML 文件带 `//go:build linux && cgo` 约束）。

## 7. 请求体与进程状态完整参考

以下字段对应 `request_types.go`，JSON 使用 snake_case。控制面 JSON 解码上限 1 MiB，拒绝未知字段与尾随数据；路径都属于 Agent 主机，相对路径按 Launcher 可执行文件目录解释。前端建议提交完整配置，避免误用服务器已有配置或 Go 零值。

### RuntimeConfig

用于 `POST /api/v1/runtime/start`。start 在上次保存的配置上解码，因此省略字段会保留已有值；首次启动的默认值不应被当成每次请求都会重新应用。

**例外：`visible_devices` 不继承。** 它每次都按 §选卡 优先级链重新解析——省略即表示「交给 Agent 决定」（前端「自动」模式正是省略该字段）。否则一旦显式选过一次卡，就再也无法回到自动放置。restart 使用已保存配置（含选卡），不使用请求体里的新表单。

#### `POST /api/v1/runtime/load`

body：`{ "model": "<动态模式模型 ID，可省略>", "visible_devices": "<选卡串，可省略>" }`。
用**保存的配置**（含真实 runtime 密码，WebUI 永远看不到）执行：停止当前 runtime →
按 §选卡 优先级链解析设备（请求体给了 `visible_devices` 就用它；缺省走自动选卡）→
重启 → 轮询 `/v1/server/status` 直到 ready（上限 10 分钟，覆盖非动态模式的多 GB 冷加载）→
动态模式下若给了 `model` 则带认证调用 `/v1/model/load`（15 分钟上限）。响应
`{ "ok": true, "visible_devices": "<实际钉定串>", "model": "<已加载模型或空串>" }`。
选卡串写入保存的配置，之后的 restart 沿用该卡。CUDA 在进程初始化时绑定设备，因此换卡
必然中断活跃推理——这是接口契约，不是缺陷。当前 WebUI 不再调用它：选卡与重启都在
「推理服务」页完成（`runtime/start` 带 `visible_devices`），该路径保留给其他客户端。

#### `POST /api/v1/runtime/state/import`

body：`{ "path": "<节点上的 .pth 绝对路径>" }`。原生 API 能从服务端路径注册 **adapter**
（`POST /v1/adapters {adapter_id, path}`），但 **state 只有 multipart 上传**，没有路径形式。
于是这一跳由 Agent 补：按 §5.5 白名单解析路径（越界 403 且不回显路径）→ 校验是普通文件、
非空、`.pth`、不超过原生 512 MiB 上限 → 以流式 multipart POST 到本机推理服务
`/v1/state/upload`（multipart 用 Bearer 头鉴权，不读 JSON `password`）→ **原样转发**推理服务的
状态码与响应体，使「已存在」「不是 state 归档」这类拒绝以原生措辞抵达前端。

能力位：`state_import`。没有该位的旧 Agent 上，WebUI 隐藏「从节点路径导入」，只留本机上传。

| 字段 | JSON 类型 | 含义 / 校验 |
| --- | --- | --- |
| `model_path` | string | 必须存在；动态加载模式为目录，否则为模型文件 |
| `vocab_path` | string | 词表文件；空值使用 `./rwkv_vocab_v20230424.txt` |
| `port` | **string** | 原生服务端口，如 `"8000"`，1–65535，不能与 Launcher 端口冲突 |
| `password` | string | 原生服务密码；与 Agent token 不同；状态接口回显为空 |
| `use_wkv32` | boolean | 注入 `--wkv32` |
| `chunk_load` | boolean | 注入 `--chunk-load` |
| `enable_dynamic_loading` | boolean | 注入 `--enable-dynamic-loading` |
| `chunk_size` | integer | prefill 大小；0 按 128，负数拒绝 |
| `state_db_path` | string | 会话数据库路径；空串不传对应 CLI 参数 |
| `tune_cache` | string | 调优缓存路径；缺省时按选卡策略处理 |
| `visible_devices` | string，可省略 | 缺省沿用 Agent 配置，空串显式不注入，见 §3 |

```json
{
  "model_path": "/data/models/model.pth",
  "vocab_path": "/data/rwkv_vocab_v20230424.txt",
  "port": "8000",
  "password": "",
  "use_wkv32": false,
  "chunk_load": false,
  "enable_dynamic_loading": false,
  "chunk_size": 128,
  "state_db_path": "rwkv_sessions.db",
  "tune_cache": "",
  "visible_devices": "0"
}
```

Start / stop / restart 成功均为 `200 {"ok":true}`。start 成功不等于模型 ready；继续轮询 runtime 状态。Client 形态下 start/restart 返回 404；runtime stop 为兼容保留无本地进程时的幂等成功，runtime 日志入口也不做 Agent gate。调用者仍应先按角色与能力选择功能。

### TuningConfig

用于 `POST /api/v1/jobs/tuning`。训练请求从零值解码，**不会自动套用前端表单的默认值**；必须提供有效数值。`state` 与 `miss` 共用同一个训练进程槽。

| 字段 | JSON 类型 | 含义 / 校验 |
| --- | --- | --- |
| `method` | string | `state` 或 `miss`；空值按 state |
| `model` | string | 已存在的 BF16 `.pth` 基模，不接受 `.rwkvq` |
| `data` | string | 已存在的 JSONL 文件；非空行恰好一个字符串 `text` 字段 |
| `output` | string | 非空输出目录路径 |
| `vocab` | string | 可选词表文件；非空时须存在 |
| `ctx` / `chunk` | integer | 均 > 0，`chunk <= ctx` |
| `epochs` | integer | > 0 |
| `batch_size` | integer | 1–128 |
| `max_steps` | integer | >= 0，0 不额外限制步数 |
| `lr` / `lr_final` | number | 均 > 0 |
| `warmup_steps` / `save_every` / `seed` | integer | 均 >= 0 |
| `optimizer` | string | `adam` 或 `muon`；空值按 adam，MiSS 仅接受 adam |
| `wkv_tape` | boolean | 共享 WKV tape 开关 |
| `rank` | integer | MiSS 必须 1–1024；state 模式不使用 |
| `alpha` | number | MiSS alpha，透传训练 CLI；state 模式不使用 |
| `targets` | string | MiSS 的 `all`，或下列目标名的逗号分隔列表，不能重复 |
| `state` | string | MiSS 可选初始 state 文件，须存在 |
| `resume` | string | MiSS 可选续训目录，须存在 |
| `visible_devices` | string，可省略 | 同 RuntimeConfig |

MiSS 目标：`att.receptance.weight`、`att.key.weight`、`att.value.weight`、`att.output.weight`、`ffn.key.weight`、`ffn.value.weight`。`state` / `resume` 字段目前仅在 MiSS 模式被使用，不要把它们解释成 state tuning 的通用续训接口。

有效的 state tuning 请求示例（路径需替换）：

```json
{
  "method": "state",
  "model": "/data/models/model.pth",
  "data": "/data/train.jsonl",
  "output": "/data/state_output",
  "vocab": "/data/rwkv_vocab_v20230424.txt",
  "ctx": 512,
  "chunk": 128,
  "epochs": 1,
  "batch_size": 16,
  "max_steps": 0,
  "lr": 0.0005,
  "lr_final": 0.0001,
  "warmup_steps": 10,
  "save_every": 100,
  "seed": 1234,
  "optimizer": "adam",
  "wkv_tape": false,
  "visible_devices": "0"
}
```

MiSS 请求在此基础上改 `method:"miss"`，补 `rank:16`、`alpha:16`、`targets:"all"`，按需补 `state` / `resume`。启动成功为 `200 {"ok":true}`。预先验证数据集：`POST /api/v1/jobs/tuning/validate`，body `{"path":"/data/train.jsonl"}`，成功返回 `{"samples":123}`；只校验格式，不代表训练已成功。

### QuantizationConfig

用于 `POST /api/v1/jobs/quantization`：

```json
{
  "input_path": "/data/models/model.pth",
  "output_path": "/data/models/model.w4a16.rwkvq",
  "format": "w4a16",
  "group_size": 128,
  "visible_devices": "0"
}
```

| 字段 | JSON 类型 | 含义 / 校验 |
| --- | --- | --- |
| `input_path` | string | 已存在的 BF16 `.pth` 模型 |
| `output_path` | string | `.rwkvq` 路径，父目录须存在；文件须尚不存在，不能覆盖输入 |
| `format` | string | `w8a16` 或 `w4a16`，空值按 w4a16 |
| `group_size` | integer | W4A16 为 32/128；0 按 128；W8A16 不使用 |
| `visible_devices` | string，可省略 | **已忽略**：`rwkv_quantize` 为纯 CPU 工具（`tools/CMakeLists.txt` 不链接 `rwkv::backend`），Agent 不做 `--card` 校验、不采样显存、不注入 `CUDA_VISIBLE_DEVICES`。字段保留仅为兼容仍在发送它的客户端 |

量化不占显卡，因此也不参与 runtime↔training 的同卡互斥。启动成功为 `200 {"ok":true}`。训练/量化的 `POST /api/v1/jobs/{id}/stop` 成功为 **HTTP 200 空响应体**，不要无条件 `response.json()`。

### ProcessStatus 与 RuntimeState

`GET /api/v1/jobs` 返回 `{"jobs":{"tuning":ProcessStatus,"quantization":ProcessStatus}}`，单项 `GET /api/v1/jobs/{id}` 直接返回状态对象。

```ts
type ProcessStatus = {
  status: "offline" | "starting" | "ready" | "stopping" | "error" | "running" | "completed";
  running: boolean;
  error: string;
  logs: string[];
  elapsed: number; // 秒
  checkpoint: string; // 最近真实保存的路径，没有则为空串
  progress: null | {
    step?: number; total?: number; epoch?: number; epochs?: number;
    loss?: number; lr?: number; tokens_per_second?: number; eta?: number;
  };
  losses: { step: number; loss: number }[];
  available?: boolean; // 对应工具是否存在
  miss_available?: boolean; // 仅 tuning，MiSS 工具是否存在
  output_path?: string; // 仅 quantization
};

type RuntimeState = ProcessStatus & {
  config: RuntimeConfig; // password 永远为空
  base_url: string; // Agent 本机 runtime 地址，只用于展示，浏览器不直接连接
  translation_adapter: boolean; // 当前旧 WebUI 兼容标记
  available: boolean;
  visible_devices: string;
  backend?: Record<string, unknown>; // ready 探测成功时的原生 /v1/server/status 响应
};
```

`GET /api/v1/node` 在 RuntimeState 上加 `role:"agent"`、`version` 和 `capabilities`；本机全套也以 `role:"agent"` 对外报告。状态数据中的未知字段应忽略，`progress` 可为 null，不得假造训练指标。

日志：`GET /api/v1/runtime/logs` 和 `GET /api/v1/jobs/{id}/logs` 均为 `text/event-stream`，每个 `data:` 内容是一行纯文本，无 JSON 包装、无 `[DONE]`、无 `Last-Event-ID` 支持。停止进程不会关闭日志连接；断开由调用方取消请求完成。重新连接会重发当前缓冲日志。

## 8. 注册表与辅助接口返回值

| 请求 | 请求体 | 成功返回 |
| --- | --- | --- |
| `GET /api/v1/backends` | 无 | `{"backends":[BackendView,...]}` |
| `POST /api/v1/backends` | `{"name":"...","base_url":"http://host:18766","token":"..."}` | 单个 BackendView，HTTP 200 |
| `PATCH /api/v1/backends/{id}` | `{"name":"...","base_url":"...","token":"..."}` 的任意子集；`token:""` = 保留原值 | 单个 BackendView（已重新探测），HTTP 200 |
| `DELETE /api/v1/backends/{id}` | 无 | `{"ok":true}` |
| `POST /api/v1/backends/{id}/probe` | 无 | 单个 BackendView，HTTP 200 |
| `POST /api/v1/node/fs` | `{}` 或 `{"path":""}` | `{"roots":["/data",...],"default":"/app/dir"}` |
| `POST /api/v1/node/fs` | `{"path":"/data"}` | `{"path":"/data","parent":"","entries":[...]}`；path 为白名单内文件时返回其父目录 |
| `POST /api/v1/node/dialog/file` / `directory` | 无 | `{"path":"..."}`；取消时可能为空串 |
| `POST /api/v1/node/dialog/reveal` | 无 | `{"ok":true}`，打开最近 checkpoint 所在目录，不接受任意 path |

`BackendView` 字段见 §3。`kind` 探测成功为 `agent` / `inference_only`；未探测或失败时可为 `""`。`last_probe` 为 Unix 秒，0 表示未探测；探测结果只存在内存中。Client 启动后每 30 秒对全部后端重跑展示型探测（I3：结果只用于显示，从不作门控），`reachable` 与 `last_probe` 随之刷新。新增后端会先保存配置，再探测，HTTP 200 并不保证 `reachable:true`。没有编辑/更新后端的接口；改 token 或地址需删除再注册，ID 会变化。`local` 不能删除。

完整联调流程、错误码、同源约束、第三方 SDK、非 `/v1` 原生路径的转发限制，见 [联调指南](integration-guide.md)。
