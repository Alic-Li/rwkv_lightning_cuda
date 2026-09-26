# HTTP API 示例

[English](http-api.md) | 简体中文 | [返回 README](../README_zh.md)

本页介绍原生推理服务；Launcher 的 Client/Agent 控制、鉴权和后端转发请看[联调指南](../RWKV_Lightning_CUDA_Launcher/docs/integration-guide.md)。

完整的接口参考（中文）见 [rwkv_lightning_api_doc.md](../rwkv_lightning_api_doc.md)。
以下示例假设服务端运行在 `8000` 端口。
如果服务端以 `--password` 启动，请通过 Bearer token 请求头或 JSON 里的 `password`
字段鉴权：

```bash
# 未设置服务密码时：
AUTH_HEADER=()
# 设置了服务密码时，用这一行替换上面的配置：
# AUTH_HEADER=(-H "Authorization: Bearer your-password")
```

以下 curl 命令使用 Bash，均引用上述 `AUTH_HEADER` 数组；请先在同一终端设置它。
上传请求使用 multipart 表单，鉴权必须放在 Bearer 请求头中，不能使用 JSON `password`。

从仓库根目录执行接口串行冒烟测试：

```bash
./test/api_endpoints_test.sh

# 自定义 host、端口或密码：
BASE_URL=http://127.0.0.1:8000 PASSWORD=rwkv7_7.2b ./test/api_endpoints_test.sh
```

## 状态与适配器端点速查

| 对象 | 方法 | 路径 | 用途 |
|---|---|---|---|
| 初始 state 文件 | `POST` | `/v1/state/upload` | 上传单个 `.pth` 文件 |
| 初始 state 文件 | `GET` / `POST` | `/v1/state/list` | 列出已上传文件 |
| 初始 state 文件 | `DELETE` / `POST` | `/v1/state/delete` | 按 `state_id` 删除文件 |
| MiSS 适配器 | `POST` | `/v1/adapters` | 按服务端目录注册推理包 |
| MiSS 适配器 | `GET` | `/v1/adapters` | 列出版本和缓存指标 |
| MiSS 适配器 | `DELETE` | `/v1/adapters` | 删除全部或指定版本的注册 |
| 会话缓存 | `POST` | `/state/status` | 列出生成期间保存的会话 |
| 会话缓存 | `POST` | `/state/delete` | 按 `session_id` 删除会话 |

- [初始状态文件：上传、列表与删除](#初始状态文件上传列表与删除)
- [MiSS 适配器：注册、列表与删除](#miss-适配器注册列表与删除)
- [有状态补全与会话缓存](#有状态补全)

## 服务状态

检查后端是否在运行、已加载的模型、支持的能力、活跃请求与暂停中的请求。

```bash
curl -sS "${AUTH_HEADER[@]}" "http://127.0.0.1:8000/v1/server/status"
```

## 模型列表

OpenAI 兼容的模型列表接口。

```bash
curl -sS "${AUTH_HEADER[@]}" "http://127.0.0.1:8000/v1/models"
```

## Token 计数

对原始文本计数 token。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"text":"hello RWKV"}'
```

对 chat messages 先套用后端 chat prompt 模板再计数。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"messages":[{"role":"user","content":"hello"}]}'
```

## 初始状态文件：上传、列表与删除

### 上传：`POST /v1/state/upload`

先上传一个序列化 RWKV state，再把返回的文件名作为 `state_id` 传给聊天请求。上传的
文件会按 PyTorch state archive 校验，保存在进程本地的临时目录中；可通过删除接口
显式移除，服务端退出时也会自动清理。上传上限 512 MiB。以 PyTorch `bfloat16` 或
`float32` 存储的 state 张量均受支持，加载时会转换为配置的 WKV 运行时精度。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/state/upload" \
  -F "file=@./rwkv-state-agentic.pth"
```

响应以上传文件名的 basename 作为 ID，例如
`{"object":"rwkv.state","state_id":"rwkv-state-agentic.pth"}`。上传同名文件会返回
HTTP 400 和 `already exists` 错误。每次请求只允许一个文件；空文件、无效 PTH 或
不含 `blocks.N.att.time_state` 张量的文件也会返回 HTTP 400。在生成请求中使用该值：

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"Continue from the supplied state."}],
    "state_id":"rwkv-state-agentic.pth",
    "stream":false,
    "max_tokens":8
  }'
```

省略 `state_id` 则使用常规的零初始化 state。所有推理接口都接受同一字符串字段：
`/v1/chat/completions`、`/v1/batch/completions`、`/translate/v1/batch-translate` 与
`/state/chat/completions`。上述每个接口还支持通过 `X-RWKV-State-Id` 请求头（代替 JSON
请求体）传递 `state_id`；各通道给出互相冲突的值会返回 HTTP 400，重复相同值则接受
（`/v1/state/delete` 额外支持 query 串中的 `?state_id=...`，适用同一规则）。
对 batch 请求，上传的 state 会在评估每条 prompt 前复制到每个批次槽位。在有状态接口
上，显式上传的 state 在该请求中优先于缓存的 `session_id` state，生成后的 state 会
回写进会话缓存。携带 `state_id` 的聊天请求默认使用不带思考前缀的经典
`User`/`Assistant` prompt，以匹配典型的 state 调优数据；显式 `think_type` 会覆盖该
行为，`think_type:"none"` 也会显式关闭前缀。

### 列表：`GET /v1/state/list`（也支持 `POST`）

```bash
curl -sS "${AUTH_HEADER[@]}" "http://127.0.0.1:8000/v1/state/list"
```

返回 `{"object":"list","data":[...]}`。每个条目包含 `state_id`、`filename`、
`size_bytes`、`tensor_count` 和 `created`（Unix 时间戳，秒）；上传响应也包含这些字段。
没有已上传文件时，`data` 是空数组。

### 删除：`DELETE /v1/state/delete`（也支持 `POST`）

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "Content-Type: application/json" \
  --data '{"state_id":"rwkv-state-agentic.pth"}'
```

成功返回 HTTP 200 和 `{"state_id":"rwkv-state-agentic.pth","deleted":true}`；
不存在时返回 HTTP 404，`deleted` 为 `false`。缺少 ID 或各通道 ID 冲突时返回 HTTP 400。
也可以用 `X-RWKV-State-Id` 请求头或 URL 查询参数传入 ID：

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "X-RWKV-State-Id: rwkv-state-agentic.pth"

curl -sS "${AUTH_HEADER[@]}" -X DELETE \
  "http://127.0.0.1:8000/v1/state/delete?state_id=rwkv-state-agentic.pth"
```

此接口管理上传的初始状态文件；生成期间缓存的 `session_id` 会话由
[`/state/status` 和 `/state/delete`](#有状态补全) 管理。

## MiSS 适配器：注册、列表与删除

### 注册：`POST /v1/adapters`

同一个接口支持 JSON 注册服务端 PTH 路径，以及 multipart 直接上传本地 PTH。
可使用 `adapter-final.pth` 或新 checkpoint 中的 `training.pth`。
`path` 是服务端路径，建议使用绝对路径；原来的双文件推理目录也兼容。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/adapters" \
  -H "Content-Type: application/json" \
  --data '{"adapter_id":"task-a","path":"/absolute/path/miss_output/checkpoint-100/training.pth"}'

curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/adapters" \
  -F 'adapter_id=task-a' -F 'file=@miss_output/adapter-final.pth'
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `adapter_id` | string | 是 | 非空的适配器名称，后续生成、删除时使用 |
| `path` | string | JSON 注册时必填 | 服务端 PTH 路径或旧推理包目录；multipart 使用 `file` |

新 PTH 内嵌元数据，单独上传即可。旧训练 checkpoint 需要同目录的 `checkpoint.json`；
远程上传旧文件时再添加 `-F 'metadata=@miss_output/checkpoint-100/checkpoint.json'`。
临时上传文件在注册后删除，RAM 只缓存声明精度的 BF16/FP16 D，不保留梯度和优化器张量。BF16 D 在 GPU miss 时于 pinned staging 中转换一次，供 FP16 推理使用。
启用密码时，multipart 请求使用 Bearer header 鉴权；上传文件受 HTTP 请求体大小限制。

成功返回 `{"adapter_id":"task-a","version":"<内容的 SHA-256 版本>"}`。同一 ID 可以注册多个内容版本，
最近注册的版本作为默认版本。注册会校验格式、形状、元数据和内容哈希并缓存到 CPU RAM，
此时不上传 GPU；无效包或 RAM 预算不足等错误返回 HTTP 400。
生成时还会检查推理包中的基模指纹是否与实际加载的模型匹配，不匹配则拒绝使用。

### 列表：`GET /v1/adapters`

```bash
curl -sS "${AUTH_HEADER[@]}" "http://127.0.0.1:8000/v1/adapters"
```

响应的 `data` 数组按注册版本列出 `id`、`version` 和 `manifest`；空列表为 `[]`。
响应顶层还包含以下进程级缓存指标：

| 字段 | 含义 |
|---|---|
| `ram_hits` | CPU RAM 缓存命中次数 |
| `gpu_hits`、`gpu_misses` | GPU 缓存命中、未命中次数 |
| `uploads` | 向 GPU 上传适配器的次数 |
| `h2d_ms` | 从主机向 GPU 传输的累计时间，单位毫秒 |
| `ram_bytes`、`gpu_bytes` | 适配器在 RAM、GPU 中的驻留字节数 |
| `gpu_peak_bytes` | 适配器 GPU 驻留字节数峰值 |

### 在生成请求中使用

所有生成接口都接受 `adapter_id`，以及可选的 `adapter_version` 和 `adapter_scale`。
省略版本时，请求准入会绑定当时最新的注册版本；`adapter_scale` 是绝对有效缩放值，
省略时使用推理包清单中的 scale。一个请求或批次绑定一个不可变适配器句柄。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"你好，请介绍一下自己。"}],
    "adapter_id":"task-a",
    "adapter_scale":1.0,
    "stream":false,
    "max_tokens":64
  }'
```

需要固定版本时，在 JSON 中加入 `"adapter_version":"<注册响应中的 version>"`。
第一次生成会把该版本完整上传 GPU，后续预填充（prefill）和逐词元解码（decode）
复用同一份显存，不会逐层或逐 token 传输。有状态会话按基模运行实例、适配器内容版本、
有效缩放值、初始 state 和 WKV 精度隔离，不兼容的缓存 state 会被拒绝。

### 删除：`DELETE /v1/adapters`

删除该 ID 的所有注册版本：

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/adapters" \
  -H "Content-Type: application/json" \
  --data '{"adapter_id":"task-a"}'
```

只删除一个版本时，额外传入 `adapter_version`：

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/adapters" \
  -H "Content-Type: application/json" \
  --data '{"adapter_id":"task-a","adapter_version":"<注册响应中的 version>"}'
```

返回 `{"deleted":true}`；ID 或版本不存在时同样返回成功。删除只移除注册，
不会删除磁盘上的推理包。已经取得句柄的请求可继续执行，新请求不能再查到已删除版本。
若删除的是最新版本，默认版本回退到其余版本中最近注册的一个。

推理包格式、缓存预算、暂停和恢复行为详见 [MiSS 中文文档](../src/miss/README.zh-CN.md)。

## 聊天补全

OpenAI 风格的聊天接口。`stream:false` 时返回单个 JSON 响应。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"Say hello in one short sentence."}],
    "think_type":"fast",
    "stream":false,
    "max_tokens":8,
    "temperature":1.0,
    "top_k":5,
    "top_p":0.3,
    "alpha_presence":0.2,
    "alpha_frequency":0.2,
    "alpha_decay":0.99,
    "stop_tokens":[0,261,24281],
    "chunk_size":1
  }'
```

`think_type` 控制 chat message prompt 的助手思考前缀：`none`、`fast`、`free`、
`preferChinese`、`en`、`enShort`/`en_short`、`enLong`/`en_long`。`fast` 使用短的闭合
思考前缀，不强制推理；其余模式会在第二个和第三个生成 token 上通过掩码 `111` 与
`754` 强制推理。若省略 `think_type`，`enable_think:true` 或 `think:true` 映射为
`free`；否则默认 `fast`。

`stream:true` 返回 SSE 分块，流以 `data: [DONE]` 结束。

```bash
curl -sS "${AUTH_HEADER[@]}" -N -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"Say hello in one short sentence."}],
    "think_type":"fast",
    "stream":true,
    "max_tokens":8,
    "temperature":1.0,
    "top_k":5,
    "top_p":0.3,
    "alpha_presence":0.2,
    "alpha_frequency":0.2,
    "alpha_decay":0.99,
    "stop_tokens":[0,261,24281],
    "chunk_size":1
  }'
```

## 批量补全

为多条 prompt 生成相互独立的续写。每个流式分块通过 `choices[].index` 标识槽位。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/batch/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
    "state_id":"rwkv-state-agentic.pth",
    "stream":false,
    "max_tokens":8,
    "temperature":1.0,
    "top_k":5,
    "top_p":0.3,
    "alpha_presence":0.2,
    "alpha_frequency":0.2,
    "alpha_decay":0.99,
    "stop_tokens":[0,261,24281],
    "chunk_size":1
  }'
```

```bash
curl -sS "${AUTH_HEADER[@]}" -N -X POST "http://127.0.0.1:8000/v1/batch/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
    "state_id":"rwkv-state-agentic.pth",
    "stream":true,
    "max_tokens":8,
    "temperature":1.0,
    "top_k":5,
    "top_p":0.3,
    "alpha_presence":0.2,
    "alpha_frequency":0.2,
    "alpha_decay":0.99,
    "stop_tokens":[0,261,24281],
    "chunk_size":1
  }'
```

## 批量翻译

兼容批量翻译风格 prompt 的接口。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/translate/v1/batch-translate" \
  -H "Content-Type: application/json" \
  --data '{
    "source_lang":"English",
    "target_lang":"Chinese",
    "text_list":["Hello","Good morning"],
    "state_id":"rwkv-state-agentic.pth"
  }'
```

## 有状态补全

通过 `session_id` 复用并更新已保存的 RWKV state。该接口只接受一条 `contents`
prompt。`session_id` 也可以通过 `X-RWKV-Session-Id` 请求头传递（`/state/delete` 同样
支持）；请求体与请求头给出互相冲突的值会返回 HTTP 400，重复相同值则接受。

```bash

curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/state/chat/completions" \
  -H "Content-Type: application/json" \
  --data "{
    \"session_id\":\"api-test\",
    \"contents\":[\"User: remember the word albatross.\\nAssistant: <think>\\n</think>\\n\"],
    \"stream\":false,
    \"max_tokens\":8,
    \"temperature\":1.0,
    \"top_k\":5,
    \"top_p\":0.3,
    \"alpha_presence\":0.2,
    \"alpha_frequency\":0.2,
    \"alpha_decay\":0.99,
    \"stop_tokens\":[0,261,24281],
    \"chunk_size\":1
  }"
```

```bash
curl -sS "${AUTH_HEADER[@]}" -N -X POST "http://127.0.0.1:8000/state/chat/completions" \
  -H "Content-Type: application/json" \
  --data "{
    \"session_id\":\"api-test\",
    \"contents\":[\"User: continue.\\nAssistant: <think>\\n</think>\\n\"],
    \"stream\":true,
    \"max_tokens\":8,
    \"temperature\":1.0,
    \"top_k\":5,
    \"top_p\":0.3,
    \"alpha_presence\":0.2,
    \"alpha_frequency\":0.2,
    \"alpha_decay\":0.99,
    \"stop_tokens\":[0,261,24281],
    \"chunk_size\":1
  }"
```

列出缓存的会话：

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/state/status" \
  -H "Content-Type: application/json" \
  --data '{}'
```

删除一个缓存的会话：

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/state/delete" \
  -H "Content-Type: application/json" \
  --data "{\"session_id\":\"api-test\"}"
```

## 停止、暂停与恢复

停止当前生成。若没有活跃请求，响应仍返回 `ok:true` 且 `stopped:false`。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/server/stop" \
  -H "Content-Type: application/json" \
  --data '{}'
```

暂停当前生成并保存当前 state。若有请求被暂停，响应中会包含 `request_id`。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/server/pause" \
  -H "Content-Type: application/json" \
  --data '{}'
```

按 `request_id` 恢复被暂停的生成。响应是 SSE 流。

```bash
curl -sS "${AUTH_HEADER[@]}" -N -X POST "http://127.0.0.1:8000/v1/server/resume" \
  -H "Content-Type: application/json" \
  --data '{
    "request_id":"req-xxxxxxxx",
    "stream":true
  }'
```

## CORS 预检

服务端在 API 路由上接受浏览器客户端的 `OPTIONS` 请求。取决于 HTTP 框架的匹配路径，
预检成功可能返回 `200` 或 `204`。

```bash
curl -sS "${AUTH_HEADER[@]}" -i -X OPTIONS "http://127.0.0.1:8000/v1/chat/completions"
```
