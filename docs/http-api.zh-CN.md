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

## 状态端点速查

| 对象 | 方法 | 路径 | 用途 |
|---|---|---|---|
| 初始 state 文件 | `POST` | `/v1/state/upload` | 上传单个 `.pth` 文件 |
| 初始 state 文件 | `GET` / `POST` | `/v1/state/list` | 列出已上传文件 |
| 初始 state 文件 | `DELETE` / `POST` | `/v1/state/delete` | 按 `state_id` 删除文件 |
| 会话缓存 | `POST` | `/state/status` | 列出生成期间保存的会话 |
| 会话缓存 | `POST` | `/state/delete` | 按 `session_id` 删除会话 |

- [初始状态文件：上传、列表与删除](#初始状态文件上传列表与删除)
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
  -F "file=@./state01.pth"
```

响应的 `state_id` 和 `filename` 为「原文件名去扩展名 + UUID-v7」，例如上传
`state01.pth` 得到 `state01-01a1086e-6d3a-744f-b79d-7e67754de680`。
`original_filename` 保留原文件的 basename（支持 POSIX / Windows 路径）。同名重复上传
会得到不同 ID，不会覆盖已有文件。每次请求只允许一个文件；空文件、无效 PTH、缺层、
重复层、非法张量形状/dtype/存储范围会返回 HTTP 400。在生成请求中使用**实际返回的** ID：

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"Continue from the supplied state."}],
    "state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680",
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

兼容性与上传路径详见 [State 上传与权重兼容性](state-upload-compatibility.zh-CN.md)。

### 列表：`GET /v1/state/list`（也支持 `POST`）

```bash
curl -sS "${AUTH_HEADER[@]}" "http://127.0.0.1:8000/v1/state/list"
```

返回 `{"object":"list","data":[...]}`。每个条目包含 `state_id`、`filename`、
`original_filename`、`size_bytes`（文件字节数）、`tensor_count`、`layers`、`heads`、
`head_size`、`created`（Unix 秒）、`created_ms`（Unix 毫秒）和 `uploaded_at`
（RFC 3339 UTC 字符串，例如 `2026-10-05T08:30:00.123Z`）；上传响应包含相同字段。
列表按 `created_ms` 从新到旧排列；同毫秒按 ID 排序。
没有已上传文件时，`data` 是空数组。

### 删除：`DELETE /v1/state/delete`（也支持 `POST`）

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "Content-Type: application/json" \
  --data '{"state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680"}'
```

成功返回 HTTP 200 和 `{"state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680","deleted":true}`；
不存在时返回 HTTP 404，`deleted` 为 `false`。缺少 ID 或各通道 ID 冲突时返回 HTTP 400。
也可以用 `X-RWKV-State-Id` 请求头或 URL 查询参数传入 ID：

```bash
curl -sS "${AUTH_HEADER[@]}" -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "X-RWKV-State-Id: state01-01a1086e-6d3a-744f-b79d-7e67754de680"

curl -sS "${AUTH_HEADER[@]}" -X DELETE \
  "http://127.0.0.1:8000/v1/state/delete?state_id=state01-01a1086e-6d3a-744f-b79d-7e67754de680"
```

此接口管理上传的初始状态文件；生成期间缓存的 `session_id` 会话由
[`/state/status` 和 `/state/delete`](#有状态补全) 管理。

原生推理 Server 统一使用当前模型权重。请求级 Adapter 和 HTTP 管理接口已移除，旧 Adapter 请求字段返回 HTTP 400。State 上传、按请求选择、动态加载、缓存和 Session 行为保持支持。MiSS/NoRA Kernel、包格式与独立训练工具保留；服务端不支持在线 Adapter Checkout 或在线权重融合，离线权重融合请使用 [`rwkv_peft_weight_merge`](peft-weight-merge.zh-CN.md)。

## 聊天补全

OpenAI 风格的聊天接口。`stream:false` 时返回单个 JSON 响应。

`choices[].finish_reason` 表示实际结束原因：采样命中配置的停止 token 时为
`stop`，耗尽 `max_tokens` 时为 `length`。最后一次允许的采样恰好命中停止
token，仍返回 `stop`。批量请求逐条记录原因。SSE 内容包为
`finish_reason:null`，最终包以空 `delta` 返回实际原因，随后发送 `[DONE]`。
显式调用服务端停止或暂停也使用 `stop`。网关原样透传这些值。
该规则覆盖聊天、批量、State 聊天和恢复续推；所有生成接口均支持互斥的 `stop_tokens` 和 `stop`，规则见下述文本停止串说明。

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

## 请求停止条件：token ID 与字符串

`/v1/chat/completions`、`/v1/batch/completions`、`/state/chat/completions`、
`/translate/v1/batch-translate` 和 `/v1/server/resume` 均可传 `stop_tokens: [0, ...]`（原有 token ID 检测路径），
或 `stop: ["</tool_call>", "\nUser:"]`（新文本检测路径）。两个字段同时出现时，
即使值为空数组或 null，也返回 HTTP 400。`stop` 必须是非空字符串组成的数组。
两者均不传时保留默认停止 token；`stop: []` 不启用配置的停止条件，也不沿用默认停止 token。

文本模式在每个生成 token 解码后检测各行尾部，与 `chunk_size` 无关；每行仅保留
最长停止串字节数的尾部，支持跨 token 和 UTF-8 字节边界。检测仅覆盖新生成的文本，
精确匹配后缀、区分大小写；多个停止串同时命中时返回请求数组中排在前面的那个。
输出保留停止串，相关 token 计入生成数量。最后一个允许的 token 命中时仍报告 `stop`。

命中后立即结束该行，响应对应 choice 返回 `finish_reason: "stop"` 和
`matched_stop: "</tool_call>"`。SSE 会立即刷新该行待输出文本，并发送空 `delta`
及上述两个字段，无需等待其他行。token 停止、取消和长度限制不返回 `matched_stop`。
翻译接口在 `translations[]` 中返回 `finish_reason` 和命中时的 `matched_stop`。
动态模型模式的聊天和批量接口同样支持这些参数。恢复接口未传停止参数时沿用暂停时配置，
并保留用于跨暂停点匹配的文本尾部；显式传入停止参数时替换配置并清空原尾部。
已结束行不再生成文本；现有固定大小的后端批次仍保留其槽位，直到全部行结束。

字符串停止请求示例（不要同时传 `stop_tokens`）：

```json
{
  "messages": [{"role": "user", "content": "Reply and end with </tool_call>."}],
  "max_tokens": 256,
  "stream": false,
  "stop": ["</tool_call>", "\nUser:"]
}
```

## 批量补全

为多条 prompt 生成相互独立的续写。每个流式分块通过 `choices[].index` 标识槽位。

```bash
curl -sS "${AUTH_HEADER[@]}" -X POST "http://127.0.0.1:8000/v1/batch/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
    "state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680",
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
    "state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680",
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
    "state_id":"state01-01a1086e-6d3a-744f-b79d-7e67754de680"
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
