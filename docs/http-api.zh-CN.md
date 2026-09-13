# HTTP API 示例

[English](http-api.md) | 简体中文 | [返回 README](../README.md)

完整的接口参考（中文）见 [rwkv_lightning_api_doc.md](../rwkv_lightning_api_doc.md)。
以下示例假设服务端运行在 `8000` 端口。
如果服务端以 `--password` 启动，请通过 Bearer token 请求头或 JSON 里的 `password`
字段鉴权：

```bash
AUTH_HEADER=(-H "Authorization: Bearer rwkv7_7.2b")
```

对所有接口跑一遍串行冒烟测试：

```bash
../test/api_endpoints_test.sh

# 自定义 host、端口或密码：
BASE_URL=http://127.0.0.1:8000 PASSWORD=rwkv7_7.2b ../test/api_endpoints_test.sh
```

## 服务状态

检查后端是否在运行、已加载的模型、支持的能力、活跃请求与暂停中的请求。

```bash
curl -sS "http://127.0.0.1:8000/v1/server/status"
```

## 模型列表

OpenAI 兼容的模型列表接口。

```bash
curl -sS "http://127.0.0.1:8000/v1/models"
```

## Token 计数

对原始文本计数 token。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"text":"hello RWKV"}'
```

对 chat messages 先套用后端 chat prompt 模板再计数。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"messages":[{"role":"user","content":"hello"}]}'
```

## Chat completions

OpenAI 风格的聊天接口。`stream:false` 时返回单个 JSON 响应。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/chat/completions" \
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

先上传一个序列化 RWKV state，再把返回的文件名作为 `state_id` 传给聊天请求。上传的
文件会按 PyTorch state archive 校验，保存在进程本地的临时目录中；可通过删除接口
显式移除，服务端退出时也会自动清理。上传上限 512 MiB。以 PyTorch `bfloat16` 或
`float32` 存储的 state 张量均受支持，加载时会转换为配置的 WKV 运行时精度。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/state/upload" \
  -F "file=@./rwkv-state-agentic.pth"
```

响应以上传文件名的 basename 作为 ID，例如
`{"object":"rwkv.state","state_id":"rwkv-state-agentic.pth"}`。上传同名文件会返回
`already exists` 错误。在生成请求中使用该值：

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/chat/completions" \
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

列出或删除已上传的 state：

```bash
curl -sS "http://127.0.0.1:8000/v1/state/list"

curl -sS -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "Content-Type: application/json" \
  --data '{"state_id":"rwkv-state-agentic.pth"}'
```

启用 `--password` 时，multipart 上传请求请使用 `Authorization: Bearer ...` 请求头。

`stream:true` 返回 SSE 分块，流以 `data: [DONE]` 结束。

```bash
curl -sS -N -X POST "http://127.0.0.1:8000/v1/chat/completions" \
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

## Batch completions

为多条 prompt 生成相互独立的续写。每个流式分块通过 `choices[].index` 标识槽位。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/batch/completions" \
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
curl -sS -N -X POST "http://127.0.0.1:8000/v1/batch/completions" \
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
curl -sS -X POST "http://127.0.0.1:8000/translate/v1/batch-translate" \
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

curl -sS -X POST "http://127.0.0.1:8000/state/chat/completions" \
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
curl -sS -N -X POST "http://127.0.0.1:8000/state/chat/completions" \
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
curl -sS -X POST "http://127.0.0.1:8000/state/status" \
  -H "Content-Type: application/json" \
  --data '{}'
```

删除一个缓存的会话：

```bash
curl -sS -X POST "http://127.0.0.1:8000/state/delete" \
  -H "Content-Type: application/json" \
  --data "{\"session_id\":\"api-test\"}"
```

## 停止、暂停与恢复

停止当前生成。若没有活跃请求，响应仍返回 `ok:true` 且 `stopped:false`。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/server/stop" \
  -H "Content-Type: application/json" \
  --data '{}'
```

暂停当前生成并保存当前 state。若有请求被暂停，响应中会包含 `request_id`。

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/server/pause" \
  -H "Content-Type: application/json" \
  --data '{}'
```

按 `request_id` 恢复被暂停的生成。响应是 SSE 流。

```bash
curl -sS -N -X POST "http://127.0.0.1:8000/v1/server/resume" \
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
curl -sS -i -X OPTIONS "http://127.0.0.1:8000/v1/chat/completions"
```
