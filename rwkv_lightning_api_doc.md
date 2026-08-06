# RWKV Lightning Backend Documentation

## Overview

RWKV Lightning provides high-performance inference implementations for RWKV language models.

Currently, RWKV Lightning supports two independent inference backends:

| Backend | Implementation | Target |
|-|-|-|
| rwkv_lightning | Python / PyTorch | Research, development, flexible deployment |
| rwkv_lightning_cuda | C++ / CUDA | High-performance production inference |

Both backends implement the same RWKV model architecture, but use different execution runtimes.

```
          RWKV Model
      +---------------+
              |
      +-------+-------+
      |               |
      v               v
+-------------+    +-----------------+
rwkv_lightning     rwkv_lightning_cuda
+-------------+    +-----------------+
PyTorch Runtime    Native CUDA Runtime
Python API         C++ API
torch Tensor       CUDA Memory
PyTorch Ops        CUDA Kernel
      |               |
      +-------+-------+
              |
             GPU

```


---

# 1. rwkv_lightning

Repository:

```
https://github.com/RWKV-Vibe/rwkv_lightning
```


## Introduction

`rwkv_lightning` is a PyTorch-based RWKV inference backend.

Features:

- Python implementation
- PyTorch acceleration
- Easy model modification
- Flexible debugging
- Research friendly
- Supports GPU inference


Recommended scenarios:

- Model research
- Algorithm verification
- New RWKV architecture testing
- Rapid prototyping

---

## API Documentation

本节描述 `rwkv_lightning` Python/PyTorch 后端的当前 HTTP API。服务基于 FastAPI，包含
原生批量推理、OpenAI 风格聊天、三级 state cache、FIM 和超大 batch 推理接口。

### 1. 启动服务

```bash
# FP16（默认）
python app.py \
  --model-path /path/to/model \
  --inference-engine fp16 \
  --port 8000 \
  --password rwkv7_7.2b

# GemLite 预打包量化模型
python app.py \
  --model-path /path/to/model-gemlite-int8 \
  --inference-engine gemlite \
  --port 8000

# CUTLASS W8A16 模型，仅支持 NVIDIA CUDA
python app.py \
  --model-path /path/to/model-w8a16 \
  --inference-engine cutlass \
  --port 8000
```

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--model-path` | 是 | 无 | 模型路径；可以带或不带 `.pth` 后缀。 |
| `--inference-engine` / `--backend` | 否 | `fp16` | 可选 `fp16`、`gemlite`、`cutlass`。 |
| `--port` | 否 | `8000` | HTTP 监听端口。 |
| `--password` | 否 | 不启用 | API 密码；省略时不执行密码校验。 |

服务固定监听 `0.0.0.0`。所有 POST 请求应使用：

```http
Content-Type: application/json
```

FastAPI 服务允许跨域请求。客户端断开连接时，推理任务会收到取消信号；尚未开始 prefill 的
请求会进入 FIFO prefill queue。若请求 batch size 超过模型的 prefill 上限，返回 HTTP `400`：

```json
{
  "error": "bsz overflow, Max bsz=32",
  "request_bsz": 64,
  "max_bsz": 32
}
```

### 2. 鉴权

普通 `/v1`、`/v2`、state、FIM 和 big-batch 路由通过 JSON 字段传递密码：

```json
{"password":"rwkv7_7.2b"}
```

`/openai/v1/*` 同时支持标准 Bearer header：

```http
Authorization: Bearer rwkv7_7.2b
```

鉴权失败返回 HTTP `401`：

```json
{"error":"Unauthorized: invalid or missing password"}
```

当前 `/v1/models` 与 `/translate/v1/batch-translate` 不校验密码；若服务暴露至公网，应在
反向代理层保护这些路由。

### 3. 路由总览

| Method | Path | 用途 | 流式 |
|---|---|---|---:|
| `GET` | `/v1/models` | 查询模型 | 否 |
| `POST` | `/v1/chat/completions` | 原生多 prompt 批量补全 | 可选 |
| `POST` | `/v2/chat/completions` | 使用 V2 采样器的批量补全 | 可选 |
| `POST` | `/translate/v1/batch-translate` | 批量翻译 | 否 |
| `POST` | `/FIM/v1/batch-FIM` | FIM 批量补全 | 可选 |
| `POST` | `/big_batch/completions` | 大 batch 流式补全 | 是 |
| `POST` | `/state/chat/completions` | 单分支 stateful 补全 | 可选 |
| `POST` | `/multi_state/chat/completions` | 可分叉对话 stateful 补全 | 可选 |
| `POST` | `/state/status` | 查询三级 state cache | 否 |
| `POST` | `/state/delete` | 删除 state cache | 否 |
| `GET` | `/openai/v1/models` | OpenAI 风格模型列表 | 否 |
| `POST` | `/openai/v1/chat/completions` | OpenAI 风格单路聊天补全 | 可选 |

### 4. 原生生成参数

原生路由复用以下请求字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `rwkv7` | 响应标签，不会在单个请求内切换已加载模型。 |
| `contents` | string[] | `[]` | 输入 prompt 数组。 |
| `max_tokens` | integer | `8192` | 每个 choice 的最大生成 token 数。 |
| `stop_tokens` | string[] | `["\nUser:"]` | 文本停止序列；与 CUDA 后端使用整数 token ID 的约定不同。 |
| `temperature` | number | `1.0` | 采样温度。 |
| `top_k` | integer | `50` | Top-K。 |
| `top_p` | number | `0.6` | Top-P。 |
| `alpha_presence` | number | `2.0` | presence repetition penalty。 |
| `alpha_frequency` | number | `0.2` | frequency repetition penalty。 |
| `alpha_decay` | number | `0.996` | repetition penalty 衰减。 |
| `stream` | boolean | `false` | 是否返回 SSE。 |
| `chunk_size` | integer | `4` | 流式输出累计多少 token 后刷新一次。 |
| `password` | string/null | `null` | 普通路由的 JSON 密码。 |

schema 还接受 `noise`、`pad_zero`、`enable_think`、`session_id`、`dialogue_idx` 和
`use_prefix_cache`；它们只在相应专用路由中生效，不应认为所有原生路由都会使用这些字段。

### 5. `POST /v1/chat/completions`

此路由虽然使用 `chat/completions` 名称，但它是原生的多 prompt batch API，输入字段是
`contents`，不是 OpenAI `messages`。

```bash
curl -N -X POST 'http://127.0.0.1:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  --data '{
    "contents":[
      "English: Hello world!\n\nChinese:",
      "English: Good morning\n\nChinese:"
    ],
    "max_tokens":1024,
    "stop_tokens":["\nUser:"],
    "temperature":0.8,
    "top_k":50,
    "top_p":0.6,
    "alpha_presence":1.0,
    "alpha_frequency":0.1,
    "alpha_decay":0.99,
    "chunk_size":8,
    "stream":true,
    "password":"rwkv7_7.2b"
  }'
```

非流式响应：

```json
{
  "id": "rwkv7-batch",
  "object": "chat.completion",
  "model": "rwkv7",
  "choices": [
    {
      "index": 0,
      "message": {"role":"assistant","content":"你好，世界！"},
      "finish_reason": "stop"
    }
  ]
}
```

流式响应是 SSE。每个事件的 `choices[].index` 对应 `contents` 的输入位置，不同 index
可以交错出现，最后以 `data: [DONE]` 结束。原生 batch SSE 不保证包含 OpenAI 风格的
`id`、`created`、`model`、最终 `finish_reason` 或 `usage`。

### 6. `POST /v2/chat/completions`

请求和响应格式与 `/v1/chat/completions` 相同，但使用 V2 batch sampler，默认参数为：

```json
{
  "top_k": 500,
  "top_p": 0.5,
  "alpha_presence": 1.0,
  "alpha_frequency": 0.1,
  "alpha_decay": 0.99
}
```

非流式响应 ID 为 `rwkv7-batch-v2`。如果显式传入采样参数，则使用请求中的值。

### 7. `POST /translate/v1/batch-translate`

该接口兼容沉浸式翻译的自定义 API：

```bash
curl -X POST 'http://127.0.0.1:8000/translate/v1/batch-translate' \
  -H 'Content-Type: application/json' \
  --data '{
    "source_lang":"en",
    "target_lang":"zh-CN",
    "text_list":["Hello world!", "Good morning"]
  }'
```

| 字段 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `source_lang` | 否 | `auto` | 源语言。常用代码会映射成英文语言名。 |
| `target_lang` | 是 | 无 | 目标语言。 |
| `text_list` | 是 | 无 | 待翻译字符串数组。 |
| `placeholders` | 否 | `null` | 当前 schema 接受，但翻译路由尚未使用。 |

接口内部使用固定参数 `max_tokens=2048`、`temperature=1.0`、`top_k=1`、
`top_p=0` 且不设置停止序列。响应：

```json
{
  "translations": [
    {"detected_source_lang":"en","text":"你好，世界！"},
    {"detected_source_lang":"en","text":"早上好"}
  ]
}
```

当 `source_lang` 为 `auto` 时，当前实现固定报告 `detected_source_lang: "en"`，并未运行
独立的语言检测器。

### 8. Stateful completion

state manager 使用三级缓存：

- L1：VRAM，默认最多 16 个 state；
- L2：RAM，当前默认最多 64 个 state；
- L3：SQLite 持久化数据库；
- 服务正常退出时会把缓存 state 持久化到数据库。

#### `POST /state/chat/completions`

```json
{
  "session_id":"session_one",
  "contents":["User: 晚饭吃什么？\n\nAssistant:"],
  "max_tokens":1024,
  "stop_tokens":["\nUser:"],
  "stream":true,
  "chunk_size":8,
  "password":"rwkv7_7.2b"
}
```

每个请求只允许一个 prompt。`session_id` 应为稳定且唯一的会话标识；首次请求创建零 state，
后续请求读取并继续更新同一 state。复用已有 state 时，服务会在没有前导空行的 prompt 前
自动补 `\n\n`。不要并发写入同一个 `session_id`。

#### `POST /multi_state/chat/completions`

该接口为一个会话保存多个可分叉的对话节点。请求必须显式提供 `session_id` 和
`dialogue_idx`：

```json
{
  "session_id":"conversation-a",
  "dialogue_idx":0,
  "contents":["User: 给我三个方案。\n\nAssistant:"],
  "stream":false,
  "max_tokens":512
}
```

根节点使用 `dialogue_idx=0`。非零节点必须已经存在，否则返回 HTTP `404`。每次成功生成后
服务分配新的 dialogue index，并将 state 保存为 `<session_id>:<new_dialogue_idx>`。非流式
响应顶层返回新的 `dialogue_idx`；流式响应会在 `[DONE]` 前插入一个元数据事件：

```json
{
  "object":"multi_state.dialogue_idx",
  "session_id":"conversation-a:1",
  "dialogue_idx":1
}
```

#### `POST /state/status`

```bash
curl -X POST 'http://127.0.0.1:8000/state/status' \
  -H 'Content-Type: application/json' \
  --data '{"password":"rwkv7_7.2b"}'
```

响应包含 `total_sessions`、`l1_cache_count`、`l2_cache_count`、`database_count` 和
`sessions`；每个 session 条目包含缓存级别及更新时间。

#### `POST /state/delete`

```json
{
  "session_id":"conversation-a",
  "delete_prefix":true,
  "password":"rwkv7_7.2b"
}
```

`delete_prefix=true` 时，除精确 ID 外还会删除所有 `<session_id>:*` 分支。成功返回 HTTP
`200`；未找到且未要求前缀删除时返回 `404`。

### 9. `POST /openai/v1/chat/completions`

这是面向 OpenAI 客户端的单 choice 文本聊天接口。它接受 `messages`、顶层 `system`，也
接受旧式 `contents` 的第一项。`system` 和 `developer` message 会合并为系统提示；当前只
处理 `user` 与 `assistant` 文本消息，不支持图片、音频、tool calls 或 structured output。

```bash
curl -N -X POST 'http://127.0.0.1:8000/openai/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer rwkv7_7.2b' \
  --data '{
    "model":"rwkv7",
    "messages":[
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"请简单介绍 RWKV。"}
    ],
    "max_tokens":512,
    "temperature":1.0,
    "top_k":20,
    "top_p":0.6,
    "enable_think":false,
    "stream":true
  }'
```

该路由的默认值与原生 batch 路由略有不同：`max_tokens=4096`、`top_k=20`、
`alpha_presence=1`、`alpha_frequency=0.1`；流式 `chunk_size` 默认 `1`，非流式默认
`16`。`enable_think=true` 时 prompt 以 `Assistant: <think` 结尾，否则使用快速思考模板。

`use_prefix_cache` 默认 `true`，用于复用匹配的 prompt 前缀 state；它与
`/state/chat/completions` 的显式 `session_id` 会话模式不同。

非流式响应包含真实随机 `chatcmpl-*` ID、`created`、单个 choice、推理层返回的
`finish_reason` 和：

```json
{
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 20,
    "total_tokens": 62
  }
}
```

流式响应先发送 `delta.role=assistant`，再发送内容增量与最终 `finish_reason`，最后发送
`data: [DONE]`。当前流式接口不解析 `stream_options.include_usage`，因此不返回最终 usage
chunk；需要流式计费统计的客户端应暂时在本地计数。

模型列表接口为：

```bash
curl 'http://127.0.0.1:8000/openai/v1/models' \
  -H 'Authorization: Bearer rwkv7_7.2b'
```

`GET /openai/v1/models` 执行 OpenAI/Bearer 鉴权；`GET /v1/models` 返回相同的核心列表结构，
但当前不执行密码校验。

### 10. `POST /big_batch/completions`

该接口面向尽可能大的并行 batch，只使用 temperature 采样，不使用 `top_k`、`top_p` 和
repetition penalty 参数。它始终返回 SSE，即使请求中的 `stream` 为 `false`。

```json
{
  "contents":[
    "English: Hello\n\nChinese:",
    "English: Good morning\n\nChinese:"
  ],
  "max_tokens":1024,
  "stop_tokens":["\nUser:"],
  "temperature":1.0,
  "chunk_size":8,
  "password":"rwkv7_7.2b"
}
```

每个 SSE choice 都带原输入的 `index`。实际可用 batch size 由显存和动态 prefill batch
上限决定。

### 11. `POST /FIM/v1/batch-FIM`

该接口用于 RWKV7 G1c 系列模型的 Fill-In-the-Middle：

```json
{
  "prefix":[
    "The rain had stopped, but the street still glistened"
  ],
  "suffix":[
    "and then the door opened."
  ],
  "max_tokens":1024,
  "temperature":0.8,
  "top_k":50,
  "top_p":0.6,
  "stream":true,
  "chunk_size":8,
  "password":"rwkv7_7.2b"
}
```

服务按位置配对 `prefix` 与 `suffix`，内部格式为：

```text
✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}
```

两数组长度应一致；当前实现使用 `zip`，长度不一致时多出的元素会被忽略。FIM 路由当前
固定传入空停止序列，因而请求中的 `stop_tokens` 不生效。非流式响应的 `object` 为
`FIM.completion`，流式响应使用通用 batch SSE 格式。

### 12. SSE 与错误处理

原生 batch SSE 的基本事件如下：

```text
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"..."}}]}

data: [DONE]
```

注意：

- 原生 `/v1`、`/v2`、state、FIM 和 big-batch SSE 是项目自有的轻量格式，不等同于完整 OpenAI SSE schema。
- `/openai/v1/chat/completions` 才提供 OpenAI 风格的 ID、role chunk 和结束 reason。
- 客户端断开时服务会取消排队或正在运行的生成；非流式连接断开可能返回内部状态 `499`。
- 流式请求在 prefill queue 阶段发生 batch overflow 时，错误可能作为 SSE `data:` 事件返回。
- FastAPI/Pydantic 字段校验错误通常返回 HTTP `422`；显式业务校验常返回 `400`、`401`、`404` 或 `500`。



# 2. rwkv_lightning_cuda


Repository:

```
https://github.com/Alic-Li/rwkv_lightning_cuda
```


---

# Introduction


`rwkv_lightning_cuda` is a native C++ CUDA implementation of RWKV inference.


Features:

- Pure C++ runtime
- Custom CUDA kernels
- Reduced framework overhead
- Optimized memory management
- High throughput inference


Recommended scenarios:


- Production deployment
- Large scale inference
- Low latency service
- Embedded inference engine


## API Documentation

本节描述当前 `rwkv_lightning_cuda` C++/CUDA 服务实际提供的 HTTP API。接口风格参考
OpenAI Chat Completions，但除 `/v1/chat/completions` 的核心文本字段外，不应假定它与
OpenAI API 完全等价。

### 1. 服务启动与基础约定

```bash
./rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000 \
  --chunk-size 128 \
  --state-db-path rwkv_sessions.db \
  --password your-password
```

| 启动参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--model-path` | 是 | 无 | RWKV `.pth` 模型路径。 |
| `--vocab-path` | 是 | 无 | RWKV tokenizer 词表路径。 |
| `--host` | 否 | `127.0.0.1` | 监听地址；需要远程访问时可显式设置为 `0.0.0.0`。 |
| `--port` | 否 | `8000` | HTTP 端口。 |
| `--chunk-size` | 否 | `128` | prompt prefill 的 token 分块大小，必须为正整数。该参数不等于请求 JSON 中的 `chunk_size`。 |
| `--state-db-path` | 否 | `rwkv_sessions.db` | 会话状态 SQLite 数据库。 |
| `--password` | 否 | 禁用鉴权 | 启用 Bearer token 或 JSON `password` 鉴权。 |
| `--wkv32` | 否 | 关闭 | 使用 FP32 WKV state 和 FP16 IO。 |

所有 JSON 请求都必须带：

```http
Content-Type: application/json
```

否则 Drogon 不会把请求体识别为 JSON，并返回：

```json
{"error":"Invalid JSON"}
```

服务支持 CORS，所有已注册路由均支持 `OPTIONS` 预检。流式响应使用
`text/event-stream; charset=utf-8`，建议客户端使用 `curl -N` 或关闭代理缓冲。

### 2. 鉴权

如果服务以 `--password` 启动，可使用以下任一方式：

```http
Authorization: Bearer your-password
```

或：

```json
{"password":"your-password"}
```

鉴权失败返回 HTTP `401`：

```json
{"error":"Unauthorized: invalid or missing password"}
```

当前 `/v1/server/status` 和 `/translate/v1/batch-translate` 不执行密码校验；前者还会返回
模型文件路径。若服务暴露到非可信网络，应由反向代理、Cloudflare Access 或防火墙保护这些路由。

### 3. 接口总览

| Method | Path | 用途 | 支持流式 |
|---|---|---|---:|
| `GET` | `/v1/models` | 查询已加载模型 | 否 |
| `POST` | `/v1/tokens/count` | 计算 token 数 | 否 |
| `POST` | `/v1/chat/completions` | OpenAI 风格单请求聊天补全 | 是 |
| `POST` | `/v1/batch/completions` | 多 prompt 批量补全 | 是 |
| `POST` | `/translate/v1/batch-translate` | 批量翻译 | 否 |
| `POST` | `/state/chat/completions` | 带 RWKV state cache 的单会话补全 | 是 |
| `POST` | `/state/status` | 查询会话缓存 | 否 |
| `POST` | `/state/delete` | 删除会话状态 | 否 |
| `GET` | `/v1/server/status` | 服务、推理请求和性能状态 | 否 |
| `POST` | `/v1/server/stop` | 停止一个或全部活动生成 | 否 |
| `POST` | `/v1/server/pause` | 暂停一个活动生成 | 否 |
| `POST` | `/v1/server/resume` | 恢复可续推的聊天流 | 固定为 SSE |

### 4. 通用生成参数

以下参数适用于 `/v1/chat/completions`、`/v1/batch/completions` 和
`/state/chat/completions`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 已加载模型名 | 作为响应中的模型标签；当前不会动态切换模型。 |
| `max_tokens` | integer | `8192` | 最多生成的 token 数。 |
| `temperature` | number | `1.0` | 采样温度。 |
| `top_k` | integer | `20` | Top-K 采样参数。 |
| `top_p` | number | `0.3` | Top-P 采样参数。 |
| `alpha_presence` | number | `2.0` | presence repetition penalty。 |
| `alpha_frequency` | number | `0.2` | frequency repetition penalty。 |
| `alpha_decay` | number | `0.996` | repetition penalty 衰减。 |
| `stop_tokens` | integer[] | `[0,261,24281]` | 停止 token ID；不是停止字符串数组。命中的 token 不出现在输出中。 |
| `stream` | boolean | `false` | 是否使用 SSE。 |
| `chunk_size` | integer | 因路由而异 | 流式输出累计多少 token 后发送一次；小于 `1` 时按 `1` 处理。 |
| `force_reasoning` | boolean | `false` | 启用内部 reasoning token mask。聊天接口使用 think 字段时会覆盖此值。 |
| `password` | string | 无 | 可选的 JSON 鉴权凭据。 |

`chunk_size` 是输出刷新粒度，不是 prefill 分块。prefill 分块由进程启动参数
`--chunk-size` 全局控制，默认 `128`。

### 5. Thinking 模式

`/v1/chat/completions` 支持：

| `think_type` | Prompt 尾部行为 | 强制 reasoning mask |
|---|---|---:|
| `fast` | `Assistant: <think></think` | 否 |
| `free` | `Assistant: <think` | 是 |
| `preferChinese` / `prefer_chinese` | `Assistant: <think>嗯` | 是 |
| `en` | 在最后用户消息添加 `(think)` | 是 |
| `enShort` / `en_short` | 添加 `(think a bit)` | 是 |
| `enLong` / `en_long` | 添加 `(think a lot)` | 是 |

兼容布尔字段 `think` 和 `enable_think`：`true` 等价于 `free`，`false` 等价于
`fast`。当有效的 `think_type` 与布尔字段同时出现时，以 `think_type` 为准。

### 6. `POST /v1/chat/completions`

接受 OpenAI 风格的 `messages`。`content` 可以是字符串，也可以是文本 part 数组：

```json
{
  "model": "api-test",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请简单介绍 RWKV。"}
  ],
  "stream": false,
  "think_type": "fast",
  "max_tokens": 512,
  "temperature": 1.0,
  "top_k": 5,
  "top_p": 0.3,
  "alpha_presence": 2.0,
  "alpha_frequency": 0.2,
  "alpha_decay": 0.99,
  "chunk_size": 1
}
```

也可传顶层 `system` 字段。为了兼容旧客户端，接口还接受 `contents` 数组，但在聊天接口中
只使用第一项，并把它作为额外的 User 消息。当前仅处理文本，不支持图片、音频、tool call、
structured output 或 `n > 1`。

非流式响应：

```json
{
  "id": "chatcmpl-rwkv-fast",
  "object": "chat.completion",
  "created": 1785945600,
  "model": "api-test",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 20,
    "total_tokens": 62
  }
}
```

流式请求示例：

```bash
curl -N 'http://127.0.0.1:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-password' \
  --data '{
    "model":"api-test",
    "messages":[{"role":"user","content":"你好"}],
    "stream":true,
    "max_tokens":128,
    "chunk_size":1
  }'
```

数据事件形如：

```text
data: {"id":"req-...","object":"chat.completion.chunk","created":1785945600,"model":"api-test","choices":[{"index":0,"delta":{"content":"你"}}]}

data: {"id":"req-...","object":"chat.completion.chunk","created":1785945600,"model":"api-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 7. `POST /v1/batch/completions`

同一次请求并行处理多个 prompt：

```json
{
  "model": "api-test",
  "contents": [
    "English: Hello\n\nChinese:",
    "English: Good morning\n\nChinese:"
  ],
  "stream": true,
  "max_tokens": 256,
  "chunk_size": 8,
  "metrics": true
}
```

`contents` 必须是非空数组。SSE 中每个 choice 使用原始输入位置作为 `index`；不同 index
的事件可能交错到达。流式 `chunk_size` 默认 `8`。非流式响应为：

```json
{
  "id": "rwkv7-fast-batch",
  "object": "chat.completion",
  "model": "api-test",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "你好"},
      "finish_reason": "stop"
    }
  ]
}
```

`metrics`、`report_metrics` 或 `include_metrics` 任一为 `true` 时，服务会记录该请求的
prefill/decode 指标，供 `/v1/server/status` 查询；这些字段不会把 metrics 或 usage 直接
加入补全响应。

### 8. `POST /translate/v1/batch-translate`

```json
{
  "source_lang": "English",
  "target_lang": "Chinese",
  "text_list": ["Hello", "Good morning"]
}
```

`target_lang` 和非空 `text_list` 为必需字段，`source_lang` 默认 `auto`。该路由使用固定生成参数：
`max_tokens=2048`、`temperature=1.0`、`top_k=1`、`top_p=0.0`、无 repetition penalty，
停止 token 为 `0`。

```json
{
  "translations": [
    {"detected_source_lang":"English","text":"你好"},
    {"detected_source_lang":"English","text":"早上好"}
  ]
}
```

当 `source_lang` 为 `auto` 时，当前 `detected_source_lang` 仍返回字符串 `auto`，服务不会
额外运行语言检测器。

### 9. Stateful API

#### `POST /state/chat/completions`

```json
{
  "session_id": "user-42",
  "contents": ["User: 请记住数字 17。\n\nAssistant:"],
  "stream": true,
  "max_tokens": 128,
  "chunk_size": 8
}
```

约束：

- `session_id` 必须非空。
- `contents` 必须恰好包含一个 prompt；stateful 推理只支持 batch size `1`。
- 首次访问创建空 state；之后从 L1 VRAM、L2 RAM 或 SQLite 中恢复，并在生成结束后写回。
- 不要并发写入同一个 `session_id`。服务允许并发请求，但同一会话的并发写回存在后完成者覆盖先完成者状态的风险。

非流式响应对象与 batch completion 类似，`id` 为 `rwkv7-fast-state`。流式格式与普通聊天
SSE 相同，流式 `chunk_size` 默认 `8`。

#### `POST /state/status`

请求体可为空对象：

```json
{}
```

响应包含三级缓存数量及会话列表：

```json
{
  "status": "success",
  "l1_cache_count": 1,
  "l2_cache_count": 0,
  "database_count": 3,
  "total_sessions": 4,
  "sessions": [
    {"session_id":"user-42","cache_level":"L1 (VRAM)"}
  ]
}
```

SQLite 中的条目还会包含 `timestamp`。

#### `POST /state/delete`

```json
{"session_id":"user-42"}
```

成功返回 HTTP `200` 和 `status: "success"`；不存在时返回 HTTP `404` 和
`status: "not_found"`。

### 10. Token 与模型接口

#### `POST /v1/tokens/count`

支持三种输入，按 `text`、`messages`、`contents` 的优先级选择：

```json
{"text":"hello RWKV"}
```

```json
{"messages":[{"role":"user","content":"hello RWKV"}]}
```

```json
{"contents":["prompt one", "prompt two"]}
```

`messages` 会先套用聊天 prompt 模板再计数；`contents` 会连接为一个字符串后计数。响应：

```json
{"tokens":5}
```

#### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "loaded-model-name",
      "object": "model",
      "created": 1785945600,
      "owned_by": "rwkv_lighting_cuda"
    }
  ]
}
```

### 11. 服务控制与并发请求

服务允许多个 generation 同时运行，不再因已有请求而返回
`{"error":"Another generation is active"}`。并发数量没有软件硬上限，但实际受 GPU 显存、
CUDA workspace、模型大小和请求 batch size 限制。

#### `GET /v1/server/status`

关键响应字段：

| 字段 | 说明 |
|---|---|
| `status` | 当前为 `running`。 |
| `api_version` / `engine_version` | API 与引擎版本。 |
| `model` | 已加载模型 ID、名称和文件路径。 |
| `capabilities` | stream、session cache、pause/resume、chunk prefill、concurrent generation 等能力。 |
| `prefill_chunk_size` | 启动时设置的 prefill 分块大小。 |
| `active_request` | 最近启动且仍活动的请求，保留用于兼容旧客户端；无请求时为 `null`。 |
| `active_requests` | 所有活动请求数组。 |
| `last_request` | 最近完成的请求快照。 |
| `paused_requests` | 可恢复的暂停请求数组。 |

活动请求包含 `id`、`endpoint`、`model`、`created`、`prompt_tokens`、
`prefilled_tokens`、`generated_tokens`、`max_tokens`、停止/暂停标记、prefill/decode
速度和进度；stateful 请求还包含 `state_key`。

#### `POST /v1/server/stop`

```json
{"request_id":"req-1785945600-1"}
```

指定 `request_id` 时只停止该请求；省略时向所有活动请求发出停止信号。停止是协作式的，
推理线程会在下一个生成迭代检查信号。

#### `POST /v1/server/pause`

```json
{"request_id":"req-1785945600-1"}
```

指定 ID 时暂停对应请求；省略时选择最近启动的活动请求。当前只有流式
`/v1/chat/completions` 会保存完整的 state 和 logits 供恢复。对 batch/stateful 流发出
pause 只会令当前生成提前结束，不会产生可供 `/v1/server/resume` 使用的记录。

#### `POST /v1/server/resume`

```json
{
  "request_id": "req-1785945600-1",
  "max_tokens": 256,
  "chunk_size": 1
}
```

也接受 `session_id` 作为 `request_id` 的兼容别名。该接口始终返回 SSE。省略
`max_tokens` 时继续生成原请求剩余的 token 数；显式传入时使用新的上限。暂停记录被取出后
即从 paused registry 删除。找不到记录返回 HTTP `404`。

### 12. SSE 结束语义、`finish_reason` 与 usage

#### 当前实现

所有流式接口使用同一个通用 SSE 收尾函数。推理任务结束后，它无条件发送：

```json
{
  "choices": [
    {"index":0,"delta":{},"finish_reason":"stop"}
  ]
}
```

然后发送 `data: [DONE]`。因此当前无法从 `finish_reason` 区分以下情况：

- 模型生成 EOS；
- 命中 `stop_tokens`；
- 达到 `max_tokens`；
- `/v1/server/stop` 或 pause 导致的管理性中止；
- 客户端/发送回调提前终止。

直接原因是推理层虽然对部分路径记录了 `stopped`、`stop_token` 和生成 token 数，但通用
`start_streaming_task` 没有接收这些结束状态，`send_finish_chunk` 的默认参数又固定为
`"stop"`。stateful 流路径目前甚至不返回结束统计。因此这是当前实现的兼容性限制，不代表
上述情况在语义上都属于自然停止。

当前流式响应也不会返回 `usage`，即使请求传入：

```json
{"stream_options":{"include_usage":true}}
```

原因是服务尚未解析 `stream_options.include_usage`，SSE 生产器也没有把 prompt/completion
token 统计传递给最终事件。普通 `/v1/chat/completions` 的非流式响应会返回 usage；batch、
stateful 和所有流式响应当前均不返回标准 usage 对象。

#### OpenAI Chat Completions 兼容目标

按照 OpenAI Chat Completions API 的定义：自然停止或命中指定停止条件应返回
`finish_reason: "stop"`，达到请求的最大生成 token 数应返回
`finish_reason: "length"`。工具调用和内容过滤分别使用 `tool_calls` 与
`content_filter`，但本服务当前不实现这两类能力。参见
[OpenAI Chat Completions API Reference](https://developers.openai.com/api/reference/resources/chat)。

目标映射应为：

| 实际结束原因 | 标准/建议 `finish_reason` |
|---|---|
| EOS 或命中 `stop_tokens` | `stop` |
| 达到 `max_tokens` | `length` |
| tool call | `tool_calls`，当前不支持 |
| content filter | `content_filter`，当前不支持 |
| 管理端 stop、pause、客户端断开 | OpenAI 没有完全对应的标准值；应另加服务自有状态字段，不能仅靠 `finish_reason` 混淆为正常 EOS。 |

当请求设置 `stream_options: {"include_usage": true}` 时，兼容实现应在
`data: [DONE]` 之前额外发送一个 `choices: []` 的 usage chunk，其中包含整个请求的
`prompt_tokens`、`completion_tokens` 和 `total_tokens`；此前的流式 chunk 应带
`usage: null`。若连接中途断开，客户端可能收不到最终 usage chunk。这也是官方 API
reference 明确说明的边界。

在上述兼容项实现之前，客户端应自行累计输出，且不要依赖当前 SSE 的
`finish_reason` 判断是否触达 `max_tokens`。

### 13. 错误与状态码

| HTTP 状态 | 常见原因 |
|---:|---|
| `200` | 请求成功；SSE 推理中的运行时异常也可能作为 `data: {"error":"..."}` 事件返回，并保持 HTTP 200。 |
| `204` | `OPTIONS` 预检。 |
| `400` | JSON 无效、缺少字段、prompt 列表为空。 |
| `401` | 密码无效或缺失。 |
| `404` | state、session 或 paused request 不存在。 |
| `500` | 暂停请求状态不完整等服务端错误。 |

标准 JSON 错误体为：

```json
{"error":"error message"}
```
