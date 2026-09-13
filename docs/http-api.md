# HTTP API examples

[English](http-api.md) | [简体中文](http-api.zh-CN.md) | [Back to README](../README.md)

The complete endpoint reference (in Chinese) lives in
[rwkv_lightning_api_doc.md](../rwkv_lightning_api_doc.md). The examples below
assume the server is running on port `8000`.
If the server was started with `--password`, pass either a Bearer token header or the `password` field in JSON:

```bash
AUTH_HEADER=(-H "Authorization: Bearer rwkv7_7.2b")
```

Run the serial smoke test for all endpoints:

```bash
../test/api_endpoints_test.sh

# Custom host, port, or password:
BASE_URL=http://127.0.0.1:8000 PASSWORD=rwkv7_7.2b ../test/api_endpoints_test.sh
```

## Service status

Check whether the backend is running, which model is loaded, supported capabilities, active request, and paused requests.

```bash
curl -sS "http://127.0.0.1:8000/v1/server/status"
```

## Model list

OpenAI-compatible model list endpoint.

```bash
curl -sS "http://127.0.0.1:8000/v1/models"
```

## Token count

Count tokens for raw text.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"text":"hello RWKV"}'
```

Count tokens for chat messages after applying the backend chat prompt template.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/tokens/count" \
  -H "Content-Type: application/json" \
  --data '{"messages":[{"role":"user","content":"hello"}]}'
```

## Chat completions

OpenAI-style chat endpoint. Use `stream:false` for one JSON response.

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

`think_type` controls the assistant think prefix for chat-message prompts:
`none`, `fast`, `free`, `preferChinese`, `en`, `enShort`/`en_short`, and
`enLong`/`en_long`. `fast` uses a short closed think prefix and does not force
reasoning. The other modes force reasoning by masking tokens `111` and `754`
on the second and third generated tokens. If `think_type` is omitted,
`enable_think:true` or `think:true` maps to `free`; otherwise the default is
`fast`.

Upload a serialized RWKV state first, then pass the returned filename as `state_id`
to a chat request. Uploaded files are validated as PyTorch state archives and
kept in a process-local temporary directory. They are removed explicitly with
the delete endpoint or automatically when the server exits. The upload limit
is 512 MiB. State tensors stored as either PyTorch `bfloat16` or `float32` are
supported; they are converted to the configured WKV runtime precision while
loading.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/state/upload" \
  -F "file=@./rwkv-state-agentic.pth"
```

The response uses the uploaded file's basename as its ID, for example
`{"object":"rwkv.state","state_id":"rwkv-state-agentic.pth"}`. Uploading a
file with a name that is already present returns an `already exists` error. Use
that value in a generation request:

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

Omit `state_id` to use the normal zero-initialized state. All inference
endpoints accept the same string field: `/v1/chat/completions`,
`/v1/batch/completions`, `/translate/v1/batch-translate`, and
`/state/chat/completions`. Every endpoint above also accepts `state_id`
through an `X-RWKV-State-Id` request header instead of the JSON body; conflicting
values across channels return HTTP 400, while repeating the same value is
accepted (`/v1/state/delete` additionally accepts `?state_id=...` in the query
string under the same rule). For a batch
request, the uploaded state is copied
to every batch slot before its prompt is evaluated. On the stateful endpoint,
an explicit uploaded state takes precedence over the cached `session_id`
state for that request, and the resulting state is cached back into the
session afterward. Chat requests carrying `state_id` default to the classic
`User`/`Assistant` prompt without a think prefix, matching typical state-tuning
data. An explicit `think_type` overrides this behavior; `think_type:"none"`
also disables the prefix explicitly.

List or delete uploaded states:

```bash
curl -sS "http://127.0.0.1:8000/v1/state/list"

curl -sS -X DELETE "http://127.0.0.1:8000/v1/state/delete" \
  -H "Content-Type: application/json" \
  --data '{"state_id":"rwkv-state-agentic.pth"}'
```

When `--password` is enabled, use an `Authorization: Bearer ...` header for the
multipart upload request.

Use `stream:true` for SSE chunks. The stream ends with `data: [DONE]`.

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

Generate independent continuations for multiple prompts. Each streamed chunk uses `choices[].index` to identify the slot.

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

## Batch translation

Compatibility endpoint for batch translation-style prompts.

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

## Stateful completions

Use `session_id` to reuse and update a saved RWKV state. This endpoint accepts exactly one prompt in `contents`. `session_id` may also be passed through an `X-RWKV-Session-Id` request header (also honored by `/state/delete`); conflicting values between the body and the header return HTTP 400, while repeating the same value is accepted.

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

List cached sessions:

```bash
curl -sS -X POST "http://127.0.0.1:8000/state/status" \
  -H "Content-Type: application/json" \
  --data '{}'
```

Delete a cached session:

```bash
curl -sS -X POST "http://127.0.0.1:8000/state/delete" \
  -H "Content-Type: application/json" \
  --data "{\"session_id\":\"api-test\"}"
```

## Stop, pause, and resume

Stop the active generation. If no request is active, the response still returns `ok:true` with `stopped:false`.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/server/stop" \
  -H "Content-Type: application/json" \
  --data '{}'
```

Pause the active generation and save the current state. The response contains `request_id` when a request was paused.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/server/pause" \
  -H "Content-Type: application/json" \
  --data '{}'
```

Resume a paused generation by `request_id`. The response is an SSE stream.

```bash
curl -sS -N -X POST "http://127.0.0.1:8000/v1/server/resume" \
  -H "Content-Type: application/json" \
  --data '{
    "request_id":"req-xxxxxxxx",
    "stream":true
  }'
```

## CORS preflight

The server accepts `OPTIONS` on API routes for browser clients. Depending on the HTTP framework path, a successful preflight may return `200` or `204`.

```bash
curl -sS -i -X OPTIONS "http://127.0.0.1:8000/v1/chat/completions"
```
