## Build

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release --target bundle_rwkv_lighting_cuda
```

Windows
```bash
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\"
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120" -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DCMAKE_CXX_FLAGS="/Zc:preprocessor" -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"

cmake --build ./build --config Release -j --target bundle_rwkv_lighting_cuda
```

Compile Go Web Frontend

```bash
## Linux
CGO_ENABLED=0 go build -ldflags="-s -w" -o rwkv_launcher main.go
## Windows
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o .\rwkv_launcher.exe .\main.go
```
## Run

Run benchmark

```bash
./build/benchmark \
  --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --model-forward \
  --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' \
  --graph-bench \
  --warmup 3 \
  --iters 10
```

Run server

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --port 8000
```

If use windows 
```bash
cd build\bundle\rwkv_lighting_cuda;
set "SCRIPT_DIR=%~dp0\";
.\build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --port 8000
```

## HTTP API examples

The examples below assume the server is running on port `8000`.
If the server was started with `--password`, pass either a Bearer token header or the `password` field in JSON:

```bash
AUTH_HEADER=(-H "Authorization: Bearer rwkv7_7.2b")
```

Run the serial smoke test for all endpoints:

```bash
./test/api_endpoints_test.sh

# Custom host, port, or password:
BASE_URL=http://127.0.0.1:8000 PASSWORD=rwkv7_7.2b ./test/api_endpoints_test.sh
```

### Service status

Check whether the backend is running, which model is loaded, supported capabilities, active request, and paused requests.

```bash
curl -sS "http://127.0.0.1:8000/v1/server/status"
```

### Model list

OpenAI-compatible model list endpoint.

```bash
curl -sS "http://127.0.0.1:8000/v1/models"
```

### Token count

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

### Chat completions

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
`fast`, `free`, `preferChinese`, `en`, `enShort`/`en_short`, and
`enLong`/`en_long`. `fast` uses a short closed think prefix and does not force
reasoning. The other modes force reasoning by masking tokens `111` and `754`
on the second and third generated tokens. If `think_type` is omitted,
`enable_think:true` or `think:true` maps to `free`; otherwise the default is
`fast`.

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

### Batch completions

Generate independent continuations for multiple prompts. Each streamed chunk uses `choices[].index` to identify the slot.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/batch/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "contents":["English: Hello\n\nChinese:","English: Good morning\n\nChinese:"],
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

### Batch translation

Compatibility endpoint for batch translation-style prompts.

```bash
curl -sS -X POST "http://127.0.0.1:8000/translate/v1/batch-translate" \
  -H "Content-Type: application/json" \
  --data '{
    "source_lang":"English",
    "target_lang":"Chinese",
    "text_list":["Hello","Good morning"]
  }'
```

### Stateful completions

Use `session_id` to reuse and update a saved RWKV state. This endpoint accepts exactly one prompt in `contents`.

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

### Stop, pause, and resume

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

### CORS preflight

The server accepts `OPTIONS` on API routes for browser clients. Depending on the HTTP framework path, a successful preflight may return `200` or `204`.

```bash
curl -sS -i -X OPTIONS "http://127.0.0.1:8000/v1/chat/completions"
```
