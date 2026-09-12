## CI / Release

GitHub Actions builds Linux and Windows CUDA packages on PRs and `main`. Push a
`v*` tag to generate a draft Release with binaries and SHA-256 checksums. See
[the release guide](docs/releasing.md) for the workflow and package contents.

## Project layout

The native code is organized by responsibility:

- `include/rwkv/`: public headers grouped into `common`, `io`, `runtime`, `inference`, and `server` APIs.
- `src/backend/`: GPU model backend integration.
- `src/inference/`: tokenization, sampling, and generation orchestration.
- `src/io/`: PTH archive and tensor readers.
- `src/server/`: HTTP API, model routing, admission control, and state storage.
- `src/app/`: executable entry points.
- `assets/`: runtime data files such as the tokenizer vocabulary.
- `cmake/`: dependency, compiler-option, and packaging modules.

The root `CMakeLists.txt` only selects the GPU backend and composes these modules. Target definitions live next to their corresponding source trees in `src/CMakeLists.txt`, `tools/CMakeLists.txt`, and `test/CMakeLists.txt`.

## Build

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release
cmake --build ./build -j --config Release --target bundle_all
```

`bundle_all` assembles every available runtime target below `build/bundle/`. Server and state-tuning bundles include `rwkv_vocab_v20230424.txt`; shared runtime libraries are placed in each bundle's `lib/` directory. Individual `bundle_<target>` targets remain available.

### CUDA W8A16 / W4A16 quantization

The build produces `build/bundle/rwkv_quantize/rwkv_quantize`, a standalone converter for BF16 RWKV
checkpoints. It writes a streaming `.rwkvq` file containing either per-output-channel
INT8 weights or grouped INT4 weights with FP16 scales; embeddings, layer norms,
LoRA factors, and other non-linear tensors remain BF16.

```bash
./build/bundle/rwkv_quantize/rwkv_quantize /path/to/model.pth /path/to/model.w8a16.rwkvq

# W4 with one FP16 scale for each 128-weight group (recommended default)
./build/bundle/rwkv_quantize/rwkv_quantize \
  --format w4a16 --group-size 128 \
  /path/to/model.pth /path/to/model.w4a16.rwkvq
```

The CUDA inference backend detects the tensor dtype in `.rwkvq` automatically,
keeps packed INT4 or INT8 weights on device, and dispatches W4A16 or W8A16 for
attention projections, FFN projections, and the output head. W4 also supports
`--group-size 32` when higher fidelity is worth the extra scales. Existing
two-positional-argument commands still export W8A16. HIP builds continue to use
the BF16/PTH path.

Windows
```bash
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\"
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120" -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DCMAKE_CXX_FLAGS="/Zc:preprocessor" -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"

cmake --build ./build --config Release -j --target bundle_rwkv_quantize bundle_rwkv_lighting_cuda
```

AMD ROCm
```bash
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
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

### Standalone state tuning (CUDA)

With `RWKV7_STATE_TUNING=ON` (the default for CUDA builds), the standalone
`rwkv_state_tune` binary trains only `blocks.N.att.time_state` from JSONL rows
of the form `{"text":"..."}`:

```bash
./build/rwkv_state_tune \
  --model /path/to/model.pth \
  --data /path/to/train.jsonl \
  --output ./state_output \
  --ctx 512 \
  --chunk 128 \
  --epochs 1 \
  --max-steps 10000 \
  --lr 0.0005 \
  --lr-final 0.0005 \
  --warmup-steps 10 \
  --save-every 100 \
  --batch-size 8
```

This first correctness-oriented version uses the existing BF16 PTH loader and
its FP16 runtime weights and rejects INT8 training. Samples are truncated at
`--ctx`; `--chunk` controls checkpoint/recompute length with reverse state
gradient propagation. `--batch-size N` accumulates N variable-length samples
per optimizer update. Checkpoints contain only state tensors and can be uploaded to the
existing inference backend. See `src/state_tuning/README.md` for implementation
details.

Run server

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000 \
  --chunk-size 128 \
  --chunk-load
```

`--chunk-size` controls prompt prefill chunking and defaults to `128` when omitted.
`--state-db-path` defaults to `rwkv_sessions.db` in the current working directory.
The W8A16 tuning cache is per-service-instance local state: unless `--tune-cache`
is provided, it is stored alongside the state database (also in the current working
directory by default). Keep the cache and state database together when deploying a
service, or set an explicit cache path.
`--chunk-load` avoids reading the complete `.pth` file or a complete large tensor into
host memory before the CUDA upload. It uses a persistent model-file stream and two
reusable 32 MiB pinned buffers to overlap disk reads, CUDA copies, and preprocessing.
Four complete transformer layers are uploaded and finalized as one batch, reducing
load-time synchronization. On Linux, consumed file-cache pages are marked reclaimable
after each read. Omit the flag to keep the original whole-file loading behavior.
Generation requests enter a FIFO admission queue. The server dynamically refreshes the
available prefill batch-size limit from free VRAM and admits requests when capacity is
available. `/v1/server/status` reports `prefill_queue` and all `active_requests` while
retaining `active_request` for compatibility.

### Dynamic model loading

By default, `--model-path` remains the path to one `.pth` or `.rwkvq` file and the original
single-model startup behavior is unchanged. To load models on demand, make it a
directory and add `--enable-dynamic-loading`:

```bash
./build/rwkv_lighting_cuda \
  --model-path /path/to/models \
  --enable-dynamic-loading \
  --chunk-load \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt
```

The directory's top-level `.pth` and `.rwkvq` files are exposed by `GET /v1/models`; their
file names without the extension are the model IDs. The response identifies the current
`loaded` model and every `available` model. Load or switch models explicitly:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/model/load" \
  -H "Content-Type: application/json" \
  --data '{"model":"rwkv7-g1i-7.2b-20260805-ctx16384"}'
```

Inference requests always use the already loaded model: their `model` field is
kept only for OpenAI compatibility and does not trigger a load or switch.
Concurrent inference shares that loaded model. Different load requests are FIFO
queued; a switch waits for active inference to finish, releases the old model
from VRAM, and then loads the selected model.

If use windows 
```bash
cd build\bundle\rwkv_lighting_cuda;
set "SCRIPT_DIR=%~dp0\";
.\build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000
```

The server binds to `127.0.0.1` by default. Use `--host 0.0.0.0` only when
you intentionally want to listen on all IPv4 interfaces.

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
`/state/chat/completions`. For a batch request, the uploaded state is copied
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

### Batch completions

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

### Batch translation

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
