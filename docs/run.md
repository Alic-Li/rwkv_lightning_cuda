# Running the server

[English](run.md) | [简体中文](run.zh-CN.md) | [Back to README](../README.md)

## Run server

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

The server binds to `127.0.0.1` by default. Use `--host 0.0.0.0` only when
you intentionally want to listen on all IPv4 interfaces.

## Run on Windows

```bash
cd build\bundle\rwkv_lighting_cuda;
set "SCRIPT_DIR=%~dp0\";
.\build/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path /path/to/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000
```

## Dynamic model loading

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

## Standalone state tuning (CUDA)

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
existing inference backend. See `../src/state_tuning/README.md` for implementation
details.
