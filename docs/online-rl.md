# Persistent BF16 online RL servers

English | [简体中文 / full API examples](online-rl.zh-CN.md)

`rwkv_miss_tune_server` trains MiSS D parameters with a fixed initial state.
`rwkv_state_tune_server` trains the initial time_state with a fixed adapter.
Both accept a cold-start adapter and state, retain Adam in memory, and share one
BF16 backbone between actor snapshots and the learner within each process.
Running both processes loads two backbones. Choose one for each training run.
The original inference server and training CLIs remain separate.

## Build or use a release

From the source checkout, with the project's CUDA and Drogon dependencies installed:

```bash
cmake -S . -B build -DRWKV7_STATE_TUNING=ON
cmake --build build --parallel 2 --target rwkv_miss_tune_server rwkv_state_tune_server
```

GitHub Actions builds and packages both servers on Linux and Windows with CUDA
12.9 and 13.2. Extract the entire release directory, including runtime libraries,
`rwkv_vocab_v20230424.txt`, `tools/online_rl.py`, and `docs/`. The examples below
run from that directory. For a source build, use `./build/<binary>` and
`--vocab ./assets/rwkv_vocab_v20230424.txt` instead. Always pass the vocabulary
explicitly in a release: the compiled default refers to the build machine.

## Start a server

MiSS (Linux shell):

```bash
./rwkv_miss_tune_server \
  --model /path/to/model.pth \
  --adapter /path/to/adapter.pth --state /path/to/state.pth \
  --vocab ./rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 --port 8848 \
  --chunk 64 --ctx 4096 --max-batch 8 \
  --cache-capacity 8 --keep-versions 2 --lr 2e-5 \
  --output ./online-rl-output/miss
```

State (Linux shell):

```bash
./rwkv_state_tune_server \
  --model /path/to/model.pth \
  --adapter /path/to/adapter.pth --state /path/to/state.pth \
  --vocab ./rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 --port 8849 \
  --chunk 64 --ctx 4096 --max-batch 8 \
  --cache-capacity 8 --keep-versions 2 --lr 0.01 \
  --output ./online-rl-output/state
```

Windows PowerShell (single-line commands avoid shell-specific continuations):

```powershell
.\rwkv_miss_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8848 --chunk 64 --ctx 4096 --max-batch 8 --lr 2e-5 --output .\online-rl-output\miss
.\rwkv_state_tune_server.exe --model C:\models\model.pth --adapter C:\models\adapter.pth --state C:\models\state.pth --vocab .\rwkv_vocab_v20230424.txt --host 127.0.0.1 --port 8849 --chunk 64 --ctx 4096 --max-batch 8 --lr 0.01 --output .\online-rl-output\state
```

Run only the command for your chosen mode. Omit `--adapter` to initialize MiSS D
at zero (or use no adapter in State mode); omit `--state` for zero initial state.
Cold adapters supply rank/alpha/targets and must match the base model.
`--help` lists all startup flags without loading a model.

`--ctx` limits complete history; oversized requests are rejected, never silently
truncated. `--chunk` controls checkpoint/recompute blocks without detaching gradients.
The batch limit applies to both rollout and train requests; samples execute
sequentially on one GPU worker. Prefix snapshots reside on CPU and are isolated
by policy version. Strict FP32 GEMM accumulation with BF16 inputs/outputs improves
actor/learner consistency but adds compute cost.

## Run the Python loop

Wait for `ready`, then use a second terminal:

```bash
curl http://127.0.0.1:8848/health
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 --data /path/to/segments.jsonl \
  --steps 10 --group-size 4 --max-turns 2 --max-new-tokens 64 \
  --temperature 1.0 --seed 1234 --timeout 3600 \
  --save --metrics ./miss-online-metrics.jsonl
```

For State use port `8849` and a separate metrics file. Python needs only `tqdm`;
HTTP uses the standard library. Each JSONL record has this structure:

```json
{"segments":[{"text":"User: question\nAssistant:","train":false},{"text":"reference answer","train":true}]}
```

Nontraining segments are observations; training segments provide reference answers
for generated assistant turns. The default judge is token unigram F1, a plumbing
example rather than a task-specific reward. Replace it with `--judge module:function`;
the callable accepts generated and reference token-ID lists and returns a scalar.
Tool observations are replayed from data; the example does not execute tool calls.

`--steps` counts processed batches, not optimizer updates. Equal group rewards
produce zero advantages and skip Adam. `--smoke` explicitly shortens the prompt
and may inject synthetic advantages; use it only for plumbing tests. Progress and
append-only JSONL metrics report reward mean/std, loss, gradient norm, learning rate,
step, throughput, and cache reuse. Full histories over server `ctx` are skipped.

## API and policy lifecycle

All POST bodies are JSON. Use the opaque `policy_version` returned by the server.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Current policy, learner step, published step and limits |
| `POST /v1/tokenize`, `/v1/detokenize` | Convert text to/from token IDs |
| `POST /v1/completions` | Batch prompts; return generated token IDs and aligned FP32 logprobs |
| `POST /v1/train` | One Adam step with tokens/turns, loss masks and signed advantages |
| `POST /v1/publish` | Publish the updated actor and optionally export files |

Sampling uses plain softmax(logits / temperature); logprobs reflect that temperature.
Use the same temperature for training. The loss is signed-advantage CE normalized
by the number of selected predictable tokens, without PPO clipping or a KL term.
Masked observations still carry gradients across chunks and turns. Training never
uses inference prefix snapshots. Collect rollouts, judge, train once, publish,
then collect fresh rollouts. A second update before publication is rejected (409).
See the [full request/response examples](online-rl.zh-CN.md#http-api).

## Save and recover

`--save` in the Python client requests exports on publication. A returned directory
contains `state.pth`, `adapter.pth` when applicable, and `policy.json`. Publication
keeps Adam intact; these files are inference exports, not optimizer checkpoints.
Restarting with `--adapter <export>/adapter.pth --state <export>/state.pth` restores
parameters but resets Adam and the server step counter. Omit adapter if none was exported.

The server disables Drogon's default 60-second idle connection timeout for long GPU
jobs. The client's `--timeout` defaults to 3600 seconds. If a request disconnects,
do not blindly retry training: it may already have updated parameters. Check
`/health`; if `step > published_step`, publish that exact step before continuing:

```bash
curl -X POST http://127.0.0.1:8848/v1/publish \
  -H 'Content-Type: application/json' \
  -d '{"expected_step":3,"save":true}'
```

Replace `3` with the observed step. The client refuses to start with an unpublished
update. After confirming which data rows completed, use `--start-row 4 --steps 7`
to process seven further batches starting at dataset line four; this is an explicit
cursor, not automatic checkpoint recovery. Missing response metrics are not fabricated.
Updating the server executable requires restarting the process.

CI validates compilation, CPU tests, bundled dependencies and relocated `--help`.
Hosted runners have no GPU; GPU training is not exercised there. Local HTTP tests
in `test/online_rl_http_test.py` cover both modes; `--check-idle-timeout` adds a
75-second silent-connection check per mode. See the
[GPU validation record](online-rl-validation.zh-CN.md).
