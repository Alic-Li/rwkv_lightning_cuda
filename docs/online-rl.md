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
The batch limit applies to both rollout and train requests. Prompts of one rollout
request are decoded together as one batch (see below); training samples still run
one after another, and rollout and training share one GPU worker. Prefix snapshots
reside on CPU and are isolated by policy version. The default `--gemm-mode bf16`
uses BF16 Tensor Core GEMMs for both rollout and training, with FP32 accumulation
and reduced-precision reductions disabled. Fixed 32-row tiles are retained to
limit numerical drift, without pedantic accumulation. Different batch/chunk
shapes can introduce BF16 rounding differences, including different samples for
the same seed. `--gemm-mode strict` restores the legacy fixed 32-row, pedantic
path for bitwise comparisons. `--resume DIR` restores a checkpoint (see below);
use the same GEMM mode recorded in its `policy.json` for numerical continuity.

Both modes directly reuse `cuda/rwkv7_statepassing_clampw.cu`: BF16 sequence
inputs/outputs and gradients, FP32 recurrent state and state adjoints. `/health`
reports the GEMM mode, kernel and dtypes. Prefill computes only the final output
head. Checkpoint sweeps and masked-only chunks skip unnecessary output heads,
while all preceding context retains full recurrent/shift backpropagation.
Zero-weight samples and loss-free trailing suffixes skip computation; State
gradients accumulate on GPU. Client metrics include rollout/train/publish times
and training token throughput. CUDA has been validated; HIP numerical and
performance behavior needs separate validation.
`rollout_loss` independently reconstructs the objective from sampled logprobs
and advantages; `loss_abs_error` compares it with the learner loss without
additional GPU work.

GPU memory, estimated from the allocation code (not measured; add about 1 GiB for
the CUDA context). 2.9B assumes 32 layers, C=2560; 7.2B assumes 32 layers, C=4096.

| Part | Formula | 2.9B | 7.2B |
|---|---|---|---|
| BF16 backbone | ~2 bytes/parameter | ~5.8 GB | ~14.4 GB |
| Chunk-boundary checkpoints | `ceil((longest sample-1)/chunk) × (L·H·16 KiB + L·2·C·2 B)` | 20.3 MiB each | 32.5 MiB each |
| ctx 8192, chunk 64 / 128 | | ~2.5 / ~1.3 GiB | ~4.1 / ~2.0 GiB |
| Learner + actor activations | `2 × chunk × L × ~42·C bytes` | ~0.45 GB at chunk 64 | ~0.7 GB at chunk 64 |
| Batched decode, B prompts | `B × L·H·16 KiB` + `B × L × ~42·C bytes` | ~0.8 GB at B=32 | ~1.2 GB at B=32 |
| MiSS rank 16, all targets | ~18 bytes/parameter | ~0.2 GB | ~0.35 GB |

Checkpoints follow the longest sample trained so far, not `--ctx` itself. For long
contexts a larger `--chunk` trades fewer checkpoints for larger activations; at ctx
8192, chunk 128 needs less memory than 64.

## Run the Python loop

Wait for `ready`, then use a second terminal:

```bash
curl http://127.0.0.1:8848/health
uv run --with tqdm python tools/online_rl.py \
  --url http://127.0.0.1:8848 --data /path/to/segments.jsonl \
  --steps 10 --group-size 4 --max-turns 2 --max-tokens 1024 \
  --temperature 1.0 --seed 1234 --timeout 3600 \
  --save --metrics ./miss-online-metrics.jsonl
```

For State use port `8849` and a separate metrics file. Python needs only `tqdm`;
HTTP uses the standard library. Each JSONL record has this structure:

```json
{"segments":[{"text":"User: question\nAssistant:","train":false},{"text":"reference answer","train":true}]}
```

Both Server and client default to string stops: `</tool_call>`, `</answer>`, and
`\n\nUser:`. The Server's `--max-tokens 1024` is a per-turn safety cap against
runaway repetition, not a target response length. The client inherits both
settings from `/health`. Override the cap with `--max-tokens` (legacy alias:
`--max-new-tokens`) or the strings with `--stop '["</tool_call>", "\n\nUser:"]'`;
`--stop '[]'` explicitly disables stops. The same JSON option configures Server
defaults. Restart an old Server before using the updated client.

The client budgets actual history lengths instead of assuming every turn fills
its cap. A truncated episode keeps its sampled tokens/logprobs and receives no
further replayed observations. Rewards still come from the judge. Metrics include
`finish_reasons`, `matched_stops`, `truncated_samples`, `truncation_reasons`, and
`completed_turns`; the progress bar shows the number of truncated samples.

Nontraining segments are observations; training segments provide reference answers
for generated assistant turns. The default judge is token unigram F1, a plumbing
example rather than a task-specific reward. Replace it with `--judge module:function`;
the callable accepts generated and reference token-ID lists and returns a scalar.
Tool observations are replayed from data; the example does not execute tool calls.
For real tool environments, tokenize each observation separately with `/v1/tokenize`
and append its IDs to the history (never re-tokenize the whole text), keep generated
tokens as returned, and send `turns` with `loss_mask` 0 for the prompt and observations
and 1 for generated turns. A complete multi-turn GRPO skeleton is in the
[Chinese guide](online-rl.zh-CN.md#真实工具环境的多轮示例).

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
| `POST /v1/logprobs` | Teacher-forced logprobs of token sequences under a policy |
| `POST /v1/train` | One Adam step with tokens/turns, loss masks and signed advantages |
| `POST /v1/publish` | Publish the updated actor; optionally export files or a full checkpoint |

Sampling uses plain softmax(logits / temperature); logprobs reflect that temperature.
Use the same temperature for training. Top-k/top-p are not offered: training CE uses
the full tempered softmax, so truncated sampling would no longer match it.

All prompts of a completions request are prefilled one by one (with the prefix
cache) and then decoded together, one forward per step. Strict mode reproduces
batch=1 logits bit for bit; BF16 Tensor Core mode permits rounding differences.
Row `i` samples from its own random stream seeded by
`(seed, i)`. String stops directly reuse the inference engine's `TextStopMatcher`:
after each token, match only the suffix of generated raw bytes, retaining at most
the longest stop's byte length. Matching spans token/UTF-8 boundaries, excludes
the prompt, is case-sensitive, and resolves overlapping suffixes in request
order. A substring inside a token does not trigger a stop. The matching token
stays in `tokens`, `logprobs`, text and the cached state. Use token IDs for
multi-turn histories.

Omitting stop rules uses Server defaults. `stop: []` disables them, including
token 0. Explicit `stop_token_ids` or `stop_token_sequences` selects the legacy
numeric path instead; combining either with `stop` returns 400, even with empty
arrays. The string list supports up to 16 strings, each 1–256 UTF-8 bytes.
`matched_stop` reports the matched rule.

Each row can generate at most `min(max_tokens, ctx - prompt_length)` tokens.
Reaching that budget returns `finish_reason: "length"`, `truncated: true`, and
`truncation_reason: "max_tokens"` or `"context_length"`. Existing history is never
trimmed; prompts exceeding `ctx` are rejected. A string hit on the last allowed
token takes precedence and returns `finish_reason: "stop"`, `truncated: false`.
`max_tokens: 0` remains prefill-only and is not marked truncated. `/health` exposes
`default_stop` and `default_max_tokens`; a completion's `max_tokens` reports its
effective per-row capacity.

The startup policy stays available as `reference_version` in `/health` and is never
retired. `/v1/logprobs` scores `sequences` under any live policy with the same
forward and reductions as training (`len(tokens)-1` values each), which lets a client
fold a KL penalty against the reference into per-token advantages. The loss is signed-advantage CE normalized
by the number of selected predictable tokens, without PPO clipping or a KL term.
Masked observations still carry gradients across chunks and turns. Training never
uses inference prefix snapshots. Collect rollouts, judge, train once, publish,
then collect fresh rollouts. A second update before publication is rejected (409).
See the [full request/response examples](online-rl.zh-CN.md#http-api).

## Save and recover

`--save` in the Python client requests exports on publication. A returned directory
contains `state.pth`, `adapter.pth` when applicable, and `policy.json`. Restarting with
`--adapter <export>/adapter.pth --state <export>/state.pth` restores parameters but
resets Adam and the server step counter. Omit adapter if none was exported.

Publishing with `"checkpoint": true` (implies `save`) also writes `training/`: for
MiSS the `rwkv_miss_tune` checkpoint (FP32 master, gradient, Adam m/v, initial state,
step); for State the FP32 state, Adam m/v and step. Restart with `--resume <directory>`
to continue with the same optimizer state and step; adapter and state default to the
files in that directory. MiSS resume checks rank, alpha, targets and the backbone.

External sampling: `/v1/train` checks only `policy_version`, so tokens sampled by
`rwkv_lighting_cuda` with the exported adapter (registered per step through
`POST /v1/adapters`, by path or multipart upload) are accepted. Its kernels are not
bit-identical to this server, and training applies no importance correction, so such
batches are slightly off-policy; compare with `/v1/logprobs` if you rely on it.

The server disables Drogon's default 60-second idle connection timeout for long GPU
jobs. The client's `--timeout` defaults to 3600 seconds. If a request disconnects,
do not blindly retry training: it may already have updated parameters. Check
`/health`; if `step > published_step`, publish that exact step before continuing:

```bash
curl -X POST http://127.0.0.1:8848/v1/publish \
  -H 'Content-Type: application/json' \
  -d '{"expected_step":3,"save":true}'
```

Use `"checkpoint":true` instead of `"save":true` to be able to `--resume` with Adam state.
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
