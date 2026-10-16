# RWKV Lightning CUDA Router

English | [简体中文](README.zh-CN.md)

An HTTP reverse proxy that balances RWKV inference servers by in-flight batch size
instead of request count. It passes every method, header, response status, and SSE
stream through unchanged.

## Run

```bash
cp config.example.toml config.toml
go run . -config config.toml
```

Build a binary with `go build .`.

`config.toml` must contain a `listen` address and one or more `[[backends]]`.
Each backend `url` is an HTTP/HTTPS URL; `weight` is its relative capacity and
defaults to 1.

## Scheduling

For every proxied request, the router determines its bsz from the first applicable
field: `contents`, `text_list`, `prompts`, or `inputs` array; then numeric `bsz` or
`batch_size`; otherwise it is 1. It picks the healthy backend with the lowest:

```
in_flight_bsz / weight
```

Ties rotate fairly. The bsz remains reserved while a normal response or SSE stream
is being copied, including backend queue time. It is released on completion, an
upstream error, or client cancellation. This makes small requests fill leftover
capacity without allowing one large batch to overload a GPU.

Transport failures mark only the affected backend unavailable for
`failure_cooldown_seconds`; HTTP error responses are forwarded and do not mark a
backend unhealthy because they may be valid request errors.

`/state/*` requests with a `session_id` in the body or an `X-RWKV-Session-Id` header
get a best-effort in-memory affinity entry for the lifetime of the router
process. Do not place stateful traffic behind a router restart unless the state
store is shared by all backends.

`/v1/state/upload`, `/v1/state/list`, and `/v1/state/delete` are sent concurrently
to every configured backend, and the router waits for every response. An upload
generates one UUID-v7 at the router and forwards it to all workers via
`X-RWKV-State-Upload-UUID`. Each worker creates the same `<filename-stem>-<uuid>`
ID, so subsequent
inference requests carrying that ID remain safely load-balanced. If a worker fails
or workers return inconsistent status/IDs, the router returns an error instead of
claiming the state is synchronized. Lists compare IDs, byte sizes and tensor
structure across workers (timestamps/order may differ). Upload failures can leave
files on successful workers: fan-out is not a distributed transaction. Upgrade
the router and all runtimes together; direct runtime uploads generate their own
UUID, so upload through the router for load-balanced inference. The example body limit is 513 MiB so it can
proxy the backend's 512 MiB state upload limit plus multipart framing.

Inference POSTs are never automatically retried. State uploads are an idempotent
exception: failed workers retry transport/read errors, timeouts and HTTP
408/429/500/502/503/504 using the same UUID and body. Deterministic 4xx errors are
not retried. Defaults: 3 attempts, 120 seconds per attempt, exponential backoff
with jitter, 250 ms base and 2000 ms cap. `Retry-After`, cancellation and the total
operation deadline are respected. See the `state_upload_*` configuration keys.
The runtime checks SHA-256, filename and size before returning an existing record;
conflicting bytes never overwrite it, and retries preserve the first upload time.

State inference is eligible only on healthy, confirmed replicas, including when
session affinity points elsewhere. Cooldown expiry alone does not confirm a State.
No ready replica means HTTP 503, with no fallback to a zero State. Partial upload
failure after retry exhaustion returns HTTP 502 JSON containing `state_id`,
`ready_backends` and `failed_backends` (name/final status, 0 when no status arrived).
When every worker deterministically rejects the upload, the runtime rejection is
preserved. Full success retains the original response shape.

Retries are bounded to the upload request; there is no indefinite background
repair after it returns. A failed replica rejoins State routing only after an
upload acknowledgment or later list confirmation. Unknown IDs, including after a
router restart, trigger list discovery with a five-second deadline before routing.
Lists still require cross-worker consistency for a successful client response.
Deletion blocks new uses of that State, uncertain deletion stays blocked, and
uniform validation/auth rejection restores previous readiness. Stale lists cannot
undo upload/delete mutations.

## Adapter replication

`POST /v1/adapters` (multipart upload or JSON path registration), `GET /v1/adapters`,
and `DELETE /v1/adapters` fan out to every configured worker, including workers in
connection cooldown. Registration succeeds only when all workers accept the same
`adapter_id` and content `version`. JSON `path` registration requires that path to
be readable on every worker. Prefer multipart for separate machines: the router
forwards the same PTH and optional legacy `checkpoint.json` bytes to each worker.

```bash
curl -sS http://127.0.0.1:8080/v1/adapters \
  -F 'adapter_id=miss-r16-lr3e-5' \
  -F 'file=@./rwkv7-g1k-7b-r16-lr3e-5-e2-miss.pth'
```

Add an `Authorization: Bearer ...` header when needed. Use the ID in generation
requests, preferably with the returned `version` as `adapter_version`. If omitted,
the router pins the request to its confirmed current content version instead of
relying on potentially different per-worker latest registrations. `adapter_scale`,
text `stop`, and SSE stop metadata are preserved.

- Uploads default to three attempts per failed worker, with a 120-second attempt
  deadline, bounded exponential backoff, jitter, and `Retry-After` support.
  `adapter_upload_*` configures this independently of State uploads. Only transport,
  response-read, timeout, and HTTP 408/429/500/502/503/504 failures retry, with the
  same ID/body; successful workers and deterministic 4xx responses do not retry.
- Partial failure returns HTTP 502 JSON with `adapter_id`, the confirmed `version`,
  `ready_backends`, and `failed_backends`. Successful replicas remain available;
  inference only uses healthy workers confirmed to hold the requested version.
  State and adapter requirements must both be satisfied on the selected worker.
  With no eligible worker, inference returns 503 without dropping either resource.
  Inference itself is never replayed automatically.
- New inference for an ID is blocked during its upload/delete. Concurrent mutations
  of the same ID return 409. Deletion broadcasts to all workers and uncertain
  results keep the deleted ID/version disabled. Deterministic rejection by all
  workers, such as unauthorized requests, restores previous readiness.
- Lists compare `(id, version, manifest)` across all workers, ignoring order and
  worker-local cache statistics. Differences return 502 instead of a random
  worker's list. On success, statistics belong to the first worker, not the cluster
  total. Stale lists cannot overwrite upload/delete results.
- After restart or an unknown adapter request, a five-second bounded list probe
  rebuilds readiness. Existing runtime lists do not expose latest-registration
  order. A sole discovered version can be selected automatically; with multiple
  versions for an ID, provide `adapter_version` or re-register through the router.

Replication is not a distributed transaction and does not retry forever after
returning an error. Re-upload with the same ID to synchronize again, then check the
router list. An outer Cloudflare 502/timeout can still occur after an upload has
actually completed; use the cluster-wide router list to confirm the result.
States must likewise be uploaded through `/v1/state/*`. Files previously uploaded
to just one worker by an older router are not automatically copied to other hosts.

## Batch load test

The included command sends non-streaming `/v1/batch/completions` requests at each
specified concurrent-request level and reports the peak sustained prompt (sample)
throughput. Supply the Cloudflare Access credentials as flags (or the equivalent
`RWKV_CF_ACCESS_CLIENT_ID` and `RWKV_CF_ACCESS_CLIENT_SECRET` environment variables):

```bash
go run ./cmd/rwkv-batch-loadtest \
  -url 'https://api-7b.rwkvos.com/v1/batch/completions' \
  -cf-client-id 'XXX' \
  -cf-client-secret 'XXX' \
  -batch-size 8 \
  -concurrency '1,2,4,8,16,32' \
  -duration 30s \
  -max-tokens 128
```

`samples/s` is completed prompts per second (`successful requests/s × batch-size`).
Use the same prompt, batch size and `max_tokens` as the intended workload; otherwise
the resulting limit is not comparable. The command does not print credentials.
