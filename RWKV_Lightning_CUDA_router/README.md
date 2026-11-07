# RWKV Lightning CUDA Router

Dynamic Adapter management and routing have been removed. `/v1/adapters` returns 404; requests containing `adapter_id`, `adapter_version`, or `adapter_scale` return 400. Remove old `adapter_upload_*` configuration keys. State APIs remain supported.

English | [简体中文](README.zh-CN.md)

An HTTP reverse proxy that balances RWKV inference servers by in-flight batch size
instead of request count. Inference and SSE pass through; State management uses a replication pipeline.

## Run

```bash
cp config.example.toml config.toml
go run . -config config.toml
```

Build a binary with `go build .`.

`config.toml` must contain a `listen` address and one or more `[[backends]]`.
Each backend `url` is an HTTP/HTTPS URL; `weight` is its relative capacity and
defaults to 1.

## Modules and shared replication pipeline

```text
main.go                         # CLI entry; go run/build . remain supported
internal/
  broadcast/                    # HTTP fan-out, retry, deadlines, response limits
  router/
    app.go, config.go            # Startup and configuration
    scheduler.go, proxy.go       # Scheduling, affinity, inference forwarding
    resource_sync.go             # Shared acknowledgments, errors, discovery
    state_*.go                  # State identities and replica tracking
    diagnostics.go              # Routing diagnostics
cmd/rwkv-batch-loadtest/         # Standalone load-test command
```

State operations follow one pipeline: validate and guard the ID, broadcast to every
worker, retry only failed uploads, wait for all results, validate identities, publish
readiness, then respond. Readiness is committed before the acknowledgment reaches
the client. State retains its shared UUID and existing retry configuration.

- Upload, list, delete, and discovery share the HTTP broadcast module. Only uploads
  retry. Lists/deletes also use the corresponding `state_upload_attempt_timeout_seconds`
  deadline (120 seconds by default); a shorter global timeout takes precedence.
  Discovery is bounded to five seconds and preserves auth headers and JSON `password`.
- Concurrent mutations of the same ID return 409; inference for that ID is blocked
  during mutations. Deletion requires a valid `deleted: true` acknowledgment.
  Uncertain deletion stays disabled; only uniform deterministic rejection restores
  the previous registration.
- Any failed or inconsistent replica prevents synchronization success. Partial
  failure returns 502 with `ready_backends` and `failed_backends`; each failure has
  a worker name, status, and reason. Uniform deterministic rejections pass through.
- Broadcasts do not follow redirects: acknowledgments must come from configured
  workers. Each response is limited to 8 MiB.
- State POST list/delete aliases and OPTIONS remain supported. Preflight does not
  mutate resource registration.

Run `go test -race ./...` and `go vet ./...` here. Tests cover the shared engine,
the State failure contract, and real HTTP sockets connecting the
router to three independent simulated workers for uploads, lists, and concurrent
inference. These require no model/GPU and do not verify a deployed production cluster.

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

An upload blocks new inference for that State until the entire broadcast and
acknowledgment validation finish. State inference then uses only healthy, confirmed
replicas, including when
session affinity points elsewhere. Cooldown expiry alone does not confirm a State.
No ready replica means HTTP 503, with no fallback to a zero State. Partial upload
failure after retry exhaustion returns HTTP 502 JSON containing `state_id`,
`ready_backends` and `failed_backends` (name, final status, and error reason; 0 when no status arrived).
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

## Diagnosing lists that fail after a successful upload

A timeout, 401/502, HTML error page, or invalid JSON list no longer erases confirmed
State registrations. A valid list confirming absence or a different identity,
or a runtime `uploaded state not found` response,
revokes the corresponding replica's readiness. Transport
failures still trigger connection cooldown; they do not establish deletion.
The list endpoint continues to report actual upstream failures and differences.

Logs identify each worker's upload attempts, status, elapsed time, and confirmed
State ID. List diagnostics include the worker and parsing/status failure. For
inconsistent lists, `failed_backends[].error` names the reference worker and the
missing, extra, or metadata-mismatched State entries. Upload contents and
credentials are not logged. `route_unavailable` records per-worker health, State readiness, and mutation state.

Generation responses include `X-RWKV-Backend`, the router-selected worker name;
`route_selected` logs the session ID and State ID. Requests with the same
`session_id` or `X-RWKV-Session-Id` remain on that worker, including concurrent
requests, while it is healthy and holds the required resources. Failure or missing
resources can move the session. Identical messages alone do not establish affinity.
Affinity itself does not prove a cache hit; inspect backend cache/generation metrics.

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
