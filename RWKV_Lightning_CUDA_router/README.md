# RWKV Lightning CUDA Router

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

`/state/*` requests with a `session_id` get a best-effort in-memory affinity entry
for the lifetime of the router process. Do not place stateful traffic behind a
router restart unless the state store is shared by all backends.

`/v1/state/upload`, `/v1/state/list`, and `/v1/state/delete` are sent concurrently
to every configured backend, and the router waits for every response. An upload
therefore creates the same filename-based `state_id` on every worker, so subsequent
inference requests carrying that ID remain safely load-balanced. If a worker fails
or workers return inconsistent status/IDs, the router returns an error instead of
claiming the state is synchronized. The example body limit is 513 MiB so it can
proxy the backend's 512 MiB state upload limit plus multipart framing.

The router does not retry POST requests: retrying after an uncertain upstream write
can execute a generation twice. Clients may safely retry connection failures.

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
