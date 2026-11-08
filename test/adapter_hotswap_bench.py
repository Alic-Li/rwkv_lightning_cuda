#!/usr/bin/env python3
"""Throughput benchmark for request-level adapter hot switching.

The server must already be running. Example:
python test/adapter_hotswap_bench.py --url http://127.0.0.1:8070 \
  --adapter miss=./exp/miss/adapter-final.pth --adapter nora=./exp/nora/adapter-final.pth

Each scenario sends --concurrency non-streaming chat requests at once. Greedy
sampling with no stop tokens makes every request decode exactly --max-tokens,
so aggregate decode throughput is comparable across scenarios. Outputs are kept
so --check can compare a request's text between solo and mixed batches.
"""
import argparse
import concurrent.futures
import http.client
import json
import time
import urllib.parse
from pathlib import Path


def call(url, path, body=None, method=None, timeout=600):
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method or ('POST' if data else 'GET'), path, body=data,
                 headers={'Content-Type': 'application/json'} if data else {})
    resp = conn.getresponse()
    raw = resp.read()
    if resp.status != 200:
        raise RuntimeError(f'{path} -> {resp.status}: {raw[:400]!r}')
    return json.loads(raw)


def run(args, assignment, prompt_suffix=''):
    def one(i, adapter):
        body = dict(max_tokens=args.max_tokens, temperature=1, top_k=1, top_p=0,
                    alpha_presence=0, alpha_frequency=0, stop_tokens=[],
                    think_type='none',
                    messages=[{'role': 'user', 'content': args.prompt + prompt_suffix}])
        if adapter:
            body['adapter_id'] = adapter
        t0 = time.monotonic()
        r = call(args.url, '/v1/chat/completions', body)
        return i, adapter, time.monotonic() - t0, r['choices'][0]['message']['content'], r.get('usage', {})

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(len(assignment)) as pool:
        results = list(pool.map(lambda p: one(*p), enumerate(assignment)))
    wall = time.monotonic() - t0
    tokens = sum(r[4].get('completion_tokens', args.max_tokens) for r in results)
    return wall, tokens, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8070')
    p.add_argument('--adapter', action='append', default=[], help='ID=PATH, registered before the run')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--prompt', default='Write a short story about a cat who learns to fly.')
    p.add_argument('--scenarios', default='base,miss,nora,mix2,mix4')
    p.add_argument('--output')
    args = p.parse_args()

    ids = []
    for spec in args.adapter:
        aid, path = spec.split('=', 1)
        call(args.url, '/v1/adapters', {'adapter_id': aid, 'path': str(Path(path).resolve())})
        ids.append(aid)

    def assignment(name, n):
        if name == 'base':
            return [None] * n
        if name in ids:
            return [name] * n
        if name == 'mix2':
            return [None if i % 2 == 0 else ids[0] for i in range(n)]
        if name == 'mix4':
            pool = [None] + ids
            return [pool[i % len(pool)] for i in range(n)]
        raise SystemExit('unknown scenario ' + name)

    run(args, [None] + ids)  # warm-up: cuBLAS tuning, adapter H2D
    report = []
    texts = {}
    for n in args.concurrency:
        for name in args.scenarios.split(','):
            if n == 1 and name.startswith('mix'):
                continue
            best = None
            for _ in range(args.repeats):
                wall, tokens, results = run(args, assignment(name, n))
                if best is None or wall < best[0]:
                    best = (wall, tokens, results)
            wall, tokens, results = best
            for _, adapter, _, text, _ in results:
                texts.setdefault((n, name, adapter), text)
            row = dict(concurrency=n, scenario=name, wall_s=round(wall, 3),
                       tokens=tokens, tok_s=round(tokens / wall, 1),
                       mean_latency_s=round(sum(r[2] for r in results) / len(results), 3))
            report.append(row)
            print(json.dumps(row), flush=True)

    # Greedy rows must not depend on which adapters share their batch. Batch
    # size itself may change kernels (sparse/dense FFN), so compare a mixed
    # scenario only with the single-adapter scenario at the same concurrency.
    mismatches = 0
    for (n, name, adapter), text in texts.items():
        if not name.startswith('mix'):
            continue
        pure = texts.get((n, adapter or 'base', adapter))
        if pure is not None and pure != text:
            mismatches += 1
            print(f'text differs: concurrency={n} scenario={name} adapter={adapter}')
    distinct = len({texts.get((n, a or 'base', a)) for n in args.concurrency
                    for a in [None] + ids} - {None})
    print(json.dumps({'distinct_pure_texts': distinct, 'batch_mismatches': mismatches}))
    if args.output:
        Path(args.output).write_text(json.dumps(dict(
            rows=report, mismatches=mismatches,
            texts=[dict(concurrency=n, scenario=s, adapter=a, text=t) for (n, s, a), t in texts.items()]),
            indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
