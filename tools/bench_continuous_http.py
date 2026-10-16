#!/usr/bin/env python3
"""Compare independent HTTP requests with one multi-row request.

Run the old and new servers on separate ports with the same model/flags, then:
  python tools/bench_continuous_http.py --urls http://127.0.0.1:8000 http://127.0.0.1:8001
No services are started or reconfigured. Uses only the Python standard library.
"""
import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls', nargs='+', default=['http://127.0.0.1:8000'])
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--batch-rows', type=int, default=64)
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.batch_rows, *args.concurrency) <= 0:
        parser.error('all sizes must be positive')
    report = []
    for url in args.urls:
        for requests, rows in [(n, 1) for n in args.concurrency] + [(1, args.batch_rows)]:
            payload = json.dumps(dict(
                contents=['User: Count integers and explain each one.\n\nAssistant:'] * rows,
                max_tokens=args.tokens, stop=[], top_k=1, top_p=0,
                temperature=1, alpha_presence=0, alpha_frequency=0,
            )).encode()

            def call(barrier=None):
                if barrier:
                    barrier.wait()
                start = time.perf_counter()
                req = urllib.request.Request(url.rstrip('/') + '/v1/batch/completions',
                                             payload, {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=600) as response:
                    result = json.load(response)
                if len(result.get('choices', [])) != rows or any(
                        choice.get('finish_reason') != 'length' for choice in result['choices']):
                    raise RuntimeError(f'incomplete benchmark request: {result}')
                return time.perf_counter() - start

            call()  # Warm kernels and the prompt cache before measuring.
            runs = []
            for _ in range(args.repeats):
                barrier = threading.Barrier(requests)
                with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
                    start = time.perf_counter()
                    latencies = list(pool.map(lambda _: call(barrier), range(requests)))
                    wall = time.perf_counter() - start
                runs.append(dict(seconds=wall, tokens_per_second=requests * rows * args.tokens / wall,
                                 request_seconds=latencies))
            item = dict(url=url, concurrent_requests=requests, rows_per_request=rows,
                        tokens_per_row=args.tokens, runs=runs,
                        median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in runs))
            report.append(item)
            print(json.dumps(item), flush=True)
    if args.output:
        with open(args.output, 'w') as output:
            json.dump(report, output, indent=2)


if __name__ == '__main__':
    main()
