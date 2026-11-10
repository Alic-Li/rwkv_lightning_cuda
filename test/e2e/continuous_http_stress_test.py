#!/usr/bin/env python3
"""Live concurrent multi-row throughput and string-stop checks (stdlib only).

Does not restart or reconfigure the server. Saves raw HTTP/SSE and timing evidence.
Throughput and overlapping SSE are indirect batching evidence, not GPU batch counters.
"""
import argparse
import concurrent.futures
import json
import statistics
import threading
import time

from string_stop_http_test import StringStopSuite


class StressSuite(StringStopSuite):
    def __init__(self, args):
        super().__init__(args)
        self.measurements = []
        self.prompts = [
            'English: Hello\n\nChinese:',
            'User: Count integers and explain each one.\n\nAssistant:',
            'User: Use the calculator tool to calculate 1+1.\n\nAssistant: <tool_call>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n',
            'User: Explain why the sky is blue.\n\nAssistant:',
        ]
        self.targets = []

    def save(self):
        super().save()
        if hasattr(self, 'measurements'):
            (self.output / 'measurements.json').write_text(json.dumps(self.measurements, indent=2))

    def parallel(self, count, task):
        barrier = threading.Barrier(count)
        def run(i):
            barrier.wait(timeout=30)
            return task(i)
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(run, i) for i in range(count)]
            # Wait for every request, even if an assertion fails in one worker.
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result(timeout=self.args.timeout))
                except Exception as exc:
                    errors.append(repr(exc))
            assert not errors, errors
            return results

    def call_batch(self, rows, extra=None, on_event=None):
        body = self.options | dict(contents=rows, max_tokens=self.args.tokens) | (extra or {})
        return self.call('/v1/batch/completions', body, on_event=on_event)

    def benchmark(self, count, rows):
        prompts = [self.prompts[1]] * rows
        self.call_batch(prompts)  # Warmup excluded from measurements.
        runs = []
        for _ in range(self.args.repeats):
            def request(i):
                begin = time.perf_counter()
                result = self.call_batch(prompts)
                texts, finishes = self.unpack(result)
                assert len(texts) == rows and all(texts), result
                assert all(c['finish_reason'] == 'length' for c in finishes), finishes
                return time.perf_counter() - begin
            begin = time.perf_counter()
            durations = self.parallel(count, request)
            elapsed = time.perf_counter() - begin
            runs.append(dict(seconds=elapsed, tokens_per_second=count*rows*self.args.tokens/elapsed,
                             request_seconds=durations))
        item = dict(kind='throughput', concurrent_requests=count, rows_per_request=rows,
                    tokens_per_row=self.args.tokens, runs=runs,
                    median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in runs))
        self.measurements.append(item)
        print(json.dumps(item), flush=True)

    def fixtures(self):
        # Obtain actual token boundaries, not arbitrary substrings inside tokens.
        result = self.call_batch(self.prompts, dict(max_tokens=48, stream=True, chunk_size=1))
        texts, finishes = self.unpack(result, True)
        assert all(c['finish_reason'] == 'length' for c in finishes), finishes
        boundaries = [[] for _ in self.prompts]
        prefixes = [''] * len(self.prompts)
        for event in result[:-1]:
            for choice in event['choices']:
                i = choice['index']
                chunk = choice['delta'].get('content', '')
                if chunk:
                    prefixes[i] += chunk
                    boundaries[i].append(prefixes[i])
        self.targets = [[boundaries[i][n] for n in (2, 7, 15)] for i in range(4)]
        # Rows 1/3 continue while rows 0/2 stop; reject an ambiguous fixture.
        assert all(t not in texts[j] for i in (0, 2) for t in self.targets[i] for j in (1, 3)), texts
        (self.output / 'stop-fixtures.json').write_text(json.dumps(dict(
            prompts=self.prompts, targets=self.targets, baseline=texts), ensure_ascii=False, indent=2))

    def stops(self, count, stream, chunk):
        row_count = self.args.rows
        def request(i):
            # Requests sharing GPU batches have different stop lists.
            level = i % 3
            stops = [self.targets[0][level], self.targets[2][level]]
            if i % 4 == 3:
                stops = ['never-match-live8000-39d7b']
            prompts = [self.prompts[(j+i) % 4] for j in range(row_count)]
            result = self.call_batch(prompts, dict(max_tokens=64, stream=stream, chunk_size=chunk,
                                              stop=stops))
            texts, finishes = self.unpack(result, stream)
            assert len(texts) == len(finishes) == row_count, (len(texts), row_count)
            stopped = 0
            for j, (text, finish) in enumerate(zip(texts, finishes)):
                should_stop = (j+i) % 4 in (0, 2) and i % 4 != 3
                assert finish['finish_reason'] == ('stop' if should_stop else 'length'), (i, j, text, finish, stops)
                if should_stop:
                    matched = finish.get('matched_stop')
                    assert matched in stops and text.endswith(matched), (i, j, text, finish)
                    assert text == matched, ('late stop or text contamination', i, j, text, matched)
                    stopped += 1
                else:
                    assert 'matched_stop' not in finish, (i, j, finish)
            if stream:
                # Verify finish delivery while non-stopped rows are still generating.
                seen_stop = False
                continued = False
                for event in result[:-1]:
                    for choice in event['choices']:
                        if choice.get('finish_reason') == 'stop':
                            seen_stop = True
                        if seen_stop and choice['delta'].get('content'):
                            continued = True
                assert not stopped or continued, ('no row continued after early stop', i)
            return dict(stopped_rows=stopped, length_rows=row_count-stopped)
        results = self.parallel(count, request)
        self.measurements.append(dict(kind='string_stop', concurrent_requests=count,
                                      rows_per_request=row_count, stream=stream, chunk_size=chunk,
                                      stopped_rows=sum(x['stopped_rows'] for x in results),
                                      length_rows=sum(x['length_rows'] for x in results)))

    def staggered(self):
        """New requests arrive after the first request has begun decoding."""
        started = threading.Event()
        records = []
        lock = threading.Lock()
        def request(i):
            begin = time.perf_counter()
            first = None
            def observe(event):
                nonlocal first
                if isinstance(event, dict) and any(c['delta'].get('content') for c in event.get('choices', [])):
                    if first is None:
                        first = time.perf_counter()
                    if i == 0:
                        started.set()
            result = self.call_batch([self.prompts[1]]*self.args.rows,
                                dict(max_tokens=256 if i == 0 else 32, stream=True, chunk_size=1), observe)
            texts, finishes = self.unpack(result, True)
            assert len(texts) == self.args.rows and all(c['finish_reason'] == 'length' for c in finishes)
            with lock:
                records.append(dict(request=i, start=begin, first_token=first, end=time.perf_counter()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            leader = pool.submit(request, 0)
            assert started.wait(60), 'leader never started'
            others = [pool.submit(request, i) for i in range(1, 16)]
            for job in [leader, *others]:
                job.result(timeout=self.args.timeout)
        lead = next(r for r in records if r['request'] == 0)
        admitted = [r for r in records if r['request'] and r['first_token'] < lead['end']]
        completed = [r for r in admitted if r['end'] < lead['end']]
        self.measurements.append(dict(kind='staggered', records=records,
                                      admitted_before_leader_end=len(admitted),
                                      completed_before_leader_end=len(completed)))
        assert len(admitted) == 15, ('new requests waited for leader completion', records)
        assert completed, ('no shorter request finished before leader', records)

    def run(self):
        self.check('initial idle server', self.initial_status)
        for count, rows in [(1, 1), (1, self.args.rows), (4, self.args.rows),
                            (16, 1), (16, self.args.rows), (1, 64)]:
            self.check(f'throughput {count} requests x {rows} rows', lambda c=count, r=rows: self.benchmark(c, r))
        self.check(f'staggered arrivals 16 requests x {self.args.rows} rows', self.staggered)
        self.check('token-boundary stop fixtures', self.fixtures)
        if self.targets:
            for count in (1, 4, 16):
                for stream in (False, True):
                    for chunk in (1, 100):
                        self.check(f'stop isolation {count} requests x {self.args.rows} rows stream={stream} chunk={chunk}',
                                   lambda c=count, s=stream, k=chunk: self.stops(c, s, k))
        self.check('final no leaked requests or reservations', self.final_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--password', default='')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', default='build/live8000-continuous-stress')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rows', type=int, default=16)
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.rows) <= 0:
        parser.error('sizes must be positive')
    suite = StressSuite(args)
    try:
        suite.run()
    finally:
        suite.save()
    print('Report:', suite.output / 'report.json', flush=True)
    return int(any(t['status'] == 'FAIL' for t in suite.results))


if __name__ == '__main__':
    raise SystemExit(main())
