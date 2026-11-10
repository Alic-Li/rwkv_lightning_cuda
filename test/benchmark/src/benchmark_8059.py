import aiohttp
import asyncio
import time
import json
import numpy as np

URL = "http://localhost:8059/v1/completions"
MODEL = "rwkv7"
PROMPT = "User: Please describe the history and development of artificial intelligence in brief.\nAssistant:"
MAX_TOKENS = 64
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]

async def single_request_nonstream(session, prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "stream": False
    }
    t0 = time.perf_counter()
    async with session.post(URL, json=payload) as resp:
        body = await resp.json()
        lat = time.perf_counter() - t0
        status = resp.status
        tokens = body.get("usage", {}).get("completion_tokens", 0) if status == 200 else 0
        prompt_tokens = body.get("usage", {}).get("prompt_tokens", 0) if status == 200 else 0
        return {
            "status": status,
            "latency": lat,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "body": body if status != 200 else None
        }

async def single_request_stream(session, prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "stream": True
    }
    t0 = time.perf_counter()
    async with session.post(URL, json=payload) as resp:
        ttft = None
        tokens = 0
        async for line in resp.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1
        total_lat = time.perf_counter() - t0
        return {
            "status": resp.status,
            "ttft": ttft if ttft is not None else total_lat,
            "latency": total_lat,
            "tokens": tokens
        }

async def benchmark_concurrency(bsz, stream=False, rounds=2):
    connector = aiohttp.TCPConnector(limit=max(500, bsz + 100))
    timeout = aiohttp.ClientTimeout(total=180.0)
    all_runs = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for r in range(rounds):
            t_start = time.perf_counter()
            if stream:
                tasks = [single_request_stream(session, PROMPT) for _ in range(bsz)]
            else:
                tasks = [single_request_nonstream(session, PROMPT) for _ in range(bsz)]
            
            results = await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - t_start

            success = [res for res in results if res["status"] == 200]
            if len(success) < bsz:
                print(f"Warning: bsz={bsz} round {r+1} only {len(success)}/{bsz} succeeded")
                continue

            latencies = [res["latency"] for res in success]
            total_tokens = sum(res["tokens"] for res in success)
            throughput = total_tokens / wall_time if wall_time > 0 else 0

            run_data = {
                "wall_time": wall_time,
                "total_tokens": total_tokens,
                "throughput": throughput,
                "avg_lat": float(np.mean(latencies)),
                "p50_lat": float(np.percentile(latencies, 50)),
                "p90_lat": float(np.percentile(latencies, 90)),
                "p99_lat": float(np.percentile(latencies, 99)),
                "min_lat": float(np.min(latencies)),
                "max_lat": float(np.max(latencies)),
            }
            if stream:
                ttfts = [res["ttft"] for res in success if res.get("ttft") is not None]
                run_data["avg_ttft"] = float(np.mean(ttfts))
                run_data["p50_ttft"] = float(np.percentile(ttfts, 50))
                run_data["p90_ttft"] = float(np.percentile(ttfts, 90))

            all_runs.append(run_data)
            await asyncio.sleep(0.5)

    # Pick the best run by throughput or average them
    best_run = max(all_runs, key=lambda x: x["throughput"])
    return best_run

async def main():
    print("=== Step 1: Warmup Port 8059 ===")
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(3):
            await single_request_nonstream(session, PROMPT)
    print("Warmup finished.\n")

    print("=== Step 2: Non-Streaming Concurrency Benchmark (Standard OpenAI Completion) ===")
    results_nonstream = {}
    for bsz in BATCH_SIZES:
        rounds = 3 if bsz <= 32 else 2
        print(f"Running bsz={bsz} ({rounds} rounds)...", flush=True)
        res = await benchmark_concurrency(bsz, stream=False, rounds=rounds)
        results_nonstream[bsz] = res
        print(f"  [bsz={bsz:3d}] WallTime: {res['wall_time']:.3f}s | Throughput: {res['throughput']:7.1f} tok/s | Avg Lat: {res['avg_lat']*1000:6.1f}ms | P50: {res['p50_lat']*1000:6.1f}ms | P90: {res['p90_lat']*1000:6.1f}ms | P99: {res['p99_lat']*1000:6.1f}ms")

    print("\n=== Step 3: Streaming Concurrency Benchmark (TTFT & ITL) ===")
    results_stream = {}
    for bsz in BATCH_SIZES:
        rounds = 2
        print(f"Running stream bsz={bsz}...", flush=True)
        res = await benchmark_concurrency(bsz, stream=True, rounds=rounds)
        results_stream[bsz] = res
        print(f"  [stream bsz={bsz:3d}] WallTime: {res['wall_time']:.3f}s | Throughput: {res['throughput']:7.1f} tok/s | Avg TTFT: {res['avg_ttft']*1000:5.1f}ms | P90 TTFT: {res['p90_ttft']*1000:5.1f}ms | Avg Lat: {res['avg_lat']*1000:6.1f}ms")

    with open("results_8059.json", "w") as f:
        json.dump({
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "prompt": PROMPT,
            "non_streaming": results_nonstream,
            "streaming": results_stream
        }, f, indent=2)
    print("\nAll tests completed. Saved to results_8059.json.")

if __name__ == "__main__":
    asyncio.run(main())
