import aiohttp
import asyncio
import time
import json
import numpy as np

URL_BATCH = "http://localhost:8060/v1/batch/completions"
URL_CHAT = "http://localhost:8060/v1/chat/completions"
PROMPT = "User: Please describe the history and development of artificial intelligence in brief.\nAssistant:"
MAX_TOKENS = 64
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]

async def worker_batch_single(session):
    payload = {
        "contents": [PROMPT],
        "max_tokens": MAX_TOKENS,
        "stop_tokens": [0],
        "stream": False
    }
    t0 = time.perf_counter()
    async with session.post(URL_BATCH, json=payload) as resp:
        body = await resp.json()
        lat = time.perf_counter() - t0
        return resp.status, lat, MAX_TOKENS

async def worker_chat_single(session):
    payload = {
        "messages": [{"role": "user", "content": "Please describe the history and development of artificial intelligence in brief."}],
        "max_tokens": MAX_TOKENS,
        "stop_tokens": [0],
        "stream": False
    }
    t0 = time.perf_counter()
    async with session.post(URL_CHAT, json=payload) as resp:
        body = await resp.json()
        lat = time.perf_counter() - t0
        toks = body.get("usage", {}).get("completion_tokens", MAX_TOKENS)
        return resp.status, lat, toks

async def run_concurrent_test(worker_fn, bsz, rounds=2):
    connector = aiohttp.TCPConnector(limit=max(500, bsz + 50))
    timeout = aiohttp.ClientTimeout(total=180.0)
    all_runs = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for _ in range(rounds):
            t0 = time.perf_counter()
            tasks = [worker_fn(session) for _ in range(bsz)]
            results = await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - t0

            success = [r for r in results if r[0] == 200]
            latencies = [r[1] for r in success]
            total_toks = sum(r[2] for r in success)
            throughput = total_toks / wall_time if wall_time > 0 else 0

            all_runs.append({
                "wall_time": wall_time,
                "throughput": throughput,
                "avg_lat": float(np.mean(latencies)),
                "p50_lat": float(np.percentile(latencies, 50)),
                "p90_lat": float(np.percentile(latencies, 90)),
                "p99_lat": float(np.percentile(latencies, 99)),
                "min_lat": float(np.min(latencies)),
                "max_lat": float(np.max(latencies)),
                "success_rate": len(success) / bsz
            })
            await asyncio.sleep(0.3)

    return max(all_runs, key=lambda x: x["throughput"])

async def main():
    print("=== Warmup 8060 Concurrent ===")
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(3):
            await worker_batch_single(session)
    print("Warmup finished.\n")

    results_batch_concurrent = {}
    results_chat_concurrent = {}

    print("=== Testing Concurrent Batch Single-item Requests (/v1/batch/completions) ===")
    for bsz in BATCH_SIZES:
        rounds = 3 if bsz <= 32 else 2
        res = await run_concurrent_test(worker_batch_single, bsz, rounds=rounds)
        results_batch_concurrent[bsz] = res
        print(f"  [Batch bsz={bsz:3d}] WallTime: {res['wall_time']:.3f}s | Throughput: {res['throughput']:8.1f} tok/s | Avg Lat: {res['avg_lat']*1000:6.1f}ms | P50: {res['p50_lat']*1000:6.1f}ms | P90: {res['p90_lat']*1000:6.1f}ms")

    print("\n=== Testing Concurrent OpenAI Chat Requests (/v1/chat/completions) ===")
    for bsz in BATCH_SIZES:
        rounds = 3 if bsz <= 32 else 2
        res = await run_concurrent_test(worker_chat_single, bsz, rounds=rounds)
        results_chat_concurrent[bsz] = res
        print(f"  [Chat  bsz={bsz:3d}] WallTime: {res['wall_time']:.3f}s | Throughput: {res['throughput']:8.1f} tok/s | Avg Lat: {res['avg_lat']*1000:6.1f}ms | P50: {res['p50_lat']*1000:6.1f}ms | P90: {res['p90_lat']*1000:6.1f}ms")

    with open("results_8060_concurrent.json", "w") as f:
        json.dump({
            "batch_concurrent": results_batch_concurrent,
            "chat_concurrent": results_chat_concurrent
        }, f, indent=2)

    print("\nConcurrent benchmark finished. Saved to results_8060_concurrent.json.")

if __name__ == "__main__":
    asyncio.run(main())
