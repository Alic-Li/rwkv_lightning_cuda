import asyncio
import httpx
import requests
import time
import json
import numpy as np

PROMPT = "User: Please describe the history and development of artificial intelligence in brief.\nAssistant:"
MAX_TOKENS = 64
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]

def benchmark_8060_single(bsz):
    url = "http://localhost:8060/v1/batch/completions"
    payload = {
        "contents": [PROMPT] * bsz,
        "max_tokens": MAX_TOKENS,
        "stop_tokens": [0],
        "stream": False,
        "metrics": True
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=180.0)
        wall_time = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            total_tokens = sum(len(c.get("message", {}).get("content", "")) > 0 for c in choices) # approximate or query status
            # Fetch server status for exact metrics
            status_resp = requests.get("http://localhost:8060/v1/server/status").json()
            last_req = status_resp.get("last_request", {})
            gen_tokens = last_req.get("generated_tokens", bsz * MAX_TOKENS)
            prefill_speed = last_req.get("prefill_speed", 0.0)
            decode_speed = last_req.get("decode_speed", 0.0)
            e2e_throughput = gen_tokens / wall_time if wall_time > 0 else 0.0
            return {
                "status": "success",
                "status_code": 200,
                "wall_time": wall_time,
                "gen_tokens": gen_tokens,
                "prefill_speed": prefill_speed,
                "decode_speed": decode_speed,
                "e2e_throughput": e2e_throughput,
            }
        else:
            return {
                "status": "failed",
                "status_code": resp.status_code,
                "wall_time": wall_time,
                "error": resp.text
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "wall_time": time.perf_counter() - t0
        }

async def _worker_8059(client, prompt):
    url = "http://localhost:8059/v1/completions"
    payload = {
        "model": "rwkv7",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True
    }
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload, timeout=180.0)
    lat = time.perf_counter() - t0
    return resp.status_code, lat, resp.json() if resp.status_code == 200 else resp.text

async def benchmark_8059_concurrent(bsz):
    limits = httpx.Limits(max_connections=max(350, bsz + 50))
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        tasks = [_worker_8059(client, PROMPT) for _ in range(bsz)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    success_latencies = [r[1] for r in results if r[0] == 200]
    total_tokens = sum(r[2]["usage"]["completion_tokens"] for r in results if r[0] == 200)

    if len(success_latencies) == bsz:
        return {
            "status": "success",
            "wall_time": wall_time,
            "gen_tokens": total_tokens,
            "e2e_throughput": total_tokens / wall_time if wall_time > 0 else 0,
            "avg_lat": np.mean(success_latencies),
            "p50_lat": np.percentile(success_latencies, 50),
            "p90_lat": np.percentile(success_latencies, 90),
            "p99_lat": np.percentile(success_latencies, 99),
            "min_lat": np.min(success_latencies),
            "max_lat": np.max(success_latencies)
        }
    else:
        return {
            "status": "partial_or_failed",
            "wall_time": wall_time,
            "success_count": len(success_latencies),
            "error": [r[2] for r in results if r[0] != 200][:3]
        }

def main():
    print("=== Warmup 8060 & 8059 ===")
    benchmark_8060_single(4)
    asyncio.run(benchmark_8059_concurrent(4))
    print("Warmup done.\n")

    results_8060 = {}
    results_8059 = {}

    print(f"Testing Batch Sizes: {BATCH_SIZES}")
    print("-" * 60)

    for bsz in BATCH_SIZES:
        print(f"Running bsz={bsz} on Port 8060 (rwkv_lightning_cuda, /v1/batch/completions)...")
        res_60 = benchmark_8060_single(bsz)
        results_8060[bsz] = res_60
        if res_60["status"] == "success":
            print(f"  -> 8060 bsz={bsz}: WallTime={res_60['wall_time']:.3f}s, Prefill={res_60['prefill_speed']:.1f} t/s, Decode={res_60['decode_speed']:.1f} t/s, E2E Throughput={res_60['e2e_throughput']:.1f} t/s")
        else:
            print(f"  -> 8060 bsz={bsz}: Status {res_60.get('status_code')}, Error={res_60.get('error')}")

        time.sleep(1)

        print(f"Running bsz={bsz} on Port 8059 (vLLM / irwkv, concurrent /v1/completions)...")
        res_59 = asyncio.run(benchmark_8059_concurrent(bsz))
        results_8059[bsz] = res_59
        if res_59["status"] == "success":
            print(f"  -> 8059 bsz={bsz}: WallTime={res_59['wall_time']:.3f}s, E2E Throughput={res_59['e2e_throughput']:.1f} t/s, AvgLat={res_59['avg_lat']:.3f}s, P90Lat={res_59['p90_lat']:.3f}s")
        else:
            print(f"  -> 8059 bsz={bsz}: Status {res_59.get('status')}, Error={res_59.get('error')}")

        time.sleep(1)
        print("-" * 60)

    # Extra test for 8060 at max supported batch size before limit (160)
    print("Testing 8060 near limit bsz=160...")
    res_60_160 = benchmark_8060_single(160)
    results_8060[160] = res_60_160
    if res_60_160["status"] == "success":
        print(f"  -> 8060 bsz=160: WallTime={res_60_160['wall_time']:.3f}s, Prefill={res_60_160['prefill_speed']:.1f} t/s, Decode={res_60_160['decode_speed']:.1f} t/s, E2E Throughput={res_60_160['e2e_throughput']:.1f} t/s")

    with open("benchmark_results.json", "w") as f:
        json.dump({"8060": results_8060, "8059": results_8059}, f, indent=2)
    print("\nBenchmark finished. Saved to benchmark_results.json.")

if __name__ == "__main__":
    main()
