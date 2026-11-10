import requests
import time
import json
import numpy as np

URL = "http://localhost:8060/v1/batch/completions"
STATUS_URL = "http://localhost:8060/v1/server/status"
PROMPT = "User: Please describe the history and development of artificial intelligence in brief.\nAssistant:"
MAX_TOKENS = 64
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]

def single_batch_nonstream(bsz):
    payload = {
        "contents": [PROMPT] * bsz,
        "max_tokens": MAX_TOKENS,
        "stop_tokens": [0],
        "stream": False,
        "metrics": True
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, timeout=300.0)
    wall_time = time.perf_counter() - t0
    
    if resp.status_code != 200:
        return {"status": resp.status_code, "error": resp.text, "wall_time": wall_time}
    
    status_data = requests.get(STATUS_URL).json()
    last_req = status_data.get("last_request", {})
    gen_tokens = last_req.get("generated_tokens", bsz * MAX_TOKENS)
    prompt_tokens = last_req.get("prompt_tokens", 0)
    prefill_speed = last_req.get("prefill_speed", 0.0)
    decode_speed = last_req.get("decode_speed", 0.0)
    e2e_throughput = gen_tokens / wall_time if wall_time > 0 else 0.0

    return {
        "status": 200,
        "wall_time": wall_time,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "e2e_throughput": e2e_throughput,
        "per_token_ms": (wall_time / MAX_TOKENS) * 1000
    }

def single_batch_stream(bsz):
    payload = {
        "contents": [PROMPT] * bsz,
        "max_tokens": MAX_TOKENS,
        "stop_tokens": [0],
        "stream": True,
        "chunk_size": 1,
        "metrics": True
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, timeout=300.0, stream=True)
    
    first_chunk_time = None
    chunks = 0
    for line in resp.iter_lines():
        if line:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
            chunks += 1
    total_time = time.perf_counter() - t0

    status_data = requests.get(STATUS_URL).json()
    last_req = status_data.get("last_request", {})
    gen_tokens = last_req.get("generated_tokens", bsz * MAX_TOKENS)
    prefill_speed = last_req.get("prefill_speed", 0.0)
    decode_speed = last_req.get("decode_speed", 0.0)

    return {
        "status": resp.status_code,
        "ttft": first_chunk_time,
        "wall_time": total_time,
        "gen_tokens": gen_tokens,
        "prefill_speed": prefill_speed,
        "decode_speed": decode_speed,
        "throughput": gen_tokens / total_time if total_time > 0 else 0
    }

def benchmark():
    print("=== Step 1: Warmup Port 8060 ===")
    single_batch_nonstream(4)
    print("Warmup finished.\n")

    print("=== Step 2: Non-Streaming Batch Completion Benchmark ===")
    results_nonstream = {}
    for bsz in BATCH_SIZES:
        rounds = 3 if bsz <= 32 else 2
        print(f"Running bsz={bsz} ({rounds} rounds)...", flush=True)
        runs = []
        for _ in range(rounds):
            res = single_batch_nonstream(bsz)
            if res["status"] == 200:
                runs.append(res)
            time.sleep(0.3)
        
        # Select best run by e2e_throughput
        best = max(runs, key=lambda x: x["e2e_throughput"])
        results_nonstream[bsz] = best
        print(f"  [bsz={bsz:3d}] WallTime: {best['wall_time']:.3f}s | Prefill: {best['prefill_speed']:8.1f} t/s | Decode: {best['decode_speed']:8.1f} t/s | E2E Throughput: {best['e2e_throughput']:8.1f} t/s")

    print("\n=== Step 3: Streaming Batch Completion Benchmark (TTFT & Stream Throughput) ===")
    results_stream = {}
    for bsz in BATCH_SIZES:
        rounds = 2
        print(f"Running stream bsz={bsz} ({rounds} rounds)...", flush=True)
        runs = []
        for _ in range(rounds):
            res = single_batch_stream(bsz)
            if res["status"] == 200:
                runs.append(res)
            time.sleep(0.3)
        best = max(runs, key=lambda x: x["throughput"])
        results_stream[bsz] = best
        print(f"  [stream bsz={bsz:3d}] WallTime: {best['wall_time']:.3f}s | TTFT: {best['ttft']*1000:6.1f}ms | Decode: {best['decode_speed']:8.1f} t/s | E2E Throughput: {best['throughput']:8.1f} t/s")

    with open("results_8060.json", "w") as f:
        json.dump({
            "max_tokens": MAX_TOKENS,
            "prompt": PROMPT,
            "non_streaming": results_nonstream,
            "streaming": results_stream
        }, f, indent=2)
    print("\nAll tests completed. Saved to results_8060.json.")

if __name__ == "__main__":
    benchmark()
