#!/usr/bin/env python3
"""Compare deterministic greedy generations from two local HTTP servers."""

import argparse
import json
import subprocess
import time
import urllib.request


PROMPTS = [
    "请用一段简短中文说明矩阵乘法的作用。",
    "User: Explain why the sky appears blue.\nAssistant:",
    "```cpp\nint main() { return 0; }\n```\n请解释这段代码。",
]


def request(url, prompt):
    body = json.dumps({
        "model": "calibration",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.001,
        "top_k": 1,
        "top_p": 1,
        "max_tokens": 64,
        "stream": False,
    }).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"missing choices: {payload}")
    return choices[0].get("text", choices[0].get("message", {}).get("content", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--int8-url", default="http://127.0.0.1:8002/v1/chat/completions")
    args = parser.parse_args()
    for index, prompt in enumerate(PROMPTS):
        try:
            fp16 = request(args.fp16_url, prompt)
            int8 = request(args.int8_url, prompt)
        except Exception as error:
            print(f"prompt={index} unavailable: {error}")
            continue
        equal = sum(a == b for a, b in zip(fp16, int8))
        total = max(len(fp16), len(int8), 1)
        first = next((position for position, (a, b) in enumerate(zip(fp16, int8)) if a != b), total)
        print(f"prompt={index} agreement={equal / total:.3f} first_divergence={first}")


if __name__ == "__main__":
    main()
