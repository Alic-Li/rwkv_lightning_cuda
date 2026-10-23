#!/usr/bin/env python3
"""Minimal rollout -> Python judge -> signed GRPO -> publish HTTP client.
Only tqdm is required. Example: uv run --with tqdm tools/online_rl.py --help
"""
import argparse
from collections import Counter
import importlib
import http.client
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

from tqdm import tqdm


class Client:
    def __init__(self, url, timeout=3600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def call(self, path, body=None):
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc
        except (http.client.HTTPException, urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Connection failed during {path}: {exc}. Request was NOT retried. "
                "A train/publish operation may already have completed; check /health "
                "before continuing. If step > published_step, publish that step first."
            ) from exc


def token_f1(generated, reference):
    """Toy offline reward; replace with a task/environment judge for real RL."""
    overlap = sum((Counter(generated) & Counter(reference)).values())
    return 2 * overlap / max(1, len(generated) + len(reference))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8848")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--start-row", type=int, default=1, help="first dataset line to process (1-based); earlier lines are skipped")
    p.add_argument("--timeout", type=float, default=3600, help="HTTP response timeout in seconds")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge", help="module:function taking (generated_token_ids, reference_token_ids), returning reward")
    p.add_argument("--metrics", type=Path, default=Path("online-rl-metrics.jsonl"))
    p.add_argument("--save", action="store_true", help="export adapter/state on each publish")
    p.add_argument("--smoke", action="store_true", help="plumbing test: shorten initial context to 64 tokens; use synthetic +/- advantages if rewards tie")
    args = p.parse_args()
    if args.start_row < 1 or not 0 < args.timeout < float("inf"):
        p.error("start-row and timeout must be positive and finite")
    if min(args.steps, args.group_size, args.max_turns, args.max_new_tokens) < 1 or args.group_size < 2:
        p.error("positive sizes and group-size >= 2 required")
    judge = token_f1
    if args.judge:
        module, name = args.judge.rsplit(":", 1)
        judge = getattr(importlib.import_module(module), name)
    api = Client(args.url, args.timeout)
    health = api.call("/health")
    if health["step"] != health["published_step"]:
        p.error(f"server has unpublished step {health['step']}; POST /v1/publish "
                f"with expected_step={health['step']} and save=true before resuming")
    if args.group_size > health["max_batch"]:
        p.error("group-size exceeds server max_batch")
    version = health["policy_version"]
    updates = 0
    with args.data.open() as source, args.metrics.open("a") as metrics, tqdm(total=args.steps, desc="Online RL", unit="batch") as bar:
        for line_number, line in enumerate(source, 1):
            if line_number < args.start_row:
                continue
            if not line.strip():
                continue
            if bar.n >= args.steps:
                break
            row = json.loads(line)
            segments = row.get("segments")
            if not segments:
                raise ValueError("example expects segments [{text, train}]; no implicit prompt/answer split")
            encoded = [(api.call("/v1/tokenize", {"text": s["text"]})["tokens"], bool(s["train"])) for s in segments]
            # Nontrain segments are offline observations. A real tool agent must
            # supply observations from its environment after executing actions.
            rounds, observation = [], []
            for ids, train in encoded:
                if train:
                    if ids:
                        rounds.append((observation, ids))
                        observation = []
                    if len(rounds) == args.max_turns:
                        break
                else:
                    observation.extend(ids)
            if not rounds or not rounds[0][0]:
                continue
            if args.smoke:
                rounds[0] = (rounds[0][0][-64:], rounds[0][1])
            bound = sum(len(obs) + args.max_new_tokens for obs, _ in rounds)
            if bound > health["ctx"]:
                tqdm.write(f"skip row {line_number}: full history bound {bound} > ctx {health['ctx']}")
                continue
            histories = [[] for _ in range(args.group_size)]
            turns = [[] for _ in histories]
            rewards = [0.0 for _ in histories]
            cached = prefilled = generated = 0
            rollout_seconds = 0.0
            started = time.monotonic()
            for turn_index, (observation, reference) in enumerate(rounds):
                for history, sample_turns in zip(histories, turns):
                    history.extend(observation)
                    if observation:
                        sample_turns.append({"tokens": observation, "loss_mask": [0] * len(observation)})
                result = api.call("/v1/completions", {
                    "policy_version": version, "prompts": histories,
                    "max_tokens": args.max_new_tokens, "temperature": args.temperature,
                    "seed": args.seed + bar.n * args.max_turns + turn_index,
                })
                assert result["policy_version"] == version
                cached += result["cached_prompt_tokens"]
                prefilled += result["prefill_tokens"]
                generated += result["generated_tokens"]
                rollout_seconds += result["seconds"]
                for i, completion in enumerate(result["completions"]):
                    ids = completion["tokens"]
                    assert len(ids) == len(completion["logprobs"])
                    rewards[i] += float(judge(ids, reference)) / len(rounds)
                    histories[i].extend(ids)
                    turns[i].append({"tokens": ids, "loss_mask": [1] * len(ids)})
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            advantages = [(r - mean) / max(std, 1e-8) for r in rewards]
            synthetic = args.smoke and std < 1e-8
            if synthetic:
                advantages = [1.0 if i % 2 == 0 else -1.0 for i in range(args.group_size)]
            train = api.call("/v1/train", {
                "policy_version": version, "temperature": args.temperature,
                "samples": [{"turns": t, "advantages": a} for t, a in zip(turns, advantages)],
            })
            if train["updated"]:
                published = api.call("/v1/publish", {"expected_step": train["step"], "save": args.save})
                version = published["policy_version"]
                updates += 1
            elapsed = time.monotonic() - started
            report = dict(row=line_number, reward=mean, reward_std=std, loss=train["loss"],
                          grad_norm=train["grad_norm"], learning_rate=train["learning_rate"],
                          step=train["step"], updated=train["updated"],
                          synthetic_advantages=synthetic, generated_tokens=generated,
                          turns=len(rounds), trained_tokens=train["trained_tokens"], total_tokens=train["total_tokens"],
                          rollout_tokens_per_second=generated / max(rollout_seconds, 1e-9),
                          seconds=elapsed, cache_hit_rate=cached / max(1, cached + prefilled),
                          policy_version=version)
            metrics.write(json.dumps(report, allow_nan=False) + "\n")
            metrics.flush()
            bar.set_postfix(reward=f"{mean:.3f}", std=f"{std:.3f}", loss=f"{train['loss']:.4f}",
                            grad=f"{train['grad_norm']:.3g}", lr=f"{train['learning_rate']:.1g}", step=train["step"],
                            cache=f"{report['cache_hit_rate']:.0%}", tok_s=f"{generated/elapsed:.1f}")
            bar.update()
    print(f"Completed {updates} optimizer updates; policy={version}; metrics={args.metrics}")


if __name__ == "__main__":
    main()
