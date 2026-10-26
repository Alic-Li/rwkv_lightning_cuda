#!/usr/bin/env python3
"""Minimal rollout -> Python judge -> signed GRPO -> publish HTTP client.
Only tqdm is required. Example: uv run --with tqdm tools/online_rl.py --help
"""
import argparse
from collections import Counter
import importlib
import http.client
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


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


def parse_stops(value):
    try:
        stops = json.loads(value)
        valid = (isinstance(stops, list) and len(stops) <= 16 and
                 all(isinstance(s, str) and 1 <= len(s.encode("utf-8")) <= 256 for s in stops))
    except (json.JSONDecodeError, UnicodeEncodeError) as exc:
        raise argparse.ArgumentTypeError("--stop requires a JSON array of strings") from exc
    if not valid:
        raise argparse.ArgumentTypeError("--stop requires up to 16 nonempty strings, each at most 256 UTF-8 bytes")
    return stops


def collect_rollouts(api, rounds, *, group_size, version, max_tokens, ctx,
                     temperature, seed, judge, stop=None):
    """Replay observations only for live episodes; a length cap ends an episode.

    Tokens completing a text stop remain in history and the training mask.
    max_tokens is a per-turn safety cap, never a requested response length.
    """
    histories = [[] for _ in range(group_size)]
    turns = [[] for _ in histories]
    rewards = [0.0 for _ in histories]
    logprob_sums = [0.0 for _ in histories]
    active = [True for _ in histories]
    truncated = [False for _ in histories]
    reasons = [None for _ in histories]
    matched = [None for _ in histories]
    truncation_reasons = [None for _ in histories]
    completed_turns = [0 for _ in histories]
    cached = prefilled = generated = 0
    seconds = 0.0
    for turn_index, (observation, reference) in enumerate(rounds):
        indices = []
        for i in range(group_size):
            if not active[i]:
                continue
            if len(histories[i]) + len(observation) >= ctx:
                # Do not append an observation that leaves no generation
                # capacity, or exceed the learner's full-history limit.
                active[i] = False
                truncated[i] = True
                reasons[i] = "length"
                matched[i] = None
                truncation_reasons[i] = "context_length"
                continue
            histories[i].extend(observation)
            if observation:
                turns[i].append({"tokens": list(observation), "loss_mask": [0] * len(observation)})
            indices.append(i)
        if not indices:
            break
        request = {"policy_version": version, "prompts": [histories[i] for i in indices],
                   "max_tokens": max_tokens, "temperature": temperature,
                   "seed": seed + turn_index}
        if stop is not None:
            request["stop"] = stop
        result = api.call("/v1/completions", request)
        assert result["policy_version"] == version
        assert len(result["completions"]) == len(indices)
        cached += result["cached_prompt_tokens"]
        prefilled += result["prefill_tokens"]
        generated += result["generated_tokens"]
        seconds += result["seconds"]
        for i, completion in zip(indices, result["completions"]):
            ids = completion["tokens"]
            assert ids and len(ids) == len(completion["logprobs"])
            rewards[i] += float(judge(ids, reference)) / len(rounds)
            logprob_sums[i] += sum(completion["logprobs"])
            histories[i].extend(ids)
            turns[i].append({"tokens": ids, "loss_mask": [1] * len(ids)})
            completed_turns[i] += 1
            reasons[i] = completion["finish_reason"]
            matched[i] = completion.get("matched_stop")
            truncated[i] = completion.get("truncated", reasons[i] == "length")
            if truncated[i]:
                active[i] = False
                truncation_reasons[i] = completion.get("truncation_reason", "max_tokens")
    return dict(turns=turns, rewards=rewards, logprob_sums=logprob_sums,
                cached=cached, prefilled=prefilled, generated=generated, seconds=seconds,
                finish_reasons=reasons, matched_stops=matched, truncated=truncated,
                truncation_reasons=truncation_reasons, completed_turns=completed_turns)


def main():
    from tqdm import tqdm

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8848")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--start-row", type=int, default=1, help="first dataset line to process (1-based); earlier lines are skipped")
    p.add_argument("--timeout", type=float, default=3600, help="HTTP response timeout in seconds")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--max-tokens", "--max-new-tokens", dest="max_new_tokens", type=int,
                   help="per-turn safety cap for repetition (default: server default_max_tokens, normally 1024)")
    p.add_argument("--stop", type=parse_stops,
                   help='JSON array of stop strings; default: server strings; [] explicitly disables stops')
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, required=True,
                   help="learning rate sent in every train request; custom loops may schedule it per step")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--judge", help="module:function taking (generated_token_ids, reference_token_ids), returning reward")
    p.add_argument("--metrics", type=Path, default=Path("online-rl-metrics.jsonl"))
    p.add_argument("--save", action="store_true", help="export adapter/state on each publish")
    p.add_argument("--smoke", action="store_true", help="plumbing test: shorten initial context to 64 tokens; use synthetic +/- advantages if rewards tie")
    args = p.parse_args()
    if not math.isfinite(args.lr) or not 1e-12 <= args.lr <= 10.0:
        p.error("lr must be finite and in 1e-12..10")
    if args.start_row < 1 or not 0 < args.timeout < float("inf"):
        p.error("start-row and timeout must be positive and finite")
    if min(args.steps, args.group_size, args.max_turns) < 1 or args.group_size < 2:
        p.error("positive sizes and group-size >= 2 required")
    if args.max_new_tokens is not None and not 1 <= args.max_new_tokens <= 1048576:
        p.error("max-tokens must be in 1..1048576")
    judge = token_f1
    if args.judge:
        module, name = args.judge.rsplit(":", 1)
        judge = getattr(importlib.import_module(module), name)
    api = Client(args.url, args.timeout)
    health = api.call("/health")
    if "default_stop" not in health or "default_max_tokens" not in health:
        p.error("server lacks default string stops; rebuild/restart the RL server before running this client")
    if args.max_new_tokens is None:
        args.max_new_tokens = health["default_max_tokens"]
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
            if len(rounds[0][0]) >= health["ctx"]:
                tqdm.write(f"skip row {line_number}: initial prompt leaves no generation capacity in ctx {health['ctx']}")
                continue
            started = time.monotonic()
            rollout = collect_rollouts(
                api, rounds, group_size=args.group_size, version=version,
                max_tokens=args.max_new_tokens, ctx=health["ctx"],
                temperature=args.temperature, seed=args.seed + bar.n * args.max_turns,
                judge=judge, stop=args.stop)
            turns, rewards = rollout["turns"], rollout["rewards"]
            cached, prefilled, generated = rollout["cached"], rollout["prefilled"], rollout["generated"]
            rollout_seconds = rollout["seconds"]
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            advantages = [(r - mean) / max(std, 1e-8) for r in rewards]
            synthetic = args.smoke and std < 1e-8
            if synthetic:
                advantages = [1.0 if i % 2 == 0 else -1.0 for i in range(args.group_size)]
            train = api.call("/v1/train", {
                "policy_version": version, "temperature": args.temperature,
                "learning_rate": args.lr,
                "samples": [{"turns": t, "advantages": a} for t, a in zip(turns, advantages)],
            })
            publish_seconds = 0.0
            if train["updated"]:
                publish_started = time.monotonic()
                published = api.call("/v1/publish", {"expected_step": train["step"], "save": args.save})
                publish_seconds = time.monotonic() - publish_started
                version = published["policy_version"]
                updates += 1
            elapsed = time.monotonic() - started
            rollout_loss = -sum(a * lp for a, lp in zip(advantages, rollout["logprob_sums"])) / train["trained_tokens"]
            report = dict(row=line_number, reward=mean, reward_std=std, loss=train["loss"],
                          rollout_loss=rollout_loss, loss_abs_error=abs(train["loss"] - rollout_loss),
                          grad_norm=train["grad_norm"], learning_rate=train["learning_rate"],
                          step=train["step"], updated=train["updated"],
                          synthetic_advantages=synthetic, generated_tokens=generated,
                          max_tokens=args.max_new_tokens,
                          stop=health["default_stop"] if args.stop is None else args.stop,
                          finish_reasons=rollout["finish_reasons"], matched_stops=rollout["matched_stops"],
                          truncated_samples=sum(rollout["truncated"]),
                          truncation_reasons=rollout["truncation_reasons"], completed_turns=rollout["completed_turns"],
                          turns=len(rounds), trained_tokens=train["trained_tokens"], total_tokens=train["total_tokens"],
                          rollout_tokens_per_second=generated / max(rollout_seconds, 1e-9),
                          rollout_seconds=rollout_seconds, train_seconds=train["seconds"],
                          publish_seconds=publish_seconds,
                          train_tokens_per_second=train["total_tokens"] / max(train["seconds"], 1e-9) if train["updated"] else 0.0,
                          seconds=elapsed, cache_hit_rate=cached / max(1, cached + prefilled),
                          policy_version=version)
            metrics.write(json.dumps(report, allow_nan=False) + "\n")
            metrics.flush()
            bar.set_postfix(reward=f"{mean:.3f}", std=f"{std:.3f}", loss=f"{train['loss']:.4f}",
                            grad=f"{train['grad_norm']:.3g}", lr=f"{train['learning_rate']:.1g}", step=train["step"],
                            cache=f"{report['cache_hit_rate']:.0%}", tok_s=f"{generated/elapsed:.1f}",
                            trunc=report["truncated_samples"])
            bar.update()
    print(f"Completed {updates} optimizer updates; policy={version}; metrics={args.metrics}")


if __name__ == "__main__":
    main()
