#!/usr/bin/env python3
"""HTTP acceptance against either tiny or real weights; stdlib only.
Starts and stops its own servers; never touches existing inference processes.
"""
import argparse
import concurrent.futures
import contextlib
import copy
import http.client
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse


class API:
    def __init__(self, url): self.url = url
    def call(self, path, body=None, status=200):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                code, data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            code, data = exc.code, json.loads(exc.read())
        assert code == status, (path, code, data)
        return data


@contextlib.contextmanager
def server(args, mode, label, adapter=None, state=None, resume=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    log_path = args.work / (label + ".log")
    command = [str(args.build / f"rwkv_{mode}_tune_server"), "--model", str(args.model),
               "--vocab", str(args.vocab), "--port", str(port), "--chunk", str(args.chunk),
               "--ctx", str(getattr(args, "ctx", 512)),
               "--output", str(args.work / label),
               "--cache-capacity", str(getattr(args, "cache_capacity", 8)),
               "--cache-memory-mb", str(getattr(args, "cache_memory_mb", 4096))]
    command += ["--gemm-mode", getattr(args, "gemm_mode", "bf16")]
    if getattr(args, "default_stop", None) is not None:
        command += ["--stop", json.dumps(args.default_stop)]
    if getattr(args, "default_max_tokens", None) is not None:
        command += ["--max-tokens", str(args.default_max_tokens)]
    if adapter: command += ["--adapter", str(adapter)]
    if state: command += ["--state", str(state)]
    if resume: command += ["--resume", str(resume)]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        api = API(f"http://127.0.0.1:{port}")
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None: raise RuntimeError(log_path.read_text())
                try:
                    api.call("/health"); break
                except (urllib.error.URLError, ConnectionError):
                    if time.monotonic() > deadline: raise TimeoutError(log_path)
                    time.sleep(.2)
            yield api
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


def check_cache_health(api):
    h = api.call("/health")
    c = h["cache"]
    assert c["scope"] == "global" and c["eviction_policy"] == "lru"
    assert c["effective_capacity"] == min(c["capacity"], c["memory_limit_bytes"] // c["entry_bytes"])
    assert c["cached_prefixes"] <= c["allocated_slots"] <= c["effective_capacity"]
    assert c["used_bytes"] == c["cached_prefixes"] * c["entry_bytes"]
    assert c["allocated_bytes"] == c["allocated_slots"] * c["entry_bytes"] <= c["memory_limit_bytes"]
    assert c["free_slots"] == c["allocated_slots"] - c["cached_prefixes"]
    assert sum(p["cached_prefixes"] for p in h["policies"]) == c["cached_prefixes"]
    assert sum(p["cache_bytes"] for p in h["policies"]) == c["used_bytes"]
    refs = [p for p in h["policies"] if p["version"] == h["reference_version"]]
    assert h["reference_cached_prefixes"] == (refs[0]["cached_prefixes"] if refs else 0)
    assert h["reference_cache_bytes"] == h["reference_cached_prefixes"] * c["entry_bytes"]
    if not refs:
        assert not h["reference_cache_enabled"]
    return h


def cache_pool_acceptance(api):
    """Exercise actual state-buffer reuse, LRU touches and disabled/small budgets."""
    h = check_cache_health(api)
    capacity = h["cache"]["effective_capacity"]
    def prefix(i):
        # Valid even for the tiny vocab, disjoint keys with a shared radix edge.
        return [98, 99 + i % 20, 99 + i // 20]
    def request(i):
        return api.call("/v1/completions", dict(prompts=[prefix(i)], max_tokens=0))
    if capacity == 0:
        for _ in range(3):
            assert request(0)["cached_prompt_tokens"] == 0
        assert check_cache_health(api)["cache"]["allocated_bytes"] == 0
        return
    assert 1 <= capacity <= 100, capacity  # deliberately small test fixture pool
    for i in range(capacity):
        assert request(i)["cached_prompt_tokens"] == 0
    assert request(0)["cached_prompt_tokens"] == 3  # oldest is now MRU
    request(capacity)  # evicts key 1, or key 0 with a single slot
    if capacity > 1:
        assert request(0)["cached_prompt_tokens"] == 3
        assert request(1)["cached_prompt_tokens"] == 0
    else:
        assert request(0)["cached_prompt_tokens"] == 0
    for cycle in range(4):
        for i in range(capacity + 1):
            request(i)
        c = check_cache_health(api)["cache"]
        assert c["allocated_slots"] == capacity and c["cached_prefixes"] == capacity
        assert c["evictions"] > 0 and c["skipped_inserts"] == 0
    # Reused snapshots must still produce the same output as a cold prefill.
    request(0)
    body = dict(prompts=[prefix(0)], max_tokens=2, stop=[], seed=17)
    warm = api.call("/v1/completions", body)["completions"][0]
    assert warm["cached_prompt_tokens"] == 3
    cold = api.call("/v1/completions", dict(body, use_cache=False))["completions"][0]
    assert warm["tokens"] == cold["tokens"]
    assert max(abs(x-y) for x,y in zip(warm["logprobs"], cold["logprobs"])) < .01


def learning_rate_acceptance(args, mode):
    """Identical initial policies/batches must update differently at different LR."""
    initial_scores, updated_scores, norms = [], [], []
    base_lr = .001 if mode in ("miss", "nora") else .01
    sequence = [98, 99, 100, 101, 102, 103]
    for factor in (1, 4):
        with server(args, mode, f"{mode}-lr-{factor}", args.adapter, args.state) as api:
            version = api.call("/health")["policy_version"]
            initial_scores.append(api.call("/v1/logprobs", {"sequences": [sequence]})["logprobs"])
            result = api.call("/v1/train", dict(
                policy_version=version, learning_rate=base_lr * factor,
                samples=[dict(tokens=sequence, loss_mask=[0, 1, 1, 1, 1, 1], advantages=1.0)]))
            assert result["step"] == 1 and result["updated"]
            norms.append(result["grad_norm"])
            api.call("/v1/publish", {"expected_step": 1})
            updated_scores.append(api.call("/v1/logprobs", {"sequences": [sequence]})["logprobs"])
    assert initial_scores[0] == initial_scores[1]
    assert math.isclose(norms[0], norms[1], rel_tol=1e-6)
    assert updated_scores[0] != updated_scores[1], "request LR did not affect the actual update"


def acceptance(api):
    h = api.call("/health"); version = h["policy_version"]
    learning_rate = .001 if h["mode"] in ("miss", "nora") else .01
    assert h["backbone_loads"] == 1 and h["dtype"] == "bfloat16"
    assert h["wkv_kernel"] == "rwkv7_statepassing_clampw"
    assert h["wkv_io_dtype"] == "bfloat16" and h["wkv_state_dtype"] == "float32"
    assert h["reference_version"] == version
    assert h["default_stop"] == ["</tool_call>", "</answer>", "\n\nUser:"]
    assert h["default_max_tokens"] == 1024
    check_cache_health(api)
    strict = h["gemm_mode"] == "strict"
    logprob_tolerance = .01 if strict else .25
    def rollout(prompt, **kwargs):
        body = dict(prompts=[prompt], max_tokens=2, seed=42, stop=[], policy_version=version)
        if "stop_token_ids" in kwargs or "stop_token_sequences" in kwargs:
            body.pop("stop")
        body.update(kwargs)
        return api.call("/v1/completions", body)
    # AB -> ABCD -> ABCDEF -> ABCDEFGH; exact prefill work is 2 each time.
    base = [98, 99, 100, 101, 102, 103, 104, 105]
    for n in (2, 4, 6, 8):
        r = rollout(base[:n], max_tokens=0)
        assert r["cached_prompt_tokens"] == n - 2 and r["prefill_tokens"] == 2, r
    cached = rollout(base)
    cold = rollout(base, use_cache=False)
    assert cold["completions"][0]["truncated"]
    assert cold["completions"][0]["truncation_reason"] == "max_tokens"
    for a, b in zip(cached["completions"], cold["completions"]):
        if strict:
            assert a["tokens"] == b["tokens"]
        if a["tokens"] == b["tokens"]:
            assert max(abs(x-y) for x,y in zip(a["logprobs"],b["logprobs"])) < logprob_tolerance
    # Complete returned generation is available as the next round's prefix.
    history = base + cached["completions"][0]["tokens"]
    inc = rollout(history + [106,107], max_tokens=0)
    assert inc["cached_prompt_tokens"] == len(history) and inc["prefill_tokens"] == 2
    # Fixed-shape requests remain reproducible. Cross-shape bitwise sampling
    # is a strict-mode guarantee; BF16 distribution drift is tested in C++.
    batch = rollout(base, use_cache=False, max_tokens=3,
                    prompts=[base, base[:4], base[:6]], seed=7)
    single = rollout(base, use_cache=False, max_tokens=3, seed=7)
    if strict:
        assert batch["completions"][0]["tokens"] == single["completions"][0]["tokens"]
        assert batch["completions"][0]["logprobs"] == single["completions"][0]["logprobs"]
    repeated = rollout(base, use_cache=False, max_tokens=3,
                       prompts=[base, base[:4], base[:6]], seed=7)
    assert repeated["completions"] == batch["completions"]
    # Stops end on the token that completes the match; that token is kept.
    reference = cold["completions"][0]["tokens"]
    sequence = rollout(base, use_cache=False, stop_token_sequences=[reference[:2]])
    row = sequence["completions"][0]
    assert row["tokens"] == reference[:2] and row["finish_reason"] == "stop"
    assert row["matched_stop"] == reference[:2]
    assert not row["truncated"] and "truncation_reason" not in row
    text = api.call("/v1/detokenize", {"tokens": reference[:2]})["text"]
    # detokenize replaces incomplete UTF-8 with U+FFFD; stop strings match
    # original bytes, so replacement text is not a valid roundtrip fixture.
    if text and "\ufffd" not in text:
        # This request must write its stopped state; a cache-disabled request
        # cannot promise that a shorter, previously unseen suffix is cached.
        row = rollout(base, use_cache=True, stop=[text])["completions"][0]
        assert row["finish_reason"] == "stop" and row["matched_stop"] == text, row
        assert row["tokens"] == reference[:len(row["tokens"])]
        assert not row["truncated"]  # a hit at max_tokens wins over length
        # The complete stop token must be consumed before caching the state.
        finished_history = base + row["tokens"]
        hit = rollout(finished_history, max_tokens=0)
        assert hit["cached_prompt_tokens"] == len(finished_history)
        assert not hit["completions"][0]["truncated"]
    # A substring inside a token is not a suffix match (same as inference).
    first = api.call("/v1/detokenize", {"tokens": reference[:1]})["text"]
    if len(first) > 1 and "\ufffd" not in first and first[0] != first[-1]:
        inside = rollout(base, max_tokens=1, use_cache=False, stop=[first[0]])["completions"][0]
        assert inside["finish_reason"] == "length" and inside["tokens"] == reference[:1], inside
    api.call("/v1/completions", dict(prompts=[base], stop=[], stop_token_ids=[]), 400)
    api.call("/v1/completions", dict(prompts=[base], stop=[], stop_token_sequences=[]), 400)
    api.call("/v1/completions", dict(prompts=[base], stop=[""]), 400)
    # Per-row context capacity is a cap, not a reason to reject the batch.
    bounded = rollout(base, use_cache=False, prompts=[[98] * (h["ctx"] - 1), base], max_tokens=2)
    near_limit, ordinary = bounded["completions"]
    assert len(near_limit["tokens"]) == 1 and near_limit["max_tokens"] == 1
    assert near_limit["truncated"] and near_limit["truncation_reason"] == "context_length"
    assert len(ordinary["tokens"]) == 2 and ordinary["truncation_reason"] == "max_tokens"
    full = rollout([98] * h["ctx"], max_tokens=1)["completions"][0]
    assert full["tokens"] == [] and full["truncation_reason"] == "context_length"
    # Teacher-forced scores reproduce the sampled log probabilities.
    scored = api.call("/v1/logprobs", dict(sequences=[base + reference], policy_version=version))
    tail = scored["logprobs"][0][-len(reference):]
    assert max(abs(x-y) for x,y in zip(tail, cold["completions"][0]["logprobs"])) < logprob_tolerance
    # Full batch validation occurs before GPU work or any optimizer mutation.
    api.call("/v1/completions", {"prompts": [[98], [-1]]}, 400)
    api.call("/v1/completions", {"prompts": [[98]], "temperature": 0}, 400)
    api.call("/v1/train", {"policy_version": version, "learning_rate": learning_rate, "samples": [
        dict(tokens=[98,99], loss_mask=[0,0], advantages=1)]}, 400)
    assert api.call("/health")["step"] == 0
    api.call("/v1/detokenize", {"tokens": [129]})  # partial/invalid byte-token text must still be valid JSON
    # Concurrent requests use the same immutable policy and deterministic seeds.
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        concurrent_results = list(pool.map(lambda _: rollout(base, use_cache=False), range(3)))
    assert all(r["completions"][0]["tokens"] == cold["completions"][0]["tokens"] for r in concurrent_results)
    # Two generated turns per episode, separated by masked observations.
    episodes = [[], []]; histories = [base[:], base[:]]; score = 0.; selected = 0
    for turn in range(2):
        r = api.call("/v1/completions", dict(prompts=histories, max_tokens=3, temperature=.7,
                    seed=91+turn, stop_token_ids=[], policy_version=version))
        for i, row in enumerate(r["completions"]):
            observation = base if turn == 0 else [106,107]
            episodes[i].append(dict(tokens=observation, loss_mask=[0]*len(observation)))
            episodes[i].append(dict(tokens=row["tokens"], loss_mask=[1]*len(row["tokens"])))
            histories[i].extend(row["tokens"])
            if turn == 0: histories[i].extend([106,107])
            advantage = (1., -.5)[i]
            score -= advantage * sum(row["logprobs"])
            selected += len(row["tokens"])
    request = dict(policy_version=version, temperature=.7, learning_rate=learning_rate,
                   samples=[dict(turns=t, advantages=a) for t,a in zip(episodes,(1.,-.5))])
    missing_lr = dict(request)
    missing_lr.pop("learning_rate")
    assert "learning_rate" in api.call("/v1/train", missing_lr, 400)["error"]
    for invalid in (None, True, "0.001", [], -1, 0, 1e-20, 10.1, 1e100):
        api.call("/v1/train", dict(request, learning_rate=invalid), 400)
    assert api.call("/health")["step"] == 0
    zero = dict(request, samples=[dict(turns=t, advantages=0.) for t in episodes])
    z = api.call("/v1/train", zero)
    assert not z["updated"] and z["step"] == 0
    train = api.call("/v1/train", request)
    assert train["step"] == 1 and math.isfinite(train["grad_norm"]) and train["grad_norm"] > 0
    assert math.isclose(train["learning_rate"], learning_rate, rel_tol=1e-6)
    # Learner CE and actor log probabilities describe the same tempered policy.
    assert abs(train["loss"] - score / selected) < .02, (train["loss"], score / selected)
    assert rollout(base, use_cache=False)["completions"] == cold["completions"]
    api.call("/v1/train", request, 409)
    api.call("/v1/publish", {"expected_step": 0}, 409)
    publication = api.call("/v1/publish", {"expected_step": 1, "save": True})
    assert publication["policy_version"] != version
    assert api.call("/v1/publish", {"expected_step": 1, "save": True}) == publication
    assert rollout(base, use_cache=False)["completions"] == cold["completions"]
    api.call("/v1/train", request, 409)
    new_version = publication["policy_version"]
    new = rollout(base, policy_version=new_version, max_tokens=0)
    assert new["cached_prompt_tokens"] == 0  # no stale cache crosses publication
    changed = rollout(base, policy_version=new_version, use_cache=False)
    assert changed["completions"][0]["logprobs"] != cold["completions"][0]["logprobs"]
    request["policy_version"] = new_version
    # There is no sticky/default learning rate, even after a successful update.
    missing_lr["policy_version"] = new_version
    api.call("/v1/train", missing_lr, 400)
    assert api.call("/health")["step"] == 1
    request["learning_rate"] = learning_rate / 2
    next_step = api.call("/v1/train", request)
    assert next_step["step"] == 2
    assert math.isclose(next_step["learning_rate"], learning_rate / 2, rel_tol=1e-6)
    # Ensure reference has cached data immediately before it is retired.
    rollout(base, max_tokens=0)
    assert check_cache_health(api)["reference_cached_prefixes"] > 0
    final = api.call("/v1/publish", {"expected_step": 2, "checkpoint": True})
    assert final["checkpoint"].endswith("training")
    retired = check_cache_health(api)
    assert retired["reference_cached_prefixes"] == 0 and not retired["reference_cache_enabled"]
    assert retired["cache"]["retired_prefixes"] > 0
    # KL weights survive; old-reference completions cannot repopulate its cache.
    reference_scores = api.call("/v1/logprobs", dict(sequences=[base + reference], policy_version=version))
    assert reference_scores["logprobs"] == scored["logprobs"]
    assert rollout(base, max_tokens=0)["cached_prompt_tokens"] == 0
    assert check_cache_health(api)["reference_cache_bytes"] == 0
    checkpoints = [Path(final["directory"])]
    for step in range(3, 7):
        request["policy_version"] = final["policy_version"]
        request["learning_rate"] = learning_rate / step
        result = api.call("/v1/train", request)
        assert result["step"] == step
        assert math.isclose(result["learning_rate"], request["learning_rate"], rel_tol=1e-6)
        final = api.call("/v1/publish", {"expected_step": step, "checkpoint": True})
        check_cache_health(api)
        checkpoints.append(Path(final["directory"]))
    if h["mode"] in ("miss", "nora"):
        configs = [json.loads((path / "training" / "checkpoint.json").read_text())["config"]
                   for path in checkpoints]
        assert len(configs[0]) == (8 if h["mode"] == "nora" else 6) and "seed" not in configs[0] and "planned_steps" not in configs[0]
        assert all(config == configs[0] for config in configs), configs
    rollout_body = dict(prompts=[base], max_tokens=2, seed=42, stop_token_ids=[])
    api.call("/v1/completions", dict(rollout_body, policy_version=new_version), 409)  # retired
    # The startup policy stays available as the reference, e.g. for KL.
    assert api.call("/v1/completions", dict(rollout_body, policy_version=version, use_cache=False)) \
        ["completions"] == cold["completions"]
    print(json.dumps(dict(mode=h["mode"], first_loss=train["loss"], rollout_loss=score/selected,
                          grad_norm=train["grad_norm"], second_grad_norm=next_step["grad_norm"],
                          first_version=version, final=final), indent=2), flush=True)
    return checkpoints


def idle_connection_acceptance(api):
    """The GPU worker is silent on the socket while computing a long request.

    Check the transport's idle policy without spending a minute of GPU compute.
    Reuse the same socket explicitly, so a reconnect cannot hide a timeout.
    """
    url = urllib.parse.urlsplit(api.url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=90)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        sock = connection.sock
        assert sock is not None
        print("Checking HTTP connection survives 75 seconds without traffic...", flush=True)
        time.sleep(75)
        connection.request("GET", "/health")
        assert connection.sock is sock, "unexpected reconnect"
        response = connection.getresponse()
        assert response.status == 200
        response.read()
    finally:
        connection.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--state", type=Path)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=7)
    p.add_argument("--gemm-mode", choices=["bf16", "strict"], default="bf16")
    p.add_argument("--check-idle-timeout", action="store_true", help="also test a 75-second silent HTTP connection per mode")
    p.add_argument("--modes", nargs="+", choices=["miss", "nora", "state"], default=["miss", "state"])
    args = p.parse_args(); args.work.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        learning_rate_acceptance(args, mode)
        for label, count, memory_mb in (("bounded-cache", 10000, 1),
                                       ("single-slot-cache", 1, 4096),
                                       ("disabled-cache", 8, 0)):
            cache_args = copy.copy(args)
            cache_args.cache_capacity = count
            cache_args.cache_memory_mb = memory_mb
            with server(cache_args, mode, f"{mode}-{label}", args.adapter, args.state) as api:
                cache_pool_acceptance(api)
        with server(args, mode, mode, args.adapter, args.state) as api:
            if args.check_idle_timeout:
                idle_connection_acceptance(api)
            checkpoints = acceptance(api)
        exported = checkpoints[-1]
        # Every checkpoint from the same writer must resume in a fresh process.
        for step, path in enumerate(checkpoints[:-1], 2):
            with server(args, mode, f"{mode}-resume-{step}", resume=path) as api:
                h = api.call("/health")
                assert h["step"] == step and h["published_step"] == step, h
        # Exercise both cold-start files produced by this process, too.
        adapter = exported / "adapter.pth"
        with server(args, mode, mode + "-reload", adapter if adapter.exists() else None, exported / "state.pth") as api:
            h = api.call("/health")
            assert h["step"] == 0  # a cold start starts new Adam moments
            # Select an actual decoded suffix, then configure it as the
            # resumed Server's default. This tests inheritance end to end.
            stop_fixture = None
            for seed in range(8):
                probe = api.call("/v1/completions", dict(prompts=[[98,99]], max_tokens=2,
                                                       seed=seed, stop=[]))["completions"][0]
                for count in range(1, 3):
                    ids = probe["tokens"][:count]
                    text = api.call("/v1/detokenize", dict(tokens=ids))["text"]
                    if text and "\ufffd" not in text and len(text.encode("utf-8")) <= 256:
                        stop_fixture = (seed, ids, text)
                        break
                if stop_fixture:
                    break
            assert stop_fixture is not None, "no valid UTF-8 default-stop fixture"
        # --resume restores the optimizer state and step of the checkpoint.
        resume_args = copy.copy(args)
        resume_args.default_stop = [stop_fixture[2]]
        resume_args.default_max_tokens = 2
        with server(resume_args, mode, mode + "-resume", resume=exported) as api:
            h = api.call("/health")
            assert h["step"] == 6 and h["published_step"] == 6, h
            assert h["default_stop"] == resume_args.default_stop and h["default_max_tokens"] == 2
            body = dict(prompts=[[98,99]], seed=stop_fixture[0])
            row = api.call("/v1/completions", body)["completions"][0]
            assert row["tokens"] == stop_fixture[1] and row["matched_stop"] == stop_fixture[2], row
            assert row["finish_reason"] == "stop" and not row["truncated"]
            disabled = api.call("/v1/completions", dict(body, stop=[]))["completions"][0]
            assert len(disabled["tokens"]) == 2 and disabled["truncation_reason"] == "max_tokens"
            numeric = api.call("/v1/completions", dict(body, stop_token_ids=[row["tokens"][0]]))["completions"][0]
            assert numeric["tokens"] == row["tokens"][:1] and numeric["finish_reason"] == "stop"
        if mode in ("miss", "nora"):
            # Old servers polluted all but their first checkpoint with nulls.
            metadata = exported / "training" / "checkpoint.json"
            original = metadata.read_text()
            legacy = json.loads(original)
            legacy["config"].update(seed=None, planned_steps=None)
            try:
                metadata.write_text(json.dumps(legacy))
                with server(args, mode, mode + "-resume-legacy", resume=exported) as api:
                    h = api.call("/health")
                    assert h["step"] == 6 and h["published_step"] == 6, h
            finally:
                metadata.write_text(original)
    print("HTTP rollout/train/publish/cache/multi-turn/cold-start acceptance passed")


if __name__ == "__main__": main()
