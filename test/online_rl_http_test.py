#!/usr/bin/env python3
"""HTTP acceptance against either tiny or real weights; stdlib only.
Starts and stops its own servers; never touches existing inference processes.
"""
import argparse
import concurrent.futures
import contextlib
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
def server(args, mode, label, adapter=None, state=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    log_path = args.work / (label + ".log")
    command = [str(args.build / f"rwkv_{mode}_tune_server"), "--model", str(args.model),
               "--vocab", str(args.vocab), "--port", str(port), "--chunk", str(args.chunk),
               "--ctx", str(getattr(args, "ctx", 512)), "--lr", ".001" if mode == "miss" else ".01",
               "--output", str(args.work / label), "--cache-capacity", "8"]
    if adapter: command += ["--adapter", str(adapter)]
    if state: command += ["--state", str(state)]
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


def acceptance(api):
    h = api.call("/health"); version = h["policy_version"]
    assert h["backbone_loads"] == 1 and h["dtype"] == "bfloat16"
    def rollout(prompt, **kwargs):
        body = dict(prompts=[prompt], max_tokens=2, seed=42, stop_token_ids=[], policy_version=version)
        body.update(kwargs)
        return api.call("/v1/completions", body)
    # AB -> ABCD -> ABCDEF -> ABCDEFGH; exact prefill work is 2 each time.
    base = [98, 99, 100, 101, 102, 103, 104, 105]
    for n in (2, 4, 6, 8):
        r = rollout(base[:n], max_tokens=0)
        assert r["cached_prompt_tokens"] == n - 2 and r["prefill_tokens"] == 2, r
    cached = rollout(base)
    cold = rollout(base, use_cache=False)
    for a, b in zip(cached["completions"], cold["completions"]):
        assert a["tokens"] == b["tokens"]
        assert max(abs(x-y) for x,y in zip(a["logprobs"],b["logprobs"])) < .01
    # Complete returned generation is available as the next round's prefix.
    history = base + cached["completions"][0]["tokens"]
    inc = rollout(history + [106,107], max_tokens=0)
    assert inc["cached_prompt_tokens"] == len(history) and inc["prefill_tokens"] == 2
    # Full batch validation occurs before GPU work or any optimizer mutation.
    api.call("/v1/completions", {"prompts": [[98], [-1]]}, 400)
    api.call("/v1/completions", {"prompts": [[98]], "temperature": 0}, 400)
    api.call("/v1/train", {"policy_version": version, "samples": [
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
    request = dict(policy_version=version, temperature=.7,
                   samples=[dict(turns=t, advantages=a) for t,a in zip(episodes,(1.,-.5))])
    zero = dict(request, samples=[dict(turns=t, advantages=0.) for t in episodes])
    z = api.call("/v1/train", zero)
    assert not z["updated"] and z["step"] == 0
    train = api.call("/v1/train", request)
    assert train["step"] == 1 and math.isfinite(train["grad_norm"]) and train["grad_norm"] > 0
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
    next_step = api.call("/v1/train", request)
    assert next_step["step"] == 2
    final = api.call("/v1/publish", {"expected_step": 2, "save": True})
    rollout_body = dict(prompts=[base], max_tokens=2, seed=42, stop_token_ids=[], policy_version=version)
    api.call("/v1/completions", rollout_body, 409)  # old version retired
    print(json.dumps(dict(mode=h["mode"], first_loss=train["loss"], rollout_loss=score/selected,
                          grad_norm=train["grad_norm"], second_grad_norm=next_step["grad_norm"],
                          first_version=version, final=final), indent=2), flush=True)
    return Path(final["directory"])


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
    p.add_argument("--check-idle-timeout", action="store_true", help="also test a 75-second silent HTTP connection per mode")
    p.add_argument("--modes", nargs="+", choices=["miss", "state"], default=["miss", "state"])
    args = p.parse_args(); args.work.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        with server(args, mode, mode, args.adapter, args.state) as api:
            if args.check_idle_timeout:
                idle_connection_acceptance(api)
            exported = acceptance(api)
        # Exercise both cold-start files produced by this process, too.
        adapter = exported / "adapter.pth"
        with server(args, mode, mode + "-reload", adapter if adapter.exists() else None, exported / "state.pth") as api:
            h = api.call("/health")
            assert h["step"] == 0  # a cold start starts new Adam moments
            api.call("/v1/completions", dict(prompts=[[98,99]], max_tokens=1))
    print("HTTP rollout/train/publish/cache/multi-turn/cold-start acceptance passed")


if __name__ == "__main__": main()
