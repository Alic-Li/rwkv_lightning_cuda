#!/usr/bin/env python3
"""Run the supported native CUDA formats against one FP16 baseline.

JSON config: source, w8a16, sb6_w4, sb6_w6, english, multilingual (file paths).
Outputs raw logs/curves plus summary.json. GPU runs are deliberately sequential.
"""
import argparse
import csv
import datetime
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--build", type=Path, default=Path("build"))
    p.add_argument("--vocab", default="assets/rwkv_vocab_v20230424.txt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--stage", choices=("all", "performance", "accuracy", "summarize"), default="all")
    a = p.parse_args()
    if a.rounds < 1:
        p.error("rounds must be positive")
    cfg = json.loads(a.config.read_text())
    models = {"fp16": cfg["source"], **{k: cfg[k] for k in ("w8a16", "sb6_w4", "sb6_w6")}}
    for path in [*models.values(), cfg["english"], cfg["multilingual"], a.vocab]:
        if not Path(path).is_file():
            p.error(f"missing input: {path}")
    a.output.mkdir(parents=True, exist_ok=True)
    cache = a.output / "w8a16.tune"

    def run(name, command):
        print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {name}", flush=True)
        (a.output / f"{name}.command.json").write_text(json.dumps([str(x) for x in command], indent=2))
        with (a.output / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    if a.stage in ("all", "performance"):
        (a.output / "gpu.txt").write_text(subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv"], text=True))
        names = list(models)
        for round_index in range(a.rounds):
            order = names[round_index % 4:] + names[:round_index % 4]
            for name in order:
                base = [a.build / "rwkv_model_bench", models[name], "--cmix-sparse", "no-fc",
                        "--tune-cache", cache]
                run(f"perf_r{round_index}_{name}_decode", base + ["--batches", "1,2,4,8,16,32,64"])
                run(f"perf_r{round_index}_{name}_prefill", base + ["--batches", "1", "--prefill-tokens", "128,512,1024"])
    if a.stage in ("all", "accuracy"):
        for corpus, length in (("english", 4096), ("multilingual", 16384)):
            cmd = [a.build / "rwkv_logit_mse_curve", cfg["source"], cfg["source"], a.vocab,
                   "--text", cfg[corpus], "--ppl", "--ppl-seq-len", str(length),
                   "--ppl-stride", "128", "--output-prefix", a.output / corpus,
                   "--tune-cache", cache]
            for name, path in models.items():
                if name != "fp16":
                    cmd += ["--candidate", f"{name}={path}"]
            run(corpus, cmd)
    result = {"config": cfg, "rounds": a.rounds, "model_bytes": {
        name: Path(path).stat().st_size for name, path in models.items()}, "performance": [], "accuracy": []}
    for name in models:
        samples = {}
        for i in range(a.rounds):
            for kind in ("decode", "prefill"):
                path = a.output / f"perf_r{i}_{name}_{kind}.log"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    m = re.fullmatch(r"batch=(\d+) decode_ms=([\d.]+) tok_s=([\d.]+)", line)
                    if m and kind == "decode":
                        samples.setdefault((kind, int(m[1]), 1), []).append(float(m[2]))
                    m = re.fullmatch(r"batch=(\d+) prefill_tokens=(\d+) prefill_ms=([\d.]+) ms_per_row=([\d.]+)", line)
                    if m and kind == "prefill":
                        samples.setdefault((kind, int(m[1]), int(m[2])), []).append(float(m[3]))
        for (kind, batch, length), ms in samples.items():
            if len(ms) != a.rounds:
                raise ValueError(f"incomplete performance samples: {name} {kind} {batch} {length}")
            median = statistics.median(ms)
            result["performance"].append(dict(model=name, kind=kind, batch=batch, tokens=length,
                latency_ms=median, tok_s=batch*length*1000/median, round_ms=ms))
    for corpus, length in (("english", 4096), ("multilingual", 16384)):
        reference = None
        for name in models:
            prefix = a.output / (corpus + ("" if name == "fp16" else f".{name}"))
            ppl_path = Path(str(prefix) + ".ppl.csv")
            if not ppl_path.exists():
                continue
            chunks = list(csv.DictReader(ppl_path.open()))
            count = sum(int(r["token_count"]) for r in chunks)
            assert count == length, (count, length)
            ppls = [math.exp(sum(math.log(float(r[col]))*int(r["token_count"]) for r in chunks)/count)
                    for col in ("fp16_ppl", "int8_ppl")]
            if reference is not None:
                assert ppls[0] == reference, "candidates must share exactly the same FP16 baseline"
            reference = ppls[0]
            curve = list(csv.DictReader(Path(str(prefix) + ".csv").open()))[:length]
            assert len(curve) == length
            result["accuracy"].append(dict(corpus=corpus, tokens=length, model=name,
                reference_ppl=ppls[0], ppl=ppls[1], delta_ppl_percent=(ppls[1]/ppls[0]-1)*100,
                logit_rmse=math.sqrt(statistics.mean(float(r["mse"]) for r in curve)),
                top1_agreement_percent=statistics.mean(int(r["top1_equal"]) for r in curve)*100))
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {a.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
