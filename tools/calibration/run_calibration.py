#!/usr/bin/env python3
"""Run local CUDA calibration commands and preserve machine-readable output."""

import argparse
import pathlib
import subprocess
import time


def run(command):
    print("$", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="build/rwkv_model_bench")
    parser.add_argument("--fp16-model", required=True)
    parser.add_argument("--int8-model", required=True)
    parser.add_argument("--bandwidth", default="build/rwkv_calibration_bandwidth")
    parser.add_argument("--report", default="tmp/W8A16_REPORT.md")
    args = parser.parse_args()
    lines = ["# W8A16 calibration report", "", f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for label, model in (("FP16", args.fp16_model), ("W8A16", args.int8_model)):
        result = subprocess.run([args.bench, model], text=True, capture_output=True)
        lines.extend([f"## {label}", "", "```text", result.stdout.rstrip(), "```", ""])
        if result.returncode != 0:
            lines.append(f"Command failed with exit code {result.returncode}: {result.stderr.strip()}")
            lines.append("")
    bandwidth = subprocess.run([args.bandwidth], text=True, capture_output=True)
    lines.extend(["## Device bandwidth", "", "```text", bandwidth.stdout.rstrip(), "```", ""])
    lines.extend([
        "## HTTP agreement",
        "",
        "Run `python3 tools/calibration/check_int8_agreement.py` with FP16 and INT8 servers",
        "after starting the rebuilt Drogon-backed HTTP target. The target is omitted only",
        "when Drogon is unavailable at configure time.",
    ])
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
