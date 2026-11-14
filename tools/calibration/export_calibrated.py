#!/usr/bin/env python3
"""GPU SB6 AW search followed by the project's C++ native archive writer."""
import argparse
import json
import subprocess
from pathlib import Path

import torch
from quantization import load_checkpoint, quantize, resolve_policy, save_json, write_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--default", choices=("w8a16", "w4a16_asym", "w6a16_asym"), default="w4a16_asym")
    parser.add_argument("--activation-stats", help="flat name -> E[x²] .pth; omit for uniform-importance SB6 search")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--quantizer", default="build/rwkv_quantize")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    weights = load_checkpoint(args.model)
    policy = resolve_policy(weights, json.loads(Path(args.policy).read_text()) if args.policy else None, args.default)
    stats = torch.load(args.activation_stats, weights_only=True) if args.activation_stats else None
    directory = Path(args.output_dir)
    manifest = {}
    with torch.inference_mode():
        for name, tier in policy.items():
            if tier in ("fp16", "bf16"):
                continue
            nu = stats[name].to(args.device) if stats is not None and tier != "w8a16" else None
            codes, scales = quantize(weights[name].to(args.device), tier, nu)
            manifest[name] = write_payload(directory, name, tier, codes, scales)
            print(f"{name}: {tier}", flush=True)
    save_json(directory / "manifest.json", manifest)
    save_json(directory / "policy.json", policy)
    save_json(directory / "run.json", vars(args))
    subprocess.run([args.quantizer, "--policy", str(directory / "policy.json"), "--calibrated",
                    str(directory / "manifest.json"), args.model, args.export], check=True)


if __name__ == "__main__":
    main()
