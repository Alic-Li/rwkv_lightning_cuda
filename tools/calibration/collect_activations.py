#!/usr/bin/env python3
"""Collect per-input-channel E[x²] for native RWKV-7 linear tensors."""
import argparse
from pathlib import Path

import torch

from model import add_model_args, setup
from quantization import save_json


class Moments:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def __call__(self, name, x):
        flat = x.detach().float().reshape(-1, x.shape[-1])
        total = flat.square().sum(0)
        self.sums[name] = self.sums.get(name, 0) + total
        self.counts[name] = self.counts.get(name, 0) + len(flat)

    def values(self):
        return {name: (value / self.counts[name]).cpu() for name, value in self.sums.items()}


@torch.no_grad()
def collect(model, data, batch_size):
    recorder = Moments()
    model.hook = recorder
    try:
        for batch in data.split(batch_size):
            hidden = model.hidden(batch)
            # Only the head input is needed, avoid materializing full-vocabulary logits.
            recorder("head.weight", hidden)
    finally:
        model.hook = None
    return recorder.values()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        model, data = setup(args)
        stats = collect(model, data, args.batch_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)
    save_json(args.output + ".json", {"run": vars(args), "tensors": len(stats)})
    print(f"saved {len(stats)} activation vectors to {args.output}")


if __name__ == "__main__":
    main()
