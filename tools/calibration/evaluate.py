#!/usr/bin/env python3
"""Teacher-forced FP32 PPL/KL of weights decoded from this project's RWKVQ."""
import argparse
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

from model import add_model_args, setup
from quantization import dequantize, save_json


def read_archive(path):
    """Stream the existing native RWKVQ v1 layout; no external container support."""
    with open(path, 'rb') as stream:
        def read(n):
            data = stream.read(n)
            if len(data) != n:
                raise ValueError("truncated RWKVQ")
            return data
        if read(8) != b'RWKVQ\0\1\0':
            raise ValueError("not a native RWKVQ archive")
        version, count = struct.unpack('<II', read(8))
        if version != 1:
            raise ValueError("unsupported archive version")
        seen = set()
        for _ in range(count):
            size, dtype, rank, group, numel = struct.unpack('<IBBHQ', read(16))
            if size == 0 or size > 1 << 20 or rank > 16 or dtype not in (0, 1, 2, 4, 6):
                raise ValueError("invalid tensor header")
            name = read(size).decode()
            shape = struct.unpack('<' + 'q' * rank, read(8 * rank))
            if name in seen or min(shape, default=0) < 0 or math.prod(shape) != numel:
                raise ValueError("invalid tensor shape or duplicate name")
            seen.add(name)
            if dtype >= 4:
                if len(shape) != 2 or group != 32 or min(shape) <= 0:
                    raise ValueError("invalid SB6 header")
                n, k = shape
                data = read(n*((k+31)//32)*(16+4*(dtype-4)))
                size, = struct.unpack('<Q', read(8))
                if size != n*((k+255)//256)*8:
                    raise ValueError("invalid SB6 metadata count")
                import sb6
                q, parts = sb6.unpack(data, read(size*2), n, k, dtype)
                yield name, sb6.dequantize(q, parts), dtype, group
            elif dtype >= 2:
                if len(shape) != 2 or group != 0:
                    raise ValueError("invalid W8 metadata")
                n, k = shape
                raw = np.frombuffer(read(numel), dtype=np.int8).copy()
                scales_count, = struct.unpack('<Q', read(8))
                if scales_count != n:
                    raise ValueError("invalid scale count")
                scales = torch.from_numpy(np.frombuffer(read(2 * scales_count), dtype='<f2').copy())
                if not torch.isfinite(scales).all() or (scales <= 0).any():
                    raise ValueError("invalid FP16 scales")
                q = torch.from_numpy(raw.reshape(shape))
                yield name, dequantize(q, scales, 'w8a16'), dtype, group
            else:
                bits = torch.from_numpy(np.frombuffer(read(2 * numel), dtype='<u2').copy())
                value = bits.view(torch.bfloat16 if dtype == 0 else torch.float16).reshape(shape).float()
                yield name, value, dtype, group
        if stream.read(1):
            raise ValueError("unexpected trailing archive data")


@torch.no_grad()
def evaluate(model, data, archive, batch_size):
    references, reference_nll = [], 0.
    count = len(data) * (data.shape[1] - 1)
    for batch in data.split(batch_size):
        logits = model.logits(model.hidden(batch))[:, :-1]
        reference_nll += F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten().to(model.device), reduction='sum').item()
        references.append(logits.log_softmax(-1).cpu())
    seen = set()
    for name, weight, _, _ in read_archive(archive):
        if name not in model.weights or weight.shape != model.weights[name].shape:
            raise ValueError(f"archive/checkpoint mismatch: {name}")
        model.weights[name] = weight.to(model.device)
        seen.add(name)
    if seen != set(model.weights):
        raise ValueError("archive missing tensors")
    nll, kl = 0., 0.
    for batch, clean in zip(data.split(batch_size), references):
        logits = model.logits(model.hidden(batch))[:, :-1]
        nll += F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten().to(model.device), reduction='sum').item()
        logp, logq = clean.to(model.device), logits.log_softmax(-1)
        kl += (logp.exp() * (logp - logq)).double().sum().item()
    return {"tokens": count, "reference_ppl": math.exp(reference_nll / count), "quantized_ppl": math.exp(nll / count),
            "ppl_change_percent": 100 * math.expm1((nll - reference_nll) / count), "kl": kl / count,
            "backend": "PyTorch FP32 native archive dequantization; verify CUDA separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        model, data = setup(args)
        result = evaluate(model, data, args.archive, args.batch_size)
    result['run'] = vars(args)
    save_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
