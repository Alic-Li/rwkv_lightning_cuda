"""Native RWKVQ quantization math. No inference kernels or external formats."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch

TIERS = ("w8a16", "bf16")
ASYM_TIERS = ("w4a16_asym", "w6a16_asym")
ALL_TIERS = TIERS + ASYM_TIERS
PROTECTED = "blocks.0.att.output.weight"
MIN_SCALE = 2.0 ** -24


def is_target(name):
    return name == "head.weight" or re.fullmatch(
        r"blocks\.\d+\.(att\.(receptance|key|value|output)|ffn\.(key|value))\.weight", name
    ) is not None


def load_checkpoint(path):
    weights = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(weights, dict) or "emb.weight" not in weights:
        raise ValueError("expected a flat RWKV-7 state dictionary")
    return weights


def resolve_policy(weights, policy=None, default="w4a16_asym"):
    if default not in ALL_TIERS + ("fp16",):
        raise ValueError(f"invalid default precision: {default}")
    policy = policy or {}
    for name, tier in policy.items():
        if name not in weights or tier not in ALL_TIERS + ("fp16",):
            raise ValueError(f"unknown tensor or precision: {name}: {tier}")
        if not is_target(name) and tier not in ("bf16", "fp16"):
            raise ValueError(f"embedding/LoRA/non-linear tensor cannot be quantized: {name}")
    result = {}
    for name, w in weights.items():
        tier = default if is_target(name) else ("fp16" if w.dtype == torch.float16 else "bf16")
        if name == PROTECTED and default.startswith("w4"):
            tier = "bf16"
        result[name] = policy.get(name, tier)
    return result


def tensor_bytes(name, shape, tier):
    """Exact native record size including name, shape, scale count and odd-K padding."""
    header = 16 + len(name.encode()) + 8 * len(shape)
    count = math.prod(shape)
    if tier in ("bf16", "fp16"):
        return header + 2 * count
    n, k = shape
    if tier == "w8a16":
        return header + count + 8 + 2 * n
    if tier in ASYM_TIERS:
        bits = int(tier[1])
        return header + n*((k+31)//32)*(16+4*(bits-4)) + 8 + n*((k+255)//256)*16
    raise ValueError(f"unsupported tier: {tier}")


def _validate(w, nu):
    if w.ndim != 2 or min(w.shape) <= 0 or not torch.isfinite(w).all():
        raise ValueError("weights must be a finite nonempty matrix")
    if nu is not None and (nu.shape != (w.shape[1],) or not torch.isfinite(nu).all() or (nu < 0).any()):
        raise ValueError("second moments must be finite, nonnegative and match K")


@torch.no_grad()
def quantize(w, tier, nu=None, row_chunk=128):
    """Return signed integer codes + FP16 scales, or the requested floating tensor."""
    _validate(w, nu)
    if tier in ("bf16", "fp16"):
        result = w.to(torch.bfloat16 if tier == "bf16" else torch.float16)
        if not torch.isfinite(result).all():
            raise ValueError("floating weight overflow")
        return result, None
    if tier in ASYM_TIERS:
        import sb6
        return sb6.quantize(w, int(tier[1]), nu)
    if tier not in TIERS:
        raise ValueError(f"unsupported tier: {tier}")
    w = w.float()
    if tier == "w8a16":
        # Keep the existing per-row, [-127,127] W8A16 grid exactly.
        unrounded = w.abs().amax(1) / 127
        scales = unrounded.half()
        # Existing C++ float_to_f16_bits rounds halfway cases up. Preserve that
        # W8 grid rather than silently switching it to PyTorch's ties-to-even.
        upper = torch.nextafter(scales, torch.full_like(scales, float("inf")))
        midpoint = (scales.float() + upper.float()) * .5
        scales = torch.where(unrounded == midpoint, upper, scales).clamp_min(MIN_SCALE)
        if not torch.isfinite(scales).all():
            raise ValueError("FP16 scale overflow")
        return (w / scales.float()[:, None]).round().clamp(-127, 127).to(torch.int8), scales
    raise ValueError(f"unsupported tier: {tier}")


def dequantize(codes, scales, tier):
    if tier in ASYM_TIERS:
        import sb6
        return sb6.dequantize(codes, scales)
    if scales is None:
        return codes.float()
    if tier == "w8a16":
        return codes.float() * scales.float()[:, None]
    raise ValueError(f"unsupported tier: {tier}")


def write_payload(directory, name, tier, codes, scales):
    """Raw native NK bytes, low nibble first; manifest is consumed by rwkv_quantize."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if tier in ASYM_TIERS:
        import sb6
        data, metadata = sb6.pack(codes, scales, int(tier[1]))
        stem = name.replace(".", "_")
        (directory / (stem + ".bin")).write_bytes(data)
        (directory / (stem + ".scales")).write_bytes(metadata)
        return {"type": tier, "shape": list(codes.shape), "data": stem + ".bin", "scales": stem + ".scales"}
    codes = codes.cpu().contiguous()
    scales = scales.cpu().half().contiguous()
    if codes.dtype != torch.int8 or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("invalid calibrated payload")
    if tier == "w8a16":
        packed = codes.view(torch.uint8)
    else:
        raise ValueError("payload requires an integer tier")
    stem = name.replace(".", "_")
    packed.numpy().tofile(directory / (stem + ".bin"))
    scales.numpy().astype("<f2").tofile(directory / (stem + ".scales"))
    return {"type": tier, "shape": list(codes.shape), "data": stem + ".bin", "scales": stem + ".scales"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
