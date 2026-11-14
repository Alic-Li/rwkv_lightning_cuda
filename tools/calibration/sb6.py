"""Unsigned 4/6-bit planes with signed-base, unsigned-six-bit SB metadata.

The native format deliberately differs from upstream rwkv-quant. Disk overhead
is exactly 16 bytes / 256 weights = .5 bpw (before row/header padding).
"""
import torch
import torch.nn.functional as F

TIERS = ('w4a16_asym', 'w6a16_asym')
MIN = 2. ** -24


def expanded(parts):
    s = (parts['d'].float().unsqueeze(-1) * parts['qs'].float()).half().float()
    m = (parts['dm'].float().unsqueeze(-1) * parts['qm'].float()).half().float()
    return s.flatten(1), m.flatten(1)


def dequantize(q, parts):
    s, m = expanded(parts)
    k = q.shape[1]
    return q.float() * s.repeat_interleave(32, 1)[:, :k] + m.repeat_interleave(32, 1)[:, :k]


def _project(w, importance, scale, offset, bits):
    d = (scale.amax(-1) / 63).clamp_min(MIN).half()
    qs = (scale / d.float().unsqueeze(-1)).round().clamp(1, 63).to(torch.uint8)
    # A common signed dm cannot encode opposite-sign group offsets. Choose the
    # sign of the largest magnitude offset, and project the opposite side to 0.
    magnitude, index = offset.abs().max(-1)
    sign = offset.gather(-1, index.unsqueeze(-1)).squeeze(-1).sign()
    dm = (magnitude / 63).clamp_min(MIN).half() * sign.half()
    denominator = dm.float().unsqueeze(-1)
    qm = torch.where(denominator != 0, offset / torch.where(denominator != 0, denominator, 1), 0).round().clamp(0, 63).to(torch.uint8)
    s = (d.float().unsqueeze(-1) * qs).half().float()
    m = (dm.float().unsqueeze(-1) * qm).half().float()
    if not torch.isfinite(s).all() or not torch.isfinite(m).all():
        raise ValueError('SB6 expanded FP16 scale/offset overflow')
    q = ((w - m.unsqueeze(-1)) / s.unsqueeze(-1)).round().clamp(0, 2**bits - 1)
    loss = ((w - (q * s.unsqueeze(-1) + m.unsqueeze(-1))).square() * importance).sum((-1, -2))
    return q.to(torch.uint8), {'d': d, 'dm': dm, 'qs': qs, 'qm': qm}, loss


def search(w, bits, nu=None, refine=True):
    """One row band; choose candidates by error AFTER both SB6/half projections."""
    if bits not in (4, 6):
        raise ValueError("SB6 bits must be 4 or 6")
    n, k = w.shape
    padded = (k + 255) // 256 * 256
    values = F.pad(w.float(), (0, padded-k)).reshape(n, -1, 8, 32)
    valid = (torch.arange(padded, device=w.device) < k).reshape(1, -1, 8, 32)
    if nu is None:
        importance = valid.float()
    else:
        importance = F.pad(nu.to(w.device).float(), (0, padded-k)).reshape(1, -1, 8, 32)
        energy = importance.sum(-1, keepdim=True)
        importance = torch.where(energy > 0, importance, valid.float())
        # Shared d/dm couple all eight groups: normalizing each group would
        # change the requested weighted objective and discard their relative
        # activation energy. Only rescale the entire superblock for stability.
        mean = importance.sum((-1, -2), keepdim=True) / valid.sum((-1, -2), keepdim=True).clamp_min(1)
        importance = importance / mean.clamp_min(1e-30)
    lower = values.masked_fill(~valid, float('inf')).amin(-1)
    upper = values.masked_fill(~valid, -float('inf')).amax(-1)
    lower = torch.where(valid.any(-1), lower, 0)
    upper = torch.where(valid.any(-1), upper, 0)
    # 41 endpoint pairs: baseline + 20 lower-only + 20 upper-only variations.
    factors = [.70 + .025 * i for i in range(21) if i != 12]
    grid = [(1., 1.)] + [(f, 1.) for f in factors] + [(1., f) for f in factors]
    best_q = best_parts = best_loss = None
    weights_sum = importance.sum(-1).clamp_min(1e-30)
    wmean = (importance * values).sum(-1) / weights_sum
    for flo, fhi in grid:
        lo, hi = lower * flo, upper * fhi
        sc = ((hi - lo) / (2**bits - 1)).clamp_min(MIN)
        q0 = ((values - lo.unsqueeze(-1)) / sc.unsqueeze(-1)).round().clamp(0, 2**bits - 1)
        candidates = [(sc, lo)]
        if refine:
            qmean = (importance * q0).sum(-1) / weights_sum
            cov = (importance * (q0-qmean.unsqueeze(-1)) * (values-wmean.unsqueeze(-1))).sum(-1)
            var = (importance * (q0-qmean.unsqueeze(-1)).square()).sum(-1)
            slope = cov / var.clamp_min(1e-30)
            good = (var > 1e-20) & (slope > 0) & torch.isfinite(slope)
            slope = torch.where(good, slope, sc)
            intercept = torch.where(good, wmean-slope*qmean, lo)
            candidates.append((slope, intercept))
        for sc, mn in candidates:
            q, parts, loss = _project(values, importance, sc, mn, bits)
            if best_loss is None:
                best_q, best_parts, best_loss = q, parts, loss
            else:
                better = loss < best_loss
                best_q = torch.where(better[..., None, None], q, best_q)
                best_parts = {name: torch.where(better if value.ndim == 2 else better.unsqueeze(-1), value, best_parts[name])
                              for name, value in parts.items()}
                best_loss = torch.minimum(best_loss, loss)
    return best_q.reshape(n, padded)[:, :k], best_parts


@torch.no_grad()
def quantize(w, bits, nu=None, row_chunk=512, refine=True):
    from quantization import _validate
    _validate(w, nu)
    if bits not in (4, 6):
        raise ValueError('SB6 bits must be 4 or 6')
    codes, collected = [], []
    for start in range(0, len(w), row_chunk):
        q, p = search(w[start:start+row_chunk], bits, nu, refine)
        codes.append(q); collected.append(p)
    return torch.cat(codes), {name: torch.cat([p[name] for p in collected]) for name in collected[0]}


def pack(q, parts, bits):
    """Return native q bytes and metadata bytes. Row and group planes are aligned."""
    if bits not in (4, 6):
        raise ValueError("SB6 bits must be 4 or 6")
    if q.dtype != torch.uint8 or (q > 2**bits-1).any():
        raise ValueError('SB6 codes outside unsigned range')
    q = F.pad(q.cpu(), (0, (-q.shape[1]) % 32)).reshape(q.shape[0], -1, 32)
    planes = [q[..., 0::2] & 15 | ((q[..., 1::2] & 15) << 4)]
    powers = 1 << torch.arange(32, dtype=torch.int64)
    for bit in range(4, bits):
        word = (((q.long() >> bit) & 1) * powers).sum(-1)
        planes.append(torch.stack([(word >> (8*j)) & 255 for j in range(4)], -1).byte())
    codes = torch.cat(planes, -1).contiguous().numpy().tobytes()
    bases = torch.stack((parts['d'], parts['dm']), -1).cpu().half().contiguous().view(torch.uint8)
    fields = [bases]
    for name in ('qs', 'qm'):
        v = parts[name].cpu().long()
        if v.shape[-1] != 8 or (v < 0).any() or (v > 63).any():
            raise ValueError('invalid SB6 six-bit metadata')
        word = (v * (1 << (6 * torch.arange(8, dtype=torch.int64)))).sum(-1)
        fields.append(torch.stack([(word >> (8*j)) & 255 for j in range(6)], -1).byte())
    return codes, torch.cat(fields, -1).contiguous().numpy().tobytes()


def unpack(data, metadata, n, k, bits):
    if bits not in (4, 6):
        raise ValueError("SB6 bits must be 4 or 6")
    groups, superblocks = (k+31)//32, (k+255)//256
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(n, groups, 16+4*(bits-4))
    q = torch.stack((raw[..., :16]&15, raw[..., :16]>>4), -1).reshape(n, groups, 32).long()
    for bit in range(4, bits):
        plane = raw[..., 16+4*(bit-4):20+4*(bit-4)].long()
        word = sum(plane[..., j] << (8*j) for j in range(4))
        q |= ((word.unsqueeze(-1) >> torch.arange(32)) & 1) << bit
    meta = torch.frombuffer(bytearray(metadata), dtype=torch.uint8).reshape(n, superblocks, 16)
    bases = meta[..., :4].contiguous().view(torch.float16)
    parts = {'d': bases[..., 0], 'dm': bases[..., 1]}
    for name, offset in [('qs', 4), ('qm', 10)]:
        word = sum(meta[..., offset+j].long() << (8*j) for j in range(6))
        parts[name] = ((word.unsqueeze(-1) >> (6 * torch.arange(8))) & 63).byte()
    if not torch.isfinite(bases).all() or (parts['d'] < 0).any():
        raise ValueError('invalid SB6 base')
    s, m = expanded(parts)
    if not torch.isfinite(s).all() or not torch.isfinite(m).all():
        raise ValueError('invalid SB6 expanded metadata')
    return q.flatten(1)[:, :k].byte(), parts
