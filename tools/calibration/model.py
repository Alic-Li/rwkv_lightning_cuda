"""FP32 RWKV-7 calibration forward, with hooks at the actual linear inputs.

World .pth naming only. Every window starts from zero recurrent/shift state.
The equations follow RWKV-7 and RafaelUI/rwkv-quant's reference (Apache-2.0);
this implementation uses only this repository's native quantization tiers.
"""
from __future__ import annotations
import ast
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from quantization import load_checkpoint


@torch.jit.script
def wkv(r, decay, k, v, a, b):
    batch, length, heads, size = r.shape
    state = torch.zeros((batch, heads, size, size), device=r.device, dtype=torch.float32)
    output = torch.empty_like(r)
    for t in range(length):
        state = (state * decay[:, t].unsqueeze(-2)
                 + (state @ a[:, t].unsqueeze(-1)) * b[:, t].unsqueeze(-2)
                 + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2))
        output[:, t] = (state @ r[:, t].unsqueeze(-1)).squeeze(-1)
    return output


class Model:
    def __init__(self, weights, device="cuda"):
        self.source = load_checkpoint(weights) if isinstance(weights, (str, Path)) else weights
        self.device = torch.device(device)
        # FP32 compute avoids rounding fake-quantized values back to BF16.
        self.weights = {k: v.to(device=self.device, dtype=torch.float32) for k, v in self.source.items()}
        self.layers = max(int(k.split('.')[1]) for k in self.weights if k.startswith("blocks.")) + 1
        self.width = self.weights["emb.weight"].shape[1]
        self.heads, self.head_size = self.weights["blocks.0.att.r_k"].shape
        self.hook = None

    def linear(self, name, x):
        if self.hook is not None:
            self.hook(name, x)
        return F.linear(x, self.weights[name])

    def norm(self, x, prefix):
        return F.layer_norm(x, (self.width,), self.weights[prefix + ".weight"].reshape(-1),
                            self.weights[prefix + ".bias"].reshape(-1))

    def embed(self, tokens):
        return self.norm(F.embedding(tokens.to(self.device), self.weights["emb.weight"]), "blocks.0.ln0")

    def attention(self, i, x, first):
        p = f"blocks.{i}.att."
        w = self.weights
        delta = F.pad(x, (0, 0, 1, -1)) - x
        mixed = {s: x + delta * w[p + "x_" + s].reshape(-1) for s in "rwkvag"}
        r = self.linear(p + "receptance.weight", mixed['r'])
        k = self.linear(p + "key.weight", mixed['k'])
        v = self.linear(p + "value.weight", mixed['v'])
        decay = torch.exp(-torch.exp(-F.softplus(-(w[p + "w0"].reshape(-1) +
                      torch.tanh(mixed['w'] @ w[p + "w1"]) @ w[p + "w2"])) - .5))
        if i == 0:
            first = v
        else:
            v = v + (first - v) * torch.sigmoid(w[p + "v0"].reshape(-1) + mixed['v'] @ w[p + "v1"] @ w[p + "v2"])
        a = torch.sigmoid(w[p + "a0"].reshape(-1) + mixed['a'] @ w[p + "a1"] @ w[p + "a2"])
        gate = torch.sigmoid(mixed['g'] @ w[p + "g1"]) @ w[p + "g2"]
        shape = (*x.shape[:2], self.heads, self.head_size)
        kk = F.normalize((k * w[p + "k_k"].reshape(-1)).reshape(shape), dim=-1)
        k = k * (1 + (a - 1) * w[p + "k_a"].reshape(-1))
        rr, kk_scaled, vv = r.reshape(shape), k.reshape(shape), v.reshape(shape)
        out = wkv(rr, decay.reshape(shape), kk_scaled, vv, -kk, kk * a.reshape(shape))
        out = F.group_norm(out.reshape(-1, self.width), self.heads,
                           w[p + "ln_x.weight"].reshape(-1), w[p + "ln_x.bias"].reshape(-1), eps=64e-5).reshape_as(x)
        bonus = ((rr * kk_scaled * w[p + "r_k"]).sum(-1, keepdim=True) * vv).reshape_as(x)
        return self.linear(p + "output.weight", (out + bonus) * gate), first

    def block(self, i, x, first):
        p = f"blocks.{i}."
        attention, first = self.attention(i, self.norm(x, p + "ln1"), first)
        x = x + attention
        normalized = self.norm(x, p + "ln2")
        shifted = normalized + (F.pad(normalized, (0, 0, 1, -1)) - normalized) * self.weights[p + "ffn.x_k"].reshape(-1)
        hidden = self.linear(p + "ffn.key.weight", shifted).relu().square()
        return x + self.linear(p + "ffn.value.weight", hidden), first

    def hidden(self, tokens):
        x, first = self.embed(tokens), None
        for i in range(self.layers):
            x, first = self.block(i, x, first)
        return self.norm(x, "ln_out")

    def logits(self, hidden):
        return self.linear("head.weight", hidden)


class Tokenizer:
    def __init__(self, vocab):
        self.trie = {}
        for line in Path(vocab).read_text().splitlines():
            index = int(line[:line.index(' ')])
            value = ast.literal_eval(line[line.index(' ') + 1:line.rindex(' ')])
            value = value.encode() if isinstance(value, str) else value
            node = self.trie
            for byte in value:
                node = node.setdefault(byte, {})
            node[None] = index

    def encode(self, text):
        data, result, start = text.encode(), [], 0
        while start < len(data):
            node, end, best = self.trie, start, None
            while end < len(data) and data[end] in node:
                node = node[data[end]]
                end += 1
                if None in node:
                    best = end, node[None]
            if best is None:
                raise ValueError(f"unencodable byte at {start}")
            start, token = best
            result.append(token)
        return result


def windows(corpus, vocab, count=16, length=256, seed=42):
    if count <= 0 or length < 2:
        raise ValueError("windows must be positive and sequence length >= 2")
    tokens = Tokenizer(vocab).encode(Path(corpus).read_text())
    # Non-overlapping windows, deterministic sampling without repetition.
    starts = list(range(0, len(tokens) - length + 1, length))
    if len(starts) < count:
        raise ValueError(f"corpus contains {len(starts)} windows, need {count}")
    random.Random(seed).shuffle(starts)
    return torch.tensor([tokens[s:s + length] for s in starts[:count]], dtype=torch.long)


def add_model_args(parser, count=16):
    parser.add_argument("--model", required=True)
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--corpus", help="UTF-8 calibration text (separate from evaluation)")
    corpus.add_argument("--tokens", help="JSON {tokens: [[token IDs], ...]} with independent windows")
    parser.add_argument("--vocab", default=str(Path(__file__).resolve().parents[2] / "assets/rwkv_vocab_v20230424.txt"))
    parser.add_argument("--windows", type=int, default=count)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def setup(args):
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tokens:
        rows = json.loads(Path(args.tokens).read_text())["tokens"]
        if args.windows <= 0 or args.seq_len < 2 or len(rows) < args.windows:
            raise ValueError("not enough token windows or invalid window dimensions")
        # Preserve the explicit order so evaluation language quotas remain intact.
        rows = rows[:args.windows]
        if any(len(row) < args.seq_len or any(type(t) is not int for t in row) for row in rows):
            raise ValueError("invalid or short token window")
        data = torch.tensor([row[:args.seq_len] for row in rows], dtype=torch.long)
    else:
        data = windows(args.corpus, args.vocab, args.windows, args.seq_len, args.seed)
    model = Model(args.model, args.device)
    if data.min() < 0 or data.max() >= model.weights["emb.weight"].shape[0]:
        raise ValueError("token ID outside model vocabulary")
    return model, data
