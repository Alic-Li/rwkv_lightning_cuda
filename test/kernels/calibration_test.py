#!/usr/bin/env python3
"""CPU numerical and native C++ exporter integration tests (requires torch)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/calibration'))
from quantization import PROTECTED, dequantize, quantize, resolve_policy
from collect_activations import collect
from evaluate import read_archive
from model import Model

def tiny_model():
    torch.manual_seed(43)
    c, f, v, rank = 64, 128, 256, 16
    w = {'emb.weight': torch.randn(v, c) * .3, 'head.weight': torch.randn(v, c) * .1,
         'ln_out.weight': torch.ones(c), 'ln_out.bias': torch.zeros(c)}
    for i in range(2):
        p = f'blocks.{i}.'
        for ln in (('ln0', 'ln1', 'ln2') if i == 0 else ('ln1', 'ln2')):
            w[p + ln + '.weight'], w[p + ln + '.bias'] = torch.ones(c), torch.zeros(c)
        for s in 'rwkvag':
            w[p + 'att.x_' + s] = torch.rand(1, 1, c)
        for name in ('receptance', 'key', 'value', 'output'):
            w[p + 'att.' + name + '.weight'] = torch.randn(c, c) * .08
        for name in 'wavg':
            w[p + 'att.' + name + '1'] = torch.randn(c, rank) * .05
            w[p + 'att.' + name + '2'] = torch.randn(rank, c) * .05
            if name != 'g':
                w[p + 'att.' + name + '0'] = torch.randn(1, 1, c) * .1
        w[p + 'att.k_k'] = torch.ones(c)
        w[p + 'att.k_a'] = torch.rand(c)
        w[p + 'att.r_k'] = torch.randn(1, 64) * .1
        w[p + 'att.ln_x.weight'], w[p + 'att.ln_x.bias'] = torch.ones(c), torch.zeros(c)
        w[p + 'ffn.x_k'] = torch.rand(c)
        w[p + 'ffn.key.weight'] = torch.randn(f, c) * .08
        w[p + 'ffn.value.weight'] = torch.randn(c, f) * .08
    return {k: x.bfloat16() for k, x in w.items()}


class ExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = Path(os.environ.get('RWKV_QUANTIZER', ROOT / 'build/rwkv_quantize'))
        if not self.binary.exists():
            self.skipTest('build rwkv_quantize or set RWKV_QUANTIZER')

    def tearDown(self):
        self.temp.cleanup()

    def export(self, weights, *args, success=True):
        source, output = self.root / 'source.pth', self.root / 'out.rwkvq'
        torch.save(weights, source)
        result = subprocess.run([str(self.binary), *args, str(source), str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return output

    def test_fp16_ties_and_legacy_w8_grid(self):
        midpoint = 1.0 + 2.0 ** -11
        values = torch.tensor([[midpoint, -midpoint, .75 * 2.0 ** -24, -.75 * 2.0 ** -24]])
        (self.root / 'policy.json').write_text(json.dumps({'head.weight': 'fp16'}))
        out = self.export({'head.weight': values, 'emb.weight': torch.ones(1, 4).half()}, '--policy', str(self.root / 'policy.json'))
        value = list(read_archive(out))[0][1]
        torch.testing.assert_close(value, values.half().float(), rtol=0, atol=0)
        values = torch.tensor([[127 * midpoint, -127 * midpoint, 1., 0.]])
        out = self.export({'head.weight': values, 'emb.weight': torch.ones(1, 4).half()})
        value = list(read_archive(out))[0][1]
        q, s = quantize(values, 'w8a16')
        self.assertEqual(s.item(), 1.0 + 2.0 ** -10)
        torch.testing.assert_close(value, dequantize(q, s, 'w8a16'), rtol=0, atol=0)


    def test_retired_formats_and_policy_are_rejected(self):
        w={'emb.weight':torch.ones(3,32).bfloat16(),'head.weight':torch.randn(3,32).bfloat16()}
        good=self.export(w).read_bytes()
        for tier in ('w4a16','w5a16_asym','w4a16_g32','w4a16_g128'):
            self.export(w,'--format',tier,success=False)
            (self.root/'policy.json').write_text(json.dumps({'head.weight':tier}))
            self.export(w,'--policy',str(self.root/'policy.json'),success=False)
            self.assertEqual((self.root/'out.rwkvq').read_bytes(),good)
            with self.assertRaises(ValueError):resolve_policy(w,default=tier)

    def test_collect_and_sb6_protection(self):
        w=tiny_model();m=Model(w,'cpu');data=torch.randint(0,256,(2,8))
        with torch.inference_mode():stats=collect(m,data,1)
        self.assertEqual(len(stats),13)
        for name,nu in stats.items():
            self.assertEqual(nu.shape,(w[name].shape[1],))
            self.assertTrue(torch.isfinite(nu).all() and (nu>=0).all())
        self.assertEqual(resolve_policy(w,default='w4a16_asym')[PROTECTED],'bf16')
        self.assertEqual(resolve_policy(w,default='w6a16_asym')[PROTECTED],'w6a16_asym')

if __name__=='__main__':unittest.main()
