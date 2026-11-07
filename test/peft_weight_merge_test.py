#!/usr/bin/env python3
"""CPU golden tests: python test/peft_weight_merge_test.py --binary PATH."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
parser.add_argument('--threads', type=int, default=2,
                    help='Merger thread count; use 1 for builds without OpenMP')
args, rest = parser.parse_known_args()
BINARY = str(Path(args.binary).resolve())
NAMES = ['att.receptance.weight', 'att.key.weight', 'att.value.weight',
         'att.output.weight', 'ffn.key.weight', 'ffn.value.weight']


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.manual_seed(42)

    def fixture(self, method='miss', dtype=torch.bfloat16, factor_dtype=torch.bfloat16, master=False, packed=False, alias=False, rank=3, scale=1.0):
        weights = {'untouched': torch.randn(3, 4).T, 'scalar': torch.tensor(0.75)}
        factors, targets, raw = {}, [], bytearray()
        for i, suffix in enumerate(NAMES):
            name = 'blocks.0.' + suffix
            # Ragged MiSS columns, non-square matrices, and offset storage.
            rows, cols = 4+i, 7+i
            weights[name] = (torch.randn(rows+1, cols)*.2).to(dtype)[1:]
            bname = name + ('.B' if method == 'nora' else '.D')
            target = dict(name=bname, layer=0, target=suffix, in_features=cols, shape=[rows, rank])
            pairs = [(bname, (rows, rank))]
            if method == 'nora':
                target.update(a_name=name+'.A', a_shape=[rank, cols])
                pairs.append((name+'.A', (rank, cols)))
            for key, shape in pairs:
                value = torch.randn(shape)*.1
                forward = value.to(factor_dtype)
                raw += forward.contiguous().view(torch.uint8).numpy().tobytes()
                if master:
                    for kind in ['master', 'gradient', 'adam_m', 'adam_v']:
                        factors[key+'.'+kind] = value if kind == 'master' else torch.zeros_like(value)
                else:
                    factors[key] = forward
            targets.append(target)
        if packed:
            names = [name for name in weights if name.startswith('blocks.')]
            weights['packed_untouched'] = torch.randn(17).to(dtype)
            names.append('packed_untouched')
            storage = torch.cat([torch.zeros(5, dtype=dtype)] + [weights[name].flatten() for name in names])
            offset = 5
            for name in names:
                old = weights[name]
                weights[name] = storage[offset:offset+old.numel()].reshape(old.shape)
                offset += old.numel()
        if alias:
            weights['overlapping_alias'] = weights['blocks.0.'+NAMES[0]]
        base, adapter = self.root/'base.pth', self.root/'adapter.pth' 
        torch.save(weights, base)
        meta = dict(format_version=1, method=method, dtype=str(factor_dtype).split('.')[-1],
                    layout='low_rank_ba' if method == 'nora' else 'modulo_rank_zero_pad',
                    kind='inference_adapter', rank=rank, alpha=rank*scale, scale=scale,
                    base_fingerprint=hashlib.sha256(base.read_bytes()).hexdigest(), targets=targets)
        if method == 'nora':
            meta['initialization'] = 'nora_init'
        canonical = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()
        meta['content_version'] = hashlib.sha256(canonical+raw).hexdigest()
        if master:
            meta['tensor_source'] = 'master'
            factors['initial_state'] = torch.zeros(10)
        torch.save(factors, adapter)
        with zipfile.ZipFile(adapter, 'a') as z:
            z.writestr('adapter/'+method+'.json', json.dumps(meta))
        return weights, factors, base, adapter, factor_dtype

    def run_merge(self, base, adapter, method, *extra, ok=True):
        output = self.root/'merged.pth'
        result = subprocess.run([BINARY, '--model', str(base), '--adapter', str(adapter),
                                 '--method', method, '--output', str(output), '--threads', str(args.threads), *extra],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout+result.stderr)
        if not ok:
            self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('*.merge-*')))
        return output

    def test_numerics_dtypes_and_checkpoint_export(self):
        for method in ['miss', 'nora']:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for factor_dtype in [torch.float16, torch.bfloat16]:
                    for master in [False, True]:
                        with self.subTest(method=method, dtype=dtype, factor_dtype=factor_dtype, master=master):
                            weights, factors, base, adapter, _ = self.fixture(method, dtype, factor_dtype, master)
                            output = self.run_merge(base, adapter, method.upper(), '--scale', '-0.5')
                            with zipfile.ZipFile(output) as z:
                                self.assertIsNone(z.testzip())
                            merged = torch.load(output, map_location='cpu', weights_only=True)
                            self.assertEqual(set(merged), set(weights))
                            for name, weight in weights.items():
                                expected = weight
                                if name.startswith('blocks.'):
                                    suffix = '.master' if master else ''
                                    b = factors[name+('.D' if method == 'miss' else '.B')+suffix].to(factor_dtype).float()
                                    delta = b[:, torch.arange(weight.shape[1])%3] if method == 'miss' else b @ factors[name+'.A'+suffix].to(factor_dtype).float()
                                    expected = (weight.float()-.5*delta).to(dtype)
                                self.assertEqual(merged[name].dtype, weight.dtype)
                                torch.testing.assert_close(merged[name], expected, rtol=1e-6 if dtype==torch.float32 else 0, atol=1e-7 if dtype==torch.float32 else 0)
                            output.unlink()

    def test_default_scale_and_zero_preserves_bits(self):
        for method in ['miss', 'nora']:
            weights, factors, base, adapter, _ = self.fixture(method)
            output = self.run_merge(base, adapter, method, '--scale', '0')
            merged = torch.load(output, weights_only=True)
            for key in weights:
                self.assertTrue(torch.equal(merged[key], weights[key]))
            output.unlink()
            output = self.run_merge(base, adapter, method)
            merged = torch.load(output, weights_only=True)
            for key in weights:
                if key.startswith('blocks.'):
                    b = factors[key+('.D' if method=='miss' else '.B')].float()
                    delta = b[:,torch.arange(weights[key].shape[1])%3] if method=='miss' else b@factors[key+'.A'].float()
                    torch.testing.assert_close(merged[key], (weights[key].float()+delta).to(weights[key].dtype), rtol=0, atol=0)
            output.unlink()

    def test_reject_wrong_method_base_and_invalid_scale(self):
        _, _, base, adapter, _ = self.fixture()
        self.run_merge(base, adapter, 'nora', ok=False)
        for scale in ['nan', 'inf', '1junk']:
            self.run_merge(base, adapter, 'miss', '--scale', scale, ok=False)
        other = self.root/'other.pth'
        torch.save({'unrelated':torch.ones(2)}, other)
        self.run_merge(other, adapter, 'miss', ok=False)

    def test_packed_storage_and_rank_larger_than_input(self):
        for method in ['miss', 'nora']:
            weights, factors, base, adapter, _ = self.fixture(method, packed=True, rank=16)
            output = self.run_merge(base, adapter, method)
            merged = torch.load(output, weights_only=True)
            for key, value in weights.items():
                expected = value
                if key.startswith('blocks.'):
                    b = factors[key+('.D' if method=='miss' else '.B')].float()
                    delta = b[:,torch.arange(value.shape[1])%16] if method=='miss' else b@factors[key+'.A'].float()
                    expected = (value.float()+delta).to(value.dtype)
                torch.testing.assert_close(merged[key], expected, rtol=0, atol=0)
            with zipfile.ZipFile(output) as z:
                self.assertIsNone(z.testzip())
            output.unlink()

    def test_package_and_checkpoint_directories(self):
        for method in ['miss', 'nora']:
            _, factors, base, adapter, _ = self.fixture(method)
            with zipfile.ZipFile(adapter) as z:
                metadata = z.read('adapter/'+method+'.json')
            # Legacy exported package: factors in adapter.pth, metadata beside it.
            torch.save(factors, adapter)
            (self.root/'adapter.json').write_bytes(metadata)
            output = self.run_merge(base, self.root, method)
            self.assertTrue(output.is_file())
            output.unlink()
            (self.root/'adapter.json').unlink()
            _, _, base, adapter, _ = self.fixture(method, master=True)
            adapter.rename(self.root/'training.pth')
            output = self.run_merge(base, self.root, method)
            self.assertTrue(output.is_file())
            output.unlink()
            (self.root/'training.pth').unlink()

    def test_overlapping_storage_is_rejected(self):
        _, _, base, adapter, _ = self.fixture(packed=True, alias=True)
        self.run_merge(base, adapter, 'miss', ok=False)

    def test_metadata_scale_is_used(self):
        weights, factors, base, adapter, _ = self.fixture(scale=0.25)
        output = self.run_merge(base, adapter, 'miss')
        merged = torch.load(output, weights_only=True)
        for key, value in weights.items():
            if key.startswith('blocks.'):
                delta = factors[key+'.D'].float()[:,torch.arange(value.shape[1])%3]
                torch.testing.assert_close(merged[key], (value.float()+.25*delta).to(value.dtype), rtol=0, atol=0)

    def test_existing_output_is_untouched(self):
        _, _, base, adapter, _ = self.fixture()
        output = self.root/'merged.pth'
        output.write_bytes(b'keep me')
        proc = subprocess.run([BINARY,'--method','miss','--model',str(base),'--adapter',str(adapter),'--output',str(output)],capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(output.read_bytes(), b'keep me')

    def test_corrupt_digest_rejected(self):
        _, factors, base, adapter, _ = self.fixture()
        with zipfile.ZipFile(adapter) as z:
            meta = z.read('adapter/miss.json')
        factors[next(iter(factors))].add_(1)
        torch.save(factors, adapter)
        with zipfile.ZipFile(adapter, 'a') as z:
            z.writestr('adapter/miss.json', meta)
        self.run_merge(base, adapter, 'miss', ok=False)

    def test_overflow_leaves_no_output(self):
        _, _, base, adapter, _ = self.fixture(dtype=torch.float16)
        self.run_merge(base, adapter, 'miss', '--scale', '1e30', ok=False)


if __name__ == '__main__':
    unittest.main(argv=[__file__]+rest)
