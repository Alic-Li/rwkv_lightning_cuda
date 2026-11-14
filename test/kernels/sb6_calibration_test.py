#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/calibration'))
sys.path.insert(0,str(Path(__file__).parent))
import sb6
from calibration_test import ExportTests
from quantization import quantize, dequantize, write_payload, tensor_bytes, resolve_policy, save_json
from evaluate import read_archive

class SB6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117);torch.set_num_threads(2)

    def test_retired_bit_width_rejected(self):
        w = torch.zeros(1, 32)
        with self.assertRaisesRegex(ValueError, '4 or 6'):
            sb6.quantize(w, 5)
        with self.assertRaisesRegex(ValueError, '4 or 6'):
            sb6.search(w, 5)
        with self.assertRaisesRegex(ValueError, '4 or 6'):
            sb6.pack(w.byte(), {}, 5)
        with self.assertRaisesRegex(ValueError, '4 or 6'):
            sb6.unpack(b'', b'', 1, 32, 5)

    def test_all_codes_planes_and_metadata(self):
        for bits in (4,6):
            q=(torch.arange(3*289).reshape(3,289)%(2**bits)).byte()
            p={'d':torch.full((3,2),.125).half(),'dm':torch.tensor([[-.25,.25]]*3).half(),
               'qs':(torch.arange(48).reshape(3,2,8)%64).byte(),'qm':(63-torch.arange(48).reshape(3,2,8)).byte()}
            data,meta=sb6.pack(q,p,bits)
            out,parts=sb6.unpack(data,meta,3,289,bits)
            self.assertTrue(torch.equal(out,q))
            for n in p:self.assertTrue(torch.equal(p[n],parts[n]))
            self.assertEqual(len(data),3*10*(16+4*(bits-4)))
            self.assertEqual(len(meta),3*2*16)

    def test_ls_projection_not_worse_and_degenerate(self):
        for bits in (4,6):
            w=torch.randn(3,269)*.2-.07;nu=torch.rand(269);nu[:32]=0
            a,p=sb6.quantize(w,bits,nu,refine=False)
            b,r=sb6.quantize(w,bits,nu,refine=True)
            # Preserve relative group energies inside each superblock;
            # only zero-energy groups have uniform fallback.
            v=nu.clone();v[:32]=1
            self.assertLessEqual(((sb6.dequantize(b,r)-w).square()*v).sum(),((sb6.dequantize(a,p)-w).square()*v).sum()+1e-6)
            for value in (0.,.125,-.125,1e-10):
                q,parts=sb6.quantize(torch.full((2,33),value),bits,torch.zeros(33))
                self.assertTrue(torch.isfinite(sb6.dequantize(q,parts)).all())

    def test_shared_metadata_preserves_group_importance(self):
        torch.manual_seed(19)
        w=torch.randn(3,256)*.2
        nu=torch.ones(256);nu[:32]=10000
        for bits in (4,6):
            q,p=sb6.quantize(w,bits)
            a,r=sb6.quantize(w,bits,nu)
            uniform=((sb6.dequantize(q,p)-w).square()*nu).sum()
            weighted=((sb6.dequantize(a,r)-w).square()*nu).sum()
            self.assertLess(weighted,uniform*.95)

class SB6ExportTests(ExportTests):
    def test_sb6_native_archive(self):
        w={'emb.weight':torch.randn(5,257).bfloat16()}
        manifest={};policy={}
        expected={}
        for bits in (4,6):
            name=f'blocks.{bits}.att.key.weight';tier=f'w{bits}a16_asym'
            w[name]=(torch.randn(3,257)*.2).bfloat16()
            q,p=quantize(w[name],tier,torch.rand(257))
            manifest[name]=write_payload(self.root,name,tier,q,p);policy[name]=tier
            expected[name]=dequantize(q,p,tier)
        save_json(self.root/'manifest.json',manifest);save_json(self.root/'policy.json',policy)
        out=self.export(w,'--policy',str(self.root/'policy.json'),'--calibrated',str(self.root/'manifest.json'))
        for name,v,dt,g in read_archive(out):
            torch.testing.assert_close(v,expected.get(name,w[name].float()),rtol=0,atol=0)
        resolved=resolve_policy(w,policy)
        self.assertEqual(out.stat().st_size,16+sum(tensor_bytes(n,v.shape,resolved[n]) for n,v in w.items()))
        # Invalid metadata must be rejected transactionally.
        name=next(iter(manifest));path=self.root/manifest[name]['scales']
        raw=bytearray(path.read_bytes());raw[:2]=b'\x00\x7c';path.write_bytes(raw)
        self.export(w,'--policy',str(self.root/'policy.json'),'--calibrated',str(self.root/'manifest.json'),success=False)


if __name__=='__main__':unittest.main()
