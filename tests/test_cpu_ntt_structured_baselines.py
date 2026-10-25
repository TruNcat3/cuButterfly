"""Exercise actual external cores against independent exact/FP references."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_ntt_baseline import NTTCore
from cpu_structured_baseline import StructuredCore
from semantic_ntt_reference import radix2_ntt_reference


class CPUExternalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT/'results/install_compare_20261013/manifest.json').read_text())['cases']

    def semantic(self, op, storage, shape, inverse=False):
        s = copy.deepcopy(next(c['descriptor']['semantics'] for c in self.cases if c['descriptor']['semantics']['operator']==op and c['descriptor']['semantics']['storage']==storage))
        s.update(shape=list(shape), batch=2, inverse=int(inverse), normalize_inverse=inverse)
        s['input_orders'] = s['output_orders'] = [0]*len(shape)
        strides = []
        distance = 1
        for n in reversed(shape):
            strides.insert(0,distance)
            distance *= n
        s['input_strides'] = s['output_strides'] = strides
        s['input_batch_stride'] = s['output_batch_stride'] = distance
        s['matrices'] = [[[1.,.25,-.5,1.]] for _ in shape]
        return s

    def test_hexl_exact_forward_inverse_both_widths_and_axes(self):
        for storage, dtype in ((8,np.uint32),(9,np.uint64)):
            for shape in ((16,),(4,8),(2,4,8)):
                for inverse in (False,True):
                    with self.subTest(storage=storage,shape=shape,inverse=inverse):
                        s = self.semantic(1,storage,shape,inverse)
                        # Exercise full residues, not only small integer inputs.
                        rng = np.random.default_rng(19)
                        source = rng.integers(0,s['modulus'],size=(2,*shape),dtype=dtype)
                        core = NTTCore(s,source)
                        try:
                            original = source.copy()
                            actual = core.execute().copy()
                            expected = radix2_ntt_reference(source,s['modulus'],inverse=inverse,normalize_inverse=inverse).astype(dtype)
                            np.testing.assert_array_equal(actual,expected)
                            np.testing.assert_array_equal(source,original)
                            np.testing.assert_array_equal(core.execute(),expected)
                            self.assertEqual(core.metadata['storage_conversion'],storage==8)
                            self.check_provenance(core.metadata)
                        finally:
                            core.close()

    def test_structured_actual_float_and_double_core(self):
        for storage,dtype,inverse in ((2,np.float32,False),(3,np.float64,False),(2,np.float32,True),(3,np.float64,True)):
            for shape in ((16,),(4,8),(2,4,8)):
                with self.subTest(storage=storage,shape=shape):
                    s = self.semantic(5,storage,shape,inverse)
                    source = np.random.default_rng(17).normal(size=(2,*shape)).astype(dtype)
                    original = source.copy()
                    expected = source.copy()
                    for axis,n in enumerate(shape,1):
                        moved = np.moveaxis(expected,axis,-1).copy()
                        rows = moved.reshape(-1,n)
                        half = 1
                        while half<n:
                            blocks = rows.reshape(-1,2*half)
                            left,right = blocks[:,:half].copy(),blocks[:,half:].copy()
                            a,b,c,d = (dtype(1/1.125),dtype(-.25/1.125),dtype(.5/1.125),dtype(1/1.125)) if inverse else (dtype(1),dtype(.25),dtype(-.5),dtype(1))
                            blocks[:,:half] = a*left + b*right
                            blocks[:,half:] = c*left + d*right
                            half *= 2
                        expected = np.moveaxis(moved,-1,axis)
                    core = StructuredCore(s,source)
                    try:
                        self.check_provenance(core.metadata)
                        np.testing.assert_allclose(core.execute(),expected,rtol=2e-6 if storage==2 else 2e-14,atol=2e-6 if storage==2 else 2e-14)
                        np.testing.assert_array_equal(source,original)
                    finally:
                        core.close()

    def check_provenance(self, metadata):
        for item in [metadata['library_identity'],metadata['external_library'],metadata['bridge_source'],*metadata['upstream_sources']]:
            self.assertEqual(item['sha256'],hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
        self.assertTrue(metadata['upstream_commit'])
        self.assertTrue(metadata['upstream_version'])

    def test_reject_noncontiguous_wrong_precision_and_compute(self):
        for op,storage,dtype,core_type in ((1,8,np.uint32,NTTCore),(1,9,np.uint64,NTTCore),(5,2,np.float32,StructuredCore),(5,3,np.float64,StructuredCore)):
            s = self.semantic(op,storage,(4,8))
            source = np.zeros((2,4,8),dtype=dtype)
            for bad in (source[:,:,::-1],source.astype(np.int64),source.tolist()):
                with self.assertRaisesRegex(ValueError,'C-contiguous'):
                    core_type(s,bad)
            for compute in (-1,5,True):
                wrong = dict(s,compute=compute)
                with self.assertRaisesRegex(ValueError,'compute'):
                    core_type(wrong,source)


if __name__ == '__main__':
    unittest.main()
