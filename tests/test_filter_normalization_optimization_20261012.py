import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from cubutterfly.real_fft import output_scale,RealFFT
spec=importlib.util.spec_from_file_location('filter_opt',ROOT/'experiments/tools/filter_normalization_optimization_20261012.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def test_scale_semantics(self):
        self.assertEqual([output_scale('forward',i,16) for i in (False,True)],[1/16,1])
        self.assertEqual([output_scale('backward',i,16) for i in (False,True)],[1,1/16])
        self.assertEqual(output_scale('ortho',True,16),1/4)
        with self.assertRaises(ValueError): output_scale('invalid',False,16)

    def test_all_contracts_same_math(self):
        rng=np.random.default_rng(123)
        for batch in (1,4):
            shape=(8,8) if batch==1 else (batch,8,8)
            image=torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
            kernel=torch.from_numpy(rng.standard_normal((3,3)).astype(np.float32))
            padding=module.base.filter_api.padding_shape(image.shape[-2:],kernel.shape)
            reference=module.base.filter_api.cpu_convolve(image.numpy(),kernel.numpy())
            for norm in ('forward','backward'):
                for cached in (False,True):
                    calls=[]
                    forward=lambda x:torch.fft.rfftn(x,dim=(-2,-1),norm=norm)
                    def hf(x):
                        calls.append('filter');return forward(x)
                    inverse=lambda x:torch.fft.irfftn(x,s=padding,dim=(-2,-1),norm=norm)
                    call=module.make_call(image,kernel,padding,forward,hf,inverse,torch,norm,cached)
                    for _ in range(3): np.testing.assert_allclose(call().numpy(),reference,atol=8e-6)
                    self.assertEqual(len(calls),1 if cached else 3)

    def test_normalization_is_configured_before_creation(self):
        import cubutterfly.real_fft as real
        transforms=[]
        class FakePlan:
            def __init__(self,library,transform,options,**kwargs): transforms.append(transform)
            def workspace_size(self): return 0
            def close(self): pass
        class FakeLibrary: pass
        with patch.object(real,'Library',FakeLibrary),patch.object(real,'Plan',FakePlan), \
             patch.object(torch.cuda,'current_stream',return_value=type('Stream',(),{'cuda_stream':5})()), \
             patch.object(torch.cuda,'current_device',return_value=0):
            fft=RealFFT(FakeLibrary(),(np.int64(8),8),np.int64(4),'cuda',norm='backward',reuse_buffers=True)
            self.assertEqual(fft.device,torch.device('cuda:0'))
            self.assertEqual([t.fft_output_scale for t in transforms],[1,1/64])
            self.assertTrue(all(not t.normalize_inverse for t in transforms))
            self.assertEqual([t.fft_type for t in transforms],[1,2])
            fft.close()

    def test_integer_contract_rejects_bool_float_and_overflow(self):
        for shape,batch in [((True,8),1),((8.5,8),1),((8,8),True),((8,8),1.5),((8,8),2**65)]:
            with self.assertRaises((TypeError,ValueError)):
                RealFFT(None,shape,batch,'cuda')

    def test_partial_construction_failure_closes_resources(self):
        import cubutterfly.real_fft as real
        closed=[]
        class FakeLibrary: pass
        class FakePlan:
            def __init__(self,*args,**kwargs):
                if closed==['created']: raise RuntimeError('construction failure')
                closed.append('created')
            def workspace_size(self): return 0
            def close(self): closed.append('closed')
        with patch.object(real,'Library',FakeLibrary),patch.object(real,'Plan',FakePlan), \
             patch.object(torch.cuda,'current_stream',return_value=type('Stream',(),{'cuda_stream':5})()):
            with self.assertRaisesRegex(RuntimeError,'construction failure'):
                RealFFT(FakeLibrary(),(8,8),1,'cuda:0')
        self.assertEqual(closed,['created','closed'])

    def test_close_attempts_all_plans_even_when_one_close_fails(self):
        calls=[]
        class FakePlan:
            def __init__(self,index): self.index=index
            def close(self):
                calls.append(self.index)
                if self.index==0: raise RuntimeError('close failure')
        fft=RealFFT.__new__(RealFFT);fft.plans=[FakePlan(0),FakePlan(1)];fft.buffers={}
        with self.assertRaisesRegex(RuntimeError,'close failure'): fft.close()
        self.assertEqual(calls,[0,1]);self.assertEqual(fft.plans,[])
        fft.close()

    def test_reuse_and_explicit_output(self):
        fft=RealFFT.__new__(RealFFT)
        fft.torch=torch;fft.device=torch.device('cpu');fft.stream=5
        fft.shape=(8,8);fft.half_shape=(8,5);fft.batch=1
        fft.reuse_buffers=True;fft.buffers={}
        class Plan:
            def execute(self,x,out): out.zero_()
        fft.plans=[Plan(),Plan()]
        x=torch.ones(8,8)
        with patch.object(torch.cuda,'current_stream',return_value=type('Stream',(),{'cuda_stream':5})()):
            a=fft.rfftn(x);b=fft.rfftn(x)
            self.assertEqual(a.data_ptr(),b.data_ptr())
            out=torch.empty(1,8,5,dtype=torch.complex64)
            self.assertEqual(fft.rfftn(x,out=out).data_ptr(),out.data_ptr())
            with self.assertRaises(ValueError):fft.rfftn(x,out=torch.empty(1,8,8))
            y=torch.empty(1,8,5,dtype=torch.complex64)
            alias=y.view(torch.float32).reshape(-1)[:64].reshape(1,8,8)
            with self.assertRaises(ValueError):fft.irfftn(y,out=alias)


if __name__=='__main__': unittest.main()
