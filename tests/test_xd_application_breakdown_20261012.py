import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1]/'experiments/tools/run_xd_application_breakdown_20261012.py'
spec = importlib.util.spec_from_file_location('xd_breakdown', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BreakdownTests(unittest.TestCase):
    def test_complex_error_preserves_imaginary_component(self):
        result = module.distribution(np.array([1+2j]), np.array([1+1j]))
        self.assertEqual(result['absolute_quantiles']['1.0'], 1)
        self.assertIsNotNone(result['relative_l2'])

    def test_cache_miss_fails_closed(self):
        class Adapter:
            def plan_metadata(self):
                return [{'selection_source':'static'}]
        with self.assertRaises(RuntimeError):
            module.require_plans([Adapter()], True)
        self.assertEqual(len(module.require_plans([Adapter()], False)), 1)

    def test_cpu_filter_and_event_order(self):
        rng=module.seeded_rng(4)
        x=rng.standard_normal((4,8,8)).astype(np.float32)
        h=rng.standard_normal((3,3)).astype(np.float32)
        shape=module.filter_api.padding_shape(x.shape[-2:],h.shape)
        order=[]
        class Event:
            def __init__(self,index): self.index=index
            def record(self): order.append(self.index)
            def synchronize(self): self.fail('inner synchronization')
        forward=lambda v:torch.fft.rfftn(v,dim=(-2,-1),norm='forward')
        inverse=lambda v:torch.fft.irfftn(v,s=shape,dim=(-2,-1),norm='forward')
        result=module.filter_call(torch.from_numpy(x),torch.from_numpy(h),shape,
                forward,forward,inverse,torch,[Event(i) for i in range(8)])
        np.testing.assert_allclose(result.numpy(),module.filter_api.cpu_convolve(x,h),atol=5e-6)
        self.assertEqual(order,list(range(8)))
        self.assertEqual(result.shape,(4,8,8))

    def test_segment_synchronizes_only_after_complete_call(self):
        calls=[]
        class Event:
            def __init__(self,**kwargs): pass
            def record(self): calls.append('record')
            def synchronize(self): calls.append('sync')
            def elapsed_time(self,other): return 2.0
        class FakeTorch:
            class cuda:
                Event=None
        FakeTorch.cuda.Event=Event
        def callback(events):
            for event in events: event.record()
            calls.append('complete')
        result=module.segment_timing(callback,FakeTorch,2)
        self.assertEqual(calls,['record']*8+['complete','sync']+['record']*8+['complete','sync'])
        self.assertIn('do not sum',result['warning'])

    def test_prepare_hashes_and_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with patch.object(module,'CASES',[((8,8),1),((4,4,4),4)]), \
                 patch.object(module,'identity',return_value=({}, {'test':'cpu'})):
                manifest=module.prepare(path)
                data=module.verify(manifest)
                self.assertEqual(len(data['filters']),4)
                self.assertEqual(len(data['direct']),2)
                artifact=Path(data['direct'][0]['artifact'])
                before=artifact.read_bytes()
                self.assertEqual(module.prepare(path),manifest)
                self.assertEqual(artifact.read_bytes(),before)
                with artifact.open('ab') as stream: stream.write(b'changed')
                with self.assertRaises(RuntimeError): module.verify(manifest)

    def test_failure_does_not_promote_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            args=type('Args',(),dict(until='2026-10-12T23:00:00+08:00',output=Path(directory),warmup=1,repeats=1,samples=1))()
            manifest={'identity':{'gpu_uuid':'GPU-test'}}
            with patch.object(module,'prepare',return_value=Path(directory)/'manifest.json'), \
                 patch.object(module,'verify',return_value=manifest), \
                 patch.object(module,'identity',return_value=({},{})), \
                 patch.object(module.subprocess,'run',return_value=type('Result',(),{'returncode':1})()), \
                 patch.object(module,'accept_release') as promote:
                with self.assertRaises(RuntimeError): module.launch(args)
                promote.assert_not_called()


if __name__=='__main__':
    unittest.main()
