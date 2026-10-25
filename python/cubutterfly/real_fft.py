"""Optional PyTorch FP32 compact real FFT integration (CUDA imports are lazy)."""
import ctypes
import hashlib
import json
import math
from pathlib import Path

from . import (AlgorithmPolicy, ComputeType, DataType, ExecutionMode, FftType,
               Library, Plan, PlanOptions, Transform, _integer)


def output_scale(norm, inverse, volume):
    if norm == 'forward':
        return 1.0 if inverse else 1.0/volume
    if norm == 'backward':
        return 1.0/volume if inverse else 1.0
    if norm == 'ortho':
        return 1.0/math.sqrt(volume)
    raise ValueError('norm must be forward, backward or ortho')


class RealFFT:
    """Out-of-place compact rank2/3 FFT; reused outputs are overwritten next call.

    Stream is fixed at construction, matching the public Plan contract. Caller
    supplied outputs must be contiguous, correctly typed, and nonaliasing.
    """
    def __init__(self, library, shape, batch, device, *, norm='backward',
                 reuse_buffers=False, cache_path=None, policy=AlgorithmPolicy.DEFAULT):
        import torch
        self.torch=torch
        self.shape=tuple(_integer(n,'extent',ctypes.c_size_t) for n in shape)
        self.batch=_integer(batch,'batch',ctypes.c_size_t)
        if len(self.shape) not in (2,3) or any(n<1 for n in self.shape) or self.batch<1:
            raise ValueError('positive rank2/3 extents and batch required')
        self.norm=norm
        self.volume=math.prod(self.shape)
        output_scale(norm,False,self.volume)
        self.device=torch.device(device)
        if self.device.type!='cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        self.stream=int(torch.cuda.current_stream(self.device).cuda_stream)
        self.library=library if isinstance(library,Library) else Library(library)
        # This optional query is not yet bound by the core lightweight frontend.
        if hasattr(self.library,'_dll'):
            query=self.library._dll.cubutterflyPlanGetSelectionSource
            query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
            query.restype=ctypes.c_int
        self.reuse_buffers=bool(reuse_buffers)
        self.half_shape=self.shape[:-1]+(self.shape[-1]//2+1,)
        self.plans=[]
        self.buffers={}
        try:
            for inverse in (False,True):
                transform=Transform(storage=DataType.FP32,compute=ComputeType.FP32,
                    extents=list(self.shape),batch=self.batch,inverse=inverse,
                    normalize_inverse=False,fft_type=FftType.C2R if inverse else FftType.R2C,
                    fft_output_scale=output_scale(norm,inverse,self.volume),execution_mode=ExecutionMode.XD)
                if isinstance(policy, str):
                    options = PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                                          mapping=policy,
                                          execution_mode=ExecutionMode.XD)
                else:
                    options = PlanOptions(policy=policy, execution_mode=ExecutionMode.XD)
                plan=Plan(self.library,transform,options,
                          stream=self.stream,cache_path=cache_path)
                self.plans.append(plan)
                size=plan.workspace_size()
                if size:
                    workspace=torch.empty(size,dtype=torch.uint8,device=self.device)
                    plan.set_workspace(workspace,size)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _execute(self,x,inverse,out=None):
        torch=self.torch
        if len(self.plans)!=2:
            raise RuntimeError('FFT is closed')
        if int(torch.cuda.current_stream(self.device).cuda_stream)!=self.stream:
            raise ValueError('FFT stream differs from construction stream')
        tail=self.half_shape if inverse else self.shape
        target=self.shape if inverse else self.half_shape
        dtype=torch.complex64 if inverse else torch.float32
        outdtype=torch.float32 if inverse else torch.complex64
        leading=tuple(x.shape[:-len(tail)])
        if tuple(x.shape[-len(tail):])!=tail or math.prod(leading)!=self.batch or x.dtype!=dtype or x.device!=self.device:
            raise ValueError('FFT input shape/dtype/device mismatch')
        flat=x.contiguous()
        if out is None:
            if self.reuse_buffers and inverse in self.buffers:
                out=self.buffers[inverse]
            else:
                out=torch.empty((self.batch,*target),dtype=outdtype,device=self.device)
                if self.reuse_buffers:
                    self.buffers[inverse]=out
        if tuple(out.shape)!=(self.batch,*target) or out.dtype!=outdtype or out.device!=self.device or not out.is_contiguous():
            raise ValueError('FFT output shape/dtype/device/contiguity mismatch')
        if out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr():
            raise ValueError('out-of-place FFT output aliases input')
        plan = self.plans[int(inverse)]
        submit = getattr(plan, "execute_steady", plan.execute)
        submit(flat, out)
        return out.reshape(*leading,*target)

    def rfftn(self,x,out=None): return self._execute(x,False,out)
    def irfftn(self,x,out=None): return self._execute(x,True,out)

    def plan_metadata(self):
        records=[]
        for inverse,plan in enumerate(self.plans):
            source=ctypes.c_int()
            self.library._call('PlanGetSelectionSource',plan.native_handle,ctypes.byref(source))
            cache=Path(plan.cache_path) if plan.cache_path else None
            records.append(dict(direction='inverse' if inverse else 'forward',norm=self.norm,
                reuse_buffers=self.reuse_buffers,selection_source={0:'measured_table',1:'user_cache',2:'runtime_measured',3:'static_model',4:'explicit',5:'unmeasured_fallback'}.get(source.value,str(source.value)),
                selection_source_id=source.value,algorithm=plan.algorithm_name(),selection_reason=plan.selection_reason(),
                descriptor=json.loads(plan.descriptor_json()),mapping=json.loads(plan.mapping_json()),
                cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest() if cache.is_file() else None) if cache else None,
                selection_scope='complete public xD plan, not rank1 calibration'))
        return records

    def close(self):
        plans,self.plans=self.plans,[]
        self.buffers.clear()
        error=None
        for plan in plans:
            try:
                plan.close()
            except Exception as exc:
                if error is None:
                    error=exc
        if error is not None:
            raise error

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
