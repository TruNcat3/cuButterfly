"""Unmodified HazyResearch butterfly CPU core for broadcast 2x2 stages."""
import ctypes as ct
from pathlib import Path
import subprocess
import numpy as np
from cpu_ntt_baseline import _packed, _fingerprint, _loaded_identity, _git_identity

ROOT = Path(__file__).resolve().parents[1] / 'results/install_compare_20261013/external/butterfly'
BRIDGE = Path(__file__).with_name('cpu_external_structured_bridge.cpp')
COMMIT = '7217b5d93bc78e1229fed3761bcc70d943f604b7'


def setup(root, *, fetch=False):
    """Build the pinned upstream CPU source without installing a package."""
    import torch
    import sysconfig
    from torch.utils.cpp_extension import include_paths, library_paths
    root = Path(root).resolve()
    source = root/'csrc/cpu/butterfly_cpu.cpp'
    if not source.exists():
        if not fetch: raise RuntimeError('missing butterfly source; use setup(root, fetch=True)')
        subprocess.run(['git','clone','--no-checkout','--depth','1','https://github.com/HazyResearch/butterfly.git',str(root)],check=True)
        subprocess.run(['git','fetch','--depth','1','origin',COMMIT],cwd=root,check=True)
        subprocess.run(['git','checkout','--detach',COMMIT],cwd=root,check=True)
    if _git_identity(root)[0] != COMMIT:
        raise RuntimeError('butterfly source must match pinned upstream commit')
    path = root/'libbutterfly_adapter.so'
    sources = [BRIDGE,source,source.with_suffix('.h')]
    if not path.exists() or any(p.stat().st_mtime_ns > path.stat().st_mtime_ns for p in sources):
        cmd = ['g++','-O2','-shared','-fPIC','-std=c++17','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-I'+str(root),'-I'+sysconfig.get_path('include')]
        cmd += ['-I'+p for p in include_paths()]
        cmd += [str(BRIDGE),str(source)]
        for p in library_paths(): cmd += ['-L'+p,'-Wl,-rpath,'+p]
        cmd += ['-ltorch','-ltorch_cpu','-lc10','-ltorch_python','-o',str(path)]
        subprocess.run(cmd,check=True)
    return path


class StructuredCore:
    def __init__(self, s, source, root=None):
        import torch
        self.root = Path(root or ROOT).resolve()
        if s['operator'] != 5 or s['storage'] not in (2, 3) or s['length_mode'] or any(s['input_orders']) or any(s['output_orders']) or not _packed(s):
            raise ValueError('butterfly adapter requires packed natural FP32/FP64 structured stages')
        self.s, self.source = s, np.asarray(source)
        dtype = np.dtype(np.float32 if s['storage'] == 2 else np.float64)
        if type(s.get('compute')) is not int or s['compute'] not in (0,1 if s['storage']==2 else 2):
            raise ValueError('structured compute must be native or match source precision')
        if not isinstance(source,np.ndarray) or self.source.dtype != dtype or self.source.shape != (s['batch'], *s['shape']) or not self.source.flags.c_contiguous:
            raise ValueError('source must be C-contiguous with matching shape and exact storage')
        self.output = np.empty_like(source)
        path = setup(self.root)
        self.lib = ct.CDLL(str(path))
        self.lib.butterfly_execute.argtypes = [ct.c_void_p]*3+[ct.c_int64]*3+[ct.c_int]
        self.weights = []
        for n, matrices in zip(s['shape'], s['matrices']):
            levels = n.bit_length()-1
            m = np.asarray(matrices, dtype=np.float64).reshape(-1,2,2)
            if len(m) not in (1, levels): raise ValueError('expected broadcast or per-stage matrices')
            if not np.isfinite(m).all(): raise ValueError('matrix entries must be finite')
            if s['inverse']:
                # Match public reference: invert coefficients in double, then
                # cast once to the requested core precision during setup.
                determinant = m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]
                if np.any(determinant == 0) or not np.isfinite(determinant).all():
                    raise ValueError('inverse matrices must be nonsingular')
                m = np.stack((m[:,1,1],-m[:,0,1],-m[:,1,0],m[:,0,0]),axis=-1).reshape(-1,2,2)/determinant[:,None,None]
            m = m.astype(dtype)
            self.weights.append(np.broadcast_to(m.reshape(1,1,len(m),1,2,2),(1,1,levels,n//2,2,2)).copy())
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        self.metadata = dict(backend='hazyresearch-butterfly-cpu', library=str(path), upstream='https://github.com/HazyResearch/butterfly', upstream_commit=commit, requested_storage=str(dtype), external_core_storage=str(dtype), inverse_coefficients_precomputed=bool(s['inverse']), inverse_normalization='matrix inverse only, matching public structured semantics', threads=1, planning_excluded=True, timing_scope='axis packing, unmodified upstream CPU butterfly, output allocation and output copy')
        identity = _loaded_identity(ct.cast(self.lib.butterfly_execute,ct.c_void_p))
        self.metadata.update(library_identity=identity, external_library=identity, bridge_source=_fingerprint(BRIDGE), upstream_version=_git_identity(self.root)[1], upstream_sources=[_fingerprint(self.root/'csrc/cpu'/name) for name in ('butterfly_cpu.cpp','butterfly_cpu.h')], adapter_source=_fingerprint(__file__), torch_version=torch.__version__, torch_runtime=_fingerprint(Path(torch.__file__).parent/'lib/libtorch_cpu.so'))

    def execute(self):
        value = self.source
        for axis, weights in enumerate(self.weights,1):
            moved = np.moveaxis(value, axis, -1)
            rows = np.ascontiguousarray(moved).reshape(-1,moved.shape[-1])
            out = np.empty_like(rows)
            self.lib.butterfly_execute(weights.ctypes.data,rows.ctypes.data,out.ctypes.data,len(rows),rows.shape[-1],rows.shape[-1].bit_length()-1,32 if rows.dtype==np.float32 else 64)
            value = np.moveaxis(out.reshape(moved.shape),-1,axis)
        np.copyto(self.output,value)
        return self.output

    def close(self):
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build pinned HazyResearch CPU butterfly and benchmark-only adapter')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args = parser.parse_args()
    print(setup(args.root,fetch=args.fetch))
