"""Intel HEXL CPU NTT, with explicit cyclic and storage adapters."""
import ctypes as ct
import ctypes.util
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from semantic_ntt_reference import primitive_power_of_two_root

ROOT = Path(__file__).resolve().parents[1] / "results/install_compare_20261013/external/hexl"
BRIDGE = Path(__file__).with_name('cpu_external_hexl_bridge.cpp')
COMMIT = 'f95acf1507b246946a34cda446f3a028fafb1c4b'


def _fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _loaded_identity(address):
    class Info(ct.Structure):
        _fields_ = [('filename',ct.c_char_p),('base',ct.c_void_p),('symbol',ct.c_char_p),('address',ct.c_void_p)]
    dl = ct.CDLL(ctypes.util.find_library('dl') or None)
    dl.dladdr.argtypes = [ct.c_void_p,ct.POINTER(Info)]
    info = Info()
    if not dl.dladdr(address,ct.byref(info)):
        raise RuntimeError('cannot identify loaded external artifact')
    return _fingerprint(info.filename.decode())


def _git_identity(root):
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    version = subprocess.check_output(['git','describe','--tags','--always'],cwd=root,text=True).strip()
    return commit, version


def setup(root, *, fetch=False):
    """Build pinned HEXL and the benchmark bridge under an explicit root."""
    root = Path(root).resolve()
    if not (root/'hexl/include/hexl/ntt/ntt.hpp').exists():
        if not fetch: raise RuntimeError('missing HEXL source; use setup(root, fetch=True)')
        subprocess.run(['git','clone','--depth','1','--branch','v1.2.5','https://github.com/intel/hexl.git',str(root)],check=True)
    commit, _ = _git_identity(root)
    if commit != COMMIT: raise RuntimeError('HEXL source must match pinned v1.2.5 commit')
    core = root/'build/hexl/libhexl.so'
    sources = list((root/'hexl').rglob('*.cpp')) + list((root/'hexl').rglob('*.hpp'))
    if not core.exists() or any(p.stat().st_mtime_ns > core.stat().st_mtime_ns for p in sources):
        subprocess.run(['cmake','-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DHEXL_TESTING=OFF','-DHEXL_BENCHMARK=OFF','-DHEXL_SHARED_LIB=ON'],check=True)
        subprocess.run(['cmake','--build',str(root/'build'),'-j8'],check=True)
    path = root/'libhexl_adapter.so'
    if not path.exists() or max(BRIDGE.stat().st_mtime_ns,core.stat().st_mtime_ns) > path.stat().st_mtime_ns:
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-I'+str(root/'hexl/include'),str(BRIDGE),'-L'+str(root/'build/hexl'),'-lhexl','-Wl,-rpath,'+str(root/'build/hexl'),'-o',str(path)],check=True)
    return path


def _packed(s):
    strides = []
    distance = 1
    for n in reversed(s['shape']):
        strides.insert(0, distance)
        distance *= n
    return (s['placement'] == 0 and s['input_strides'] == strides
            and s['output_strides'] == strides
            and s['input_batch_stride'] == distance
            and s['output_batch_stride'] == distance)


class NTTCore:
    def __init__(self, s, source, root=None):
        self.root = Path(root or ROOT).resolve()
        if s['operator'] != 1 or s['storage'] not in (8, 9) or s['length_mode'] or any(s['input_orders']) or any(s['output_orders']) or not _packed(s) or s['ntt_input_layout'] or s['ntt_output_layout']:
            raise ValueError('HEXL adapter requires packed natural unsigned cyclic NTT')
        self.s = s
        self.source = np.asarray(source)
        dtype = np.dtype(np.uint32 if s['storage'] == 8 else np.uint64)
        if type(s.get('compute')) is not int or s['compute'] not in (0,3 if s['storage']==8 else 4):
            raise ValueError('NTT compute must be native or exact modular integer')
        if not isinstance(source,np.ndarray) or self.source.dtype != dtype or self.source.shape != (s['batch'], *s['shape']) or not self.source.flags.c_contiguous:
            raise ValueError('source must be C-contiguous with matching shape and exact storage')
        if np.any(self.source >= s['modulus']):
            raise ValueError('HEXL input must contain canonical residues')
        self.output = np.empty_like(source)
        path = setup(self.root)
        self.lib = ct.CDLL(str(path))
        self.lib.hexl_create.argtypes = [ct.c_uint64]*3
        self.lib.hexl_create.restype = ct.c_void_p
        self.lib.hexl_execute.argtypes = [ct.c_void_p]*3+[ct.c_uint64, ct.c_int, ct.c_int]
        self.lib.hexl_close.argtypes = [ct.c_void_p]
        self.lib.hexl_library_address.restype = ct.c_void_p
        self.plans = []
        roots = []
        q = int(s['modulus'])
        for n in s['shape']:
            omega = primitive_power_of_two_root(q, n)
            psi = primitive_power_of_two_root(q, 2*n)
            if pow(psi, 2, q) != omega:
                raise ValueError('HEXL square-root adapter cannot match requested cyclic root')
            handle = self.lib.hexl_create(n, q, psi)
            if not handle:
                raise RuntimeError('HEXL plan initialization failed')
            self.plans.append(handle)
            roots.append({'n': n, 'cyclic_root': omega, 'negacyclic_root': psi})
        self.metadata = dict(backend='intel-hexl-cpu', library=str(path), upstream='https://github.com/intel/hexl', upstream_commit='f95acf1507b246946a34cda446f3a028fafb1c4b', external_core_storage='uint64', requested_storage=str(dtype), storage_conversion=s['storage']==8, precision_status='adapted-storage-widening' if s['storage']==8 else 'native-word64', conversion='uint32 -> uint64 exact widening -> uint32 exact narrowing' if s['storage']==8 else 'none', roots=roots, threads=1, planning_excluded=True, timing_scope='axis packing, width conversion, cyclic twist, HEXL execute, bit reversal, inverse normalization and output copy')
        self.metadata.update(library_identity=_loaded_identity(ct.cast(self.lib.hexl_execute,ct.c_void_p)), external_library=_loaded_identity(self.lib.hexl_library_address()), bridge_source=_fingerprint(BRIDGE), upstream_version=_git_identity(self.root)[1], upstream_sources=[_fingerprint(p) for p in sorted((self.root/'hexl').rglob('*')) if p.suffix in ('.cpp','.hpp')], adapter_source=_fingerprint(__file__), native_word32=False, all_storage_adapter_overheads_timed=True)

    def execute(self):
        value = self.source
        for axis, handle in enumerate(self.plans, 1):
            moved = np.moveaxis(value, axis, -1)
            rows = np.ascontiguousarray(moved, dtype=np.uint64).reshape(-1, moved.shape[-1])
            out = np.empty_like(rows)
            self.lib.hexl_execute(handle, rows.ctypes.data, out.ctypes.data, len(rows), int(self.s['inverse']), int(self.s['normalize_inverse']))
            value = np.moveaxis(out.reshape(moved.shape), -1, axis)
        np.copyto(self.output, value, casting='unsafe')
        return self.output

    def close(self):
        for handle in self.plans:
            self.lib.hexl_close(handle)
        self.plans = []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build pinned HEXL and benchmark-only CPU adapter')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args = parser.parse_args()
    print(setup(args.root,fetch=args.fetch))
