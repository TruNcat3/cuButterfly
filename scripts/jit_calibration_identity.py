"""Bind opt-in calibration to immutable broker compilation inputs."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import semantic_compile_broker as broker


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _linkage(root):
    tree = ast.parse((Path(root) / 'scripts/compile_module.py').read_text())
    options = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and node.value.startswith('--cudart=')}
    if not options:
        return 'static-default'
    if options == {'--cudart=shared'}:
        return 'shared'
    if options == {'--cudart=static'}:
        return 'static'
    raise ValueError('ambiguous JIT CUDA runtime linkage')


def runtime_identity(document):
    """Validate current inputs against a broker and fingerprint linker inputs."""
    config = document['config']
    if (broker._sources(config) != document['sources_sha256']
            or broker._stat(config['nvcc']) != document['compiler_stat']):
        raise ValueError('JIT calibration broker inputs changed')
    nvcc = Path(config['nvcc']).resolve(strict=True)
    toolkit = nvcc.parent.parent
    linkage = _linkage(config['root'])
    names = ('libcudart.so',) if linkage == 'shared' else ('libcudart_static.a',)
    runtime_files = {}
    for name in names:
        directories = [toolkit / 'lib64', toolkit / 'targets/x86_64-linux/lib']
        if linkage == 'shared':
            directories = [Path(p) for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p] + directories
        selected = next((directory / name for directory in directories
                         if (directory / name).is_file()), None)
        if selected is None:
            raise ValueError('cannot identify JIT CUDA runtime input: ' + name)
        resolved = selected.resolve(strict=True)
        runtime_files[name] = {'path': str(resolved), 'sha256': _sha(resolved)}
    return {'schema': 'cubutterfly-jit-calibration-identity-v1',
            'broker_id': document['id'], 'config': dict(config),
            'sources_sha256': document['sources_sha256'],
            'generator_sha256': _sha(Path(config['root']) / 'scripts/compile_module.py'),
            'compiler_version': document['compiler_version'],
            'compiler_sha256': _sha(nvcc), 'cuda_runtime_linkage': linkage,
            'cuda_runtime_files': runtime_files,
            'ld_library_path': os.environ.get('LD_LIBRARY_PATH', '')}


def calibration_identity():
    """Historical calibration remains unchanged unless binding is requested."""
    directory = os.environ.get('CUBUTTERFLY_CALIBRATION_BROKER')
    if not directory:
        return None
    document = broker._document(Path(directory))
    config = document['config']
    for variable, field in (('CUBUTTERFLY_TEMPLATE_ROOT', 'root'),
                            ('CUBUTTERFLY_MATHDX_ROOT', 'mathdx'),
                            ('CUBUTTERFLY_NVCC', 'nvcc'),
                            ('CUBUTTERFLY_JIT_CACHE', 'cache')):
        selected = os.environ.get(variable)
        if not selected or Path(selected).resolve() != Path(config[field]).resolve():
            raise ValueError('calibration JIT environment differs from broker: ' + variable)
    return runtime_identity(document)


def validate_timing(calibration, document):
    current = runtime_identity(document)
    recorded = calibration.get('measurement_identity', {}).get('jit_runtime')
    if recorded is None:
        if current['cuda_runtime_linkage'] == 'shared':
            raise ValueError('shared JIT timing requires runtime-bound calibration')
        return current
    if recorded != current:
        raise ValueError('calibration JIT runtime identity mismatch')
    return current
