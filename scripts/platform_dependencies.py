"""CPU-only, conservative provisioning for the existing platform runner.

Public functions accept the runner's (command, env=None, label=...) callback.
Existing sources are never reset and existing binaries are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GPUNTT = 'd03c5eaeadaa780d153496afcb3b6a9b79a13a63'
DAO = 'e7706faf8d1c3b9f241e36860640ad1dac644ede'
VKFFT = '066a17c17068c0f11c9298d848c2976c71fad1c1'


def _env(args, **extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(args.nvcc.parent.parent),
               PATH=os.pathsep.join((str(args.prefix / 'bin'), str(args.nvcc.parent),
                                    os.environ.get('PATH', ''))), **extra)
    for name in ('LD_PRELOAD', 'CUBUTTERFLY_HARDWARE_MODEL', 'CUBUTTERFLY_PROFILE_ROOT',
                 'CUBUTTERFLY_HARDWARE_PROFILE', 'CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_REGISTRY',
                 'CUBUTTERFLY_PREPARE_BROKER', 'CUBUTTERFLY_CALIBRATION_BROKER',
                 'CUBUTTERFLY_JIT_CACHE_ONLY', 'CUBUTTERFLY_COMPILE_BUDGET_FILE',
                 'CUBUTTERFLY_COMPILE_BUDGET_SECONDS', 'CUBUTTERFLY_PREPARE_EVENTS'):
        env.pop(name, None)
    return env


def _read(command, args):
    return subprocess.check_output([str(x) for x in command], env=_env(args), text=True).strip()


def _local(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    forbidden = (Path('/sdb'), Path('/mnt/sdb'))
    if any(p == path or p in path.parents for p in forbidden):
        raise ValueError('Dependency artifacts must not use /sdb or /mnt/sdb')
    resolved = path.resolve()
    if any(p == resolved or p in resolved.parents for p in forbidden):
        raise ValueError('Dependency artifacts must not use /sdb or /mnt/sdb')
    return resolved


def _hash(path):
    path = _local(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _setup(target):
    return dict(python_version='3.11', torch_index_url='https://download.pytorch.org/whl/cu128',
                gpuntt_revision=GPUNTT, dao_revision=DAO, vkfft_revision=VKFFT,
                ) | target.get('setup', {})


def _source(args, run, directory, url, revision):
    directory = _local(directory)
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('baseline revision must be a complete pinned Git commit')
    if directory.exists():
        if not (directory / '.git').exists():
            raise ValueError(f'Existing source is not a Git checkout: {directory}; choose a fresh source directory')
        head = _read(['git', '-C', directory, 'rev-parse', 'HEAD'], args)
        dirty = _read(['git', '-C', directory, 'status', '--porcelain'], args)
        if head != revision or dirty:
            raise ValueError(f'Pinned source differs or is dirty: {directory}; preserve it and choose a fresh directory')
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    run(['git', 'clone', '--filter=blob:none', url, str(directory)], env=_env(args), label='clone-' + directory.name)
    # Only a newly created checkout may be changed.
    run(['git', '-C', str(directory), 'checkout', '--detach', revision], env=_env(args), label='pin-' + directory.name)
    if (_read(['git', '-C', directory, 'rev-parse', 'HEAD'], args) != revision
            or _read(['git', '-C', directory, 'status', '--porcelain'], args)):
        raise ValueError(f'Could not prepare clean pinned source: {directory}')


def _packages(args):
    probe = '''import importlib.metadata as m,json
names=['cmake','ninja','packaging','wheel','setuptools','numpy','torch','fast-hadamard-transform']
out={}
for name in names:
 try: out[name]=m.version(name)
 except m.PackageNotFoundError: out[name]=None
print(json.dumps(out))'''
    return json.loads(_read([args.python, '-c', probe], args))


def _torch_target(args, target):
    flags = _read([args.python, '-c', "import torch; print(getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or '')"], args)
    if f'sm_{target["cuda_arch"]}' not in flags.split():
        raise ValueError(f'Torch does not contain sm_{target["cuda_arch"]}; select a compatible target environment. Existing Torch was preserved.')


def prepare_environment(args, target, run):
    """Provision absent dependencies without upgrading an existing environment."""
    setup = _setup(target)
    if not (args.prefix / 'conda-meta').is_dir():
        if args.prefix.exists():
            raise ValueError(f'Existing prefix is not a conda environment: {args.prefix}')
        conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
        if not conda:
            raise ValueError('Conda is required to create the target prefix; set CONDA_EXE')
        run([conda, 'create', '-p', str(args.prefix), 'python=' + setup['python_version'], 'pip', '-y'],
            env=_env(args), label='create-environment')
    packages = _packages(args)
    requirements = {'cmake': 'cmake', 'ninja': 'ninja', 'packaging': 'packaging',
                    'wheel': 'wheel', 'setuptools': 'setuptools<70', 'numpy': 'numpy<2'}
    missing = [spec for name, spec in requirements.items() if not packages.get(name)]
    if missing:
        run([str(args.python), '-m', 'pip', 'install', *missing], env=_env(args), label='python-prerequisites')
    if not packages.get('torch'):
        run([str(args.python), '-m', 'pip', 'install', 'torch', '--index-url', setup['torch_index_url']],
            env=_env(args), label='torch')
    _torch_target(args, target)
    version = target.get('mathdx_version', '25.6.0')
    # install_cufftdx.sh expects the pip --target root, not the include root.
    if args.mathdx.parts[-2:] != ('nvidia', 'mathdx'):
        raise ValueError('--mathdx must name the package include root <destination>/nvidia/mathdx')
    destination = args.mathdx.parent.parent
    header = args.mathdx / 'include/cufftdx.hpp'
    if destination.exists():
        metadata = list(destination.glob('nvidia_mathdx-*.dist-info/METADATA'))
        if not header.is_file() or not any(f'Version: {version}' in p.read_text().splitlines() for p in metadata):
            raise ValueError(f'Existing MathDx is incomplete or differs from {version}: {destination}; choose a fresh destination')
    else:
        run(['bash', str(ROOT / 'scripts/install_cufftdx.sh')],
            env=_env(args, MATHDX_ROOT=str(destination), MATHDX_VERSION=version, PYTHON=str(args.python)), label='mathdx')
    _source(args, run, ROOT / 'external/VkFFT', 'https://github.com/DTolm/VkFFT.git', setup['vkfft_revision'])


def prepare_baseline_sources(args, target, run):
    setup = _setup(target)
    if setup['gpuntt_revision'] != GPUNTT:
        raise ValueError('Existing GPU-NTT builder requires revision ' + GPUNTT)
    _source(args, run, ROOT / ('external/GPU-NTT-' + setup['gpuntt_revision']),
            'https://github.com/Alisah-Ozcan/GPU-NTT.git', setup['gpuntt_revision'])
    _source(args, run, ROOT / ('external/fast-hadamard-transform-' + setup['dao_revision']),
            'https://github.com/Dao-AILab/fast-hadamard-transform.git', setup['dao_revision'])


def _identity(args, target, revision):
    return dict(revision=revision, cuda_arch=target['cuda_arch'], python=str(args.python),
                python_sha256=_hash(args.python), nvcc=str(args.nvcc), nvcc_sha256=_hash(args.nvcc))


def _gpuntt_sources():
    return [ROOT / name for name in (
        'benchmarks/gpuntt_crypto/semantic_bridge.cu',
        'benchmarks/gpuntt_crypto/build_semantic_bridge.sh',
        'benchmarks/gpuntt_crypto/CMakeLists.txt', 'scripts/semantic_gpuntt_baseline.py',
        'src/reference.cpp', 'include/cuntt/ntt.hpp')]


def _gpuntt_identity(args, target):
    return dict(_identity(args, target, _setup(target)['gpuntt_revision']),
                helper_sha256=_hash(args.build_dir / 'libcuntt.so'),
                sources_sha256={str(path): _hash(path) for path in _gpuntt_sources()})


def _validate_gpuntt(library, helper, arch, nvcc):
    record = json.loads(Path(str(library) + '.manifest.json').read_text())
    if (record.get('schema') != 'cubutterfly-gpuntt-build-v1' or record.get('vendor_revision') != GPUNTT
            or record.get('architecture') != f'sm{arch} + compute{arch} PTX'
            or Path(record['root_helper_library']['path']) != helper
            or Path(record['binary']['path']) != library or not record.get('sources')):
        raise ValueError('GPU-NTT manifest identity differs; use a fresh build directory')
    command = shlex.split(record.get('build_command', ''))
    flags = [f'--generate-code=arch=compute_{arch},code=sm_{arch}',
             f'--generate-code=arch=compute_{arch},code=compute_{arch}']
    recorded_sources = {entry['path'] for entry in record['sources']}
    if (not command or command[0] != str(nvcc) or not all(flag in command for flag in flags)
            or not {str(path) for path in _gpuntt_sources()}.issubset(recorded_sources)):
        raise ValueError('GPU-NTT native compiler/target/source provenance differs')
    for entry in [record['binary'], record['vendor_archive'], record['root_helper_library'], *record['sources']]:
        if _hash(entry['path']) != entry['sha256'] or Path(entry['path']).stat().st_size != entry['bytes']:
            raise ValueError('GPU-NTT input/output changed: ' + entry['path'])


def build_baselines(args, target, run):
    """Build pinned comparators after libcuntt.so exists; return library Paths."""
    setup = _setup(target)
    helper = args.build_dir / 'libcuntt.so'
    identity = _gpuntt_identity(args, target)
    standard = args.build_dir / 'libgpuntt_semantic_bridge.so'

    def reuse(path, adopt=False):
        receipt_path = Path(str(path) + '.platform.json')
        try:
            if not path.is_file() or not Path(str(path) + '.manifest.json').is_file():
                return False
            _validate_gpuntt(path, helper, target['cuda_arch'], args.nvcc)
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                return saved.get('identity') == identity
            if adopt:
                _write(receipt_path, dict(identity=identity,
                    provenance='adopted native CPU build provenance; no calibration or GPU validation'))
                return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return False

    if reuse(standard, adopt=True):
        library = standard
    else:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        directory = args.build_dir / 'baselines' / key
        library = directory / 'libgpuntt_semantic_bridge.so'
        attempt = 0
        while not reuse(library, adopt=True) and any(
                path.exists() or path.is_symlink() for path in
                (library, Path(str(library) + '.manifest.json'), Path(str(library) + '.platform.json'))):
            # Preserve even interrupted content-addressed attempts. Never replace
            # their binary or receipts when retrying the same inputs.
            attempt += 1
            directory = args.build_dir / 'baselines' / key / ('retry-' + str(attempt))
            library = directory / 'libgpuntt_semantic_bridge.so'
        if not reuse(library, adopt=True):
            run(['bash', str(ROOT / 'benchmarks/gpuntt_crypto/build_semantic_bridge.sh')], env=_env(args,
                CUDA_ARCH=str(target['cuda_arch']), NVCC=str(args.nvcc), PYTHON=str(args.python),
                BUILD_JOBS=str(args.jobs), CUBUTTERFLY_BUILD_DIR=str(args.build_dir), CUNTT_LIBRARY=str(helper),
                GPUNTT_SOURCE_DIR=str(ROOT / ('external/GPU-NTT-' + setup['gpuntt_revision'])),
                GPUNTT_BUILD_DIR=str(directory / 'build'), GPUNTT_OUTPUT=str(library)), label='gpuntt-build')
            _validate_gpuntt(library, helper, target['cuda_arch'], args.nvcc)
            _write(Path(str(library) + '.platform.json'), dict(identity=identity,
                provenance='built by platform dependency helper; no calibration or GPU validation'))
    dao_receipt = args.prefix / 'share/cuButterfly/dao-platform-build.json'
    dao_identity = _identity(args, target, setup['dao_revision'])
    packages = _packages(args)
    reuse_dao = False
    if dao_receipt.exists():
        try:
            record = json.loads(dao_receipt.read_text())
            reuse_dao = (record.get('identity') == dao_identity
                         and bool(packages.get('fast-hadamard-transform'))
                         and bool(record.get('artifacts'))
                         and all(_hash(entry['path']) == entry['sha256'] for entry in record['artifacts']))
        except (OSError, KeyError, json.JSONDecodeError):
            reuse_dao = False
    if not reuse_dao:
        source = ROOT / ('external/fast-hadamard-transform-' + setup['dao_revision'])
        run([str(args.python), '-m', 'pip', 'install', '--no-build-isolation', '--no-cache-dir', '--no-deps', '--force-reinstall', str(source)],
            env=_env(args, FAST_HADAMARD_TRANSFORM_FORCE_BUILD='TRUE', MAX_JOBS=str(args.jobs),
                     TORCH_CUDA_ARCH_LIST=str(target['cuda_arch'] / 10)), label='dao-build')
        # Read installed distribution files without importing its CUDA extension.
        probe = "import importlib.metadata as m,json; d=m.distribution('fast-hadamard-transform'); print(json.dumps([str(d.locate_file(p).resolve()) for p in d.files if str(p).endswith('.so')]))"
        artifacts = json.loads(_read([args.python, '-c', probe], args))
        if not artifacts:
            raise ValueError('Dao installation has no compiled extension; no receipt published')
        _write(dao_receipt, dict(identity=dao_identity, artifacts=[dict(path=p, sha256=_hash(p)) for p in artifacts]))
    vkfft = args.build_dir / 'libvkfft_xd_bridge.so'
    if not vkfft.is_file():
        raise ValueError('VkFFT bridge is missing; build the main library with --with-vkfft')
    return dict(vkfft_library=vkfft, gpuntt_library=library)
