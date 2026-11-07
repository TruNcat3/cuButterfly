#!/usr/bin/env python3
"""Bind existing CPU preparation and calibrated comparison to a target platform.

plan/doctor/build/prepare are CPU-only. run checks the selected device before
delegating calibration, confirmation and timing to run_full_semantic_refresh.
This finite operator track is not the entire paper's migration certificate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def safe_path(value):
    path = Path(os.path.abspath(Path(value).expanduser()))
    forbidden = (Path('/sdb'), Path('/mnt/sdb'))
    # Reject the spelling before resolving symlinks or inspecting that mount.
    if any(p == path or p in path.parents for p in forbidden):
        raise ValueError('select local SSD or NAS, not /sdb or /mnt/sdb')
    resolved = path.resolve()
    if any(p == resolved or p in resolved.parents for p in forbidden):
        raise ValueError('select local SSD or NAS, not /sdb or /mnt/sdb')
    return resolved


def config(path):
    value = read(path)
    if value.get('schema') != 'cubutterfly-research-platform-v1':
        raise ValueError('unsupported platform schema')
    if type(value['cuda_arch']) is not int or value['cuda_arch'] <= 0:
        raise ValueError('cuda_arch must be a positive integer')
    re.compile(value['device_name_pattern'])
    scope = read(ROOT / value['scope'])
    from semantic_campaign_manifest import load_config
    for mode in ('quick', 'full'):
        item = value['modes'][mode]
        source = ROOT / item['config']
        load_config(source)
        if sha(source) != scope['files'][source.name]:
            raise ValueError('frozen paper population differs: ' + mode)
    return value


def environment(args, gpu=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu if gpu else '',
               CUBUTTERFLY_COMPILE_MODE='research')
    for name in ('CUBUTTERFLY_HARDWARE_MODEL', 'CUBUTTERFLY_PROFILE_ROOT',
                 'CUBUTTERFLY_HARDWARE_PROFILE', 'CUBUTTERFLY_COST_MODEL',
                 'CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_PREPARE_BROKER',
                 'CUBUTTERFLY_CALIBRATION_BROKER', 'CUBUTTERFLY_JIT_CACHE_ONLY',
                 'CUBUTTERFLY_PREPARE_EVENTS', 'CUBUTTERFLY_COMPILE_BUDGET_FILE',
                 'CUBUTTERFLY_COMPILE_BUDGET_SECONDS', 'LD_PRELOAD'):
        env.pop(name, None)
    env.update(CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT), CUBUTTERFLY_MATHDX_ROOT=str(args.mathdx),
               CUBUTTERFLY_NVCC=str(args.nvcc), CUBUTTERFLY_JIT_CACHE=str(args.output / 'modules'))
    env['CUDA_HOME'] = str(args.nvcc.parent.parent)
    env['PATH'] = str(args.nvcc.parent) + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = str(args.build_dir) + ':' + str(args.nvcc.parent.parent / 'lib64') + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


@contextmanager
def process_environment(env):
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def commands(args, target):
    mode = target['modes'][args.mode]
    folder = args.output / args.mode
    python = str(args.python)
    build = ['bash', str(ROOT / 'scripts/prepare_research_target.sh'),
             '--build-dir', str(args.build_dir), '--cuda-arch', str(target['cuda_arch']),
             '--prefix', str(args.prefix), '--python', python, '--nvcc', str(args.nvcc),
             '--mathdx-root', str(args.mathdx), '--jobs', str(args.jobs),
             '--shared', '--fft-codegen-mode', 'on-demand', '--install']
    if args.with_vkfft:
        build.append('--with-vkfft')
    if args.json_root:
        build += ['--json-root', str(args.json_root)]
    refresh = [python, str(ROOT / 'scripts/run_full_semantic_refresh.py'),
               '--build-dir', str(args.build_dir), '--output-dir', str(folder / 'run'),
               '--manifest', str(folder / 'manifest.json'), '--preparation', str(folder / 'prepared'),
               '--gpu-uuid', args.gpu or 'GPU-REQUIRED-AT-RUN',
               '--workloads', str(ROOT / target['workloads']),
               '--mathdx', str(args.mathdx), '--nvcc', str(args.nvcc),
               '--jit-cache', str(args.output / 'modules'), '--compile-jobs', str(args.jobs),
               '--cpu-workers', str(args.jobs)]
    for key in ('search_budget', 'comparison_search_budget', 'stage_point_budget'):
        refresh += ['--' + key.replace('_', '-'), str(mode[key])]
    for key in ('vkfft_library', 'gpuntt_library'):
        if getattr(args, key):
            refresh += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return dict(build=build,
        python_install=[python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(ROOT / 'python')],
        manifest=[python, str(ROOT / 'scripts/semantic_campaign_manifest.py'), '--config', str(ROOT / mode['config']),
                  '--library', str(args.build_dir / 'libcuntt.so'), '--output', str(folder / 'manifest.json')],
        references=[python, str(ROOT / 'scripts/semantic_preparation_queue.py'), '--manifest', str(folder / 'manifest.json'),
                    '--directory', str(folder / 'prepared'), '--jobs', str(args.jobs)],
        run=refresh,
        report=[python, str(ROOT / 'scripts/summarize_semantic_campaign.py'),
                '--results', str(folder / 'run/measurement/results.json'), '--output', str(folder / 'report/summary.json')])


def device_matches(target, observed):
    if int(observed['compute_cap'].replace('.', '')) != target['cuda_arch']:
        raise ValueError('device compute capability differs from target/build')
    if not re.search(target['device_name_pattern'], observed['name'], re.IGNORECASE):
        raise ValueError('selected GPU model differs from declared platform')
    lower, upper = target['memory_mib_range']
    if not lower <= int(observed['memory_mib']) <= upper:
        raise ValueError('selected GPU memory differs from declared platform')


def detect(args, target):
    if not args.gpu or not args.gpu.startswith('GPU-'):
        raise ValueError('--gpu must be an explicit physical GPU UUID')
    raw = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
        '--query-gpu=uuid,name,compute_cap,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True).strip()
    lines = raw.splitlines()
    if len(lines) != 1:
        raise ValueError('expected exactly one physical GPU')
    uuid, name, capability, memory, driver = [v.strip() for v in lines[0].split(',')]
    observed = dict(uuid=uuid, name=name, compute_cap=capability, memory_mib=int(memory), driver=driver)
    if uuid != args.gpu:
        raise ValueError('GPU UUID mismatch')
    device_matches(target, observed)
    return observed


def doctor(args, target):
    checks = []
    def item(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))
    item('conda_prefix', (args.prefix / 'conda-meta').is_dir(), str(args.prefix))
    item('python', args.python.is_file(), str(args.python))
    item('mathdx', (args.mathdx / 'include/cufftdx.hpp').is_file(), str(args.mathdx))
    try:
        codes = subprocess.check_output([str(args.nvcc), '--list-gpu-code'],
                                       text=True, env=environment(args), stderr=subprocess.STDOUT)
        item('nvcc_target', f'sm_{target["cuda_arch"]}' in codes.split(), str(args.nvcc))
    except (OSError, subprocess.CalledProcessError) as error:
        item('nvcc_target', False, str(error))
    # Compiled-in Torch metadata only; never initialize CUDA or query a device.
    probe = "import importlib.metadata as m,json; names=['numpy','torch','cupy-cuda12x','cupy-cuda13x','fast-hadamard-transform']; out={}; exec(\"for n in names:\\n try: out[n]=m.version(n)\\n except m.PackageNotFoundError: out[n]=None\"); print(json.dumps(out))"
    try:
        packages = json.loads(subprocess.check_output([str(args.python), '-c', probe], text=True, env=environment(args)))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        packages = dict(error=str(error))
    item('numpy', bool(packages.get('numpy')), packages)
    item('torch', bool(packages.get('torch')), packages.get('torch'))
    torch_probe = '''import json, torch
print(json.dumps({'cuda': torch.version.cuda,
                  'arch_flags': getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or ''}))'''
    try:
        torch_build = json.loads(subprocess.check_output([str(args.python), '-c', torch_probe],
            text=True, env=environment(args), stderr=subprocess.PIPE))
        item('torch_target', f'sm_{target["cuda_arch"]}' in torch_build['arch_flags'].split(), torch_build)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        item('torch_target', False, str(error))
    baseline_gaps = []
    if not packages.get('fast-hadamard-transform'):
        baseline_gaps.append('Dao FWHT package is absent; build it for this target')
    for name in ('vkfft_library', 'gpuntt_library'):
        path = getattr(args, name)
        if not path or not path.is_file():
            baseline_gaps.append(name + ' is not prepared')
    return dict(status='cpu-prerequisites-present' if all(c['ok'] for c in checks) else 'blocked-prerequisites',
        checks=checks, baseline_gaps=baseline_gaps, gpu_access=False,
        scope='Metadata/compiler-target check only; build runs a real cuFFTDx compile probe; runtime remains unverified')


def precompile(args, target):
    from research_compile_requests import compile_request
    from compile_module import compile_mapping
    modules = {}
    for point in read(ROOT / 'config/research_campaign/stage_points.json')['points']:
        request = compile_request(dict(point, mapping_json=point['mapping']), 'research')
        if request is None:
            continue
        key = json.dumps(request, sort_keys=True)
        if key in modules:
            continue
        module = compile_mapping(request, root=ROOT, mathdx=args.mathdx, nvcc=args.nvcc,
                                 sm=target['cuda_arch'], cache=args.output / 'modules')
        modules[key] = dict(path=str(module), sha256=sha(module), request=request)
    write(args.output / args.mode / 'precompiled.json', dict(status='cpu-compiled',
        cuda_arch=target['cuda_arch'], modules=list(modules.values()),
        scope='Declared stage proposals only; generated candidates can require additional CPU compilation'))


def validate_precompiled(path, arch):
    receipt = read(path)
    if receipt.get('status') != 'cpu-compiled' or receipt.get('cuda_arch') != arch:
        raise ValueError('prepared module receipt differs from target; run prepare again')
    for module in receipt.get('modules', []):
        artifact = safe_path(module['path'])
        if not artifact.is_file() or sha(artifact) != module['sha256']:
            raise ValueError('prepared module missing or changed; run prepare again')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'doctor', 'build', 'prepare', 'run', 'report'))
    parser.add_argument('--target', type=Path, default=ROOT / 'config/platforms/pro6000_server96.json')
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--prefix', type=Path, default=Path(sys.prefix))
    parser.add_argument('--python', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda-12.8/bin/nvcc'))
    parser.add_argument('--mathdx', type=Path, default=ROOT / 'external/mathdx-25.6.0/nvidia/mathdx')
    parser.add_argument('--json-root', type=Path)
    parser.add_argument('--with-vkfft', action='store_true')
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--allow-baseline-gaps', action='store_true')
    parser.add_argument('--gpu')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(argv)
    args.target = safe_path(args.target)
    target = config(args.target)
    args.build_dir = args.build_dir or ROOT / f'build-{target["id"]}-sm{target["cuda_arch"]}'
    args.output = args.output or ROOT / 'results' / target['id']
    args.python = args.python or args.prefix / 'bin/python'
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, safe_path(value))
    if args.jobs < 1:
        raise ValueError('--jobs must be positive')
    plan = commands(args, target)
    if args.action == 'plan':
        print(json.dumps(dict(target=target, commands=plan, gpu_access=False,
            scope='Main operator comparison only; full is 66 requests, not all paper tracks'), indent=2))
        return 0
    if args.action == 'doctor':
        result = doctor(args, target);print(json.dumps(result, indent=2))
        return 0 if result['status']=='cpu-prerequisites-present' else 2
    if args.action == 'report':
        subprocess.run(plan['report'], check=True, env=environment(args));return 0
    if args.action == 'build':
        if not (args.prefix / 'conda-meta').is_dir():
            raise ValueError('prepare the shared conda prefix first')
        for step in ('build', 'python_install'):
            subprocess.run(plan[step], check=True, env=environment(args), cwd=ROOT)
        return 0
    from run_current_recalibration import validate_build
    if validate_build(args.build_dir) != target['cuda_arch']:
        raise ValueError('build architecture differs from platform configuration')
    folder = args.output / args.mode
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(target_sha256=sha(args.target), config_sha256=sha(ROOT / target['modes'][args.mode]['config']),
        library_sha256=sha(args.build_dir / 'libcuntt.so'), nvcc=str(args.nvcc), mathdx=str(args.mathdx),
        cuda_arch=target['cuda_arch'], prefix=str(args.prefix), python=str(args.python))
    bound = folder / 'binding.json'
    if bound.exists() and read(bound) != binding:
        raise ValueError('prepared build/config changed; use a fresh output directory')
    write(bound, binding)
    if args.action == 'prepare':
        if not (folder / 'manifest.json').exists():
            subprocess.run(plan['manifest'], check=True, env=environment(args), cwd=ROOT)
        from run_semantic_campaign import validate_fingerprints
        manifest = read(folder / 'manifest.json')
        validate_fingerprints(manifest, args.build_dir / 'libcuntt.so')
        if manifest['metrics']['unique_cases'] != target['modes'][args.mode]['requests']:
            raise ValueError('prepared request count differs from declared scope')
        subprocess.run(plan['references'], check=True, env=environment(args), cwd=ROOT)
        with process_environment(environment(args)):
            precompile(args, target)
        return 0
    if not (folder / 'precompiled.json').is_file():
        raise ValueError('run prepare first')
    validate_precompiled(folder / 'precompiled.json', target['cuda_arch'])
    observed = detect(args, target)
    readiness = doctor(args, target)
    if readiness['status'] != 'cpu-prerequisites-present':
        raise ValueError('CPU/toolchain prerequisites are missing: ' + json.dumps(readiness))
    if readiness['baseline_gaps'] and not args.allow_baseline_gaps:
        raise ValueError('baseline prerequisites missing: ' + '; '.join(readiness['baseline_gaps']) +
                         '; prepare them or explicitly use --allow-baseline-gaps for a diagnostic')
    write(folder / 'device.json', observed)
    write(folder / 'dependencies.json', readiness)
    refresh = plan['run']
    if (folder / 'run/status.json').exists():
        refresh += ['--resume']
    subprocess.run(refresh + ['--prepare-only'], check=True, env=environment(args, gpu=True), cwd=ROOT)
    if '--resume' not in refresh:
        refresh += ['--resume']
    subprocess.run(refresh, check=True, env=environment(args, gpu=True), cwd=ROOT)
    subprocess.run(plan['report'], check=True, env=environment(args), cwd=ROOT)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
