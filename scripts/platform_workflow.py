"""Installation orchestration for research_platform; measurements stay in its runner."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

import research_platform as platform
import platform_dependencies as dependencies

ROOT = platform.ROOT


class WorkflowCancelled(RuntimeError):
    pass


def live_cpu_group(group):
    """Ignore unreaped zombies; wait for active descendants of our CPU group."""
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == group:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def entry_command(args, action):
    command = [str(args.python), str(ROOT / 'scripts/research_platform.py'), action]
    for key in ('target', 'mode', 'prefix', 'python', 'build_dir', 'output', 'nvcc',
                'mathdx', 'json_root', 'vkfft_library', 'gpuntt_library', 'gpu', 'jobs'):
        value = getattr(args, key, None)
        if value is not None:
            command += ['--' + key.replace('_', '-'), str(value)]
    if args.with_vkfft:
        command.append('--with-vkfft')
    if args.allow_baseline_gaps:
        command.append('--allow-baseline-gaps')
    return command


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another workflow owns {path}; follow its status instead of starting it twice') from error
        yield


class Commands:
    """Log independent CPU jobs; propagate cancellation only to owned children."""
    def __init__(self, args):
        self.args = args
        self.children = set()
        self.mutex = threading.Lock()
        self.cancelled = threading.Event()

    def stop(self):
        self.cancelled.set()
        with self.mutex:
            children = list(self.children)
        for child in children:
            if child.poll() is None:
                # Watcher/refresh own their cleanup handlers. CPU children have
                # no such handler; terminate their own process group as well.
                try:
                    if child.platform_gpu_driver:
                        child.terminate()
                    else:
                        os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in children:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('Owned driver cleanup did not finish; inspect its log before resuming') from error
            if not child.platform_gpu_driver:
                deadline = time.monotonic() + 5
                while live_cpu_group(child.pid) and time.monotonic() < deadline:
                    time.sleep(.05)
                if live_cpu_group(child.pid):
                    # These descendants share the new session we created;
                    # never send this escalation to a GPU watcher/driver.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 5
                    while live_cpu_group(child.pid) and time.monotonic() < deadline:
                        time.sleep(.05)
                    if live_cpu_group(child.pid):
                        raise RuntimeError('Owned CPU process group did not exit')

    def __call__(self, command, env=None, label='step', gpu_driver=False):
        if self.cancelled.is_set():
            raise WorkflowCancelled('workflow cancelled')
        path = self.args.output / 'workflow/logs' / (label + '.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{label}] {shlex.join([str(x) for x in command])}\n  log: {path}', flush=True)
        with path.open('a') as log:
            log.write('\n' + shlex.join([str(x) for x in command]) + '\n'); log.flush()
            with self.mutex:
                if self.cancelled.is_set():
                    raise WorkflowCancelled('workflow cancelled')
                child = subprocess.Popen([str(x) for x in command], cwd=ROOT,
                    env=env or platform.environment(self.args), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child.platform_gpu_driver = gpu_driver
                self.children.add(child)
            try:
                while True:
                    if self.cancelled.is_set():
                        self.stop()
                        raise WorkflowCancelled('workflow cancelled')
                    try:
                        code = child.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                with self.mutex:
                    self.children.discard(child)
        if code:
            raise subprocess.CalledProcessError(code, command)


def build_identity(args, target):
    # Source inputs, not a Git HEAD alone: a dirty research checkout must not
    # silently reuse a binary compiled before an edit. Do not traverse results.
    files = [ROOT / 'CMakeLists.txt']
    extensions = {'.cpp', '.cu', '.cuh', '.hpp', '.h', '.cmake', '.in', '.py', '.json'}
    for name in ('src', 'include', 'cmake', 'config'):
        base = platform.safe_path(ROOT / name)
        files += [path for path in base.rglob('*') if path.suffix in extensions and path.is_file()]
    files += [ROOT / 'scripts/prepare_research_target.sh']
    source = hashlib.sha256()
    for path in sorted(set(files)):
        platform.safe_path(path)
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(platform.sha(path).encode())
    return dict(source_sha256=source.hexdigest(), cuda_arch=target['cuda_arch'],
        prefix=str(args.prefix), python=str(args.python), nvcc=str(args.nvcc),
        nvcc_sha256=platform.sha(args.nvcc), mathdx=str(args.mathdx),
        mathdx_header_sha256=platform.sha(args.mathdx / 'include/cufftdx.hpp'),
        json_root=str(args.json_root) if args.json_root else None, with_vkfft=True)


def ensure_build(args, target, run):
    receipt = args.output / 'workflow/build.json'
    identity = build_identity(args, target)
    library = args.build_dir / 'libcuntt.so'
    if receipt.exists():
        previous = platform.read(receipt)
        if previous.get('identity') != identity:
            raise ValueError('Build inputs changed; preserve this run and choose a fresh --output and --build-dir')
        for item in previous.get('artifacts', []):
            path = platform.safe_path(item['path'])
            if not path.is_file() or platform.sha(path) != item['sha256']:
                raise ValueError('Prepared build artifact changed: ' + str(path))
        if not previous.get('artifacts'):
            raise ValueError('Build receipt has no artifacts')
        print('[build] reusing verified target build', flush=True)
        return
    # An already bound manual run cannot be rebuilt behind the runner's back.
    if (args.output / args.mode / 'binding.json').exists():
        raise ValueError('Existing manually prepared run has no workflow build receipt; keep using run, or choose fresh --output and --build-dir for all')
    run(entry_command(args, 'build'), label='build-install')
    artifacts = [library, args.build_dir / 'libvkfft_xd_bridge.so', args.build_dir / 'CMakeCache.txt']
    platform.write(receipt, dict(identity=identity,
        artifacts=[dict(path=str(p), sha256=platform.sha(p)) for p in artifacts]))


def select_devices(args, target):
    """Resolve physical selectors once; downstream commands use exact UUIDs."""
    requested = args.gpus.split(',') if args.gpus else ([args.gpu] if args.gpu else [])
    if requested and any(not selector.strip() for selector in requested):
        raise ValueError('empty GPU selector')
    if not requested:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,compute_cap,memory.total',
            '--format=csv,noheader,nounits'], text=True)
        for line in raw.strip().splitlines():
            uuid, name, capability, memory = [value.strip() for value in line.split(',')]
            try:
                platform.device_matches(target, dict(name=name, compute_cap=capability, memory_mib=int(memory)))
            except ValueError:
                continue
            requested.append(uuid)
        if len(requested) != 1:
            raise ValueError('Use --gpu INDEX/UUID or --gpus INDEX,INDEX to select among matching GPUs')
    devices = []
    for selector in requested:
        uuid = subprocess.check_output(['nvidia-smi', '-i', selector.strip(),
            '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True).strip()
        device_args = copy.copy(args); device_args.gpu = uuid
        observed = platform.detect(device_args, target)
        if uuid in [device['uuid'] for device in devices]:
            raise ValueError('the same physical GPU was selected twice')
        devices.append(observed)
    return devices


def prepare_device(args, observed):
    """Share immutable CPU references; each UUID owns calibration and timing."""
    device_args = copy.copy(args)
    device_args.gpu = observed['uuid']
    device_args.output = args.output / 'devices' / observed['uuid']
    source = args.output / args.mode
    destination = device_args.output / args.mode
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('binding.json', 'manifest.json', 'precompiled.json'):
        value = platform.read(source / name)
        path = destination / name
        if path.exists() and platform.read(path) != value:
            raise ValueError('Device preparation differs from the shared CPU inputs: ' + str(path))
        platform.write(path, value)
    for name, original in (('prepared', source / 'prepared'), ('modules', args.output / 'modules')):
        path = destination / name if name == 'prepared' else device_args.output / name
        if path.exists() or path.is_symlink():
            if platform.safe_path(path) != platform.safe_path(original):
                raise ValueError('Device CPU-cache link differs: ' + str(path))
        else:
            path.symlink_to(original, target_is_directory=True)
    return device_args


def export_runtime(args):
    """Publish an explicit application binding only after calibrated completion."""
    from run_current_recalibration import gate
    folder = args.output / args.mode
    run = folder / 'run'
    state = platform.read(run / 'status.json')
    if not str(state.get('status', '')).startswith('complete') or state.get('remaining') != 0:
        raise ValueError('Cannot export runtime before the finite comparison completes')
    gate('finalize', run / 'calibration')
    values = dict(CUBUTTERFLY_LIBRARY=str(args.build_dir / 'libcuntt.so'),
        CUBUTTERFLY_HARDWARE_MODEL=str(run / 'calibration/hardware/hardware_model.json'),
        CUBUTTERFLY_HARDWARE_PROFILE=str(run / 'calibration/hardware/hardware_profile.json'),
        CUBUTTERFLY_COST_MODEL=str(run / 'calibration/finalized/cost_model.json'),
        CUBUTTERFLY_REGISTRY=str(run / 'calibration/finalized/registry.json'),
        CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT), CUBUTTERFLY_MATHDX_ROOT=str(args.mathdx),
        CUBUTTERFLY_NVCC=str(args.nvcc), CUBUTTERFLY_JIT_CACHE=str(args.output / 'modules'),
        CUBUTTERFLY_COMPILE_MODE='research')
    for key, value in values.items():
        if key != 'CUBUTTERFLY_COMPILE_MODE' and not platform.safe_path(value).exists():
            raise ValueError('Runtime artifact missing: ' + value)
    path = folder / 'runtime_environment.sh'
    path.write_text('# Calibrated target binding; retain the referenced artifacts.\n' +
        ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items()))
    platform.write(folder / 'runtime_binding.json', dict(gpu_uuid=args.gpu, variables=values,
        library_sha256=platform.sha(args.build_dir / 'libcuntt.so'),
        scope='Measured registry and target profile for this finite campaign; not exhaustive tuning'))
    return path


def measure_device(args, target, observed, run):
    args = prepare_device(args, observed)
    uuid = args.gpu
    # The shared root/build locks also protect source preparation. This lock
    # prevents another instance of this entry timing the same GPU separately.
    with lock(Path('/tmp') / f'cubutterfly-platform-{os.getuid()}-{uuid}.lock'):
        run(entry_command(args, 'run') + ['--prepare-run-only'], label=uuid + '-bind')
        refresh = platform.commands(args, target)['run'] + ['--resume']
        watch = [str(args.python), str(ROOT / 'scripts/watch_idle_campaign.py'),
            '--campaign-dir', str(args.output / args.mode / 'run'), '--gpu-uuid', uuid,
            '--wait-hours', str(args.wait_hours), '--', *refresh]
        run(watch, env=platform.environment(args, gpu=True), label=uuid + '-measure', gpu_driver=True)
        run(entry_command(args, 'report'), label=uuid + '-report')
        return dict(gpu_uuid=uuid, status='complete',
            summary=str(args.output / args.mode / 'report/summary.json'),
            runtime_environment=str(export_runtime(args)))


def workflow(args, target):
    args.with_vkfft = True
    if platform.safe_path(args.python) != platform.safe_path(args.prefix / 'bin/python'):
        raise ValueError('--python must select the chosen conda prefix interpreter')
    if args.gpu and args.gpus:
        raise ValueError('choose --gpu or --gpus, not both')
    if args.vkfft_library or args.gpuntt_library or args.allow_baseline_gaps:
        raise ValueError('all prepares its pinned complete baselines; use individual actions for custom baselines')
    if args.dry_run:
        print(json.dumps(dict(target=target['id'], mode=args.mode,
            requests=target['modes'][args.mode]['requests'], jobs=args.jobs,
            prefix=str(args.prefix), build_dir=str(args.build_dir), output=str(args.output),
            phases=['environment-and-pinned-sources', 'build-and-install',
                    'parallel-baseline-build-and-CPU-preparation'] + ([] if args.prepare_only else
                    ['resolve-device-identity', 'wait-for-idle', 'calibrate-confirm-compare', 'report-and-runtime-binding']),
            commands=platform.commands(args, target), gpu_access=False,
            wait_hours=args.wait_hours, prepare_only=args.prepare_only), indent=2))
        return 0
    run = Commands(args)
    state_file = args.output / 'workflow/status.json'
    state = dict(schema='cubutterfly-platform-workflow-v1', status='starting',
        pid=os.getpid(), target=target['id'], mode=args.mode, jobs=args.jobs,
        requests=target['modes'][args.mode]['requests'], started_unix=time.time())
    def save(status, **values):
        state.update(status=status, updated_unix=time.time(), **values)
        platform.write(state_file, state)
    def interrupt(number, frame):
        # Never acquire a child-registry lock or raise between Popen and
        # registration from a signal handler. Ordinary wait loops do cleanup.
        run.cancelled.set()
    handlers = {}
    with ExitStack() as stack:
        stack.enter_context(lock(args.output / 'workflow.lock'))
        stack.enter_context(lock(args.build_dir / '.platform-build.lock'))
        # Dependency/package installation mutates the prefix; serialize other
        # workflows sharing that environment through completion of this run.
        prefix_key = hashlib.sha256(str(args.prefix).encode()).hexdigest()[:16]
        stack.enter_context(lock(args.prefix.parent / ('.cubutterfly-' + prefix_key + '.lock')))
        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            # Fail compiler/target incompatibility before any environment download.
            codes = subprocess.check_output([str(args.nvcc), '--list-gpu-code'], text=True,
                env=platform.environment(args))
            if f'sm_{target["cuda_arch"]}' not in codes.split():
                raise ValueError('Selected CUDA compiler does not support this target; supply --nvcc')
            save('preparing-environment')
            dependencies.prepare_environment(args, target, run)
            # A fresh conda prefix creates bin/python as a symlink. Normalize
            # after creation so first-run and resume bindings use the same path.
            args.python = platform.safe_path(args.prefix / 'bin/python')
            dependencies.prepare_baseline_sources(args, target, run)
            save('building')
            ensure_build(args, target, run)
            save('preparing-cpu')
            baseline_args, cpu_args = copy.copy(args), copy.copy(args)
            if args.jobs > 1:
                baseline_args.jobs = max(1, args.jobs // 2)
                cpu_args.jobs = args.jobs - baseline_args.jobs
                with ThreadPoolExecutor(max_workers=2) as pool:
                    baseline = pool.submit(dependencies.build_baselines, baseline_args, target, run)
                    prepare = pool.submit(run, entry_command(cpu_args, 'prepare') + ['--parallel-prepare'], label='prepare-cpu')
                    libraries = baseline.result(); prepare.result()
            else:
                libraries = dependencies.build_baselines(baseline_args, target, run)
                run(entry_command(cpu_args, 'prepare'), label='prepare-cpu')
            for key, value in libraries.items():
                setattr(args, key, value)
            platform.write(args.output / 'workflow/baselines.json', {key: str(value) for key, value in libraries.items()})
            if args.prepare_only:
                save('cpu-ready', gpu_access=False)
                return 0
            devices = select_devices(args, target)
            # Bounded aggregate CPU workers even when several devices measure.
            if len(devices) > args.jobs:
                raise ValueError('--jobs must be at least the number of selected GPUs')
            args.jobs = max(1, args.jobs // len(devices))
            save('waiting-or-measuring', devices=devices)
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(measure_device, args, target, observed, run) for observed in devices]
                outcomes, errors = [], []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except Exception as error:
                        errors.append(str(error))
                if run.cancelled.is_set():
                    raise WorkflowCancelled('workflow cancelled')
                save('incomplete' if errors else 'complete', results=outcomes, errors=errors)
                if errors:
                    return 2
            print(json.dumps(state, indent=2))
            return 0
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, WorkflowCancelled))
            run.stop()
            save('interrupted' if interrupted else 'failed', error=str(error))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
