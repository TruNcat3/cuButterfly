#!/usr/bin/env python3
"""Resume current-build calibration and a prepared semantic comparison cohort.

Runs the operator track followed by explicitly prepared mechanism/application
jobs. Use watch_idle_campaign.py to wait for exclusive hardware. Existing
calibration gates, CPU artifacts and asynchronous compiler are reused.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import run_current_recalibration as calibration
from run_isolated_gpu import descendant, gpu_pids, stop_owned
import semantic_compile_broker as broker
import semantic_preparation_queue as preparation

ROOT = Path(__file__).resolve().parents[1]
GPU_BUSY_EXIT = 76  # Explicit async exclusive-GPU interruption; distinct from worker recycle.


class ExternalCompileProducer:
    """Observe a separately supervised CPU producer without taking ownership."""

    external = True

    def __init__(self, directory, broker_id):
        self.directory = Path(directory).resolve()
        self.broker_id = broker_id
        receipt = calibration.read(self.directory / 'producer_status.json')
        self.pid = receipt['pid']
        if type(self.pid) is not int or self.pid <= 1:
            raise RuntimeError('invalid external compile producer PID')
        self.start_ticks = self._start_ticks()
        self._validate()

    def _start_ticks(self):
        fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            raise RuntimeError('external compile producer is a zombie')
        return fields[19]

    def _validate(self):
        receipt = calibration.read(self.directory / 'producer_status.json')
        age = time.time_ns() - receipt['heartbeat_ns']
        if (receipt.get('schema') != 'cubutterfly-semantic-compile-producer-v1'
                or receipt.get('broker_id') != self.broker_id or receipt.get('pid') != self.pid
                or receipt.get('status') != 'running' or not 0 <= age <= 60_000_000_000
                or self._start_ticks() != self.start_ticks):
            raise RuntimeError('external compile producer identity or heartbeat is invalid')
        command = Path(f'/proc/{self.pid}/cmdline').read_bytes().decode().split('\0')
        expected_script = str(ROOT / 'scripts/semantic_compile_producer.py')
        if (expected_script not in command or '--broker' not in command
                or Path(command[command.index('--broker') + 1]).resolve() != self.directory):
            raise RuntimeError('external compile producer command does not match broker')
        with (self.directory / 'standalone_producer.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        raise RuntimeError('external compile producer does not hold its broker lock')

    def poll(self):
        self._validate()
        return None


def stop_producer(producer):
    # An external CPU service must survive GPU occupation and driver restarts.
    if not getattr(producer, 'external', False):
        stop_owned(producer)


class PhaseFailure(RuntimeError):
    """A measured command failed, distinct from identity/ownership violations."""


def followup_plan(args):
    if not getattr(args, 'followup_plan', None):
        return dict(jobs=[], input_paths=[])
    plan = calibration.read(args.followup_plan)
    if plan.get('schema') != 'cubutterfly-refresh-followups-v1':
        raise ValueError('unsupported followup plan schema')
    replacements = dict(output_dir=str(args.output_dir), build_dir=str(args.build_dir))
    includes = plan.get('include_prepared_plans', [])
    if not isinstance(includes, list) or not all(isinstance(x, str) for x in includes):
        raise ValueError('include_prepared_plans must be an explicit path array')
    for name in includes:
        path = (ROOT / name.format(**replacements)).resolve(strict=True)
        child = calibration.read(path)
        if (child.get('schema') != 'cubutterfly-refresh-followups-v1'
                or child.get('include_prepared_plans')):
            raise ValueError('prepared followup plan requires the same schema and no nested includes')
        plan.setdefault('input_paths', []).extend([str(path), *child.get('input_paths', [])])
        plan.setdefault('jobs', []).extend(child.get('jobs', []))
    return plan


def followup_dependencies(args):
    """Require a topological, finite list; missing/failed prerequisites block use."""
    known = {'calibrate', 'finalize', 'xd', 'compare'}
    dependencies = {}
    for job in followup_plan(args).get('jobs', []):
        required = job.get('depends_on', [])
        if (not isinstance(required, list) or not all(isinstance(x, str) for x in required)
                or len(set(required)) != len(required) or not set(required) <= known):
            raise ValueError('followup dependencies must name distinct earlier phases: ' + job['id'])
        dependencies[job['id']] = required
        known.add(job['id'])
    return dependencies


def followup_inputs(args):
    paths = followup_plan(args).get('input_paths', [])
    if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
        raise ValueError('followup input_paths must be an explicit path array')
    replacements = dict(output_dir=str(args.output_dir), build_dir=str(args.build_dir))
    result = [(ROOT / path.format(**replacements)).resolve(strict=True) for path in paths]
    if any(not path.is_file() for path in result):
        raise ValueError('followup input_paths must identify files')
    return result


def followup_producers(args):
    result = {}
    for job in followup_plan(args).get('jobs', []):
        enabled = job.get('compile_producer', False)
        if not isinstance(enabled, bool):
            raise ValueError('compile_producer must be boolean')
        if enabled:
            path = job.get('compile_priority_report')
            if not isinstance(path, str):
                raise ValueError('compile producer requires an explicit priority report')
            result[job['id']] = (ROOT / path.format(output_dir=str(args.output_dir))).resolve()
    return result


def followup_recycles(args):
    result = set()
    for job in followup_plan(args).get('jobs', []):
        code = job.get('worker_recycle_exit')
        if code is not None:
            if type(code) is not int or code != 75:
                raise ValueError('only explicit async worker recycle exit 75 is supported')
            result.add(job['id'])
    return result


def followup_jobs(args):
    plan = followup_plan(args)
    followup_dependencies(args)
    followup_producers(args)
    followup_recycles(args)
    jobs = plan.get('jobs', [])
    names = set()
    result = []
    for job in jobs:
        name = job['id']
        if (not name or name in names or name in ('calibrate', 'finalize', 'xd', 'compare')
                or not all(c.isalnum() or c in '-_' for c in name)):
            raise ValueError('invalid or duplicate followup id')
        names.add(name)
        argv = job['argv']
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise ValueError('followup requires an explicit argument array')
        script = (ROOT / argv[0]).resolve(strict=True)
        if not script.is_relative_to(ROOT) or script.suffix != '.py':
            raise ValueError('followup entrypoint must be a repository Python script')
        replacements = dict(gpu_uuid=args.gpu_uuid, output_dir=str(args.output_dir),
                            build_dir=str(args.build_dir))
        command = [sys.executable, str(script)] + [item.format(**replacements) for item in argv[1:]]
        result.append((name, command))
    return result


def validate_preparation(manifest, library, directory):
    from semantic_campaign_manifest import validate_manifest_hash
    from run_semantic_campaign import PROTOCOL, validate_fingerprints
    if not validate_manifest_hash(manifest):
        raise ValueError('manifest checksum mismatch')
    validate_fingerprints(manifest, library)
    config, _ = preparation._config(directory)
    if config['manifest'] != manifest or config['source_sha256'] != preparation._sources():
        raise ValueError('prepared CPU inputs do not match this manifest and source')
    if any(config.get(key) != PROTOCOL[key] for key in ('reference_max_elements', 'ntt_max_operations')):
        raise ValueError('prepared reference bounds differ from the measurement protocol')
    records = [preparation.record_for(directory, case['id']) for case in manifest['cases']]
    if not records or any(not record or record['status'] != 'ready' for record in records):
        raise ValueError('all requested CPU inputs and references must be ready before GPU launch')
    return len(records)


def build_commands(args):
    output = args.output_dir / 'calibration'
    options = SimpleNamespace(build_dir=args.build_dir, output_dir=output,
        gpu_uuid=args.gpu_uuid, workloads=args.workloads,
        stage_point_budget=args.stage_point_budget, search_budget=args.search_budget,
        vkfft_binary=None, gpuntt_binary=None, fht_python=None,
        resume=False, include_xd=True)
    commands = dict(calibration.commands(options))
    # Partial checkpoints retain their own pre/post measurement guards and exact
    # identities. Never adopt another build or silently skip a failed gate.
    if (output / 'initial/search_coverage.json').exists():
        commands['calibrate'].append('--resume-search')
    if getattr(args, 'calibration_cpu_input', None):
        commands['calibrate'] += ['--cpu-preparation-input', str(args.calibration_cpu_input),
            '--retry-compilation-gaps']
        if getattr(args, 'calibration_cpu_inventory_only', False):
            commands['calibrate'].append('--cpu-preparation-inventory-only')
        else:
            commands['calibrate'] += ['--cpu-preparation-dir', str(args.calibration_cpu_dir)]
        # Research preparation is explicitly requested; do not apply the
        # lightweight installation's cumulative compilation cap to this run.
        commands['calibrate'][commands['calibrate'].index('--compile-seconds') + 1] = '0'
    if (output / 'finalized/confirmation_identity.json').exists():
        commands['finalize'].append('--resume')
    compare = [sys.executable, str(ROOT / 'scripts/run_async_semantic_campaign.py'),
        '--manifest', str(args.manifest), '--library', str(args.build_dir / 'libcuntt.so'),
        '--output', str(args.output_dir / 'measurement'), '--preparation', str(args.preparation),
        '--compile-broker', str(args.output_dir / 'compile'), '--gpu-uuid', args.gpu_uuid,
        '--calibration-manifest', str(output / 'finalized/calibration_manifest.json'),
        '--timing', '--cpu-workers', str(args.cpu_workers), '--in-flight', str(args.cpu_workers)]
    for name in ('vkfft_library', 'gpuntt_library'):
        if getattr(args, name):
            compare += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if (args.output_dir / 'measurement/results.json').exists():
        compare.append('--resume')
    return [('calibrate', commands['calibrate']), ('finalize', commands['finalize']),
            ('xd', commands['xd']), ('compare', compare)]


def build_environment(args, config):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        CUBUTTERFLY_COMPILE_MODE='research', CUBUTTERFLY_JIT_PYTHON=sys.executable,
        CUBUTTERFLY_CALIBRATION_BROKER=str(args.output_dir / 'compile'),
        CUBUTTERFLY_TEMPLATE_ROOT=config['root'], CUBUTTERFLY_MATHDX_ROOT=config['mathdx'],
        CUBUTTERFLY_NVCC=config['nvcc'], CUBUTTERFLY_JIT_CACHE=config['cache'])
    for key in ('CUBUTTERFLY_HARDWARE_MODEL', 'CUBUTTERFLY_PROFILE_ROOT',
                'CUBUTTERFLY_HARDWARE_PROFILE', 'CUBUTTERFLY_COST_MODEL', 'LD_PRELOAD',
                'CUBUTTERFLY_PREPARE_BROKER', 'CUBUTTERFLY_PREPARE_EVENTS',
                'CUBUTTERFLY_COMPILE_BUDGET_FILE', 'CUBUTTERFLY_COMPILE_BUDGET_SECONDS'):
        env.pop(key, None)
    env['LD_LIBRARY_PATH'] = str(args.build_dir) + (':' + env['LD_LIBRARY_PATH']
                                                 if env.get('LD_LIBRARY_PATH') else '')
    # Installed editable frontend is checked again by the semantic worker.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return env


def identity(args, document, env):
    paths = [args.manifest, args.workloads, args.preparation / 'config.json', Path(__file__)]
    paths += [args.build_dir / name for name in calibration.ARTIFACTS]
    paths += [ROOT / 'scripts' / name for name in (*calibration.SOURCE_INPUTS,
        'calibrate_xd_public.py', 'run_async_semantic_campaign.py', 'run_semantic_campaign.py',
        'semantic_cpu_pipeline.py', 'semantic_input_profiles.py', 'semantic_compile_broker.py',
        'semantic_compile_producer.py', 'semantic_preparation_queue.py', 'jit_calibration_identity.py',
        'semantic_resource_compile_proxy.py')]
    paths += [getattr(args, name) for name in ('vkfft_library', 'gpuntt_library') if getattr(args, name)]
    if getattr(args, 'calibration_cpu_input', None):
        paths.append(args.calibration_cpu_input)
    if getattr(args, 'followup_plan', None):
        paths.append(args.followup_plan)
        paths += [Path(command[1]) for _, command in followup_jobs(args)]
        paths += followup_inputs(args)
    result = dict(schema='cubutterfly-semantic-refresh-v1', gpu_uuid=args.gpu_uuid,
        files={str(path.resolve()): calibration.digest(path) for path in paths},
        broker_id=document['id'], broker_sources_sha256=document['sources_sha256'],
        stage_point_budget=args.stage_point_budget, calibration_search_budget=args.search_budget,
        comparison_search_budget=args.comparison_search_budget,
        calibration_cpu_input=str(args.calibration_cpu_input) if getattr(args, 'calibration_cpu_input', None) else None,
        calibration_cpu_dir=str(args.calibration_cpu_dir) if getattr(args, 'calibration_cpu_dir', None) else None,
        calibration_cpu_inventory_only=bool(getattr(args, 'calibration_cpu_inventory_only', False)),
        environment={k: v for k, v in env.items() if k.startswith('CUBUTTERFLY_') or
                     k in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')})
    if getattr(args, 'external_compile_producer', False):
        result['external_compile_producer'] = True
    return result


def result_summary(report, count):
    remaining = report.get('remaining_case_ids')
    rows = report.get('rows')
    if not isinstance(remaining, list) or not isinstance(rows, list):
        raise ValueError('comparison result is missing its coverage ledger')
    if remaining or len(rows) != count or not str(report.get('status', '')).startswith('complete'):
        raise ValueError('comparison has not completed the declared finite population')
    return dict(status='complete-operator-track', remaining=0, measured_rows=len(rows),
        comparison_status=report['status'],
        scope_note='Only the frozen operator track; mechanism and end-to-end tracks have separate receipts')


def orchestrate(args):
    build_sm = calibration.validate_build(args.build_dir)
    manifest = calibration.read(args.manifest)
    count = validate_preparation(manifest, args.build_dir / 'libcuntt.so', args.preparation)
    active_broker = args.output_dir / 'compile'
    document = broker.initialize(active_broker, dict(root=str(ROOT), mathdx=str(args.mathdx),
        nvcc=str(args.nvcc), cache=str(args.jit_cache), sm=build_sm))
    env = build_environment(args, document['config'])
    current = identity(args, document, env)
    if (getattr(args, 'calibration_cpu_input', None) and not args.prepare_only
            and not getattr(args, 'calibration_cpu_inventory_only', False)):
        preparation = calibration.read(args.calibration_cpu_dir / 'status.json')
        if (not preparation.get('scoring_complete') or not preparation.get('compile_ready')
                or preparation.get('input_sha256') != calibration.digest(args.calibration_cpu_input)):
            raise ValueError('CPU calibration scoring and priority compilation must finish before GPU work')
    recycle_jobs = followup_recycles(args)
    path = args.output_dir / 'status.json'
    if path.exists():
        state = calibration.read(path)
        if not args.resume or state['identity'] != current:
            raise ValueError('resume requires unchanged build, source, GPU, inputs and timing policy')
        if state['status'] == 'failed':
            raise ValueError('failed phase requires inspection; automatic failure retry is disabled')
    else:
        state = dict(identity=current, status='prepared', phases={}, gpu_uuid=args.gpu_uuid,
            population=count, remaining=count, measured_rows=0,
            scope='finite operator cohort after current calibration; not full-paper completion',
            checkpoint_policy='only inner pre/post-guarded completed samples; interrupted work remains pending')

    def save():
        state['updated_unix'] = time.time()
        calibration.write(path, state)

    def verify_inputs():
        for name, expected in current['files'].items():
            if calibration.digest(name) != expected:
                raise ValueError('campaign input changed: ' + name)
        if broker._sources(document['config']) != document['sources_sha256']:
            raise ValueError('compiled template sources changed during campaign')

    def run(name, command):
        if state['phases'].get(name, {}).get('status') == 'complete':
            return 0
        verify_inputs()
        if gpu_pids(args.gpu_uuid):
            state.update(status='waiting-for-gpu', phase=name, retryable=True)
            save()
            raise RuntimeError('target GPU is occupied')
        for key in ('error', 'retryable', 'foreign_pids'):
            state.pop(key, None)
        phase = dict(status='running', command=command, started_unix=time.time())
        state['phases'][name] = phase
        state.update(status='running', phase=name)
        save()
        with (args.output_dir / (name + '.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            phase['pid'] = child.pid
            save()
            try:
                while True:
                    if producer is not None and producer.poll() is not None:
                        raise RuntimeError('CPU compile producer exited unexpectedly; see compile_producer.log')
                    foreign = [pid for pid in gpu_pids(args.gpu_uuid)
                        if Path(f'/proc/{pid}').exists() and not descendant(pid, child.pid)]
                    if foreign:
                        stop_owned(child)
                        phase['status'] = 'interrupted'
                        state.update(status='interrupted-by-foreign-gpu-work', retryable=True,
                                     foreign_pids=foreign)
                        raise RuntimeError('foreign GPU work; stopped only the owned measurement group')
                    if child.poll() is not None:
                        break
                    time.sleep(1)
            finally:
                stop_owned(child)
                phase['returncode'] = child.returncode
                phase['finished_unix'] = time.time()
                save()
        busy_receipt = None
        if child.returncode == GPU_BUSY_EXIT and name == 'compare':
            try:
                receipt = calibration.read(args.output_dir / 'measurement/gpu_interruption.json')
                if (receipt['status'] == 'interrupted-by-foreign-gpu-work'
                        and receipt['gpu_uuid'] == args.gpu_uuid
                        and receipt['pid'] == child.pid
                        and phase['started_unix'] <= receipt['observed_unix'] <= time.time()):
                    busy_receipt = receipt
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if busy_receipt is not None:
            verify_inputs()
            phase['status'] = 'interrupted'
            state.update(status='interrupted-by-foreign-gpu-work', retryable=True,
                         gpu_interruption=busy_receipt)
            save()
            raise RuntimeError('foreign GPU work detected by comparison worker; retry when idle')
        if child.returncode == 75 and (name == 'compare' or name in recycle_jobs):
            verify_inputs()
            phase['status'] = 'worker-recycle-required'
            save()
            return 75
        verify_inputs()
        if child.returncode:
            phase['status'] = 'failed'
            save()
            raise PhaseFailure(f'{name} exited {child.returncode}; see {name}.log')
        if producer is not None and producer.poll() is not None:
            raise RuntimeError('CPU compile producer exited before comparison acceptance')
        phase['status'] = 'complete'
        save()
        return 0

    save()
    if args.prepare_only:
        calibration.write(args.output_dir / 'commands.json', dict(phases=build_commands(args),
            followups=followup_jobs(args), dependencies=followup_dependencies(args)))
        print(json.dumps(dict(status=state['status'], prepared_cases=count, gpu_measurements_started=False)))
        return 0
    producer = None

    def start_producer(priority_report):
        if getattr(args, 'external_compile_producer', False):
            return ExternalCompileProducer(active_broker, document['id'])
        with (args.output_dir / 'compile_producer.log').open('a') as log:
            return subprocess.Popen([sys.executable,
                str(ROOT / 'scripts/semantic_compile_producer.py'), '--broker', str(active_broker),
                '--jobs', str(args.compile_jobs), '--watch', '--priority-report', str(priority_report)],
                cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)

    try:
        for name, command in build_commands(args):
            env.pop('CUBUTTERFLY_HARDWARE_MODEL', None)
            hardware = args.output_dir / 'calibration/hardware/hardware_model.json'
            if name != 'calibrate' and hardware.is_file():
                env['CUBUTTERFLY_HARDWARE_MODEL'] = str(hardware)
            profile = 'initial' if name == 'calibrate' else 'finalized'
            env['CUBUTTERFLY_REGISTRY'] = str(args.output_dir / 'calibration' / profile / 'registry.json')
            if name == 'compare':
                env['CUBUTTERFLY_SEARCH_BUDGET'] = str(args.comparison_search_budget)
                if state['phases'].get(name, {}).get('status') != 'complete':
                    producer = start_producer(args.output_dir / 'measurement/results.json')
            for _ in range(128):
                if run(name, command) != 75:
                    break
                if '--resume' not in command:
                    command.append('--resume')
            else:
                raise RuntimeError('semantic worker recycling limit reached')
            if name != 'compare':
                calibration.gate(name, args.output_dir / 'calibration', args.workloads)
        report = calibration.read(args.output_dir / 'measurement/results.json')
        state.update(result_summary(report, count))
        if producer is not None:
            stop_producer(producer)
            producer = None
        jobs = followup_jobs(args)
        dependencies = followup_dependencies(args)
        producers = followup_producers(args)
        state['operator_track_complete'] = True
        state['followup_failures'] = state.get('followup_failures', {})
        state['followup_blocked'] = {}
        for index, (name, command) in enumerate(jobs):
            state.update(status='running-followups', remaining=len(jobs) - index)
            save()
            if name in state['followup_failures']:
                continue  # A failed experiment is retained, never silently retried.
            blocked = [dependency for dependency in dependencies.get(name, [])
                       if state['phases'].get(dependency, {}).get('status') != 'complete']
            if blocked:
                state['phases'][name] = dict(status='blocked-by-dependency', depends_on=blocked)
                state['followup_blocked'][name] = blocked
                save()
                continue
            try:
                if name in producers and state['phases'].get(name, {}).get('status') != 'complete':
                    producer = start_producer(producers[name])
                for _ in range(128):
                    if run(name, command) != 75:
                        break
                else:
                    raise PhaseFailure(f'{name} exhausted its explicit worker recycle limit')
            except PhaseFailure as error:
                state['followup_failures'][name] = str(error)
                save()
            finally:
                if producer is not None:
                    stop_producer(producer)
                    producer = None
        if jobs:
            state.update(status='complete-runnable-refresh-with-gaps' if
                         state['followup_failures'] or state['followup_blocked']
                         else 'complete-runnable-refresh', remaining=0,
                         full_refresh_complete=False,
                         scope_note='Operator cohort and explicitly queued followups only; consult full refresh plan for unresolved application and profile coverage')
        save()
        return 0
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            state.update(status='interrupted-calibration', retryable=True)
        elif state.get('status') not in ('waiting-for-gpu', 'interrupted-by-foreign-gpu-work'):
            state.update(status='failed', retryable=False)
        state['error'] = str(error)
        save()
        raise
    finally:
        if producer is not None:
            stop_producer(producer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'output-dir', 'manifest', 'preparation'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--workloads', type=Path, default=calibration.WORKLOADS)
    parser.add_argument('--mathdx', type=Path, default=ROOT / 'external/mathdx/nvidia/mathdx')
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    parser.add_argument('--jit-cache', type=Path, default=Path.home() / '.cache/cubutterfly/modules')
    parser.add_argument('--vkfft-library', type=Path)
    parser.add_argument('--gpuntt-library', type=Path)
    parser.add_argument('--followup-plan', type=Path,
                        help='explicit finite followup jobs; failed jobs remain gaps, not victories')
    parser.add_argument('--calibration-cpu-input', type=Path,
                        help='frozen offline calibration input manifest')
    parser.add_argument('--calibration-cpu-dir', type=Path,
                        help='completed CPU scoring and priority-compilation directory')
    parser.add_argument('--calibration-cpu-inventory-only', action='store_true',
                        help='reuse frozen candidates and existing modules; score current feedback in memory')
    parser.add_argument('--stage-point-budget', type=int, default=24)
    parser.add_argument('--search-budget', type=int, default=32)
    parser.add_argument('--comparison-search-budget', type=int, default=512)
    parser.add_argument('--compile-jobs', type=int, default=4)
    parser.add_argument('--external-compile-producer', action='store_true',
                        help='require a live independent CPU producer for this broker; never stop it with GPU work')
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true', help='validate and bind CPU inputs; no CUDA work')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.calibration_cpu_inventory_only:
        if not args.calibration_cpu_input or args.calibration_cpu_dir:
            parser.error('inventory-only requires --calibration-cpu-input without --calibration-cpu-dir')
    elif bool(args.calibration_cpu_input) != bool(args.calibration_cpu_dir):
        parser.error('--calibration-cpu-input and --calibration-cpu-dir must be specified together')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if min(args.stage_point_budget, args.search_budget, args.comparison_search_budget,
           args.compile_jobs, args.cpu_workers) < 1:
        parser.error('budgets and worker counts must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 73
        def interrupted(number, frame):
            raise KeyboardInterrupt('refresh received signal ' + str(number))
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            return orchestrate(args)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
