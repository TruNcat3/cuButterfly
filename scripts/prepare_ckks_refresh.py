#!/usr/bin/env python3
"""CPU-only CKKS consumer build, request validation and dependent GPU queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/wt/yes/envs/cubutterfly/bin/python')
ENV_ALLOWLIST = ('HOME', 'XDG_CACHE_HOME', 'CUBUTTERFLY_JIT_PYTHON',
                 'CUBUTTERFLY_TEMPLATE_ROOT', 'CUBUTTERFLY_MATHDX_ROOT',
                 'CUBUTTERFLY_NVCC', 'CUBUTTERFLY_JIT_CACHE', 'CUBUTTERFLY_REGISTRY',
                 'CUBUTTERFLY_PLATFORM_TUNING', 'CUBUTTERFLY_STATIC_RANKER')


def safe_path(path):
    path = Path(path).resolve()
    if path == Path('/sdb') or Path('/sdb') in path.parents:
        raise ValueError(f'forbidden asset location: {path}')
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(safe_path(path).read_text())


def write_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name + '.', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    temporary.replace(path)


def historical_options(metadata):
    """Replay semantic and timing argv, never historical binary/output identities."""
    argv = metadata['benchmark_argv'][1:]
    index = argv.index('--output')
    return argv[:index] + argv[index + 2:]


def without_options(argv, flags=(), values=()):
    result = []
    index = 0
    while index < len(argv):
        item = argv[index]
        index += 1
        if item in values:
            index += 1
        elif item not in flags:
            result.append(item)
    return result


def prepare(config, *, build=False, jobs=2):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES='' explicitly")
    runner = load_module('ckks_refresh_runner', ROOT / 'paper/experiments/run_heongpu_ckks_e2e.py')
    requests = load_module('ckks_refresh_requests', ROOT / 'scripts/prepare_rns_requests.py')
    output = safe_path(ROOT / config['output'])
    output.mkdir(parents=True, exist_ok=True)
    library = safe_path(ROOT / config['library'])
    source = safe_path(ROOT / config['heongpu_source'])
    cached = safe_path(ROOT / config['cached_build'])
    # Resolve explicit assets before hashing or passing them to the consumer build.
    assets = [library, source / 'src/lib/host/ckks/operator.cu', cached / 'src/libheongpu.a',
              cached / 'thirdparty/GPU-NTT/src/libntt-1.0.a',
              cached / 'thirdparty/GPU-FFT/src/libfft-1.0.a',
              cached / 'thirdparty/RNGonGPU/src/librngongpu-1.0.a',
              cached / '_deps/rmm-build/librmm.so', cached / '_deps/rapids_logger-build/librapids_logger.so',
              Path('/tmp/heongpu-e13-gf2x/root/usr/lib/x86_64-linux-gnu/libgf2x.so.3.0.0'),
              Path('/tmp/heongpu-e13-prefix-20261003')]
    for asset in assets:
        if not safe_path(asset).exists():
            raise FileNotFoundError(asset)
    manifest_path = safe_path(ROOT / config['requests'])
    manifest = read_json(manifest_path)
    for entry in manifest['requests']:
        safe_path(manifest_path.parent / entry['request_file'])
    validated = requests._load_requests(manifest_path)
    if len(validated) != config['expected_requests']:
        raise ValueError('historical request population changed')
    request_dir = output / 'bootstrap_prepared/requests'
    for entry in manifest['requests']:
        destination = request_dir / entry['request_file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(manifest_path.parent / entry['request_file']), destination)
    # Historical mappings remain provenance only; the current cache starts empty.
    write_json(request_dir / 'manifest.json', manifest)
    request_receipt = {'unique_requests': len(validated),
                       'forward': sum(not item[2]['inverse'] for item in validated),
                       'inverse': sum(item[2]['inverse'] for item in validated),
                       'in_place': sum(item[1] == 'in_place' for item in validated),
                       'separate': sum(item[1] == 'separate' for item in validated),
                       'mapping_status': 'historical_prior_only_current_mapping_requires_GPU_tuning',
                       'manifest': runner.file_identity(manifest_path),
                       'request_files': [runner.file_identity(safe_path(manifest_path.parent / entry['request_file']))
                                         for entry in manifest['requests']]}
    write_json(output / 'request_validation.json', request_receipt)
    wrapper = ROOT / 'paper/experiments/run_heongpu_ckks_e2e.py'
    builds = {name: output / 'consumer_builds' / name for name in ('short_chain', 'bootstrap_v1_small')}
    common = [str(PYTHON), str(wrapper), '--cuntt-library', str(library),
              '--heongpu-source', str(source), '--cached-build', str(cached), '--arch', config['arch']]
    build_receipts = []
    previous_builds = {item['id']: item for item in read_json(output / 'cpu_build_receipt.json')} if (output / 'cpu_build_receipt.json').exists() else {}
    for profile, directory in builds.items():
        command = common + ['--build-dir', str(directory), '--build-only', '--jobs', str(jobs),
                            '--output', str(output / f'{profile}_build_unused.csv')]
        if profile == 'bootstrap_v1_small':
            command += ['--parameter-set', 'bootstrap-v1-small', '--sizes', '4096', '--aux-primes', '3']
        record = {'id': f'build_{profile}', 'argv': command, 'gpu_required': False,
                  'status': 'not_requested', 'CUDA_VISIBLE_DEVICES': '',
                  'libcuntt': runner.file_identity(library)}
        if not build and record['id'] in previous_builds:
            previous = previous_builds[record['id']]
            if previous.get('status') == 'completed' and all(
                    previous.get(key, {}).get('sha256') == runner.file_identity(path).get('sha256')
                    for key, path in [('libcuntt', library), ('cubutterfly_heongpu_ckks', directory / 'cubutterfly_heongpu_ckks'),
                                      ('rns_request_compare', directory / 'rns_request_compare')]):
                record = previous
        if build:
            started = time.monotonic()
            log = output / f'build_{profile}.log'
            env = os.environ.copy()
            env.setdefault('CUBUTTERFLY_COMPILE_MODE', 'research')
            with log.open('w') as handle:
                result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record.update(status='completed' if result.returncode == 0 else 'failed',
                          returncode=result.returncode, wall_seconds=time.monotonic() - started,
                          log=str(log))
            for artifact in ('cubutterfly_heongpu_ckks', 'rns_request_compare'):
                record[artifact] = runner.file_identity(directory / artifact)
            if result.returncode == 0:
                dynamic = subprocess.run(['readelf', '-d', str(directory / 'cubutterfly_heongpu_ckks')],
                                         check=True, capture_output=True, text=True, env=env)
                record['consumer_dynamic_linkage'] = dynamic.stdout
                record['generated_cmake'] = runner.file_identity(directory / 'wrapper/CMakeLists.txt')
                record['bootstrap_header'] = runner.file_identity(directory / 'wrapper/ckks_bootstrap_parameters.hpp')
                record['correspondence'] = 'built_from_recorded_generated_CMake_in_isolated_directory; GPU correctness pending'
            if result.returncode:
                build_receipts.append(record)
                write_json(output / 'cpu_build_receipt.json', build_receipts)
                raise RuntimeError(f'isolated consumer build failed: {log}')
        build_receipts.append(record)
        write_json(output / 'cpu_build_receipt.json', build_receipts)
    queue = []
    cache = output / 'rns_cache'
    gpu_env = {'CUDA_VISIBLE_DEVICES': '<exclusive-full-GPU-UUID>',
               'CUBUTTERFLY_COMPILE_MODE': os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research'),
               'CUBUTTERFLY_RNS_CACHE': str(cache),
               'CUBUTTERFLY_RNS_TUNE_REPETITIONS': str(config['tune_repetitions'])}
    gpu_checks = []
    for target in ('test_heongpu_ntt_adapter', 'test_heongpu_stage_batch',
                   'test_heongpu_graph_schedule', 'test_heongpu_graph_host_memory'):
        check_id = f'correctness_{target}'
        gpu_checks.append(check_id)
        queue.append({'id': check_id, 'depends_on': ['build_short_chain'],
                      'gpu_required': True, 'status': 'pending_gpu', 'env': gpu_env,
                      'argv': [str(builds['short_chain'] / target)],
                      'acceptance': 'exit 0; save stdout/stderr; no performance claim; bound current libcuntt and exclusive GPU identity'})
    queue.append({'id': 'tune_bootstrap_118', 'depends_on': ['build_bootstrap_v1_small'],
                  'gpu_required': True, 'status': 'pending_gpu',
                  'env': gpu_env,
                  'argv': [str(PYTHON), str(ROOT / 'scripts/prepare_rns_requests.py'), 'prepare',
                           '--manifest', str(request_dir / 'manifest.json'), '--library', str(library),
                           '--cache-directory', str(cache), '--policy', 'refresh',
                           '--repetitions', str(config['tune_repetitions']),
                           '--output', str(output / 'bootstrap_prepared/tuning.json')],
                  'acceptance': '118/118 status=ok; current library/device/protocol; tuning is not correctness evidence'})
    replay = dict(queue[-1])
    replay.update(id='replay_bootstrap_118', depends_on=['tune_bootstrap_118'],
                  acceptance='118/118 status=ok and cache_hit=true in a new process; no tuning; not correctness evidence')
    replay['argv'] = list(replay['argv'])
    replay['argv'][replay['argv'].index('refresh')] = 'require'
    replay['argv'][-1] = str(output / 'bootstrap_prepared/cache_replay.json')
    queue.append(replay)
    historical = []
    count = 0
    for cohort in config['auto_cohorts']:
        path = safe_path(ROOT / config['auto_history'] / f'{cohort}.metadata.json')
        meta = read_json(path)
        options = historical_options(meta)
        batch_count = len(options[options.index('--batches') + 1].split(','))
        count += batch_count * 4
        historical.append((cohort, 'short_chain', options, meta, path))
    if count != config['expected_auto_configurations']:
        raise ValueError('historical auto configuration count changed')
    bootstrap_path = safe_path(ROOT / config['bootstrap_history'])
    bootstrap_meta = read_json(bootstrap_path)
    historical.append(('bootstrap_v1_small_b1_b8', 'bootstrap_v1_small',
                       historical_options(bootstrap_meta), bootstrap_meta, bootstrap_path))
    for cohort, profile, options, meta, history_path in historical:
        cohort_dir = output / ('auto_schedule_36' if profile == 'short_chain' else 'bootstrap_prepared')
        cohort_dir.mkdir(parents=True, exist_ok=True)
        base = common + ['--build-dir', str(builds[profile]), '--skip-build']
        if profile == 'short_chain':
            # Eager prepare-only cannot construct stage-batched request shapes.
            queue.append({'id': f'populate_{cohort}', 'depends_on': [f'build_{profile}'],
                          'gpu_required': True, 'status': 'pending_gpu',
                          'env': dict(gpu_env, CUBUTTERFLY_RNS_CACHE_POLICY='prefer',
                                      CUBUTTERFLY_RNS_REQUEST_EXPORT=str(cohort_dir / f'{cohort}_population.ndjson')),
                          'argv': base + options + ['--output', str(cohort_dir / f'{cohort}_population.csv')],
                          'evidence_role': 'request discovery and current mapping population; excluded from final independent measurement',
                          'acceptance': 'full five-path run succeeds; scalar and stage requests exported; retain original selection/timing budget; no reduced workload'})
        prep_options = without_options(options, flags=('--graph', '--stage-batch', '--auto-schedule', '--thermal-guard'))
        prep = base + prep_options + ['--prepare-only', '--prepare-backend', 'both',
                                     '--output', str(cohort_dir / f'{cohort}_prepare.csv')]
        prep_id = f'prepare_{cohort}'
        queue.append({'id': prep_id, 'depends_on': [f'build_{profile}'] +
                      (['replay_bootstrap_118'] if profile == 'bootstrap_v1_small' else [f'populate_{cohort}']),
                      'gpu_required': True, 'status': 'pending_gpu',
                      'env': dict(gpu_env, CUBUTTERFLY_RNS_CACHE_POLICY='prefer',
                                  CUBUTTERFLY_RNS_REQUEST_EXPORT=str(cohort_dir / f'{cohort}_requests.ndjson')),
                      'argv': prep, 'historical_metadata': runner.file_identity(history_path),
                      'acceptance': 'full native/cuButterfly call exact ciphertext; changed input and restored input; preparation metadata completed'})
        measure = base + options + ['--after-preparation', str(cohort_dir / f'{cohort}_prepare.metadata.json'),
                                    '--output', str(cohort_dir / f'{cohort}.csv')]
        queue.append({'id': f'measure_{cohort}', 'depends_on': [prep_id, *gpu_checks],
                      'gpu_required': True, 'status': 'pending_gpu',
                      'env': dict(gpu_env, CUBUTTERFLY_RNS_CACHE_POLICY='require',
                                  CUBUTTERFLY_RNS_REQUEST_EXPORT=str(cohort_dir / f'{cohort}_measurement_requests.ndjson')),
                      'argv': measure, 'historical_thermal_protocol': meta.get('thermal_protocol'),
                      'acceptance': ('completed, ordinary, performance_eligible=true; exact_ciphertext=1 and mismatch_count=0; '
                                     'all required cache hits; no online tuning; retain cohort separately; '
                                     + ('five paths/config, 14 selection and 100 holdout samples/config; selection 7x5, holdout 20x10'
                                        if profile == 'short_chain' else
                                        'independent_public_API_loop Graph; N4096 B1/B8; warmup1 repetitions5 graph-replays1'))})
        if profile == 'bootstrap_v1_small':
            queue[-2]['acceptance'] += '; extract exported requests and require the exact same 118 config/placement request IDs as the CPU-validated manifest; stop and report any changed population'
    result = {'schema_version': 1, 'status': 'cpu_prepared_gpu_dependencies_pending',
              'output_dir': str(output),
              'offline_mapping_ready': False, 'auto_configurations': count,
              'bootstrap_application_batches': [1, 8], 'distinct_rns_requests': len(validated),
              'libcuntt': runner.file_identity(library),
              'cpu_builds': build_receipts, 'queue': queue,
              'queue_policy': 'Execute only after dependencies pass; replace GPU UUID, verify exclusive model/memory identity; commands require GPU and are never executed by this CPU script.',
              'scope_note': 'Historical mapping fields are prior provenance, not current confirmed mappings. 118 requests are not 118 independent applications.'}
    static_paths = [Path(__file__), ROOT / 'configs/experiments/ckks_refresh_20261025.json', wrapper,
                    ROOT / 'paper/experiments/summarize_heongpu_ckks_e2e.py',
                    ROOT / 'paper/experiments/ckks_tfhe_workloads.json',
                    ROOT / 'paper/experiments/heongpu_ckks_e2e_bench.cu',
                    ROOT / 'paper/experiments/gpu_thermal_guard.hpp',
                    ROOT / 'scripts/prepare_rns_requests.py', output / 'cpu_build_receipt.json',
                    output / 'request_validation.json', output / 'cpu_tests.xml',
                    source / 'src/include/heongpu/host/ckks/operator.cuh', request_dir / 'manifest.json', *assets[:-1]]
    static_paths += [path for path in (ROOT / 'adapters/heongpu').rglob('*') if path.is_file() and path.suffix in ('.cu', '.cpp', '.hpp', '.patch') and not path.name.startswith('tfhe_')]
    static_paths += [ROOT / path for path in ('python/cubutterfly/rns.py', 'python/cubutterfly/__init__.py',
                                            'include/cuntt/rns_c_api.h', 'src/rns_c_api.cpp')]
    static_paths += [ROOT / path for path in ('include/cuntt/rns_ntt.hpp', 'include/cuntt/rns_tuning.hpp',
                                            'src/rns_tuning.cpp', 'src/rns_ntt.cu', 'src/ntt_hybrid2d.cuh', 'src/rns_fused_ctgs.cuh')]
    static_paths += [request_dir / entry['request_file'] for entry in manifest['requests']]
    for directory in builds.values():
        static_paths += [directory / target for target in ('cubutterfly_heongpu_ckks', 'rns_request_compare',
                         'test_heongpu_ntt_adapter', 'test_heongpu_stage_batch', 'test_heongpu_graph_schedule',
                         'test_heongpu_graph_host_memory', 'wrapper/CMakeLists.txt', 'wrapper/ckks_bootstrap_parameters.hpp')]
    result['static_inputs'] = [runner.file_identity(safe_path(path)) for path in sorted(set(static_paths)) if safe_path(path).is_file()]
    result['normalized_followups'] = []
    for item in queue:
        item['receipt'] = str(output / 'job_receipts' / f"{item['id']}.json")
        result['normalized_followups'].append({
            'id': 'ckks-' + item['id'],
            'argv': ['scripts/prepare_ckks_refresh.py', '--output-dir', str(output), '--run-job', item['id'], '--gpu-uuid', '{gpu_uuid}'],
            'depends_on': ['ckks-' + dependency for dependency in item['depends_on'] if not dependency.startswith('build_')],
            'input_paths': [entry['path'] for entry in result['static_inputs']],
            'expected_receipt': item['receipt']})
    write_json(output / 'gpu_queue.json', result)
    write_json(output / 'normalized_followups.json', {
        'schema': 'cubutterfly-refresh-followups-v1',
        'jobs': result['normalized_followups'],
        'input_paths': [entry['path'] for entry in result['static_inputs']] + [str(output / 'gpu_queue.json')]})
    return result


def option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def validate_job(job, plan, gpu_uuid):
    """Reject successful processes whose requested evidence is incomplete."""
    runner = load_module('ckks_job_runner', ROOT / 'paper/experiments/run_heongpu_ckks_e2e.py')
    requests = load_module('ckks_job_requests', ROOT / 'scripts/prepare_rns_requests.py')
    argv = job['argv']
    name = job['id']
    if name.startswith('correctness_'):
        return {'validation': 'test_process_exit_zero', 'performance_evidence': False}
    path = safe_path(option(argv, '--output'))
    if name in ('tune_bootstrap_118', 'replay_bootstrap_118'):
        data = read_json(path)
        manifest = requests._load_requests(option(argv, '--manifest'))
        expected = {entry[0] for entry in manifest}
        rows = data['results']
        if (len(rows) != 118 or len({row['id'] for row in rows}) != 118 or
                {row['id'] for row in rows} != expected or any(row['status'] != 'ok' for row in rows)):
            raise ValueError('RNS tuning/replay did not cover all 118 successful unique requests')
        if data['policy'] != option(argv, '--policy') or safe_path(data['library']) != safe_path(plan['libcuntt']['path']):
            raise ValueError('RNS result identity/policy differs')
        if data.get('repetitions') != int(option(argv, '--repetitions', '5')) or data.get('available_request_count') != 118 or data.get('limit') is not None:
            raise ValueError('RNS preparation protocol/population differs')
        if name.startswith('replay_') and any(row.get('cache_hit') is not True for row in rows):
            raise ValueError('RNS strict replay requires 118 cache hits')
        if name.startswith('tune_') and any(row.get('cache_hit') is not False for row in rows):
            raise ValueError('RNS refresh must tune all 118 requests rather than replay old cache hits')
        return {'validation': '118 unique request IDs, statuses and replay cache hits', 'result': runner.file_identity(path)}
    meta = read_json(path.with_suffix('.metadata.json'))
    if meta.get('status') != 'completed' or meta.get('returncode') != 0 or meta.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('CKKS metadata failed/incomplete/wrong GPU')
    for key in ('libcuntt', 'benchmark_binary'):
        identity = meta['artifacts'][key]
        if identity.get('sha256') != runner.file_identity(safe_path(identity['path'])).get('sha256'):
            raise ValueError(f'CKKS metadata artifact changed: {key}')
    if meta['artifacts']['libcuntt']['sha256'] != plan['libcuntt']['sha256']:
        raise ValueError('CKKS metadata library does not match frozen library')
    expected_binary = safe_path(option(argv, '--build-dir')) / 'cubutterfly_heongpu_ckks'
    if safe_path(meta['benchmark_argv'][0]) != expected_binary:
        raise ValueError('CKKS consumer differs from job')
    # Wrapper adds explicit defaults; compare every explicitly requested argument.
    for index, token in enumerate(argv[2:]):
        if token.startswith('--') and token not in ('--heongpu-source', '--cached-build', '--cuntt-library', '--build-dir', '--arch', '--skip-build', '--after-preparation'):
            if token not in meta['benchmark_argv']:
                raise ValueError(f'CKKS missing requested argument: {token}')
            offset = index + 3
            if offset < len(argv) and not argv[offset].startswith('--') and option(meta['benchmark_argv'], token) != argv[offset]:
                raise ValueError(f'CKKS requested argument differs: {token}')
    export = safe_path(job['env']['CUBUTTERFLY_RNS_REQUEST_EXPORT'])
    observations = [requests.loads(line) for line in export.read_text().splitlines() if line.strip()]
    if not observations:
        raise ValueError('missing current consumer request observations')
    exported = requests.extract([export], path.parent / (path.stem + '_extracted_requests'))
    if 'bootstrap_v1_small' in name:
        historical = read_json(Path(plan['output_dir']) / 'bootstrap_prepared/requests/manifest.json')
        if {entry['id'] for entry in exported['requests']} != {entry['id'] for entry in historical['requests']}:
            raise ValueError('current bootstrap population differs from exact historical 118 requests')
    if name.startswith('prepare_'):
        if meta.get('measurement_kind') != 'preparation-diagnostic' or meta.get('performance_eligible') is not False:
            raise ValueError('preparation must remain diagnostic')
        log = path.with_suffix('.log').read_text()
        probes = [line for line in log.splitlines() if line.startswith('preparation_probe,phase=result,')]
        benchmark = meta['benchmark_argv']
        scenes = option(benchmark, '--scenes', 'keyswitch,square_relinearize_rescale,square_rescale_keyswitch' +
                        (',rotate_rows_shift1' if '--rotate' in benchmark else '')).split(',')
        expected_probes = (len(scenes) * len(option(benchmark, '--sizes').split(',')) *
                           len(option(benchmark, '--batches').split(',')) * 2 * max(2, int(option(benchmark, '--warmup', '3'))))
        if len(probes) != expected_probes or any('reference_output_changes=0,' not in line for line in probes):
            raise ValueError('preparation repeatability probe missing or failed')
        return {'validation': 'complete diagnostic metadata, repeatability probes and request population', 'unique_requests': exported['unique_request_count']}
    if meta.get('measurement_kind') != 'ordinary' or meta.get('performance_eligible') is not True:
        raise ValueError('measurement is not eligible ordinary timing')
    if meta.get('thermal_protocol') != 'cool-before-selection-and-measurement-rounds-v2':
        raise ValueError('thermal protocol differs')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get('exact_ciphertext') != '1' or row.get('mismatch_count') != '0' or
                       row.get('graph_status') != 'reserved_scratch_changed_restored_input_exact' for row in rows):
        raise ValueError('ciphertext/changed-input/restored-input validation failed')
    if name.startswith('measure_'):
        if meta['rns_preparation'].get('CUBUTTERFLY_RNS_CACHE_POLICY') != 'require' or any(item.get('cache_hit') is not True for item in observations):
            raise ValueError('formal measurement requires all cache hits without tuning')
    sys.path.insert(0, str(ROOT / 'paper/experiments'))
    summary = load_module('ckks_job_summary', ROOT / 'paper/experiments/summarize_heongpu_ckks_e2e.py')
    summary.require_performance_evidence(meta, path)
    if 'bootstrap_v1_small' not in name:
        summary.parse_rows(rows, meta, path)
        summary.validate_selection_evidence(rows, meta, path)
    else:
        profile = runner.bootstrap_profile('bootstrap_v1_small')
        if len(rows) != 2 or {row['batch'] for row in rows} != {'1', '8'}:
            raise ValueError('bootstrap must contain exactly B1 and B8')
        for row in rows:
            required = {'scene': 'regular_bootstrapping', 'N': '4096', 'parameter_set': 'bootstrap-v1-small',
                        'baseline': 'HEonGPU_native', 'batch_submission': 'independent_public_API_loop',
                        'mode': 'cuda_graph', 'repetitions': '5', 'graph_replays_per_sample': '1',
                        'Q_bits': profile['Q_bits'], 'P_bits': profile['P_bits'], 'keyswitch_method': 'II',
                        'ntt_execution': 'native-rns', 'sec_level': 'none', 'scale': '1099511627776'}
            if any(row.get(key) != value for key, value in required.items()):
                raise ValueError('bootstrap parameter/timing contract differs')
            native, treatment = float(row['native_ms']), float(row['cubutterfly_ms'])
            if not all(math.isfinite(value) and value > 0 for value in (native, treatment)) or not math.isclose(native / treatment, float(row['speedup']), rel_tol=1e-8):
                raise ValueError('bootstrap timing/speedup invalid')
    return {'validation': 'exact full matrix, ordinary eligible metadata, selection/holdout when applicable, changed/restored input, request export and formal cache hits',
            'rows': len(rows), 'unique_requests': exported['unique_request_count']}


def job_artifacts(output, job):
    paths = [output / 'job_receipts' / f"{job['id']}.log"]
    destination = option(job['argv'], '--output')
    if destination:
        path = safe_path(destination)
        paths += [path, path.with_suffix('.log'), path.with_suffix('.metadata.json'),
                  Path(str(path) + '.selection.csv'), Path(str(path) + '.samples.csv'),
                  path.parent / (path.stem + '_extracted_requests')]
    export = job.get('env', {}).get('CUBUTTERFLY_RNS_REQUEST_EXPORT')
    if export:
        paths.append(safe_path(export))
    result = sorted(set(safe_path(path) for path in paths))
    if any(not path.is_relative_to(output) for path in result):
        raise ValueError('job outputs must stay inside the CKKS output directory')
    return result


def begin_attempt(output, job, binding):
    """Only an exact previous owner can authorize moving partial artifacts."""
    attempt_path = output / 'job_receipts' / f"{job['id']}.attempt.json"
    artifacts = job_artifacts(output, job)
    previous = read_json(attempt_path) if attempt_path.exists() else None
    existing = [path for path in artifacts if path.exists()]
    if previous and previous.get('status') != 'completed' and previous.get('binding') == binding:
        if previous.get('artifacts') != [str(path) for path in artifacts]:
            raise ValueError('previous attempt artifact ownership differs')
        if previous.get('status') == 'running':
            try:
                os.kill(previous['owner_pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('previous running attempt owner is still alive')
        archive = output / 'attempt_archives' / job['id'] / previous['attempt_id']
        archive.mkdir(parents=True, exist_ok=True)
        if (archive / 'attempt.json').exists():
            if read_json(archive / 'attempt.json') != previous:
                raise ValueError('attempt archive ownership differs')
        else:
            shutil.copyfile(attempt_path, archive / 'attempt.json')
        for index, path in enumerate(artifacts):
            if path.exists():
                destination = archive / f'{index:02d}_{path.name}'
                if destination.exists():
                    raise FileExistsError(f'archive target already exists: {destination}')
                shutil.move(str(path), str(destination))
        if Path(job['receipt']).exists() and not (archive / 'job_receipt.json').exists():
            shutil.copyfile(job['receipt'], archive / 'job_receipt.json')
    elif existing or previous:
        raise FileExistsError('unknown, changed-binding or completed job artifacts cannot be replaced')
    attempt = {'attempt_id': uuid.uuid4().hex, 'status': 'running', 'binding': binding,
               'artifacts': [str(path) for path in artifacts], 'owner_pid': os.getpid(),
               'started_unix': time.time()}
    write_json(attempt_path, attempt)
    return attempt_path, attempt


def run_job(output, job_id, gpu_uuid):
    output = safe_path(output)
    plan = read_json(output / 'gpu_queue.json')
    runner = load_module('ckks_execute_runner', ROOT / 'paper/experiments/run_heongpu_ckks_e2e.py')
    runner.target_gpu_uuid({'CUDA_VISIBLE_DEVICES': gpu_uuid})
    jobs = {job['id']: job for job in plan['queue']}
    job = jobs[job_id]
    receipt = {'id': job_id, 'status': 'failed', 'returncode': 1, 'gpu_uuid': gpu_uuid, 'argv': job['argv']}
    attempt_path = None
    attempt = None
    try:
        for identity in plan['static_inputs']:
            if not identity.get('sha256') or runner.file_identity(safe_path(identity['path'])).get('sha256') != identity['sha256']:
                raise ValueError(f"frozen input changed: {identity['path']}")
        if any(item.get('status') != 'completed' for item in plan['cpu_builds']):
            raise ValueError('CPU consumer builds have not completed')
        inherited = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        for value in inherited.values():
            if value:
                safe_path(value)
        env = {**inherited, 'PATH': '/home/wt/yes/envs/cubutterfly/bin:/usr/local/cuda/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
               'LD_LIBRARY_PATH': ':'.join([str(ROOT / 'results/e13_assets/heongpu-build-e13/_deps/rmm-build'),
                                          str(ROOT / 'results/e13_assets/heongpu-build-e13/_deps/rapids_logger-build'),
                                          str(Path(plan['libcuntt']['path']).parent),
                                          '/tmp/heongpu-e13-gf2x/root/usr/lib/x86_64-linux-gnu',
                                          '/tmp/heongpu-e13-prefix-20261003/usr/lib/x86_64-linux-gnu']),
               **job['env'], 'CUDA_VISIBLE_DEVICES': gpu_uuid}
        before = runner.gpu_observation(gpu_uuid)
        receipt['gpu_observation_before'] = before
        if not before['query_ok'] or before['foreign_processes']:
            raise ValueError('GPU occupancy preflight failed')
        identity = runner.read_only_probe(['nvidia-smi', f'--id={gpu_uuid}', '--query-gpu=uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits'])
        receipt['gpu_identity'] = identity
        if identity.get('returncode') != 0:
            raise ValueError('GPU hardware identity query failed')
        binding = {'job_id': job_id, 'argv': job['argv'], 'gpu_uuid': gpu_uuid,
                   'static_inputs_sha256': hashlib.sha256(json.dumps(plan['static_inputs'], sort_keys=True).encode()).hexdigest(),
                   'environment': env}
        attempt_path, attempt = begin_attempt(output, job, binding)
        receipt['attempt_id'] = attempt['attempt_id']
        receipt['inherited_environment'] = inherited
        log = output / 'job_receipts' / f'{job_id}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as handle:
            result = subprocess.run(job['argv'], env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        receipt.update(returncode=result.returncode, log=str(log))
        if result.returncode == 0:
            after = runner.gpu_observation(gpu_uuid)
            receipt['gpu_observation_after'] = after
            if not after['query_ok'] or after['foreign_processes']:
                raise ValueError('GPU occupancy postflight failed')
            receipt['validation'] = validate_job(job, plan, gpu_uuid)
            receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed', returncode=receipt['returncode'] or 1, error=str(error))
    if attempt is not None:
        attempt.update(status=receipt['status'], returncode=receipt['returncode'], finished_unix=time.time())
        write_json(attempt_path, attempt)
    write_json(job['receipt'], receipt)
    print(json.dumps(receipt))
    return receipt['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/ckks_refresh_20261025.json')
    parser.add_argument('--build', action='store_true', help='actually compile isolated consumers on CPU; never execute them')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run-job')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.run_job:
        if not args.output_dir or not args.gpu_uuid or args.build:
            parser.error('--run-job requires --output-dir --gpu-uuid and forbids --build')
        return run_job(args.output_dir, args.run_job, args.gpu_uuid)
    config = read_json(args.config)
    if args.output_dir:
        config['output'] = str(args.output_dir)
    result = prepare(config, build=args.build, jobs=args.jobs)
    print(json.dumps({key: result[key] for key in ('status', 'auto_configurations', 'distinct_rns_requests', 'offline_mapping_ready')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
