#!/usr/bin/env python3
"""Finite fixed-leaf endpoint screen and independent guarded holdouts.

Prepare is CPU-only. Run gives screen and three holdouts one shared deadline.
Only clean guard releases contribute to selection or holdout summaries.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_controlled_experiment as controlled

EXPECTED_LIBRARY = '775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2'
GPU = os.environ.get("CUBUTTERFLY_GPU_UUID", "")
DEFAULT_CONFIG = ROOT / 'configs/experiments/nonfft_endpoint_control_20261107.json'
DEFAULT_BUILD = ROOT / 'results/a100_fp16_acceptance_20261105/build'
DEFAULT_OUTPUT = ROOT / 'results/nonfft_endpoint_control_20261107'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    controlled._atomic_write(Path(path), value)


def sha(path):
    return controlled._sha256(Path(path))


def clean_environment(build):
    env = os.environ.copy()
    for suffix in ('HARDWARE_MODEL', 'PROFILE_ROOT', 'HARDWARE_PROFILE', 'COST_MODEL',
                   'REGISTRY', 'PLAN_CACHE', 'PLATFORM_TUNING', 'JIT_CACHE', 'E13_ALGORITHM_POLICY'):
        env.pop('CUBUTTERFLY_' + suffix, None)
    env.update(LD_LIBRARY_PATH=str(build.resolve()), CUBUTTERFLY_COMPILE_MODE='research',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def make_study(config):
    cases = []
    for cell in config['cells']:
        variants = []
        fwht = cell['workload']['operator'] == 'fwht'
        for threads in cell['threads']:
            for organization in cell['organization']:
                mapping = copy.deepcopy(cell['mapping'])
                if fwht:
                    mapping.update(tile_threads=threads, reorder_columns=organization)
                else:
                    for subgraph in mapping['subgraph_mappings']:
                        subgraph.update(threads_per_block=threads, data_time=organization)
                variants.append({'id': f't{threads}-o{organization}', 'mapping': mapping})
        comparisons = []
        baseline = variants[0]
        for variant in variants[1:]:
            axes = sorted(controlled._mapping_diff(baseline['mapping'], variant['mapping'],
                          include_derived=False, nested_roots={'subgraph_mappings'} if not fwht else set()))
            comparisons.append({'baseline': baseline['id'], 'treatment': variant['id'], 'changed_axes': axes})
        cases.append({'id': cell['id'], 'experiment': 'fixed-arithmetic-leaf-endpoint-control',
                      'workload': cell['workload'], 'variants': variants, 'comparisons': comparisons})
    study = {'schema': controlled.SCHEMA, 'scope': config['scope'],
             'protocol': config['protocol'], 'cases': cases}
    controlled._validate_study(study)
    return study


def resources(config):
    rows = []
    for cell in config['cells']:
        fwht = cell['workload']['operator'] == 'fwht'
        n, batch = 1 << cell['workload']['logN'], cell['workload']['batch']
        values_bytes = n * batch * (4 if fwht else 8)
        for threads in cell['threads']:
            for organization in cell['organization']:
                rows.append({'cell': cell['id'], 'threads': threads, 'organization': organization,
                    'full_array_bytes': values_bytes,
                    'logical_data_path_bytes': 4 * values_bytes,
                    'logical_data_path_note': 'two logical regions, input-read/boundary-write then boundary-read/output-write; excludes coefficient/ready metadata and cache behavior; NOT measured HBM traffic',
                    'logical_regions': 2,
                    'expected_launches': 2 if fwht else 1,
                    'cta_organization': (f'prefix CTAs={batch * 256}; suffix CTAs={batch * 256 // organization}; each suffix CTA groups {organization} columns'
                        if fwht else f'cooperative persistent producer/consumer grid capped by occupancy and target_ctas_per_sm; each role processes up to {organization} transforms per packet; launch grid is not simply batch/data_time'),
                    'shared_memory_static_lower_bound_bytes': (max(1024, 1024 * organization) if fwht else (4 * 1025 * 8 if cell['leaf'] == 'homogeneous-radix4' else 0)),
                    'shared_memory_note': 'source-derived tile allocation lower bound, not measured allocated/shared residency',
                    'wrapper': ('same prefix/suffix templates with runtime column grouping' if fwht else
                        ('dt1 mature specialized 10x10; dt>1 homogeneous packet-loop wrapper; same arithmetic leaf' if cell['leaf'] == 'homogeneous-radix4' else 'same homogeneous warp wrapper for every dt'))})
    return rows


def snapshot(config_path, build):
    paths = [Path(__file__), config_path, ROOT / 'scripts/run_isolated_gpu.py',
             ROOT / 'python/cubutterfly/__init__.py', ROOT / 'src/ntt.cu', ROOT / 'src/butterfly.cu']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    binaries = {name: {'path': str((build / name).resolve()), 'sha256': sha(build / name)}
                for name in ('cubutterfly_bench', 'cuntt_bench', 'libcuntt.so')}
    if binaries['libcuntt.so']['sha256'] != EXPECTED_LIBRARY:
        raise RuntimeError('library differs from authorized 775a build')
    links = {}
    for name in ('cubutterfly_bench', 'cuntt_bench'):
        proc = subprocess.run(['ldd', str(build / name)], env=clean_environment(build), capture_output=True, text=True, check=True)
        if f'libcuntt.so => {build.resolve()}/libcuntt.so ' not in proc.stdout:
            raise RuntimeError('benchmark does not resolve the pinned current library: ' + name)
        links[name] = re.sub(r" \(0x[0-9a-fA-F]+\)", "", proc.stdout)
    return {'sources': sources, 'controlled_sources': controlled._source_identity(),
            'binaries': binaries, 'dynamic_links': links, 'gpu_uuid': GPU,
            'environment': {k: v for k, v in clean_environment(build).items()
                            if k.startswith('CUBUTTERFLY_') or k in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


def prepare(config_path, build, output):
    config = read(config_path)
    study = make_study(config)
    identity = snapshot(config_path, build)
    frozen = {'schema': 'nonfft-endpoint-freeze-v1', 'identity': identity, 'config': config,
              'study': study, 'resource_expectations': resources(config),
              'budget': {'ceiling_per_side_per_core': 6, 'no_repeat_padding': True,
                         'pool_shared_between_sides': True,
                         'counts': {c['id']: {'conventional': len(c['threads']),
                                    'extended': len(c['threads']) * len(c['organization'])} for c in config['cells']}},
              'selection': 'one screen trial; minimize accepted kernel_ms within each side; screen never reported as gain',
              'holdout': 'three independent guarded processes; deduplicate identical winners; reverse candidate order in process 2',
              'timing_boundary': 'C++ CUDA events around repeat full device execution calls / repeat, excludes H2D/D2H, setup, CPU correctness and public Plan oracle; may include host submission gaps; not CUDA Graph or pure arithmetic-only timing',
              'numerical_policy': 'mandatory existing full-batch C++ verification; NTT exact; FWHT retain existing correct/max_error semantics with no new threshold, plus full-output public Plan FP32 CPU-oracle errors saved separately without threshold'}
    destination = output / 'frozen.json'
    if destination.exists():
        if read(destination) != frozen:
            raise RuntimeError('frozen inputs changed; refusing to overwrite')
        return frozen
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'screen_study.json', study)
    write(output / 'screen_prepared.json', controlled.prepare_document(study, output / 'screen_study.json', build))
    write(destination, frozen)
    return frozen


def select_winners(screen, config):
    selections = []
    by_id = {case['id']: case for case in screen['cases']}
    for cell in config['cells']:
        case = by_id[cell['id']]
        accepted = [v for v in case['variants'] if v.get('preflight', {}).get('status') == 'accepted'
                    and len(v.get('trials', [])) == 1 and v['trials'][0].get('performance_accepted')]
        choices = {}
        for side in ('conventional', 'extended'):
            pool = [v for v in accepted if side == 'extended' or v['id'].endswith('-o1')]
            if not pool:
                raise RuntimeError(f'no accepted {side} candidate: {cell["id"]}')
            winner = min(pool, key=lambda v: (v['trials'][0]['kernel_ms'], v['id']))
            choices[side] = {'id': winner['id'], 'mapping': winner['preflight']['resolved_mapping'],
                             'screen_kernel_ms_selection_only': winner['trials'][0]['kernel_ms']}
        selections.append({'cell': cell['id'], 'workload': cell['workload'], 'choices': choices,
                           'same_winner': choices['conventional']['mapping'] == choices['extended']['mapping'],
                           'attempted_unique': len(case['variants']), 'accepted_unique': len(accepted)})
    return {'schema': 'nonfft-endpoint-winners-v1', 'screen_used_for_selection_only': True, 'cells': selections}


def fp32_reference(values):
    import numpy as np
    result = np.array(values, dtype=np.float32, copy=True)
    width = 1
    while width < result.shape[-1]:
        groups = result.reshape(*result.shape[:-1], -1, 2, width)
        left, right = groups[..., 0, :].copy(), groups[..., 1, :].copy()
        groups[..., 0, :], groups[..., 1, :] = left + right, left - right
        width *= 2
    return result


def public_fwht_errors(screen, build, output):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'python'))
    from cubutterfly import Library, Plan, PlanOptions, Transform, Operator, DataType
    output.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(20261107).standard_normal((16, 65536), dtype=np.float32)
    reference = fp32_reference(values)
    np.save(output / 'input.npy', values)
    np.save(output / 'reference_fp32.npy', reference)
    library = Library(build / 'libcuntt.so')
    source = torch.from_numpy(values).cuda()
    rows = []
    for case in screen['cases']:
        if case['workload']['operator'] != 'fwht':
            continue
        for variant in case['variants']:
            if variant.get('preflight', {}).get('status') != 'accepted':
                continue
            row = {'cell': case['id'], 'variant': variant['id'], 'threshold_assessed': False}
            try:
                mapping = variant['preflight']['resolved_mapping']
                with Plan(library, Transform(op=Operator.FWHT, storage=DataType.FP32,
                          extents=[65536], batch=16), PlanOptions(mapping=json.dumps(mapping)),
                          stream=torch.cuda.current_stream().cuda_stream) as plan:
                    workspace = torch.empty(plan.workspace_size(), dtype=torch.uint8, device='cuda')
                    if workspace.numel():
                        plan.set_workspace(workspace.data_ptr(), workspace.numel())
                    actual = torch.empty_like(source)
                    plan.execute(source.data_ptr(), actual.data_ptr())
                    torch.cuda.synchronize()
                    got = actual.cpu().numpy()
                    difference = got - reference
                    name = f'{case["id"]}_{variant["id"]}_error_fp32.npy'
                    np.save(output / name, difference)
                    row.update(status='complete', finite=bool(np.isfinite(got).all()),
                        max_abs=float(np.abs(difference).max()),
                        relative_l2=float(np.linalg.norm(difference.astype(np.float64)) / np.linalg.norm(reference.astype(np.float64))),
                        elements=int(got.size), error_file=name, error_sha256=sha(output / name),
                        mapping=json.loads(plan.mapping_json()))
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            write(output / 'errors.json', {'reference': 'CPU numpy FP32 Sylvester stages, unnormalized; all 1048576 elements',
                    'note': 'raw errors retained without any newly introduced threshold; outside timing', 'rows': rows})
    return rows


def stage_screen(root, build):
    study_path = root / 'screen_study.json'
    study = read(study_path)
    attempt = Path(os.environ['XD_REPORT_DIR'])
    document = controlled.run_document({}, study, study_path, build, attempt / 'screen.json', GPU, resume=False)
    public_fwht_errors(document, build, attempt / 'fwht_errors')
    winners = select_winners(document, read(root / 'frozen.json')['config'])
    write(attempt / 'winners.json', winners)


def stage_holdout(root, build, index):
    winners = read(root / 'winners.json')
    attempt = Path(os.environ['XD_REPORT_DIR'])
    device = controlled._query_hardware(build / 'cubutterfly_bench', GPU)
    document = {'schema': 'nonfft-endpoint-holdout-v1', 'process_index': index,
                'status': 'partial', 'device': device, 'frozen_sha256': sha(root / 'frozen.json'),
                'winners_sha256': sha(root / 'winners.json'), 'pid': os.getpid(), 'cases': [],
                'protocol': {'trials': 1, 'warmup': 100, 'repeat': 100, 'verify_batches': 0}}
    for cell in winners['cells']:
        unique = {choice['id']: choice for choice in cell['choices'].values()}
        entries = sorted(unique.values(), key=lambda v: v['id'], reverse=index % 2 == 1)
        case = {'id': cell['cell'], 'workload': cell['workload'], 'variants': [], 'comparisons': [],
                'binary': str(controlled._binary_for(build, cell['workload']))}
        for choice in entries:
            commands = controlled._commands(Path(case['binary']), cell['workload'], choice['mapping'], document['protocol'])
            case['variants'].append({'id': choice['id'], 'mapping': choice['mapping'],
                'preflight_command': commands['preflight'], 'timed_command': commands['timed']})
        document['cases'].append(case)
        expected = controlled._expected_semantics(cell['workload'])
        order = [v['id'] for v in case['variants']]
        case['order'] = order
        for variant in case['variants']:
            controlled._preflight_variant(case, variant, expected, device, build)
            write(attempt / 'holdout.json', document)
            controlled._measure_trial(case, variant, 0, expected, device, document['protocol'], order)
            write(attempt / 'holdout.json', document)
            if not variant.get('trials', [{}])[0].get('performance_accepted'):
                raise RuntimeError('holdout candidate failed: ' + cell['cell'] + '/' + variant['id'])
            if variant['preflight']['resolved_mapping'] != choice_mapping(cell, variant['id']):
                raise RuntimeError('holdout mapping changed from selected winner')
    document['status'] = 'complete'
    write(attempt / 'holdout.json', document)


def choice_mapping(cell, identifier):
    return next(c['mapping'] for c in cell['choices'].values() if c['id'] == identifier)


def summarize(root):
    winners = read(root / 'winners.json')
    documents = [read(root / f'holdout_{i + 1}.json') for i in range(3)]
    rows = []
    for cell in winners['cells']:
        pairs = []
        for document in documents:
            case = next(c for c in document['cases'] if c['id'] == cell['cell'])
            times = {v['id']: v['trials'][0]['kernel_ms'] for v in case['variants']}
            conventional = times[cell['choices']['conventional']['id']]
            extended = times[cell['choices']['extended']['id']]
            pairs.append({'process': document['process_index'], 'conventional_ms': conventional,
                          'extended_ms': extended, 'ratio': conventional / extended})
        rows.append({'cell': cell['cell'], 'same_winner': cell['same_winner'], 'holdout_pairs': pairs,
                     'median_paired_ratio': statistics.median(p['ratio'] for p in pairs),
                     'interpretation': 'identical mapping; ratio 1 by identity, no claimed gain' if cell['same_winner'] else 'independent holdout only; inspect all three paired ratios and variability'})
    write(root / 'summary.json', {'schema': 'nonfft-endpoint-result-v1', 'status': 'complete',
          'frozen_sha256': sha(root / 'frozen.json'), 'winners_sha256': sha(root / 'winners.json'),
          'timing_boundary': read(root / 'frozen.json')['timing_boundary'], 'rows': rows})


def guarded(root, build, name, mode, deadline, index=None):
    guard = root / 'guards' / name
    if (guard / 'release').is_dir():
        return guard / 'release'
    command = [sys.executable, str(ROOT / 'scripts/run_isolated_gpu.py'), '--gpu-uuid', GPU,
        '--report-dir', str(guard), '--idle-seconds', '5', '--until', deadline, '--',
        sys.executable, str(Path(__file__).resolve()), '--mode', mode, '--output', str(root),
        '--build-dir', str(build), '--gpu-uuid', GPU]
    if index is not None:
        command += ['--index', str(index)]
    result = subprocess.run(command, env=clean_environment(build))
    if result.returncode:
        raise RuntimeError(f'{name} guard failed or deadline expired; no automatic budget restart')
    return guard / 'release'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'run', 'stage-screen', 'stage-holdout'), default='prepare')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--index', type=int, choices=(0, 1, 2))
    parser.add_argument("--gpu-uuid", default=GPU, required=not GPU)
    args = parser.parse_args()
    globals()["GPU"] = args.gpu_uuid
    root, build = args.output.resolve(), args.build_dir.resolve()
    if args.mode.startswith('stage-'):
        frozen = read(root / 'frozen.json')
        if snapshot(args.config.resolve(), build) != frozen['identity']:
            raise RuntimeError('frozen source/build/link identity changed')
        if args.mode == 'stage-screen':
            stage_screen(root, build)
        else:
            stage_holdout(root, build, args.index)
        return
    frozen = prepare(args.config.resolve(), build, root)
    print(json.dumps({'frozen_sha256': sha(root / 'frozen.json'), 'config_sha256': sha(args.config), 'mode': args.mode}), flush=True)
    if args.mode == 'prepare':
        return
    state_path = root / 'run_state.json'
    if state_path.exists():
        deadline = read(state_path)['deadline']
    else:
        deadline = (datetime.now(timezone.utc) + timedelta(minutes=90)).isoformat()
        write(state_path, {'deadline': deadline, 'status': 'running', 'gpu_uuid': GPU})
    if datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
        raise RuntimeError('original 90 minute deadline expired')
    release = guarded(root, build, 'screen', 'stage-screen', deadline)
    write(root / 'screen.json', read(release / 'screen.json'))
    write(root / 'winners.json', read(release / 'winners.json'))
    for index in range(3):
        release = guarded(root, build, f'holdout_{index + 1}', 'stage-holdout', deadline, index)
        write(root / f'holdout_{index + 1}.json', read(release / 'holdout.json'))
    summarize(root)
    write(state_path, {'deadline': deadline, 'status': 'complete', 'gpu_uuid': GPU})


if __name__ == '__main__':
    main()
