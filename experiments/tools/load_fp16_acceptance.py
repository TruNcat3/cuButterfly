"""Read the committed nine-request public DEFAULT FP16 cohort without CUDA.

MEASURE and installed DEFAULT are separate evidence, never latency alternatives.
Incomplete campaigns and uncommitted child results are rejected.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT / 'results/a100_fp16_acceptance_20261105'
CASES = ((4096, 256), (4096, 2048), (8192, 256), (16384, 64), (16384, 512),
         (32768, 32), (32768, 256), (262144, 4), (262144, 32))
UUID = 'GPU-00000000-0000-0000-0000-000000000001'
GPU = 'NVIDIA A100 80GB PCIe'
PROTOCOL = '20 complete calls per CUDA graph; 7 replays; 10 warmup'
COHORT = 'fp16_public_default9_20261105'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def positive(value):
    value = float(value)
    require(math.isfinite(value) and value > 0, 'Invalid Graph latency')
    return value


def graph_latency(result):
    timing = result['graph_timing']
    require(timing['protocol'] == PROTOCOL, 'Incomplete Graph protocol')
    samples = list(map(positive, timing['samples_ms']))
    require(len(samples) == 7, 'Expected seven Graph replays')
    median = positive(timing['median_ms'])
    require(math.isclose(statistics.median(samples), median, rel_tol=1e-12), 'Graph median mismatch')
    return median * 1000


def load_fp16_acceptance(directory=None):
    try:
        return _load_fp16_acceptance(directory)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError(f'Incomplete or malformed FP16 acceptance evidence: {error}') from error


def _load_fp16_acceptance(directory=None):
    directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
    hashes = {}

    def read(relative):
        path = directory / relative
        require(path.is_file(), f'Incomplete FP16 acceptance: missing {relative}')
        hashes[str(relative)] = sha(path)
        try:
            return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'Invalid acceptance artifact: {relative}') from error

    def committed(relative, marker):
        result = read(relative)
        commit = read('post_completed/' + marker + '.json')
        require(commit.get('result_sha256') == hashes[relative], 'Uncommitted or changed child result: ' + relative)
        require(Path(commit['result']).as_posix().endswith('/' + relative), 'Parent commit result path mismatch')
        require(bool(commit.get('attempt')), 'Missing parent attempt receipt')
        return result

    summary = read('summary.json')
    require(summary.get('status') == 'complete-error-report-only', 'FP16 summary is not complete')
    state = read('status.json')
    require(state.get('status') == 'complete' and state.get('phase') == 'all', 'Parent campaign has not committed completion')
    installation = read('installation.json')
    require(installation.get('status') == 'installed-validated', 'Installed acceptance is incomplete')
    require(summary['installation'] == installation, 'Summary installation receipt mismatch')
    for field, path in [('stage_model', 'stage/model-receipt.json'), ('tuning', 'tuning/receipt.json')]:
        require(summary[field] == read(path), 'Summary receipt mismatch: ' + field)
    tuning_commit = read('post_completed/tuning.json')
    require(tuning_commit.get('status') == 'complete' and
            tuning_commit['receipt_sha256'] == hashes['tuning/receipt.json'] and
            tuning_commit['registry_sha256'] == sha(directory / 'registry.json'), 'Tuning parent commit mismatch')
    identity = read('build/identity_receipt.json')
    library_hash = identity['library_sha256']
    runtime = identity['runtime_fingerprint'] + ':research'
    require(installation['library_sha256'] == library_hash and
            installation['runtime_fingerprint'] == identity['runtime_fingerprint'] and
            tuning_commit['library_sha256'] == library_hash, 'Library identity mismatch')
    hardware = read('stage/calibration_device.json')
    require(hardware['device'] == GPU and hardware['global_memory_bytes'] == 84974239744 and
            hardware['compute_capability'] == '8.0', 'GPU hardware identity mismatch')
    manifests = {folder: read(folder + '/manifest.json') for folder in ('public', 'installed_public', 'external')}
    external = manifests['external']
    require(external['gpu_uuid'] == UUID and external['library_sha256'] == library_hash, 'External identity mismatch')
    expected = {f'n{n}-b{b}': (n, b) for n, b in CASES}
    summary_rows = {f"n{row['n']}-b{row['batch']}": row for row in summary['rows']}
    require(len(summary['rows']) == len(summary_rows) == 9 and set(summary_rows) == set(expected), 'Nine-request summary denominator mismatch')
    evidence = {'default': [], 'measure': [], 'installed': []}
    pairs = []

    def terminal(result, vendor=False):
        require(result['status'] == ('complete' if vendor else 'error-report-only'), 'Unsuccessful result')
        require(result['gpu_uuid'] == UUID and result['library_sha256'] == library_hash, 'Result GPU/library mismatch')
        if not vendor:
            require(result['gpu_name'] == GPU and result['runtime_fingerprint'] == runtime, 'Public runtime/hardware mismatch')
        require(result['all_outputs_finite'] is True, 'Nonfinite output')
        flags = ('replay_byte_equal', 'input_and_output_guards_passed') if vendor else ('direct_graph_byte_equal', 'input_output_guards_passed')
        require(all(result.get(flag) is True for flag in flags), 'Replay or guard check missing')
        return graph_latency(result)

    def output(relative, result):
        path = directory / Path(relative).parent / 'output.npy'
        require(path.is_file() and sha(path) == result['output_sha256'], 'Output artifact checksum mismatch')

    for case, (n, batch) in expected.items():
        tasks = [task for task in external['tasks'] if task['id'] == case + '-cufft']
        require(len(tasks) == 1, 'Missing or duplicate cuFFT task')
        task = tasks[0]
        require(task['variant'] == 'cufft' and task['n'] == n and task['batch'] == batch, 'cuFFT request mismatch')
        for kind in ('input', 'reference'):
            require(sha(task[kind + '_file']) == task[kind + '_sha256'], 'Prepared ' + kind + ' checksum mismatch')
        vendor_path = f'external/{case}-cufft/result.json'
        vendor = committed(vendor_path, case + '-cufft')
        require(vendor['manifest_sha256'] == hashes['external/manifest.json'] and vendor['design_point'] == task and
                vendor['vendor_sha256'] == external['vendor_sha256'] and vendor['input_sha256'] == task['input_sha256'] and
                vendor['reference_file'] == task['reference_file'] and vendor['mode'] == 'measure', 'cuFFT preparation mismatch')
        baseline_us = terminal(vendor, True)
        output(vendor_path, vendor)
        row = summary_rows[case]
        require(math.isclose(row['cufft_us'], baseline_us, rel_tol=1e-12), 'Summary baseline timing mismatch')
        for label, folder, policy in [('default', 'public', 'default'), ('measure', 'public', 'measure'),
                                      ('installed', 'installed_public', 'default')]:
            manifest = manifests[folder]
            require(manifest['schema'] == 'fp16-public-acceptance-v1' and manifest['installed'] == (label == 'installed') and
                    manifest['expected_runtime_fingerprint'] == runtime, 'Public preparation identity mismatch')
            require(manifest['bindings'].get(manifest['library']) == library_hash, 'Public library binding mismatch')
            entries = [e for e in manifest['cases'] if e['id'] == case]
            require(len(entries) == 1, 'Missing or duplicate public request')
            entry = entries[0]
            require((entry['n'], entry['batch'], entry['storage'], entry['compute'], entry['inverse'], entry['normalize_inverse'], entry['scope']) ==
                    (n, batch, 4, 0, False, False, 'public-selection'), 'Public request semantics mismatch')
            require(all(entry[k] == task[k] for k in ('input_file', 'input_sha256', 'reference_file', 'reference_sha256')), 'Paired input/reference mismatch')
            path = f'{folder}/runs/{case}/{policy}/result.json'
            result = committed(path, f'{folder}-{policy}-{case}')
            require(result['manifest_sha256'] == hashes[folder + '/manifest.json'] and result['id'] == case and result['policy'] == policy and
                    result['installed'] == (label == 'installed') and result['scope'] == 'public-selection' and
                    result['selection_source'] == (2 if policy == 'measure' else 0) and result['threshold'] is None,
                    'Public selection is not the requested DEFAULT/MEASURE path')
            if label == 'installed':
                require(Path(result['frontend']).is_relative_to(Path(installation['profile']).parents[3]), 'Installed frontend remains outside prefix')
            own_us = terminal(result)
            output(path, result)
            require(math.isclose(row['latency_us'][label], own_us, rel_tol=1e-12) and
                    math.isclose(row['speedup'][label], baseline_us / own_us, rel_tol=1e-12) and
                    row['mappings'][label] == result['actual_mapping'], 'Summary public evidence mismatch')
            pair = dict(operator='fft', precision='fp16', case=case, cohort=COHORT,
                        selection_kind='public_DEFAULT' if label == 'default' else 'public_MEASURE' if label == 'measure' else 'installed_DEFAULT',
                        variant=label, status='accuracy-review-required', numeric_policy='error-report-only',
                        source=path, source_sha256=hashes[path],
                        baseline='cufft', baseline_source=vendor_path, baseline_source_sha256=hashes[vendor_path],
                        own_us=own_us, baseline_us=baseline_us, speedup=baseline_us / own_us,
                        N=n, B=batch, actualN=n, actualB=batch, library_hash=library_hash,
                        mapping_json=json.dumps(result['actual_mapping'], sort_keys=True), gpu={'name': GPU, 'memory_bytes': 84974239744},
                        gpu_uuid=UUID, prepared_sha256=task['input_sha256'], reference_sha256=task['reference_sha256'],
                        semantics=dict(operator=0, storage=4, compute=0, shape=[n], batch=batch,
                                       inverse=0, normalize_inverse=False, fft_output_scale=1.0),
                        precision_contract='native FP16 storage arithmetic; same prepared CPU reference; vendor arithmetic unspecified',
                        cpu_error=result['cpu_error'], cufft_cpu_error=row['cufft_cpu_error'], included_in_figure=label == 'default')
            evidence[label].append(pair)
            if label == 'default':
                pairs.append(pair)
    # The parent's completion covers semantic and prior-installation continuity
    # controls too; require their committed records without adding timing pairs.
    for folder in ('public', 'installed_public'):
        require({entry['id'] for entry in manifests[folder]['cases']} == set(expected) |
                {'fp16-inverse', 'bf16-inverse', 'bf16-forward', 'fp16-fp32'}, 'Semantic control denominator mismatch')
        for entry in manifests[folder]['cases']:
            if entry['id'] in expected:
                continue
            require(entry['scope'] == 'legacy-scalar-explicit-smoke', 'Unexpected public control')
            relative = f"{folder}/runs/{entry['id']}/default/result.json"
            control = committed(relative, folder + '-' + entry['id'])
            terminal(control)
            output(relative, control)
        require(len(manifests[folder]['cases']) == 13, 'Missing semantic controls')
    for placement in ('development', 'installed'):
        for case in ('fft-fp64-n20-b16', 'fwht-fp32-n16-b16', 'fwht-fp32-n15-b32'):
            relative = f'legacy/{placement}/{case}/result.json'
            control = committed(relative, f'legacy-{placement}-{case}')
            terminal(control)
            output(relative, control)
    values = [pair['speedup'] for pair in pairs]
    public_summary = [dict(operator='fft', precision='fp16', baseline='cufft', cohort=COHORT,
        selection_kind='public_DEFAULT', pairs=9, gm=math.exp(statistics.mean(map(math.log, values))),
        wins=sum(value > 1 for value in values), min=min(values), max=max(values)),
        dict(operator='fft', precision='bf16', baseline='cufft', cohort=COHORT, selection_kind='public_DEFAULT',
             pairs=0, gm=None, wins=0, min=None, max=None, status='pending')]
    return dict(requests=[dict(pair) for pair in pairs], pairs=pairs, summary=public_summary,
        evidence=evidence, identity=dict(schema='fp16-public-acceptance-comparison-v1', cohort=COHORT,
            selection_kind='public_DEFAULT', requests=9, default_requests=9, source_sha256=hashes,
            runtime_fingerprint=runtime, library_sha256=library_hash, gpu_uuid=UUID,
            graph_protocol=PROTOCOL, bf16_status='pending; semantic smokes are not an automatic comparison cohort',
            limitations=['Nine finite FP16 forward requests; not full-matrix coverage',
                         'Numerical errors are report-only; no accuracy acceptance claim',
                         'MEASURE and installed DEFAULT retained separately; only development DEFAULT enters pairs',
                         'cuFFT internal arithmetic is unspecified']))
