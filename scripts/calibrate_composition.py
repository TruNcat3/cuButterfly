#!/usr/bin/env python3
"""Measure paired execution schedules against a frozen independent-stage model.

No overlap timing is used to fit its own prediction. The report distinguishes
finished measurements, model holdouts, final floating accuracy acceptance and
unidentified concurrent resource parameters.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from calibration_space import command_for
from benchmark_semantics import normalize_benchmark_semantics
from composition_inventory import build_inventory
from hardware_registry import canonical_semantics
from run_comprehensive_suite import require_exclusive_gpu
from stage_composition import resolve_batch_services
import stage_cost_model

SCHEMA = 'cubutterfly-composition-calibration-v1'
VERSION = '0.9.3.2'
THRESHOLDS = dict(median_relative_error=.10, p90_relative_error=.20,
                  top1_accuracy=.90, minimum_pairs=3)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_runtime_fingerprint(build_dir):
    """Return the runtime identity emitted by this build, including policy.

    Composition timings must never combine records from a prior binary.  The
    generated header is the same source used by the C++ runtime identity, so
    this remains auditable without launching a benchmark during preparation.
    """
    header = Path(build_dir) / 'generated' / 'build_fingerprint.hpp'
    if not header.is_file():
        return None
    match = re.search(r'CUBUTTERFLY_RUNTIME_FINGERPRINT\s+"([0-9a-f]+)"',
                      header.read_text())
    if not match:
        raise ValueError('build has no runtime fingerprint identity')
    mode = os.environ.get('CUBUTTERFLY_COMPILE_MODE')
    if mode is None:
        policy = Path(build_dir) / 'generated' / 'jit_config.hpp'
        if not policy.is_file():
            raise ValueError('build has no compile-policy identity')
        policy_match = re.search(r'CUBUTTERFLY_DEFAULT_CODEGEN_MODE\s+"([^"]+)"',
                                 policy.read_text())
        if not policy_match:
            raise ValueError('build has no compile-policy identity')
        mode = policy_match.group(1)
    if mode not in ('precompiled', 'auto', 'on-demand', 'research'):
        raise ValueError('invalid compile policy identity')
    return match.group(1) + ':' + mode


def _current_runtime_records(records, runtime_fingerprint):
    """Filter composition sources to verified samples from this build.

    Historical build priors remain in their source checkpoint, but are not
    eligible for whole-plan timing. Records without a verified sample are
    retained for the normal inventory omission accounting.
    """
    kept, dropped = [], 0
    for record in records:
        samples = record.get('samples', []) if isinstance(record, dict) else []
        verified = [sample for sample in samples
                    if isinstance(sample, dict) and sample.get('correct') in (True, '1', 1)
                    and sample.get('runtime_fingerprint')]
        if verified and any(sample.get('runtime_fingerprint') != runtime_fingerprint
                            for sample in verified):
            dropped += 1
            continue
        kept.append(record)
    return kept, dropped


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def prediction_is_covered(value, point):
    mapping = json.loads(point['mapping_json'])
    # No service projection currently exists for factor slices. Never qualify
    # them by scaling a whole-group service as if it were a measured slice.
    if int(mapping.get('factor_slices', 1)) > 1:
        return False
    return bool(value.get('kernel_ms', 0) > 0
                and not value.get('whole_plan_fallback_used', True)
                and not value.get('missing_primitives')
                and not value.get('extrapolated_primitives')
                and value.get('schedule_model') != 'serial-upper-estimate')


def summarize(inventory, rows):
    by_id = {row['id']: row for row in rows}
    requests = inventory['requests']
    overlap = [r for r in requests if r['role'] == 'overlap']
    measured = [r for r in rows if r['status'] == 'measured']
    failures = [r for r in rows if r['status'] in ('failed', 'correctness-failure')]
    paired = {}
    for request in requests:
        paired.setdefault(request['pair_id'], []).append(request)
    pairs = []
    errors = []
    for identifier, members in paired.items():
        samples = [by_id.get(r['id']) for r in members]
        complete = all(s and s['status'] == 'measured' for s in samples)
        covered = complete and all(s['prediction_covered'] for s in samples)
        item = dict(pair_id=identifier, family=members[0]['family'], complete=complete,
                    prediction_covered=covered, workload=members[0]['workload'])
        if complete:
            winner = min(samples, key=lambda s: s['median_kernel_ms'])
            role_rows = {s['role']: s for s in samples}
            item.update(measured_winner=winner['role'],
                        overlap_speedup_vs_bulk=role_rows['bulk']['median_kernel_ms'] /
                        role_rows['overlap']['median_kernel_ms'])
            if 'sliced-serial' in role_rows:
                item['overlap_speedup_vs_same_slices'] = (
                    role_rows['sliced-serial']['median_kernel_ms'] /
                    role_rows['overlap']['median_kernel_ms'])
        if covered:
            predicted_winner = min(samples, key=lambda s: s['prediction']['kernel_ms'])
            item.update(predicted_winner=predicted_winner['role'],
                        top1_match=predicted_winner['role'] == winner['role'],
                        latency_regret=predicted_winner['median_kernel_ms'] / winner['median_kernel_ms'])
            # Count each unique execution once in error statistics; paired
            # rankings may legitimately compare one bulk execution to several tiles.
        pairs.append(item)
    unique = {}
    for row in measured:
        if row['prediction_covered']:
            unique[digest(row['point'])] = abs(row['prediction']['kernel_ms'] / row['median_kernel_ms'] - 1)
    errors = list(unique.values())
    eligible = [p for p in pairs if p['prediction_covered']]
    missing = sum(not row['prediction_covered'] for row in measured)
    median, p90 = percentile(errors, .5), percentile(errors, .9)
    top1 = statistics.mean(p['top1_match'] for p in eligible) if eligible else None
    all_measured = len(measured) == len(requests)
    validation_status = ('unvalidated' if not overlap else
                         'incomplete' if not all_measured else
                         'uncovered' if missing else
                         'insufficient-holdout' if len(eligible) < THRESHOLDS['minimum_pairs'] else
                         'passed' if median <= THRESHOLDS['median_relative_error'] and
                         p90 <= THRESHOLDS['p90_relative_error'] and top1 >= THRESHOLDS['top1_accuracy'] else
                         'failed')
    status = ('failed' if failures else 'missing-overlap-population' if not overlap else
              'complete-observed' if all_measured else 'incomplete')
    return dict(status=status, requested_requests=len(requests), measured_requests=len(measured),
                overlap_requests=len(overlap), measured_overlap_requests=sum(
                    by_id.get(r['id'], {}).get('status') == 'measured' for r in overlap),
                failed_requests=len(failures), missing_predictions=missing,
                inventory_complete=False, concurrency_parameters_identified=False,
                validation=dict(status=validation_status, thresholds=THRESHOLDS,
                    scope='predeclared paired whole-plan holdouts; no fitting to these timings',
                    pairs=len(pairs), covered_pairs=len(eligible), median_relative_error=median,
                    p90_relative_error=p90, top1_accuracy=top1,
                    max_latency_regret=max((p['latency_regret'] for p in eligible), default=None)),
                pairs=pairs)


def _benchmark_semantics(value):
    normalized = normalize_benchmark_semantics(value)
    result = canonical_semantics(normalized)
    matrices = normalized.get('stage_matrix', [])
    if isinstance(matrices, str):
        matrices = [matrices]
    # The CLI accepts repeated comma-separated matrices; CSV uses colon/x.
    # Compare their exact double values instead of ignoring this semantic field.
    result['stage_matrices'] = tuple(tuple(float(v).hex() for v in matrix.split(','))
                                    for matrix in matrices)
    return result


def validate_sample(point, sample, hardware, runtime_fingerprint):
    actual = dict(sample)
    if point['operator'] == 'ntt':
        actual.update(operator='ntt', precision='word' + sample['word_bits'],
                      direction='inverse' if sample['inverse'] == '1' else 'forward',
                      placement=sample['output_order'])
    if actual.get('correct') != '1' or int(actual.get('verified_batches', -1)) != int(point['batch']):
        raise ValueError('full-batch benchmark verification failed')
    if actual.get('runtime_fingerprint') != runtime_fingerprint:
        raise ValueError('benchmark and described runtime differ')
    if json.loads(actual['mapping_json']) != json.loads(point['mapping_json']):
        raise ValueError('benchmark resolved a different execution mapping')
    desired = _benchmark_semantics(point)
    observed = _benchmark_semantics(actual)
    for key, value in desired.items():
        if value != observed.get(key):
            raise ValueError('benchmark semantics differ: ' + key)
    for key in ('device', 'compute_capability'):
        if str(actual.get(key)) != str(hardware[key]):
            raise ValueError('benchmark hardware differs: ' + key)
    latency = float(actual['kernel_ms'])
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError('invalid whole-plan timing')
    actual['measurement_exclusive_gpu'] = '1'
    return actual


def run_calibration(build_dir, output_dir, profile, records, *, trials=3, warmup=100,
                    repeat=100, resume=False, run=None, guard=None, max_elements=1048576):
    if trials < 1 or warmup < 0 or repeat < 1:
        raise ValueError('invalid composition timing protocol')
    build_dir, output_dir = Path(build_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'composition_calibration.json'
    source_runtime = profile.get('runtime_fingerprint') or _build_runtime_fingerprint(build_dir)
    if records:
        records, dropped_records = _current_runtime_records(records, source_runtime)
    else:
        dropped_records = 0
    if source_runtime and not profile.get('runtime_fingerprint'):
        profile['runtime_fingerprint'] = source_runtime
    inventory = build_inventory(records, max_elements=max_elements)
    profile = copy.deepcopy(profile)
    if inventory['requests'] and not profile.get('runtime_fingerprint'):
        fingerprints = {sample.get('runtime_fingerprint') for record in records
                        if record.get('correct') for sample in record.get('samples', [])
                        if sample.get('runtime_fingerprint')}
        if len(fingerprints) != 1:
            raise ValueError('composition profile requires one verified source runtime identity')
        profile['runtime_fingerprint'] = fingerprints.pop()
    sources = [Path(__file__), Path(__file__).with_name('composition_inventory.py'),
               Path(__file__).with_name('stage_composition.py'), Path(stage_cost_model.__file__),
               Path(__file__).with_name('stage_service_model.py'),
               Path(__file__).with_name('calibration_space.py'),
               Path(__file__).with_name('benchmark_semantics.py'),
               Path(__file__).with_name('hardware_registry.py'),
               Path(__file__).with_name('schedule_cost_model.py'),
               Path(__file__).with_name('pipeline_schedule_model.py')]
    inputs = dict(version=VERSION, inventory_sha256=digest(inventory), profile_sha256=digest(profile),
                  records_sha256=digest(records), sources={p.name: file_hash(p) for p in sources},
                  protocol=dict(trials=trials, warmup=warmup, repeat=repeat, verify_batches=0),
                  environment={k:v for k,v in os.environ.items()
                               if k != 'CUBUTTERFLY_COMPILE_BUDGET_FILE' and
                               (k.startswith('CUBUTTERFLY_') or k in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING'))})
    from jit_calibration_identity import calibration_identity
    jit_runtime = calibration_identity()
    if jit_runtime is not None:
        inputs["jit_runtime"] = jit_runtime
    binaries = [build_dir / name for name in ('cubutterfly_bench', 'cuntt_bench', 'cubutterfly_stage_microbench')]
    libraries = sorted(build_dir.glob('libcuntt.so*'))
    if not libraries and (build_dir / 'libcuntt.a').is_file():
        libraries = [build_dir / 'libcuntt.a']
    if inventory['requests']:
        if not libraries:
            raise ValueError('composition measurements require the library identity')
        inputs['binaries'] = {str(p): file_hash(p) for p in binaries + libraries}
    report = dict(schema=SCHEMA, identity=inputs, scope=inventory['scope'], status='running',
                  inventory=inventory, rows=[], model_policy='frozen independent-stage model; no overlap fit',
                  floating_acceptance='benchmark diagnostic only; campaign error-report acceptance remains pending',
                  timing_excludes='compilation, planning, input transfers and CPU reference checking',
                  source_runtime_identity=source_runtime,
                  dropped_prior_records=dropped_records)
    if output.exists():
        if not resume:
            raise ValueError('composition report exists; use --resume after identity verification')
        report = json.loads(output.read_text())
        if report['identity'] != inputs:
            raise ValueError('composition resume identity mismatch')

    def save():
        report['summary'] = summarize(inventory, report['rows'])
        write(output, report)

    if not inventory['requests']:
        report['status'] = 'missing-overlap-population'
        save()
        return report
    signatures = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                  for p in binaries + libraries + sources}
    run = run or (lambda command, **kwargs: subprocess.run(command, text=True, capture_output=True, **kwargs))
    guard = guard or require_exclusive_gpu
    call_number = len(list(output_dir.glob('call-*.json')))

    def check_files():
        for path, signature in signatures.items():
            p = Path(path)
            if (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) != signature:
                raise ValueError('composition measurement source changed: ' + path)

    def checked_run(command):
        nonlocal call_number
        check_files()
        guard()
        result = run(command, check=False)
        call_number += 1
        write(output_dir / f'call-{call_number:05d}.json', dict(command=list(map(str, command)),
              returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        guard()
        check_files()
        result.check_returncode()
        return result

    try:
        hardware = next(csv.DictReader(io.StringIO(checked_run(
            [str(build_dir / 'cubutterfly_bench'), '--device-identity']).stdout)))
        for key in ('device', 'compute_capability', 'global_memory_bytes'):
            if str(hardware[key]) != str(profile[key]):
                raise ValueError('composition profile belongs to another device: ' + key)
        model = stage_cost_model.fit([], profile)
        model_id = digest(model)
        if report.get('prediction_model_sha256', model_id) != model_id:
            raise ValueError('frozen composition prediction model changed')
        report.update(hardware=hardware, prediction_model_sha256=model_id)
        save()
        descriptions = {}

        def describe(point):
            key = digest(point)
            if key not in descriptions:
                descriptions[key] = json.loads(checked_run([str(build_dir / 'cubutterfly_stage_microbench'),
                    '--describe-only', '--point-json', json.dumps(point)]).stdout)
            return descriptions[key]

        by_id = {row['id']: row for row in report['rows']}
        if len(by_id) != len(report['rows']) or set(by_id) - {r['id'] for r in inventory['requests']}:
            raise ValueError('composition request index mismatch')
        measured_cache = {digest(r['point']): r for r in report['rows'] if r['status'] == 'measured'}
        for request in inventory['requests']:
            row = by_id.get(request['id'])
            if row and row['status'] == 'measured':
                if row['point'] != request['point'] or len(row['samples']) != trials:
                    raise ValueError('completed composition row identity/trial count mismatch')
                for previous in row['samples']:
                    validate_sample(row['point'], previous, hardware, profile['runtime_fingerprint'])
                median = statistics.median(float(s['kernel_ms']) for s in row['samples'])
                if row['median_kernel_ms'] != median:
                    raise ValueError('completed composition median mismatch')
                if row.get('evidence_sha256') != digest({k:v for k,v in row.items() if k != 'evidence_sha256'}):
                    raise ValueError('completed composition evidence checksum mismatch')
                continue
            if row and row['status'] != 'running':
                raise ValueError('failed or unavailable request requires investigation before retry')
            point = request['point']
            if not row:
                row = {**request, 'status': 'running', 'samples': []}
                report['rows'].append(row)
                save()
            description = describe(point)
            if description.get('status') != 'resolved':
                row.update(status='unavailable', reason=description.get('reason', 'unresolved physical plan'))
                save()
                continue
            actual = description['sample']
            if actual.get('runtime_fingerprint') != profile.get('runtime_fingerprint'):
                raise ValueError('stage probe and calibration profile runtime differ')
            if json.loads(actual['mapping_json']) != json.loads(point['mapping_json']):
                raise ValueError('describe changed the requested mapping')
            projection = resolve_batch_services(point, description, describe)
            sample = {**actual, **json.loads(actual['mapping_json']),
                      'execution_groups_json': json.dumps(description['groups']),
                      'stage_service_projection': projection, 'descriptor_source': 'actual-stage-probe'}
            try:
                prediction = stage_cost_model.predict(sample, model, explain=True)
                covered = prediction_is_covered(prediction, point)
                if projection.get('status') == 'unsupported':
                    covered = False
                row.update(prediction=prediction, prediction_covered=covered)
            except (ValueError, KeyError, TypeError) as error:
                row.update(prediction={}, prediction_covered=False, prediction_error=repr(error))
            row.update(stage_service_projection=projection, physical_groups=description['groups'])
            save()  # Freeze the prediction before observing any whole-plan timing.
            cached = measured_cache.get(digest(point))
            if cached:
                row.update(samples=copy.deepcopy(cached['samples']), reused_request_id=cached['id'])
            binary = build_dir / ('cuntt_bench' if point['operator'] == 'ntt' else 'cubutterfly_bench')
            command = command_for(binary, point) + ['--warmup', str(warmup), '--repeat', str(repeat),
                                                  '--verify', '--verify-batches', '0', '--csv']
            row['command'] = command
            for previous in row['samples']:
                validate_sample(point, previous, hardware, actual['runtime_fingerprint'])
            if len(row['samples']) > trials:
                raise ValueError('extra trials in resumed composition row')
            while len(row['samples']) < trials:
                result = checked_run(command)
                values = list(csv.DictReader(io.StringIO(result.stdout)))
                if len(values) != 1:
                    raise ValueError('benchmark must return exactly one result')
                row['samples'].append(validate_sample(point, values[0], hardware, actual['runtime_fingerprint']))
                save()
            row.update(status='measured', median_kernel_ms=statistics.median(float(s['kernel_ms']) for s in row['samples']))
            row['evidence_sha256'] = digest({k:v for k,v in row.items() if k != 'evidence_sha256'})
            measured_cache[digest(point)] = row
            save()
            print(f"composition {len(measured_cache)} unique executions: {request['family']} {request['role']}", flush=True)
        check_files()
        report['status'] = 'execution-finished'
        save()
        return report
    except BaseException as error:
        report.update(status='stopped-investigation-required', error=repr(error))
        if 'row' in locals() and row['status'] == 'running':
            row.update(status='failed', error=repr(error))
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--operator-calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-elements', type=int, default=1048576)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=100)
    args = parser.parse_args(argv)
    records = json.loads(args.operator_calibration.read_text())
    if args.prepare_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / 'composition_inventory.json'
        if output.exists():
            raise ValueError('use a fresh preparation output directory')
        inventory = build_inventory(records, max_elements=args.max_elements)
        inventory['sources'] = dict(operator_calibration=str(args.operator_calibration.resolve()),
                                    operator_sha256=file_hash(args.operator_calibration),
                                    profile=str(args.profile.resolve()), profile_sha256=file_hash(args.profile))
        write(output, inventory)
        print(json.dumps(inventory['coverage'], indent=2))
        return 0
    profile = json.loads(args.profile.read_text())
    report = run_calibration(args.build_dir, args.output_dir, profile, records,
        trials=args.trials, warmup=args.warmup, repeat=args.repeat,
        resume=args.resume, max_elements=args.max_elements)
    print(json.dumps(report['summary'], indent=2))
    return 0 if report['summary']['status'] == 'complete-observed' else 1


if __name__ == '__main__':
    sys.exit(main())
