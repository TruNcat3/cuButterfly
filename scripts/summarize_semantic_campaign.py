#!/usr/bin/env python3
"""CPU-only, integrity-checked summary of one semantic campaign checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


REVIEW = 'accuracy-review-required'
GAPS = {'unsupported', 'missing-lowering', 'runner-gap', 'validation-unavailable',
        'resource-limit', 'missing-baseline'}
ERRORS = {'correctness-failure', 'runtime-error', 'runner-error', 'baseline-error'}


def digest(value):
    serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            checksum.update(block)
    return checksum.hexdigest()


def load_checkpoint(path):
    """Reject incomplete integrity metadata rather than admitting legacy rows."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get('schema') != 'cubutterfly-semantic-results-v1':
        raise ValueError('unsupported results schema')
    indices = report['rows']
    if report.get('rows_sha256') != digest(indices):
        raise ValueError('rows checksum mismatch')
    rows, seen = [], set()
    for index in indices:
        case_id = index['id']
        if case_id in seen:
            raise ValueError('duplicate case id: ' + case_id)
        seen.add(case_id)
        relative = 'rows/' + case_id + '.json'
        if index.get('path') != relative or Path(case_id).name != case_id or case_id in ('.', '..'):
            raise ValueError('invalid row checkpoint path')
        row_path = path.parent / relative
        if not row_path.is_file():
            raise ValueError('missing row checkpoint: ' + relative)
        envelope = json.loads(row_path.read_text())
        row = envelope['row']
        checksum = digest(row)
        if checksum != envelope.get('sha256') or checksum != index.get('sha256'):
            raise ValueError('row checkpoint checksum mismatch: ' + relative)
        if row.get('id') != case_id or row.get('status') != index['status']:
            raise ValueError('row checkpoint ID/status mismatch: ' + relative)
        names = [backend['name'] for backend in row.get('baselines', [])]
        if len(names) != len(set(names)) or 'cubutterfly' in names:
            raise ValueError('duplicate backend in case: ' + case_id)
        for implementation in [row, *row.get('baselines', [])]:
            artifact = implementation.get('error_artifact')
            if artifact is not None:
                artifact_path = Path(artifact['path'])
                if not artifact_path.is_absolute():
                    artifact_path = path.parent / artifact_path
                if not artifact_path.is_file() or file_digest(artifact_path) != artifact['sha256']:
                    raise ValueError('missing or corrupted error artifact: ' + str(artifact_path))
        rows.append(row)
    return report, rows


def category(implementation):
    value = implementation.get('category', 'unreported')
    return 'native-contract' if value == 'native' else value


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def value_range(values):
    finite = [value for value in values if finite_number(value)]
    return {'min': min(finite) if finite else None, 'max': max(finite) if finite else None,
            'finite_count': len(finite), 'unavailable_or_nonfinite_count': len(values) - len(finite)}


def median_ms(implementation):
    value = implementation.get('timing', {}).get('median_ms')
    return value if finite_number(value) and value > 0 else None


def exact_pass(implementation):
    correctness = implementation.get('correctness', {})
    return (implementation.get('status') == 'pass' and correctness.get('exact') is True
            and correctness.get('mismatches') == 0)


def summarize(results_path):
    report, rows = load_checkpoint(results_path)
    groups, pairs, cases = {}, {}, []
    baseline_counts = Counter()
    population_counts = Counter()
    for row in rows:
        semantics = row['descriptor']['semantics']
        axes = tuple(semantics[key] for key in ('operator', 'storage', 'compute', 'fft_type'))
        integer = semantics['storage'] in (8, 9)
        population_counts.update(row.get('populations', []))
        case = {'id': row['id'], 'status': row['status'], 'populations': row.get('populations', []),
                'backends': []}
        implementations = [('cubutterfly', row), *[(b['name'], b) for b in row.get('baselines', [])]]
        for name, implementation in implementations:
            status = implementation['status']
            effective = REVIEW if not integer and status == 'pass' else status
            key = (*axes, category(implementation))
            group = groups.setdefault(key, {})
            backend = group.setdefault(name, {'status_counts': Counter(), 'recorded_status_counts': Counter(),
                                              'correctness': [], 'timings': []})
            backend['status_counts'][effective] += 1
            backend['recorded_status_counts'][status] += 1
            correctness = implementation.get('correctness')
            if correctness is not None:
                backend['correctness'].append(correctness)
            timing = median_ms(implementation)
            if timing is not None:
                backend['timings'].append(timing)
            case['backends'].append({'name': name, 'status': effective, 'recorded_status': status,
                                     'category': category(implementation), 'correctness': correctness,
                                     'error_artifact': implementation.get('error_artifact'),
                                     'timing': implementation.get('timing')})
            if name == 'cubutterfly':
                continue
            baseline_counts['total'] += 1
            bucket = ('not_applicable' if status == 'not-applicable' else
                      'awaiting_accuracy_review' if effective == REVIEW else 'missing' if status in GAPS
                      else 'errors' if status in ERRORS else 'exact_pass' if integer and exact_pass(implementation)
                      else 'other')
            baseline_counts[bucket] += 1
            pair_key = (*axes, category(row), category(implementation), name)
            pair = pairs.setdefault(pair_key, {'timed': [], 'diagnostic': [], 'qualified': []})
            own_ms, baseline_ms = median_ms(row), median_ms(implementation)
            if own_ms is None or baseline_ms is None:
                continue
            observation = {'cubutterfly_ms': own_ms, 'baseline_ms': baseline_ms,
                           'baseline_over_cubutterfly_ratio': baseline_ms / own_ms}
            pair['timed'].append(observation)
            if integer:
                if exact_pass(row) and exact_pass(implementation):
                    pair['qualified'].append(observation)
            elif (row['status'] in ('pass', REVIEW) and status in ('pass', REVIEW)
                  and row.get('correctness') is not None and correctness is not None):
                pair['diagnostic'].append(observation)
        cases.append(case)
    grouped = []
    for key, backends in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(('op', 'storage', 'compute', 'fft_type', 'category'), key))
        result['backends'] = {}
        for name, backend in sorted(backends.items()):
            metrics = backend.pop('correctness')
            timings = backend.pop('timings')
            result['backends'][name] = dict(backend, all_element_error_summary={
                'case_count': len(metrics),
                'max_abs': value_range([m.get('max_abs') for m in metrics]),
                'max_relative_nonzero_reference': value_range([m.get('max_relative_nonzero_reference') for m in metrics]),
                'l2_relative': value_range([m.get('l2_relative') for m in metrics]),
                'quantiles': 'per-case all-element distributions in cases[].backends[].correctness.error_distributions; no pooled quantile approximation'},
                actual_timing_count=len(timings), median_ms_range=value_range(timings))
        grouped.append(result)
    paired = []
    for key, pair in sorted(pairs.items(), key=lambda item: str(item[0])):
        integer = key[1] in (8, 9)
        observations = pair['qualified'] if integer else pair['diagnostic']
        result = dict(zip(('op', 'storage', 'compute', 'fft_type', 'cubutterfly_category',
                           'baseline_category', 'baseline'), key))
        result.update(actual_timing_paired_count=len(pair['timed']), paired_count=len(observations),
                      qualified_count=len(pair['qualified']),
                      accuracy_status='exact-only' if integer else REVIEW,
                      ratio_role='observed diagnostic; pending accuracy acceptance' if not integer else 'both exact validations passed',
                      cubutterfly_median_ms_range=value_range([o['cubutterfly_ms'] for o in observations]),
                      baseline_median_ms_range=value_range([o['baseline_ms'] for o in observations]),
                      observed_baseline_over_cubutterfly_ratio_range=value_range([o['baseline_over_cubutterfly_ratio'] for o in observations]))
        paired.append(result)
    remaining = report.get('remaining_case_ids', [])
    return {'schema': 'cubutterfly-semantic-summary-v1',
            'source': {'path': str(Path(results_path).resolve()), 'rows_sha256': report['rows_sha256'],
                       'results_sha256': file_digest(Path(results_path)), 'identity': report.get('identity')},
            'scope': report.get('scope', report.get('manifest_metrics')),
            'population': report.get('identity', {}).get('population'),
            'population_counts': dict(population_counts), 'status': report.get('status'),
            'row_count': len(rows), 'remaining': len(remaining), 'remaining_case_ids': remaining,
            'invalid_count': report.get('invalid_request_count', 0),
            'invalid_requests_source': report.get('invalid_requests_source'),
            'recorded_status_counts': dict(Counter(row['status'] for row in rows)),
            'status_counts': dict(Counter(REVIEW if row['descriptor']['semantics']['storage'] not in (8, 9)
                                        and row['status'] == 'pass' else row['status'] for row in rows)),
            'baseline_counts': {key: baseline_counts[key] for key in
                                ('total', 'not_applicable', 'missing', 'errors', 'awaiting_accuracy_review', 'exact_pass', 'other')},
            'timing_boundary': 'complete reported execution, including actual order/layout adapters; no reconstructed core-only timing',
            'accuracy_policy': 'floating errors require review against reference backends before thresholds; gaps are neither passes nor failures; integer/NTT qualification requires both exact passes',
            'groups': grouped, 'pairs': paired, 'cases': cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.results.resolve() == args.output.resolve():
        parser.error('output must differ from results')
    summary = summarize(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
