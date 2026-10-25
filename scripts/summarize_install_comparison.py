#!/usr/bin/env python3
"""Summarize retained installation comparisons without invoking a GPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np

from summarize_semantic_campaign import digest, load_checkpoint


OPERATORS = {0: 'fft', 1: 'ntt', 2: 'fwht', 3: 'subset-zeta',
             4: 'superset-zeta', 5: 'structured-2x2'}
STORAGE = {2: 'fp32', 3: 'fp64', 6: 'complex64', 7: 'complex128',
           8: 'uint32', 9: 'uint64'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def artifact_check(artifact):
    raw = Path(artifact['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact['sha256']:
        raise ValueError('artifact checksum mismatch: ' + artifact['path'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        actual, reference = arrays['actual'], arrays['reference']
        if actual.dtype != reference.dtype or actual.shape != reference.shape:
            raise ValueError('same-precision artifact contract mismatch')
        result = dict(actual_nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
                      reference_nonfinite=int(np.count_nonzero(~np.isfinite(reference))))
        if actual.dtype.kind == 'u':
            result['exact_mismatches'] = int(np.count_nonzero(actual != reference))
            if result['exact_mismatches']:
                raise ValueError('integer mismatch in retained artifact')
        return result


def summarize(gpu_path, cpu_dir, output):
    checkpoint, measured = load_checkpoint(gpu_path)
    review_path = cpu_dir / 'gpu_same_precision_review.json'
    cpu_path = cpu_dir / 'results.json'
    table_path = cpu_dir / 'unified_comparison.csv'
    review = json.loads(review_path.read_text())
    cpu = json.loads(cpu_path.read_text())
    if review['identity']['gpu_results_sha256'] != sha(gpu_path):
        raise ValueError('review does not match current checkpoint')
    if digest(review['rows']) != review['rows_sha256']:
        raise ValueError('review row checksum mismatch')
    if checkpoint['identity']['manifest_sha256'] != review['identity']['manifest_sha256']:
        raise ValueError('review manifest mismatch')
    if cpu['identity']['manifest_sha256'] != review['identity']['cpu_source_manifest_sha256']:
        raise ValueError('CPU source manifest mismatch')
    by_case = {row['id']: row for row in measured}
    reviewed = {(r['case_id'], r['backend']): r for r in review['rows']}
    cpu_cases = defaultdict(list)
    for row in cpu['rows']:
        cpu_cases[row['case_id']].append(row)
    with table_path.open(newline='') as handle:
        table = {r['case_id']: r for r in csv.DictReader(handle)}
    if set(by_case) != set(table) or set(by_case) != set(cpu_cases):
        raise ValueError('joined/CPU/checkpoint case populations differ')

    classes, gaps, errors = [], [], []
    group = defaultdict(list)
    integer_counts = Counter()
    verified = 0

    def base(case_id, backend, platform, status):
        row = by_case[case_id]
        s = row['descriptor']['semantics']
        return dict(case_id=case_id, population=';'.join(row['populations']),
                    backend=backend, platform=platform, status=status,
                    operator=OPERATORS[s['operator']], precision=STORAGE[s['storage']],
                    shape='x'.join(map(str, s['shape'])), batch=s['batch'],
                    direction='inverse' if s['inverse'] else 'forward')

    def numerical(row, platform):
        nonlocal verified
        case_id, backend = row['case_id'], row['backend']
        s = row['semantics']
        if s != by_case[case_id]['descriptor']['semantics']:
            raise ValueError('numerical row semantics mismatch')
        if 'error_artifact' not in row:
            raise ValueError('measured numerical row lacks retained artifact')
        checked = artifact_check(row['error_artifact'])
        verified += 1
        if s['storage'] in (8, 9):
            if row['status'] != 'pass' or row['correctness'].get('exact') is not True:
                raise ValueError('integer row lacks exact validation')
            integer_counts[backend] += 1
            return
        e = base(case_id, backend, platform, row['status'])
        e.update(checked, reference_kind=row['reference']['kind'],
                 reference_dtype=row['reference'].get('dtype'),
                 artifact=row['error_artifact']['path'],
                 artifact_sha256=row['error_artifact']['sha256'])
        c = row['correctness']
        for key in ('elements', 'max_abs', 'max_relative_nonzero_reference',
                    'max_abs_at_zero_reference', 'rms_abs', 'l2_relative'):
            e[key] = c[key]
        for name, distribution in c['error_distributions'].items():
            for key in ('elements', 'zero_count', 'nonfinite_count'):
                e[name + '_' + key] = distribution[key]
            for q, value in distribution['quantiles'].items():
                e[name + '_q' + q] = value
        errors.append(e)
        group[(backend, e['operator'], e['precision'])].append(e)

    for row in measured:
        case_id = row['id']
        for backend, implementation in [('cubutterfly', row)] + [(b['name'], b) for b in row['baselines']]:
            status = implementation['status']
            if table[case_id][backend + '_status'] != status:
                raise ValueError('joined GPU status mismatch')
            classification = base(case_id, backend, 'gpu', status)
            classification['reason'] = implementation.get('reason', '')
            classification['support_status'] = implementation.get('support_status', '')
            classes.append(classification)
            if status in ('pass', 'accuracy-review-required'):
                numerical(reviewed[(case_id, backend)], 'gpu')
            if status == 'missing-baseline':
                coverage = cpu_cases[case_id]
                g = dict(classification)
                g.update(cpu_backends=';'.join(r['backend'] for r in coverage),
                         cpu_statuses=';'.join(r['status'] for r in coverage),
                         cpu_semantics_match=all(r['semantics'] == row['descriptor']['semantics'] for r in coverage),
                         cpu_input_match=all(r['input_sha256'] == reviewed[(case_id, 'cubutterfly')]['input_sha256'] for r in coverage),
                         coverage='CPU external measured; GPU gap remains')
                if not g['cpu_semantics_match'] or not g['cpu_input_match']:
                    raise ValueError('CPU gap coverage mismatch')
                gaps.append(g)
    for row in cpu['rows']:
        if table[row['case_id']][row['backend'] + '_status'] != row['status']:
            raise ValueError('joined CPU status mismatch')
        if row['input_sha256'] != reviewed[(row['case_id'], 'cubutterfly')]['input_sha256']:
            raise ValueError('CPU/GPU inputs differ')
        classes.append(base(row['case_id'], row['backend'], 'cpu', row['status']))
        numerical(row, 'cpu')
    groups = []
    for (backend, operator, precision), rows in sorted(group.items()):
        g = dict(backend=backend, operator=operator, precision=precision, rows=len(rows),
                 elements=sum(r['elements'] for r in rows),
                 actual_nonfinite=sum(r['actual_nonfinite'] for r in rows),
                 reference_nonfinite=sum(r['reference_nonfinite'] for r in rows))
        for metric in ('max_abs', 'max_relative_nonzero_reference', 'l2_relative'):
            values = [r[metric] for r in rows]
            g.update({metric + '_min': min(values), metric + '_median': statistics.median(values),
                      metric + '_max': max(values)})
        groups.append(g)
    summary = dict(schema='cubutterfly-install-comparison-analysis-v1', gpu_calls=0,
                   scope='66 retained scenarios; no floating acceptance threshold',
                   inputs={str(p): sha(p) for p in (gpu_path, review_path, cpu_path, table_path)},
                   gpu_identity=checkpoint['identity'], cpu_identity=cpu['identity'],
                   cases=len(measured), classifications=len(classes),
                   gpu_status_counts=dict(Counter(r['status'] for r in classes if r['platform'] == 'gpu')),
                   cpu_status_counts=dict(Counter(r['status'] for r in classes if r['platform'] == 'cpu')),
                   gpu_gap_slots=len(gaps), gpu_gap_cases=len({r['case_id'] for r in gaps}),
                   cpu_covered_gpu_gap_slots=sum(r['cpu_semantics_match'] and r['cpu_input_match'] for r in gaps),
                   verified_same_precision_artifacts=verified,
                   integer_exact_rows=dict(integer_counts), floating_rows=len(errors),
                   floating_actual_nonfinite=sum(r['actual_nonfinite'] for r in errors),
                   floating_reference_nonfinite=sum(r['reference_nonfinite'] for r in errors),
                   aggregation='Per-row quantiles retained; group statistics describe row maxima/L2 values, not pooled element quantiles.',
                   error_groups=groups)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'classifications.csv', classes)
    write_csv(output / 'gpu_gaps_cpu_coverage.csv', gaps)
    write_csv(output / 'floating_errors.csv', errors)
    write_csv(output / 'floating_error_groups.csv', groups)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return {key: summary[key] for key in ('cases', 'classifications', 'gpu_gap_slots', 'gpu_gap_cases',
             'cpu_covered_gpu_gap_slots', 'verified_same_precision_artifacts', 'floating_rows',
             'floating_actual_nonfinite', 'floating_reference_nonfinite')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-results', type=Path, default=Path('results/install_compare_repair_20261014/measurement/results.json'))
    parser.add_argument('--cpu-dir', type=Path, default=Path('results/install_compare_20261013/cpu_external_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/install_compare_repair_20261014/analysis'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.gpu_results, args.cpu_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
