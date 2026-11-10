#!/usr/bin/env python3
"""CPU-only snapshot of released representative profiles (partial campaigns allowed).

Usage: --campaign PATH --output DIRECTORY. Profiler values are diagnostics only.
Only .sum absolute counts are totaled; percentages/rates/resources stay per kernel.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sqlite3
import subprocess
from collections import Counter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(raw):
    try:
        value = float(str(raw).replace(',', ''))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def parse_ncu(path, metrics):
    """Accept NCU long CSV or raw wide CSV with a separate physical-units row."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith('"ID",') or line.startswith('ID,'))
    rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    long = 'Metric Name' in rows[0] if rows else False
    units = {} if long else next((row for row in rows if not row.get('ID')), {})
    values = []
    for row in rows:
        if not row.get('ID') or not row.get('Kernel Name'):
            continue
        names = [row['Metric Name']] if long else metrics
        for metric in names:
            if metric not in metrics:
                continue
            raw = row.get('Metric Value') if long else row.get(metric)
            values.append(dict(kernel_id=row['ID'], kernel_name=row['Kernel Name'],
                               metric=metric, unit=row.get('Metric Unit') if long else units.get(metric),
                               raw_value=raw, value=number(raw)))
    # NCU leaves these absolute event-count units blank in raw wide output.
    for value in values:
        value['raw_unit'] = value['unit']
        if not value['unit'] and value['metric'].endswith('.sum'):
            if 'wavefronts_' in value['metric']:
                value['unit'] = 'wavefront'
            elif 'bank_conflicts_' in value['metric']:
                value['unit'] = 'conflict'
    present = {(v['kernel_id'], v['metric']) for v in values}
    kernels = {v['kernel_id']: v['kernel_name'] for v in values}
    for kernel_id, kernel_name in kernels.items():
        for metric in metrics:
            if (kernel_id, metric) not in present:
                values.append(dict(kernel_id=kernel_id, kernel_name=kernel_name, metric=metric,
                                   unit=None, raw_unit=None, raw_value=None, value=None))
    return values


def analyze_nsys(report_path, task, output):
    """Export with NSYS CPU tooling; all timeline quantities are diagnostics."""
    output.mkdir(parents=True, exist_ok=True)
    database = report_path.with_suffix('.sqlite')
    if not database.is_file():
        database = output / 'capture.sqlite'
        if not database.is_file():
            with (output / 'export.log').open('w') as log:
                result = subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output',
                                         str(database), str(report_path)], stdout=log,
                                        stderr=subprocess.STDOUT, timeout=120)
                if result.returncode:
                    raise ValueError('NSYS SQLite export failed; see ' + str(output / 'export.log'))
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        strings = dict(connection.execute('select id,value from StringIds'))
        kernels = ([dict(r) for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
                   if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else [])
        runtime = [dict(r) for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start')]
        api = Counter(strings.get(r['nameId'], str(r['nameId'])) for r in runtime)
        copies = [dict(r) for r in connection.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
        kinds = dict(connection.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')) if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
        copy_counts = Counter(kinds.get(r.get('copyKind'), str(r.get('copyKind'))) for r in copies)
        first = min((r['start'] for r in kernels), default=None)
        for i, kernel in enumerate(kernels):
            kernel['name'] = strings.get(kernel['demangledName'])
            kernel['relative_start_ns'] = kernel['start'] - first
            kernel['relative_end_ns'] = kernel['end'] - first
            kernel['duration_ns'] = kernel['end'] - kernel['start']
            kernel['gap_from_previous_end_ns'] = kernel['start'] - kernels[i-1]['end'] if i else None
        issues = []
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            issues.append('kernel activity table missing; kernel capture cannot be validated')
        if len(kernels) != task['expected_kernels']:
            issues.append('kernel count mismatch')
        if len({(r['deviceId'], r['contextId'], r['streamId']) for r in kernels}) != 1:
            issues.append('expected one device/context/stream')
        host_copies = [r for r in copies if r.get('copyKind') in (1, 2)]
        if host_copies:
            issues.append('captured HtoD/DtoH copies')
        allocation_api = {name: count for name, count in api.items()
                          if any(token in name for token in ('Malloc', 'Free', 'MemAlloc', 'MemFree'))}
        if allocation_api:
            issues.append('captured allocation/free API')
        graph_launches = sum(count for name, count in api.items() if name.startswith('cudaGraphLaunch'))
        kernel_launches = sum(count for name, count in api.items() if name.startswith('cudaLaunchKernel'))
        if task['family'] == 'factor' and (graph_launches != 1 or kernel_launches != 0):
            issues.append('expected exactly one graph launch and no direct kernel launch')
        if task['family'] == 'subgraph' and (kernel_launches != task['expected_kernels'] or graph_launches != 0):
            issues.append('direct kernel launch count mismatch')
        return dict(task_id=task['id'], status='invalid' if issues else 'complete', issues=issues,
                    report=str(report_path), report_sha256=sha(report_path), sqlite=str(database),
                    sqlite_sha256=sha(database), kernels=kernels, runtime=runtime,
                    kernel_count=len(kernels) if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else None,
                    kernel_table_present='CUPTI_ACTIVITY_KIND_KERNEL' in tables,
                    copy_table_present='CUPTI_ACTIVITY_KIND_MEMCPY' in tables,
                    api_counts=dict(api), copy_counts=dict(copy_counts), copy_count=len(copies),
                    copies=copies, allocation_api_counts=allocation_api,
                    scope='Profiler diagnostics only; no inference about intra-kernel overlap.')
    finally:
        connection.close()


def summarize(campaign, nsys_output=None):
    data = json.loads(campaign.read_text())
    if data['schema'] != 'cubutterfly-representative-profile-campaign-v1':
        raise ValueError('unexpected campaign schema')
    digest = sha(campaign)
    provenance = {str(campaign): digest}
    tasks, samples, ncu, totals, controls, nsys = [], [], [], [], [], []
    def read(path):
        provenance[str(path)] = sha(path)
        return json.loads(path.read_text())
    for task in data['tasks']:
        release = campaign.parent / 'jobs' / task['id'] / 'release'
        entry = dict(task, status='missing', release=str(release))
        tasks.append(entry)
        if not (release / 'complete.json').is_file():
            continue
        try:
            receipt = read(release / 'complete.json')
            if receipt.get('campaign_sha256') != digest or receipt.get('gpu_uuid') != data['gpu_uuid']:
                raise ValueError('release receipt identity mismatch')
            metadata = read(release / 'task.json')
            if metadata.get('task') != task or metadata.get('campaign_sha256') != digest:
                raise ValueError('task metadata identity mismatch')
            if task['kind'] == 'ordinary':
                path = release / ('stdout.txt' if task['family'] == 'subgraph' else 'application/results.json')
                if task['family'] == 'subgraph':
                    provenance[str(path)] = sha(path)
                    text = path.read_text()
                    rows = list(csv.DictReader(io.StringIO(text[text.index('device,'):])))
                    expected = {f'{core}/{org}' for core in ('native', 'cufftdx-thread')
                                for org in ('stagewise-hbm', 'resident-prefix-2')}
                    parsed = [(f"{r['core']}/{r['organization']}", r['kernel_ms'], r) for r in rows]
                else:
                    observed = read(path)
                    parsed = [(r['id'], r['median_ms'], r) for r in observed['results']]
                    expected = {f'{core}-t{t}-d{d}' for core in ('native', 'cufftdx')
                                for t, d in ((1, 0), (4, 0), (4, 1), (4, 2))}
                    groups = read(release / 'application/control_groups.json')
                    if not controls:
                        controls = groups
                    elif controls != groups:
                        raise ValueError('control groups differ across processes')
                if len(parsed) != len(expected) or {r[0] for r in parsed} != expected:
                    raise ValueError('ordinary variant identities incomplete')
                local = []
                for variant, timing, raw in parsed:
                    timing = number(timing)
                    if timing is None or timing <= 0:
                        raise ValueError('missing/nonpositive ordinary time')
                    local.append(dict(task_id=task['id'], family=task['family'],
                                      process=task['process'], variant=variant, time_ms=timing,
                                      raw=raw, source=str(path)))
                samples.extend(local)
            elif task['kind'] == 'ncu':
                path = release / 'ncu.csv'
                provenance[str(path)] = sha(path)
                values = parse_ncu(path, data['metrics'])
                ids = {v['kernel_id'] for v in values}
                entry['kernel_count'] = len(ids)
                if len(ids) != task['expected_kernels']:
                    raise ValueError('kernel count mismatch')
                for value in values:
                    value['task_id'] = task['id']
                ncu.extend(values)
                entry['unavailable_metric_count'] = sum(v['value'] is None for v in values)
                entry['unavailable_metrics'] = [v for v in values if v['value'] is None]
                entry['missing_metric_authority'] = 'Recomputed from ncu.csv; capture_scope.json legacy unavailable_metrics parser is invalid for wide CSV.'
                for metric in data['metrics']:
                    selected = [v for v in values if v['metric'] == metric]
                    units = {v['unit'] for v in selected}
                    if (metric.endswith('.sum') and len(units) == 1 and
                            next(iter(units), None) in {'byte', 'sector', 'request', 'inst', 'wavefront', 'conflict'}):
                        valid = len(selected) == len(ids) and all(v['value'] is not None for v in selected)
                        totals.append(dict(task_id=task['id'], metric=metric, unit=selected[0]['unit'],
                                           value=sum(v['value'] for v in selected) if valid else None,
                                           status='complete' if valid else 'missing'))
            else:
                if not (release / 'capture.nsys-rep').is_file():
                    raise ValueError('NSYS report missing')
                report_path = release / 'capture.nsys-rep'
                entry['nsys_report'] = str(report_path)
                provenance[str(report_path)] = sha(report_path)
                if nsys_output is not None:
                    analysis = analyze_nsys(report_path, task, nsys_output / task['id'])
                    nsys.append(analysis)
                    if analysis['status'] == 'invalid':
                        raise ValueError('; '.join(analysis['issues']))
            entry['status'] = 'complete'
        except (OSError, ValueError, KeyError, StopIteration, sqlite3.Error, subprocess.TimeoutExpired) as error:
            entry.update(status='invalid', error=str(error))
    aggregates = []
    for family, expected_count in [('subgraph', 4), ('factor', 8)]:
        variants = sorted({r['variant'] for r in samples if r['family'] == family})
        for variant in variants:
            rows = [r for r in samples if r['family'] == family and r['variant'] == variant]
            times = [r['time_ms'] for r in rows]
            aggregates.append(dict(family=family, variant=variant, processes=len(times), expected_processes=3,
                                   status='complete' if len(times) == 3 else 'partial',
                                   median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times)))
    index = {(r['family'], r['variant']): r for r in aggregates}
    pairs = [('subgraph', f'{core}/stagewise-hbm', f'{core}/resident-prefix-2')
             for core in ('native', 'cufftdx-thread')]
    pairs += [('factor', group['baseline'], member) for group in controls for member in group['differences']]
    ratios = []
    for family, baseline, variant in pairs:
        a, b = index.get((family, baseline)), index.get((family, variant))
        complete = a is not None and b is not None and a['status'] == b['status'] == 'complete'
        ratios.append(dict(family=family, baseline=baseline, variant=variant,
                           status='complete' if complete else 'missing',
                           baseline_over_variant=a['median_ms'] / b['median_ms'] if complete else None))
    return dict(schema='cubutterfly-representative-profile-summary-v1',
                status='complete' if all(t['status'] == 'complete' for t in tasks) else 'partial',
                campaign=str(campaign), campaign_sha256=digest, campaign_metadata=data,
                provenance_sha256=provenance, tasks=tasks, ordinary_samples=samples,
                ordinary_aggregates=aggregates, matched_ratios=ratios,
                ncu_kernel_metrics=ncu, ncu_absolute_totals=totals, nsys_diagnostics=nsys,
                nsys_analysis_requested=nsys_output is not None,
                policy='No numerical acceptance thresholds; raw errors retained. Profiler diagnostics never ordinary timing. No averages of kernel rates or percentages. NSYS analysis optional and strictly diagnostic.')


def apply_nsys_repair(report, campaign, output):
    """Replace only the explicitly superseded timeline captures, never timings."""
    repair = summarize(campaign, output)
    metadata = repair['campaign_metadata']
    identity = metadata.get('capture_repair', {})
    if identity.get('parent_campaign_sha256') != report['campaign_sha256']:
        raise ValueError('NSYS repair has a different parent campaign')
    if metadata['gpu_uuid'] != report['campaign_metadata']['gpu_uuid']:
        raise ValueError('NSYS repair GPU mismatch')
    for filename, digest in report['campaign_metadata']['file_hashes'].items():
        if filename.endswith('/scripts/run_representative_profile_campaign.py'):
            continue  # The explicit node-tracing correction changes the runner.
        if metadata['file_hashes'].get(filename) != digest:
            raise ValueError('NSYS repair changed measured inputs: ' + filename)
    expected = {t['id']: t for t in report['campaign_metadata']['tasks']
                if t['family'] == 'factor' and t['kind'] == 'nsys'}
    if {t['id']: t for t in metadata['tasks']} != expected or len(expected) != 8:
        raise ValueError('NSYS repair changed the eight-case scope')
    replacement = {t['id']: t for t in repair['tasks']}
    for task in report['tasks']:
        if task['id'] in replacement:
            old = dict(task)
            task.clear()
            task.update(replacement[old['id']])
            task['superseded_capture'] = old
    report['superseded_nsys_diagnostics'] = [d for d in report['nsys_diagnostics']
                                             if d['task_id'] in replacement]
    report['nsys_diagnostics'] = [d for d in report['nsys_diagnostics']
                                 if d['task_id'] not in replacement] + repair['nsys_diagnostics']
    report['capture_repair'] = dict(campaign=str(campaign),
                                    campaign_sha256=repair['campaign_sha256'], **identity)
    report['provenance_sha256'].update(repair['provenance_sha256'])
    report['status'] = 'complete' if all(t['status'] == 'complete' for t in report['tasks']) else 'partial'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analyze-nsys', action='store_true', help='Export/reuse CPU SQLite and validate NSYS scope')
    parser.add_argument('--nsys-repair-campaign', type=Path,
                        help='Explicit same-build eight-case node-tracing repair; retains prior captures')
    args = parser.parse_args()
    report = summarize(args.campaign.resolve(), args.output.resolve() / 'nsys' if args.analyze_nsys else None)
    if args.nsys_repair_campaign:
        if not args.analyze_nsys:
            parser.error('--nsys-repair-campaign requires --analyze-nsys')
        apply_nsys_repair(report, args.nsys_repair_campaign.resolve(), args.output.resolve() / 'nsys_repair')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for key in ('tasks', 'ordinary_samples', 'ordinary_aggregates', 'matched_ratios', 'ncu_kernel_metrics', 'ncu_absolute_totals', 'nsys_diagnostics'):
        rows = report[key]
        with (args.output / (key + '.csv')).open('w', newline='') as handle:
            fields = sorted({k for row in rows for k in row})
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()} for row in rows)
    print(json.dumps(dict(status=report['status'], tasks=len(report['tasks']),
                          complete=sum(t['status'] == 'complete' for t in report['tasks']))))


if __name__ == '__main__':
    main()
