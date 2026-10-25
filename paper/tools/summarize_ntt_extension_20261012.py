#!/usr/bin/env python3
"""Audit all three NTT process trials and retain every paired win or loss."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_semantic_campaign as runner

BASE = ROOT / 'results/paper_targeted_20261012/ntt_extension'


def main():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    freeze = json.loads((BASE / 'freeze.json').read_text())
    by_id = {c['id']: [] for c in manifest['cases']}
    identities = []
    setup = {}
    for number in (1, 2, 3):
        directory = BASE / f'trial-{number}'
        report = json.loads((directory / 'results.json').read_text())
        assert report['status'] == 'complete' and not report['remaining_case_ids']
        identity = report['identity']
        assert identity['gpu_uuid'] == freeze['gpu_uuid']
        assert identity['library_sha256'] == freeze['library_sha256']
        assert identity['calibration_sha256'] == freeze['calibration_sha256']
        assert identity['protocol']['warmup_calls'] == 10
        assert identity['protocol']['graph_calls'] == 20
        assert identity['protocol']['replays'] == 7
        identities.append(identity)
        selected = None
        if number > 1:
            selected = json.loads((BASE / 'selected_mappings.json').read_text())
        setup[str(number)] = {'selection_source': 'public_MEASURE_512' if number == 1 else 'frozen_trial1',
                              'accepted_plan_build_ms': {}, 'initial_plan_build_total_ms': 0.0}
        for entry in report['rows']:
            document = json.loads((directory / entry['path']).read_text())
            row = document['row']
            setup[str(number)]['accepted_plan_build_ms'][row['id']] = row['plan_build_ms']
            setup[str(number)]['initial_plan_build_total_ms'] += row['plan_build_ms']
            assert document['sha256'] == runner.digest(row) == entry['sha256']
            baseline = next(b for b in row['baselines'] if b['name'] == 'gpu-ntt')
            if selected is not None:
                if row['mapping'] != selected['mappings'][row['id']]:
                    raise AssertionError('resolved mapping differs from frozen trial1 mapping')
            assert row['status'] == baseline['status'] == 'pass'
            for result in (row['correctness'], baseline['correctness']):
                assert result['exact'] and result['mismatches'] == 0
                assert result['elements'] == row['descriptor']['semantics']['batch'] * row['descriptor']['semantics']['shape'][0]
            cb = row['timing']['median_ms'] * 1000
            gpu = baseline['timing']['median_ms'] * 1000
            by_id[row['id']].append({'trial': number, 'cubutterfly_us': cb, 'gpu_ntt_us': gpu,
                                   'speedup': gpu / cb, 'exact': True,
                                   'algorithm': row['algorithm'], 'mapping': row['mapping']})
    rows = []
    for case in manifest['cases']:
        s = case['descriptor']['semantics']
        trials = by_id[case['id']]
        assert len(trials) == 3
        cb = statistics.median(t['cubutterfly_us'] for t in trials)
        gpu = statistics.median(t['gpu_ntt_us'] for t in trials)
        rows.append({'id': case['id'], 'group': 'original4' if s['batch'] == 1 else 'additional8',
                     'N': s['shape'][0], 'B': s['batch'], 'direction': 'inverse' if s['inverse'] else 'forward',
                     'cubutterfly_us': cb, 'gpu_ntt_us': gpu, 'speedup': gpu / cb, 'trials': trials})
    groups = {}
    for group in ('original4', 'additional8'):
        ratios = [r['speedup'] for r in rows if r['group'] == group]
        trial_gms = [math.exp(statistics.mean(math.log(r['trials'][i]['speedup']) for r in rows if r['group'] == group)) for i in range(3)]
        groups[group] = {'points': len(ratios), 'geometric_mean': math.exp(statistics.mean(map(math.log, ratios))),
                         'point_range': [min(ratios), max(ratios)], 'wins': sum(v > 1 for v in ratios),
                         'losses': sum(v < 1 for v in ratios), 'ties': sum(v == 1 for v in ratios),
                         'trial_geometric_means': trial_gms, 'trial_geometric_mean_range': [min(trial_gms), max(trial_gms)]}
    result = {'status': 'complete-exact', 'aggregation': 'per point median of three independent process medians, then GPU-NTT/cuButterfly; group GM across points',
              'baseline': 'GPU-NTT Merge only', 'groups': groups, 'rows': rows,
              'trial_identity_sha256': [runner.digest(i) for i in identities],
              'setup': setup, 'setup_scope': 'accepted initial plan construction; compilation, failed pending attempts, replay, allocation, CPU oracle excluded from GPU timing'}
    events = [json.loads(line) for path in (BASE / 'trial-1/compile_events').glob('*.jsonl')
              for line in path.read_text().splitlines() if line.strip()]
    request_ids = sorted({event['request_id'] for event in events if event.get('request_id')})
    durations = []
    status_counts = {}
    for request_id in request_ids:
        record = json.loads((ROOT / 'results/plan_batch_trace_20261010/shared_compile/records' / (request_id + '.json')).read_text())
        status_counts[record['status']] = status_counts.get(record['status'], 0) + 1
        if record.get('started_ns') and record.get('finished_ns'):
            durations.append((record['finished_ns'] - record['started_ns']) / 1e9)
    result['compile_setup'] = {'observed_trial1_unique_requests': len(request_ids), 'broker_status_counts': status_counts,
                               'sum_per_request_compile_seconds': sum(durations),
                               'duration_scope': 'sum of shared broker request intervals; parallel/overlapping, not wall time or marginal campaign cost',
                               'request_ids': request_ids, 'observed_event_count': len(events),
                               'trial2_trial3': 'no repeated search; frozen trial1 mappings; no new compile-event files'}
    (BASE / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
