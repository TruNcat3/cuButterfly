#!/usr/bin/env python3
"""Verify portable Full66 source checkpoints and render per-request evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/snapshots/pro6000_full66_20261118'
REPORT = ROOT / 'experiments/reports/PRO6000_FULL66_20261118.md'
GPU = 'GPU-a257f4db-8c62-f26b-9c08-a4045615a140'
OPS = ('FFT', 'NTT', 'FWHT', 'subset-zeta', 'superset-zeta', 'structured-2x2')
STORAGE = ('FP16', 'BF16', 'FP32', 'FP64', 'complex-FP16', 'complex-BF16', 'complex-FP32', 'complex-FP64', 'uint32', 'uint64')
BASELINES = ('cufft', 'vkfft', 'daofht', 'gpu-ntt')
GOOD = {'pass', 'accuracy-review-required'}
GAPS = {'unsupported', 'missing-lowering', 'runner-gap', 'validation-unavailable', 'resource-limit', 'missing-baseline'}
ERRORS = {'correctness-failure', 'runtime-error', 'runner-error', 'baseline-error'}
RANK_ENV = {'CUBUTTERFLY_STATIC_RANKER', 'CUBUTTERFLY_STATIC_RANKER_SHA256',
            'CUBUTTERFLY_STATIC_RANK_WORKERS', 'CUBUTTERFLY_STATIC_RANK_CACHE',
            'CUBUTTERFLY_STATIC_RANK_AUDIT_DIR'}
IDENTITY_DIFFERENCES = {'manifest_sha256', 'preparation', 'preparation_config_sha256',
                        'preparation_source_manifest_sha256', 'runtime_environment'}
PROTOCOLS = {'20 complete calls per CUDA graph; 7 replays; 10 warmup',
             '7 single event calls; reset enqueued before start event; 10 warmup'}


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unpack(root, record):
    path = Path(record['path'])
    assert not path.is_absolute() and '..' not in path.parts
    raw = (Path(root) / path).read_bytes()
    assert sha(raw) == record['sha256'], str(path)
    raw = gzip.decompress(raw)
    assert sha(raw) == record['uncompressed_sha256'], str(path)
    if record.get('encoding','json')=='utf8':
        return raw.decode('utf8')
    assert record.get('encoding','json')=='json'
    return json.loads(raw)


def timing(value):
    timing = value.get('timing')
    if timing is None:
        return None
    samples = timing['samples_ms']
    assert len(samples) == 7 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples)
    assert timing['median_ms'] == statistics.median(samples)
    assert timing['protocol'] in PROTOCOLS
    return timing['median_ms'] * 1000


def exact(value):
    return value.get('correctness', {}).get('exact') is True and value['correctness'].get('mismatches') == 0


def known_negative(row):
    s = row['descriptor']['semantics']
    bad = [b for b in row.get('baselines', []) if b['status'] in ERRORS]
    replay = row.get('mapping_replay_correctness', {})
    return (row['status'] == 'correctness-failure' and s['operator'] == 1 and s['storage'] == 9
            and s['modulus'] == 576460756061519873 and s['inverse'] == 0
            and s['input_orders'] == s['output_orders'] == [0]
            and (s['shape'], s['batch']) in (([4096], 1), ([65536], 1), ([65536], 4))
            and exact(row) and replay.get('exact') is True and replay.get('mismatches') == 0
            and len(bad) == 1 and bad[0]['name'] == 'gpu-ntt' and bad[0]['status'] == 'correctness-failure'
            and bad[0].get('error', '').startswith('integer mismatch at '))


def compatible(first, current):
    assert {k: v for k, v in first.items() if k not in IDENTITY_DIFFERENCES} == {
        k: v for k, v in current.items() if k not in IDENTITY_DIFFERENCES}, 'Scientific runtime identities differ'
    assert {k: v for k, v in first['runtime_environment'].items() if k not in RANK_ENV} == {
        k: v for k, v in current['runtime_environment'].items() if k not in RANK_ENV}, 'Non-ranking runtime environment differs'


def verify(root, allow_partial=False):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['schema'] == 'cubutterfly-pro6000-full66-publication-v1'
    assert manifest['gpu_uuid'] == GPU
    assert manifest['requested_baselines'] == list(BASELINES)
    population = unpack(root, manifest['population'])
    assert population['manifest_sha256'] == digest({k: v for k, v in population.items() if k != 'manifest_sha256'})
    cases = {c['id']: c for c in population['cases']}
    assert len(cases) == len(population['cases']) == 66
    assert {c['descriptor']['semantics']['operator'] for c in cases.values()} == set(range(6))
    for identifier, case in cases.items():
        assert digest(case['descriptor']['semantics']) == identifier
    index = unpack(root, manifest['observations'])
    assert index['parent_manifest_sha256'] == manifest['population']['uncompressed_sha256']
    observed_ids = [item['id'] for item in index['rows']]
    assert len(observed_ids) == len(set(observed_ids)) == index['observed_cases']
    assert len(index['remaining']) == len(set(index['remaining']))
    assert set(observed_ids).isdisjoint(index['remaining'])
    assert set(observed_ids) | set(index['remaining']) == set(cases)
    assert index['declared_cases'] == 66
    if not allow_partial:
        assert len(observed_ids) == 66 and not index['remaining'], 'Full66 population still incomplete'
        assert manifest.get('publication_gate')!='partial-preview','Explicit partial preview is not a complete publication'
        if manifest.get('publication_gate')=='complete-with-array-replay-audit':
            assert manifest.get('numerical_array_audit'),'Final CPU array replay is missing'
    sources = {}
    first = None
    for source in manifest['sources']:
        assert source['original_path'] not in sources
        checkpoint = unpack(root, source['checkpoint'])
        assert source['checkpoint']['uncompressed_sha256'] == source['original_sha256']
        assert checkpoint['schema'] == 'cubutterfly-semantic-results-v1'
        assert checkpoint['rows_sha256'] == digest(checkpoint['rows'])
        identity = checkpoint['identity']
        assert identity['gpu_uuid'] == manifest['gpu_uuid'] and identity['library_sha256'] == manifest['library_sha256']
        assert identity['gpu_name'] == 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
        assert identity['compute_capability'] == '12.0' and identity['gpu_total_memory'] == 101973950464
        assert identity['compile_mode'] == 'research' and identity['timing'] is True
        env=identity['runtime_environment']
        if env.get('CUBUTTERFLY_STATIC_RANKER'):
            if manifest.get('cpu_ranker_source'):
                helper_sha=manifest['cpu_ranker_source']['uncompressed_sha256']
                unpack(root,manifest['cpu_ranker_source'])
            else:
                helper_sha=sha((ROOT/'experiments/tools/pro6000_parallel_static_ranker.py').read_bytes())
            assert env['CUBUTTERFLY_STATIC_RANKER_SHA256']==helper_sha,'CPU helper differs from frozen source'
        if first is not None:
            compatible(first, identity)
        else:
            first = identity
        source_manifest = unpack(root, source['population'])
        assert digest(source_manifest) == identity['manifest_sha256']
        source_cases = {c['id']: c for c in source_manifest['cases']}
        records = {}
        assert len(checkpoint['rows']) == len(source['rows'])
        for entry in checkpoint['rows']:
            identifier = entry['id']
            assert identifier not in records and entry['path'] == f'rows/{identifier}.json'
            envelope = unpack(root, source['rows'][identifier])
            row = envelope['row']
            assert entry['sha256'] == envelope['sha256'] == digest(row)
            assert row['id'] == identifier and row['status'] == entry['status']
            assert row['descriptor'] == source_cases[identifier]['descriptor'] == cases[identifier]['descriptor']
            records[identifier] = row
        sources[source['original_path']] = (source, checkpoint, records)
    rows = []
    for item in index['rows']:
        source, checkpoint, records = sources[item['results']]
        assert item['results_sha256'] == source['original_sha256']
        row = records[item['id']]
        assert item['row_sha256'] == digest(row) and bool(item['negative_result']) == known_negative(row)
        s = row['descriptor']['semantics']
        plan = row.get('plan_descriptor')
        if plan:
            assert plan['semantics'] == s
            assert plan['selection']['algorithm_policy'] == item['selection_policy'] == 1
        else:
            assert row['status'] in GAPS and item['selection_policy'] is None
        names = [b['name'] for b in row.get('baselines', [])]
        assert len(names) == len(set(names))
        if row['status'] in GAPS:
            assert set(names) <= set(BASELINES)
        else:
            assert names == list(BASELINES)
        assert row['status'] in GOOD | GAPS or known_negative(row)
        if row['status'] not in GAPS:
            assert timing(row) is not None and row.get('correctness') and row.get('mapping_replay_correctness')
            if s['storage'] in (8, 9):
                assert exact(row) and row['mapping_replay_correctness'].get('exact') is True and row['mapping_replay_correctness'].get('mismatches') == 0
        for baseline in row.get('baselines', []):
            value = timing(baseline)
            assert baseline['status'] not in ERRORS or known_negative(row)
            if value is not None:
                assert baseline['status'] in GOOD and baseline.get('correctness')
                assert baseline['timing']['protocol'] == row['timing']['protocol']
                if s['storage'] in (8, 9):
                    assert exact(baseline)
        rows.append((row, item, checkpoint['identity']))
    assert sum(known_negative(row) for row, _, _ in rows) == index['negative_results']
    if manifest.get('numerical_array_audit'):
        array_audit=unpack(root,manifest['numerical_array_audit'])
        audit_cases={r['case_id']:r for r in array_audit['cases']}
        assert len(audit_cases)==len(array_audit['cases'])
        assert set(audit_cases)<=set(observed_ids),'Array audit population differs'
        if array_audit['status']=='complete-recorded-array-metrics-replayed':
            assert set(audit_cases)==set(observed_ids) and len(audit_cases)==66,'Array audit population differs'
        assert array_audit['primary_check_full_source_sha256']==first['sources']['run_semantic_campaign.py']
        assert array_audit['numpy']==first['numpy']
        for row,item,_ in rows:
            if row['id'] not in audit_cases:continue
            evidence=audit_cases[row['id']]
            assert evidence['row_sha256']==item['row_sha256']
            implementations={'cubutterfly':row,**{b['name']:b for b in row.get('baselines',[])}}
            assert {r['name'] for r in evidence['implementations']}==set(implementations)
            for result in evidence['implementations']:
                implementation=implementations[result['name']]
                if result['classification']=='no-measurement-array':
                    assert not implementation.get('error_artifact') and not implementation.get('timing')
                    continue
                assert result['array_sha256']==implementation['error_artifact']['sha256']
                if result['classification']=='recorded-metrics-reproduced':
                    assert result['metrics_sha256']==digest(implementation['correctness'])
                else:
                    assert result['classification']=='recorded-integer-failure-reproduced'
                    assert result['name']=='gpu-ntt' and known_negative(row)
                    assert result['error']==implementation['error'] and result['mismatches']>0
    # Calibration warning/coverage is evidence, never upgraded for publication.
    calibration = unpack(root, manifest['calibration_manifest'])
    assert digest(calibration) == first['calibration_sha256']
    assert calibration['cost_model_status'] == 'calibrated-local-warning'
    assert not calibration['service_gaps']
    assert calibration['selector_replay']['status'] == 'passed'
    for evidence in manifest.get('supporting_evidence', []):
        unpack(root, evidence['record'])
    return manifest, population, rows, index


def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render(root, allow_partial=False):
    manifest, population, observations, index = verify(root, allow_partial)
    cases, pairs = [], []
    for row, provenance, identity in observations:
        s = row['descriptor']['semantics']
        label = f"{OPS[s['operator']]} {STORAGE[s['storage']]} {'x'.join(map(str, s['shape']))} B={s['batch']} {'inverse' if s['inverse'] else 'forward'}"
        own = timing(row)
        cases.append(dict(case_id=row['id'], workload=label, status=row['status'], own_us=own,
                          plan_build_ms=row.get('plan_build_ms'),workspace_bytes=row.get('workspace_bytes'),
                          input_bytes=row.get('input_bytes'),output_bytes=row.get('output_bytes'),
                          selection_policy=provenance['selection_policy'], selection_scope=row.get('selection_scope'),
                          semantics=json.dumps(s, sort_keys=True), mapping=json.dumps(row.get('mapping'), sort_keys=True),
                          correctness=json.dumps(row.get('correctness'), sort_keys=True),
                          mapping_replay_correctness=json.dumps(row.get('mapping_replay_correctness'), sort_keys=True),
                          source=provenance['results'], source_row_sha256=provenance['row_sha256'],
                          source_identity_sha256=digest(identity)))
        for name in BASELINES:
            baseline = next((b for b in row.get('baselines', []) if b['name'] == name),
                            dict(name=name, status='not-reached-after-own-gap',
                                 reason=row.get('error', row.get('reason', 'own implementation unavailable'))))
            value = timing(baseline)
            qualified = row['status'] not in GAPS and own is not None and value is not None and baseline['status'] in GOOD
            ratio = value / own if qualified else None
            pairs.append(dict(case_id=row['id'], workload=label, baseline=baseline['name'],
                              status=baseline['status'], own_us=own, baseline_us=value, baseline_over_own=ratio,
                              ratio_role=('integer exact with exact replay' if s['storage'] in (8, 9)
                                          else 'diagnostic floating error review') if qualified else 'excluded',
                              reason=baseline.get('error', baseline.get('reason', '')),
                              correctness=json.dumps(baseline.get('correctness'), sort_keys=True),
                              source=provenance['results']))
    counts = Counter(row['descriptor']['semantics']['operator'] for row, _, _ in observations)
    declared = Counter(c['descriptor']['semantics']['operator'] for c in population['cases'])
    text = ['# PRO6000 Full66 per-request comparisons', '',
            f"Observed **{len(cases)}/66** requests; remaining {len(index['remaining'])}; retained negative requests {index['negative_results']}.", '',
            'These are public MEASURE (policy 1) comparisons using target calibration and a search budget of 16, separate from DEFAULT replay and fixed A100 mapping transfer. Planning, compilation, allocation and CPU checks are outside the original complete-call timing. The original successful plan-construction attempt is retained as plan_build_ms in cases.csv, together with workspace and input/output sizes; this field does not sum earlier compilation/queue/retry waits and is not total cold-start wall time. Each source keeps its measured identity; CPU parallel ranking was introduced by an explicit checkpoint transition after full request/score/order equivalence testing.', '',
            'The original formal campaign retains its recorded interrupted or failed state. Full population observation does not imply full numerical acceptance or a globally validated model. The model remains calibrated-local-warning; floating ratios require error review, integer ratios require exact outputs and exact mapping replay. Failed or missing comparators have no ratio. No aggregate across source cohorts is reported.', '',
            '| Operator | Declared | Observed |', '|---|---:|---:|']
    text += [f'| {name} | {declared[i]} | {counts[i]} |' for i, name in enumerate(OPS)]
    text += ['', '| Request | Status | cuButterfly us |', '|---|---|---:|']
    text += [f"| {r['workload']} | {r['status']} | {r['own_us']:.4f} |" if r['own_us'] is not None else f"| {r['workload']} | {r['status']} | missing |" for r in cases]
    text += ['', '| Request | Baseline | Status | Baseline us | Baseline/cuButterfly |', '|---|---|---|---:|---:|']
    for row in pairs:
        if row['status'] == 'not-applicable':
            continue
        value = 'missing' if row['baseline_us'] is None else f"{row['baseline_us']:.4f}"
        ratio = 'excluded' if row['baseline_over_own'] is None else f"{row['baseline_over_own']:.4f}x"
        text.append(f"| {row['workload']} | {row['baseline']} | {row['status']} | {value} | {ratio} |")
    text += ['', 'Raw source checkpoints, row envelopes, full numerical error distributions, actual mappings, calibration limitations and source identities are losslessly bundled. Error-array paths/hashes remain original provenance; the local export audit verifies those files before freezing. This checkpoint format has no per-row GPU guard receipt: exclusivity evidence is the unchanged recorded runner and its protocol, not a newly invented guard field.', '',
             'The CPU array audit recomputes recorded numerical metrics and stored error arrays from the original lossless actual/reference outputs. It preserves the original floating review and integer failures; it does not generate a new oracle or rerun GPU mapping replay.', '',
             'The [CPU ranking helper](../tools/pro6000_parallel_static_ranker.py) matches the measured helper bytes. The snapshot also retains its source receipt and the measured checkout’s workflow/build patch, including the two untracked score-cache helpers. The patch identifies the original measured source state; rebuilding on another host requires a compatible toolchain and fresh target validation.', '',
             '[Per-request semantics, mappings and errors](../snapshots/pro6000_full66_20261118/cases.csv), [all comparator records](../snapshots/pro6000_full66_20261118/baselines.csv), and [source manifest](../snapshots/pro6000_full66_20261118/manifest.json).', '',
             '```bash', 'python3 experiments/tools/report_pro6000_full66.py --check', '```', '']
    return csv_text(cases), csv_text(pairs), '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cases, pairs, text = render(args.snapshot, args.allow_partial)
    outputs = {args.snapshot / 'cases.csv': cases, args.snapshot / 'baselines.csv': pairs, REPORT: text}
    for path, value in outputs.items():
        if args.write:
            path.write_text(value)
        else:
            assert path.read_text() == value, 'Derived report/CSV differs: ' + str(path)
    print('Verified Full66 source identities, requests, timings, negative results and calibration scope; CPU only')


if __name__ == '__main__':
    main()
