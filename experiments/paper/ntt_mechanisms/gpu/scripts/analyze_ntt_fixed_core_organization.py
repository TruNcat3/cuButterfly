#!/usr/bin/env python3
"""Post-hoc fixed-primitive organization statistics; no GPU access or raw edits."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def number(v):
    x = float(str(v).replace(',', '').strip())
    if not math.isfinite(x):
        raise ValueError('nonfinite value')
    return x


def profile(path, metrics, expected):
    out = {'status': 'pending', 'expected_nodes': expected, 'nodes': [], 'gaps': []}
    if not path.exists():
        return out
    out['csv_sha256'] = sha(path)
    try:
        raw = path.read_text()
        rows = list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])))
        nodes = {}
        long_format = bool(rows and 'Metric Name' in rows[0])
        for row in rows:
            if not row.get('ID', '').strip():  # NCU wide CSV has a units row.
                continue
            key = row['ID']
            node = nodes.setdefault(key, {'id': key, 'kernel': row.get('Kernel Name'), 'metrics': {}})
            entries = [(row.get('Metric Name'), row.get('Metric Value'))] if long_format else [(m, row.get(m)) for m in metrics]
            for m, v in entries:
                if m not in metrics:
                    continue
                if m in node['metrics']:
                    out['gaps'].append(f'duplicate metric: node {key}, {m}')
                try:
                    node['metrics'][m] = number(v)
                except (ValueError, TypeError):
                    out['gaps'].append(f'invalid metric: node {key}, {m}: {v!r}')
        if len(nodes) != expected:
            out['gaps'].append(f'node count {len(nodes)} != {expected}')
        for key, node in nodes.items():
            for m in metrics:
                if m not in node['metrics']:
                    out['gaps'].append(f'missing metric: node {key}, {m}')
        out['nodes'] = list(nodes.values())
        out['status'] = 'partial' if out['gaps'] else 'complete'
        # Only comparable absolute counts are summed, and only a complete capture.
        if out['status'] == 'complete':
            sums = {m: sum(n['metrics'][m] for n in nodes.values()) for m in (
                'dram__bytes_read.sum', 'dram__bytes_write.sum',
                'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum')}
            out['complete_call_absolute_counts'] = sums
            out['dram_read_write_bytes'] = sums['dram__bytes_read.sum'] + sums['dram__bytes_write.sum']
            out['l2_read_write_sectors'] = sums['lts__t_sectors_op_read.sum'] + sums['lts__t_sectors_op_write.sum']
    except (ValueError, KeyError, csv.Error) as e:
        out['status'] = 'partial'
        out['gaps'].append(str(e))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    campaign = a.campaign.resolve()
    data = json.loads(campaign.read_text())
    digest = sha(campaign)
    root = campaign.parent
    dest = a.output or root / 'analysis'
    dest.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'ntt-fixed-core-posthoc-v1', 'campaign': str(campaign),
              'campaign_sha256': digest, 'gpu_uuid': data['gpu_uuid'], 'request': data['request'],
              'scope': data['core_scope'], 'controlled_pair': data['controlled_pair'],
              'partitions': data['partitions'], 'file_hashes': data['file_hashes'],
              'estimator': 'Per process median of all seven CUDA-event samples; organization effect = median of three split process medians / median of three resident process medians. Paired process ratios retained independently.',
              'ordinary': [], 'profiles': [], 'pairs': [], 'gaps': []}
    for t in data['tasks']:
        release = root / 'jobs' / t['id'] / 'release'
        receipt = release / 'complete.json'
        if not receipt.exists():
            state = 'blocked' if (release / 'blocked.json').exists() else 'pending'
            item = {'task': t, 'status': state}
            (report['ordinary'] if t['kind'] == 'ordinary' else report['profiles']).append(item)
            continue
        item = {'task': t, 'status': 'invalid'}
        try:
            if json.loads(receipt.read_text())['campaign_sha256'] != digest:
                raise ValueError('receipt campaign hash mismatch')
            resultpath = release / 'result.json'
            result = json.loads(resultpath.read_text())
            item.update(result=result, result_sha256=sha(resultpath))
            for k in ('core', 'local_log', 'organization'):
                if result[k] != t[k]:
                    raise ValueError(f'result/task {k} mismatch')
            if result.get('correctness_exact_mismatches') != 0:
                raise ValueError('exact correctness missing/failed')
            if t['kind'] == 'ordinary':
                if result.get('profile_only'):
                    raise ValueError('profile run cannot supply timing')
                values = [number(v) for v in result['samples_us']]
                if len(values) != 7 or any(v <= 0 for v in values):
                    raise ValueError('expected seven positive finite samples')
                item.update(status='complete', process_median_us=median(values), raw_samples_us=values)
            else:
                item.update(profile(release / 'ncu.csv', data['metrics'], 2 if t['organization'] == 'resident-prefix' else 3))
        except (ValueError, KeyError, OSError) as e:
            item.update(status='invalid', reason=str(e))
        (report['ordinary'] if t['kind'] == 'ordinary' else report['profiles']).append(item)
    paired = []
    for core in ('native', 'gpuntt', 'phantom'):
        for local in (8, 10):
            group = [r for r in report['ordinary'] if r['task']['core'] == core and r['task']['local_log'] == local]
            orgs = {o: {r['task']['process']: r for r in group if r['task']['organization'] == o and r['status'] == 'complete'} for o in ('split-prefix', 'resident-prefix')}
            pair = {'core': core, 'local_log': local, 'local_points': 1 << local, 'status': 'pending', 'process_pairs': []}
            for process in range(3):
                split = orgs['split-prefix'].get(process)
                resident = orgs['resident-prefix'].get(process)
                if split and resident:
                    row = {'core': core, 'local_log': local, 'process': process,
                           'split_median_us': split['process_median_us'], 'resident_median_us': resident['process_median_us'],
                           'speedup': split['process_median_us'] / resident['process_median_us']}
                    paired.append(row)
                    pair['process_pairs'].append(row)
            if len(pair['process_pairs']) == 3:
                splitmed = median(r['split_median_us'] for r in pair['process_pairs'])
                resmed = median(r['resident_median_us'] for r in pair['process_pairs'])
                pair.update(status='complete', split_median_us=splitmed, resident_median_us=resmed, speedup_ratio_of_medians=splitmed / resmed)
            report['pairs'].append(pair)
    report['status'] = 'complete' if all(x['status'] == 'complete' for x in report['ordinary'] + report['profiles']) else 'pending-or-partial'
    (dest / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (dest / 'paired.csv').open('w') as f:
        fields = ['core', 'local_log', 'process', 'split_median_us', 'resident_median_us', 'speedup']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(paired)
    lines = ['# A100 fixed-primitive organization control', '', f'Status: {report["status"]}.', '',
             'Fixed identity is the named radix-two arithmetic primitive, including the same canonical wrapper within each pair. This is not a vendor whole-kernel or whole-library comparison. Grouping changes tile size, shared storage, grid and materialized boundaries together.', '',
             '| Core | Prefix points | Split us | Resident us | Speedup |', '|---|---:|---:|---:|---:|']
    for r in report['pairs']:
        if r['status'] == 'complete':
            lines.append(f'| {r["core"]} | {r["local_points"]} | {r["split_median_us"]:.3f} | {r["resident_median_us"]:.3f} | {r["speedup_ratio_of_medians"]:.4f}x |')
        else:
            lines.append(f'| {r["core"]} | {r["local_points"]} | pending | pending | pending |')
    lines += ['', 'Ordinary timing uses all raw samples and three independent process medians. NCU timing is excluded. Complete traffic sums only absolute DRAM bytes/L2 sectors across one validated capture; registers, shared storage, active warps and stalls remain per-node observations. Cache and clocks are uncontrolled.', '', '## Capture status', '']
    for r in report['profiles']:
        lines.append(f'- {r["task"]["id"]}: {r["status"]}' + (': ' + '; '.join(r.get('gaps', [])) if r.get('gaps') else ''))
    (dest / 'README.md').write_text('\n'.join(lines) + '\n')
    print(dest / 'analysis.json')
    print(report['status'])

if __name__ == '__main__':
    main()
