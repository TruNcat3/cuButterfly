#!/usr/bin/env python3
"""Freeze audited application supplements; export only, never execute CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root=ROOT):
    audit_path = root / 'results/a100_e2e_closure_20261107/audit.json'
    fno_path = root / 'results/e2e_unified_20261013/fno_hotpath_summary.json'
    audit, fno = (json.loads(p.read_text()) for p in (audit_path, fno_path))
    if len(audit['results']) != 14 or len(audit['jobs']) != 9 or any(j['state'] != 'complete' or not all(j['report_hashes_match'].values())
                   or not j['queue_hash_matches'] or not j['preflight_hash_matches']
                   for j in audit['jobs']):
        raise ValueError('Expected complete nine-process/14-row audit')
    if not all(fno[field] for field in ('all_outputs_finite', 'model_state_unchanged',
            'graph_exact_vs_same_backend_eager', 'all_output_hashes_match_original_control',
            'all_cache_mapping_hashes_match_original_control')):
        raise ValueError('FNO numerical/lifecycle controls are incomplete')
    sources = {}
    def source(path):
        sources[str(path.relative_to(root))] = digest(path)
    source(audit_path)
    source(fno_path)
    rows = []
    for row in audit['results']:
        row = dict(row, cohort='nov2-application-closure', timing_mode='graph')
        source(root / row['source'])
        row['source_sha256'] = digest(root / row['source'])
        row['role'] = 'diagnostic' if 'diagnostic' in row['baseline'] else 'external'
        rows.append(row)
    for row in fno['results']:
        for mode in ('graph', 'eager'):
            rows.append(dict(cohort='historical-fno-hotpath', scenario=f"FNO {row['size']}^2",
                size=row['size'], batch=fno['batch'], baseline='Torch', timing_mode=mode,
                baseline_ms=row['wall_median_ms']['torch_' + mode],
                own_ms=row['wall_median_ms']['cubutterfly_' + mode],
                speedup=row['speedup_torch_over_cubutterfly'][mode],
                trial_speedups=[r[mode] for r in row['paired_trial_speedups']],
                independent_processes=fno['independent_processes_per_size'],
                library_sha256=fno['library_sha256'], gpu_uuid=fno['gpu_uuid'],
                timing_scope=fno['timing_scope'], aggregation=fno['aggregation'],
                source=str(fno_path.relative_to(root)), source_sha256=digest(fno_path), role='external'))
    return dict(schema='paper-application-closeout-v1', sources=sources, records=rows,
        note='Build-prior measurements, not installed-Nov5 timings. No pooling with snapshot or legacy records. Diagnostic same-padding filter comparisons are separate. FNO Graph and eager are separate contracts.',
        limitations=audit['limitations'])


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'paper-application-closeout-v1' or len(data['records']) != 18:
        raise ValueError('Expected frozen 14 closure and four FNO mode rows')
    for row in data['records']:
        if row['baseline_ms'] <= 0 or row['own_ms'] <= 0:
            raise ValueError('Nonpositive application latency')
        if abs(row['baseline_ms'] / row['own_ms'] - row['speedup']) > 1e-8:
            raise ValueError('Application ratio differs from recorded latencies')
        if data['sources'].get(row['source']) != row['source_sha256']:
            raise ValueError('Missing frozen source hash')
    return data



def table(data):
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Cohort / request & Mode & $B$ & Own ms & Base ms & Ratio & Build\\\midrule"]
    for row in data['records']:
        if row['cohort'] == 'historical-fno-hotpath':
            label = 'FNO ' + str(row['size'])
        elif 'Llama' in row['scenario']:
            label = 'QuaRot ' + row['job_id'].split('_')[-1]
        elif 'filter' in row['scenario']:
            label = row['scenario'].replace('fixed31 filter ', 'Filter ').replace('^2', '')
            label += ' diag.' if row['role'] == 'diagnostic' else ' CuPy'
            label += ' ' + row['job_id'].split('_')[-1]
        else:
            label = 'Bootstrap'
        lines.append(f"{label} & {row['timing_mode']} & {row['batch']} & "
                     f"{row['own_ms']:.4f} & {row['baseline_ms']:.4f} & "
                     f"{row['speedup']:.6f} & {row['library_sha256'][:6]}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return '\n'.join(lines) + '\n'


def cohort_table(data):
    groups = {}
    for row in data['records']:
        if row['timing_mode'] != 'graph' or row['role'] != 'external':
            continue
        name = (f"FNO {row['size']}" if 'size' in row else 'QuaRot' if 'Llama' in row['scenario']
                else row['scenario'].replace('fixed31 filter ', 'Filter ').replace('^2', '')
                if 'filter' in row['scenario'] else 'Bootstrap')
        groups.setdefault((row['cohort'], name, row['batch']), []).append(row)
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Scenario & Type & Baseline & Ratio\\\midrule"]
    for (cohort, name, batch), rows in groups.items():
        values = [row['speedup'] for row in rows]
        low, high = f"{min(values):.3f}", f"{max(values):.3f}"
        ratio = low if low == high else f"{low}--{high}"
        if name == 'QuaRot':
            kind, baseline = 'FWHT', 'QuaRot/Dao'
        elif name == 'Bootstrap':
            kind, baseline = 'NTT', 'HEonGPU'
        else:
            kind, baseline = 'FFT', 'Torch' if name.startswith('FNO') else 'CuPy'
        if name.startswith(('FNO ', 'Filter ')):
            title, size = name.split(' ')
            name = title + r" $" + size + r"^2$"
        lines.append(f"{name}, B{batch} & {kind} & {baseline} & {ratio}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return '\n'.join(lines) + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/generated/current_refresh/application_closeout.json')
    parser.add_argument('--fp16-acceptance', type=Path,
                        help='Optionally freeze the validated nine-request public cohort for results-free redraw')
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(), indent=2) + '\n')
    data = load(args.output)
    args.output.with_suffix(".tex").write_text(table(data))
    (args.output.parent / "application_cohorts.tex").write_text(cohort_table(data))
    if args.fp16_acceptance:
        from load_fp16_acceptance import load_fp16_acceptance
        frozen = load_fp16_acceptance(args.fp16_acceptance)
        (args.output.parent / 'fp16_comparison.json').write_text(json.dumps(frozen, indent=2) + '\n')


if __name__ == '__main__':
    main()
