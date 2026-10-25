#!/usr/bin/env python3
"""Audit normal timing and per-call profiler denominators without hit averages."""
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_controlled_experiment as controlled
OUT = ROOT / 'results/paper_targeted_20261012/mechanism'


def csv_rows(text, header):
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(header))
    return list(csv.DictReader(io.StringIO('\n'.join(lines[index:]))))


def main():
    journal = json.loads((OUT / 'timing_attempt3.json').read_text())
    timing, profiles = [], []
    for case in journal['cases']:
        variants = {item['id']: item for item in case['variants']}
        for comparison in case['comparisons']:
            baseline, treatment = [variants[comparison[key]] for key in ['baseline', 'treatment']]
            diff = controlled._mapping_diff(baseline['preflight']['resolved_mapping'], treatment['preflight']['resolved_mapping'], include_derived=False)
            derivative = None
            if case['workload']['operator'] == 'fwht' and diff == {'stage_partition', 'boundaries'}:
                a, b = [v['preflight']['resolved_mapping']['boundaries'] for v in [baseline, treatment]]
                partitions = [v['preflight']['resolved_mapping']['stage_partition'] for v in [baseline, treatment]]
                if all(len(boundaries) == len(partition)-1 for boundaries, partition in zip([a,b],partitions)) and all(item == dict(layout='direct-strided', residency='global-scratch', twiddle='table') for item in a+b):
                    derivative = 'Default global-scratch boundary descriptor count equals physical group count minus one in both variants; stage_partition changes count, with no boundary-policy substitution.'
                    diff.remove('boundaries')
            accepted = diff == set(comparison['changed_axes']) and all(v['preflight']['status'] == 'accepted' and len(v['trials']) == 3 and all(t['performance_accepted'] for t in v['trials']) for v in [baseline, treatment])
            medians = [statistics.median(float(t['kernel_ms']) for t in v['trials']) for v in [baseline, treatment]]
            timing.append(dict(case=case['id'], baseline=baseline['id'], treatment=treatment['id'],
                accepted=accepted, original_runner_status=comparison['status'], actual_changed_axes=sorted(diff),
                boundary_derivative_audit=derivative, baseline_us=medians[0]*1000, treatment_us=medians[1]*1000,
                baseline_over_treatment=medians[0]/medians[1], independent_process_trials=3,
                correctness='fullbatch oracle preflight accepted for both variants'))
        for variant in variants.values():
            directory = 'profiles_fft' if case['workload']['operator'] == 'fft' else 'profiles_workspace_fix' if case['id'] == 'fwht-fp32-n8-b1' else 'profiles'
            stem = OUT / directory / (case['id'] + '.' + variant['id'])
            metrics = {}
            rows = csv_rows(Path(str(stem)+'.ncu.csv').read_text(), '"ID"')
            ids = sorted(set(row['ID'] for row in rows))
            for row in rows:
                metrics.setdefault(row['Metric Name'], []).append(float(row['Metric Value'].replace(',', '')))
            sums = {key: sum(values) for key, values in metrics.items() if key.endswith('.sum')}
            report = Path(str(stem) + '.nsys.nsys-rep')
            result = subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv', str(report)], text=True, capture_output=True, check=True)
            Path(str(stem)+'.nsys.trace.csv').write_text(result.stdout)
            trace = csv_rows(result.stdout, 'Start (ns)')
            intervals = sorted((int(row['Start (ns)']), int(row['Start (ns)'])+int(row['Duration (ns)'])) for row in trace if row['Name'])
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            profiles.append(dict(case=case['id'], variant=variant['id'], source_directory=directory,
                logical_calls=1, warmup_calls=100, ncu_kernel_count=len(ids), nsys_kernel_count=len(intervals),
                per_call_counter_sums=sums, l2_read_write_bytes_from_32byte_sectors=32*(sums.get('lts__t_sectors_op_read.sum',0)+sums.get('lts__t_sectors_op_write.sum',0)),
                per_kernel_registers=metrics.get('launch__registers_per_thread'),
                per_kernel_shared_bytes=metrics.get('launch__shared_mem_per_block_allocated'),
                per_kernel_long_scoreboard_pct=metrics.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),
                nsys_kernel_union_us=sum(end-start for start,end in merged)/1000,
                nsys_first_kernel_to_last_kernel_us=(max(end for _,end in intervals)-min(start for start,_ in intervals))/1000,
                nsys_kernel_duration_sum_us=sum(end-start for start,end in intervals)/1000,
                profiler_duration_is_normal_latency=False,
                kernel_names=[row['Name'] for row in trace if row['Name']],
                percentage_aggregation='No aggregate percentage; raw per-kernel scoreboard only. No L2 hit percentage requested.'))
    factorial = next(c for c in journal['cases'] if c['workload']['operator']=='fft')
    cells = {v['id']: statistics.median(float(t['kernel_ms']) for t in v['trials']) for v in factorial['variants']}
    interaction = (cells['native-serial']/cells['native-overlap'])/(cells['imported-serial']/cells['imported-overlap'])
    controlled._atomic_write(OUT/'summary.json', dict(schema='cubutterfly-mechanism-cohort-summary-v1',
        timing_source='timing_attempt3.json', binary_identity=journal['identity']['binaries'],
        frozen_library_sha256='24486975585fa2ab6399a0cfeae4b9449c398490cb76c5ea5a73a3560abe8c66',
        timing=timing, profiles=profiles, fft_ratio_of_dataflow_speedups=interaction,
        interpretation='Internal fixed-core organization controls only; stagewise is deliberately unfused and underfills tiny stage CTAs. Ratios are not external library speedups or tuned-optimum comparisons.'))
    print(json.dumps({'timing_comparisons':len(timing), 'profile_variants':len(profiles), 'all_accepted':all(t['accepted'] for t in timing)}))


if __name__ == '__main__':
    main()
