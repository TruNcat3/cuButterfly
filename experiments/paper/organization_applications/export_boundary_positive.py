#!/usr/bin/env python3
"""Export the frozen FFT local-bank positive control without GPU or raw-path access."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'inputs/boundary_positive.json'
STEM = ROOT / 'output/boundary_positive_summary_20261119'
CONFLICT = 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{}.sum'
RESOURCE_KEYS = ('launch__registers_per_thread', 'launch__shared_mem_per_block_dynamic')


def export(output_dir):
    global STEM
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    STEM = output_dir / "boundary_positive_summary_20261119"
    payload = SOURCE.read_bytes()
    data = json.loads(payload)
    timing_identity = data['identities']['bank_timing']
    profile_identity = data['identities']['bank_profile']
    for key in ('binary_sha256', 'library_sha256', 'source_sha256', 'config_sha256'):
        assert timing_identity[key] == profile_identity[key], key
    requests = []
    flat = []
    for batch in (4, 16):
        runs = [r for r in data['bank_timing'] if r['batch'] == batch]
        assert len(runs) == 3 and len({r['run'] for r in runs}) == 3
        process_records = []
        for run in runs:
            assert run['logN'] == 20 and run['precision'] == 'fp32' and run['operator'] == 'fft'
            assert run['byte_identical_outputs'] and run['strict_control_valid']
            assert run['global_boundary_layout_fixed'] and run['capture_calls'] == 20
            assert run['trials'] == 7 and run['warmup_graph_launches'] == 10
            cases = {c['mode']: c for c in run['cases']}
            assert set(cases) == {'linear', 'xor'}
            for case in cases.values():
                assert len(case['trials_ms']) == 7 and case['graph_nodes'] == 60
                assert all(math.isfinite(v) and v > 0 for v in case['trials_ms'])
                assert case['median_ms'] == statistics.median(case['trials_ms'])
                assert case['verified_batches'] == batch and case['cpu_error']['nonfinite'] == 0
                assert case['linear_output_error']['max_abs'] == 0
            lm, xm = (cases[k]['median_ms'] for k in ('linear', 'xor'))
            process_records.append({'run': run['run'], 'linear_ms': lm, 'xor_ms': xm,
                                    'linear_over_xor': lm / xm,
                                    'cpu_errors': {k: cases[k]['cpu_error'] for k in cases},
                                    'byte_identical_outputs': run['byte_identical_outputs']})
        ratios = [r['linear_over_xor'] for r in process_records]
        reference = next(r for r in data['ratios'] if r['study'] == 'bank' and r['batch'] == batch)
        assert ratios == reference['process_ratios']
        assert statistics.median(ratios) == reference['median']
        assert min(ratios) == reference['min'] and max(ratios) == reference['max']
        profiles = {mode: data['profiles'][f'profile-n20-b{batch}-{mode}'] for mode in ('linear', 'xor')}
        resources = []
        for left, right in zip(profiles['linear']['kernels'], profiles['xor']['kernels']):
            assert left['role'] == right['role']
            assert left['actual_launch'] == right['actual_launch']
            assert all(left['metrics'][k] == right['metrics'][k] for k in RESOURCE_KEYS)
            resources.append({'role': left['role'], 'actual_launch': left['actual_launch'],
                              'registers_per_thread': left['metrics'][RESOURCE_KEYS[0]],
                              'dynamic_shared_bytes': left['metrics'][RESOURCE_KEYS[1]],
                              'linear_xor_equal': True})
        assert len(resources) == 3
        for profile in profiles.values():
            assert profile['correctness']['byte_identical_outputs']
            assert profile['correctness']['profile_complete_calls'] == 1
        conflicts = {}
        for op in ('ld', 'st'):
            counts = {mode: profiles[mode]['kernels'][0]['metrics'][CONFLICT.format(op)] for mode in profiles}
            assert 0 < counts['xor'] < counts['linear']
            conflicts[op] = {**counts, 'reduction_percent': 100 * (1 - counts['xor'] / counts['linear'])}
        row = {'logN': 20, 'batch': batch, 'processes': process_records,
               'linear_median_ms': statistics.median(r['linear_ms'] for r in process_records),
               'xor_median_ms': statistics.median(r['xor_ms'] for r in process_records),
               'paired_speedup_median': statistics.median(ratios),
               'paired_speedup_min': min(ratios), 'paired_speedup_max': max(ratios),
               'prefix_bank_conflicts': conflicts, 'ncu_resources': resources}
        requests.append(row)
        for record in process_records:
            flat.append({'logN': 20, 'batch': batch, 'run': record['run'],
                         'process_linear_ms': record['linear_ms'], 'process_xor_ms': record['xor_ms'],
                         'process_paired_speedup': record['linear_over_xor'],
                         **{k: row[k] for k in ('linear_median_ms', 'xor_median_ms', 'paired_speedup_median', 'paired_speedup_min', 'paired_speedup_max')},
                         **{f'prefix_{op}_{key}': value for op, counts in conflicts.items() for key, value in counts.items()}})
    output = {'schema': 'boundary-positive-summary-v1',
              'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(payload).hexdigest(),
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'embedded_raw_hashes_verified': False,
              'provenance_note': 'Sanitized numerical subset; original source hashes recorded separately in source_provenance.json. Embedded raw inputs are not bundled or revalidated.',
              'timing_identity': timing_identity, 'profile_identity': profile_identity,
              'scope': 'Independent FP32 FFT prefix shared-staging slot permutation control; fixed cuFFTDx Block arithmetic, 8+12 partition, EPT16/256 threads, global tiled-transpose boundary, no overlap.',
              'timing': 'Three fresh processes; seven event intervals per mode, each enclosing a graph of twenty complete calls (60 nodes); ten graph warmups per mode. Per-call time includes prefix, suffix and terminal transpose; capture, CPU reference and copies excluded. Modes rotate by trial; batch order reversed in process 1.',
              'statistics': 'Median/min/max of three within-process Linear/XOR ratios; range is not a confidence interval. Displayed latency medians are independently aggregated.',
              'correctness_policy': 'All batches checked against same-precision CPU reference; finite outputs and byte-identical Linear/XOR outputs required; CPU errors retained without introducing a new acceptance threshold.',
              'mapping_limit': 'Local complex-FP32 tile slot XOR is not a direct measurement of Algorithm 1 physical-bank single-word theorem. For 16 slots, index=16*element+(slot XOR (element & 15)); complex values span two four-byte banks. Nonzero residual conflicts and vector/concurrent access require empirical qualification.',
              'provenance_limit': 'Frozen identities hash the control source, executable and library; they do not provide a complete per-file library source snapshot. No online raw paths are accessed by this exporter.',
              'requests': requests}
    STEM.with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n')
    with STEM.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print('Exported two requests / six paired process records; numerical controls verified (sanitized subset).')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    export(parser.parse_args().output_dir)
