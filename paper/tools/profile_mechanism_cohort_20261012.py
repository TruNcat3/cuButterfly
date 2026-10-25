#!/usr/bin/env python3
"""Finite NCU/NSYS capture of accepted operator control variants."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/tools'))
from run_mechanism_cohort_20261012 import OUT, environment, previous
sys.path.insert(0, str(ROOT / 'scripts'))
from run_comprehensive_suite import require_exclusive_gpu
import run_controlled_experiment as controlled
import os

METRICS = ['dram__bytes_read.sum', 'dram__bytes_write.sum', 'lts__t_sectors_op_read.sum',
           'lts__t_sectors_op_write.sum', 'launch__registers_per_thread',
           'launch__shared_mem_per_block_allocated',
           'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
           'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--output-dir', type=Path, default=OUT / 'profiles')
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    output = args.output_dir
    output.mkdir(exist_ok=True)
    os.environ.update(environment(output / 'events.jsonl'))
    records = []
    for case in journal['cases']:
        if args.case and case['id'] != args.case:
            continue
        if case['workload']['operator'] not in {'fft', 'fwht', 'ntt'}:
            continue
        for variant in case['variants']:
            if variant.get('preflight', {}).get('status') != 'accepted':
                continue
            stem = output / (case['id'] + '.' + variant['id'])
            for tool in ['ncu', 'nsys']:
                record_path = Path(str(stem) + '.' + tool + '.json')
                if record_path.exists():
                    records.append(json.loads(record_path.read_text()))
                    continue
                driver = [str(previous.PYTHON), str(ROOT / 'paper/tools/profile_mechanism_call_20261012.py'),
                          '--journal', str(args.journal.resolve()), '--case', case['id'], '--variant', variant['id'],
                          '--output', str(stem) + '.' + tool + '.capture.json']
                if tool == 'ncu':
                    command = ['ncu', '--profile-from-start', 'off', '--target-processes', 'all',
                               '--kernel-name-base', 'demangled', '--metrics', ','.join(METRICS), '--csv',
                               '--log-file', str(stem) + '.ncu.csv'] + driver
                else:
                    command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                               '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                               '--output', str(stem) + '.nsys'] + driver
                require_exclusive_gpu()
                result = subprocess.run(command, text=True, capture_output=True)
                require_exclusive_gpu()
                record = dict(case=case['id'], variant=variant['id'], tool=tool, returncode=result.returncode,
                              command=command, logical_calls=1, warmup=100, normal_timing_accepted=False,
                              stdout=result.stdout, stderr=result.stderr)
                controlled._atomic_write(record_path, record)
                records.append(record)
                print(case['id'], variant['id'], tool, result.returncode, flush=True)
    controlled._atomic_write(output / 'manifest.json', dict(journal=str(args.journal.resolve()),
        journal_sha256=controlled._sha256(args.journal), metrics=METRICS,
        capture='one synchronized logical call after100 warmup, cudaProfilerApi range', records=records))


if __name__ == '__main__':
    main()
