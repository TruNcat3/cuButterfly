#!/usr/bin/env python3
"""Finite current-build residency controls; no planner or core mutations."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/paper_targeted_20261012/mechanism'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_controlled_experiment as controlled
from run_comprehensive_suite import require_exclusive_gpu

spec = importlib.util.spec_from_file_location('e2_environment', ROOT / 'results/paper_targeted_20261011/followup_prepare/e2_current/run_e2_safe.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def study():
    cases = []
    for op, precision in [('fwht', 'fp32'), ('ntt', 'word64')]:
        for log_n, batch in [(8, 1), (16, 16)]:
            workload = dict(operator=op, precision=precision, logN=log_n, batch=batch,
                            placement='out-of-place', direction='forward', normalization='none')
            mapping = dict(schema_version=1, kind='ntt' if op == 'ntt' else 'butterfly',
                           backend='shared-iterative', compute_unit='radix2')
            if op == 'ntt':
                workload.update(modulus=576460756061519873, input_order='natural', output_order='natural')
                mapping.update(threads_per_block=128, dataflow_layout='linear', modular_multiply='shoup')
            else:
                mapping.update(tile_threads=128, fft_core='scalar', shared_layout='linear', local_exchange='shared')
            variants = [dict(id=name, mapping=dict(mapping, stage_partition=partition))
                        for name, partition in [('stagewise', [1] * log_n), ('resident8', [8] * (log_n // 8))]]
            cases.append(dict(id=f'{op}-{precision}-n{log_n}-b{batch}', experiment='E2-residency-fixed-core',
                              workload=workload, variants=variants,
                              comparisons=[dict(baseline='stagewise', treatment='resident8', changed_axes=['stage_partition'])]))
    source = json.loads((ROOT / 'results/paper_targeted_20261011/followup_prepare/e2_current/study.json').read_text())['cases'][0]
    fft = dict(id='fft-fp32-n8-b16384-factorial', experiment='E2-core-dataflow-2x2',
               workload=dict(source['workload'], batch=16384), variants=[], comparisons=[])
    for core_label, core in [('native', 'register-tile'), ('imported', 'cufftdx-block')]:
        for flow, overlap in [('serial', False), ('overlap', True)]:
            mapping = copy.deepcopy(source['variants'][0]['mapping'])
            mapping.update(fft_core=core, factor_slices=2, factor_overlap=overlap, data_tiles_per_cta=1)
            fft['variants'].append(dict(id=f'{core_label}-{flow}', mapping=mapping))
        fft['comparisons'].append(dict(baseline=f'{core_label}-serial', treatment=f'{core_label}-overlap', changed_axes=['factor_overlap']))
    for flow in ['serial', 'overlap']:
        fft['comparisons'].append(dict(baseline=f'native-{flow}', treatment=f'imported-{flow}', changed_axes=['fft_core']))
    cases.append(fft)
    return dict(schema=controlled.SCHEMA, protocol=dict(trials=3, warmup=100, repeat=100, verify_batches=0),
                scope='Four scalar-radix2 same-core residency pairs plus one factorized FFT core x slice-overlap factorial; finite cohort, no retuning.', cases=cases)


def environment(events):
    result = previous.env(events)
    result['LD_LIBRARY_PATH'] = str(previous.LIB.parent) + ':' + result.get('LD_LIBRARY_PATH', '')
    result['CUBUTTERFLY_COMPILE_SOCKET'] = '/tmp/cubutterfly-gpu2-20261011-r3.sock'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if previous.sha(previous.LIB) != previous.EXPECTED_LIB:
        raise RuntimeError('frozen runtime identity changed')
    calibration = json.loads(previous.CAL.read_text())
    if calibration['measurement_identity']['library_sha256'] != previous.EXPECTED_LIB:
        raise RuntimeError('calibration runtime mismatch')
    study_path = OUT / 'study.json'
    value = study()
    if study_path.exists() and json.loads(study_path.read_text()) != value:
        raise RuntimeError('cohort changed; choose a new directory')
    controlled._atomic_write(study_path, value)
    prepared = controlled.prepare_document(value, study_path, ROOT / 'build-e13-cufftdx')
    if not (OUT / 'prepared.json').exists():
        controlled._atomic_write(OUT / 'prepared.json', prepared)
    if not args.run:
        print(json.dumps({'status': 'prepared', 'cases': len(value['cases']), 'variants': 12}))
        return
    events = OUT / f'events_attempt{args.attempt}.jsonl'
    os.environ.update(environment(events))
    require_exclusive_gpu()
    command = [str(previous.PYTHON), str(ROOT / 'scripts/run_controlled_experiment.py'), '--mode', 'run',
               '--study', str(study_path), '--build-dir', str(ROOT / 'build-e13-cufftdx'),
               '--gpu-uuid', previous.GPU, '--output', str(OUT / f'timing_attempt{args.attempt}.json')]
    if args.resume:
        command.append('--resume')
    completed = subprocess.run(command, text=True, capture_output=True)
    (OUT / f'run_attempt{args.attempt}.log').write_text(completed.stdout + '\n' + completed.stderr)
    require_exclusive_gpu()
    status, detail = previous.check_events(events)
    controlled._atomic_write(OUT / f'run_audit_attempt{args.attempt}.json', dict(returncode=completed.returncode, preparation_status=status,
        preparation_detail=detail, command=command, library_sha256=previous.sha(previous.LIB),
        calibration_sha256=previous.sha(previous.CAL), gpu_uuid=previous.GPU))
    print(completed.stdout, completed.stderr, status, detail)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
