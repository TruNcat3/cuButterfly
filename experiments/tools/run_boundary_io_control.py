#!/usr/bin/env python3
"""Bounded existing-core external-boundary control, under run_isolated_gpu.py.

Preserves every process/trial and all negative results. The local-bank study
uses a separate driver and is not treated as another point of this control.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=os.environ.get('XD_REPORT_DIR'))
    parser.add_argument('--phase', choices=('timing', 'profile'), required=True)
    parser.add_argument('--ncu', default='/usr/local/cuda/bin/ncu')
    args = parser.parse_args()
    if args.output is None:
        parser.error('provide --output or an isolated GPU guard')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    spec = config['global_boundary']
    root = Path(__file__).resolve().parents[2]
    binary = args.binary.resolve()
    identity = dict(schema=config['schema'], phase=args.phase,
                    timestamp=datetime.now().astimezone().isoformat(),
                    gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    binary=str(binary), binary_sha256=sha(binary),
                    library_sha256=sha(binary.parent/'libcuntt.so'),
                    config=config, config_sha256=sha(args.config),
                    source_sha256={p: sha(root/p) for p in
                                   ['apps/cubutterfly_boundary_bench.cu',
                                    'src/shared_iterative.cuh', 'src/operator_traits.cuh',
                                    'include/cuntt/iterative_layout.hpp']},
                    timing_boundary='median of seven event intervals; each replays a complete-call Graph twenty times; capture, CPU reference and copies excluded',
                    control='Only external boundary changes: canonical strided / separate transpose / mapped producer stores. All share the writer-aligned stage core.',
                    process_order='requests reversed in process 1; modes rotate by trial within each fresh process',
                    calls=[])
    rows = []

    def record(label, cmd, profile=False, expected_modes=None):
        completed = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=600)
        (args.output/(label+'.stdout')).write_text(completed.stdout)
        (args.output/(label+'.stderr')).write_text(completed.stderr)
        identity['calls'].append(dict(label=label, command=list(map(str,cmd)),
                                      returncode=completed.returncode, diagnostic=profile))
        (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
        if completed.returncode:
            raise RuntimeError(f'{label} failed; raw stdout/stderr retained')
        if not profile:
            observed=list(csv.DictReader(io.StringIO(completed.stdout)))
            assert {r['mode'] for r in observed} == expected_modes, label
            for r in observed:
                assert int(r['nonfinite']) == 0, label
                if r['mode'] != 'cufft':
                    assert float(r['max_abs_same_core']) == 0, label
                rows.append(dict(run=label, **r))
            (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('PASS',label,flush=True)

    def arguments(op, request, repeat, trials, mode):
        return [op,spec['precision'],request['log_n'],request['batch'],request['split'],
                spec['tiles'],spec['threads'],spec['inverse'],repeat,trials,mode]

    if args.phase == 'timing':
        points=[(op,request) for op in spec['operators'] for request in spec['requests']]
        for process in range(spec['processes']):
            for op,request in (points if process % 2 == 0 else list(reversed(points))):
                label=f"p{process}-{op}-n{request['log_n']}-b{request['batch']}"
                modes=set(spec['modes']) | ({'cufft'} if op=='fft' else set())
                record(label,[binary,*arguments(op,request,spec['repeat'],spec['trials'],'all')],
                       expected_modes=modes)
        with (args.output/'rows.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        request=spec['profile_request']
        for op in spec['operators']:
            for mode in spec['modes']:
                label=f'profile-{op}-{mode}'
                count=3 if mode=='materialized' else 2
                # Two canonical launches, then one uncaptured complete-call
                # warmup in Graph.capture. The next launches are the same mode.
                skip=2+count
                command=[args.ncu,'--csv','--page','raw','--cache-control','none',
                         '--clock-control','none','--launch-skip',str(skip),
                         '--launch-count',str(count),'--metrics',','.join(config['profile_metrics']),
                         '--log-file',args.output/(label+'.csv'),
                         binary,*arguments(op,request,2,1,mode)]
                record(label,command,profile=True)
    (args.output/'complete.json').write_text(json.dumps(dict(status='complete',phase=args.phase,
                                              calls=len(identity['calls']),rows=len(rows)),indent=2)+'\n')


if __name__ == '__main__':
    main()
