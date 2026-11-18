#!/usr/bin/env python3
"""One independent CPU confirmation process: 75 original + 8 cyclic denominators."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def cpu_ticks(cores):
    ticks={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            core=int(fields[0][3:])
            if core in cores:
                values=list(map(int,fields[1:]));ticks[core]=(sum(values[:8]),sum(values[3:5]))
    return ticks


def load_probe(cores):
    first=cpu_ticks(cores);time.sleep(.2);last=cpu_ticks(cores)
    busy={str(c):100*(1-(last[c][1]-first[c][1])/max(1,last[c][0]-first[c][0])) for c in cores}
    return dict(selected_core_busy_percent=busy,loadavg=os.getloadavg(),
                measured_before_row_seconds=.2,reserved=False,exclusive=False)


def process_snapshot():
    observed=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            ticks=int(fields[11])+int(fields[12])
            cmd=(path/'comm').read_text().strip()
            affinity=sorted(os.sched_getaffinity(int(path.name)))
            observed.append(dict(pid=int(path.name),command=cmd,cpu_ticks=ticks,affinity=affinity))
        except (OSError,ValueError,IndexError,PermissionError):continue
    return dict(clock_ticks_per_second=os.sysconf('SC_CLK_TCK'),at=time.time(),processes=observed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measured-root',type=Path,required=True)
    p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--process',type=int,choices=(1,2,3),required=True)
    args=p.parse_args()
    root=args.measured_root.resolve();out=args.followup.resolve()/'cpu-confirmation'/f'process-{args.process}'
    assert not (out/'results.json').exists(),'Independent process output must be fresh'
    out.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(root/'scripts'),str(root/'python')]
    import numpy as np
    import run_semantic_campaign as semantic
    import semantic_preparation_queue as prep
    import run_comparator_extension as ext
    # CPU adapters are the reviewed PR versions; frozen input loaders stay bound above.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    import cpu_external_baselines as cpu
    from cpu_fxt_baseline import FP16FWHTAdapter
    from types import SimpleNamespace
    deps=root/'external/figure9-cpu'
    options=SimpleNamespace(fxt_library=deps/'install/lib/libfxt.so',hexl_root=deps/'hexl',butterfly_root=deps/'butterfly')
    affinity=list(range(16,24));os.sched_setaffinity(0,affinity)
    manifest=json.loads((root/'results/pro6000-issue9/main/full/manifest.json').read_text())
    old=json.loads((root/'results/pro6000-figure9-20261118/cpu-full66/results.json').read_text())
    old_mma=json.loads((root/'results/pro6000-figure9-20261118/cpu-mma9-adapted/results.json').read_text())
    old_rows={(r['case_id'],r['backend']):r for r in old['rows']}
    items=[]
    for case in manifest['cases']:
        payload=prep.load_case(root/'results/pro6000-issue9/main/full/prepared',case)
        spec=cpu.backend_specs(payload['s'],[8])[0]
        r=old_rows[case['id'],spec[0]]
        assert hashlib.sha256(payload['source'].tobytes()).hexdigest()==r['input_sha256']
        items.append((case['id'],payload,spec,r))
    m,loaded=ext.load_prepared(root/'results/pro6000-issue9-gpu0/mma9-local-paired')
    for case in m['config']['cases']:
        payload=loaded[case['id']]
        r=next(r for r in old_mma['rows'] if r['case_id']==case['id'])
        assert hashlib.sha256(payload['source'].tobytes()).hexdigest()==r['input_sha256']
        items.append((case['id'],payload,('fxt-fp16-io-fp32-cpu','adapter','FP16FWHTAdapter',1),r))
    m,loaded=ext.load_prepared(args.followup/'cyclic-ntt8')
    for case in m['config']['cases']:
        items.append((case['id'],loaded[case['id']],cpu.backend_specs(case['descriptor']['semantics'],[8])[0],None))
    assert len(items)==83 and len({i[0] for i in items})==83
    identity=dict(host=cpu.host_identity(),pid=os.getpid(),process_number=args.process,
        runner_sha256=semantic.sha256_file(__file__),
        cpu_adapter_sha256=semantic.sha256_file(cpu.__file__),
        protocol=dict(warmup=10,samples=7,calls=20,fft_threads=8,other_threads=1),
        affinity_by_group=dict(full66=affinity,mma9=[16],cyclic_ntt8=affinity),
        aggregation='median of seven samples per process, then median of three independent process medians',
        cpu_reserved=False,cpu_exclusive=False,
        limitation='No host/core reservation claimed; pre-row occupancy and process activity retained. Provisional unless a low-interference window is confirmed.')
    report=dict(schema='pro6000-figure9-cpu-confirmation-v1',identity=identity,status='running',rows=[],
        process_snapshot_start=process_snapshot(),expected_rows=83)
    cpu.write(out/'results.json',report)
    for identifier,payload,spec,old_row in items:
        s,source=payload['s'],payload['source']
        adapted=s['storage']==0
        os.sched_setaffinity(0,[16] if adapted else affinity)
        row=dict(case_id=identifier,backend=spec[0],semantics=s,
            input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
            affinity=sorted(os.sched_getaffinity(0)),load_before=load_probe(sorted(os.sched_getaffinity(0))),
            old_median_ms=old_row['timing']['median_ms'] if old_row else None,status='pending')
        core=None
        try:
            if adapted:
                expected=payload['expected'];reference=payload['reference_policy']
                core=FP16FWHTAdapter(s,source,options.fxt_library)
            else:
                expected,reference=cpu.same_precision_reference(s,source,payload['expected'])
                core=cpu.construct(spec,s,source,options)
            info=core.metadata;row['metadata']=info() if callable(info) else info
            row['reference']=reference
            row['reference_sha256']=hashlib.sha256(expected.tobytes()).hexdigest()
            row['timing']=cpu.measure(core,source,expected,s,10,7,20)
            row['correctness']=semantic.check_full(core.output,expected,s['storage'],'error-report',True)
            row['error_artifact']=semantic.retain_error_artifact(out/'errors',identifier,spec[0],core.output,expected)
            row['status']='pass' if s['storage'] in (8,9) else 'accuracy-review-required'
        except Exception as error:
            row.update(status='error',error=repr(error))
        finally:
            if core is not None:core.close()
        row['loadavg_end']=os.getloadavg()
        report['rows'].append(row);cpu.write(out/'results.json',report)
        print(args.process,len(report['rows']),identifier,row['status'],flush=True)
    os.sched_setaffinity(0,affinity)
    report['process_snapshot_end']=process_snapshot()
    report['status']='complete-provisional' if all(r['status'] in ('pass','accuracy-review-required') for r in report['rows']) else 'complete-with-failures'
    cpu.write(out/'results.json',report)


if __name__=='__main__':
    main()
