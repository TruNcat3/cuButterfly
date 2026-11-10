#!/usr/bin/env python3
"""Explicit isolated NCU diagnostics. Profiler times are NOT normal timings."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'experiments/provenance/PERFORMANCE_BOTTLENECK_AUDIT_20261012.json'
OUTPUT=ROOT/'results/performance_bottleneck_profile_20261012/guarded_v1'
GPU='GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'
GUARD=ROOT/'scripts/run_isolated_gpu.py'
PYTHON='/home/wt/yes/envs/cubutterfly/bin/python'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def resolved(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def validate_record(record):
    label=record['label']
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',label): raise ValueError('unsafe label')
    if record['gpu_uuid']!=GPU or record['environment'].get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('prepared command targets wrong GPU')
    for name in ('source','binary_current_identity'):
        item=record[name];path=resolved(item['path'])
        if digest(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise ValueError(f'{label}: {name} identity mismatch')
    if record['binary_expected_sha256']!=record['binary_current_identity']['sha256']:
        raise ValueError('binary differs from recorded normal-timing executable')
    argv=record['argv']
    if argv.count('--log-file')!=1 or argv.count('--metrics')!=1:
        raise ValueError('ambiguous NCU command')
    binary=resolved(record['binary_current_identity']['path'])
    if str(binary) not in argv: raise ValueError('profile executable absent from argv')
    if record['expected_kernel_groups']!=2 or argv[argv.index('--launch-count')+1]!='2':
        raise ValueError('expected exactly two stage groups')


def command_for(record,attempt):
    argv=list(record['argv'])
    argv[argv.index('--log-file')+1]=str(attempt/f"{record['label']}.ncu.csv")
    return argv


def parse_metrics(path,record):
    required=record['argv'][record['argv'].index('--metrics')+1].split(',')
    header=None;groups={}
    with Path(path).open(newline='') as stream:
        rows=list(csv.reader(stream))
    # NCU's --csv output is wide (one row per kernel and one column per
    # metric). Some older NCU releases emit a narrow Metric Name/Value table;
    # accept both representations while requiring the same two kernel rows.
    wide=next((row for row in rows if 'ID' in row and 'Kernel Name' in row
               and set(required).issubset(row)), None)
    if wide is not None:
        id_index=wide.index('ID'); kernel_index=wide.index('Kernel Name')
        for row in rows:
            if len(row)!=len(wide) or not row[kernel_index]: continue
            key=row[id_index]; kernel=row[kernel_index]
            if key == 'ID' and kernel == 'Kernel Name': continue
            if not key or not re.search(r'shared_(iterative|resident)_kernel',kernel):
                raise ValueError('unexpected captured kernel')
            group=groups.setdefault(key,dict(kernel=kernel,metrics={}))
            if group['kernel']!=kernel: raise ValueError('ambiguous kernel metric')
            for metric in required:
                try: value=float(row[wide.index(metric)].replace(',','').strip())
                except (ValueError,IndexError) as exc:
                    raise ValueError(f'missing/non-numeric metric {metric}') from exc
                if not math.isfinite(value): raise ValueError('nonfinite profiler metric')
                if metric in group['metrics']: raise ValueError('duplicate kernel metric')
                group['metrics'][metric]=dict(value=value,unit='')
    else:
      for row in rows:
            if 'Metric Name' in row and 'Metric Value' in row and 'Kernel Name' in row and 'ID' in row:
                header=row;continue
            if header is None or len(row)!=len(header): continue
            item=dict(zip(header,row));metric=item['Metric Name']
            if metric not in required: continue
            key=item['ID'];kernel=item['Kernel Name']
            if not key or not re.search(r'shared_(iterative|resident)_kernel',kernel):
                raise ValueError('unexpected captured kernel')
            try: value=float(item['Metric Value'].replace(',','').strip())
            except ValueError as exc: raise ValueError(f'missing/non-numeric metric {metric}') from exc
            if not math.isfinite(value): raise ValueError('nonfinite profiler metric')
            group=groups.setdefault(key,dict(kernel=kernel,metrics={}))
            if group['kernel']!=kernel or metric in group['metrics']:
                raise ValueError('ambiguous or duplicate kernel metric')
            group['metrics'][metric]=dict(value=value,unit=item.get('Metric Unit',''))
    if len(groups)!=record['expected_kernel_groups']:
        raise ValueError(f'expected two captured groups; found {len(groups)}')
    result=[]
    for index,(launch_id,group) in enumerate(groups.items()):
        if set(group['metrics'])!=set(required): raise ValueError('missing requested stage metrics')
        result.append(dict(label=record['label'],stage_group=index,capture_launch_id=launch_id,
            kernel=group['kernel'],metrics=group['metrics'],
            stage_assignment='capture order; two full requested stage groups, not inferred kernel argument values',
            timing_kind='NCU replay diagnostic, not normal timing'))
    return result


def prepare(output,audit):
    document=json.loads(audit.read_text());records=document['ncu_prepared']
    if len(records)!=8 or len({r['label'] for r in records})!=8:
        raise ValueError('expected eight unique prepared profile commands')
    for record in records: validate_record(record)
    paths={str(audit.resolve()),str(Path(__file__).resolve()),str(GUARD)}
    paths.update(str(resolved(r['argv'][0])) for r in records)
    paths.add(str(ROOT/'results/search_budget_20261011/frozen_runtime/libcuntt.so'))
    spec=dict(schema='bottleneck-ncu-diagnostics-v1',audit_path=str(audit.resolve()),
        hashes={path:digest(path) for path in sorted(paths)},records=records,
        claim_boundary='NCU counters and replay times only; never substitute for normal application/operator timings')
    output.mkdir(parents=True,exist_ok=True);path=output/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec: raise ValueError('immutable spec changed; choose new output root')
    save(path,spec);return path


def verify(spec):
    for path,expected in spec['hashes'].items():
        if digest(path)!=expected: raise ValueError(f'profile source/tool/runtime changed: {path}')
    for record in spec['records']: validate_record(record)


def write_csv(path,rows):
    metrics=sorted({metric for row in rows for metric in row['metrics']})
    fields=['label','stage_group','capture_launch_id','kernel','timing_kind']
    fields+=[column for metric in metrics for column in (metric,metric+'__unit')]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record={key:row[key] for key in ('label','stage_group','capture_launch_id','kernel','timing_kind')}
            for metric,value in row['metrics'].items():
                record[metric]=value['value'];record[metric+'__unit']=value['unit']
            writer.writerow(record)


def child(spec_path):
    spec=json.loads(spec_path.read_text());verify(spec)
    attempt=Path(os.environ['XD_REPORT_DIR']).resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU: raise ValueError('child must run under GPU2 guard')
    summary=dict(status='running',spec_sha256=digest(spec_path),claim_boundary=spec['claim_boundary'],profiles=[],stage_rows=[])
    for record in spec['records']:
        verify(spec)
        label=record['label'];argv=command_for(record,attempt)
        csv_path=attempt/f'{label}.ncu.csv'
        if csv_path.exists(): raise ValueError('refusing to overwrite prior NCU capture')
        outcome=dict(label=label,command=argv,status='failed')
        try:
            child_env = dict(os.environ, **record['environment'])
            # The research compile transport requires an attempt-local event
            # sink. The prepared historical command omitted this environment
            # entry, which made NCU fail before launching any kernel.
            child_env['CUBUTTERFLY_PREPARE_EVENTS'] = str(attempt / 'preparation_events.jsonl')
            with (attempt/f'{label}.stdout.log').open('x') as log:
                completed=subprocess.run(argv,cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
            outcome['returncode']=completed.returncode
            if completed.returncode: raise ValueError('NCU subprocess failed')
            rows=parse_metrics(csv_path,record)
            summary['stage_rows'].extend(rows)
            outcome.update(status='complete',csv_sha256=digest(csv_path),captured_groups=len(rows))
        except (OSError,ValueError) as exc:
            outcome['error']=str(exc)
        summary['profiles'].append(outcome)
        save(attempt/'profile_summary.json',summary)
        write_csv(attempt/'stage_metrics.csv',summary['stage_rows'])
    summary['status']='complete' if all(p['status']=='complete' for p in summary['profiles']) else 'failed'
    save(attempt/'profile_summary.json',summary)
    return 0 if summary['status']=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--child',action='store_true')
    parser.add_argument('--audit',type=Path,default=AUDIT);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--spec',type=Path);parser.add_argument('--until',default='2026-10-12T18:00:00+08:00')
    args=parser.parse_args()
    if args.child: return child(args.spec)
    path=prepare(args.output,args.audit)
    if args.prepare: print(path);return 0
    deadline=datetime.fromisoformat(args.until)
    if deadline.tzinfo is None: parser.error('--until must include timezone')
    if deadline<=datetime.now().astimezone(): parser.error('--until must be a future finite deadline')
    command=[PYTHON,str(GUARD),'--gpu-uuid',GPU,'--report-dir',str((args.output/'guard').resolve()),
        '--until',args.until,'--idle-seconds','3','--',PYTHON,str(Path(__file__).resolve()),'--child','--spec',str(path.resolve())]
    return subprocess.run(command,cwd=ROOT).returncode


if __name__=='__main__': raise SystemExit(main())
