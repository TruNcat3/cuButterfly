#!/usr/bin/env python3
"""Run a bounded prefix-bank control under the existing exclusive GPU guard."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,default=os.environ.get('XD_REPORT_DIR'))
    p.add_argument('--phase',choices=('timing','profile'),required=True)
    p.add_argument('--ncu',default='/usr/local/cuda/bin/ncu')
    a=p.parse_args()
    if a.output is None:p.error('provide --output or isolated GPU guard')
    a.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(a.config.read_text());spec=config['local_bank']
    source=Path(__file__).resolve().parents[1]/'experiments/boundary_bank_control.cpp'
    identity=dict(schema=config['schema'],study='local-bank',phase=a.phase,
                  timestamp=datetime.now().astimezone().isoformat(),
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  binary=str(a.binary),binary_sha256=sha(a.binary),library_sha256=sha(a.library),
                  source_sha256=sha(source),config=config,config_sha256=sha(a.config),calls=[])
    cases=[]
    def record(label,command,profile):
        done=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=600)
        (a.output/(label+'.stdout')).write_text(done.stdout)
        (a.output/(label+'.stderr')).write_text(done.stderr)
        identity['calls'].append(dict(label=label,command=list(map(str,command)),returncode=done.returncode))
        (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        if done.returncode:raise RuntimeError(label+' failed; raw records retained')
        report=json.loads(done.stdout)
        assert report['strict_control_valid'] and report['byte_identical_outputs'],label
        assert len(report['cases'])==2
        maps=[dict(r['mapping_json']) for r in report['cases']]
        assert [m.pop('shared_layout') for m in maps]==['linear','xor-swizzle'],label
        assert maps[0]==maps[1], 'undeclared mapping difference: '+label
        for r in report['cases']:
            assert r['cpu_error']['nonfinite']==0 and r['verified_batches']==report['batch']
        (a.output/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        cases.append(dict(run=label,**report))
        (a.output/'rows.json').write_text(json.dumps(cases,indent=2)+'\n')
        print('PASS',label,flush=True)
    if a.phase=='timing':
        for process in range(spec['processes']):
            for request in (spec['requests'] if process%2==0 else list(reversed(spec['requests']))):
                label=f"p{process}-n{request['log_n']}-b{request['batch']}"
                record(label,[a.binary,'--logN',request['log_n'],'--batch',request['batch'],
                              '--repeat',spec['repeat'],'--trials',spec['trials']],False)
    else:
        for request in spec['requests']:
            for mode in spec['modes']:
                label=f"profile-n{request['log_n']}-b{request['batch']}-{mode}"
                record(label,[a.ncu,'--csv','--page','raw','--profile-from-start','off',
                              '--cache-control','none','--clock-control','none','--metrics',
                              ','.join(config['profile_metrics']),'--log-file',a.output/(label+'.csv'),
                              a.binary,'--logN',request['log_n'],'--batch',request['batch'],
                              '--only',mode,'--profile'],True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='complete',phase=a.phase,calls=len(cases)),indent=2)+'\n')


if __name__=='__main__':main()
