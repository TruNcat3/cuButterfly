#!/usr/bin/env python3
"""Run a bounded prepared tile population or export validated measured candidates.

No CUDA work occurs without --execute. No busy-GPU waiting or implicit retry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import validate_hadacore_tile as validation

RESOURCE_FIELDS=('registers_per_thread','threads','static_shared_bytes','dynamic_shared_bytes',
                 'local_bytes_per_thread','max_threads_per_block','active_blocks_per_sm',
                 'binary_version','ptx_version')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def result_path(output,task):
    return Path(output)/'rows'/f'{task["case"]}-groups{task["groups_per_block"]}.json'


def checked_row(output,build,task,gpu_uuid):
    """CPU-only acceptance, binding every measured row to its current artifacts."""
    output,build=Path(output),Path(build)
    prepared,x,y,build_record=validation.validate_inputs(task['N'],task['B'],output,build,gpu_uuid)
    path=result_path(output,task);row=read(path)
    if row.get('status')!='validated-exact-upstream':raise ValueError('result is not validated-exact-upstream')
    for key in ('exact_repeat','input_immutable','buffer_guards','graph_reproduced',
                'upstream_graph_reproduced','exact_upstream_match','finite'):
        if row.get(key) is not True:raise ValueError('missing validation: '+key)
    if row.get('guards')!={'before':True,'after':True}:raise ValueError('exclusive guards incomplete')
    expected=dict(case=task['case'],groups_per_block=task['groups_per_block'],gpu_uuid=gpu_uuid,
                  contract=validation.CONTRACT,build_sha256=digest(build/'build.json'),
                  provider_sha256=digest(validation.PROVIDER),payload_sha256=prepared['artifact_sha256'],
                  runner_sha256=digest(validation.__file__),library=str((build/'libhadacore_tile.so').resolve()),
                  library_sha256=digest(build/'libhadacore_tile.so'))
    for key,value in expected.items():
        if row.get(key)!=value:raise ValueError('result identity mismatch: '+key)
    error_path=Path(row['error_artifact'])
    if digest(error_path)!=row.get('error_artifact_sha256'):raise ValueError('error artifact hash mismatch')
    resources=row.get('resources',{})
    if any(key not in resources for key in RESOURCE_FIELDS):raise ValueError('incomplete runtime resources')
    if resources['threads']<=0 or resources['active_blocks_per_sm']<=0:raise ValueError('invalid runtime resources')
    identity=row.get('gpu_identity',{})
    if identity.get('uuid')!=gpu_uuid or not identity.get('name') or not identity.get('memory_bytes') or not identity.get('sm'):
        raise ValueError('incomplete GPU identity')
    mapping=row.get('mapping',{})
    if mapping.get('log_n')!=task['N'].bit_length()-1 or mapping.get('tile_count')!=task['B'] or mapping.get('groups_per_block')!=task['groups_per_block']:
        raise ValueError('mapping differs from requested task')
    timing=row.get('timing',{});median=timing.get('median_ms')
    if not isinstance(median,(int,float)) or not math.isfinite(median) or median<=0:raise ValueError('missing measured local timing')
    samples=timing.get('samples_ms',[])
    if len(samples)!=7 or any(not math.isfinite(v) or v<=0 for v in samples):raise ValueError('incomplete timing samples')
    return row,path


def population(output):
    preparation=read(Path(output)/'prepare.json')
    tasks=preparation['gpu_tasks']
    keys=[(t['N'],t['B'],t['groups_per_block']) for t in tasks]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate prepared tasks')
    if any(t['gpu_uuid']!=preparation['gpu_uuid'] or t['arithmetic_contract']!=validation.CONTRACT for t in tasks):
        raise ValueError('prepared task contract/identity mismatch')
    return preparation,tasks


def export_profile(output,build,profile_path,native_profile=None):
    preparation,tasks=population(output);gpu_uuid=preparation['gpu_uuid']
    rows=[];pending=[];identity=None
    for task in tasks:
        p=result_path(output,task)
        if not p.exists():pending.append(dict(case=task['case'],groups_per_block=task['groups_per_block'],status='pending'));continue
        try:row,path=checked_row(output,build,task,gpu_uuid)
        except (ValueError,KeyError,OSError) as exc:
            pending.append(dict(case=task['case'],groups_per_block=task['groups_per_block'],status='invalid-result',reason=str(exc)));continue
        if identity is not None and identity!=row['gpu_identity']:raise ValueError('mixed hardware identities')
        identity=row['gpu_identity']
        rows.append(dict(log_n=task['N'].bit_length()-1,batch=task['B'],chunks=row['mapping']['chunks_per_warp'],
            groups_per_block=task['groups_per_block'],status='validated-exact-upstream',median_ms=row['timing']['median_ms'],
            timing_protocol=row['timing']['protocol'],
            evidence_path=str(path.resolve()),evidence_sha256=digest(path),resources=row['resources']))
    if not rows:raise ValueError('no validated current measured rows; profile not written')
    profile=dict(schema='cubutterfly-hadacore-profile-v1',gpu_uuid=gpu_uuid,sm=identity['sm'],gpu_identity=identity,
        library=str((Path(build)/'libhadacore_tile.so').resolve()),library_sha256=digest(Path(build)/'libhadacore_tile.so'),
        arithmetic_contract=validation.CONTRACT,rows=rows,pending=pending,
        total_prepared_tasks=len(tasks),measured_candidates=len(rows),complete=not pending)
    if native_profile is not None:
        native=read(native_profile)
        if native.get('schema')!='cubutterfly-hadacore-native-profile-v1' or native.get('gpu_identity')!=identity:
            raise ValueError('native profile hardware/schema mismatch')
        if digest(native['library'])!=native['library_sha256']:
            raise ValueError('native profile library changed')
        accepted=[]
        for control in native['native_rows']:
            if digest(control['evidence_path'])!=control['evidence_sha256']:
                raise ValueError('native evidence hash mismatch')
            evidence=read(control['evidence_path'])
            if (evidence.get('status')!='measured-error-report' or
                evidence.get('arithmetic_policy')!='STRICT' or
                evidence.get('library_sha256')!=native['library_sha256'] or
                control.get('library_sha256')!=native['library_sha256'] or
                evidence.get('gpu_identity')!=identity or
                evidence.get('mapping')!=control['mapping'] or
                evidence.get('case')!=f'N{2**control["log_n"]}xB{control["batch"]}' or
                evidence.get('guards')!={'before':True,'after':True} or
                any(evidence.get(k) is not True for k in
                    ('finite','exact_repeat','input_immutable','buffer_guards','graph_reproduced')) or
                evidence['timing']['median_ms']!=control['median_ms'] or
                evidence['timing']['protocol']!=control['timing_protocol']):
                raise ValueError('native control does not match verified evidence')
            if digest(evidence['error_artifact'])!=evidence['error_artifact_sha256']:
                raise ValueError('native error evidence changed')
            accepted.append(control)
        profile['native_rows']=accepted
        profile['native_profile_sha256']=digest(native_profile)
        profile['paired_requests']=len({(r['log_n'],r['batch']) for r in rows} &
                                        {(r['log_n'],r['batch']) for r in accepted})
    write(profile_path,profile);return profile


def execute_population(output,build,upstream_library,python=sys.executable,retry_failed=False):
    output,build=Path(output),Path(build);prepared,tasks=population(output)
    checkpoint=dict(schema='hadacore-calibration-checkpoint-v1',status='running',total=len(tasks),completed=0,jobs={})
    checkpoint_path=output/'calibration-checkpoint.json'
    for task in tasks:
        name=f'{task["case"]}-groups{task["groups_per_block"]}';p=result_path(output,task)
        if p.exists():
            try:checked_row(output,build,task,prepared['gpu_uuid'])
            except (ValueError,KeyError,OSError) as exc:
                if not retry_failed:
                    checkpoint.update(status='blocked-existing-result',blocked_job=name,reason=str(exc));write(checkpoint_path,checkpoint);return checkpoint
                archive=output/'history'/f'{name}-{time.time_ns()}.json';archive.parent.mkdir(exist_ok=True);p.replace(archive)
            else:
                checkpoint['jobs'][name]={'status':'reused'};checkpoint['completed']+=1;write(checkpoint_path,checkpoint);continue
        cmd=[python,str(Path(validation.__file__).resolve()),'--run','--execute','--case',task['case'],
             '--groups-per-block',str(task['groups_per_block']),'--build',str(build),'--output',str(output),
             '--gpu-uuid',prepared['gpu_uuid'],'--upstream-library',str(upstream_library)]
        log=output/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        checkpoint['jobs'][name]={'status':'running','command':cmd,'log':str(log)};write(checkpoint_path,checkpoint)
        with log.open('ab') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        try:
            if result.returncode:raise ValueError(f'validation subprocess exit {result.returncode}; see {log}')
            checked_row(output,build,task,prepared['gpu_uuid'])
        except (ValueError,KeyError,OSError) as exc:
            checkpoint['jobs'][name]['status']='failed-or-interrupted'
            checkpoint.update(status='stopped',blocked_job=name,reason=str(exc));write(checkpoint_path,checkpoint);return checkpoint
        checkpoint['jobs'][name]['status']='validated';checkpoint['completed']+=1;write(checkpoint_path,checkpoint)
    checkpoint['status']='complete';write(checkpoint_path,checkpoint);return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--export-only',action='store_true')
    p.add_argument('--upstream-library',type=Path);p.add_argument('--profile',type=Path)
    p.add_argument('--native-profile',type=Path,help='Merge verified same-device STRICT public controls')
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    if a.export_only:
        if not a.profile:p.error('--export-only requires --profile')
        report=export_profile(a.output,a.build,a.profile,a.native_profile)
    elif a.execute:
        if not a.upstream_library:p.error('--execute requires --upstream-library')
        report=execute_population(a.output,a.build,a.upstream_library,retry_failed=a.retry_failed)
        if report['status']=='complete' and a.profile:export_profile(a.output,a.build,a.profile,a.native_profile)
    else:
        prepared,tasks=population(a.output);report=dict(status='prepared',gpu_execution=False,total=len(tasks),gpu_uuid=prepared['gpu_uuid'])
    print(json.dumps(report,indent=2))
    if report.get('status') in ('stopped','blocked-existing-result'):sys.exit(1)
if __name__=='__main__':main()
