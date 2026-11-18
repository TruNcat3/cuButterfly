"""CPU-only exact parallel execution of the repository's static ranker.

Each contiguous shard calls the unchanged serial function. Global stable tie
order, confidence labels and the complete accepted/rejected/unranked partition
are retained. Cache hits require the exact request, code, profiles and backend.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import rank_static_candidates as serial
import stage_schedule_native as native
REQUEST=None
VERSION='exact-contiguous-static-rank-v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()

def profiles(directory):
    if directory is None:return {}
    directory=Path(directory)
    return {str(directory/name):file_sha(directory/name) if (directory/name).exists() else None
            for name in ('calibration_device.json','hardware_profile.json','cost_model.json')}

def current_profiles(recorded):
    return {name:file_sha(name) if Path(name).exists() else None for name in recorded}

def dependencies():
    os.environ['CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY']='1'
    sources={p.name:file_sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}
    source=ROOT/'scripts/stage_schedule_native.cpp';sources[source.name]=file_sha(source)
    identifier=hashlib.sha256(source.read_bytes()+os.uname().machine.encode()).hexdigest()[:24]
    library=Path(tempfile.gettempdir())/f'cubutterfly-stage-schedule-{os.getuid()}'/(identifier+'.so')
    return dict(version=VERSION,wrapper_sha256=file_sha(__file__),sources=sources,
        python=sys.version,machine=platform.machine(),backend=native.backend(),
        native_library_sha256=file_sha(library) if library.exists() else None,
        environment={k:v for k,v in sorted(os.environ.items()) if k.startswith('CUBUTTERFLY_')
            or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED')})

def validate_indices(request):
    if not isinstance(request,dict) or request.get('schema')!=serial.REQUEST_SCHEMA:
        raise ValueError('unsupported static-ranking request schema')
    candidates=request['candidates']
    if not isinstance(candidates,list):raise ValueError('candidates must be an array')
    indices=[item['index'] for item in candidates]
    if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('candidate indices must be distinct nonnegative integers')

def shard(bounds):
    start,stop=bounds
    return serial.rank(dict(REQUEST,candidates=REQUEST['candidates'][start:stop]))

def combine(parts):
    if not parts:raise ValueError('No ranking partitions')
    first=parts[0]
    for part in parts:
        for key in ('schema','profile_source','reason','schedule_backend'):
            if part.get(key)!=first.get(key):raise ValueError('Shard model/backend metadata differ')
    result=dict(first)
    for category in ('ranked','rejected','unranked'):
        result[category]=[row for part in parts for row in part[category]]
    # Parts are contiguous and ordered. Python's stable sort consequently
    # preserves the original candidate order for equal floating point scores.
    result['ranked'].sort(key=lambda row:row['predicted_kernel_ms'])
    modes={row['confidence'] for row in result['ranked']}
    result['mode']=('mixed-static-model-and-hardware-rate-prior' if len(modes)>1
                    else next(iter(modes)) if modes else 'unavailable')
    return result

def rank_parallel(request,workers=8):
    global REQUEST
    validate_indices(request)
    if type(workers) is not int or not 1<=workers<=32:raise ValueError('workers must be in [1,32]')
    before=profiles(request.get('profile_dir'))
    count=len(request['candidates'])
    if workers==1 or count<workers:
        result=serial.rank(request)
    else:
        # Validate the hardware/model boundary before forking, without scoring
        # or copying the large candidate array. No CUDA API is called here.
        serial.rank(dict(request,candidates=[]))
        REQUEST=request;step=(count+workers-1)//workers
        bounds=[(start,min(count,start+step)) for start in range(0,count,step)]
        try:
            with multiprocessing.get_context('fork').Pool(len(bounds)) as pool:
                result=combine(pool.map(shard,bounds))
        finally:REQUEST=None
    if current_profiles(before)!=before:raise ValueError('Ranking profile changed during scoring')
    return result

def cache_read(path,request_sha,dependency):
    if not path.exists():return None
    with gzip.open(path,'rt') as stream:entry=json.load(stream)
    identity=entry['identity']
    if entry['identity_sha256']!=digest(identity) or entry['result_sha256']!=digest(entry['result']):
        raise ValueError('Ranking cache checksum mismatch')
    if identity.get('request_sha256')!=request_sha or identity.get('dependencies')!=dependency:
        return None
    if current_profiles(identity['profile_files'])!=identity['profile_files']:return None
    return entry['result']

def cache_write(path,request_sha,dependency,profile_files,result):
    identity=dict(request_sha256=request_sha,dependencies=dependency,profile_files=profile_files)
    entry=dict(identity=identity,identity_sha256=digest(identity),result=result,result_sha256=digest(result))
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with gzip.open(temporary,'wt') as stream:json.dump(entry,stream,allow_nan=False,separators=(',',':'))
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)

def run(request_path,output_path,workers=8,cache_directory=None,audit_directory=None):
    start=time.monotonic();request_path=Path(request_path);output_path=Path(output_path)
    request_sha=file_sha(request_path);dependency=dependencies();hit=False;cache_error=None
    cache_directory=Path(cache_directory) if cache_directory else None
    if cache_directory:cache_directory.mkdir(parents=True,exist_ok=True)
    lock_path=cache_directory/(request_sha+'.lock') if cache_directory else None
    lock=lock_path.open('a') if lock_path else nullcontext()
    with lock as handle:
        if lock_path:fcntl.flock(handle,fcntl.LOCK_EX)
        cache_path=cache_directory/(request_sha+'.json.gz') if cache_directory else None
        result=None
        if cache_path:
            try:result=cache_read(cache_path,request_sha,dependency)
            except (OSError,ValueError,KeyError,TypeError,EOFError) as error:
                cache_error=repr(error)
                if cache_path.exists():cache_path.replace(cache_path.with_name(cache_path.name+f'.invalid-{time.time_ns()}'))
            hit=result is not None
        if result is None:
            request=serial._read(request_path);before=profiles(request.get('profile_dir'))
            result=rank_parallel(request,workers)
            if current_profiles(before)!=before:raise ValueError('Ranking profile changed before publication')
            if dependencies()!=dependency:raise ValueError('Ranking source/backend changed during scoring')
            if cache_path:cache_write(cache_path,request_sha,dependency,before,result)
    temporary=output_path.with_name(output_path.name+f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
        temporary.replace(output_path)
    finally:temporary.unlink(missing_ok=True)
    audit=dict(version=VERSION,request_sha256=request_sha,request_bytes=request_path.stat().st_size,
        result_sha256=digest(result),workers=workers,cache_hit=hit,cache_error=cache_error,
        elapsed_seconds=time.monotonic()-start,counts={key:len(result[key]) for key in ('ranked','rejected','unranked')},
        dependencies=dependency,scope='Exact original static ranking function on contiguous CPU shards; stable tie order and original coverage labels retained. Planning only, no GPU measurement or acceptance.')
    if audit_directory:
        directory=Path(audit_directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/(str(time.time_ns())+'-'+request_sha+'.json')).write_text(json.dumps(audit,indent=2)+'\n')
    return audit

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=int(os.environ.get('CUBUTTERFLY_STATIC_RANK_WORKERS','8')))
    args=parser.parse_args();expected=os.environ.get('CUBUTTERFLY_STATIC_RANKER_SHA256')
    if expected and file_sha(__file__)!=expected:raise ValueError('Parallel ranker source checksum differs')
    run(args.request,args.output,args.workers,os.environ.get('CUBUTTERFLY_STATIC_RANK_CACHE'),os.environ.get('CUBUTTERFLY_STATIC_RANK_AUDIT_DIR'))
if __name__=='__main__':main()
