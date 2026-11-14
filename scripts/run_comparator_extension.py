#!/usr/bin/env python3
"""Fresh paired comparator campaign with CPU-bound references and guarded resume."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import run_semantic_campaign as runner
import semantic_cpu_pipeline as pipeline

REVISION = 'comparator-extension-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def runtime_binding_paths(config):
    """Small runtime inputs only; large per-case evidence stays provider-validated."""
    def local_path(value):
        path = Path(os.path.abspath(value))
        if any(path == root or root in path.parents for root in (Path('/sdb'), Path('/mnt/sdb'))):
            raise ValueError('runtime binding path outside allowed storage: '+str(path))
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in (Path('/sdb'), Path('/mnt/sdb'))):
            raise ValueError('runtime binding path outside allowed storage: '+str(path))
        return resolved
    paths = {local_path(config['library'])}
    environment = config['environment']
    for key, value in {**config, **environment}.items():
        if value and isinstance(value, str) and (key.endswith('_library') or key.endswith('_LIBRARY')):
            paths.add(local_path(value))
    for key in ('CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_HARDWARE_MODEL', 'CUBUTTERFLY_HADACORE_PROFILE'):
        if environment.get(key):
            paths.add(local_path(environment[key]))
    if environment.get('CUBUTTERFLY_PROFILE_ROOT'):
        root = local_path(environment['CUBUTTERFLY_PROFILE_ROOT'])
        if not root.is_dir():
            raise ValueError('runtime binding profile root missing: '+str(root))
        paths.update(local_path(p) for p in root.rglob('*.json'))
    # Bind provider/native shared libraries only. Historical JSON provenance
    # and potentially huge per-case evidence remain outside runtime discovery.
    scanned = set()
    while True:
        pending = [p for p in paths-scanned if p.suffix == '.json']
        if not pending:
            break
        for path in pending:
            scanned.add(path)
            if not path.is_file():
                raise ValueError('runtime binding file missing: '+str(path))
            def visit(value):
                if isinstance(value, dict):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, list):
                    for item in value:
                        visit(item)
                elif isinstance(value, str) and value.startswith('/') and Path(value).suffix == '.so':
                    paths.add(local_path(value))
            visit(json.loads(path.read_text()))
    return paths


def check_runtime_bindings(config, preparing=False):
    if 'runtime_bindings' not in config:
        return
    bindings = config['runtime_bindings']
    if not isinstance(bindings, dict) or any(not isinstance(k, str) or not Path(k).is_absolute()
                                            or not isinstance(v, str) for k,v in bindings.items()):
        raise ValueError('runtime_bindings must map absolute file paths to SHA256 hashes')
    required = runtime_binding_paths(config)
    actual = {}
    for path in sorted(required | {Path(p) for p in bindings}):
        if any(path == root or root in path.parents for root in (Path('/sdb'), Path('/mnt/sdb'))):
            raise ValueError('runtime binding path outside allowed storage: '+str(path))
        if not path.is_file():
            raise ValueError('runtime binding file missing: '+str(path))
        actual[str(path)] = sha(path)
    if preparing:
        if any(actual.get(path) != expected for path,expected in bindings.items()):
            raise ValueError('runtime binding checksum mismatch during preparation')
        config['runtime_bindings'] = actual
    elif actual != bindings:
        raise ValueError('runtime binding population/checksum mismatch')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def checked_config(config):
    for key in ('library', 'gpu_uuid', 'environment', 'cases'):
        if key not in config:
            raise ValueError('missing configuration field '+key)
    if not isinstance(config['environment'], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in config['environment'].items()):
        raise ValueError('environment must map strings to strings')
    ids = [c['id'] for c in config['cases']]
    if len(set(ids)) != len(ids) or any(not i or Path(i).name != i or i in ('.','..') for i in ids):
        raise ValueError('case IDs must be unique safe filenames')
    for c in config['cases']:
        if not isinstance(c.get('baselines'), list):
            raise ValueError('each case requires explicit baselines')
        runner.selected_baselines(c['baselines'])
        if 'semantics' not in c['descriptor']:
            raise ValueError('each case requires a canonical descriptor')
    return config


def primary_reference(prepared):
    s, source = prepared['s'], prepared['source']
    if s['operator'] == 0:
        from cpu_external_baselines import same_precision_reference
        return same_precision_reference(s, source)
    if s['operator'] == 2 and s['storage'] == 0:
        if s['length_mode'] or s['compute'] != 0:
            raise ValueError('FP16 FWHT reference requires unembedded native FP16 compute')
        work = source.astype(np.float16)
        for axis,n in enumerate(s['shape'], 1):
            if n < 1 or n & (n-1):
                raise ValueError('FWHT axes must be powers of two')
            moved = np.moveaxis(work, axis, -1).copy()
            rows = moved.reshape(-1,n)
            half = 1
            while half < n:
                blocks = rows.reshape(-1,2*half)
                left, right = blocks[:,:half].copy(), blocks[:,half:].copy()
                blocks[:,:half], blocks[:,half:] = left+right, left-right
                half *= 2
            if s['inverse'] and s['normalize_inverse']:
                np.multiply(rows, np.float16(1/n), out=rows)
            work = np.moveaxis(rows.reshape(moved.shape),-1,axis)
        return work, dict(kind='FP16-per-stage-rounded-Sylvester', dtype='float16', numpy=np.__version__)
    return prepared['expected'], dict(kind='semantic-cpu-reference')


def prepare(config, output):
    config = checked_config(copy.deepcopy(config))
    check_runtime_bindings(config, preparing=True)
    output = Path(output)
    manifest = dict(schema=REVISION, config=config, config_sha256=digest(config),
                    preparation_sources={p.name:sha(p) for p in (Path(__file__),Path(pipeline.__file__))}, cases=[])
    for case in config['cases']:
        record = dict(id=case['id'], semantics_sha256=digest(case['descriptor']['semantics']),
                      descriptor_sha256=digest(case['descriptor']))
        try:
            payload = pipeline.prepare_case(case, reference_max_elements=config.get('prepare_reference_limit',8388608))
            diagnostic = payload['expected']
            payload['expected'], reference_info = primary_reference(payload)
            path = output/'prepared'/(case['id']+'.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.npz.tmp')
            with temp.open('wb') as f:
                np.savez(f, **{k:payload[k] for k in ('source','expected','encoded_source','input_indices','output_indices')},
                         diagnostic_expected=diagnostic)
            temp.replace(path)
            record.update(status='prepared', path=str(path.relative_to(output)), sha256=sha(path), reference=reference_info)
        except runner.Gap as exc:
            record.update(status='preparation-gap', error=str(exc), gap_status=getattr(exc,'status','validation-unavailable'))
        manifest['cases'].append(record)
    write(output/'preparation.json', manifest)
    return manifest


def load_prepared(output):
    """Validate and load every array before importing Torch or initializing CUDA."""
    output = Path(output)
    manifest = json.loads((output/'preparation.json').read_text())
    config = checked_config(manifest['config'])
    if digest(config) != manifest['config_sha256']:
        raise ValueError('configuration checksum mismatch')
    check_runtime_bindings(config)
    records = manifest['cases']
    if [r['id'] for r in records] != [c['id'] for c in config['cases']]:
        raise ValueError('preparation case population mismatch')
    loaded = {}
    for case,r in zip(config['cases'],records):
        if r['semantics_sha256'] != digest(case['descriptor']['semantics']) or r['descriptor_sha256'] != digest(case['descriptor']):
            raise ValueError('prepared semantic binding mismatch')
        if r['status'] != 'prepared':
            continue
        expected_path = 'prepared/'+case['id']+'.npz'
        if r['path'] != expected_path or sha(output/r['path']) != r['sha256']:
            raise ValueError('prepared array checksum/path mismatch')
        with np.load(output/r['path'], allow_pickle=False) as arrays:
            payload = {k:arrays[k].copy() for k in arrays.files}
        payload.update(case_id=case['id'], s=copy.deepcopy(case['descriptor']['semantics']), reference_policy=r['reference'])
        loaded[case['id']] = payload
    return manifest, loaded


def source_fingerprints(config):
    directory = Path(__file__).resolve().parent
    names = {'run_comparator_extension.py', 'run_semantic_campaign.py',
             'semantic_cpu_pipeline.py'}
    baselines = {name for case in config['cases'] for name in case['baselines']}
    operators = {case['descriptor']['semantics']['operator'] for case in config['cases']}
    if 0 in operators:
        names.update(('cpu_external_baselines.py', 'semantic_preparation_queue.py',
                      'semantic_campaign_manifest.py'))
    if 1 in operators:
        names.add('semantic_ntt_reference.py')
    if 'cufft' in baselines:
        names.update(('diagnose_xd_fft.py', 'benchmark_xd.py', 'cufft_low_precision_baseline.py'))
    if 'vkfft' in baselines:
        names.add('vkfft_xd.py')
    if baselines & {'cufft', 'vkfft'}:
        names.add('semantic_fft_baseline.py')
    if 'daofht' in baselines:
        names.add('semantic_dao_baseline.py')
    if 'gpu-ntt' in baselines:
        names.add('semantic_gpuntt_baseline.py')
    for name in baselines & set(runner.EXTERNAL_CORE_PLUGINS):
        names.add(runner.EXTERNAL_CORE_PLUGINS[name][0]+'.py')
    return {name:sha(directory/name) for name in sorted(names)}


def environment_identity(configured):
    prefixes = ('CUDA_', 'CUBUTTERFLY_', 'TORCH_', 'OMP_', 'MKL_', 'OPENBLAS_')
    return dict(configured=dict(configured), effective={
        key:value for key,value in sorted(os.environ.items()) if key.startswith(prefixes)})


def run(output, case_id=None):
    output = Path(output)
    manifest, loaded = load_prepared(output)
    config = manifest['config']
    if case_id is not None and case_id not in {c['id'] for c in config['cases']}:
        raise ValueError('unknown --case')
    os.environ.update(config['environment'])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_uuid']
    # Check before Torch import and any CUDA initialization. Busy work remains pending.
    try:
        runner.exclusive_gpu(config['gpu_uuid'])
    except RuntimeError as exc:
        pending = [c['id'] for c in config['cases']
                   if not (output/'rows'/(c['id']+'.json')).exists()]
        write(output/'pending.json',dict(config_sha256=manifest['config_sha256'],
              pending=pending,status='busy',phase='before-torch-import',reason=str(exc)))
        return 76
    # Import only after the complete CPU population has been loaded.
    import torch
    import cubutterfly as cb
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties,'uuid',''))
    if not runner.same_gpu_uuid(config['gpu_uuid'],actual_uuid):
        raise ValueError('CUDA UUID mismatch')
    paths = dict(main=config['library'])
    paths.update({k:v for k,v in config.items() if k.endswith('_library') and v})
    for key,value in config['environment'].items():
        if key.endswith('_LIBRARY') and value:
            paths[key]=value
    identity = dict(revision=REVISION, config_sha256=manifest['config_sha256'],
        preparation_sha256=sha(output/'preparation.json'), sources=source_fingerprints(config),
        libraries={k:dict(path=str(Path(v).resolve()), sha256=sha(v) if Path(v).is_file() else None) for k,v in paths.items()},
        environment=environment_identity(config['environment']), python=sys.version, numpy=np.__version__,
        torch=torch.__version__, torch_cuda=torch.version.cuda, frontend_sha256=sha(cb.__file__),
        gpu_uuid=actual_uuid, gpu_name=properties.name, gpu_memory=properties.total_memory,
        compute_capability=[properties.major,properties.minor], protocol=dict(runner.PROTOCOL,
            reference_max_elements=config.get('prepare_reference_limit',8388608),selection_policy=0,
            primary_reference='prepared same-dtype FFT / per-stage FP16 FWHT'))
    if 'runtime_bindings' in config:
        identity['runtime_bindings'] = config['runtime_bindings']
    token = digest(identity)
    completed = set()
    for case in config['cases']:
        path=output/'rows'/(case['id']+'.json')
        if path.exists():
            envelope=json.loads(path.read_text())
            if envelope.get('identity_sha256') != token or envelope.get('identity') != identity:
                raise ValueError('resume identity mismatch for '+case['id'])
            if envelope.get('guards') != dict(before=True, after=True):
                raise ValueError('resume row lacks successful exclusivity guards')
            if envelope.get('row', {}).get('id') != case['id']:
                raise ValueError('resume row case ID mismatch')
            if envelope.get('row_sha256') != digest(envelope['row']):
                raise ValueError('resume row checksum mismatch')
            completed.add(case['id'])
    pending=[c['id'] for c in config['cases'] if c['id'] not in completed]
    write(output/'pending.json',dict(identity_sha256=token, pending=pending))
    library = None
    stream = None
    records={r['id']:r for r in manifest['cases']}
    for case in config['cases']:
        if case['id'] in completed or (case_id is not None and case['id'] != case_id):
            continue
        try:
            runner.exclusive_gpu(actual_uuid)
        except RuntimeError as exc:
            write(output/'pending.json',dict(identity_sha256=token,pending=pending,status='busy',reason=str(exc)))
            return 76
        if library is None:
            library = cb.Library(Path(config['library']))
            stream = torch.cuda.Stream()
        if case['id'] not in loaded:
            row=dict(id=case['id'], status=records[case['id']].get('gap_status','validation-unavailable'),
                     preparation=records[case['id']], descriptor=case['descriptor'],baselines=[])
        else:
            runtime_case=dict(case,populations=case.get('populations',['comparator-extension']))
            with torch.cuda.stream(stream):
                row=runner.run_case(runtime_case, library, torch, stream, timing=True,
                    reference_max_elements=config.get('prepare_reference_limit',8388608),
                    baseline_names=case['baselines'], prepared=loaded[case['id']], selection_policy=0,
                    vkfft_library=config.get('vkfft_library'),gpuntt_library=config.get('gpuntt_library'),
                    error_dir=output/'errors',float_verdict='error-report')
                stream.synchronize()
        try:
            runner.exclusive_gpu(actual_uuid)
        except RuntimeError as exc:
            write(output/'pending.json',dict(identity_sha256=token,pending=pending,status='busy',reason=str(exc)))
            return 76
        check_runtime_bindings(config)
        write(output/'rows'/(case['id']+'.json'),dict(identity=identity,identity_sha256=token,
              row=row,row_sha256=digest(row),guards=dict(before=True,after=True),
              prepared_sha256=records[case['id']].get('sha256')))
        pending.remove(case['id'])
        write(output/'pending.json',dict(identity_sha256=token,pending=pending))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case')
    args=parser.parse_args(argv)
    if args.prepare:
        if args.config is None:
            parser.error('--prepare requires --config')
        prepare(json.loads(args.config.read_text()),args.output)
        return 0
    return run(args.output,args.case)


if __name__ == '__main__':
    raise SystemExit(main())
