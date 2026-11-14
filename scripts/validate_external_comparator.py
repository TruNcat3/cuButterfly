#!/usr/bin/env python3
"""Validate prepared external FFT/FWHT adapters without a cuButterfly plan.

Out-of-place cases reuse the runner's full direct and CUDA graph replay checks.
In-place cases retain direct/event verification but report an explicit graph gap.
Timing collected to exercise the existing verification path is diagnostic only.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import run_comparator_extension as campaign
import run_semantic_campaign as runner
import semantic_cpu_pipeline as pipeline

BACKENDS = {'gpu-fft':'gpufft_baseline', 'hadacore':'hadacore_baseline'}


def physical_buffers(prepared):
    s=prepared['s']
    sizes={side:(int(prepared[side+'_indices'].max(initial=-1))+1)*runner.side_dtype(s,side).itemsize
           for side in ('input','output')}
    if s['placement']:
        sizes['input']=sizes['output']=max(sizes.values())
    source=pipeline.prepared_physical_bytes(prepared,'input',prepared['source'],sizes['input'])
    output=source.copy() if s['placement'] else np.full(sizes['output'],0xA5,dtype=np.uint8)
    return source,output


def validate(prepared_directory, backend, output):
    if backend not in BACKENDS:
        raise ValueError('backend must be gpu-fft or hadacore')
    prepared_directory,output=Path(prepared_directory).resolve(),Path(output).resolve()
    if output==prepared_directory or prepared_directory in output.parents:
        raise ValueError('validation output must be independent of the prepared campaign')
    manifest,loaded=campaign.load_prepared(prepared_directory)
    config=manifest['config']
    selected=[c for c in config['cases'] if backend in c['baselines']]
    if not selected:
        raise ValueError('backend is not declared in any prepared case')
    records={r['id']:r for r in manifest['cases']}
    os.environ.update(config['environment'])
    os.environ['CUDA_VISIBLE_DEVICES']=config['gpu_uuid']
    pending=[c['id'] for c in selected]
    def busy(phase,token=None):
        try:
            runner.exclusive_gpu(config['gpu_uuid'])
            return False
        except RuntimeError as exc:
            campaign.write(output/'pending.json',dict(status='busy',phase=phase,backend=backend,
                pending=pending,config_sha256=manifest['config_sha256'],identity_sha256=token,reason=str(exc)))
            return True
    if busy('before-torch-import'):
        return 76
    plugin=importlib.import_module(BACKENDS[backend])
    import torch
    properties=torch.cuda.get_device_properties(0)
    actual_uuid=str(getattr(properties,'uuid',''))
    if not runner.same_gpu_uuid(config['gpu_uuid'],actual_uuid):
        raise ValueError('CUDA UUID mismatch')
    library=Path(plugin.DEFAULT_LIBRARY).resolve()
    paths=(Path(__file__),Path(runner.__file__),Path(pipeline.__file__),Path(campaign.__file__),Path(plugin.__file__))
    identity=dict(schema='external-comparator-validation-v1',backend=backend,
        preparation_sha256=campaign.sha(prepared_directory/'preparation.json'),
        prepared_arrays={c['id']:records[c['id']].get('sha256') for c in selected},
        library=dict(path=str(library),sha256=campaign.sha(library) if library.is_file() else None),
        sources={p.name:campaign.sha(p) for p in paths},gpu_uuid=actual_uuid,
        gpu_name=properties.name,gpu_memory=properties.total_memory,
        torch=torch.__version__,torch_cuda=torch.version.cuda,python=sys.version,numpy=np.__version__,
        environment=campaign.environment_identity(config['environment']),
        protocol=dict(runner.PROTOCOL,purpose='external adapter validation only; no own plan or paired performance',
                      float_verdict='error-report',graph='out-of-place graph; explicit gap for in-place'))
    token=campaign.digest(identity)
    for case in selected:
        path=output/'rows'/(case['id']+'.json')
        if not path.is_file():
            continue
        envelope=json.loads(path.read_text())
        if envelope.get('identity')!=identity or envelope.get('identity_sha256')!=token:
            raise ValueError('validation resume identity mismatch')
        if envelope.get('row_sha256')!=campaign.digest(envelope['row']) or envelope['row'].get('id')!=case['id']:
            raise ValueError('validation resume row checksum/binding mismatch')
        if envelope.get('guards')!={'before':True,'after':True}:
            raise ValueError('validation resume missing guards')
        pending.remove(case['id'])
    campaign.write(output/'pending.json',dict(identity_sha256=token,pending=pending))
    stream=None
    for case in selected:
        identifier=case['id']
        if identifier not in pending:
            continue
        if busy('before-case',token):
            return 76
        row=dict(id=identifier,descriptor=case['descriptor'],backend=backend,purpose='validation-only',baselines=[])
        if identifier not in loaded:
            row.update(status=records[identifier].get('gap_status','validation-unavailable'),preparation=records[identifier],graph_status='not-run')
        else:
            payload=loaded[identifier]
            source_bytes,output_initial=physical_buffers(payload)
            if stream is None:
                stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                input_saved=torch.from_numpy(source_bytes).to('cuda')
                try:
                    baseline_rows=runner.run_baselines(case,torch,stream,payload['source'],payload['expected'],
                        input_saved,output_initial,timing=True,vkfft_library=None,
                        float_verdict='error-report',error_dir=output/'errors',baseline_names=[backend])
                except runner.BaselineCorrectnessFailure as exc:
                    baseline_rows=exc.rows
                stream.synchronize()
            for result in baseline_rows:
                if 'timing' in result:
                    result['validation_timing_diagnostic']=result.pop('timing')
            row['baselines']=baseline_rows
            result=baseline_rows[0]
            row['status']=result['status']
            verified=result['status'] in ('pass','accuracy-review-required')
            protocol=result.get('validation_timing_diagnostic',{}).get('protocol','')
            graph_covered=verified and 'CUDA graph' in protocol
            row['graph_status']=('direct-and-graph-replay-verified' if graph_covered else
                'not-covered-by-recorded-protocol' if verified else 'not-completed')
            if verified and not graph_covered:
                row['status']='graph-validation-gap'
        if busy('after-case',token):
            return 76
        campaign.write(output/'rows'/(identifier+'.json'),dict(identity=identity,identity_sha256=token,
            row=row,row_sha256=campaign.digest(row),guards=dict(before=True,after=True)))
        pending.remove(identifier)
        campaign.write(output/'pending.json',dict(identity_sha256=token,pending=pending))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_directory',type=Path)
    parser.add_argument('--backend',choices=sorted(BACKENDS),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    return validate(args.prepared_directory,args.backend,args.output)


if __name__=='__main__':
    raise SystemExit(main())
