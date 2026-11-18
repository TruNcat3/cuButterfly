#!/usr/bin/env python3
"""One original GPU-NTT complete call on a frozen cyclic NTT8 failing request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measured-root',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--case-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',default='GPU-a257f4db-8c62-f26b-9c08-a4045615a140')
    a=p.parse_args()
    root=a.measured_root.resolve();sys.path[:0]=[str(root/'scripts'),str(root/'python')]
    import numpy as np
    import run_semantic_campaign as c
    import run_comparator_extension as ext
    import semantic_gpuntt_baseline as gpu_ntt
    m,loaded=ext.load_prepared(a.preparation)
    case=next(case for case in m['config']['cases'] if case['id']==a.case_id)
    s=case['descriptor']['semantics'];assert s['storage']==9 and not s['inverse']
    payload=loaded[a.case_id]
    hashes=json.loads((a.preparation/'input-hashes.json').read_text())[a.case_id]
    assert hashlib.sha256(payload['source'].tobytes()).hexdigest()==hashes['input_sha256']
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid;c.exclusive_gpu(a.gpu_uuid)
    import torch
    stream=torch.cuda.Stream();core=None
    with torch.cuda.stream(stream):
        src=torch.from_numpy(np.ascontiguousarray(payload['source'])).to('cuda')
        saved=src.clone();dst=torch.empty_like(src)
        try:
            core=gpu_ntt.build_core(torch,s,src,dst,stream,m['config']['gpuntt_library'])
            core();stream.synchronize()
            actual=dst.cpu().numpy();expected=payload['expected']
            mismatches=np.flatnonzero(actual.reshape(-1)!=expected.reshape(-1))
            result=dict(case_id=a.case_id,semantics=s,input_sha256=hashes['input_sha256'],
                reference_sha256=hashes['reference_sha256'],gpu_uuid=a.gpu_uuid,
                bridge_sha256=c.sha256_file(m['config']['gpuntt_library']),
                runner_sha256=c.sha256_file(__file__),metadata=core.metadata(),
                input_preserved=torch.equal(src,saved),exact=not len(mismatches),
                mismatches=len(mismatches),first_bad=int(mismatches[0]) if len(mismatches) else None,
                first_actual=int(actual.reshape(-1)[mismatches[0]]) if len(mismatches) else None,
                first_expected=int(expected.reshape(-1)[mismatches[0]]) if len(mismatches) else None,
                output=c.retain_error_artifact(a.output.parent/'errors',a.case_id,'gpu-ntt-repro',actual,expected),
                protocol='one complete direct call; no performance timing and no search')
            c.exclusive_gpu(a.gpu_uuid)
            result['guards']=dict(before=True,after=True)
            ext.write(a.output,result)
            print(json.dumps({k:result[k] for k in ('case_id','exact','mismatches','first_bad')}),flush=True)
        finally:
            if core is not None:core.close()


if __name__=='__main__':main()
