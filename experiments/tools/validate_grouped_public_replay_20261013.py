#!/usr/bin/env python3
"""Validate automatic grouped xD RealFFT cache replay against an app mapping."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CACHE = ROOT / "results/e2e_fft_opt_20261013/poisson_2048_b16_t0/release/plans.cache"
SOURCE_REPORT = ROOT / "results/e2e_fft_opt_20261013/poisson_2048_b16_t0/release/periodic_poisson.json"

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def errors(torch, a, b):
    import numpy as np
    d=(a-b).abs()
    q=np.quantile(d.numpy(),[.5,.9,.99])
    return dict(max_abs=float(d.max()),relative_l2=float(torch.linalg.vector_norm((a-b).reshape(-1))/torch.linalg.vector_norm(b.reshape(-1))),p50_abs=float(q[0]),p90_abs=float(q[1]),p99_abs=float(q[2]),quantiles='full-array CPU numpy.quantile',elements=d.numel())

def main(argv=None):
    global SOURCE_CACHE, SOURCE_REPORT
    p=argparse.ArgumentParser(); p.add_argument('--library',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',type=int,default=0)
    p.add_argument('--source-cache',type=Path,default=SOURCE_CACHE)
    p.add_argument('--source-report',type=Path,default=SOURCE_REPORT)
    args=p.parse_args(argv)
    SOURCE_CACHE, SOURCE_REPORT = args.source_cache, args.source_report
    if not args.output.is_absolute() and os.environ.get('XD_REPORT_DIR'):
        args.output = Path(os.environ['XD_REPORT_DIR']) / args.output
    import torch
    sys.path.insert(0,str(ROOT/'python')); from cubutterfly.real_fft import RealFFT
    torch.cuda.set_device(args.device); device=torch.device('cuda',args.device); props=torch.cuda.get_device_properties(device)
    source=json.loads(SOURCE_REPORT.read_text()); expected={x['direction']:x for x in source['setup']['cubutterfly']['plans']}
    report=dict(schema='grouped-public-realfft-replay-20261013-v1',status='running',library=str(args.library.resolve()),library_sha256=digest(args.library),runner_sha256=digest(Path(__file__)),source_cache_sha256=digest(SOURCE_CACHE),source_report_sha256=digest(SOURCE_REPORT),gpu_uuid=str(props.uuid),gpu_name=props.name,cases=[])
    try:
      if source['library_sha256'] != digest(args.library):
        raise AssertionError('source application and replay library identities differ')
      with tempfile.TemporaryDirectory(prefix='grouped-replay-') as td:
        cache=Path(td)/'plans.cache'; shutil.copyfile(SOURCE_CACHE,cache)
        adapters=[RealFFT(str(args.library),(2048,2048),16,device,norm='backward',reuse_buffers=True,cache_path=str(cache)) for _ in range(2)]
        metas=[a.plan_metadata() for a in adapters]
        for meta in metas:
          for rec in meta:
            if rec['selection_source']!='user_cache': raise AssertionError('selection did not replay user cache')
        for direction,index in [('forward',0),('inverse',1)]:
          src=expected[direction]['mapping']['physical'];
          for meta in metas:
            got=meta[index]['mapping']['physical'];
            if got != src: raise AssertionError(f'{direction} mapping differs from source app mapping')
            if direction=='inverse' and not got.get('c2r_prefix_scale'): raise AssertionError('inverse replay lacks prefix scale')
        gen=torch.Generator(device=device).manual_seed(20261013); x=torch.randn((16,2048,2048),device=device,generator=gen); xf=x.cpu(); refs=torch.fft.rfftn(xf,dim=(-2,-1),norm='backward'); invref=torch.fft.irfftn(refs,s=(2048,2048),dim=(-2,-1),norm='backward')
        rows=[]; outputs=[]
        for adapter in adapters:
          inp=x.clone(); f=adapter.rfftn(inp); torch.cuda.synchronize(); f_before=f.cpu().clone()
          y=adapter.irfftn(f); torch.cuda.synchronize(); fc=f.cpu(); yc=y.cpu()
          if not torch.equal(inp,x): raise AssertionError('forward/inverse mutated input')
          if not torch.equal(fc,f_before): raise AssertionError('inverse mutated spectrum input')
          finite_forward=bool(torch.isfinite(fc).all()); finite_inverse=bool(torch.isfinite(yc).all())
          if not finite_forward or not finite_inverse: raise AssertionError('nonfinite forward/inverse output')
          outputs.append((fc,yc))
          rows.append(dict(forward=errors(torch,fc,refs),inverse=errors(torch,yc,invref),finite_forward=finite_forward,finite_inverse=finite_inverse,input_preserved=True,spectrum_preserved=True,forward_mapping=adapter.plan_metadata()[0],inverse_mapping=adapter.plan_metadata()[1]))
        report['two_adapters_exact_forward']=torch.equal(outputs[0][0],outputs[1][0]); report['two_adapters_exact_inverse']=torch.equal(outputs[0][1],outputs[1][1])
        if not report['two_adapters_exact_forward'] or not report['two_adapters_exact_inverse']: raise AssertionError('fresh adapters returned different outputs')
        report['source_cache_sha256_after']=digest(SOURCE_CACHE); report['source_report_sha256_after']=digest(SOURCE_REPORT)
        if report['source_cache_sha256_after']!=report['source_cache_sha256'] or report['source_report_sha256_after']!=report['source_report_sha256']: raise AssertionError('original source cache/report changed')
        report['cases']=rows; report['selection_sources']=[[m['selection_source'] for m in meta] for meta in metas]; report['status']='pass'
        for a in adapters: a.close()
    except Exception as e:
      report.update(status='failed',error_type=type(e).__name__,error=str(e))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'status':report['status'],'output':str(args.output)},indent=2)); return 0 if report['status']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())
