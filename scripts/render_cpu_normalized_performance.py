#!/usr/bin/env python3
"""Render CPU-normalized performance boxplots from frozen paired timings."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path('paper/generated/current_refresh/cpu_normalized.json')
DEFAULT_PREFIX = Path('figures/cpu_normalized_performance')
GPU_ORDER = ['cuButterfly','cufft','vkfft','daofht','gpu-ntt']

def finite_pos(x):
    try: return math.isfinite(float(x)) and float(x) > 0
    except Exception: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output-prefix', type=Path, default=DEFAULT_PREFIX)
    ap.add_argument('--fftw-threads', type=int, choices=[1,8], default=8)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    if data.get('schema') != 'cpu-normalized-performance-v1':
        raise ValueError('input schema must be cpu-normalized-performance-v1')
    records = data.get('records', [])
    if not isinstance(records, list): raise ValueError('records must be a list')
    seen = set(); rows=[]
    for r in records:
        if not isinstance(r, dict): raise ValueError('record must be object')
        required=['case_id','operator','precision','backend','cpu_us','backend_us']
        miss=[k for k in required if k not in r]
        if miss: raise ValueError(f'record missing {miss}')
        key=(r['case_id'],r['backend'])
        if key in seen: raise ValueError(f'duplicate case/backend: {key}')
        seen.add(key)
        if not finite_pos(r['cpu_us']) or not finite_pos(r['backend_us']):
            raise ValueError(f'non-positive/non-finite timing: {key}')
        ratio=float(r.get('speedup', float(r['cpu_us'])/float(r['backend_us'])))
        expected=float(r['cpu_us'])/float(r['backend_us'])
        if not finite_pos(ratio) or abs(ratio-expected)>max(1e-8,1e-6*expected):
            raise ValueError(f'invalid speedup for {key}')
        rr=dict(r); rr['speedup']=ratio
        # FFT CPU denominator is explicitly selected by thread view.
        cb=str(rr.get('cpu_backend',''))
        if rr['operator'].lower() == 'fft' and cb and cb != f'fftw-cpu-t{args.fftw_threads}':
            continue
        rows.append(rr)
    if not rows: raise ValueError('no records remain after CPU thread selection')
    groups=defaultdict(list)
    for r in rows: groups[(str(r['operator']),str(r['precision']))].append(r)
    # A family uses one explicitly named CPU denominator; never select a
    # per-case fastest CPU implementation.
    for gkey, rs in groups.items():
        cpu_backends={str(r.get('cpu_backend','')) for r in rs}
        if len(cpu_backends) > 1:
            raise ValueError(f'CPU denominator varies within family {gkey}: {sorted(cpu_backends)}')
    plot_rows=[]; audit=[]
    for gkey, rs in sorted(groups.items()):
        by_backend=defaultdict(dict)
        for r in rs: by_backend[r['backend']][r['case_id']]=r
        own=set(by_backend.get('cuButterfly',{}))
        ext=set().union(*(set(v) for k,v in by_backend.items() if k!='cuButterfly'))
        inter=own & ext
        # external comparison intersection requires each included backend to have exact IDs.
        participating=[b for b in GPU_ORDER if b in by_backend]
        common=set(own)
        for b in participating:
            common &= set(by_backend[b])
        common &= inter
        if common:
            for b in participating:
                vals=[by_backend[b][c]['speedup'] for c in sorted(common)]
                plot_rows.append((gkey,'common',b,vals,sorted(common)))
            audit.append({'group':gkey,'panel':'common','case_ids':sorted(common),'backends':participating})
        own_only=sorted(own-common)
        if own_only:
            vals=[by_backend['cuButterfly'][c]['speedup'] for c in own_only]
            plot_rows.append((gkey,'cuButterfly-only','cuButterfly',vals,own_only))
            audit.append({'group':gkey,'panel':'cuButterfly-only','case_ids':own_only,'backends':['cuButterfly']})
    if not plot_rows: raise ValueError('no plottable groups')
    panels=[('common',[x for x in plot_rows if x[1]=='common']),('cuButterfly-only',[x for x in plot_rows if x[1]=='cuButterfly-only'])]
    panels=[x for x in panels if x[1]]
    fig, axes=plt.subplots(1,len(panels),sharey=True,figsize=(max(10,sum(len(x[1]) for x in panels)*1.25),6.5),squeeze=False)
    axes=axes[0]
    for ax,(panel_name,prows) in zip(axes,panels):
        positions=list(range(1,len(prows)+1)); labels=[]
        for pos,(g,p,b,vals,cases) in zip(positions,prows):
            ax.boxplot(vals,positions=[pos],widths=.65,patch_artist=True,showfliers=False,whis=(0,100),
                       boxprops={'facecolor':'#4c78a8' if b=='cuButterfly' else '#bdbdbd','alpha':.8},
                       medianprops={'color':'black','linewidth':1.4}, flierprops={'marker':'o','markersize':3,'alpha':.45})
            # Show every request so the box never hides the finite workload
            # population; deterministic jitter only separates coincident values.
            import numpy as np
            jitter = np.random.default_rng(410 + pos).uniform(-.17, .17, len(vals))
            ax.scatter([pos + j for j in jitter], vals, s=13,
                       color='#1f5f9d' if b == 'cuButterfly' else '#666666',
                       edgecolors='white', linewidths=.35, zorder=3)
            labels.append(f'{g[0]}\n{g[1]}\n{b}\nn={len(cases)}')
        ax.axhline(1,color='black',linestyle='--',linewidth=1)
        ax.set_yscale('log'); ax.set_xticks(positions); ax.set_xticklabels(labels,rotation=45,ha='right'); ax.grid(axis='y',which='both',alpha=.2)
        ax.set_title('Shared request intersection' if panel_name=='common' else 'cuButterfly requests without complete external comparison')
    axes[0].set_ylabel('Speedup relative to CPU (CPU time / backend time)')
    fig.suptitle('CPU-normalized performance (paired requests)')
    fig.text(0.01,0.038,'CPU denominator: FFTW-t8 (FFT), FXT-t1 (FWHT/zeta), HEXL-t1 (NTT), author Butterfly-t1 (2x2).',fontsize=8)
    fig.text(0.01,0.012,'Provisional shared-host CPU timings; CPU wall/device-resident GPU timing; host-device transfers excluded.',fontsize=8)
    fig.text(0.99,0.985,'blue: cuButterfly   gray: external GPU',ha='right',va='top',fontsize=8)
    fig.tight_layout()
    prefix=args.output_prefix; prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','png','pdf'):
        path = prefix.with_suffix('.'+ext)
        fig.savefig(path,dpi=220)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    csvpath=prefix.with_suffix('.csv')
    with csvpath.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['operator','precision','panel','backend','case_id','speedup'])
        for g,p,b,vals,cases in plot_rows:
            for c,v in zip(cases,vals): w.writerow([g[0],g[1],p,b,c,v])
    manifest={'schema':'cpu-normalized-performance-render-v1',
              'input':str(args.input),
              'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'fftw_threads':args.fftw_threads,
              'qualification':'shared-host provisional CPU timings; CPU/GPU timing boundaries are separate',
              'cpu_denominators':{'fft':'fftw-cpu-t8','fwht':'fxt-cpu-t1','ntt':'hexl-cpu-t1',
                                  'subset-zeta':'fxt-cpu-t1','superset-zeta':'fxt-cpu-t1',
                                  'structured-2x2':'butterfly-author-cpu-t1'},
              'records':len(records),'groups':audit,'outputs':{}}
    for ext in ('svg','png','pdf','csv'):
        p=prefix.with_suffix('.'+ext); manifest['outputs'][ext]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    prefix.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
