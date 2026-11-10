#!/usr/bin/env python3
"""Render CPU-normalized performance boxplots from frozen paired timings."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_INPUT = Path('experiments/snapshots/current_refresh/cpu_normalized.json')
DEFAULT_PREFIX = Path('figures/cpu_normalized_performance')
GPU_ORDER = ['cuButterfly','cufft','vkfft','daofht','gpu-ntt']
NAMES = {'cuButterfly': 'cuButterfly', 'cufft': 'cuFFT', 'vkfft': 'VkFFT',
         'daofht': 'DaoFHT', 'gpu-ntt': 'GPU-NTT'}
STYLES = {'cuButterfly': ('#0072B2', 'o', ''), 'cufft': ('#A34D00', 's', '//'),
          'vkfft': ('#007B59', '^', '..'), 'daofht': ('#AA3377', 'D', '\\\\'),
          'gpu-ntt': ('#555555', 'P', 'xx')}
GROUP_ORDER = [('fft', 'fp32'), ('fft', 'fp64'), ('fwht', 'fp32'), ('fwht', 'fp64'),
               ('ntt', 'word32'), ('ntt', 'word64'), ('subset-zeta', 'uint32'),
               ('subset-zeta', 'word64'), ('superset-zeta', 'uint32'),
               ('superset-zeta', 'word64'), ('structured-2x2', 'fp32'),
               ('structured-2x2', 'fp64')]
GROUP_LABELS = {('fft', 'fp32'): 'FFT-F32', ('fft', 'fp64'): 'FFT-F64',
                ('fwht', 'fp32'): 'FWHT-F32', ('fwht', 'fp64'): 'FWHT-F64',
                ('ntt', 'word32'): 'NTT-w32', ('ntt', 'word64'): 'NTT-w64',
                ('subset-zeta', 'uint32'): 'SZ-u32', ('subset-zeta', 'word64'): 'SZ-w64',
                ('superset-zeta', 'uint32'): 'PZ-u32', ('superset-zeta', 'word64'): 'PZ-w64',
                ('structured-2x2', 'fp32'): '2x2-F32', ('structured-2x2', 'fp64'): '2x2-F64'}

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
    by_group={}
    for gkey in GROUP_ORDER:
        rs=groups.get(gkey, [])
        by_backend=defaultdict(dict)
        for r in rs: by_backend[r['backend']][r['case_id']]=r
        if not by_backend: continue
        by_group[gkey]=by_backend
        audit.append({'group':gkey, 'panel':'single-axis',
                      'backends': [b for b in GPU_ORDER if b in by_backend],
                      'case_ids': {b: sorted(by_backend[b]) for b in GPU_ORDER if b in by_backend}})
        for backend in GPU_ORDER:
            if backend in by_backend:
                cases=sorted(by_backend[backend])
                plot_rows.append((gkey, backend, [by_backend[backend][c]['speedup'] for c in cases], cases))
    if not plot_rows: raise ValueError('no plottable groups')
    shown_groups=[g for g in GROUP_ORDER if g in by_group]
    fig, ax=plt.subplots(figsize=(9.4, 3.7))
    fig.subplots_adjust(left=.075, right=.995, bottom=.30, top=.72)
    all_values=[]
    for i,gkey in enumerate(shown_groups):
        series=[row for row in plot_rows if row[0] == gkey]
        offsets=np.linspace(-.30,.30,len(series)) if len(series)>1 else [0]
        notes=[]
        for offset,(group,backend,vals,cases) in zip(offsets,series):
            color,marker,hatch=STYLES[backend]
            all_values.extend(vals)
            box=ax.boxplot(vals, positions=[i+offset], widths=.16, patch_artist=True,
                           showfliers=False, whis=(0,100), manage_ticks=False,
                           boxprops={'edgecolor':color,'facecolor':'white','linewidth':.8},
                           medianprops={'color':'black','linewidth':.85},
                           whiskerprops={'color':color,'linewidth':.7},
                           capprops={'color':color,'linewidth':.7})
            box['boxes'][0].set_hatch(hatch)
            jitter=np.random.default_rng(410+i*7+len(backend)).uniform(-.035,.035,len(vals))
            ax.scatter(i+offset+jitter, vals, s=10, marker=marker, color=color,
                       edgecolors='white', linewidths=.25, zorder=3)
            gm=math.exp(sum(math.log(v) for v in vals)/len(vals))
            short={'cuButterfly':'B','cufft':'C','vkfft':'V','daofht':'D','gpu-ntt':'G'}[backend]
            notes.append(f'{short} {gm:.2f}x/{len(vals)}')
        ax.text(i, .985, '\n'.join(notes), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=5.8, linespacing=1.05)
    if not all_values: raise ValueError('no positive speedups to plot')
    ax.axhline(1,color='#333333',linestyle='--',linewidth=1)
    ax.set_yscale('log'); ax.set_ylim(min(all_values)/1.35,max(all_values)*1.5)
    ax.set_xlim(-.55,len(shown_groups)-.45)
    ax.set_ylabel('CPU-relative speedup (CPU time / backend time; log scale)', fontsize=7.5)
    ax.set_xticks(range(len(shown_groups)))
    ax.set_xticklabels([f'{GROUP_LABELS[g]}\n(n={len(by_group[g].get("cuButterfly",{}))})' for g in shown_groups], fontsize=6.5)
    ax.grid(axis='y',which='both',alpha=.25); ax.set_axisbelow(True)
    for boundary in [i-.5 for i in range(2,len(shown_groups)) if GROUP_LABELS[shown_groups[i-1]][:3] != GROUP_LABELS[shown_groups[i]][:3]]:
        ax.axvline(boundary,color='#AAAAAA',linestyle=':',linewidth=.7)
    group_titles=[(0,2,'FFT'),(2,4,'FWHT'),(4,6,'NTT'),(6,10,'Zeta'),(10,12,'Structured 2x2')]
    for left,right,title in group_titles:
        valid=[i for i,g in enumerate(shown_groups) if left<=GROUP_ORDER.index(g)<right]
        if valid:
            ax.text((min(valid)+max(valid))/2,-.30,title,transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=6.5,fontweight='bold')
    handles=[Line2D([],[],color=c,marker=m,linestyle='none',markersize=4,label=NAMES[n])
             for n,(c,m,_) in STYLES.items()]
    fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False,fontsize=7)
    fig.text(.5,.835,'CPU-normalized complete-call distributions; letters above each group give GM / n',ha='center',fontsize=6.2)
    fig.text(.01,.012,'CPU: FFTW-t8 (FFT), FXT-t1 (FWHT/zeta), HEXL-t1 (NTT), author Butterfly-t1 (2x2). Shared-host provisional; GPU transfers excluded.',fontsize=6.1)
    prefix=args.output_prefix; prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('svg','png','pdf'):
        path = prefix.with_suffix('.'+ext)
        fig.savefig(path,dpi=220)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    csvpath=prefix.with_suffix('.csv')
    with csvpath.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['operator', 'precision', 'panel', 'backend', 'case_id', 'speedup'])
        for g,b,vals,cases in plot_rows:
            for c,v in zip(cases,vals): w.writerow([g[0],g[1],'single-axis',b,c,v])
    manifest={'schema':'cpu-normalized-performance-render-v2',
              'input':str(args.input),
              'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'fftw_threads':args.fftw_threads,
              'qualification':'shared-host provisional CPU timings; CPU/GPU timing boundaries are separate',
              'axis_scale':'log',
              'cpu_denominators':{'fft':'fftw-cpu-t8','fwht':'fxt-cpu-t1','ntt':'hexl-cpu-t1',
                                  'subset-zeta':'fxt-cpu-t1','superset-zeta':'fxt-cpu-t1',
                                  'structured-2x2':'butterfly-author-cpu-t1'},
              'records':len(records),'groups':audit,'outputs':{}}
    for ext in ('svg','png','pdf','csv'):
        p=prefix.with_suffix('.'+ext); manifest['outputs'][ext]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    prefix.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
