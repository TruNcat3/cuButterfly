#!/usr/bin/env python3
"""Render bounded, fixed-core attribution from frozen, independently scoped controls."""
from pathlib import Path
import csv, json, hashlib, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from paper_plot_style import RC

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'
OUTPUT=ROOT/'output'
SOURCES=['inputs/motivation_organization_a100.csv','inputs/principle_core_organization.csv','inputs/resource_decisions_20261107.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def gm(v): return math.exp(statistics.mean(math.log(x) for x in v))
def main():
    rows=[]
    raw=list(csv.DictReader((ROOT/SOURCES[0]).open()))
    assert len(raw)==162
    grouped={}
    for r in raw: grouped.setdefault((r['case'],r['variant']),[]).append(r)
    for (case,variant),rs in grouped.items():
        if variant=='local0': continue
        bs={int(r['trial']):float(r['kernel_ms']) for r in grouped[(case,'local0')]}
        vs={int(r['trial']):float(r['kernel_ms']) for r in rs}
        ratio=statistics.median(bs.values())/statistics.median(vs.values())
        assert len(bs)==len(vs)==3
        for trial in sorted(bs):
            rows.append(dict(panel='a',cohort='M1-A10040',operator=rs[0]['operator'],core='scalar-radix2',variant=variant,case=case,process=trial,baseline_ms=bs[trial],variant_ms=vs[trial],ratio=bs[trial]/vs[trial],case_ratio=ratio,kind='paired-process'))
    f=list(csv.DictReader((ROOT/SOURCES[1]).open()))
    assert len(f)==12
    for r in f:
        for trial,ratio in enumerate(json.loads(r['trial_ratios'])):
            rows.append(dict(panel='b',cohort='FFT-Sep17-A10080',operator='fft',core=r['core'],variant='resident-prefix',case=f"N{r['N']}-B{r['B']}",process=trial,baseline_ms='',variant_ms='',ratio=ratio,case_ratio=float(r['ratio']),kind='paired-process'))
    resource=json.loads((ROOT/SOURCES[2]).read_text())['prospective_measured_selection']
    identity=[]
    for r in resource['rows']:
        if r['workload']['operator']=='ntt':
            assert r['same_winner'] and r['choices']['conventional']['mapping']==r['choices']['extended']['mapping']
            identity.append(dict(core=r['leaf'],same_winner=True,ratio=1.0,interpretation='identical mapping measured once; no gain claim'))
            continue
        assert r['workload']['operator']=='fwht'
        for p in r['holdout_pairs']:
            rows.append(dict(panel='b',cohort='FWHT-Nov7-A10080',operator='fwht',core=r['leaf'],variant='matched-tuning-extended',case='N65536-B16',process=p['process'],baseline_ms=p['conventional_ms'],variant_ms=p['extended_ms'],ratio=p['ratio'],case_ratio=r['median_paired_ratio'],kind='paired-process'))
    assert len(rows)==150
    plt.rcParams.update(RC)
    fig=plt.figure(figsize=(3.45,1.32))
    axs=[fig.add_axes([.125,.31,.355,.46]),fig.add_axes([.52,.31,.465,.46])]
    colors=['#0072B2','#D55E00']; markers=['o','^']
    def points(ax,x,selected,color,marker):
        cases=list(dict.fromkeys(r['case'] for r in selected))
        med=[next(r['case_ratio'] for r in selected if r['case']==c) for c in cases]
        vals=[r['ratio'] for r in selected]
        ax.scatter(x+np.linspace(-.09,.09,len(vals)),vals,s=5,color=color,alpha=.30,linewidths=0,zorder=2)
        if len(med)>1:
            ax.boxplot([med],positions=[x],widths=.23,whis=(0,100),showfliers=False,manage_ticks=False,patch_artist=True,boxprops=dict(facecolor='white',edgecolor=color,linewidth=.8),medianprops=dict(color=color,linewidth=1),whiskerprops=dict(color=color,linewidth=.7),capprops=dict(color=color,linewidth=.7))
        ax.scatter(x+np.linspace(-.055,.055,len(med)),med,s=14,marker=marker,edgecolors=color,facecolors='white',linewidths=.7,zorder=4)
    for i,op in enumerate(('fft','fwht','ntt')):
        for j,v in enumerate(('local2','local4')):
            points(axs[0],i+(-.16 if j==0 else .16),[r for r in rows if r['panel']=='a' and r['operator']==op and r['variant']==v],colors[j],markers[j])
    axs[0].set_xticks(range(3),['FFT\nFP64','FWHT\nFP64','NTT\nword64'],fontsize=6.5)
    axs[0].set_xlim(-.55,2.55)
    fig.text(.125,.97,'(a) Fixed-kernel grouping',fontsize=7.2,va='top')
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([],[],color=colors[j],marker=markers[j],markerfacecolor='white',ls='none',label=v) for j,v in enumerate(('local2','local4'))],loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,fontsize=6.5,frameon=False,handletextpad=.25,columnspacing=1.2,borderpad=0)
    groups=[('fft','native'),('fft','cufftdx-thread'),('fwht','radix2'),('fwht','radix4')]
    for i,(op,core) in enumerate(groups):
        selected=[r for r in rows if r['panel']=='b' and r['operator']==op and (r['core']==core or r['core']=='fwht-'+core)]
        if not selected:
            selected=[r for r in rows if r['panel']=='b' and r['operator']==op and core in r['core']]
        assert selected,(op,core)
        points(axs[1],i,selected,colors[0 if op=='fft' else 1],'D')
    axs[1].set_xticks(range(4),['FFT\nregister','FFT\ncuFFTDx\nThread','FWHT\nradix-2','FWHT\nradix-4'],fontsize=6.5)
    axs[1].set_xlim(-.35,3.35)
    axs[1].axvline(1.5,color='#aaaaaa',ls=':',lw=.7)
    fig.text(.52,.97,'(b) Across kernels',fontsize=7.2,va='top')

    for ax in axs:
        ax.axhline(1,color='#555555',ls='--',lw=.7);ax.set_ylim(.50,1.74);ax.set_yticks([.5,1,1.5]);ax.tick_params(axis='y',labelsize=6.5)
        ax.set_ylabel('Speedup (x)' if ax is axs[0] else '',fontsize=7)
        if ax is axs[1]: ax.set_yticklabels([])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    prefix=OUTPUT/'cross_operator_attribution'
    for ext in ('pdf','svg','png'): fig.savefig(prefix.with_suffix('.'+ext),dpi=300,facecolor='white')
    out=OUTPUT/'cross_operator_attribution.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    meta=dict(schema='cross-operator-attribution-v1',dimensions_inches=[3.45,1.32],shared_y_limits=[0.50,1.74],axis_scope='Common point/box scale includes parity and every raw ratio and case summary',minimum_font_points=6.5,sources={s:sha(ROOT/s) for s in SOURCES},rows=len(rows),panel_a=dict(raw_paired_ratios=108,case_summaries=36,estimator='median(local0 three process times) / median(variant three process times); faint raw paired trial ratios',hardware='A100 40GB',scope='scalar radix2 fixed; N2^16/2^20 B1/4/16; local2/local4 both retained including losses'),panel_b=dict(fft_raw_paired_ratios=36,fft_cases_per_core=6,fft_estimator='source ratio of cell medians and all source paired trial ratios',fwht_raw_paired_ratios=6,fwht_estimator='median of three process-paired ratios; same leaf within each arm, geometry may change',hardware='separate A10080 cohorts',tuning='FFT fixed four cells; FWHT matched conventional and extended pools'),ntt_identity=identity,exclusions=['No cuFFT internal core ablation: cuFFTDx Thread is an exposed local implementation, distinct from cuFFT.','M3 forced FWHT c4 / NTT dt2 negative controls remain in CORE_CROSSING_RESULTS_20261027.md; different cohort, not pooled here.','No performance inference for unmeasured hardware, arbitrary hierarchy depth, or no-gain NTT endpoint identity.'],outputs={str(prefix.with_suffix('.'+ext).relative_to(OUTPUT)):sha(prefix.with_suffix('.'+ext)) for ext in ('pdf','svg','png')})
    (OUTPUT/'cross_operator_attribution.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),outputs=str(prefix))))
if __name__=='__main__':main()
