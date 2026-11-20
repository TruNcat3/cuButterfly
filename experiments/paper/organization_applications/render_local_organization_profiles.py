#!/usr/bin/env python3
"""Render frozen M1 resource diagnostics; kernel spans are not trial uncertainty."""
from pathlib import Path
import hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper_plot_style import RC
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'
OUTPUT=ROOT/'output'
SOURCE=ROOT/'inputs/narrative_scaling_20261027.json'
METRICS=[('launch__registers_per_thread','Registers' ),('sm__warps_active.avg.pct_of_peak_sustained_active','Active warps (%)'),('smsp__warp_issue_stalled_barrier_per_warp_active.pct','Barrier stalls (%)')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=json.loads(SOURCE.read_text()); rows=d['profile_records']
 expected={(op,v,k) for op in ('fft','fwht','ntt') for v in ('local0','local2','local4') for k in (0,1)}
 assert len(rows)==18 and {(r['operator'],r['variant'],r['group']) for r in rows}==expected
 for r in rows:
  assert all(math.isfinite(r['metrics'][m]) and r['metrics'][m]>=0 for m,_ in METRICS)
 with plt.rc_context(RC):
  fig,axs=plt.subplots(1,3,figsize=(3.45,1.45))
  fig.subplots_adjust(left=.095,right=.985,bottom=.16,top=.67,wspace=.48)
  colors=['#555555','#0072B2','#D55E00'];marks=['o','s','^']
  for ax,(metric,title) in zip(axs,METRICS):
   for i,op in enumerate(('fft','fwht','ntt')):
    for j,var in enumerate(('local0','local2','local4')):
     vals=[r['metrics'][metric] for r in rows if r['operator']==op and r['variant']==var]
     x=i+(j-1)*.20
     ax.plot([x,x],[min(vals),max(vals)],color=colors[j],lw=.9,zorder=2)
     for k,val in enumerate(vals):
      ax.scatter(x+(-.025 if k==0 else .025),val,s=12,marker=marks[j],edgecolor=colors[j],facecolor='white' if k==0 else colors[j],linewidth=.65,zorder=3)
   ax.set_title(title,fontsize=7.2,pad=4);ax.set_xticks(range(3),['FFT','FWHT','NTT']);ax.tick_params(axis='both',labelsize=6.5,length=2,pad=2)
   ax.set_xlim(-.35,2.35);ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  axs[0].set_yticks([0,50,100,150]);axs[1].set_yticks([0,50,100]);axs[2].set_yticks([0,25,50]);axs[2].set_ylim(0,60)
  fig.legend(handles=[Line2D([],[],color=c,marker=m,markerfacecolor='white',ls='none',label=v,markersize=3.5) for c,m,v in zip(colors,marks,('local0','local2','local4'))],loc='upper center',bbox_to_anchor=(.5,1.015),ncol=3,frameon=False,fontsize=6.5,columnspacing=1,handletextpad=.3)
  fig.legend(handles=[Line2D([],[],color='#555555',marker='o',markerfacecolor=fill,ls='none',label=label,markersize=3) for fill,label in [('white','Kernel 0'),('#555555','Kernel 1')]],loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,fontsize=6.5,columnspacing=1,handletextpad=.3)
  prefix=OUTPUT/'local_organization_profiles'
  for ext in ('pdf','svg','png'):fig.savefig(prefix.with_suffix('.'+ext),dpi=350,facecolor='white')
 metadata=dict(schema='local-organization-profile-figure-v1',source=str(SOURCE.relative_to(ROOT)),source_sha256=sha(SOURCE),renderer_sha256=sha(Path(__file__)),size_inches=[3.45,1.45],minimum_font_points=6.5,records=18,profile_records=rows,metrics=[dict(name=m,title=t) for m,t in METRICS],scope=d.get('profile_scope'),source_profile_statistic=d.get('profile_statistic'),statistic='All two kernel records per operator/organization plotted, connected by observed min-max; neither trial uncertainty nor confidence interval. Open/filled markers distinguish the two kernels.',limitations=['Profiler diagnostics only, not ordinary latency ratios.','Barrier stalls are per active warp and not a fraction of total elapsed time.','Active warps report the named NCU activity metric, not theoretical occupancy.','Independent A10040 M1 cohort, no cross-device aggregation.'],outputs={str(prefix.with_suffix('.'+e).relative_to(OUTPUT)):sha(prefix.with_suffix('.'+e)) for e in ('pdf','svg','png')})
 (OUTPUT/'local_organization_profiles.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print(json.dumps(dict(records=18,size_inches=[3.45,1.45])))
if __name__=='__main__':main()
