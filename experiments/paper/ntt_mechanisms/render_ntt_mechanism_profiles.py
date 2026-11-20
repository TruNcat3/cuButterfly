#!/usr/bin/env python3
"""Frozen A100 fixed-primitive and NTT local-layout controls."""
import argparse, csv, hashlib, json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from paper_plot_style import RC
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/ntt_fixed_core_a100_20261118/analysis.json'
LOCAL_BANK=ROOT/'data/ntt_local_bank_control_a100_20261119_strong/analysis.json'
STEM='ntt_mechanism_profiles_20261118'
CORES=['native','gpuntt','phantom']; NAMES=['cuButterfly','GPU-NTT','PhantomFHE']
COLORS=['#0072B2','#D55E00','#009E73']; HATCH=['','///','xx']; MARKERS=['o','s','^']
ACTIVE='sm__warps_active.avg.pct_of_peak_sustained_active'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 global SOURCE,LOCAL_BANK
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--fixed-analysis',type=Path,default=SOURCE,help='measured fixed-core analysis JSON for panels a-d')
 parser.add_argument('--layout-analysis',type=Path,default=LOCAL_BANK,help='measured local-layout analysis JSON for panels e-f')
 parser.add_argument('--output-dir',type=Path,default=ROOT/'output')
 args=parser.parse_args(); SOURCE=args.fixed_analysis.resolve();LOCAL_BANK=args.layout_analysis.resolve(); output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
 data=json.loads(SOURCE.read_text());bank=json.loads(LOCAL_BANK.read_text());assert data['status']=='complete' and bank['status']=='complete'
 rows=[]
 def add(panel,kind,value,**kw):rows.append(dict(panel=panel,kind=kind,value=value,**kw))
 def ordinary(core,local,org):
  a=sorted([r for r in data['ordinary'] if r['task']['core']==core and r['task']['local_log']==local and r['task']['organization']==org],key=lambda r:r['task']['process'])
  assert len(a)==3 and all(r['status']=='complete' for r in a);return [r['process_median_us'] for r in a]
 def profile(core,local,org):
  a=next(r for r in data['profiles'] if r['task']['core']==core and r['task']['local_log']==local and r['task']['organization']==org);assert a['status']=='complete';return a
 with plt.rc_context({**RC,'font.size':6.5,'axes.labelsize':6.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none'}):
  fig=plt.figure(figsize=(3.45,3.02))
  axes=[fig.add_axes([x,y,.345,.185]) for y in [.705,.405,.055] for x in [.13,.64]]
  titles=['NTT organization gain','NTT DRAM traffic','NTT resident latency','NTT prefix concurrency','NTT local layout trade-off','NTT conflict balance']
  for i,ax in enumerate(axes):
   ax.set_title(f'({chr(97+i)}) {titles[i]}',loc='left',fontsize=7,pad=3)
   ax.tick_params(length=2,pad=1.5);ax.grid(axis='y',color='.9',lw=.4);ax.set_axisbelow(True)
  a,b,c,d,e,f=axes
  def bar(ax,x,v,k,points=None):
   ax.bar(x,v,.24,color=COLORS[k],hatch=HATCH[k],edgecolor='white',linewidth=.3,zorder=3)
   if points is not None:
    ax.vlines(x,min(points),max(points),color='black',lw=.6,zorder=4)
    ax.scatter([x-.045,x,x+.045],points,s=6,marker=MARKERS[k],facecolors='white',edgecolors='black',lw=.4,zorder=5)
  for k,core in enumerate(CORES):
   base=ordinary(core,8,'resident-prefix')
   for j,local in enumerate([8,10]):
    x=j+(k-1)*.26
    split=ordinary(core,local,'split-prefix');resident=ordinary(core,local,'resident-prefix')
    value=median(split)/median(resident);pts=[u/v for u,v in zip(split,resident)];bar(a,x,value,k,pts)
    add('a','summary',value,core=core,prefix_points=1<<local,denominator='resident complete-call us',numerator='split complete-call us',estimator='ratio of process medians')
    for proc,(u,v,z) in enumerate(zip(split,resident,pts)):add('a','process',z,core=core,prefix_points=1<<local,process=proc,numerator=u,denominator=v,unit='speedup')
    old=profile(core,local,'split-prefix');new=profile(core,local,'resident-prefix')
    value=100*(1-new['dram_read_write_bytes']/old['dram_read_write_bytes']);bar(b,x,value,k)
    add('b','profile',value,core=core,prefix_points=1<<local,numerator=new['dram_read_write_bytes'],denominator=old['dram_read_write_bytes'],unit='percent reduction',metric='complete-call DRAM read+write bytes')
    value=median(resident)/median(base);pts=[u/v for u,v in zip(resident,base)];bar(c,x,value,k,pts)
    add('c','summary',value,core=core,prefix_points=1<<local,denominator=median(base),numerator=median(resident),estimator='ratio of process medians',unit='latency / resident256')
    for proc,(u,v,z) in enumerate(zip(resident,base,pts)):add('c','process',z,core=core,prefix_points=1<<local,process=proc,numerator=u,denominator=v,unit='latency / resident256')
    node=next(n for n in new['nodes'] if str(n['id'])=='0');value=node['metrics'][ACTIVE];bar(d,x,value,k)
    add('d','profile',value,core=core,prefix_points=1<<local,kernel_id=0,metric=ACTIVE,unit='percent peak active warps',denominator='peak sustained active warps')
  for ax in [a,b,c,d]:ax.set(xticks=[0,1],xticklabels=['256','1024'],xlim=(-.53,1.53));ax.set_xlabel('Prefix points',labelpad=0)
  a.set(ylim=(0,3.5),yticks=[0,1,2,3],ylabel='Split / resident latency')
  b.set(ylim=(0,70),yticks=[0,30,60],ylabel='DRAM reduction (%)')
  c.set(ylim=(0,1.5),yticks=[0,.5,1,1.5],ylabel='Relative latency')
  d.set(ylim=(0,100),yticks=[0,50,100],ylabel='Active warps (%)')
  for ax in [a,c]:ax.axhline(1,color='.35',ls=':',lw=.65,zorder=4)
  for i,request in enumerate(bank['requests']):
   batch=request['batch']; log_n=request['logN']
   linear=request['layouts']['linear']; xor=request['layouts']['hermes-xor']
   baseline=linear['median_kernel_ms']
   value=xor['median_kernel_ms']/baseline
   add('e','summary',value,logN=log_n,batch=batch,stage_partition=request['partition'],numerator=xor['median_kernel_ms'],denominator=baseline,estimator='ratio of layout process medians',unit='XOR / linear latency')
   for k,(layout,record) in enumerate([('linear',linear),('hermes-xor',xor)]):
    pts=[v/baseline for v in record['process_kernel_ms']]
    x=i+(k-.5)*.32
    color=['#444444','#7B3294'][k];marker=['o','D'][k]
    e.vlines(x,min(pts),max(pts),color=color,lw=.8,zorder=3)
    e.scatter([x-.055,x,x+.055],pts,s=9,marker=marker,facecolors='white',edgecolors=color,lw=.6,zorder=4)
    e.plot([x-.10,x+.10],[median(pts)]*2,color=color,lw=1.4,zorder=5)
    add('e','layout_summary',median(pts),logN=log_n,batch=batch,layout=layout,numerator=record['median_kernel_ms'],denominator=baseline,minimum=min(pts),maximum=max(pts),estimator='layout median / linear median; range of independent processes',unit='latency / linear median')
    for proc,(raw,v) in enumerate(zip(record['process_kernel_ms'],pts)):add('e','process',v,logN=log_n,batch=batch,layout=layout,process=proc,numerator=raw,denominator=baseline,unit='latency / linear median')
   for j,op in enumerate(['load','store']):
    metric=f'shared_{op}_conflicts'
    old=linear['profile'][metric];new=xor['profile'][metric]
    value=100*(1-new/old)
    f.bar(j+(i-.5)*.32,value,.29,color='#7B3294',hatch='////' if i else '',edgecolor='white',linewidth=.3,zorder=3)
    add('f','profile',value,logN=log_n,batch=batch,kernel_ids=xor['profile']['kernel_ids'],linear_kernel_ids=linear['profile']['kernel_ids'],profile_scope='sum of final two stage kernels covering complete NTT',metric=metric,numerator=new,denominator=old,unit='conflict reduction (+) / increase (-) percent')
  e.set(ylim=(.96,1.20),yticks=[1,1.1,1.2],ylabel='Latency / linear median',xticks=[0,1],xticklabels=['B4','B16'],xlim=(-.5,1.5));e.axhline(1,color='.35',ls=':',lw=.65)
  e.legend(handles=[Line2D([],[],color=co,marker=ma,markerfacecolor='white',markersize=2.5,lw=.8,label=la) for co,ma,la in [('#444444','o','Linear'),('#7B3294','D','XOR')]],loc='upper left',bbox_to_anchor=(0,1.03),ncol=2,fontsize=6.5,handlelength=.7,columnspacing=.5,handletextpad=.25)
  f.set(ylim=(-40,110),yticks=[-25,0,50,100],ylabel='Conflict change (%)',xticks=[0,1],xticklabels=['Load','Store']);f.axhline(0,color='.35',ls=':',lw=.65)
  f.legend(handles=[Patch(facecolor='#7B3294',hatch=ha,edgecolor='white',label=la) for ha,la in [('', 'B4'),('////','B16')]],loc='upper right',ncol=2,fontsize=6.5,handlelength=.8,columnspacing=.5)
  fig.legend(handles=[Patch(facecolor=co,hatch=ha,edgecolor='white',label=na) for co,ha,na in zip(COLORS,HATCH,NAMES)],loc='upper center',bbox_to_anchor=(.53,1.005),ncol=3,frameon=False,handlelength=.9,columnspacing=.8)
  for ext in ['pdf','png','svg']:fig.savefig(output/f'{STEM}.{ext}',dpi=300,facecolor='white')
  plt.close(fig)
 csvpath=output/f'{STEM}.csv'
 with csvpath.open('w',newline='') as out:
  writer=csv.DictWriter(out,sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
 manifest=dict(dimensions_inches=[3.45,3.02],minimum_text_pt=6.5,source_hashes={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name):sha(p) for p in [SOURCE,LOCAL_BANK]},renderer_sha256=sha(Path(__file__)),chart_csv_sha256=sha(csvpath),panels=dict(zip('abcdef',titles)),selection='Panels a-d use three fixed NTT arithmetic primitives and two stage organizations; panels e-f use fixed NTT radix2-shoup shared-iterative local-layout controls at two batch sizes.',scope='A100; NTT word64 q576460756061519873, natural cyclic forward. Panels a-d use N=2^16,B=16 and fixed prefixes 256/1024; panels e-f use N=2^20, B=4/16, partition 10+10, and 256 threads. Only dataflow_layout changes in e-f; global I/O, stage partition, threads, arithmetic, and storage allocation remain fixed.',estimators='a ratio of three process medians; c same-provider resident median normalized to resident256 median; overlaid points/minmax use corresponding process pairs, not confidence intervals. e ratio of layout process medians; each layout point is its ordinary device latency divided by that batch linear median, with median and min/max; processes are independent, not recorded pairs. b complete-call DRAM byte reduction; d prefix node0 only; f sum of final two stage-kernel conflict counters from separate complete-NTT NCU captures, positive for reduction and negative for increase.',uncertainty='e three independent timing processes per layout and batch (12 total); ranges show observed process spread, not confidence intervals. NCU sums two final stage kernels per configuration in separate captures; clocks/cache uncontrolled. No profiler time enters performance.',counts=dict(csv_rows=len(rows),ntt_organization_processes=36,ntt_organization_profiles=12,local_layout_processes=12,local_layout_indexwise_diagnostic_ratios=6,local_layout_profiles=4),alt_text='Six compact panels show NTT organization gains and resource costs. The lower controls compare linear and XOR shared layouts within one fixed NTT core at N=2^20 and B=4/16: XOR increases complete-call latency by about 4.7–7.4 percent, reduces shared-load conflicts by about 87 percent, and increases shared-store conflicts by about 13–14 percent.')
 (output/f'{STEM}.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(output/f'{STEM}.png')
if __name__=='__main__':main()
