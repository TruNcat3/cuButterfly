#!/usr/bin/env python3
"""Render fixed-core mechanism evidence from frozen observations, without a GPU.

Bars for performance use ratios of process medians. Points retain all three
within-process ratios. Profile percentages remain separate for each kernel.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PAPER = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#D55E00']
HATCHES = ['', '///']
CONFIGS = ['t1-d0', 't4-d0', 't4-d1', 't4-d2']
STALL = 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'
SHARED = 'launch__shared_mem_per_block_allocated'
ACTIVE = 'sm__warps_active.avg.pct_of_peak_sustained_active'
REGISTERS = 'launch__registers_per_thread'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=PAPER / 'snapshots/representative_profiles_20261108.json')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    assert len(data['tasks']) == 30
    assert all(t['status'] == 'complete' for t in data['tasks'] if t['kind'] in ('ordinary', 'ncu'))
    plotted, export = {}, []

    def samples(variant):
        rows = sorted([r for r in data['ordinary_samples'] if r['variant'] == variant], key=lambda r: r['process'])
        assert len(rows) == 3 and [r['process'] for r in rows] == [0, 1, 2]
        for r in rows:
            plotted['ordinary/' + r['task_id'] + '/' + variant] = dict(panel_source='ordinary', **r)
        return [r['time_ms'] for r in rows]

    def metrics(task, name):
        rows = sorted([r for r in data['ncu_kernel_metrics'] if r['task_id'] == task and r['metric'] == name], key=lambda r: int(r['kernel_id']))
        assert rows and all(r['value'] is not None for r in rows)
        for r in rows:
            plotted['ncu/' + task + '/' + r['kernel_id'] + '/' + name] = dict(panel_source='NCU diagnostic', **r)
        return rows

    def total(task, names):
        return sum(r['value'] for name in names for r in metrics(task, name))

    def ratio_bar(ax, x, baseline, variant, core, panel, width=.30):
        a, b = samples(baseline), samples(variant)
        value = median(a) / median(b)
        ratios = [first / second for first, second in zip(a, b)]
        ax.bar(x, value, width, color=COLORS[core], edgecolor='black', linewidth=.45, hatch=HATCHES[core], zorder=2)
        ax.vlines(x, min(ratios), max(ratios), color='black', linewidth=.65, zorder=4)
        ax.hlines([min(ratios), max(ratios)], x-.035, x+.035, color='black', linewidth=.65, zorder=4)
        ax.scatter([x-.055, x, x+.055], ratios, s=8, marker=['o', 's'][core], facecolor='white', edgecolor='black', linewidth=.45, zorder=5)
        ax.text(x, max(value, max(ratios)) + .045,
                f'{value:.2f}' if panel == 'd' else f'{value:.3f}',
                ha='center', va='bottom', fontsize=6.1,
                rotation=90 if panel == 'd' else 0)
        for process, (aa, bb, rr) in enumerate(zip(a, b, ratios)):
            export.append(dict(panel=panel, kind='ordinary', core=core, baseline=baseline, variant=variant, process=process,
                               baseline_ms=aa, variant_ms=bb, value=rr, unit='speedup', estimator='within-process ratio'))
        export.append(dict(panel=panel, kind='ordinary_summary', core=core, baseline=baseline, variant=variant,
                           value=value, minimum=min(ratios), maximum=max(ratios), unit='speedup', estimator='ratio of process medians'))
        return value

    def kernel_points(ax, metric, panel):
        for i, config in enumerate(CONFIGS):
            for core, name in enumerate(['native', 'cufftdx']):
                task = 'factor-' + name + '-' + config + '-ncu'
                rows = metrics(task, metric)
                assert len(rows) == 2 and [r['kernel_id'] for r in rows] == ['0', '1']
                for stage, row in enumerate(rows):
                    ax.scatter(i+(core-.5)*.34+(stage-.5)*.10, row['value'], marker=['o', '^'][stage], s=19,
                               color=COLORS[core], edgecolors='black', linewidth=.45, zorder=3)
                    export.append(dict(panel=panel, kind='profile', core=core, variant=config, kernel_id=stage,
                                       metric=metric, value=row['value'], unit=row['unit'], estimator='single capture, per kernel'))

    def base_axis(ax, title=None, top=None):
        if title:
            ax.set_title(title, loc='left', pad=6, fontsize=7.6)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='.89', linewidth=.5)
        ax.tick_params(length=2, pad=2)
        if top is not None:
            ax.set_ylim(0, top)

    style = {'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
             'legend.fontsize': 6.4, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'hatch.linewidth': .45}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.05, 4.10))
        # Matched columns: performance, communication/supply, resource cost.
        columns = [(.080, .245), (.407, .245), (.744, .238)]
        axes = [fig.add_axes([left, .640, width, .215]) for left, width in columns]
        axes += [fig.add_axes([left, .165, width, .260]) for left, width in columns[:2]]
        shared_ax = fig.add_axes([columns[2][0], .330, columns[2][1], .095])
        active_ax = fig.add_axes([columns[2][0], .165, columns[2][1], .115])
        a, b, c, d, e = axes
        base_axis(a, '(a) Same-core performance', 1.65)
        a.axhline(1, color='.3', ls='--', lw=.75, zorder=1)
        a.set_ylabel('Speedup (×)')
        ps_speedups=[]
        for core, name in enumerate(['native', 'cufftdx-thread']):
            ps_speedups.append(ratio_bar(a, core, name+'/stagewise-hbm', name+'/resident-prefix-2', core, 'a', .48))
        a.set_xticks([0, 1]); a.set_xticklabels(['Native', 'cuFFTDx'])
        a.set_xlim(-.65, 1.65); a.set_yticks([0, .5, 1, 1.5])
        a.text(.5, .91, 'vs. external steps; 5 → 4 kernels', transform=a.transAxes, ha='center', fontsize=6.0)

        base_axis(b, '(b) Communication retained', 120)
        b.set_ylabel('Traffic retained (%)')
        b.axhline(100, color='.3', ls='--', lw=.75)
        traffic=[]
        for i, (layer, names, scale) in enumerate([
                ('DRAM', ['dram__bytes_read.sum', 'dram__bytes_write.sum'], 1/2**30),
                ('L2', ['lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum'], 32/2**30)]):
            for core, name in enumerate(['native', 'cufftdx-thread']):
                old=total('subgraph-'+name+'-stagewise-hbm-ncu', names)*scale
                new=total('subgraph-'+name+'-resident-prefix-2-ncu', names)*scale
                value=100*new/old; x=i+(core-.5)*.34
                b.bar(x, value, .30, color=COLORS[core], edgecolor='black', linewidth=.45, hatch=HATCHES[core])
                b.text(x, value+3, f'{value:.1f}', fontsize=6.1, ha='center')
                traffic.append(dict(panel='b', kind='profile_summary', core=core, metric=layer, baseline_GiB=old,
                                    variant_GiB=new, value=value, unit='percent', estimator='ratio of full-call count sums'))
        export.extend(traffic)
        b.set_xticks([0,1]);b.set_xticklabels(['DRAM', 'L2 sectors']);b.set_xlim(-.6,1.6);b.set_yticks([0,50,100])
        b.text(.5, .96, 'Read + write; separate levels', transform=b.transAxes, ha='center', va='top', fontsize=6.2)

        base_axis(c, '(c) Local resource demand', 110)
        c.set_ylabel('Registers / thread')
        for core, name in enumerate(['native', 'cufftdx-thread']):
            old=metrics('subgraph-'+name+'-stagewise-hbm-ncu', REGISTERS)
            new=metrics('subgraph-'+name+'-resident-prefix-2-ncu', REGISTERS)
            for i,row in enumerate([old[1],old[2],new[1]]):
                x=i+(core-.5)*.34
                c.bar(x,row['value'],.30,color=COLORS[core],edgecolor='black',linewidth=.45,hatch=HATCHES[core])
                c.text(x,row['value']+3,f"{row['value']:.0f}",ha='center',fontsize=6.2)
                export.append(dict(panel='c',kind='profile',core=core,variant=['external step 1','external step 2','resident prefix'][i],
                                   kernel_id=row['kernel_id'],metric=REGISTERS,value=row['value'],unit=row['unit'],estimator='single capture, per kernel'))
        c.set_xticks([0,1,2]);c.set_xticklabels(['Step 1','Step 2','Resident\nprefix']);c.set_yticks([0,50,100]);c.set_xlim(-.65,2.65)
        c.text(.04,.96,'Only the replaced prefix',transform=c.transAxes,va='top',fontsize=6.2)

        base_axis(d, '(d) Speedup over same-core 1/0', 1.55)
        d.set_ylabel('Speedup (×)')
        d.axhline(1,color='.3',ls='--',lw=.75)
        for i, config in enumerate(CONFIGS):
            for core,name in enumerate(['native','cufftdx']):
                ratio_bar(d,i+(core-.5)*.34,name+'-t1-d0',name+'-'+config,core,'d')
        d.set_yticks([0,.5,1,1.5])
        base_axis(e,'(e) Memory dependency waits',100)
        e.set_ylabel('Long-scoreboard stall (%)')
        kernel_points(e,STALL,'e')
        e.set_yticks([0,25,50,75,100])
        e.legend(handles=[Line2D([],[],marker=m,linestyle='',color='black',markersize=3.5,label=f'Factor {i+1}') for i,m in enumerate(['o','^'])],
                 loc='upper right',frameon=False,handletextpad=.25,borderpad=.1)

        base_axis(shared_ax,'(f) Buffer and concurrency cost',85)
        base_axis(active_ax,top=22)
        shared=[]
        for i, config in enumerate(CONFIGS):
            rows=[r for core in ['native','cufftdx'] for r in metrics('factor-'+core+'-'+config+'-ncu',SHARED)]
            values=[r['value']/1024 for r in rows]
            assert len(values)==4 and len(set(values))==1
            shared.append(values[0]);shared_ax.bar(i,values[0],.5,color='.75',edgecolor='black',linewidth=.45)
            shared_ax.text(i,values[0]+3,f'{values[0]:g}',ha='center',fontsize=6.2)
            export.append(dict(panel='f-top',kind='profile',variant=config,metric=SHARED,value=values[0],unit='KiB/block',
                               estimator='identical across both cores and both factor kernels'))
        shared_ax.set_ylabel('Shared KiB',labelpad=1,fontsize=6.2);shared_ax.set_yticks([0,40,80]);shared_ax.tick_params(axis='x',bottom=False,labelbottom=False)
        kernel_points(active_ax,ACTIVE,'f-bottom')
        active_ax.set_ylabel('Warps (%)',labelpad=1,fontsize=6.2);active_ax.set_yticks([0,10,20])
        for ax in [d,e,shared_ax,active_ax]:
            ax.set_xlim(-.6,3.6);ax.set_xticks(range(4))
            if ax is not shared_ax:
                ax.set_xticklabels(['1/0','4/0','4/1','4/2']);ax.set_xlabel('Tiles t / prefetch depth d',labelpad=2)
        # Discrete configuration categories: no interpolation curves.
        fig.text(.08,.962,'Resident subgraphs  |  FFT FP32, N = 65,536, B = 256',weight='bold',fontsize=8)
        fig.legend(handles=[Patch(facecolor=col,edgecolor='black',hatch=h,label=name) for col,h,name in zip(COLORS,HATCHES,['Native core','cuFFTDx core'])],
                   loc='upper left',bbox_to_anchor=(.074,.945),frameon=False,ncol=2,handlelength=1.4,columnspacing=1.4)
        fig.text(.08,.540,'Tile supply  |  FFT FP64, N = 65,536, B = 16',weight='bold',fontsize=8)
        prefetch = [median(samples(core+'-t4-d0')) / median(samples(core+'-t4-d1'))
                    for core in ['native', 'cufftdx']]
        endpoint = [median(samples(core+'-t1-d0')) / median(samples(core+'-t4-d1'))
                    for core in ['native', 'cufftdx']]
        fig.text(.08,.496,f'Fixed 4-tile prefetch control: {prefetch[0]:.3f}× / {prefetch[1]:.3f}×; '
                 f'strong-endpoint gain: {endpoint[0]:.3f}× / {endpoint[1]:.3f}×.',fontsize=6.9)
        fig.text(.08,.056,'Performance: ratio of medians; 3 paired process points and min–max. Dashed line: unchanged baseline.',fontsize=6.5)
        fig.text(.08,.022,'Profiles: one complete capture per configuration; factor markers are kernels, not statistical error bars.',fontsize=6.5)
        out=PAPER/'figures/representative_profiles_20261108'
        out.parent.mkdir(parents=True,exist_ok=True)
        for extension in ['pdf','png','svg']:
            fig.savefig(out.with_suffix('.'+extension),dpi=300,facecolor='white')
        svg_path = out.with_suffix('.svg')
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
        plt.close(fig)
    csv_path=PAPER/'generated/representative_profiles_chart_data_20261108.csv'
    with csv_path.open('w',newline='') as handle:
        fields=sorted({k for row in export for k in row})
        writer=csv.DictWriter(handle,fields,lineterminator='\n');writer.writeheader();writer.writerows(export)
    metadata={
        'input_status':data['status'],'input':str(args.input.resolve()),'input_sha256':hashlib.sha256(raw).hexdigest(),
        'dimensions_inches':[7.05,4.10],'png_dpi':300,'matplotlib_version':matplotlib.__version__,
        'destination':'Internal IEEE two-column systems/architecture manuscript; no venue-compliance claim',
        'ordinary_latency':'Bars: median(baseline process times)/median(variant process times). Points and min/max: three within-process baseline/variant ratios. Repeated replays inside a process are not independent samples. No CI or significance claim.',
        'traffic':'Each resident complete-call sum is normalized to the same core external-step sum; DRAM and L2 stay separate. Only absolute counts summed across kernels. L2 sectors *32 bytes; GiB=bytes/2**30. Absolute values in companion CSV.',
        'resources':'Registers for the replaced prefix only. Factor shared allocation includes allocation overhead (17/49/65 KiB versus 16/48/64 requested); points show each factor kernel separately. No percentage averaging.',
        'diagnostics':'NCU replay caches/clocks uncontrolled; no profiler durations or uncertainty distributions plotted. Counter association is not an additive causal decomposition.',
        'exclusions':'None among 12 declared ordinary variants. Prefix register panel shows replaced steps only, with common permutation and unchanged suffix intentionally outside that local-resource comparison; traffic remains full-call.',
        'plotted_source_rows':list(plotted.values()),'source_hashes':data['provenance_sha256'],
        'chart_csv':str(csv_path.relative_to(PAPER)),'chart_csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'design_references':[{'title':'Welder','venue':'OSDI 2023','figures':[13], 'url':'https://www.usenix.org/system/files/osdi23-shi.pdf'},
                             {'title':'ROLLER','venue':'OSDI 2022','figures':[17,18], 'url':'https://www.usenix.org/system/files/osdi22-zhu.pdf'}],
        'alt_text':'Two rows align performance with communication or supply and resource cost. FP32 resident prefixes improve both fixed cores by about 1.19x while retaining 77% DRAM and 87% L2 traffic, despite more registers. FP64 depth-one prefetch is 1.12–1.15x faster than the stronger single-tile endpoint; depth two further lowers long-scoreboard stalls but uses 65 versus 49 KiB and has lower active-warp fractions and slower full execution.'}
    (PAPER/'generated/representative_profiles_figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
