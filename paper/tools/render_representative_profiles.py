#!/usr/bin/env python3
"""Render frozen representative profiles; ordinary timing and NCU diagnostics stay separate."""
import argparse
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
CONFIGS = ['t1-d0', 't4-d0', 't4-d1', 't4-d2']
STALL = 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'
SHARED = 'launch__shared_mem_per_block_allocated'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=PAPER / 'generated/representative_profiles_20261108.json')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    assert len(data['tasks']) == 30
    assert all(t['status'] == 'complete' for t in data['tasks'] if t['kind'] in ('ordinary', 'ncu'))
    plotted = []

    def samples(variant, scale=1):
        rows = sorted([r for r in data['ordinary_samples'] if r['variant'] == variant], key=lambda r: r['process'])
        assert len(rows) == 3 and len({r['process'] for r in rows}) == 3
        plotted.extend(dict(panel_source='ordinary', **r) for r in rows)
        return [r['time_ms'] * scale for r in rows]

    def metrics(task, metric):
        rows = [r for r in data['ncu_kernel_metrics'] if r['task_id'] == task and r['metric'] == metric]
        assert rows and all(r['value'] is not None for r in rows)
        plotted.extend(dict(panel_source='NCU diagnostic', **r) for r in rows)
        return sorted(rows, key=lambda r: int(r['kernel_id']))

    def total(task, names, factor):
        return sum(r['value'] for name in names for r in metrics(task, name)) * factor

    def timing(ax, x, vals, core, width):
        m = median(vals)
        ax.bar(x, m, width, color=COLORS[core], alpha=.78, edgecolor='black', linewidth=.45, hatch=['', '///'][core], zorder=2)
        ax.errorbar(x, m, yerr=[[m-min(vals)], [max(vals)-m]], color='black', capsize=2, lw=.65, zorder=4)
        ax.scatter([x-.065, x, x+.065], vals, s=10, marker=['o', 's'][core], facecolor='white', edgecolor='black', linewidth=.5, zorder=5)

    style = {'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'hatch.linewidth': .45}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.6))
        fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.82, wspace=.40, hspace=.95)
        titles = ['(a) Ordinary latency', '(b) Complete DRAM traffic', '(c) Complete L2 traffic', '(d) Ordinary latency', '(e) Long-scoreboard stall', '(f) Allocated shared memory']
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, loc='left', pad=5)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='.88', lw=.5)
            ax.set_ylim(bottom=0)
            ax.tick_params(length=2, pad=2)
        orgs = ['stagewise-hbm', 'resident-prefix-2']
        for core, name in enumerate(['native', 'cufftdx-thread']):
            for org, organization in enumerate(orgs):
                x = org + (core-.5)*.34
                variant = name+'/'+organization
                task = 'subgraph-'+name+'-'+organization+'-ncu'
                timing(axes[0, 0], x, samples(variant), core, .30)
                for ax, names, factor in [(axes[0, 1], ['dram__bytes_read.sum', 'dram__bytes_write.sum'], 1/2**30), (axes[0, 2], ['lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum'], 32/2**30)]:
                    value = total(task, names, factor)
                    ax.bar(x, value, .30, color=COLORS[core], alpha=.78, edgecolor='black', linewidth=.45, hatch=['', '///'][core])
        for ax in axes[0]:
            ax.set_xticks([0,1])
            ax.set_xticklabels(['External\nsteps', 'Resident\nprefix'])
            ax.set_xlim(-.6, 1.6)
            ax.set_ylim(0, max(p.get_height() for p in ax.patches)*1.18)
        axes[0,0].set_ylabel('Latency (ms)')
        axes[0,1].set_ylabel('Read + write (GiB)')
        axes[0,2].set_ylabel('Read + write (GiB)')
        shared_by_config = []
        for i, config in enumerate(CONFIGS):
            shared = []
            for core, name in enumerate(['native', 'cufftdx']):
                variant = name+'-'+config
                task = 'factor-'+variant+'-ncu'
                timing(axes[1,0], i+(core-.5)*.34, samples(variant, 1000), core, .30)
                rows = metrics(task, STALL)
                assert len(rows) == 2 and {r['kernel_id'] for r in rows} == {'0', '1'}
                for stage, row in enumerate(rows):
                    axes[1,1].scatter(i+(core-.5)*.32+(stage-.5)*.10, row['value'], marker=['o','^'][stage], s=21, color=COLORS[core], edgecolors='black', linewidth=.45, zorder=3)
                sm = metrics(task, SHARED)
                assert len(sm) == 2
                shared.extend(r['value']/1024 for r in sm)
            assert len(set(shared)) == 1, 'Do not collapse unequal allocations'
            shared_by_config.append(shared[0])
        axes[1,2].bar(range(4), shared_by_config, width=.55, color='.7', edgecolor='black', linewidth=.5, hatch='..')
        for i, v in enumerate(shared_by_config):
            axes[1,2].text(i, v+1.5, f'{v:g}', ha='center', fontsize=7)
        for ax in axes[1]:
            ax.set_xticks(range(4))
            ax.set_xticklabels(['1/0','4/0','4/1','4/2'])
            ax.set_xlabel('Configuration t / d', labelpad=2)
            ax.set_xlim(-.6,3.6)
        axes[1,0].set_ylabel('Latency (µs)')
        axes[1,0].set_ylim(0,155)
        axes[1,1].set_ylabel('Per-kernel stall (%)')
        axes[1,1].set_ylim(0,100)
        axes[1,2].set_ylabel('KiB / compute group')
        axes[1,2].set_ylim(0,84)
        axes[1,2].text(.5,.96,'Both cores; both stages', transform=axes[1,2].transAxes, ha='center', va='top', fontsize=6.5)
        axes[1,1].legend(handles=[Line2D([],[],marker=m, linestyle='', color='black', markersize=4,label=f'Stage {i+1}') for i,m in enumerate(['o','^'])], loc='upper right', frameon=False, fontsize=6, handletextpad=.3, borderpad=.1)
        fig.text(.075,.965,'FP32  •  N = 65,536, B = 256', weight='bold', fontsize=8)
        fig.text(.075,.458,'FP64  •  N = 65,536, B = 16', weight='bold', fontsize=8)
        fig.legend(handles=[Patch(facecolor=c, edgecolor='black', hatch=h, label=n) for c,h,n in zip(COLORS,['','///'],['Native core','cuFFTDx core'])], loc='upper right', bbox_to_anchor=(.99,1), frameon=False, ncol=2, handlelength=1.3, columnspacing=1)
        fig.text(.075,.035,'Latency: median bar, 3 process points, min–max whisker.   NCU panels: diagnostics from complete captures.',fontsize=7)
        out = PAPER/'figures/representative_profiles_20261108'
        out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out.with_suffix('.pdf'), facecolor='white')
        fig.savefig(out.with_suffix('.png'), dpi=300, facecolor='white')
        plt.close(fig)
    metadata = {
        'input_status': data['status'], 'unused_nsys_tasks': [t for t in data['tasks'] if t['kind'] == 'nsys'],
        'input': str(args.input.resolve()), 'input_sha256': hashlib.sha256(raw).hexdigest(),
        'dimensions_inches': [7.05,3.6], 'png_dpi':300, 'matplotlib_version':matplotlib.__version__,
        'destination':'Provisional manuscript figure; venue requirements not specified',
        'ordinary_latency': 'Median of three independent process timing samples; all three points and min/max shown; no confidence intervals. ms to microseconds: multiply by 1000.',
        'diagnostics':'NCU counters are separate from ordinary performance. Complete capture sums read and write counters across all 5 stagewise / 4 resident kernels. No profiler durations plotted.',
        'dram_conversion':'sum(dram__bytes_read.sum + dram__bytes_write.sum) / 2**30 GiB',
        'l2_conversion':'sum(lts__t_sectors_op_read.sum + lts__t_sectors_op_write.sum) * 32 bytes/sector / 2**30 GiB; full 32-byte sector conversion, not useful payload',
        'stall':'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct; each kernel separately, never averaged; stages 1/2 correspond to capture kernel IDs 0/1',
        'shared':'launch__shared_mem_per_block_allocated in byte/block /1024. One bar per configuration only after verifying equality across both cores and both stages; block is compute group.',
        'shared_KiB':shared_by_config,
        'exclusions':'None among requested variants; all 36 ordinary observations and all requested NCU metric rows retained.',
        'source_hashes': data['provenance_sha256'], 'plotted_source_rows': plotted,
        'alt_text':'Two rows compare both cores. P–S resident prefix reduces ordinary latency and total memory traffic. P–P t4/d0 is slower than t1/d0; d1 reduces latency and long-scoreboard stall, while d2 raises allocation and latency relative to d1. All stages and process samples are shown.'}
    meta = PAPER/'generated/representative_profiles_figure_metadata.json'
    meta.write_text(json.dumps(metadata,indent=2)+'\n')
    print(out)

if __name__ == '__main__':
    main()
