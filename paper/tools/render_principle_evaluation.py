#!/usr/bin/env python3
"""Render separated operator, fixed-core, and application evidence cohorts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'results/paper_targeted_20261012/unified_report'
CORE = ROOT/'results/paper_completion_20260917/core_dataflow_ablation'
POISSON_REPORT = ROOT/'results/e2e_current_20261017/poisson_2048_b16_adaptive_trace/report.json'
GENERATED = ROOT/'paper/generated'
FIGURES = ROOT/'paper/figures'
STYLE = {
    'cuButterfly': ('#0072B2', 'o', '-'), 'cuFFT': ('#A34D00', 's', '--'),
    'VkFFT': ('#007B59', '^', ':'), 'Dao FHT': ('#AA3377', 'D', '-.'),
    'GPU-NTT': ('#555555', 'P', '--'),
    'CuPy native padding': ('#A34D00', 's', '--'),
    'Torch same padding (diagnostic)': ('#007B59', '^', ':'),
}
GROUPS = [('fft', 'fp32'), ('fft', 'fp64'), ('fwht', 'fp32'), ('fwht', 'fp64'),
          ('ntt', 'word32'), ('ntt', 'word64'), ('subset-zeta', 'uint32'),
          ('superset-zeta', 'uint32'), ('structured-2x2', 'fp32')]
FIELDS = {'cuButterfly': 'cubutterfly', 'cuFFT': 'cufft', 'VkFFT': 'vkfft',
          'Dao FHT': 'daofht', 'GPU-NTT': 'gpu-ntt'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    path = GENERATED/f'principle_{name}.csv'
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'nonpositive/nonfinite measured latency: {value}')
    return value


def values(row, field):
    if not row.get(field):
        return []
    return [positive(x) for x in json.loads(row[field]) if x is not None]


def gm(items):
    return math.exp(statistics.mean(math.log(x) for x in items)) if items else None


def axis_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#DADADA', linewidth=0.45)
    ax.set_axisbelow(True)


def legend(fig, names, y=0.99, ncol=None):
    handles = [Line2D([], [], color=STYLE[x][0], marker=STYLE[x][1], linestyle=STYLE[x][2],
                      markersize=4, label=x) for x in names]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y),
               ncol=ncol or len(names), frameon=False, fontsize=7)


def export(fig, name, description, sources, metadata, height, width=7):
    paths = []
    for extension in ('pdf', 'png', 'svg'):
        path = FIGURES/f'principle_{name}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white', transparent=False)
        if extension == 'png':
            with Image.open(path) as bitmap:
                if bitmap.mode == 'RGBA' and bitmap.getchannel('A').getextrema() != (255,255):
                    raise ValueError('figure unexpectedly contains transparent pixels')
                rgb = bitmap.convert('RGB')
            rgb.save(path, dpi=(300,300))
        paths.append(path)
    metadata['figures'][name] = dict(width_inches=width, height_inches=height, png_dpi=300,
                                   description=description, sources=[str(x.relative_to(ROOT)) for x in sources],
                                   outputs={str(x.relative_to(ROOT)): digest(x) for x in paths})
    plt.close(fig)


def operator_plot(rows, metadata):
    rows = [x for x in rows if x['cohort'] == 'E1-20261011-v4-complete']
    if len(rows) != 36:
        raise ValueError(f'expected exactly 36 E1 requests, found {len(rows)}')
    long, paired = [], []
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.subplots_adjust(left=.08, right=.99, bottom=.23, top=.72)
    for group_index, (operator, precision) in enumerate(GROUPS):
        group = [x for x in rows if (x['operator'], x['precision']) == (operator, precision)]
        for row in group:
            if row['complete_call'] != 'True' or row['public_selector'] != 'True':
                raise ValueError('E1 figure requires complete calls through the public selector')
            for system, prefix in FIELDS.items():
                long.append(dict(cohort=row['cohort'], id=row['id'], operator=operator, precision=precision,
                                 N=row['N'], B=row['B'], direction=row['direction'], system=system,
                                 latency_us=row.get(prefix+'_us'), process_us=row.get(prefix+'_process_us'),
                                 availability=row.get(prefix+'_availability'), source_paths=row['source_paths']))
        systems = [name for name, prefix in FIELDS.items() if any(x.get(prefix+'_us') for x in group)]
        offsets = np.linspace(-.24, .24, len(systems)) if len(systems) > 1 else [0]
        annotations = []
        for offset, system in zip(offsets, systems):
            prefix = FIELDS[system]
            latencies = []
            ratios = []
            for row in group:
                measured = row.get(prefix+'_us')
                if not measured:
                    continue
                latency = positive(measured)
                latencies.append(latency)
                if system != 'cuButterfly':
                    ratio = latency/positive(row['cubutterfly_us'])
                    ratios.append(ratio)
                    paired.append(dict(operator=operator, precision=precision, id=row['id'], baseline=system,
                                       baseline_us=latency, cubutterfly_us=row['cubutterfly_us'], ratio=ratio))
            color, marker, _ = STYLE[system]
            box = ax.boxplot([latencies], positions=[group_index+offset], widths=.15, patch_artist=True,
                             showfliers=False, whis=(0, 100), manage_ticks=False,
                             medianprops=dict(color='black', linewidth=.9),
                             boxprops=dict(edgecolor=color, facecolor='white', linewidth=.8),
                             whiskerprops=dict(color=color, linewidth=.7), capprops=dict(color=color, linewidth=.7))
            box['boxes'][0].set_hatch({'cuButterfly': '', 'cuFFT': '//', 'VkFFT': '..', 'Dao FHT': '\\\\', 'GPU-NTT': 'xx'}[system])
            jitter = np.linspace(-.035, .035, len(latencies))
            ax.scatter(group_index+offset+jitter, latencies, s=9, marker=marker, color=color, zorder=3)
            if ratios:
                short = {'cuFFT':'C', 'VkFFT':'V', 'Dao FHT':'D', 'GPU-NTT':'G'}[system]
                annotations.append(f'{short}: {gm(ratios):.2f}x / {len(ratios)}')
        if not annotations:
            annotations = ['no matched', 'baseline']
        expected = {'fwht': 'Dao FHT', 'ntt': 'GPU-NTT'}.get(operator)
        if expected and expected not in systems:
            annotations.append('unsupported')
        ax.text(group_index, 1.02, '\n'.join(annotations), transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=6.3, linespacing=1.1)
    labels = ['FFT\nFP32 (6)', 'FFT\nFP64 (6)', 'FWHT\nFP32 (6)', 'FWHT\nFP64 (2)',
              'NTT\nword32 (4)', 'NTT\nword64 (4)', 'Subset\nzeta U32 (3)', 'Superset\nzeta U32 (3)', '2x2\nFP32 (2)']
    ax.set_xticks(range(len(GROUPS)), labels)
    ax.set_yscale('log', base=10)
    measured_all = [positive(x['latency_us']) for x in long if x['latency_us']]
    ax.set_ylim(min(measured_all)/1.4, max(measured_all)*1.3)
    ax.set_xlim(-.55, len(GROUPS)-.45)
    ax.set_ylabel('Complete-call latency (us; log10)')
    axis_style(ax)
    for boundary in (1.5, 3.5, 5.5, 6.5, 7.5):
        ax.axvline(boundary, color='#BDBDBD', linestyle=':', linewidth=.65)
    legend(fig, list(FIELDS), ncol=5)
    fig.text(.5, .87, 'Paired GM / requests: C=cuFFT, V=VkFFT, D=Dao FHT, G=GPU-NTT (baseline/cuButterfly)', ha='center', fontsize=6.7)
    fig.text(.5, .015, 'E1 current build: 36 requests; boxes show workload spread (IQR, min/max), not uncertainty. Floating accuracy unassessed.', ha='center', fontsize=6.6)
    write_csv('e1_latency', long)
    write_csv('e1_pairs', paired)
    summaries = []
    for operator, precision in GROUPS:
        for baseline in list(FIELDS)[1:]:
            ratios = [x['ratio'] for x in paired if (x['operator'], x['precision'], x['baseline']) == (operator, precision, baseline)]
            summaries.append(dict(operator=operator, precision=precision, baseline=baseline, n=len(ratios),
                                  paired_geometric_mean=gm(ratios), status='matched' if ratios else 'no matched measured baseline'))
    write_csv('e1_pair_summary', summaries)
    metadata['e1'] = dict(requests=36, groups=9, aggregation='one measured complete-call latency per workload; box IQR and min/max across requests',
                         pairing='baseline_us / cubutterfly_us on identical request; separate GM per operator/precision/competitor',
                         missing='unsupported/not-applicable retained; absent box is not zero or a win')
    export(fig, 'e1_latency', 'Unified absolute latency distribution by operator and precision; missing external baselines explicit.',
           [REPORT/'operators.csv'], metadata, 3)


def ntt_extension_plot(rows, metadata):
    selected = [r for r in rows if r['cohort'] == 'NTT-20261012-extension-snapshot']
    if len(selected) != 12:
        raise ValueError('expected separate 12-point NTT extension')
    raw = json.loads((ROOT/'results/paper_targeted_20261012/ntt_extension/summary.json').read_text())
    group_by_id = {r['id']: r['group'] for r in raw['rows']}
    selected.sort(key=lambda r: (group_by_id[r['id']] != 'original4', int(r['N']), int(r['B']), r['direction']))
    figure_rows = []
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.subplots_adjust(left=.08, right=.99, bottom=.3, top=.81)
    for system, prefix in (('cuButterfly', 'cubutterfly'), ('GPU-NTT', 'gpu-ntt')):
        means, lo, hi = [], [], []
        for row in selected:
            process = values(row, prefix+'_process_us')
            center = positive(row[prefix+'_us'])
            means.append(center); lo.append(center-min(process)); hi.append(max(process)-center)
            for trial, latency in enumerate(process, 1):
                figure_rows.append(dict(cohort=row['cohort'], group=group_by_id[row['id']], id=row['id'], N=row['N'], B=row['B'], direction=row['direction'],
                                        system=system, trial=trial, latency_us=latency))
        color, marker, line = STYLE[system]
        ax.errorbar(np.arange(12)+(-.08 if system == 'cuButterfly' else .08), means, yerr=[lo, hi],
                    color=color, marker=marker, linestyle='none', capsize=2, markersize=4, linewidth=.8)
    ax.set_xticks(range(12), [f'{int(r["N"]).bit_length()-1}/{r["B"]}\n{r["direction"][0].upper()}' for r in selected], fontsize=7)
    ax.set_xlabel('log2(N) / batch; F=forward, I=inverse')
    ax.set_ylabel('Complete-call latency (us; log10)')
    ax.set_yscale('log', base=10); axis_style(ax)
    ax.axvline(3.5, color='#777777', linestyle=':', linewidth=.8)
    for x, name in ((1.5, 'original 4'), (7.5, 'additional 8')):
        subset = [r for r in selected if group_by_id[r['id']] == ('original4' if x == 1.5 else 'additional8')]
        ax.text(x, .99, f'{name}: GPU-NTT/cuB GM={gm([positive(r["gpu-ntt_us"])/positive(r["cubutterfly_us"]) for r in subset]):.3f}x',
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7)
    legend(fig, ['cuButterfly', 'GPU-NTT'])
    fig.text(.5, .015, 'Separate NTT extension; exact checks. Median and min/max of 3 processes, not CI; never pooled with E1.', ha='center', fontsize=6.8)
    write_csv('ntt_extension', figure_rows)
    export(fig, 'ntt_extension', 'Separate 12-point NTT extension, split original four and additional eight; three-process ranges.',
           [REPORT/'operators.csv', ROOT/'results/paper_targeted_20261012/ntt_extension/summary.json'], metadata, 2.5)


def core_plot(metadata):
    document = json.loads((CORE/'summary.json').read_text())
    raw = read_csv(CORE/'measurements.csv')
    if len(document['groups']) != 6 or any('80GB' not in r['device_name'] for r in raw):
        raise ValueError('fixed-core source must be the six A100 80GB workloads')
    rows = []
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fig.subplots_adjust(left=.18, right=.99, bottom=.27, top=.78)
    labels = []
    for index, group in enumerate(document['groups']):
        labels.append(f'2^{group["logN"]}\nB={group["batch"]}')
        for core, color, marker, offset in (('native', '#0072B2', 'o', -.07), ('cufftdx-thread', '#A34D00', 's', .07)):
            cells = group['cells']
            conventional = positive(cells[core+'/stagewise-hbm']['median_kernel_ms'])
            resident = positive(cells[core+'/resident-prefix-2']['median_kernel_ms'])
            ratio = conventional/resident
            values_by_org = {}
            for organization in ('stagewise-hbm', 'resident-prefix-2'):
                observations = [r for r in raw if int(r['logN']) == group['logN'] and int(r['batch']) == group['batch'] and r['core'] == core and r['organization'] == organization]
                if len(observations) != 3 or any(r['correct'] != '1' for r in observations):
                    raise ValueError('fixed-core three-trial correctness contract failed')
                values_by_org[organization] = {int(r['trial']): positive(r['kernel_ms']) for r in observations}
                if not math.isclose(statistics.median(values_by_org[organization].values()), positive(cells[core+'/'+organization]['median_kernel_ms']), rel_tol=1e-10):
                    raise ValueError('summary cell differs from raw trial median')
            trials = [values_by_org['stagewise-hbm'][t]/values_by_org['resident-prefix-2'][t] for t in sorted(values_by_org['stagewise-hbm'])]
            ax.errorbar(index+offset, ratio, yerr=[[ratio-min(trials)], [max(trials)-ratio]], color=color,
                        marker=marker, markersize=4.5, capsize=3, linewidth=.9)
            rows.append(dict(logN=group['logN'], N=2**group['logN'], B=group['batch'], core=core,
                             conventional_us=conventional*1000, resident_us=resident*1000, ratio=ratio,
                             trial_ratios=json.dumps(trials), min_ratio=min(trials), max_ratio=max(trials), n=3,
                             hardware='NVIDIA A100 80GB PCIe', source=str((CORE/'summary.json').relative_to(ROOT))))
    ax.axhline(1, color='#777777', linestyle=':', linewidth=.8)
    ax.set_ylim(.98, max(r['max_ratio'] for r in rows)*1.07)
    ax.set_xticks(range(6), labels, fontsize=7)
    ax.set_ylabel('Stagewise/resident ratio')
    axis_style(ax)
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none', label=label) for color, marker, label in
               [('#0072B2', 'o', 'Fixed native codelet'), ('#A34D00', 's', 'Fixed cuFFTDx Thread codelet')]]
    fig.legend(handles=handles, loc='upper center', ncol=1, frameon=False, fontsize=7)
    fig.text(.5, .025, 'Historical Sep17 | A100 80GB | FP32 FFT\nMatched stagewise control; 3 trial min/max, not CI.', ha='center', fontsize=6.3)
    write_csv('core_organization', rows)
    metadata['core_organization'] = dict(formula='same-core stagewise cell median / resident cell median', uncertainty='min/max of three trial-paired ratios; not CI',
                                         interpretation='finite matched control, not external-library native performance or current-build throughput',
                                         omitted='A100 40GB historical ratios excluded')
    export(fig, 'core_organization', 'Same-core organization ratios for six historical 80GB FFT workloads using two local codelets.',
           [CORE/'summary.json', CORE/'measurements.csv', CORE/'runtime_manifest.json'], metadata, 2.5, width=3.5)


def application_plots(rows, metadata):
    selected = [r for r in rows if r['cohort'] == 'applications-20261012-filter-v2']
    selected.sort(key=lambda r: (int(re.search(r'image(\d+)', r['input'])[1]), int(r['B'])))
    if len(selected) != 9 or {(int(re.search(r'image(\d+)', r['input'])[1]), int(r['B'])) for r in selected} != {(n,b) for n in (512,1024,2048) for b in (1,4,16)}:
        raise ValueError('expected exactly nine filter-v2 size/batch points')
    fields = {'cuButterfly': 'cubutterfly', 'CuPy native padding': 'cupy_native', 'Torch same padding (diagnostic)': 'torch_samepad_diagnostic'}
    long = []
    for row in selected:
        size = int(re.search(r'image(\d+)', row['input'])[1]); batch = int(row['B'])
        ours = values(row, 'cubutterfly_process_us')
        for system, prefix in fields.items():
            latencies = values(row, prefix+'_process_us')
            if len(latencies) != 3 or len(ours) != 3:
                raise ValueError('filter-v2 requires three real process values per method')
            for trial, (latency, own) in enumerate(zip(latencies, ours), 1):
                long.append(dict(cohort=row['cohort'], input=row['input'], size=size, B=batch, process=trial,
                                 method=system, latency_us=latency, images_per_second=batch*1e6/latency,
                                 baseline_over_cubutterfly=latency/own if system != 'cuButterfly' else 1,
                                 status=row['status'], source_paths=row['source_paths']))
    write_csv('application_processes', long)
    for quantity, height in (('throughput_ratio', 3), ('latency', 2.5)):
        fig, axes = plt.subplots(2 if quantity == 'throughput_ratio' else 1, 3, figsize=(7, height), squeeze=False)
        fig.subplots_adjust(left=.09, right=.99, bottom=.25, top=.8, wspace=.32, hspace=.37)
        for column, size in enumerate((512,1024,2048)):
            axis = axes[0,column]
            axis.set_title(f'{size} x {size}, filter 31 x 31', fontsize=8)
            for system in fields:
                color, marker, linestyle = STYLE[system]
                field = 'images_per_second' if quantity == 'throughput_ratio' else 'latency_us'
                medians, lows, highs = [], [], []
                for batch in (1,4,16):
                    process = [r[field] for r in long if (r['size'],r['B'],r['method']) == (size,batch,system)]
                    center = statistics.median(process)
                    medians.append(center); lows.append(center-min(process)); highs.append(max(process)-center)
                    axis.scatter(np.full(3, (1,4,16).index(batch)) + np.linspace(-.025,.025,3), process, marker=marker, s=7, color=color)
                axis.errorbar(range(3), medians, yerr=[lows, highs], color=color, marker=marker, linestyle=linestyle,
                              markersize=3.5, linewidth=.8, capsize=2)
            axis.set_xticks(range(3), ['1','4','16'])
            axis.set_yscale('log', base=10)
            axis_style(axis)
            if quantity == 'throughput_ratio':
                ratio_ax = axes[1,column]
                for system in list(fields)[1:]:
                    color, marker, linestyle = STYLE[system]
                    centers, lows, highs = [], [], []
                    for batch in (1,4,16):
                        process = [r['baseline_over_cubutterfly'] for r in long if (r['size'],r['B'],r['method']) == (size,batch,system)]
                        center = statistics.median(process)
                        centers.append(center); lows.append(center-min(process)); highs.append(max(process)-center)
                        ratio_ax.scatter(np.full(3,(1,4,16).index(batch))+np.linspace(-.025,.025,3), process, marker=marker,s=7,color=color)
                    ratio_ax.errorbar(range(3), centers, yerr=[lows,highs], color=color, marker=marker,linestyle=linestyle,markersize=3.5,linewidth=.8,capsize=2)
                ratio_ax.axhline(1,color='#777777',linestyle=':',linewidth=.7)
                ratio_ax.set_xticks(range(3),['1','4','16']); ratio_ax.set_xlabel('Independent image batch')
                axis_style(ratio_ax)
            else:
                axis.set_xlabel('Independent image batch')
        # Shared scales keep the three size panels directly comparable.
        field = 'images_per_second' if quantity == 'throughput_ratio' else 'latency_us'
        data = [r[field] for r in long]
        for axis in axes[0]:
            axis.set_ylim(min(data)/1.35,max(data)*1.35)
        axes[0,0].set_ylabel('Images/s (log10)' if quantity == 'throughput_ratio' else 'Batch latency (us; log10)')
        if quantity == 'throughput_ratio':
            ratios = [r['baseline_over_cubutterfly'] for r in long if r['method'] != 'cuButterfly']
            for axis in axes[1]:
                axis.set_ylim(min(.9,min(ratios)*.9),max(1.1,max(ratios)*1.12))
            axes[1,0].set_ylabel('Baseline/cuB\nlatency ratio')
        legend(fig,list(fields),ncol=3)
        fig.text(.5,.055,'Filter-v2; median and min/max of 3 independent processes, not CI. Floating tolerance unassessed: exploratory.',ha='center',fontsize=6.7)
        fig.text(.5,.015,'CuPy native FFT padding is the primary baseline; Torch same padding is a separate diagnostic. Shared filter recomputed per call.',ha='center',fontsize=6.4)
        name = 'application_throughput_ratio' if quantity == 'throughput_ratio' else 'application_latency'
        export(fig,name,'Nine batch filtering workloads with three real process measurements; primary and diagnostic baselines distinct.',
               [REPORT/'applications.csv'],metadata,height)
    metadata['applications'] = dict(cohort='applications-20261012-filter-v2', points=9, independent_processes_per_point=3,
                                    throughput='B * 1e6 / process latency_us', ratio='baseline process latency / cuButterfly process latency; matched process index',
                                    interval='median and min/max; not CI', accuracy='no explicit floating threshold; exploratory measured results only')
    return selected


def poisson_endpoint_plot(report_path, metadata):
    """Render the measured 2-D Poisson endpoint as a separate application cohort.

    This report is a single A100 process run.  The twenty wall-clock samples per
    backend/mode are repeated steady-state calls, so the figure reports their
    distribution without treating them as independent process replicates.
    """
    report = json.loads(Path(report_path).read_text())
    required = ('application', 'source', 'size', 'batch', 'gpu_name', 'gpu_uuid',
                'library_sha256', 'timing_scope', 'setup_search_capture_excluded',
                'timings', 'speedup')
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f'Poisson endpoint report missing fields: {missing}')
    if report['size'] != 2048 or report['batch'] != 16:
        raise ValueError('Poisson endpoint figure requires the 2048/B16 report')
    if not report['setup_search_capture_excluded']:
        raise ValueError('Poisson endpoint requires setup/search/capture to be excluded')

    methods = (('cuButterfly', 'cubutterfly'), ('Torch', 'torch'), ('CuPy', 'cupy'))
    modes = ('eager', 'graph')
    rows = []
    summary = []
    for method, prefix in methods:
        for mode in modes:
            key = f'{prefix}_{mode}'
            timing = report['timings'].get(key)
            if not timing or not isinstance(timing.get('wall_ms'), list):
                raise ValueError(f'Poisson endpoint missing wall samples for {key}')
            samples = [positive(value) for value in timing['wall_ms']]
            if len(samples) != 20:
                raise ValueError(f'Poisson endpoint requires 20 wall samples for {key}, found {len(samples)}')
            median = statistics.median(samples)
            expected = positive(timing['wall_median_ms'])
            if not math.isclose(median, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{key} report median differs from raw wall samples')
            for trial, latency in enumerate(samples, 1):
                rows.append(dict(cohort='e2e-20261017-poisson-2048-b16', application=report['application'],
                                 size=report['size'], B=report['batch'], method=method, mode=mode,
                                 trial=trial, wall_ms=latency,
                                 batch_per_second=report['batch'] * 1000.0 / latency,
                                 gpu_name=report['gpu_name'], gpu_uuid=report['gpu_uuid'],
                                 report_path=str(Path(report_path).relative_to(ROOT))))
            summary.append(dict(cohort='e2e-20261017-poisson-2048-b16', application=report['application'],
                                size=report['size'], B=report['batch'], method=method, mode=mode,
                                samples=len(samples), median_wall_ms=median,
                                min_wall_ms=min(samples), max_wall_ms=max(samples),
                                batch_per_second=report['batch'] * 1000.0 / median,
                                speedup_vs_cubutterfly_graph=(
                                    positive(report['timings']['cubutterfly_graph']['wall_median_ms']) / median),
                                report_path=str(Path(report_path).relative_to(ROOT))))

    write_csv('application_poisson_endpoint', rows)
    write_csv('application_poisson_endpoint_summary', summary)
    by_method = {
        method: {row['mode']: row for row in summary if row['method'] == method}
        for method, _ in methods
    }
    tex = [r'\begingroup', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
           r'Backend & Eager ms & Graph ms & Graph batch/s & Graph/cuB \\', r'\midrule']
    for method, _ in methods:
        eager = by_method[method]['eager']
        graph = by_method[method]['graph']
        label = r'\system{}' if method == 'cuButterfly' else method
        ratio = graph['speedup_vs_cubutterfly_graph']
        tex.append(f'{label} & {eager["median_wall_ms"]:.3f} & {graph["median_wall_ms"]:.3f} & '
                   f'{graph["batch_per_second"]:.1f} & {ratio:.3f}\\\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
            r'\parbox{\linewidth}{N=2048, B=16 periodic spectral Poisson solve on A100 80GB. '
            r'Values are medians of 20 repeated steady-state wall samples in one process; '
            r'Graph/cuB is a latency ratio, so values below one favor cuButterfly. Setup, search and capture are excluded.}',
            r'\endgroup']
    (GENERATED/'principle_application_poisson_endpoint.tex').write_text('\n'.join(tex)+'\n')
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.subplots_adjust(left=.09, right=.99, bottom=.25, top=.79)
    positions = []
    labels = []
    colors = {'eager': '#999999', 'graph': '#0072B2'}
    for index, (method, _) in enumerate(methods):
        for offset, mode in ((-.18, 'eager'), (.18, 'graph')):
            samples = [r['wall_ms'] for r in rows if r['method'] == method and r['mode'] == mode]
            position = index + offset
            positions.append(position)
            labels.append(f'{method}\n{mode}')
            box = ax.boxplot([samples], positions=[position], widths=.28, patch_artist=True,
                             showfliers=False, whis=(0, 100), manage_ticks=False,
                             medianprops=dict(color='black', linewidth=.9),
                             boxprops=dict(edgecolor=colors[mode], facecolor=colors[mode], alpha=.55, linewidth=.8),
                             whiskerprops=dict(color=colors[mode], linewidth=.7),
                             capprops=dict(color=colors[mode], linewidth=.7))
            ax.scatter(np.full(len(samples), position) + np.linspace(-.045, .045, len(samples)), samples,
                       s=6, color=colors[mode], alpha=.65, zorder=3)
            median = statistics.median(samples)
            ax.text(position, max(samples) + .03 * (max(samples) - min(samples)), f'{median:.3f}',
                    ha='center', va='bottom', fontsize=6.2, rotation=90)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Complete-call wall latency (ms)')
    ax.set_title('2-D periodic spectral Poisson solve | N=2048, B=16 | A100 80GB', fontsize=8.5, pad=8)
    axis_style(ax)
    ax.set_ylim(bottom=min(r['wall_ms'] for r in rows) * .98, top=max(r['wall_ms'] for r in rows) * 1.09)
    legend_handles = [Line2D([], [], color=colors[m], marker='s', linestyle='none', markersize=5,
                             label=m) for m in modes]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=2,
               frameon=False, fontsize=7)
    fig.text(.5, .055, '20 repeated steady-state calls in one process; boxes show min/max and median, not CI.',
             ha='center', fontsize=6.6)
    fig.text(.5, .015, 'Wall scope includes device input/output copies, complete solve, and synchronization; setup/search/capture excluded.',
             ha='center', fontsize=6.2)
    export(fig, 'application_poisson_endpoint',
           'Measured 2-D periodic spectral Poisson endpoint at N=2048 and B=16; eager and CUDA Graph modes across three backends.',
           [Path(report_path)], metadata, 2.8)
    metadata['poisson_endpoint'] = dict(
        cohort='e2e-20261017-poisson-2048-b16', application=report['application'], size=report['size'], batch=report['batch'],
        hardware=dict(name=report['gpu_name'], uuid=report['gpu_uuid'], memory_bytes=report['gpu_memory_bytes']),
        samples='20 repeated steady-state wall_ms samples per backend/mode in one process; not independent process replicates',
        timing_scope=report['timing_scope'], setup_search_capture_excluded=report['setup_search_capture_excluded'],
        report_path=str(Path(report_path).relative_to(ROOT)), report_sha256=digest(report_path),
        library_sha256=report['library_sha256'], source=report['source'])


def application_table(all_rows, filter_rows):
    historical = [r for r in all_rows if r['cohort'] in ('applications-20261012-QuaRot','applications-20261011-FNO-attempt03') and r.get('cubutterfly_ms')]
    rows = []
    tex = [r'\begingroup',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{llrrrrrr}',r'\hline',
           r'Workload / version & Primary baseline & $B$ & $n$ & cuB ms & Base ms & Base/cuB & cuB items/s \\',r'\hline']
    for row in filter_rows+historical:
        if row['cohort'].endswith('filter-v2'):
            size = re.search(r'image(\d+)',row['input'])[1]
            label = f'Filter {size}, v2'; baseline='CuPy native'; prefix='cupy_native'
        elif 'QuaRot' in row['cohort']:
            label = 'QuaRot '+row['input'].split('phase=')[-1]+', Oct12 B1'; baseline='Dao FWHT';prefix='dao_baseline'
        else:
            label = 'FNO '+row['input'].split(',')[0]+', Oct11';baseline='Torch native';prefix='torch_native'
        own = positive(row['cubutterfly_ms']); base=positive(row[prefix+'_ms'])
        throughput=positive(row['cubutterfly_throughput'])
        rows.append(dict(cohort=row['cohort'], workload=label, baseline=baseline, B=row['B'], n=row['trials'],
                         cubutterfly_ms=own, baseline_ms=base, baseline_over_cubutterfly=base/own,
                         cubutterfly_items_per_second=throughput, throughput_unit=row['throughput_unit'],
                         status=row['status'], source_paths=row['source_paths']))
        tex.append(f'{label} & {baseline} & {row["B"]} & {row["trials"]} & {own:.3f} & {base:.3f} & {base/own:.3f} & {throughput:.1f} '+r'\\')
    tex += [r'\hline',r'\end{tabular}',r'\par\smallskip',
            r'\parbox{\linewidth}{Filter v2: nine exploratory FP32 requests (floating tolerance unassessed), three processes, shared filter; primary CuPy uses its native FFT padding. Torch same-padding diagnostic is shown separately in the figures. QuaRot Oct12: historical B1, two process trials, random weights, implementation consistency only; throughput is tokens/s. FNO Oct11: historical B4, three process medians, checkpoint missing, throughput is samples/s. Filter throughput is images/s. Versions are not pooled.}',r'\endgroup']
    (GENERATED/'principle_application_table.tex').write_text('\n'.join(tex)+'\n')
    write_csv('application_table',rows)


def captions():
    text = {
        'PrincipleEOneCaption': 'Current E1: 36 complete semantic requests, grouped by operator and precision. Boxes show workload latency quartiles with min/max whiskers; dots are individual requests, not independent replicates. Paired geometric means and counts are reported separately for each competitor. Missing baselines are explicit. Floating results retain measured errors but have no accepted accuracy threshold.',
        'PrincipleNTTExtensionCaption': 'The separate 12-point NTT extension on A100 80GB. Original four and additional eight requests are not pooled with E1. Points show median complete-call latency across three independent processes; bars show process min/max, not confidence intervals. Both implementations pass exact elementwise checks.',
        'PrincipleCoreOrganizationCaption': 'Historical A100 80GB FP32 FFT fixed-core organization control (September 17). Each ratio is the stagewise-HBM median latency divided by the resident-prefix median latency for the same local codelet. Bars show min/max of three trial-paired ratios, not confidence intervals. The stagewise control is a matched organization, not an external library native implementation; current-build throughput is not inferred.',
        'PrincipleApplicationThroughputCaption': 'Filter-v2 independent image batches with one shared 31 by 31 filter, recomputed per complete call. Three panels separate image sizes; upper panels show throughput and lower panels show process-paired baseline/cuButterfly latency ratios. CuPy native padding is the primary application baseline; Torch with identical padding is a separate diagnostic. Medians and min/max use three real processes, not confidence intervals. Floating tolerance is unassessed, so these measurements are exploratory.',
        'PrincipleApplicationLatencyCaption': 'Whole-batch filter-v2 latency for nine independent-image requests. Padding, image and shared-filter transforms, multiplication, inverse, and crop/copy are included; host transfer, plan construction, compilation, and tuning are excluded. CuPy uses native FFT padding; Torch same-padding results are diagnostic. Points and ranges show three-process median and min/max, not confidence intervals. Floating accuracy acceptance remains unassessed.',
    }
    (GENERATED/'principle_captions.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in text.items())+'\n')


def main():
    global REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=REPORT)
    args=parser.parse_args()
    REPORT=args.report_dir.resolve()
    GENERATED.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'savefig.facecolor':'white'})
    sources=[REPORT/'operators.csv',REPORT/'applications.csv',CORE/'summary.json',CORE/'measurements.csv',CORE/'runtime_manifest.json',
             ROOT/'results/paper_targeted_20261012/ntt_extension/summary.json',ROOT/'docs/research_principles.md',
             POISSON_REPORT,Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    metadata=dict(schema='cubutterfly-principle-figures-v1',figures={},source_hashes=hashes,
                  software=dict(matplotlib=matplotlib.__version__,numpy=np.__version__),
                  export='7 inch double-column figures; 3.5 inch fixed-core figure; opaque PDF/PNG/SVG; 300 DPI RGB PNG; embedded TrueType PDF; editable SVG text',
                  publisher_status='general manuscript figures; venue-specific requirements pending review',
                  policy='no pooled competitor ratios, no pooled historical/current cohorts, no invented missing observations')
    operators=read_csv(REPORT/'operators.csv');applications=read_csv(REPORT/'applications.csv')
    operator_plot(operators,metadata);ntt_extension_plot(operators,metadata);core_plot(metadata)
    selected=application_plots(applications,metadata);poisson_endpoint_plot(POISSON_REPORT,metadata)
    application_table(applications,selected);captions()
    for path in sources:
        if digest(path)!=hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError(f'source changed while rendering: {path}; rerun against a stable report')
    metadata['generated_hashes']={str(p.relative_to(ROOT)):digest(p) for p in sorted(GENERATED.glob('principle_*')) if p.suffix in ('.csv','.tex')}
    (GENERATED/'principle_figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(figures=list(metadata['figures']),metadata='paper/generated/principle_figure_metadata.json')))


if __name__=='__main__':
    main()
