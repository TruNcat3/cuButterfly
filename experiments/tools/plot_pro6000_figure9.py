#!/usr/bin/env python3
"""Plot verified PRO6000 Figure 9 tables; requires matplotlib, never a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import report_pro6000_figure9 as report

NAMES = {'cubutterfly': 'cuButterfly', 'cufft': 'cuFFT', 'vkfft': 'VkFFT',
         'gpu-fft': 'GPU-FFT', 'gpu-ntt': 'GPU-NTT', 'daofht': 'Dao FHT',
         'hadacore': 'HadaCore'}
COLORS = {'cubutterfly': '#2463a6', 'cufft': '#e49b36', 'vkfft': '#409a84',
          'gpu-fft': '#9873b9', 'gpu-ntt': '#9873b9', 'daofht': '#409a84',
          'hadacore': '#ba694a'}


def panel(ax, rows, names, title, denominator):
    values = [[r['cpu_normalized'] for r in rows if r['backend'] == name
               and r['cpu_normalized'] is not None] for name in names]
    all_values = [v for group in values for v in group]
    lower = min([1, *all_values]) / 2.3
    upper = max([1, *all_values]) * 3.5
    for x, (name, group) in enumerate(zip(names, values), 1):
        if not group:
            ax.text(x, lower * 1.4, 'unsupported', ha='center', color='#666', fontsize=8)
            continue
        bp = ax.boxplot([group], positions=[x], widths=.46, whis=(0, 100),
                        showfliers=False, patch_artist=True,
                        medianprops={'color': '#142638', 'linewidth': 1.4})
        bp['boxes'][0].set(facecolor=COLORS[name], alpha=.32, edgecolor=COLORS[name])
        # Each point is a request, not an independent timing repeat.
        offsets = [((j * 7) % 11 - 5) * .019 for j in range(len(group))]
        ax.scatter([x + d for d in offsets], group, s=15, color=COLORS[name],
                   alpha=.85, zorder=3, edgecolors='white', linewidths=.3)
        ax.text(x, 1.015, f'GM {report.geomean(group):.2f}×',
                transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
    ax.set(yscale='log', ylim=(lower, upper), xlim=(.45, len(names) + .55))
    ax.set_xticks(range(1, len(names) + 1),
                  [f'{NAMES[n]}\nn={len(v)}' for n, v in zip(names, values)], fontsize=9)
    ax.axhline(1, color='#6a737d', linestyle='--', linewidth=.8)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.grid(axis='y', which='major', alpha=.19)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, loc='left', pad=27, fontsize=11, fontweight='bold')
    ax.set_xlabel(denominator, fontsize=8, labelpad=7)
    ax.set_ylabel('CPU time / GPU complete-call time', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)


def save(fig, output, stem, tables):
    files = []
    for suffix in ('svg', 'pdf', 'png'):
        path = output / f'{stem}.{suffix}'
        metadata = {'Date': None} if suffix == 'svg' else (
            {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
        fig.savefig(path, dpi=180, metadata=metadata)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        files.append({'path': path.name, 'sha256': report.sha(path.read_bytes())})
    plt.close(fig)
    return {'files': files, 'tables_sha256': report.digest(tables),
            'matplotlib_version': matplotlib.__version__,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=report.ROOT / 'figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'pro6000-figure9-20261118', 'font.size': 9,
                         'pdf.fonttype': 42})
    tables, _ = report.render()
    rows = tables['backends']
    fig, axes = plt.subplots(6, 2, figsize=(12.8, 18))
    for op, pair, axis_row in zip(report.OPS,
            [('FP32', 'FP64'), ('word32', 'word64'), ('FP32', 'FP64'),
             ('word32', 'word64'), ('word32', 'word64'), ('FP32', 'FP64')], axes):
        for precision, ax in zip(pair, axis_row):
            cohort = 'fft17-final' if op == 'FFT' else 'full66'
            selected = [r for r in rows if (r['cohort'], r['operator'], r['precision'])
                        == (cohort, op, precision)]
            assert selected
            names = (['cubutterfly', 'cufft', 'vkfft', 'gpu-fft'] if op == 'FFT' else
                     ['cubutterfly', 'gpu-ntt'] if op == 'NTT' else
                     ['cubutterfly', 'daofht', 'hadacore'] if op == 'FWHT' else ['cubutterfly'])
            cpu = ('FFTW 3.3.10, 8 threads' if op == 'FFT' else
                   'HEXL 1.2.5, uint32 I/O → uint64 core, 1 thread' if (op, precision) == ('NTT', 'word32') else
                   'HEXL 1.2.5, 1 thread' if op == 'NTT' else
                   'Author Butterfly CPU, 1 thread; external GPU: gap' if op == 'structured-2x2' else
                   'FXT, 1 thread; external GPU: gap' if 'zeta' in op else 'FXT, 1 thread')
            panel(ax, selected, names, f'{op} / {precision} · {cohort}', cpu)
    fig.suptitle('PRO6000: CPU-normalized complete operator calls',
                 y=.987, fontsize=17, fontweight='bold')
    fig.text(.5, .968, 'Local CPU: Intel Xeon Platinum 8488C · shared-host provisional timings',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.075, right=.98, top=.925, bottom=.14, hspace=1.05, wspace=.23)
    fig.text(.075, .018,
        'Boxes: workload IQR and median; whiskers: full range; dots: exact requests. GM and n are per backend/panel.\n'
        'FFT: GPU0, final library a00188ff…; other panels: GPU1, original library 3b0668f2…. No pooled cohort/build mean.\n'
        'GPU: warm CUDA Graph, 7 × 20 complete calls, 10 warmups; CPU: wall-clock, same repeats. Initial transfers excluded.\n'
        'Integer outputs exact. Three GPU-NTT forward failures have no GPU ratio; own and replay independently exact.\n'
        'Floating results remain accuracy-review-required. HadaCore FP16 is outside Full66 FP32/FP64; see separate MMA9 panel.\n'
        'CPU/GPU values include required adapters; word32 HEXL is storage adapted. Request identities and all gaps are in the tables.',
        fontsize=9, linespacing=1.5)
    receipts = {'main': save(fig, args.output_dir, 'pro6000_figure9', tables)}
    fig, ax = plt.subplots(figsize=(6.9, 5.8))
    panel(ax, [r for r in rows if r['cohort'] == 'mma9-original'],
          ['cubutterfly', 'hadacore'], 'ALLOW_FP16_MMA9 · independent GPU0 cohort',
          'CPU FXT adapter: FP16 I/O, FP32 arithmetic, 1 thread')
    fig.subplots_adjust(left=.14, right=.97, top=.83, bottom=.32)
    fig.text(.14, .055,
        'CPU widening, complete FP32 FWHT and final narrowing are timed.\n'
        'This CPU reference is adapted, not native FP16 arithmetic.\n'
        'GPU library 3b0668f2…; all 9 requests retained; floating accuracy review pending.\n'
        'Box/range/dots: workloads, not confidence intervals. No pooled main/MMA9 mean.',
        fontsize=9, linespacing=1.5)
    receipts['mma9'] = save(fig, args.output_dir, 'pro6000_figure9_mma9', tables)
    (args.output_dir / 'pro6000_figure9.manifest.json').write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + '\n')
    print('Exported verified main and adapted-MMA9 plots as SVG, PDF and PNG')


if __name__ == '__main__':
    main()
