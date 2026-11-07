#!/usr/bin/env python3
"""CPU-only rendering of frozen FFT-L controls, separate from the main cohort."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper/generated/fft_l_20261104'
OUT = ROOT / 'paper/figures'
rows = list(csv.DictReader((DATA / 'boundary_confirmed.csv').open()))
profile = json.loads((DATA / 'profile_summary.json').read_text())
with plt.rc_context({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.88))
    fig.subplots_adjust(left=.13, right=.985, bottom=.23, top=.73, wspace=.57)
    ax = axes[0]
    batches = [4, 64, 256]
    for precision, color, marker in [('fp16', '#0072B2', 'o'), ('bf16', '#A34700', 's')]:
        values = [float(next(r for r in rows if r['case'] == f'{precision}-1d-b{b}')['organization_speedup']) for b in batches]
        ax.plot(range(3), values, color=color, marker=marker, markersize=3.4,
                linewidth=1.1, label=precision.upper())
    ax.axhline(1, color='#666666', linestyle='--', linewidth=.6)
    ax.set_xticks(range(3), [str(b) for b in batches])
    ax.set(xlabel='Batch', ylabel='Control / revised', ylim=(0, 3.4), yticks=[0, 1, 2, 3])
    ax.set_title('(a) A100 80GB', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2,
              frameon=False, fontsize=6.1, handlelength=1.2, columnspacing=.7)
    ax = axes[1]
    for variant, offset, color, hatch, label in [('old', -.18, '#999999', '', 'Original'),
                                               ('new', .18, '#0072B2', '//', 'Revised')]:
        part = [r for r in profile if r['variant'] == variant]
        values = [r['store_sectors'] / r['store_requests'] for r in part]
        assert values == ([32, 32] if variant == 'old' else [4, 4])
        bars = ax.bar(np.arange(2) + offset, values, .34, color=color, hatch=hatch, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value+1, f'{value:g}', ha='center', fontsize=6.3)
    ax.set_xticks([0, 1], ['Prefix', 'Suffix'])
    ax.set(xlabel='Stage band', ylabel='Store sectors / request', ylim=(0, 39), yticks=[0, 16, 32])
    ax.set_title('(b) A100 40GB', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2,
              frameon=False, fontsize=6.1, handlelength=1.1, columnspacing=.6)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#dddddd', linewidth=.45)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'fft_l_boundary_20261104.{ext}', dpi=300, facecolor='white')
    plt.close(fig)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'script_sha256': sha(Path(__file__)),
    'inputs': {str((DATA/f).relative_to(ROOT)): sha(DATA/f) for f in ('boundary_confirmed.csv', 'profile_summary.json', 'provenance.json')},
    'size_inches': [3.45, 1.88],
    'panel_a': 'All six confirmed N4096 native FFT-L cases. Control/revised complete-Graph medians, B4 endpoint retains G1/C1; B64/256 revised G32/C8. Fixed scalar radix4, 256 threads. Different builds preserve byte equality. One process per endpoint; workload points are not uncertainty.',
    'panel_b': 'Separate A100 40GB FP16 N4096 B256 NCU kernel captures. Each stage store_sectors/store_requests. No aggregation of counter units, no NCU timings used as ordinary performance.',
    'selection': 'All final-confirmed boundary cases, including BF16 B4 regression. No external superiority claim; full cuFFT comparisons retained in CSV and prose.',
    'outputs': {str(p.relative_to(ROOT)): sha(p) for p in OUT.glob('fft_l_boundary_20261104.*')},
}
(DATA/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Rendered FFT-L boundary evidence at column width.')
