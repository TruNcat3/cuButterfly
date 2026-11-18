#!/usr/bin/env python3
"""Export the small Figure 9 record overlay without output arrays or binaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import report_pro6000_figure9 as report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--standalone', action='store_true', help='Include older source records and every verifier dependency; omit array payloads')
    args = parser.parse_args()
    tables, text = report.render()
    assert json.loads(gzip.decompress((report.SNAP / 'tables.json.gz').read_bytes())) == tables
    assert json.loads((report.SNAP / 'identities.json').read_text()) == tables['identities']
    assert not subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=report.ROOT, text=True).strip(), 'Commit the reviewed data before packaging'
    assert report.REPORT.read_text() == text
    for name, values in tables.items():
        if name != 'identities':
            assert (report.SNAP / f'{name}.csv').read_text() == report.csv_text(values)
    plots = json.loads((report.ROOT / 'figures/pro6000_figure9.manifest.json').read_text())
    for receipt in plots.values():
        assert receipt['tables_sha256'] == report.digest(tables)
        for item in receipt['files']:
            assert report.sha((report.ROOT / 'figures' / item['path']).read_bytes()) == item['sha256']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ('pro6000-figure9-followup-standalone-20261118.zip' if args.standalone else 'pro6000-figure9-20261118.zip')
    assert not archive.exists(), 'Use a fresh output directory; never overwrite a published asset'
    files = sorted(p for p in report.SNAP.rglob('*') if p.is_file())
    files += sorted((report.ROOT / 'figures').glob('pro6000_figure9*'))
    files += [report.REPORT, Path(__file__),
              report.ROOT / 'experiments/tools/report_pro6000_figure9.py',
              report.ROOT / 'experiments/tools/plot_pro6000_figure9.py']
    if args.standalone:
        import report_pro6000_figure9_followup as followup
        followup.check()
        confirmed_tables,_=followup.render()
        receipts=json.loads((report.ROOT/'figures/pro6000_figure9_confirmed.manifest.json').read_text())
        for receipt in receipts.values():
            assert receipt['tables_sha256']==report.digest(confirmed_tables)
            for item in receipt['files']:
                assert report.sha((report.ROOT/'figures'/item['path']).read_bytes())==item['sha256']
        for name in ('pro6000_full66_20261118','pro6000_extensions_20261117','pro6000_fixed_mapping_20261117','pro6000_figure9_followup_20261118'):
            files += [p for p in (report.ROOT/'experiments/snapshots'/name).rglob('*') if p.is_file() and p.suffix not in ('.npz','.npy')]
        files += [report.ROOT/'experiments/snapshots/current_refresh/snapshot.json',
                  report.ROOT/'config/research_campaign/mapping_seeds.json',followup.REPORT]
        files += [report.ROOT/'experiments/tools'/name for name in (
            'report_pro6000_full66.py','report_pro6000_extensions.py','report_pro6000_fixed_samples.py',
            'report_pro6000_figure9_followup.py','run_pro6000_cyclic_ntt8.py','confirm_pro6000_cpu.py',
            'reproduce_pro6000_ntt8_failure.py')]
        files += [report.ROOT/'experiments/reports'/name for name in (
            'PRO6000_FULL66_20261118.md','PRO6000_EXTENSIONS_20261117.md','PRO6000_FIXED_MAPPING_SAMPLES_20261117.md')]
    files=sorted(set(files))
    inventory = []
    for path in files:
        assert path.suffix not in ('.npz', '.npy', '.so', '.a', '.o')
        raw = path.read_bytes()
        inventory.append({'path': str(path.relative_to(report.ROOT)), 'bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = {
        'schema': 'pro6000-figure9-self-contained-v2' if args.standalone else 'pro6000-figure9-performance-overlay-v1',
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=report.ROOT, text=True).strip(),
        'request_counts': {'full66': 66, 'fft17-original': 17, 'fft17-final': 17,
                           'mma9-original': 9, 'fixed-mapping': 25},
        'backend_cells': len(tables['backends']), 'local_cpu_records': len(tables['cpu']),
        'files': inventory,
        'scope': ('Self-contained metadata/timing verification. Includes all old cohort records and verifier dependencies. Original array hashes retained as receipts; payloads omitted, not revalidated.' if args.standalone else
                  'Record overlay. Reused older snapshots are in the source commit and original performance release. No output-array payloads or binaries.')}
    readme = (
        'PRO6000 Figure 9 performance records\n\n'
        'Open requests.csv/backends.csv/cpu.csv/statistics.csv directly. tables.json.gz\n'
        'contains their JSON equivalents. All rows retain cohort/build/platform identities,\n'
        'seven samples, errors, failures, support boundaries and exact CPU input joins.\n'
        'CPU timings are provisional shared-host records; floats require numerical review.\n\n'
        'For deterministic verification, check out source_commit from FIGURE9_MANIFEST.json,\n'
        'extract this archive into that checkout and run:\n'
        '  python3 experiments/tools/report_pro6000_figure9.py --check\n'
        'The verifier reuses the earlier Full66/FFT17/MMA9/fixed-mapping snapshots already\n'
        'present in that commit. Those original performance records are also in the\n'
        'pro6000-performance-20261118 release. This overlay does not duplicate large arrays.\n\n'
        'Verify every inventory member using its SHA256 in FIGURE9_MANIFEST.json.\n')
    if args.standalone:
        readme=('PRO6000 Figure 9 self-contained performance records\n\n'
                'Extract into an empty directory, then run with standard-library Python:\n'
                '  python3 experiments/tools/report_pro6000_figure9.py --check\n\n'
                'No source checkout, downloads, third-party Python dependencies, GPU, output\n'
                'arrays or library binaries are required. Every prior cohort source record,\n'
                'the four report verifiers and required A100 snapshot/mapping catalog are included.\n'
                'Both the original Figure9 derived tables and the finite follow-up are checked.\n'
                'Original array SHA receipts are verified against their source metadata; absent\n'
                'array payloads are not claimed to have been replayed. All compressed source\n'
                'records have compressed and decompressed checksums. FIGURE9_MANIFEST.json\n'
                'contains a complete inventory of this package.\n\n'
                'CPU medians use the median of three independent process medians; shared-host\n'
                'provisional labels remain. NTT8 has six valid pairs and two external failures;\n'
                'own/replay are exact. Original Full66 failures and old cohort means remain.\n')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            z.write(path, str(path.relative_to(report.ROOT)))
        z.writestr('FIGURE9_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('FIGURE9_README.txt', readme)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_text(f'{checksum}  {archive.name}\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in inventory:
            raw = z.read(entry['path'])
            assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
        if args.standalone:
            with tempfile.TemporaryDirectory(prefix='pro6000-figure9-clean-') as clean:
                z.extractall(clean)
                env=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
                subprocess.run([sys.executable,'-S','experiments/tools/report_pro6000_figure9.py','--check'],
                               cwd=clean,env=env,check=True)
    print(json.dumps({'path': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': checksum, 'files': len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
