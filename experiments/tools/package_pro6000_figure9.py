#!/usr/bin/env python3
"""Export the small Figure 9 record overlay without output arrays or binaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import report_pro6000_figure9 as report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
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
    archive = args.output_dir / 'pro6000-figure9-20261118.zip'
    assert not archive.exists(), 'Use a fresh output directory; never overwrite a published asset'
    files = sorted(p for p in report.SNAP.rglob('*') if p.is_file())
    files += sorted((report.ROOT / 'figures').glob('pro6000_figure9*'))
    files += [report.REPORT, Path(__file__),
              report.ROOT / 'experiments/tools/report_pro6000_figure9.py',
              report.ROOT / 'experiments/tools/plot_pro6000_figure9.py']
    inventory = []
    for path in files:
        assert path.suffix not in ('.npz', '.npy', '.so', '.a', '.o')
        raw = path.read_bytes()
        inventory.append({'path': str(path.relative_to(report.ROOT)), 'bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = {
        'schema': 'pro6000-figure9-performance-overlay-v1',
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=report.ROOT, text=True).strip(),
        'request_counts': {'full66': 66, 'fft17-original': 17, 'fft17-final': 17,
                           'mma9-original': 9, 'fixed-mapping': 25},
        'backend_cells': len(tables['backends']), 'local_cpu_records': len(tables['cpu']),
        'files': inventory,
        'scope': 'Record overlay. Reused older snapshots are in the source commit and original performance release. No output-array payloads or binaries.'}
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
    print(json.dumps({'path': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': checksum, 'files': len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
