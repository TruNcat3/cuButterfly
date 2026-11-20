#!/usr/bin/env python3
"""Verify and regenerate frozen evaluation figures without CUDA or network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(manifest: dict) -> int:
    if manifest.get('schema') != 'cubutterfly-paper-evidence-v1':
        raise ValueError('unsupported evidence manifest schema')
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'missing or invalid bundled file: {relative}')
        if digest(path) != expected:
            raise ValueError(f'bundled file changed: {relative}')
    return len(manifest['files'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='check bundled source/data hashes without importing plotting packages')
    parser.add_argument('--figures', nargs='+', choices=['8', '9', '10', '11', '12', 'boundary-table'],
                        default=['8', '9', '10', '11', '12', 'boundary-table'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    verified = verify_bundle(manifest)
    print(f'Verified {verified} bundled files (sanitized bytes, not original source hashes).', flush=True)
    if args.check_only:
        return
    output = args.output_dir.resolve()
    # Keep generated output separate from immutable inputs and runnable source.
    protected = [ROOT / x for x in ('operator_comparison', 'organization_applications', 'ntt_mechanisms')]
    if output == ROOT or any(output == p or output.is_relative_to(p) for p in protected):
        parser.error('--output-dir must be separate from bundled source/data directories')
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.figures)
    runs = []
    for group in manifest['render_groups']:
        if not selected.intersection(group['figures']):
            continue
        destination = output / group['id']
        command = [sys.executable, str(ROOT / group['script']), '--output-dir', str(destination)]
        subprocess.run(command, cwd=ROOT, check=True)
        if group.get('verify_script'):
            subprocess.run([sys.executable, str(ROOT / group['verify_script']),
                            '--output-dir', str(destination)], cwd=ROOT, check=True)
        for expected in group['expected_outputs']:
            if not (destination / expected).is_file():
                raise RuntimeError(f'render did not produce {group["id"]}/{expected}')
        runs.append({'id': group['id'], 'figures': group['figures'],
                     'outputs': {str(p.relative_to(output)): digest(p)
                                 for p in sorted(destination.rglob('*')) if p.is_file()}})
    report = {'schema': 'cubutterfly-paper-reproduction-v1', 'mode': 'offline-frozen-evidence',
              'bundle_manifest_sha256': digest(ROOT / 'manifest.json'),
              'verified_input_files': verified, 'runs': runs,
              'measurement_performed': False,
              'note': 'Rebuilds measured evidence figures; does not rerun GPU experiments or validate a new platform.'}
    (output / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Complete: {len(runs)} render groups; report in {output / "reproduction.json"}')


if __name__ == '__main__':
    main()
