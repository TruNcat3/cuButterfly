#!/usr/bin/env python3
"""Reclassify frozen numeric evidence without changing its verdict or artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_fft_range_control as control
from semantic_input_profiles import _digest
from semantic_real_input_profiles import classify_numeric_failure

r = control.r


def audit(paths, library):
    rows, seen = [], set()
    for source_path in paths:
        path = Path(source_path).resolve(strict=True)
        row, checksum = control.load_source_row(path)
        if row['status'] != 'correctness-failure':
            raise ValueError('audit requires original correctness-failure rows')
        report = json.loads((path.parent.parent / 'results.json').read_text())
        binding = SimpleNamespace(row=path, library=library,
                                  gpu_uuid=report['identity']['gpu_uuid'])
        evidence = control.bind_source_measurement(binding, row, checksum)
        key = (binding.gpu_uuid, row['id'])
        if key in seen:
            raise ValueError('duplicate GPU/semantic request')
        seen.add(key)
        s = row['descriptor']['semantics']
        # Overflow is evidence here, never an instruction to modify the source.
        with np.errstate(over='ignore', invalid='ignore'):
            source = r.make_source(s)
        with np.load(row['error_artifact']['path'], allow_pickle=False) as artifact:
            classification = classify_numeric_failure(
                source, artifact['reference'], artifact['actual'], r.side_storage(s, 'output'))
        rows.append({'id': row['id'], 'gpu_uuid': binding.gpu_uuid,
                     'row_path': str(path), 'row_checksum': checksum,
                     'original_status': row['status'],
                     'original_diagnostics': row.get('numeric_diagnostics'),
                     'diagnostics': classification,
                     'source_sha256': _digest(source),
                     'source_evidence': evidence,
                     'original_error_artifact': row['error_artifact']})
    return {'schema': 'cubutterfly-numeric-evidence-audit-v2',
            'verdict_policy': 'original failures unchanged; classification is not acceptance',
            'library_sha256': r.sha256_file(library),
            'generator_sha256': r.sha256_file(r.__file__),
            'classifier_sha256': r.sha256_file(Path(__file__).with_name('semantic_real_input_profiles.py')),
            'counts': dict(Counter(row['diagnostics']['classification'] for row in rows)),
            'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, nargs='+', required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('audit output already exists')
    result = audit(args.rows, args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'counts': result['counts']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
