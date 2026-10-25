#!/usr/bin/env python3
"""Regenerate a build-bound manifest without changing its frozen experiment scope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import semantic_campaign_manifest as campaign
import semantic_preparation_queue as preparation


def _write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('comparison binding already exists with different inputs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(source_path, config_path, library_path, preparation_dir, output_path, receipt_path):
    paths = [Path(p).resolve() for p in
             (source_path, config_path, library_path, preparation_dir, output_path, receipt_path)]
    source_path, config_path, library_path, preparation_dir, output_path, receipt_path = paths
    if len(set(paths)) != len(paths):
        raise ValueError('comparison binding outputs must be separate from source inputs')
    source = json.loads(source_path.read_text())
    if not campaign.validate_manifest_hash(source):
        raise ValueError('source manifest checksum mismatch')
    config = campaign.load_config(config_path)
    if config != source['config']:
        raise ValueError('comparison configuration changed')
    # Query public descriptors and capabilities again. Merely replacing an old
    # library hash could conceal changes to lowering support or normalization.
    current = campaign.generate_manifest(config, library_path, config_path=config_path)
    scope = lambda value: {k: v for k, v in value.items()
                           if k not in ('fingerprints', 'manifest_sha256')}
    if scope(source) != scope(current):
        raise ValueError('current public descriptor/capability scope differs from source manifest')
    prepared, prepared_sha = preparation._config(preparation_dir)
    if prepared['source_sha256'] != preparation._sources():
        raise ValueError('CPU preparation source identity changed')
    if prepared['manifest'] != source:
        raise ValueError('CPU preparation belongs to another source manifest')
    records = {}
    for case in current['cases']:
        record = preparation.record_for(preparation_dir, case['id'])
        if record is None or record['status'] == 'failed':
            raise ValueError('CPU preparation is not reusable: ' + case['id'])
        if record['case_sha256'] != campaign.semantic_id(case):
            raise ValueError('CPU preparation case changed: ' + case['id'])
        records[case['id']] = campaign.semantic_id(record)
    receipt = dict(schema='cubutterfly-comparison-build-binding-v1',
        source_manifest=campaign.fingerprint(source_path),
        source_manifest_sha256=campaign.semantic_id(source),
        current_manifest_sha256=campaign.semantic_id(current),
        source_library=source['fingerprints']['library'],
        current_library=current['fingerprints']['library'],
        scope_sha256=campaign.semantic_id(scope(current)), cases=len(current['cases']),
        preparation=str(preparation_dir), preparation_config_sha256=prepared_sha,
        preparation_records=records,
        policy='same complete experiment scope; retain original CPU inputs and measurements; GPU timings belong to current build')
    _write_once(output_path, current)
    _write_once(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'config', 'library', 'preparation', 'output', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(args.source_manifest, args.config, args.library, args.preparation,
                      args.output, args.receipt)
    print(json.dumps({key: receipt[key] for key in ('cases', 'current_manifest_sha256', 'policy')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
