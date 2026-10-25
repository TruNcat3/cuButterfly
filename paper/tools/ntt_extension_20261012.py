#!/usr/bin/env python3
"""Freeze and run the explicit twelve-contract word64 cyclic NTT cohort."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import run_semantic_campaign as runner
import semantic_campaign_manifest as manifests
import semantic_preparation_queue as preparation

OUTPUT = ROOT / 'results/paper_targeted_20261012/ntt_extension'
E1 = ROOT / 'results/paper_targeted_20261011/operator_prepare/e1_fixed_36_v4/manifest.json'
LIBRARY = ROOT / 'results/search_budget_20261011/frozen_runtime/libcuntt.so'
CALIBRATION = ROOT / 'results/plan_batch_trace_20261010/recalibration_gpu2_finalized_bound_20261011/calibration_manifest.json'
BROKER = ROOT / 'results/plan_batch_trace_20261010/shared_compile'
UUID = 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'
LIBRARY_SHA = '24486975585fa2ab6399a0cfeae4b9449c398490cb76c5ea5a73a3560abe8c66'
BOUNDS = {'reference_max_elements': 1 << 24, 'ntt_max_operations': 1 << 27}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def mapping_selector(original, mappings):
    def selected_descriptor(descriptor, timing=False, mapping=None):
        if timing and mapping is None:
            mapping = mappings[runner.digest(descriptor['semantics'])]
        return original(descriptor, timing, mapping)
    return selected_descriptor


def freeze():
    if runner.sha256_file(LIBRARY) != LIBRARY_SHA:
        raise ValueError('frozen runtime SHA mismatch')
    original = json.loads(E1.read_text())
    populations = []
    for batch in (1, 16, 256):
        for source in original['config']['populations']:
            if source['regimes'][0]['transform'].get('word_bits') != 64:
                continue
            item = copy.deepcopy(source)
            item['batches'] = [batch]
            item['name'] = ('original4' if batch == 1 else 'additional8') + '-N' + str(item['shapes'][0][0]) + '-B' + str(batch) + '-' + item['directions'][0]
            populations.append(item)
    config = {'schema': manifests.CONFIG_SCHEMA, 'populations': populations}
    save(OUTPUT / 'config.json', config)
    manifest = manifests.generate_manifest(config, LIBRARY, config_path=OUTPUT / 'config.json')
    assert len(manifest['cases']) == 12 and not manifest['invalid_requests']
    old_ids = {c['id'] for c in original['cases'] if c['descriptor']['semantics']['word_bits'] == 64}
    assert old_ids == {c['id'] for c in manifest['cases'] if c['descriptor']['semantics']['batch'] == 1}
    manifest['paper_cohort'] = {'original4': sorted(old_ids), 'additional8': [c['id'] for c in manifest['cases'] if c['id'] not in old_ids]}
    manifest['manifest_sha256'] = manifests.manifest_hash(manifest)
    save(OUTPUT / 'manifest.json', manifest)
    calibration = json.loads(CALIBRATION.read_text())
    profile = json.loads(Path(calibration['hardware_rates_source']).read_text())
    assert profile['provenance']['cuda_visible_devices'] == UUID
    assert calibration['measurement_identity']['library_sha256'] == LIBRARY_SHA
    save(OUTPUT / 'freeze.json', {'manifest_sha256': manifest['manifest_sha256'], 'library_sha256': LIBRARY_SHA,
         'wrapper_sha256': runner.sha256_file(__file__), 'gpu_uuid': UUID,
         'calibration': str(CALIBRATION), 'calibration_sha256': runner.digest(calibration),
         'protocol': dict(runner.PROTOCOL, **BOUNDS), 'independent_process_trials': 3,
         'baseline': 'GPU-NTT Merge only', 'bounds_reason': 'all elements and radix-two butterflies for exact N65536 B256 oracle',
         'separate_groups': ['original4', 'additional8'], 'search_budget': 512})
    preparation.initialize(OUTPUT / 'prepared', manifest, **BOUNDS)
    print(json.dumps({'status': 'frozen', 'cases': 12, 'gpu_uuid': UUID}))


def prepare():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    result = preparation.prepare_manifest(manifest, OUTPUT / 'prepared', jobs=1)
    save(OUTPUT / 'cpu_preparation.json', result)
    print(json.dumps(result))


def trial(number):
    if number not in (1, 2, 3):
        raise ValueError('only three predeclared trials')
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    for case in manifest['cases']:
        record = preparation.record_for(OUTPUT / 'prepared', case['id'])
        if not record or record['status'] != 'ready':
            raise ValueError('all CPU oracles must be ready before GPU execution')
    calibration = json.loads(CALIBRATION.read_text())
    os.environ['LD_LIBRARY_PATH'] = calibration['measurement_identity']['jit_runtime']['ld_library_path']
    os.environ['CUBUTTERFLY_SEARCH_BUDGET'] = '512'
    runner.PROTOCOL.update(BOUNDS)
    import run_async_semantic_campaign as asynchronous
    if number > 1:
        first = OUTPUT / 'trial-1'
        report = json.loads((first / 'results.json').read_text())
        if report['status'] != 'complete' or report['remaining_case_ids']:
            raise ValueError('trial1 must complete before freezing selected mappings')
        rows = {}
        for entry in report['rows']:
            document = json.loads((first / entry['path']).read_text())
            if document['sha256'] != runner.digest(document['row']) or entry['sha256'] != document['sha256']:
                raise ValueError('trial1 row checksum mismatch')
            row = document['row']
            if row['status'] != 'pass':
                raise ValueError('only exact passing selected mappings can be replayed')
            rows[row['id']] = row
        selected = {'selection_source': 'frozen_trial1', 'scope': 'selected mapping stability across independent processes; not search variability',
                    'trial1_report_sha256': runner.sha256_file(first / 'results.json'),
                    'wrapper_sha256': runner.sha256_file(__file__),
                    'mappings': {key: row['mapping'] for key, row in rows.items()},
                    'row_sources': {entry['id']: {'path': str(first / entry['path']), 'row_sha256': entry['sha256']} for entry in report['rows']}}
        selected_path = OUTPUT / 'selected_mappings.json'
        if selected_path.exists():
            if json.loads(selected_path.read_text()) != selected:
                raise ValueError('frozen selected mappings identity changed')
        else:
            save(selected_path, selected)
        runner.selection_descriptor = mapping_selector(runner.selection_descriptor, selected['mappings'])
        save(OUTPUT / f'trial-{number}-selection.json', selected)
    return asynchronous.main(['--manifest', str(OUTPUT / 'manifest.json'), '--library', str(LIBRARY),
        '--output', str(OUTPUT / f'trial-{number}'), '--preparation', str(OUTPUT / 'prepared'),
        '--compile-broker', str(BROKER), '--gpu-uuid', UUID,
        '--calibration-manifest', str(CALIBRATION),
        '--gpuntt-library', str(ROOT / 'build-e13-cufftdx/libgpuntt_semantic_bridge.so'),
        '--timing', '--cpu-workers', '1', '--in-flight', '1', '--worker-max-rss-bytes', str(16 * 1024**3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'prepare', 'trial'))
    parser.add_argument('--number', type=int)
    args = parser.parse_args()
    return freeze() if args.mode == 'freeze' else prepare() if args.mode == 'prepare' else trial(args.number)


if __name__ == '__main__':
    raise SystemExit(main())
