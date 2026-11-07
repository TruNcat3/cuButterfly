#!/usr/bin/env python3
"""Load frozen FFT-L public replay and separate fixed controls without a GPU.

Usage: ``from load_fft_l_comparison import load_fft_l_comparison``;
``snapshot = load_fft_l_comparison()`` returns requests, pairs, summary, identity.
Default requests/pairs contain only one public winner replay. control_pairs and
control_summary preserve the separate 22-request fixed control. Requests/pairs use
the current-refresh operator schema, with a separate cohort;
callers must explicitly include this cohort and must not relabel it main66/AUTO.
Run this module to regenerate comparison_summary.json and comparison_pairs.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT / 'paper/generated/fft_l_20261104'
FROZEN_CSV_SHA256 = '3bfe5f7fb86e2da4da76429be1baa34803e3d8f6eb8defefc87d65d09804b93c'
SELECTION = 'fixed-two-band-control'
COHORT = 'fft_l_fixed_control22'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    number = float(value)
    require(math.isfinite(number) and number > 0, 'Invalid latency')
    return number


def load_fft_l_comparison(directory=None):
    """Return a JSON-serializable snapshot. directory may hold an offline copy.

    summary contains public replay coverage; control_summary retains 11 fixed pairs per precision.
    identity includes the public replay; it is excluded from fixed-control aggregates.
    Frozen CSV is pinned here; provenance originally hashes raw sources only.
    Its index-aligned scale_contracts independently bind semantics and timings.
    """
    directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
    csv_path = directory / 'scale_full.csv'
    provenance_path = directory / 'provenance.json'
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    require(csv_hash == FROZEN_CSV_SHA256, 'Frozen scale_full.csv SHA256 mismatch')
    provenance = json.loads(provenance_path.read_text())
    rows = list(csv.DictReader(csv_path.open(newline='')))
    contracts = provenance['scale_contracts']
    require(len(rows) == len(contracts) == provenance['scale_rows'] == 111,
            'Frozen matrix row count mismatch')
    sources = {item['path']: item['sha256'] for item in provenance['source_files']}
    require(len(sources) == len(provenance['source_files']), 'Duplicate provenance sources')
    index = {}
    for i, (row, contract) in enumerate(zip(rows, contracts)):
        key = (row['case'], row['variant'])
        require(key not in index, 'Duplicate case/variant')
        require(row['summary_pointer'] == f'/rows/{i}', 'Contract alignment mismatch')
        require(row['source'] in sources, 'Missing hashed source')
        semantics = contract['semantics']
        n, batch = int(row['N']), int(row['B'])
        require(semantics['shape'] == [n] and semantics['batch'] == batch
                and semantics['storage'] == {'fp16': 4, 'bf16': 5}[row['precision']]
                and semantics['operator'] == 0 and semantics['inverse'] == 0
                and semantics['normalize_inverse'] is False
                and semantics['fft_output_scale'] == 1.0, 'Request semantics mismatch')
        require(contract['library_sha256'] == row['library_sha256'], 'Library mismatch')
        if row['graph_us']:
            require(row['all_outputs_finite'] == 'True', 'Nonfinite outputs')
            require(contract['gpu'] == {'name': 'NVIDIA A100 80GB PCIe',
                                       'memory_bytes': 84974239744}, 'Hardware mismatch')
            for field, timing in [('graph_us', 'graph_timing'), ('direct_us', 'direct_timing')]:
                measured = positive(row[field])
                evidence = contract[timing]
                require(math.isclose(measured, positive(evidence['median_ms']) * 1000,
                                     rel_tol=1e-12), 'Provenance timing mismatch')
                require(math.isclose(statistics.median(evidence['samples_ms']),
                                     evidence['median_ms'], rel_tol=1e-12), 'Median mismatch')
        index[key] = (row, contract)
    cases = sorted({row['case'] for row in rows})
    require(len(cases) == provenance['scale_requests'] == 22, 'Request denominator mismatch')
    require(sum(bool(row['graph_us']) for row in rows) == 87, 'Timing coverage mismatch')

    def pair(case, variant):
        own, oc = index[(case, variant)]
        vendor, vc = index[(case, 'cufft')]
        for field in ('semantics', 'gpu', 'gpu_uuid', 'prepared_sha256',
                      'library_sha256', 'vendor_sha256', 'precision_contract'):
            require(oc[field] == vc[field] and oc[field] is not None,
                    f'Pair contract mismatch: {case}/{field}')
        require(len(oc['prepared_sha256']) == 64, 'Missing prepared identity')
        mapping = json.loads(own['mapping_json'])
        require(mapping['contract'] == oc['semantics'], 'Native mapping contract mismatch')
        own_us, baseline_us = positive(own['graph_us']), positive(vendor['graph_us'])
        return dict(operator='fft', precision=own['precision'], case=case,
                    cohort=COHORT, selection_kind=SELECTION, variant=variant,
                    status='accuracy-review-required', source=own['source'],
                    source_sha256=sources[own['source']], baseline='cufft',
                    baseline_source=vendor['source'],
                    baseline_source_sha256=sources[vendor['source']],
                    own_us=own_us, baseline_us=baseline_us, speedup=baseline_us / own_us,
                    N=int(own['N']), B=int(own['B']), actualN=int(own['N']), actualB=int(own['B']),
                    library_hash=own['library_sha256'], mapping_json=own['mapping_json'],
                    semantics=oc['semantics'], gpu=oc['gpu'], gpu_uuid=oc['gpu_uuid'],
                    prepared_sha256=oc['prepared_sha256'],
                    precision_contract=oc['precision_contract'],
                    native_l2=float(own['native_l2']), mathematical_l2=float(own['mathematical_l2']))

    pairs = [pair(case, 'two-grouped') for case in cases]
    summary = []
    for precision in ('fp16', 'bf16'):
        values = [p['speedup'] for p in pairs if p['precision'] == precision]
        require(len(values) == 11, 'Per-precision denominator mismatch')
        summary.append(dict(operator='fft', precision=precision, baseline='cufft',
                            cohort=COHORT, selection_kind=SELECTION, pairs=11,
                            gm=math.exp(statistics.mean(map(math.log, values))),
                            wins=sum(v > 1 for v in values), min=min(values), max=max(values)))
    replay_keys = [case for case, variant in index if variant == 'public-replay']
    require(len(replay_keys) == 1, 'Expected exactly one public replay')
    replay = pair(replay_keys[0], 'public-replay')
    replay.update(cohort='fft_l_public_winner_replay', selection_kind='public_winner_replay',
                  included_in_control_summary=False)
    identity = dict(schema='fft-l-comparison-v2', selection_kind='public_winner_replay',
                    variant='public-replay', cohort='fft_l_public_winner_replay', requests=1,
                    fixed_control_scope=dict(selection_kind=SELECTION, cohort=COHORT,
                                             variant='two-grouped', requests=22),
                    source_requests=22, timed_rows=87,
                    missing_rows=24, frozen_rows=111, csv_sha256=csv_hash,
                    provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                    csv_hash_binding='Pinned loader hash; original provenance hashes raw sources, not CSV',
                    public_replay=replay,
                    limitations=['Archived controls fix the variant across 22 requests; no per-request minimum selection',
                                 'Not an AUTO matrix; repaired AUTO calibration is incomplete',
                                 'Separate from main66; builds and selection protocols differ',
                                 'Native storage arithmetic; vendor internal arithmetic unspecified',
                                 'Finite outputs with reported errors; accuracy acceptance remains pending',
                                 'Prepared inputs repeat one vector across batches; not independent random batches',
                                 'Frozen evidence adaptation only; no new experiments'])
    public_summary = [dict(operator='fft', precision='fp16', baseline='cufft',
                           cohort='fft_l_public_winner_replay', selection_kind='public_winner_replay',
                           pairs=1, gm=replay['speedup'], wins=int(replay['speedup'] > 1),
                           min=replay['speedup'], max=replay['speedup']),
                      dict(operator='fft', precision='bf16', baseline='cufft',
                           cohort='fft_l_public_winner_replay', selection_kind='public_winner_replay',
                           pairs=0, gm=None, wins=0, min=None, max=None, status='pending')]
    identity.update(default_selection_kind='public_winner_replay', default_requests=1,
                    bf16_status='pending; repaired AUTO calibration and 32-request remeasurement incomplete',
                    evidence_classes=['public_winner_replay', 'fixed_control'])
    return dict(requests=[dict(replay)], pairs=[replay], summary=public_summary,
                control_pairs=pairs, control_summary=summary, identity=identity)


def main():
    snapshot = load_fft_l_comparison()
    (DEFAULT_DIRECTORY / 'comparison_summary.json').write_text(
        json.dumps(dict(summary=snapshot['summary'], control_summary=snapshot['control_summary'], identity=snapshot['identity']), indent=2) + '\n')
    with (DEFAULT_DIRECTORY / 'comparison_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshot['pairs'][0]))
        writer.writeheader()
        for pair in snapshot['pairs'] + snapshot['control_pairs']:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in pair.items()})
    print(json.dumps(snapshot['summary'], indent=2))


if __name__ == '__main__':
    main()
