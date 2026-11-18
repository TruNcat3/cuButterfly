"""Verify the frozen two-request search regression and its paired denominator."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments/snapshots/pro6000_search_fix_20261118'


@pytest.fixture(scope='module')
def snapshot():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    decoded = {}
    for record in manifest['records']:
        raw = (ROOT/record['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['sha256']
        raw = gzip.decompress(raw)
        assert hashlib.sha256(raw).hexdigest() == record['uncompressed_sha256']
        decoded[record['original']] = raw
    return manifest, decoded


def test_search_budget_and_projection_evidence(snapshot):
    manifest, decoded = snapshot
    assert manifest['original_scientific_files_unchanged'] == 57
    searches = [raw for path, raw in decoded.items() if path.endswith('.jsonl')]
    assert len(searches) == 2
    for raw in searches:
        events = [json.loads(line) for line in raw.splitlines()]
        summary = next(e for e in events if e['phase'] == 'summary')
        assert summary['accepted'] == summary['attempted'] == summary['search_budget'] == 16
        assert summary['rejected'] == 0 and summary['probe_limit'] == 64
        assert summary['projection_rejected_candidate_count'] == 8
        results = [e for e in events if e['phase'] == 'candidate_result']
        assert len(results) == 16
        assert all(e['outcome'] == 'accepted' for e in results)
        assert any(e['mapping']['backend'] == 'online-reorder' and e['mapping']['local_stages'] == 6
                   for e in results)


@pytest.mark.parametrize('inverse', [0, 1])
def test_automatic_replay_and_three_paired_rounds(snapshot, inverse):
    manifest, decoded = snapshot
    paired = [json.loads(raw) for path, raw in decoded.items()
              if '/paired/trial' in path and f'-{inverse}-' in Path(path).name]
    assert len(paired) == 9
    sources = set()
    for row in paired:
        assert row['guards'] == {'before': True, 'after': True}
        assert row['gpu_uuid'] == manifest['gpu_uuid']
        assert row['actual_library']['sha256'] == manifest['library_sha256']
        assert row['status'] == 'complete-diagnostic-accuracy-review-required'
        timing = row['timing']
        assert len(timing['samples_ms']) == 7 and all(v > 0 for v in timing['samples_ms'])
        assert statistics.median(timing['samples_ms']) == timing['median_ms']
        sources.add((row['source_envelope_sha256'], row['payload_sha256']))
        if row['backend'] == 'default':
            assert row['actual_mapping']['stage_partition'] == [6, 6]
            assert row['actual_mapping']['compute_unit'] == 'radix2'
            assert 'cache' in row['selection_reason']
        elif row['backend'] == 'recorded':
            assert row['actual_mapping'] == row['original_mapping']
        else:
            assert row['backend'] == 'gpu-fft'
    assert len(sources) == 1
    search = next(json.loads(raw) for path, raw in decoded.items()
                  if path.endswith(f'/{inverse}-remeasure-timing.json'))
    replay = next(row for row in paired if row['backend'] == 'default')
    assert search['actual_mapping'] == replay['actual_mapping']
