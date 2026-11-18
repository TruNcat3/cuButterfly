"""Figure 9 must preserve failures, exact CPU joins and independent denominators."""
import copy
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/tools'))
import report_pro6000_figure9 as report


@pytest.fixture(scope='module')
def data():
    return report.verify()


@pytest.fixture(scope='module')
def tables():
    return report.render()[0]


def test_complete_populations_and_all_backend_cells(tables):
    counts = Counter(r['cohort'] for r in tables['requests'])
    assert counts == {'full66': 66, 'fft17-original': 17, 'mma9-original': 9,
                      'fft17-final': 17, 'fixed-mapping': 25}
    assert len(tables['backends']) == 134 * 7
    assert len(tables['cpu']) == 83 + 9
    assert Counter(r['operator'] for r in tables['requests'] if r['cohort'] == 'full66') == {
        'FFT': 17, 'NTT': 13, 'FWHT': 13, 'subset-zeta': 8, 'superset-zeta': 8,
        'structured-2x2': 7}


def test_final_fft_is_complete_not_two_point_replacement(tables):
    rows = [r for r in tables['backends'] if r['cohort'] == 'fft17-final'
            and r['backend'] in ('cufft', 'vkfft', 'gpu-fft')]
    assert len(rows) == 51
    assert sum(r['gpu_speedup'] is not None for r in rows) == 47
    gaps = [r for r in rows if r['gpu_speedup'] is None]
    assert len(gaps) == 4
    assert all(r['backend'] == 'gpu-fft' and len(json.loads(r['shape'])) > 1
               and r['reason'] and r['support_status'] == 'unsupported-semantic' for r in gaps)
    assert all(r['gpu_uuid'].removeprefix('GPU-') ==
               '1067ab39-e411-e141-1775-64426bca5701' for r in rows)
    assert all(r['library_sha256'] ==
               'a00188ff3415760359a52eb2e812153e0189c957ce5139c9ed4963d7d15c2c0b' for r in rows)


def test_cpu_join_rejects_input_or_precision_substitution(data):
    _, inputs, _, cpu, mma, _, _ = data
    identifier, item = next(iter(inputs.items()))
    good, _ = report.cpu_reference(item['semantics'], identifier, item['input_sha256'], cpu, mma)
    assert good['backend'] == 'fftw-cpu-t8'
    with pytest.raises(AssertionError, match='input mismatch'):
        report.cpu_reference(item['semantics'], identifier, '0' * 64, cpu, mma)
    wrong = dict(item['semantics'], storage=7)
    with pytest.raises(AssertionError, match='semantic'):
        report.cpu_reference(wrong, identifier, item['input_sha256'], cpu, mma)


def test_external_ntt_failure_does_not_poison_exact_own_or_create_ratio(tables):
    rows = [r for r in tables['backends'] if r['cohort'] == 'full66'
            and r['campaign_status'] == 'correctness-failure']
    assert len(rows) == 3 * 7
    bad = [r for r in rows if r['backend'] == 'gpu-ntt']
    own = [r for r in rows if r['backend'] == 'cubutterfly']
    assert all(r['status'] == 'correctness-failure' and r['gpu_speedup'] is None
               and r['cpu_normalized'] is None and not r['backend_usable'] for r in bad)
    assert all(r['own_usable'] and r['backend_usable'] and r['cpu_normalized'] > 0
               and json.loads(r['correctness'])['exact'] and r['backend_status']=='pass'
               and r['status']=='pass' and r['source_backend_status']=='correctness-failure'
               and not r['reason'] for r in own)


def test_all_derived_statuses_respect_backend_ownership(tables):
    for cohort, expected in [('full66',3),('fixed-mapping',2)]:
        requests=[r for r in tables['requests'] if r['cohort']==cohort and r['campaign_status']=='correctness-failure']
        assert len(requests)==expected
        assert all(r['own_backend_status']=='pass' for r in requests)
        assert not any(r['backend']=='cubutterfly' and r['backend_status']=='correctness-failure'
                       for r in tables['coverage'] if r['cohort']==cohort)
        assert sum(r['requests'] for r in tables['coverage'] if r['cohort']==cohort
                   and r['backend']=='gpu-ntt' and r['backend_status']=='correctness-failure')==expected


def test_statistics_never_pool_cohorts_precisions_or_metrics():
    rows = [dict(cohort=c, operator='FFT', precision=p, backend='cufft',
                 gpu_speedup=v, cpu_normalized=v * 100 if v is not None else None)
            for c, p, v in [('old', 'FP32', 2), ('old', 'FP32', 8),
                            ('new', 'FP32', .25), ('old', 'FP64', 16),
                            ('new', 'FP32', None)]]
    result = report.statistics_rows(rows)
    by_key = {(r['cohort'], r['precision'], r['metric']): r for r in result}
    assert len(result) == 6
    assert by_key['old', 'FP32', 'gpu_speedup']['geomean'] == pytest.approx(4)
    assert by_key['old', 'FP32', 'gpu_speedup']['pairs'] == 2
    assert by_key['new', 'FP32', 'gpu_speedup']['pairs'] == 1
    assert by_key['old', 'FP32', 'cpu_normalized']['geomean'] == pytest.approx(400)


def test_references_and_identity_registry_keep_original_roles(tables, data):
    _, inputs, _, _, _, _, _ = data
    for row in tables['requests']:
        if row['cohort'].startswith('fft17-'):
            assert row['reference_sha256'] == inputs[row['case_id']]['fft17_reference_sha256']
            assert 'native-precision' in row['reference_role']
        assert row['source_identity_sha256'] in tables['identities']
        assert row['cpu_identity_sha256'] in tables['identities']
    assert all(report.digest(v) == k for k, v in tables['identities'].items())
    assert all(r['a100_input_sha256'] is None for r in tables['a100_intersections'])
    assert all(r['a100_over_pro6000'] is None or r['exact_semantic_match']
               for r in tables['a100_intersections'])


def test_cpu_adapted_contract_is_explicit(tables):
    rows = [r for r in tables['cpu'] if r['cpu_cohort'] == 'mma9-fp16-adapted']
    assert len(rows) == 9
    for row in rows:
        meta = tables['identities'][row['metadata_sha256']]
        assert not meta['native_fp16'] and meta['external_core_storage'] == 'float32'
        assert meta['all_conversion_and_scaling_timed']
        assert row['status'] == 'accuracy-review-required'


def test_full_population_cache_binding_and_checksums_cannot_be_dropped(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / 'snapshot'
    shutil.copytree(report.SNAP, root)
    manifest = json.loads((root / 'manifest.json').read_text())
    record = manifest['fft17_final']['rows'][0]
    path = root / record['path']
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    envelope['identity']['runtime_bindings'].pop(next(
        k for k in envelope['identity']['runtime_bindings'] if k.endswith('plans.json')))
    envelope['identity_sha256'] = report.digest(envelope['identity'])
    raw = json.dumps(envelope, sort_keys=True).encode()
    packed = gzip.compress(raw, mtime=0)
    path.write_bytes(packed)
    record.update(sha256=report.sha(packed), uncompressed_sha256=report.sha(raw))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(report, 'SNAP', root)
    with pytest.raises((AssertionError, KeyError)):
        report.verify()


def test_derived_json_contains_exact_csv_table_population(tables):
    payload = json.loads(gzip.decompress((report.SNAP / 'tables.json.gz').read_bytes()))
    assert payload == tables
