"""Finite NTT8 support, per-backend failures, and unbiased CPU aggregation."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/tools'))
import report_pro6000_figure9 as prior
import report_pro6000_figure9_followup as followup


@pytest.fixture(scope='module')
def tables():return followup.render()[0]


def test_predeclared_eight_exact_a100_semantic_intersections(tables):
    assert len(tables['ntt8_backends'])==16 and len(tables['a100_intersections'])==8
    assert all(r['exact_semantic_match'] for r in tables['a100_intersections'])
    cells={(r['word_bits'],r['batch'],r['inverse']) for r in tables['ntt8_backends']}
    assert cells=={(bits,batch,direction) for bits in (32,64) for batch in (1,16) for direction in (0,1)}
    assert {r['modulus'] for r in tables['ntt8_backends'] if r['word_bits']==32}=={1073479681}
    assert {r['modulus'] for r in tables['ntt8_backends'] if r['word_bits']==64}=={582090251837636609}


def test_only_six_exact_external_pairs_have_speedup(tables):
    baseline=[r for r in tables['ntt8_backends'] if r['backend']=='gpu-ntt']
    assert sum(r['gpu_speedup'] is not None for r in baseline)==6
    failed=[r for r in baseline if r['backend_status']=='correctness-failure']
    assert len(failed)==2 and all(r['word_bits']==64 and not r['inverse'] for r in failed)
    assert all(r['latency_us'] is None and r['cpu_normalized'] is None and r['reason'] for r in failed)
    assert all(r['backend_status']=='pass' and json.loads(r['correctness'])['exact']
               and json.loads(r['replay_correctness'])['exact'] for r in tables['ntt8_backends'] if r['backend']=='cubutterfly')
    assert {(r['precision'],r['pairs']) for r in tables['statistics'] if r['metric']=='gpu_speedup'}=={('word32',4),('word64',2)}


def test_cpu_uses_three_process_medians_and_keeps_old_values(tables):
    assert len(tables['cpu_confirmation'])==83 and len(tables['cpu_processes'])==249
    assert len({r['pid'] for r in tables['cpu_processes']})==3
    assert sum(r['old_us'] is not None for r in tables['cpu_confirmation'])==75
    for row in tables['cpu_confirmation']:
        medians=json.loads(row['process_medians_us'])
        assert len(medians)==3
        assert row['median_of_medians_us']==statistics.median(medians)
        assert row['minimum_process_us']==min(medians) and row['maximum_process_us']==max(medians)
        assert row['status']=='provisional-shared-host'
        if row['operator']=='FFT':assert row['backend']=='fftw-cpu-t8'
    assert any(r['median_of_medians_us']>r['minimum_process_us'] for r in tables['cpu_confirmation'])


def test_normalization_reuses_gpu_timings_and_exact_cpu_inputs(tables):
    old,_=prior.render()
    old_rows={(r['cohort'],r['case_id'],r['backend']):r for r in old['backends']}
    assert len(tables['normalized_backends'])==75*7+8*2
    assert {r['case_id'] for r in tables['normalized_backends']}=={r['case_id'] for r in tables['cpu_confirmation']}
    for row in tables['normalized_backends']:
        key=row['cohort'],row['case_id'],row['backend']
        if key in old_rows:
            assert row['latency_us']==old_rows[key]['latency_us']
            assert row['gpu_speedup']==old_rows[key]['gpu_speedup']
            assert row['input_sha256']==old_rows[key]['input_sha256']
        if row['cpu_normalized'] is not None:
            assert row['cpu_normalized']==row['cpu_reference_us']/row['latency_us']


def test_metadata_only_fixed_verifier_does_not_claim_array_replay(tmp_path):
    root=tmp_path/'fixed';shutil.copytree(prior.fixed.DEFAULT,root)
    for path in (root/'errors').glob('*.npz'):path.unlink()
    with pytest.raises(FileNotFoundError):prior.fixed.verify(root)
    rows,*_=prior.fixed.verify(root,require_array_payloads=False)
    assert len(rows)==25
    provenance=json.loads((root/'provenance.json').read_text())
    provenance['bundled_error_arrays'][0]['sha256']='0'*64
    (root/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(AssertionError):prior.fixed.verify(root,require_array_payloads=False)


def test_metadata_only_extension_checks_array_receipts_without_payloads(tmp_path):
    root=tmp_path/'extensions';shutil.copytree(prior.extensions.DEFAULT,root)
    for path in (root/'array_samples').glob('*.npz'):path.unlink()
    with pytest.raises(FileNotFoundError):prior.extensions.verify(root)
    manifest,data=prior.extensions.verify(root,require_array_payloads=False)
    assert len(data['fft17']['records'])==17 and len(data['mma9']['records'])==9
    manifest['array_samples'][0]['sha256']='0'*64
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):prior.extensions.verify(root,require_array_payloads=False)


def test_published_json_matches_verified_derived_tables(tables):
    assert json.loads(gzip.decompress((followup.SNAP/'tables.json.gz').read_bytes()))==tables


def test_status_fix_preserves_all_prior_78_numeric_statistics():
    # Original statistics.csv from reviewed source commit 0c451388f598ec6b519ee315b3c120495d771b59.
    assert prior.sha((prior.SNAP/'statistics.csv').read_bytes())=='fd7c515ee82116b107871112ded6d6ef8aa0290bea06b8c23b4f414d9f144ab9'


def test_corrupted_followup_record_is_rejected(tmp_path,monkeypatch):
    root=tmp_path/'snapshot';shutil.copytree(followup.SNAP,root)
    manifest=json.loads((root/'manifest.json').read_text())
    record=manifest['cyclic_ntt8']['rows'][0]
    path=root/record['path'];path.write_bytes(path.read_bytes()[:-1])
    monkeypatch.setattr(followup,'SNAP',root)
    with pytest.raises(AssertionError):followup.verify()
