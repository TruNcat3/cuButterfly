"""Missing bindings can be filled without expanding or changing the population."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import supplement_semantic_gpuntt as supplement


def row(identifier, status='missing-baseline', support='missing-external-binding', bits=32):
    return dict(id=identifier, status='pass', descriptor={'semantics': dict(
        operator=1, shape=[65536], batch=7, word_bits=bits, modulus=998244353, inverse=0)},
        timing=dict(median_ms=2.0, protocol='frozen'), baselines=[dict(
        name='gpu-ntt', status=status, support_status=support,
        timing=dict(median_ms=3.0, protocol='frozen'))])


def test_only_missing_bindings_are_selected():
    rows = [row('missing'), row('valid', 'pass', 'supported'),
            row('range', support='unsupported-semantic'),
            row('failed', 'correctness-failure', 'supported')]
    original = copy.deepcopy(rows)
    assert [r['id'] for r in supplement.select_missing(rows)] == ['missing']
    assert rows == original


def test_unverified_library_trial_cannot_be_used_as_pair():
    trial = row('unverified')
    trial['status'] = 'correctness-failure'
    with pytest.raises(ValueError, match='validated NTT'):
        supplement.select_missing([trial])


def test_merged_statistics_keep_unsupported_out_and_originals_unchanged():
    rows = [row('missing'), row('valid', 'pass', 'supported', bits=64),
            row('range', support='unsupported-semantic')]
    original = copy.deepcopy(rows)
    replacement = row('missing', 'pass', 'supported')
    replacement['baselines'][0]['timing']['median_ms'] = 1.0
    summary = supplement.summarize(rows, [replacement])
    assert rows == original
    assert summary['paired']['count'] == 2
    assert summary['paired']['wins'] == 1
    assert summary['paired']['geometric_mean'] == pytest.approx((1.5 * 0.5) ** 0.5)
    assert summary['original_pairs']['geometric_mean'] == 1.5
    assert summary['supplemented_pairs']['geometric_mean'] == 0.5
    assert summary['unpaired'][0]['id'] == 'range'


def test_different_timing_protocol_cannot_be_merged():
    original = row('missing')
    replacement = row('missing', 'pass', 'supported')
    replacement['baselines'][0]['timing']['protocol'] = 'another'
    with pytest.raises(ValueError, match='protocols differ'):
        supplement.summarize([original], [replacement])


def test_source_row_corruption_is_detected(tmp_path):
    trial = row('missing')
    index = supplement.runner.write_case_checkpoint(tmp_path, trial)
    report = dict(status='complete-with-gaps', remaining_case_ids=[], rows=[index],
                  rows_sha256=supplement.runner.digest([index]), identity={'manifest_sha256': 'abc'})
    source_path, manifest_path = tmp_path / 'results.json', tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(dict(manifest_sha256='abc', cases=[dict(
        id=trial['id'], descriptor=trial['descriptor'])])))
    report['identity']['manifest_sha256'] = supplement.runner.digest(json.loads(manifest_path.read_text()))
    source_path.write_text(json.dumps(report))
    corrupted = dict(trial, timing={'median_ms': 0.01})
    (tmp_path / index['path']).write_text(json.dumps(dict(row=corrupted, sha256=index['sha256'])))
    with pytest.raises(ValueError, match='checksum'):
        supplement.load_source(source_path, manifest_path)


def test_resource_limit_checkpoint_is_not_accepted_as_completed(tmp_path, monkeypatch):
    source_path = tmp_path / 'source.json'
    source_path.write_text('{}')
    gpu_uuid = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'
    source = dict(identity=dict(gpu_uuid=gpu_uuid, protocol={}, library=str(source_path),
        library_sha256=supplement.runner.sha256_file(source_path)))
    monkeypatch.setattr(supplement, 'load_source', lambda *args: (source, {}, [row('oom')]))
    monkeypatch.setattr(supplement.gpuntt, 'backend_inventory', lambda *args: {'available': True})
    # The common runner intentionally accepts resource-limit as a terminal gap.
    # The baseline supplement must apply its stricter all-pass rule on top.
    monkeypatch.setattr(supplement.runner, 'validate_resume', lambda *args: {'oom'})
    (tmp_path / 'results.json').write_text(json.dumps(dict(rows=[dict(id='oom', status='resource-limit')])))
    args = SimpleNamespace(output=tmp_path, source_report=source_path, manifest=source_path,
                           gpu_uuid=gpu_uuid, gpuntt_library=source_path)
    with pytest.raises(ValueError, match='failed trial'):
        supplement.run(args)


def test_saved_numerical_failure_blocks_automatic_retry(tmp_path):
    (tmp_path / 'verification_failure.json').write_text('{}')
    with pytest.raises(ValueError, match='verification failure'):
        supplement.run(SimpleNamespace(output=tmp_path))
