import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quarot_followup', ROOT / 'scripts/run_quarot_fused_followup.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def jobs(tmp_path):
    library, extension = tmp_path / 'lib.so', tmp_path / 'extension.so'
    library.write_bytes(b'library')
    extension.write_bytes(b'extension')
    extension.with_name('build_receipt.json').write_text(json.dumps({
        'protocol': 'quarot_current_stream_only_v1', 'extension_sha256': queue.digest(extension)}))
    return queue.prepare(queue.load_config(queue.DEFAULT_CONFIG), library, extension, 'python', 'GPU-test')


def test_fixed_config_preserves_complete_precision_and_batch_contract(tmp_path):
    prepared = jobs(tmp_path)
    assert len(prepared) == 4
    for job in prepared:
        command = job['command']
        values = dict(zip(command[2::2], command[3::2]))
        assert values['--compute-contract'] == 'public_fp32_final'
        assert values['--execution-mode'] == 'graph'
        assert int(values['--batch-size']) == job['identity']['job']['batch_size']
        assert values['--calls'] == '7'
        assert values['--validation-warmup'] == '4'
        assert '--atol' not in command and '--rtol' not in command
    config = json.loads(queue.DEFAULT_CONFIG.read_text())
    config['contract']['compute_contract'] = 'native'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='fixed four-job'):
        queue.load_config(path)


def test_one_deadline_is_passed_unchanged_to_every_guard(tmp_path, monkeypatch):
    prepared = jobs(tmp_path)
    commands = []
    monkeypatch.setattr(queue, 'release_matches', lambda directory, identity: False if not Path(directory).exists() else True)
    cutoff = '2026-11-07T23:00:00+08:00'
    result = queue.execute(prepared, tmp_path / 'output', 'GPU-test', cutoff, 'python',
                           now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
                           run=lambda cmd, **kw: (commands.append(cmd) or SimpleNamespace(returncode=0)))
    assert result == 0 and len(commands) == 4
    assert all(cmd[cmd.index('--until') + 1] == cutoff for cmd in commands)
    assert queue.execute(prepared, tmp_path / 'expired', 'GPU-test', cutoff, 'python',
                         now=lambda: datetime(2027, 1, 1, tzinfo=timezone.utc),
                         run=lambda *a, **kw: pytest.fail('expired queue ran')) == 2


def test_release_mismatch_is_rejected_and_compatible_release_skipped(tmp_path):
    job = jobs(tmp_path)[0]
    directory = tmp_path / 'output' / job['id']
    (directory / 'release').mkdir(parents=True)
    identity = job['identity']
    (directory / 'identity.json').write_text(json.dumps(identity))
    (directory / 'status.json').write_text(json.dumps({'state': 'complete', 'gpu_uuid': 'GPU-test'}))
    report = {k: identity[k] for k in ['library_sha256', 'runner_sha256', 'adapter_sha256', 'checks_sha256', 'gpu_uuid']}
    report.update(status='pass_timing_numerical_unassessed', compute_contract='public_fp32_final', execution_mode='graph',
                  quarot_extension_sha256=identity['extension_sha256'],
                  payload={'batch_size': 1, 'prefill': 128, 'decode': 8}, variant_order='baseline-first',
                  seed=20261003, validation_warmup_sequences=4, attention_backend='torch_sdpa',
                  timing_calls=7, timing_warmup_sequences=2)
    (directory / 'release/report.json').write_text(json.dumps(report))
    assert queue.release_matches(directory, identity)
    assert queue.execute([job], tmp_path / 'output', 'GPU-test', '2026-11-07T23:00:00+08:00', 'python',
                         run=lambda *a, **kw: pytest.fail('completed job reran')) == 0
    historical = dict(identity)
    historical.pop('selection_environment')
    (directory / 'identity.json').write_text(json.dumps(historical))
    assert queue.release_matches(directory, historical)
    historical_job = dict(job, identity=historical)
    with pytest.raises(ValueError, match='historical identity is read-only'):
        queue.execute([historical_job], tmp_path / 'output', 'GPU-test',
                      '2026-11-07T23:00:00+08:00', 'python')
    (directory / 'identity.json').write_text(json.dumps(identity))
    report['compute_contract'] = 'native'
    (directory / 'release/report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='identity'):
        queue.release_matches(directory, identity)


@pytest.mark.parametrize('variable', ['CUBUTTERFLY_MODEL_FINALISTS', 'CUBUTTERFLY_SEARCH_BUDGET',
                                      'CUBUTTERFLY_E13_ALGORITHM_POLICY'])
def test_changed_selection_environment_rejected_before_release_or_execution(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, 'before')
    prepared = jobs(tmp_path)
    monkeypatch.setenv(variable, 'after')
    monkeypatch.setattr(queue, 'release_matches', lambda *args: pytest.fail('checked release with stale environment'))
    with pytest.raises(ValueError, match='environment changed after prepare'):
        queue.execute(prepared, tmp_path / 'out', 'GPU-test', '2026-11-07T23:00:00+08:00', 'python')
    assert not (tmp_path / 'out').exists()


def test_static_config_content_change_invalidates_same_path(tmp_path, monkeypatch):
    tuning = tmp_path / 'tuning.json'
    tuning.write_text('{"rules": []}')
    monkeypatch.setenv('CUBUTTERFLY_PLATFORM_TUNING', str(tuning))
    prepared = jobs(tmp_path)
    bound = prepared[0]['identity']['selection_environment']
    assert bound['static_files']['CUBUTTERFLY_PLATFORM_TUNING']['sha256'] == queue.digest(tuning)
    tuning.write_text('{"rules": [1]}')
    with pytest.raises(ValueError, match='environment changed after prepare'):
        queue.execute(prepared, tmp_path / 'out', 'GPU-test', '2026-11-07T23:00:00+08:00', 'python')


def test_custom_mutable_plan_cache_rejected_without_reading_or_creating_it(tmp_path, monkeypatch):
    cache = tmp_path / 'plans.json'
    monkeypatch.setenv('CUBUTTERFLY_PLAN_CACHE', str(cache))
    with pytest.raises(ValueError, match='custom CUBUTTERFLY_PLAN_CACHE is unsupported'):
        jobs(tmp_path)
    assert not cache.exists()


def test_model_sibling_changes_and_new_cost_model_are_bound(tmp_path, monkeypatch):
    model = tmp_path / 'hardware_model.json'
    model.write_text('{}')
    monkeypatch.setenv('CUBUTTERFLY_HARDWARE_MODEL', str(model))
    before = queue.selection_environment()
    cost = tmp_path / 'cost_model.json'
    assert before['static_files']['hardware_model_profile_siblings']['cost_model.json']['sha256'] is None
    cost.write_text('{"coefficients": [1]}')
    after = queue.selection_environment()
    assert after != before
    cost.write_text('{"coefficients": [2]}')
    assert queue.selection_environment() != after


def test_default_profile_and_ranker_content_changes_are_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    profile = tmp_path / 'config/hardware/test-gpu'
    profile.mkdir(parents=True)
    (profile / 'hardware_model.json').write_text('{}')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    ranker = scripts / 'rank_static_candidates.py'
    ranker.write_text('# original')
    before = queue.selection_environment()
    (profile / 'cost_model.json').write_text('{}')
    after = queue.selection_environment()
    assert before != after
    ranker.write_text('# changed')
    assert queue.selection_environment() != after


@pytest.mark.parametrize('variable', ['CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_HARDWARE_PROFILE'])
def test_unsupported_static_override_rejected_instead_of_assumed_consumed(tmp_path, monkeypatch, variable):
    path = tmp_path / 'static.json'
    path.write_text('{}')
    monkeypatch.setenv(variable, str(path))
    with pytest.raises(ValueError, match='not a supported runtime override'):
        queue.selection_environment()
