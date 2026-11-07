import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import profile_current_mechanism as profile


def journal_fixture(tmp_path, *, measured=False):
    binary = tmp_path / 'cubutterfly_bench'
    binary.write_bytes(b'test benchmark; never executed')
    protocol = dict(trials=3, warmup=10, repeat=20, verify_batches=0)
    workload = dict(operator='fwht', precision='fp32', logN=8, batch=4,
                    placement='out-of-place', normalization='none')
    cases = []
    for index in range(6):
        variants = []
        for name, partition in [('stagewise', [1] * 8), ('grouped', [4, 4])]:
            mapping = dict(schema_version=1, kind='butterfly', backend='shared-iterative',
                           compute_unit='radix2', stage_partition=partition)
            commands = profile.controlled._commands(binary, workload, mapping, protocol)
            variant = dict(id=name, mapping=mapping, timed_command=commands['timed'],
                           preflight_command=commands['preflight'])
            if measured:
                sample = dict(workload)
                sample.update(profile.controlled._expected_semantics(workload))
                sample.update(mapping_json=json.dumps(mapping), runtime_fingerprint='fixture',
                              correct='1', verified_batches='4', kernel_ms='0.1')
                variant['preflight'] = dict(status='accepted', sample=sample, resolved_mapping=mapping,
                                             runtime_fingerprint='fixture')
                variant['trials'] = [dict(trial=i, status='measured', performance_accepted=True,
                                          sample=copy.deepcopy(sample), resolved_mapping=mapping) for i in range(3)]
            variants.append(variant)
        cases.append(dict(id=f'case-{index}', binary=str(binary), workload=workload, variants=variants,
                          status='complete' if measured else 'pending'))
    identity = dict(protocol=protocol, gpu_uuid='GPU-test', device={}, binaries={binary.name: dict(
        path=str(binary), sha256=profile._sha256(binary) if measured else None)})
    return dict(status='complete' if measured else 'prepared', completed=measured,
                identity=identity, protocol=protocol, cases=cases)


def test_default_prepare_is_cpu_only_all_six_twelve(tmp_path, monkeypatch):
    journal = journal_fixture(tmp_path)
    source = tmp_path / 'journal.json'
    source.write_text(json.dumps(journal))
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU preparation cannot execute subprocesses or GPU guards')
    monkeypatch.setattr(profile.subprocess, 'run', forbidden)
    monkeypatch.setattr(profile, 'require_exclusive_gpu', forbidden)
    output = tmp_path / 'profile'
    assert profile.main(['--journal', str(source), '--output-dir', str(output)]) == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['selected_case_count'] == 6
    assert manifest['full_population'] and not manifest['gpu_queried']
    assert len(manifest['variants']) == 12
    assert len(list(output.glob('*/*/manifest.json'))) == 12
    assert 'lts__t_sector_hit_rate.pct' in manifest['metrics']
    assert 'setup and teardown' in manifest['capture_scope']
    for row in manifest['variants']:
        assert '--launch-count' not in row['commands']['ncu']
        assert row['commands']['ncu'].count('--target-processes') == 1
        assert row['command'][row['command'].index('--warmup') + 1] == '0'
        assert row['command'][row['command'].index('--repeat') + 1] == '1'


def test_case_selection_rejects_unknown_and_preserves_variants(tmp_path):
    journal = journal_fixture(tmp_path)
    assert [case['id'] for case in profile._selected_cases(journal, ['case-3'])] == ['case-3']
    with pytest.raises(ValueError, match='unknown --case'):
        profile._selected_cases(journal, ['missing'])


@pytest.mark.parametrize('mutation', ['prepared', 'preflight', 'batch', 'trials', 'fingerprint', 'gpu'])
def test_run_rejects_bad_measured_journal_without_cuda(tmp_path, mutation):
    journal = journal_fixture(tmp_path, measured=True)
    profile._validate_measured(journal, journal['cases'], 'GPU-test')
    variant = journal['cases'][0]['variants'][0]
    if mutation == 'prepared':
        journal['status'] = 'prepared'
    elif mutation == 'preflight':
        variant['preflight']['status'] = 'rejected'
    elif mutation == 'batch':
        variant['preflight']['sample']['verified_batches'] = '1'
    elif mutation == 'trials':
        variant['trials'].pop()
    elif mutation == 'fingerprint':
        variant['trials'][0]['sample']['runtime_fingerprint'] = 'changed'
    else:
        journal['identity']['gpu_uuid'] = 'GPU-other'
    with pytest.raises((ValueError, RuntimeError)):
        profile._validate_measured(journal, journal['cases'], 'GPU-test')


def test_binary_tamper_rejected(tmp_path):
    journal = journal_fixture(tmp_path, measured=True)
    case = journal['cases'][0]
    profile._binary_identity(journal, case, True)
    Path(case['binary']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='binary hash mismatch'):
        profile._binary_identity(journal, case, True)


def test_guard_selects_uuid_and_restores_parent_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'original')
    seen = []
    monkeypatch.setattr(profile, 'require_exclusive_gpu', lambda: seen.append(profile.os.environ['CUDA_VISIBLE_DEVICES']))
    profile._guard('GPU-selected')
    assert seen == ['GPU-selected']
    assert profile.os.environ['CUDA_VISIBLE_DEVICES'] == 'original'


def test_failures_persist_per_tool_and_never_complete(tmp_path, monkeypatch):
    journal = journal_fixture(tmp_path, measured=True)
    source = tmp_path / 'journal.json'
    source.write_text(json.dumps(journal))
    monkeypatch.setattr(profile, '_tool_identity', lambda tool, run: {'version': 'mocked'})
    calls = []
    def execute(command, gpu_uuid, artifact):
        calls.append((command[0], gpu_uuid))
        return dict(status='failed', returncode=7, stdout='', stderr='fixture error')
    monkeypatch.setattr(profile, '_execute', execute)
    output = tmp_path / 'profile'
    assert profile.main(['--journal', str(source), '--output-dir', str(output), '--run',
                         '--gpu-uuid', 'GPU-test', '--case', 'case-1']) == 1
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert len(manifest['failures']) == 4 and len(calls) == 4
    assert all(row['status'] == 'failed' for row in manifest['variants'])
    assert all(json.loads(path.read_text())['status'] == 'failed' for path in output.glob('*/*/manifest.json'))


def test_success_without_artifact_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, '_guard', lambda uuid: None)
    monkeypatch.setattr(profile.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    result = profile._execute(['mock-profiler'], 'GPU-test', tmp_path / 'absent')
    assert result['status'] == 'failed'


def test_forbidden_sdb_asset(tmp_path):
    link = tmp_path / 'asset'
    link.symlink_to('/sdb/asset')
    with pytest.raises(ValueError, match='/sdb'):
        profile._sha256(link)
