import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'install_compare_driver', ROOT / 'results/install_compare_repair_20261014/run.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def prior(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    out = tmp_path / 'campaign'
    out.mkdir()
    identity = {}
    for path in (out / 'manifest.json', out / 'campaign_config.json',
                 tmp_path / 'config/current_recalibration_workloads.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        identity[str(path)] = driver.digest(path)
    (out / 'initial').mkdir()
    (out / 'initial/search_measurements.json').write_text('[{"origin_identity":"old-build"}]')
    (out / 'initial/stage_calibration.json').write_text('{"identity":"old-stage"}')
    state = dict(status='interrupted-by-foreign-gpu-work', identity=identity,
        gpu_uuid=driver.GPU, updated_unix=1791978587.2992363,
        phases={name: {'status': 'complete'} for name in ('stage-scope', 'calibrate', 'finalize', 'compare')},
        foreign_pids=[123], error='foreign GPU activity')
    driver.write(out / 'status.json', state)
    return out, state


def test_adoption_preserves_origin_and_reopens_current_build(tmp_path, monkeypatch):
    out, state = prior(tmp_path, monkeypatch)
    for name in ('finalized', 'measurement'):
        (out / name).mkdir()
        (out / name / 'results.json').write_text('{"identity":"old-build"}')
    for name in ('schedule_calibration.json', 'pipeline_schedule_calibration.json'):
        (out / 'initial' / name).write_text('{"probe_sha256":"old-probe"}')
    (out / 'initial/composition_calibration').mkdir()
    (out / 'initial/composition_calibration/identity.json').write_text('{"build":"old-build"}')
    current = dict(state['identity'], library='new-build')
    before = (out / 'initial/search_measurements.json').read_bytes()
    adopted = driver.adopt_current_build(state, current, out)
    archive = Path(adopted['current_build_adoption']['archive'])
    assert json.loads((archive / 'status.json').read_text()) == state
    assert (archive / 'initial/search_measurements.json').read_bytes() == before
    assert (out / 'initial/search_measurements.json').read_bytes() == before
    assert adopted['current_build_adoption']['previous_identity'] == state['identity']
    assert adopted['identity'] == current
    assert adopted['phases'] == {'stage-scope': dict(
        status='superseded-by-current-calibrator',
        reason='Frozen old-build stage scope is archived; current full stage inventory must be checked by calibrate.')}
    assert (archive / 'finalized/results.json').exists()
    assert (archive / 'measurement/results.json').exists()
    assert not (out / 'finalized').exists() and not (out / 'measurement').exists()
    assert 'foreign_pids' not in adopted and 'error' not in adopted
    for name in ('schedule_calibration.json', 'pipeline_schedule_calibration.json', 'composition_calibration'):
        assert not (out / 'initial' / name).exists()
        assert (archive / 'initial' / name).exists()
        assert (archive / 'retired-caches' / name).exists()
    assert state['status'] == 'interrupted-by-foreign-gpu-work'


@pytest.mark.parametrize('change', ['gpu', 'manifest', 'workloads'])
def test_adoption_cannot_change_gpu_or_semantic_scope(tmp_path, monkeypatch, change):
    out, state = prior(tmp_path, monkeypatch)
    current = dict(state['identity'])
    gpu = driver.GPU
    if change == 'gpu':
        gpu = 'different-GPU'
    else:
        path = out / 'manifest.json' if change == 'manifest' else tmp_path / 'config/current_recalibration_workloads.json'
        current[str(path)] = 'changed'
    with pytest.raises(ValueError, match='cannot change'):
        driver.adopt_current_build(state, current, out, gpu)
    assert not (out / 'attempts').exists()


def test_cpu_preparation_update_preserves_gpu_checkpoints_and_rejects_binary_changes(tmp_path, monkeypatch):
    out, state = prior(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'OUT', out)
    cpu_script = str(tmp_path / 'scripts/calibrate_local_hardware.py')
    state['identity'][cpu_script] = 'old-tools'
    current = dict(state['identity'], **{cpu_script:'new-tools', str(out/'cpu_inputs/active.json'):'input-sha'})
    resumed = driver.resume_interrupted(state, current, cpu_preparation=True)
    assert resumed['phases'] == state['phases']
    assert resumed['cpu_preparation_updates'][-1]['previous_identity'][cpu_script] == 'old-tools'
    assert (out/'initial/search_measurements.json').read_text() == '[{"origin_identity":"old-build"}]'
    with pytest.raises(ValueError, match='identity changed'):
        driver.resume_interrupted(state, dict(current, library='changed'), cpu_preparation=True)


@pytest.mark.parametrize('resume_case', ['complete', 'busy', 'foreign', 'failure', 'signal', 'reused-complete'])
def test_driver_adoption_reaches_calibrate_finalize_compare_without_gpu(tmp_path, monkeypatch, resume_case):
    real_out = ROOT / 'results/install_compare_repair_20261014'
    protected = [real_out / 'status.json', *sorted((real_out / 'initial').rglob('*'))]
    real_hashes = {path: driver.digest(path) for path in protected if path.is_file()}
    real_attempts = set((real_out / 'attempts').iterdir())
    out, state = prior(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'OUT', out)
    (out / 'run.py').write_text('# CPU test driver identity\n')
    monkeypatch.setattr(driver, '__file__', str(out / 'run.py'))
    monkeypatch.setattr(driver, 'BUILD', tmp_path / 'build')
    monkeypatch.setattr(driver, 'SOURCE', tmp_path / 'source')
    monkeypatch.setattr(driver, 'BROKER', tmp_path / 'broker')
    driver.write(tmp_path / 'broker/config.json', dict(config=dict(root='frozen-root', mathdx='frozen-mathdx',
        nvcc='frozen-nvcc', sm=80, cache='frozen-cache')))
    import semantic_compile_broker
    initialized = []
    def initialize(path, config):
        initialized.append((path, config))
        document = dict(id='current-broker', config=config)
        driver.write(path / 'config.json', document)
        return document
    monkeypatch.setattr(semantic_compile_broker, 'initialize', initialize)
    real_digest = driver.digest
    monkeypatch.setattr(driver, 'digest', lambda path: real_digest(path) if Path(path).exists() else 'current-input')
    monkeypatch.setattr(driver, 'gpu_pids', lambda gpu: [])
    monkeypatch.setattr(driver, 'stop_owned', lambda child: None)
    monkeypatch.setattr(driver, 'gate', lambda *args: None)
    import rebind_semantic_campaign
    def prepare(source, config, library, preparation, output, receipt):
        driver.write(output, dict(build='current', cases=66))
        driver.write(receipt, dict(source=str(source), cases=66))
        return dict(cases=66)
    monkeypatch.setattr(rebind_semantic_campaign, 'prepare', prepare)
    if resume_case == 'reused-complete':
        workload = tmp_path / 'config/current_recalibration_workloads.json'
        cell = dict(operator='fft', precision='fp32', logN=12, batch=4)
        driver.write(workload, dict(workloads=[cell]))
        state['identity'][str(workload)] = real_digest(workload)
        driver.write(out / 'status.json', state)
        driver.write(out / 'initial/search_coverage.json', dict(workloads=[dict(workload=cell, complete=True)]))
        for name in ('operator_calibration.json', 'calibration_device.json', 'calibration_manifest.json'):
            driver.write(out / 'initial' / name, {})
    commands = []
    stopped = []
    monkeypatch.setattr(driver, 'stop_owned', lambda child: stopped.append(child))
    mode = [None]
    class Child:
        pid = 999999
        returncode = 0
        def poll(self):
            if mode[0] == 'signal':
                raise KeyboardInterrupt('mock SIGTERM')
            return 0
    def popen(command, **kwargs):
        commands.append(command)
        child = Child()
        if any('run_async_semantic_campaign.py' in item for item in command):
            driver.write(out / 'measurement/results.json', dict(status='complete', rows=[{}]*66, remaining_case_ids=[]))
            if mode[0] == 'failure':
                child.returncode = 1
        return child
    monkeypatch.setattr(driver.subprocess, 'Popen', popen)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--adopt-current-build'] +
                        (['--reuse-completed-search'] if resume_case == 'reused-complete' else []))
    driver.main()
    assert not any('complete_stage_scope.py' in item for command in commands for item in command)
    calibrations = [command for command in commands if any('calibrate_local_hardware.py' in item for item in command)]
    if resume_case == 'reused-complete':
        assert not calibrations
    else:
        assert '--resume-search' in calibrations[0] and '--resume-build-priors' in calibrations[0]
    finalize = next(command for command in commands if any('finalize_recalibration.py' in item for item in command))
    assert '--revalidate-mappings' in finalize and '--resume' not in finalize
    comparison = next(command for command in commands if any('run_async_semantic_campaign.py' in item for item in command))
    assert '--resume' not in comparison
    assert comparison[comparison.index('--manifest')+1] == str(out / 'comparison_manifest.json')
    assert (out / 'manifest.json').read_text() == '{}'
    result = json.loads((out / 'status.json').read_text())
    assert result['status'] == 'complete' and result['measured_rows'] == 66
    assert result['current_build_adoption']['previous_status'] == 'interrupted-by-foreign-gpu-work'
    if resume_case == 'reused-complete':
        assert result['phases']['calibrate']['status'] == 'retained-as-mapping-seeds'
        assert result['completed_search_seeds']['workloads'] == 1
    assert result['active_broker'] == str(out / 'compile')
    assert comparison[comparison.index('--compile-broker')+1] == str(out / 'compile')
    assert initialized[0][1]['root'] == 'frozen-root'
    assert (tmp_path / 'broker/config.json').exists()
    assert result['identity'][str(out / 'compile/config.json')] == driver.digest(out / 'compile/config.json')
    assert Path(result['current_build_adoption']['archive']).is_relative_to(out)
    assert all(Path(path).is_relative_to(tmp_path) for path in result['identity'])
    assert initialized[0][0] == out / 'compile'
    assert result['gpu_uuid'] == driver.GPU
    assert not any(str(ROOT / 'results/install_compare_repair_20261014') in str(item)
                   for command in commands for item in command)
    # Reopen only the pending comparison; completed calibration/cache provenance stays untouched.
    result.update(status='interrupted-by-foreign-gpu-work', retryable=True, error='old interruption')
    result['phases']['compare']['status'] = 'quarantined-pending-review'
    driver.write(out / 'status.json', result)
    completed = {name: result['phases'][name] for name in ('calibrate', 'finalize')}
    (out / 'initial/schedule_calibration.json').write_text('preserved schedule')
    cache_hash = driver.digest(out / 'initial/schedule_calibration.json')
    broker_hash = driver.digest(out / 'compile/config.json')
    mode[0] = resume_case
    calls = []
    def gpu_pids(gpu):
        calls.append(gpu)
        return [os.getpid()] if resume_case == 'busy' or (resume_case == 'foreign' and len(calls) > 1) else []
    monkeypatch.setattr(driver, 'gpu_pids', gpu_pids)
    monkeypatch.setattr(driver, 'descendant', lambda *args: False)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--resume', '--resume-after-gpu-interruption'])
    count = len(commands)
    if resume_case in ('complete', 'reused-complete'):
        driver.main()
    else:
        with pytest.raises(KeyboardInterrupt if resume_case == 'signal' else RuntimeError):
            driver.main()
    resumed = json.loads((out / 'status.json').read_text())
    expected = dict(complete='complete', **{'reused-complete': 'complete'}, busy='waiting-for-gpu', foreign='interrupted-by-foreign-gpu-work',
                    failure='failed', signal='interrupted-calibration')[resume_case]
    assert resumed['status'] == expected
    assert resumed.get('retryable', False) == (resume_case in ('busy', 'foreign', 'signal'))
    assert all(resumed['phases'][name] == phase for name, phase in completed.items())
    assert len(resumed['interruption_resumes']) == 1
    assert driver.digest(out / 'initial/schedule_calibration.json') == cache_hash
    assert driver.digest(out / 'compile/config.json') == broker_hash
    assert len(initialized) == 1
    assert not any('calibrate_local_hardware.py' in item for command in commands[count:] for item in command)
    assert not any('finalize_recalibration.py' in item for command in commands[count:] for item in command)
    if resume_case == 'signal':
        assert len(stopped) >= 3  # Both comparison child and producer are cleaned up.
    assert {path: driver.digest(path) for path in real_hashes} == real_hashes
    assert set((real_out / 'attempts').iterdir()) == real_attempts


@pytest.mark.parametrize('change', ['incomplete', 'duplicate', 'different-workload', 'failed-gate'])
def test_reuse_completed_search_rejects_invalid_source_before_archiving(tmp_path, monkeypatch, change):
    out, state = prior(tmp_path, monkeypatch)
    workload = tmp_path / 'config/current_recalibration_workloads.json'
    cell = dict(operator='fft', precision='fp32', logN=12, batch=4)
    driver.write(workload, dict(workloads=[cell]))
    state['identity'][str(workload)] = driver.digest(workload)
    cells = [dict(workload=cell, complete=True)]
    if change == 'incomplete':
        cells[0]['complete'] = False
    elif change == 'duplicate':
        cells.append(cells[0])
    elif change == 'different-workload':
        cells[0]['workload'] = dict(cell, batch=8)
    driver.write(out / 'initial/search_coverage.json', dict(workloads=cells))
    if change == 'failed-gate':
        def gate(*args):
            raise ValueError('source calibration not ready')
        monkeypatch.setattr(driver, 'gate', gate)
    with pytest.raises(ValueError, match='scope|not ready'):
        driver.adopt_current_build(state, state['identity'], out, reuse_completed_search=True)
    assert not (out / 'attempts').exists()


def test_resume_allows_only_explicit_driver_identity_change(tmp_path, monkeypatch):
    out, state = prior(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, 'OUT', out)
    monkeypatch.setattr(driver, '__file__', str(out / 'run.py'))
    state['identity'][driver.__file__] = 'old-driver'
    current = dict(state['identity'], **{driver.__file__: 'new-driver'})
    resumed = driver.resume_interrupted(state, current)
    assert resumed['driver_identity_updates'][0]['previous_sha256'] == 'old-driver'
    assert resumed['driver_identity_updates'][0]['current_sha256'] == 'new-driver'
    assert resumed['phases'] == state['phases']
    with pytest.raises(ValueError, match='input identity changed'):
        driver.resume_interrupted(state, dict(current, library='changed-library'))
    with pytest.raises(ValueError, match='target GPU'):
        driver.resume_interrupted(dict(state, gpu_uuid='foreign-GPU'), current)


def test_driver_lock_rejects_duplicate_before_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'OUT', tmp_path)
    monkeypatch.setattr(driver, '_main', lambda: pytest.fail('duplicate initialized campaign'))
    status = tmp_path / 'status.json'
    status.write_text('{"status":"running","owner":"existing-driver"}')
    original = status.read_bytes()
    with (tmp_path / 'driver.lock').open('a') as lock:
        driver.fcntl.flock(lock, driver.fcntl.LOCK_EX | driver.fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            driver.main()
        assert error.value.code == 73
        assert status.read_bytes() == original


@pytest.mark.parametrize('sig', [driver.signal.SIGTERM, driver.signal.SIGINT])
def test_driver_signals_interrupt_and_restore_handlers(tmp_path, monkeypatch, sig):
    monkeypatch.setattr(driver, 'OUT', tmp_path)
    previous = driver.signal.getsignal(sig)
    monkeypatch.setattr(driver, '_main', lambda: driver.signal.raise_signal(sig))
    with pytest.raises(KeyboardInterrupt, match='received signal'):
        driver.main()
    assert driver.signal.getsignal(sig) == previous
    # Signal exit also releases the process lock.
    monkeypatch.setattr(driver, '_main', lambda: None)
    driver.main()
