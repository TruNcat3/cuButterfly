"""CPU subprocesses exercise refresh orchestration without CUDA or compilation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_full_semantic_refresh as refresh
import run_semantic_campaign as semantic
import semantic_campaign_manifest as manifests
import watch_idle_campaign as monitor


def arguments(tmp_path):
    build = tmp_path / 'build'
    build.mkdir()
    for name in refresh.calibration.ARTIFACTS:
        (build / name).parent.mkdir(parents=True, exist_ok=True)
        (build / name).write_text('test artifact')
    preparation = tmp_path / 'preparation'
    preparation.mkdir()
    (preparation / 'config.json').write_text('{}')
    output = tmp_path / 'output'
    output.mkdir()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"cases": [{"id": "one"}, {"id": "two"}]}')
    workloads = tmp_path / 'workloads.json'
    workloads.write_text('{"workloads": []}')
    return SimpleNamespace(build_dir=build, output_dir=output,
        manifest=manifest, preparation=preparation, workloads=workloads,
        gpu_uuid='GPU-test', mathdx=tmp_path / 'mathdx', nvcc=tmp_path / 'nvcc',
        jit_cache=tmp_path / 'cache', stage_point_budget=24, search_budget=32,
        comparison_search_budget=512, compile_jobs=2, cpu_workers=3,
        vkfft_library=None, gpuntt_library=None, followup_plan=None,
        prepare_only=False, resume=False)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    bound = tmp_path / 'bound-input'
    bound.write_text('original')
    config = dict(root=str(ROOT), mathdx=str(args.mathdx), nvcc=str(args.nvcc),
                  cache=str(args.jit_cache), sm=80)
    document = dict(config=config, id='broker-test', sources_sha256='sources-test')
    monkeypatch.setattr(refresh.calibration, 'validate_build', lambda path: None)
    monkeypatch.setattr(refresh, 'validate_preparation', lambda *a: 2)
    monkeypatch.setattr(refresh.broker, 'initialize', lambda *a: document)
    monkeypatch.setattr(refresh.broker, '_sources', lambda *a: 'sources-test')
    monkeypatch.setattr(refresh, 'identity', lambda *a: dict(
        files={str(bound): refresh.calibration.digest(bound)}, broker_id=document['id']))
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: [])
    gates = []
    monkeypatch.setattr(refresh.calibration, 'gate', lambda name, *a: gates.append(name))
    real_popen = subprocess.Popen
    children = []
    producer_code = 'import time; time.sleep(60)'

    def spawn(command, **kwargs):
        producer = any(str(x).endswith('semantic_compile_producer.py') for x in command)
        actual = [sys.executable, '-c', context.producer_code] if producer else command
        child = real_popen(actual, **kwargs)
        children.append((producer, child, command[:], kwargs['env'].copy()))
        return child

    monkeypatch.setattr(refresh.subprocess, 'Popen', spawn)
    real_sleep = refresh.time.sleep
    monkeypatch.setattr(refresh.time, 'sleep', lambda duration: real_sleep(.005))

    def command(name, code=''):
        code = (f'from pathlib import Path; import json; '
                f'p=Path({str(args.output_dir / "executed")!r}); '
                f'p.open("a").write({name + chr(10)!r})' + ('; ' + code if code else ''))
        if name == 'compare' and not code.endswith('raise SystemExit(75)'):
            report = dict(status='complete', rows=[{'id': 'one'}, {'id': 'two'}],
                          remaining_case_ids=[])
            code += (f'; p=Path({str(args.output_dir / "measurement/results.json")!r}); '
                     f'p.parent.mkdir(exist_ok=True); p.write_text({json.dumps(report)!r})')
        return [sys.executable, '-c', code]

    plan = [(name, command(name)) for name in ('calibrate', 'finalize', 'xd', 'compare')]
    monkeypatch.setattr(refresh, 'build_commands', lambda a: copy.deepcopy(context.plan))
    monkeypatch.setattr(refresh, 'followup_jobs', lambda a: copy.deepcopy(context.jobs))
    context = SimpleNamespace(args=args, bound=bound, document=document, gates=gates,
        children=children, producer_code=producer_code, command=command, plan=plan, jobs=[],
        state=lambda: json.loads((args.output_dir / 'status.json').read_text()),
        executed=lambda: (args.output_dir / 'executed').read_text().splitlines())
    yield context
    for _, child, _, _ in children:
        refresh.stop_owned(child)


def test_real_cpu_children_complete_all_gates_and_cleanup_producer(lifecycle):
    c = lifecycle
    assert refresh.orchestrate(c.args) == 0
    assert c.executed() == ['calibrate', 'finalize', 'xd', 'compare']
    assert c.gates == ['calibrate', 'finalize', 'xd']
    state = c.state()
    assert monitor.complete(state)
    assert monitor.exit_action(0, state, 0) == 'finished'
    assert state['remaining'] == 0 and state['measured_rows'] == 2
    assert all(p.poll() is not None for _, p, _, _ in c.children)
    producers = [(cmd, env) for producer, _, cmd, env in c.children if producer]
    assert len(producers) == 1
    command, env = producers[0]
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert command[command.index('--jobs') + 1] == '2'
    assert command[command.index('--priority-report') + 1].endswith('measurement/results.json')
    compare_env = next(env for producer, _, _, env in c.children
                       if not producer and env.get('CUBUTTERFLY_SEARCH_BUDGET') == '512')
    assert compare_env['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    assert compare_env['CUBUTTERFLY_REGISTRY'].endswith('finalized/registry.json')


def test_phase_failure_never_advances_and_cannot_automatically_resume(lifecycle):
    c = lifecycle
    c.plan[0] = ('calibrate', c.command('calibrate', 'raise SystemExit(7)'))
    with pytest.raises(RuntimeError, match='calibrate exited 7'):
        refresh.orchestrate(c.args)
    assert c.executed() == ['calibrate'] and not c.gates
    assert c.state()['status'] == 'failed' and c.state()['retryable'] is False
    c.args.resume = True
    with pytest.raises(ValueError, match='failed phase requires inspection'):
        refresh.orchestrate(c.args)
    assert c.executed() == ['calibrate']


def test_gate_failure_after_zero_exit_blocks_following_phase(lifecycle, monkeypatch):
    c = lifecycle
    def gate(*args):
        raise ValueError('stage calibration incomplete')
    monkeypatch.setattr(refresh.calibration, 'gate', gate)
    with pytest.raises(ValueError, match='stage calibration incomplete'):
        refresh.orchestrate(c.args)
    assert c.executed() == ['calibrate']
    assert c.state()['status'] == 'failed'


def test_foreign_gpu_interruption_stops_only_owned_child_then_resumes(lifecycle, monkeypatch):
    c = lifecycle
    c.plan[0] = ('calibrate', c.command('calibrate', 'import time; time.sleep(60)'))
    observations = iter([[], [os.getpid()]])
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: next(observations, []))
    with pytest.raises(RuntimeError, match='foreign GPU work'):
        refresh.orchestrate(c.args)
    state = c.state()
    assert state['phases']['calibrate']['status'] == 'interrupted'
    assert state['foreign_pids'] == [os.getpid()]
    assert c.children[0][1].poll() is not None
    assert monitor.exit_action(1, state, 0) == 'retry'
    c.plan[0] = ('calibrate', c.command('calibrate'))
    c.args.resume = True
    assert refresh.orchestrate(c.args) == 0
    assert monitor.complete(c.state())


def test_busy_gpu_wait_state_is_retryable_without_launching(lifecycle, monkeypatch):
    c = lifecycle
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: [os.getpid()])
    with pytest.raises(RuntimeError, match='occupied'):
        refresh.orchestrate(c.args)
    assert not c.children
    assert c.state()['remaining'] == 2
    assert monitor.exit_action(1, c.state(), 0) == 'retry'


def test_resume_skips_completed_children_but_rechecks_gates(lifecycle):
    c = lifecycle
    assert refresh.orchestrate(c.args) == 0
    children = len(c.children)
    c.args.resume = True
    assert refresh.orchestrate(c.args) == 0
    assert c.executed() == ['calibrate', 'finalize', 'xd', 'compare']
    assert c.gates == ['calibrate', 'finalize', 'xd'] * 2
    assert len([p for producer, p, _, _ in c.children[children:] if not producer]) == 0


def test_worker_recycle_retries_comparison_only_with_resume(lifecycle):
    c = lifecycle
    marker = c.args.output_dir / 'recycled'
    code = (f'p=Path({str(marker)!r}); '
            'first=not p.exists(); p.touch(); '
            'exec("raise SystemExit(75)" if first else "pass")')
    c.plan[-1] = ('compare', c.command('compare', code))
    assert refresh.orchestrate(c.args) == 0
    assert c.executed() == ['calibrate', 'finalize', 'xd', 'compare', 'compare']
    compare_commands = [cmd for producer, _, cmd, env in c.children
                        if not producer and env.get('CUBUTTERFLY_SEARCH_BUDGET') == '512']
    assert '--resume' not in compare_commands[0]
    assert '--resume' in compare_commands[1]


def test_resume_rejects_changed_bound_input_before_launch(lifecycle):
    c = lifecycle
    c.args.prepare_only = True
    assert refresh.orchestrate(c.args) == 0
    c.bound.write_text('changed')
    c.args.resume = True
    with pytest.raises(ValueError, match='resume requires unchanged'):
        refresh.orchestrate(c.args)
    assert not c.children


def test_prepare_only_records_commands_without_gpu_or_subprocess(lifecycle, monkeypatch):
    c = lifecycle
    c.args.prepare_only = True
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: pytest.fail('GPU query during prepare-only'))
    assert refresh.orchestrate(c.args) == 0
    assert not c.children and not c.gates
    assert c.state()['status'] == 'prepared'
    assert set(dict(json.loads((c.args.output_dir / 'commands.json').read_text())['phases'])) == {
        'calibrate', 'finalize', 'xd', 'compare'}


def test_keyboard_interruption_preserves_completed_phases_on_resume(lifecycle, monkeypatch):
    c = lifecycle
    c.plan[2] = ('xd', c.command('xd', 'import time; time.sleep(60)'))
    def observe(uuid):
        if c.children and c.children[-1][2] == c.plan[2][1]:
            raise KeyboardInterrupt('test interruption')
        return []
    monkeypatch.setattr(refresh, 'gpu_pids', observe)
    with pytest.raises(KeyboardInterrupt, match='test interruption'):
        refresh.orchestrate(c.args)
    state = c.state()
    assert state['status'] == 'interrupted-calibration'
    assert state['phases']['calibrate']['status'] == 'complete'
    assert state['phases']['finalize']['status'] == 'complete'
    assert 'compare' not in state['phases']
    assert all(child.poll() is not None for _, child, _, _ in c.children)
    assert monitor.exit_action(1, state, 0) == 'retry'
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: [])
    c.plan[2] = ('xd', c.command('xd'))
    c.args.resume = True
    assert refresh.orchestrate(c.args) == 0
    assert c.executed().count('calibrate') == 1
    assert c.executed().count('finalize') == 1
    assert c.executed()[-2:] == ['xd', 'compare']


def test_broker_identity_collision_rejected_before_any_phase(lifecycle, monkeypatch):
    c = lifecycle
    def initialize(*args):
        raise ValueError('broker identity changed; use a new broker directory')
    monkeypatch.setattr(refresh.broker, 'initialize', initialize)
    with pytest.raises(ValueError, match='broker identity changed'):
        refresh.orchestrate(c.args)
    assert not c.children and not c.gates


def test_existing_broker_rejects_changed_compiler_sources(tmp_path, monkeypatch):
    compiler = tmp_path / 'nvcc'
    compiler.write_text('fake CPU compiler identity')
    config = dict(root=str(tmp_path), mathdx=str(tmp_path / 'mathdx'),
                  nvcc=str(compiler), cache=str(tmp_path / 'cache'), sm=80)
    monkeypatch.setattr(refresh.broker.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout='fake nvcc version'))
    monkeypatch.setattr(refresh.broker, '_sources', lambda c: 'original-source')
    first = refresh.broker.initialize(tmp_path / 'broker', config)
    assert refresh.broker.initialize(tmp_path / 'broker', config) == first
    monkeypatch.setattr(refresh.broker, '_sources', lambda c: 'changed-source')
    with pytest.raises(ValueError, match='broker identity changed'):
        refresh.broker.initialize(tmp_path / 'broker', config)


def test_compare_cannot_complete_after_bound_input_changes(lifecycle):
    c = lifecycle
    c.plan[-1] = ('compare', c.command('compare',
        f'Path({str(c.bound)!r}).write_text("changed during comparison")'))
    with pytest.raises(ValueError, match='campaign input changed'):
        refresh.orchestrate(c.args)
    assert c.state()['status'] == 'failed'


def test_failed_cpu_producer_cannot_publish_success(lifecycle):
    c = lifecycle
    c.producer_code = 'raise SystemExit(9)'
    c.plan[-1] = ('compare', c.command('compare', 'import time; time.sleep(.1)'))
    with pytest.raises(RuntimeError, match='producer'):
        refresh.orchestrate(c.args)
    assert c.state()['status'] == 'failed'


def test_build_commands_bind_paths_external_libraries_and_resume_receipts(tmp_path):
    args = arguments(tmp_path)
    args.vkfft_library = tmp_path / 'vkfft.so'
    args.gpuntt_library = tmp_path / 'gpuntt.so'
    for relative in ('initial/search_coverage.json', 'finalized/confirmation_identity.json'):
        path = args.output_dir / 'calibration' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (args.output_dir / 'measurement').mkdir()
    (args.output_dir / 'measurement/results.json').write_text('{}')
    commands = dict(refresh.build_commands(args))
    assert list(commands) == ['calibrate', 'finalize', 'xd', 'compare']
    assert '--resume-search' in commands['calibrate']
    assert '--resume' in commands['finalize'] and '--resume' in commands['compare']
    for flag, value in (('--manifest', args.manifest), ('--preparation', args.preparation),
                        ('--vkfft-library', args.vkfft_library),
                        ('--gpuntt-library', args.gpuntt_library)):
        assert commands['compare'][commands['compare'].index(flag) + 1] == str(value)
    assert '--timing' in commands['compare']
    assert commands['compare'][commands['compare'].index('--cpu-workers') + 1] == '3'
    assert commands['compare'][commands['compare'].index('--calibration-manifest') + 1].endswith(
        'calibration/finalized/calibration_manifest.json')


def test_identity_binds_build_broker_and_measurement_budgets(tmp_path):
    args = arguments(tmp_path)
    document = dict(id='broker-one', sources_sha256='source-one')
    initial = refresh.identity(args, document, {'CUDA_VISIBLE_DEVICES': 'GPU-test'})
    (args.build_dir / 'libcuntt.so').write_text('different binary')
    assert refresh.identity(args, document, {})['files'] != initial['files']
    for field in ('stage_point_budget', 'search_budget', 'comparison_search_budget'):
        changed = copy.copy(args)
        setattr(changed, field, getattr(changed, field) + 1)
        assert refresh.identity(changed, document, {}) != refresh.identity(args, document, {})
    assert refresh.identity(args, dict(document, id='broker-two'), {}) != refresh.identity(args, document, {})
    assert refresh.identity(args, dict(document, sources_sha256='changed'), {}) != refresh.identity(args, document, {})


@pytest.mark.parametrize('difference', ['manifest', 'sources', 'pending', 'checksum', 'fingerprint',
    'reference_max_elements', 'ntt_max_operations', 'none'])
def test_preparation_mismatch_fails_before_gpu_use(tmp_path, monkeypatch, difference):
    args = arguments(tmp_path)
    manifest = json.loads(args.manifest.read_text())
    config = dict(manifest=copy.deepcopy(manifest), source_sha256={'source': 'same'},
        reference_max_elements=semantic.PROTOCOL['reference_max_elements'],
        ntt_max_operations=semantic.PROTOCOL['ntt_max_operations'])
    monkeypatch.setattr(manifests, 'validate_manifest_hash', lambda m: difference != 'checksum')
    def fingerprints(*a):
        if difference == 'fingerprint':
            raise ValueError('library fingerprint changed')
    monkeypatch.setattr(semantic, 'validate_fingerprints', fingerprints)
    monkeypatch.setattr(refresh.preparation, '_config', lambda p: (config, 'config-hash'))
    monkeypatch.setattr(refresh.preparation, '_sources', lambda: {'source': 'same'})
    monkeypatch.setattr(refresh.preparation, 'record_for', lambda *a:
                        {'status': 'failed' if difference == 'pending' else 'ready'})
    if difference == 'manifest':
        config['manifest'] = {'cases': []}
    if difference == 'sources':
        config['source_sha256'] = {'source': 'old'}
    if difference in ('reference_max_elements', 'ntt_max_operations'):
        config[difference] += 1
    if difference == 'none':
        assert refresh.validate_preparation(manifest, args.build_dir / 'libcuntt.so', args.preparation) == 2
        return
    with pytest.raises(ValueError):
        refresh.validate_preparation(manifest, args.build_dir / 'libcuntt.so', args.preparation)


@pytest.mark.parametrize('report', [dict(status='complete', rows=[{}, {}]),
    dict(status='complete', rows=[{}], remaining_case_ids=[]),
    dict(status='running', rows=[{}, {}], remaining_case_ids=[]),
    dict(status='complete', rows=[{}, {}], remaining_case_ids=['two'])])
def test_incomplete_comparison_cannot_claim_track_complete(report):
    with pytest.raises(ValueError):
        refresh.result_summary(report, 2)


def test_environment_clears_stale_measurement_overrides(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    for key in ('CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_HARDWARE_PROFILE',
                'CUBUTTERFLY_PREPARE_BROKER', 'CUBUTTERFLY_PREPARE_EVENTS', 'LD_PRELOAD'):
        monkeypatch.setenv(key, 'stale')
    config = dict(root=str(ROOT), mathdx=str(args.mathdx), nvcc=str(args.nvcc), cache=str(args.jit_cache))
    env = refresh.build_environment(args, config)
    assert not any(key in env for key in ('LD_PRELOAD', 'CUBUTTERFLY_PREPARE_EVENTS',
        'CUBUTTERFLY_PREPARE_BROKER', 'CUBUTTERFLY_COST_MODEL', 'CUBUTTERFLY_HARDWARE_PROFILE'))
    assert env['CUBUTTERFLY_COMPILE_MODE'] == 'research'
    assert env['CUBUTTERFLY_CALIBRATION_BROKER'] == str(args.output_dir / 'compile')


def test_followup_failures_remain_gaps_and_neither_success_nor_failure_reruns(lifecycle):
    c = lifecycle
    c.jobs = [('good', c.command('good')), ('bad', c.command('bad', 'raise SystemExit(7)')),
              ('last', c.command('last'))]
    assert refresh.orchestrate(c.args) == 0
    state = c.state()
    assert c.executed()[-3:] == ['good', 'bad', 'last']
    assert state['status'] == 'complete-runnable-refresh-with-gaps'
    assert state['full_refresh_complete'] is False
    assert state['operator_track_complete'] is True
    assert set(state['followup_failures']) == {'bad'}
    assert state['phases']['bad']['status'] == 'failed'
    assert monitor.complete(state)
    before = c.executed()
    c.args.resume = True
    assert refresh.orchestrate(c.args) == 0
    assert c.executed() == before
    assert c.state()['followup_failures'] == state['followup_failures']


def test_cpu_producer_is_stopped_before_followup_launch(lifecycle, monkeypatch):
    c = lifecycle
    c.jobs = [('after', c.command('after'))]
    spawn = refresh.subprocess.Popen
    def checked_spawn(command, **kwargs):
        if command == c.jobs[0][1]:
            producers = [child for producer, child, _, _ in c.children if producer]
            assert producers and all(child.poll() is not None for child in producers)
        return spawn(command, **kwargs)
    monkeypatch.setattr(refresh.subprocess, 'Popen', checked_spawn)
    assert refresh.orchestrate(c.args) == 0
    assert c.state()['status'] == 'complete-runnable-refresh'
    assert c.state()['full_refresh_complete'] is False


def test_followup_interruption_resumes_only_unfinished_job(lifecycle, monkeypatch):
    c = lifecycle
    c.jobs = [('one-job', c.command('one-job')),
              ('two-job', c.command('two-job', 'import time; time.sleep(60)')),
              ('three-job', c.command('three-job'))]
    def observe(uuid):
        if c.children and c.children[-1][2] == c.jobs[1][1]:
            raise KeyboardInterrupt('followup interrupted')
        return []
    monkeypatch.setattr(refresh, 'gpu_pids', observe)
    with pytest.raises(KeyboardInterrupt, match='followup interrupted'):
        refresh.orchestrate(c.args)
    state = c.state()
    assert state['operator_track_complete'] is True
    assert state['remaining'] == 2
    assert state['phases']['one-job']['status'] == 'complete'
    assert 'three-job' not in state['phases']
    assert monitor.exit_action(1, state, 0) == 'retry'
    c.jobs[1] = ('two-job', c.command('two-job'))
    c.args.resume = True
    monkeypatch.setattr(refresh, 'gpu_pids', lambda uuid: [])
    assert refresh.orchestrate(c.args) == 0
    assert c.executed().count('one-job') == 1
    assert c.executed().count('calibrate') == 1
    assert c.executed()[-2:] == ['two-job', 'three-job']
    assert c.state()['full_refresh_complete'] is False


@pytest.mark.parametrize('exit_code', [0, 7])
def test_followup_identity_violation_is_fatal_even_when_command_fails(lifecycle, exit_code):
    c = lifecycle
    c.jobs = [('changed', c.command('changed',
        f'Path({str(c.bound)!r}).write_text("changed"); raise SystemExit({exit_code})'))]
    with pytest.raises(ValueError, match='campaign input changed'):
        refresh.orchestrate(c.args)
    assert c.state()['status'] == 'failed'
    assert 'changed' not in c.state().get('followup_failures', {})


def test_followup_ownership_error_is_fatal_and_later_job_never_starts(lifecycle, monkeypatch):
    c = lifecycle
    c.jobs = [('ownership', c.command('ownership')), ('later', c.command('later'))]
    original = refresh.stop_owned
    def stop(child):
        original(child)
        if any(cmd == c.jobs[0][1] and owned is child for _, owned, cmd, _ in c.children):
            raise RuntimeError('refusing to signal an unowned process group')
    monkeypatch.setattr(refresh, 'stop_owned', stop)
    with pytest.raises(RuntimeError, match='unowned process group'):
        refresh.orchestrate(c.args)
    monkeypatch.setattr(refresh, 'stop_owned', original)
    assert 'later' not in c.executed()
    assert c.state()['status'] == 'failed'
    assert not c.state()['followup_failures']


def write_followup_plan(args, jobs):
    args.followup_plan = args.output_dir / 'followups.json'
    args.followup_plan.write_text(json.dumps(dict(
        schema='cubutterfly-refresh-followups-v1', jobs=jobs)))


def test_followup_arguments_and_identity_bind_plan_and_entrypoint(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    script = tmp_path / 'scripts/job.py'
    script.parent.mkdir()
    script.write_text('print("one")')
    monkeypatch.setattr(refresh, 'ROOT', tmp_path)
    digest = refresh.calibration.digest
    monkeypatch.setattr(refresh.calibration, 'digest', lambda p:
                        digest(p) if Path(p).exists() else 'unused-fixture-source')
    write_followup_plan(args, [dict(id='job', argv=['scripts/job.py', '--gpu', '{gpu_uuid}',
        '--output', '{output_dir}/job', '--build', '{build_dir}'])])
    [(name, command)] = refresh.followup_jobs(args)
    assert name == 'job'
    assert command == [sys.executable, str(script), '--gpu', 'GPU-test',
                       '--output', str(args.output_dir / 'job'), '--build', str(args.build_dir)]
    document = dict(id='broker', sources_sha256='sources')
    initial = refresh.identity(args, document, {})
    assert initial['files'][str(args.followup_plan)] == digest(args.followup_plan)
    assert initial['files'][str(script)] == digest(script)
    script.write_text('print("two")')
    assert refresh.identity(args, document, {}) != initial
    after_script = refresh.identity(args, document, {})
    write_followup_plan(args, [dict(id='job', argv=['scripts/job.py', '--different'])])
    assert refresh.identity(args, document, {}) != after_script


@pytest.mark.parametrize('name', ['calibrate', 'finalize', 'xd', 'compare'])
def test_followup_cannot_shadow_builtin_phase(tmp_path, name):
    args = arguments(tmp_path)
    write_followup_plan(args, [dict(id=name, argv=['scripts/run_full_semantic_refresh.py'])])
    with pytest.raises(ValueError, match='reserved|invalid'):
        refresh.followup_jobs(args)


@pytest.mark.parametrize('jobs', [
    [dict(id='duplicate', argv=['scripts/run_full_semantic_refresh.py'])] * 2,
    [dict(id='../outside', argv=['scripts/run_full_semantic_refresh.py'])],
    [dict(id='job', argv=['/etc/passwd'])],
    [dict(id='job', argv='scripts/run_full_semantic_refresh.py')]])
def test_followup_rejects_ambiguous_or_invalid_job_contract(tmp_path, jobs):
    args = arguments(tmp_path)
    write_followup_plan(args, jobs)
    with pytest.raises((ValueError, FileNotFoundError, IsADirectoryError)):
        refresh.followup_jobs(args)
