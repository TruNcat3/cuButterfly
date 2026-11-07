import csv
import io
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import calibrate_local_hardware as calibration


@pytest.mark.parametrize('extra', [[], ['--cpu-preparation-input', 'inventory'],
                                  ['--cpu-preparation-inventory-only'],
                                  ['--cpu-preparation-dir', 'prepared']])
def test_seed_only_cli_budget_and_inventory_conflicts(monkeypatch, extra):
    argv = ['calibrate', '--build-dir', '.', '--profile-dir', '.', '--seed-only']
    argv += extra or ['--seed-budget', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit, match='2'):
        calibration.parse_args()


@pytest.mark.parametrize('report_only', [False, True])
def test_seed_only_keeps_measurement_validation_finalists_and_resume(tmp_path, monkeypatch, report_only):
    for name in ('cubutterfly_bench', 'cuntt_bench'):
        (tmp_path / name).write_bytes(b'cpu-test')
    workload = dict(operator='fft', precision='fp16', logN=8, batch=2,
                    direction='forward', normalization='none', placement='out-of-place')
    mapping = dict(schema_version=1, kind='butterfly', backend='shared-iterative',
                   fft_core='scalar', stage_partition=[4, 4], tile_threads=128)
    from hardware_registry import canonical_semantics
    semantics = canonical_semantics(workload)
    snapshot = dict(seeds=[dict(semantics=semantics, mapping=mapping, provenance={'source': 'test', 'kind': 'explicit'}),
                          dict(semantics=semantics, mapping=dict(mapping, tile_threads=256),
                               provenance={'source': 'second', 'kind': 'registry-winner'})])
    spec = tmp_path / 'workloads.json'
    spec.write_text(json.dumps(dict(schema='cubutterfly-install-search-v1', scope='test', workloads=[workload])))
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=spec, resume_search=False,
        seed_only=True, seed_budget=1, _prepared_seed_snapshot=snapshot,
        low_fft_error_report_only=report_only,
        search_budget=0, search_seconds=0, search_finalists=1, compile_seconds=0,
        operator_trials=2, operator_warmup=3, operator_repeat=5, verify_batches=0,
        cost_model='legacy', search_strategy='evolutionary', stage_calibration='skip')
    measured = []
    def forbidden(*args, **kwargs):
        pytest.fail('seed-only enumerated runtime inventory or exploration')
    monkeypatch.setattr(calibration, 'runtime_candidates', forbidden)
    import evolutionary_search
    monkeypatch.setattr(evolutionary_search, 'evolutionary_candidates', forbidden)
    monkeypatch.setattr(calibration, 'require_exclusive_gpu', lambda: None)
    def fake_run(command, check=True):
        if '--list-processing-units' in command:
            text = 'precision,logN,threads,ept\n'
        elif '--list-register-tile-mappings' in command:
            text = 'logN,local_stages\n'
        else:
            assert ('--low-fft-error-report-only' in command) == report_only
            requested = json.loads(command[command.index('--mapping-json') + 1])
            measured.append(requested)
            sample = dict(workload, backend=requested['backend'], fft_core=requested['fft_core'],
                          correct='1', kernel_ms='0.25', mapping_json=json.dumps(requested),
                          warmup='3', repeat='5')
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(sample))
            writer.writeheader(); writer.writerow(sample)
            text = stream.getvalue()
        return SimpleNamespace(returncode=0, stdout=text, stderr='')
    monkeypatch.setattr(calibration, 'run', fake_run)
    output = tmp_path / 'coverage.json'
    rows = calibration.run_compiled_search(args, output)
    assert len(rows) == 1 and rows[0]['status'] == 'measured'
    assert rows[0]['trials'] == 2
    assert measured and all(m['tile_threads'] == 128 for m in measured)
    audit = json.loads(output.read_text())
    assert audit['search_scope'] == 'seed-only' and not audit['inventory_complete']
    assert audit['search_protocol']['seed_only'] is True
    assert audit['workloads'][0]['screened'] == 1
    assert audit['workloads'][0]['seed_coverage']['budget_omitted'] == 1
    count = len(measured)
    args.resume_search = True
    calibration.run_compiled_search(args, output)
    assert len(measured) == count
    args.seed_only = False
    with pytest.raises(ValueError, match='cannot resume'):
        calibration.run_compiled_search(args, output)
    args.seed_only = True
    args.low_fft_error_report_only = not report_only
    with pytest.raises(ValueError, match='cannot resume'):
        calibration.run_compiled_search(args, output)


@pytest.mark.parametrize('budget', [0, 1])
def test_seed_only_no_usable_seeds_is_error(budget):
    with pytest.raises(ValueError, match='no usable mapping seeds'):
        calibration.workload_search_inputs(SimpleNamespace(seed_only=True, seed_budget=budget),
                                          {'seeds': []}, {'operator': 'fft', 'logN': 8}, None)


@pytest.mark.parametrize('operator,precision,applicable', [
    ('fft', 'fp16', True), ('fft', 'bf16', True), ('fft', 'fp16-fp32', True),
    ('fft', 'fp32', False), ('fwht', 'fp16', False), ('ntt', 'uint32', False)])
def test_numeric_policy_only_flags_low_fft(operator, precision, applicable):
    args = SimpleNamespace(low_fft_error_report_only=True)
    point = dict(operator=operator, precision=precision)
    command, policy = calibration.low_fft_numeric_policy(args, point, ['bench'])
    assert ('--low-fft-error-report-only' in command) == applicable
    assert ('not-applicable' in policy) != applicable
    if not applicable:
        with pytest.raises(ValueError, match='only to FP16/BF16 FFT'):
            calibration.low_fft_numeric_policy(args, point, ['bench', '--low-fft-error-report-only'])
