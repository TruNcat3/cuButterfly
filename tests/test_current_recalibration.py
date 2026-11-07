import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("current_recalibration", ROOT / "scripts/run_current_recalibration.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments(tmp_path):
    build = tmp_path / "build"
    for name in runner.ARTIFACTS:
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact")
    (build / "CMakeCache.txt").write_text(
        "CUBUTTERFLY_BUILD_SHARED:BOOL=ON\nCUBUTTERFLY_ENABLE_CUFFTDX:BOOL=ON\n"
        "CUBUTTERFLY_ENABLE_XD_CUFFTDX:BOOL=ON\nCMAKE_CUDA_ARCHITECTURES:STRING=80\n")
    return SimpleNamespace(build_dir=build, output_dir=tmp_path / "output", gpu_uuid="GPU-test",
                           resume=False, dry_run=False, vkfft_binary=None, fht_python=None,
                           gpuntt_binary=None, include_xd=False, workloads=runner.WORKLOADS,
                           stage_point_budget=8, search_budget=16)


def stage_profile(sampling=None):
    service = dict(curves={"key": {}}, calibration_status="complete", validated=True,
                   validation=dict(passed=True), coverage=dict(coverage_complete=True,
                   validation_complete=True, requested_static_count=24, described_static_count=24))
    if sampling is not None:
        service.update(calibration_status="complete-sampled", validated=False,
                       selected_scope_status="complete", selected_scope_validated=True,
                       inventory_complete=False, stage_point_sampling=sampling)
    return dict(stage_service=service)


def test_custom_workloads_and_budgets_propagate(tmp_path):
    args=arguments(tmp_path)
    args.workloads=tmp_path / "custom-workloads.json"
    runner.write(args.workloads,dict(scope="two finite cells",workloads=[
        dict(precision="fp32",logN=12,batch=4),dict(operator="fwht",precision="fp32",logN=12,batch=4)]))
    args.stage_point_budget=24
    args.search_budget=32
    identity=runner.identity(args)
    assert identity["workloads_sha256"]==runner.digest(args.workloads)
    assert identity["protocol"]["stage_point_budget"]==24
    assert identity["protocol"]["search_budget"]==32
    plan=runner.commands(args)
    for command in (plan[0][1],plan[2][1]):
        workload_flag="--search-workloads" if command is plan[0][1] else "--workloads"
        assert command[command.index(workload_flag)+1]==str(args.workloads)
        assert command[command.index("--stage-point-budget")+1]=="24"
        assert command[command.index("--search-budget")+1]=="32"
    sampling = dict(selected=24, inventory_complete=False, scope="stratified-sample")
    runner.write(args.output_dir / "initial/calibration_manifest.json",dict(
        operator_result=dict(status="complete"), stage_point_sampling_summary=sampling))
    runner.write(args.output_dir / "initial/calibration_device.json", stage_profile(sampling))
    runner.write(args.output_dir / "initial/search_coverage.json",dict(workloads=[dict(complete=True)]*2))
    runner.gate("calibrate",args.output_dir,args.workloads)
    runner.write(args.output_dir / "comparison.json",dict(status="complete",cells=[{},{}]))
    runner.gate("compare",args.output_dir,args.workloads)
    runner.write(args.output_dir / "comparison.json",dict(status="complete",cells=[{}]))
    with pytest.raises(ValueError,match="comparison cells"):
        runner.gate("compare",args.output_dir,args.workloads)
    args.dry_run=True
    args.output_dir=tmp_path / "dry-run"
    runner.orchestrate(args,mock.Mock(side_effect=AssertionError("must not run")))
    assert runner.read(args.output_dir / "manifest.json")["scope"]=="two finite cells"


@pytest.mark.parametrize("field",["stage_point_budget","search_budget"])
def test_resume_rejects_changed_budget(tmp_path,field):
    args=arguments(tmp_path)
    args.dry_run=True
    runner.orchestrate(args)
    args.resume=True
    setattr(args,field,getattr(args,field)+1)
    with pytest.raises(ValueError,match="same build"):
        runner.orchestrate(args)


def test_workloads_and_finite_commands(tmp_path):
    workloads = runner.read(runner.WORKLOADS)["workloads"]
    assert len(workloads) == 23
    assert workloads[:19] == runner.read(ROOT / "config/install_search_workloads.json")["workloads"]
    assert {(w["logN"], w["batch"], w["accumulation"]) for w in workloads[19:]} == {
        (n, b, "fp32") for n in (5, 6) for b in (128, 4096)}
    command = runner.commands(arguments(tmp_path))[0][1]
    for flag, value in (("--search-budget", "16"), ("--search-finalists", "2"),
                        ("--stage-point-budget", "8"), ("--compile-seconds", "1200"),
                        ("--verify-batches", "2")):
        assert command[command.index(flag) + 1] == value
    assert "--search-only" in command


def test_dry_run_has_no_subprocess(tmp_path):
    args = arguments(tmp_path)
    args.dry_run = True
    execute = mock.Mock(side_effect=AssertionError("must not run"))
    assert runner.orchestrate(args, execute) == 0
    assert runner.read(args.output_dir / "status.json")["status"] == "dry-run"
    execute.assert_not_called()
    assert runner.environment(args)["CUBUTTERFLY_REGISTRY"] == str(args.output_dir / "initial/registry.json")


def test_resume_rejects_changed_binary(tmp_path):
    args = arguments(tmp_path)
    args.dry_run = True
    runner.orchestrate(args)
    (args.build_dir / "libcuntt.so").write_text("changed")
    args.resume = True
    with pytest.raises(ValueError, match="same build"):
        runner.orchestrate(args)


def test_phase_failure_stops_comparison(tmp_path):
    args = arguments(tmp_path)
    execute = mock.Mock(return_value=SimpleNamespace(returncode=7))
    with pytest.raises(RuntimeError, match="calibrate exited 7"):
        runner.orchestrate(args, execute)
    assert execute.call_count == 1
    state = runner.read(args.output_dir / "status.json")
    assert state["status"] == "failed"
    assert state["phases"]["calibrate"]["status"] == "failed"
    assert "compare" not in state["phases"]


def test_zero_exit_with_incomplete_search_stops(tmp_path):
    args = arguments(tmp_path)
    runner.write(args.output_dir / "initial/calibration_manifest.json", {"operator_result": {"status": "complete"}})
    runner.write(args.output_dir / "initial/search_coverage.json", {"workloads": [{"complete": False}] * 23})
    # Existing payload belongs to the matching checkpoint, not a new output.
    runner.write(args.output_dir / "status.json", {"identity": runner.identity(args), "status": "running", "phases": {}})
    args.resume = True
    execute = mock.Mock(return_value=SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="23 search cells"):
        runner.orchestrate(args, execute)
    assert execute.call_count == 1


def test_finalized_gate_requires_public_replay(tmp_path):
    args = arguments(tmp_path)
    runner.write(args.output_dir / "finalized/calibration_manifest.json", {
        "status": "ready-with-measured-mappings", "selector_replay": {"status": "failed"}})
    with pytest.raises(ValueError, match="selector replay"):
        runner.gate("finalize", args.output_dir)


def test_stage_gate_accepts_only_the_complete_declared_finite_sample():
    sampling = dict(selected=24, inventory_complete=False, scope="stratified-sample")
    profile = stage_profile(sampling)
    runner.require_stage_ready(profile, sampling)
    import copy
    for changes in (dict(selected_scope_validated=False), dict(selected_scope_status="complete-bounded"),
                    dict(calibration_status="incomplete-one-point-only"),
                    dict(ranking_scope=dict(validated=True), selected_scope_validated=False)):
        broken = copy.deepcopy(profile)
        broken['stage_service'].update(changes)
        with pytest.raises(ValueError):
            runner.require_stage_ready(broken,sampling)
    for changes in (dict(requested_static_count=23),dict(described_static_count=23),
                    dict(pending_description_count=1),dict(adapter_gap_count=1),
                    dict(unsupported_group_count=1),dict(omitted_count=1),
                    dict(one_point_curves=['one']),dict(validation_complete=False)):
        broken = copy.deepcopy(profile)
        broken['stage_service']['coverage'].update(changes)
        with pytest.raises(ValueError):
            runner.require_stage_ready(broken,sampling)
    with pytest.raises(ValueError,match='declared sampling'):
        runner.require_stage_ready(profile,dict(sampling,selected=12))


def test_global_scope_requires_global_inventory_and_validation():
    sampling = dict(selected=24,inventory_complete=True,scope='all')
    profile = stage_profile()
    profile['stage_service'].update(stage_point_sampling=sampling,inventory_complete=True)
    runner.require_stage_ready(profile,sampling)
    profile['stage_service']['inventory_complete'] = False
    with pytest.raises(ValueError,match='global inventory'):
        runner.require_stage_ready(profile,sampling)


@pytest.mark.parametrize('status',[None,'incomplete','incomplete-one-point-only','complete',
                                   'complete-sampled','complete-bounded','unknown','failed'])
def test_release_composition_uses_allowlist(status):
    with pytest.raises(ValueError,match='composition'):
        runner.require_release_ready(stage_profile(),dict(status=status),[])


def test_release_requires_complete_confirmed_services():
    profile = stage_profile()
    # A finite confirmed-mapping population need not certify the global inventory.
    profile['stage_service']['inventory_complete'] = False
    runner.require_release_ready(profile,dict(status='complete-observed'),[])
    with pytest.raises(ValueError,match='gaps'):
        runner.require_release_ready(profile,dict(status='complete-observed'),[dict(reason='missing')])
    profile['stage_service']['validated'] = False
    with pytest.raises(ValueError,match='validated'):
        runner.require_release_ready(profile,dict(status='complete-observed'),[])


def test_driver_finalized_gate_checks_service_scope_and_composition(tmp_path):
    args = arguments(tmp_path)
    final = args.output_dir / 'finalized'
    manifest = dict(status='ready-with-measured-mappings',selector_replay=dict(status='passed'),
                    registry_promoted_records=1,hardware_rates_remeasured=True,service_gaps=[],
                    composition_calibration=dict(status='complete-observed'))
    runner.write(final/'calibration_manifest.json',manifest)
    runner.write(final/'registry.json',{})
    profile = stage_profile()
    runner.write(final/'calibration_device.json',profile)
    runner.gate('finalize',args.output_dir)
    manifest['composition_calibration']['status'] = 'incomplete-uncovered'
    runner.write(final/'calibration_manifest.json',manifest)
    with pytest.raises(ValueError,match='composition'):
        runner.gate('finalize',args.output_dir)
    manifest['composition_calibration']['status'] = 'complete-observed'
    runner.write(final/'calibration_manifest.json',manifest)
    profile['stage_service']['coverage']['coverage_complete'] = False
    runner.write(final/'calibration_device.json',profile)
    with pytest.raises(ValueError,match='declared scope'):
        runner.gate('finalize',args.output_dir)


def test_selected_semantics_and_csv(tmp_path):
    workload = dict(operator="structured-2x2", precision="fp32", logN=12, batch=1024,
                    placement="out-of-place", normalization="none", stage_matrix="1,0.25,-0.5,1")
    sample = {k: str(v) for k, v in workload.items() if k != "stage_matrix"}
    sample.update(stage_matrices="1:0.25:-0.5:1", direction="forward", accumulation="native",
                  element_stride="1", batch_stride="4096", modulus="0")
    record = dict(status="measured", correct=True, median_kernel_ms=2, samples=[sample], name="winner")
    cells = runner.comparison_cells([record], [workload])
    assert cells[0]["selected"]["name"] == "winner"
    with pytest.raises(ValueError, match="no confirmed"):
        runner.comparison_cells([], [workload])
    cells[0].update(measurements=[dict(name="cuButterfly", sample={"kernel_ms": "2"}),
                                 dict(name="reference", sample={"kernel_ms": "4"})],
                    baseline_plan=dict(available=[dict(name="reference", external=True)],
                                       unavailable=[dict(name="missing", reason="unsupported")]))
    path = tmp_path / "comparison.csv"
    runner.write_comparison_csv(path, {"cells": cells})
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["speedup"] == "2.0"
    assert rows[1]["reference_ms"] == ""
    assert rows[1]["status"] == "unavailable"


def test_all_23_requested_semantics_match_confirmed_samples():
    from hardware_registry import canonical_semantics
    workloads = runner.read(runner.WORKLOADS)["workloads"]
    records = []
    for index, workload in enumerate(workloads):
        sample = canonical_semantics({"operator": "fft", **workload})
        if workload.get("stage_matrix"):
            sample["stage_matrices"] = ":".join(format(float(v), ".17g") for v in workload["stage_matrix"].split(","))
        records.append(dict(name=str(index), status="measured", correct=True,
                            median_kernel_ms=1, samples=[sample]))
    cells = runner.comparison_cells(records, workloads)
    assert len(cells) == 23
    assert [c["selected"]["name"] for c in cells] == list(map(str, range(23)))


def test_completed_phases_are_preserved_on_resume(tmp_path):
    args = arguments(tmp_path)
    args.resume = True
    runner.write(args.output_dir / "status.json", {
        "identity": runner.identity(args), "status": "running", "gaps": [],
        "phases": {name: {"status": "complete"} for name in ("calibrate", "finalize", "compare")}})
    runner.write(args.output_dir / "finalized/calibration_manifest.json", {})
    runner.write(args.output_dir / "comparison.json", {"gaps": []})
    execute = mock.Mock(side_effect=AssertionError("completed phase rerun"))
    with mock.patch.object(runner, "gate") as gate:
        assert runner.orchestrate(args, execute) == 0
        assert gate.call_count == 3
    execute.assert_not_called()


@pytest.mark.parametrize('targets,sm', [('80', 80), ('120', 120),
    ('120-real;120-virtual', 120), ('120;120-real', 120)])
def test_validate_build_derives_unique_sm(tmp_path, targets, sm):
    args = arguments(tmp_path)
    cache = args.build_dir / 'CMakeCache.txt'
    cache.write_text(cache.read_text().replace('STRING=80', 'STRING=' + targets))
    assert runner.validate_build(args.build_dir) == sm


@pytest.mark.parametrize('targets', ['', 'native', 'all', 'sm_120', '120a',
    '80;120', '120-real;80-virtual', '120;', '0'])
def test_validate_build_rejects_ambiguous_or_missing_architecture(tmp_path, targets):
    args = arguments(tmp_path)
    cache = args.build_dir / 'CMakeCache.txt'
    cache.write_text(cache.read_text().replace('STRING=80', 'STRING=' + targets))
    with pytest.raises(ValueError, match='CMAKE_CUDA_ARCHITECTURES'):
        runner.validate_build(args.build_dir)
    cache.write_text('\n'.join(line for line in cache.read_text().splitlines()
                               if not line.startswith('CMAKE_CUDA_ARCHITECTURES')))
    with pytest.raises(ValueError, match='CMAKE_CUDA_ARCHITECTURES'):
        runner.validate_build(args.build_dir)


def test_environment_uses_only_this_runs_post_calibration_model(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setenv('CUBUTTERFLY_HARDWARE_MODEL', '/foreign/a100.json')
    monkeypatch.setenv('CUBUTTERFLY_PROFILE_ROOT', '/foreign/profiles')
    for finalized in (False, True):
        env = runner.environment(args, finalized)
        assert 'CUBUTTERFLY_HARDWARE_MODEL' not in env
        assert 'CUBUTTERFLY_PROFILE_ROOT' not in env
    hardware = args.output_dir / 'hardware/hardware_model.json'
    hardware.parent.mkdir(parents=True)
    hardware.write_text('{}')
    assert 'CUBUTTERFLY_HARDWARE_MODEL' not in runner.environment(args)
    assert runner.environment(args, True)['CUBUTTERFLY_HARDWARE_MODEL'] == str(hardware)
