import csv
import copy
import io
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_local_hardware as calibration
import evolutionary_search
import stage_cost_model
from calibration_space import candidate_id, mapping_point


def _profile():
    return {
        "device": "Mock A100",
        "compute_capability": "8.0",
        "global_memory_bytes": 40 << 30,
        "sm_count": 2,
        "capabilities": {
            "global_feedback_bytes_per_second": 1.0e12,
            "equivalent_butterflies_per_second": 1.0e12,
            "interstage_shared_bytes_per_second": 1.0e12,
            "cta_barriers_per_second": 1.0e9,
        },
    }


def _point(workload, mapping):
    return mapping_point(workload, mapping)


@pytest.mark.parametrize("resume_budget", [6, 0])
def test_staged_evolutionary_search_uses_probe_groups_records_failures_and_resumes(
    tmp_path, monkeypatch, resume_budget
):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for binary in ("cubutterfly_bench", "cuntt_bench", "cubutterfly_plan_probe"):
        (build_dir / binary).write_bytes(binary.encode())

    output = tmp_path / "search_coverage.json"
    profile_path = tmp_path / "calibration_device.json"
    profile_path.write_text(json.dumps(_profile()) + "\n")
    workloads_path = tmp_path / "workloads.json"
    workload = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": 4,
        "direction": "forward",
        "normalization": "none",
        "placement": "out-of-place",
        "element_stride": 1,
        "batch_stride": 256,
    }
    workloads_path.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "scope": "staged-test",
        "workloads": [workload],
    }) + "\n")

    base_mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "fft_core": "scalar",
        "compute_unit": "radix2",
        "stage_partition": [4, 4],
        "tile_threads": 128,
        "stage_overlap": False,
        "batch_tile_count": 1,
    }
    good_mapping = dict(base_mapping, tile_threads=256)
    bad_mapping = dict(base_mapping, tile_threads=512, proposal="unavailable-neighbor")
    next_mapping = dict(base_mapping, tile_threads=1024, proposal="resume-frontier")
    base = _point(workload, base_mapping)
    good = _point(workload, good_mapping)
    bad = _point(workload, bad_mapping)
    next_point = _point(workload, next_mapping)
    resume_points = [base] + [
        _point(workload, dict(base_mapping, tile_threads=threads, proposal=f"resume-{threads}"))
        for threads in (640, 768, 896, 1024, 1152)
    ]

    groups = [{
        "stage_count": 4,
        "grid_ctas": 2,
        "threads": 128,
        "live_shared_bytes": 1024,
        "launch_count": 1,
        "core": "scalar",
        "shared_layout": "linear",
        "compiler_resources_known": True,
    }]
    probe_calls = []
    benchmark_calls = []

    def fake_run(command, check=True):
        command = [str(value) for value in command]
        if command[0].endswith("cubutterfly_plan_probe") and "--hardware" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "sm_count": 2,
                    "max_blocks_per_sm": 32,
                    "max_threads_per_sm": 2048,
                    "shared_bytes_per_sm": 164 * 1024,
                    "registers_per_sm": 65536,
                }),
                stderr="",
            )
        if command[0].endswith("cubutterfly_plan_probe") and "--point-json" in command:
            requested = json.loads(command[command.index("--point-json") + 1])
            probe_calls.append(requested)
            requested_mapping = json.loads(requested["mapping_json"])
            if requested_mapping.get("proposal") == "unavailable-neighbor":
                response = {"status": "unavailable", "reason": "mock shape rejected"}
            else:
                    response = {
                        "status": "resolved",
                        "mapping_json": requested["mapping_json"],
                        "execution_groups_json": groups,
                        "runtime_fingerprint": "mock-runtime-v1",
                    }
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        if "--list-processing-units" in command:
            return SimpleNamespace(returncode=0, stdout="precision,logN,threads,ept\n", stderr="")
        if "--list-register-tile-mappings" in command:
            return SimpleNamespace(returncode=0, stdout="logN,local_stages\n", stderr="")
        if "--csv" in command:
            benchmark_calls.append(command)
            requested = json.loads(command[command.index("--mapping-json") + 1])
            sample = {
                "operator": "fft",
                "precision": "fp32",
                "logN": "8",
                "batch": "4",
                "backend": requested["backend"],
                "fft_core": requested.get("fft_core", "scalar"),
                "compute_unit": requested.get("compute_unit", "auto"),
                "core": requested.get("fft_core", "scalar"),
                "shared_layout": requested.get("shared_layout", "linear"),
                "correct": "1",
                "kernel_ms": "0.25" if requested.get("tile_threads") == 256 else "0.5",
                "mapping_json": json.dumps(requested, sort_keys=True, separators=(",", ":")),
                "execution_groups_json": json.dumps(groups),
                "runtime_fingerprint": "mock-runtime-v1",
                "warmup": "3",
                "repeat": "5",
            }
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(sample))
            writer.writeheader()
            writer.writerow(sample)
            return SimpleNamespace(returncode=0, stdout=buffer.getvalue(), stderr="")
        raise AssertionError(f"unexpected mock command: {command}")

    monkeypatch.setattr(calibration, "run", fake_run)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: [base])

    evolutionary_calls = []

    def fake_evolutionary(parents, inventory, measured_ids, score, limit=64):
        evolutionary_calls.append((parents, inventory, measured_ids, limit))
        if inventory:
            return [inventory[0]]
        return [good, bad]

    monkeypatch.setattr(evolutionary_search, "evolutionary_candidates", fake_evolutionary)

    real_predict = stage_cost_model.predict
    projected_samples = []
    resolved_samples = []

    def recording_predict(sample, model, explain=False):
        if sample.get("descriptor_source") == "projected":
            projected_samples.append(sample)
            if sample.get("tile_threads") == 512:
                # This syntactically cheap neighbor is rejected by the real
                # constructor. Its estimate must lose authority after probing.
                return 0.0
            raise ValueError("unresolved plan descriptor")
        resolved_samples.append(sample)
        return real_predict(sample, model, explain)

    monkeypatch.setattr(stage_cost_model, "predict", recording_predict)

    args = SimpleNamespace(
        build_dir=build_dir,
        search_workloads=workloads_path,
        resume_search=False,
        search_budget=2,
        search_seconds=0,
        search_finalists=1,
        operator_warmup=3,
        operator_repeat=5,
        operator_trials=1,
        verify_batches=0,
        seed_budget=0,
        compile_seconds=0,
        cost_model="staged",
        search_strategy="evolutionary",
    )

    result = calibration.run_compiled_search(args, output)
    assert len(result) == 2
    assert len(benchmark_calls) == 2
    assert evolutionary_calls
    assert evolutionary_calls[-1][0] == [base]
    assert {candidate_id(row["configuration"]) for row in result} == {
        candidate_id(base), candidate_id(good)
    }

    coverage = json.loads(output.read_text())
    cell = coverage["workloads"][0]
    assert cell["complete"] is True
    records = {record["id"]: record for record in cell["candidates"]}
    assert records[candidate_id(base)]["cost_breakdown"]["descriptor_source"] == "resolved-plan"
    assert records[candidate_id(good)]["cost_breakdown"]["descriptor_source"] == "resolved-plan"
    assert records[candidate_id(good)]["cost_breakdown"]["missing_primitives"] == []

    descriptors = json.loads((tmp_path / "plan_descriptors.json").read_text())["descriptors"]
    assert descriptors[candidate_id(good)]["status"] == "resolved"
    assert descriptors[candidate_id(good)]["execution_groups_json"] == groups
    assert descriptors[candidate_id(bad)] == {
        "status": "unavailable",
        "reason": "mock shape rejected",
    }
    assert resolved_samples and all(sample["execution_groups_json"] == groups for sample in resolved_samples)
    assert projected_samples

    # A matching completed cell and descriptor identity must reuse both
    # measurements and descriptors on resume.  The hardware capability probe is
    # intentionally repeated; benchmark and point probes are not.
    args.resume_search = True
    benchmark_count = len(benchmark_calls)
    point_probe_count = len(probe_calls)
    resumed = calibration.run_compiled_search(args, output)
    assert len(resumed) == 2
    assert len(benchmark_calls) == benchmark_count
    assert len(probe_calls) == point_probe_count

    # Simulate an interruption after the first persisted screen. The old
    # second selection remains in the audit but is outside the persisted
    # prefix; the resumed frontier adds a newly enumerated candidate.
    coverage = json.loads(output.read_text())
    partial_cell = coverage["workloads"][0]
    partial_cell["complete"] = False
    partial_cell["screened"] = 1
    prefix_candidate = next(candidate for candidate in partial_cell["candidates"]
                             if candidate.get("selection_index") == 0)
    prefix_metadata = copy.deepcopy(prefix_candidate)
    output.write_text(json.dumps(coverage) + "\n")
    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: [base, next_point])
    benchmark_count = len(benchmark_calls)
    args.search_strategy = "evolutionary"
    resumed = calibration.run_compiled_search(args, output)
    assert len(resumed) == 2
    assert len(benchmark_calls) == benchmark_count + 1
    assert all(json.loads(command[command.index("--mapping-json") + 1]).get("tile_threads") != 128
               for command in benchmark_calls[benchmark_count:])
    resumed_coverage = json.loads(output.read_text())
    resumed_cell = resumed_coverage["workloads"][0]
    resumed_by_id = {candidate["id"]: candidate for candidate in resumed_cell["candidates"]}
    assert resumed_by_id[prefix_candidate["id"]]["selection_reason"] == prefix_metadata["selection_reason"]
    assert resumed_by_id[prefix_candidate["id"]]["selection_index"] == prefix_metadata["selection_index"]
    assert resumed_by_id[candidate_id(next_point)]["selection_index"] == 1
    journal = json.loads((tmp_path / "search_measurements.json").read_text())
    assert {row["name"] for row in journal} >= {candidate_id(base), candidate_id(good), candidate_id(next_point)}

    # A protocol mismatch must start a fresh selection traversal, while valid
    # timing rows remain eligible for normal cache reuse.
    mismatch = json.loads(output.read_text())
    mismatch["workloads"][0]["complete"] = False
    mismatch["workloads"][0]["screened"] = 1
    mismatch["search_protocol"]["selection_policy_version"] = "old-selection-policy"
    output.write_text(json.dumps(mismatch) + "\n")
    before_mismatch_evolutionary = len(evolutionary_calls)
    before_mismatch_benchmarks = len(benchmark_calls)
    args.search_strategy = "model"
    calibration.run_compiled_search(args, output)
    assert len(evolutionary_calls) == before_mismatch_evolutionary
    assert len(benchmark_calls) == before_mismatch_benchmarks
    mismatch_cell = json.loads(output.read_text())["workloads"][0]
    assert candidate_id(good) not in {candidate["id"] for candidate in mismatch_cell["candidates"]}

    # An interrupted cell with four persisted screens must resume at the
    # fifth selection. The i=4 exploration turn keeps the same stratified
    # frontier as an uninterrupted run; only the i=5 evolutionary proposal is
    # rebuilt, and the two remaining candidates are the only new timings.
    def prepare_search_root(root):
        build = root / "build"
        build.mkdir(parents=True)
        for binary in ("cubutterfly_bench", "cuntt_bench", "cubutterfly_plan_probe"):
            (build / binary).write_bytes(binary.encode())
        (root / "calibration_device.json").write_text(json.dumps(_profile()) + "\n")
        return build

    partial_root = tmp_path / "interrupted-prefix"
    partial_build = prepare_search_root(partial_root)
    partial_output = partial_root / "search_coverage.json"
    partial_args = copy.copy(args)
    partial_args.build_dir = partial_build
    partial_args.resume_search = False
    partial_args.search_budget = resume_budget
    partial_args.search_strategy = "evolutionary"
    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: resume_points)
    benchmark_start = len(benchmark_calls)
    benchmark_attempts = 0

    def interrupting_run(command, check=True):
        nonlocal benchmark_attempts
        if "--csv" in command:
            benchmark_attempts += 1
            if benchmark_attempts == 5:
                raise KeyboardInterrupt()
        return fake_run(command, check)

    monkeypatch.setattr(calibration, "run", interrupting_run)
    with pytest.raises(KeyboardInterrupt):
        calibration.run_compiled_search(partial_args, partial_output)
    partial_audit = json.loads(partial_output.read_text())
    partial_cell = partial_audit["workloads"][0]
    assert partial_cell["screened"] == 4
    partial_journal = json.loads((partial_root / "search_measurements.json").read_text())
    assert len(partial_journal) == 4
    prefix_ids = {row["name"] for row in partial_journal}
    evolutionary_before_resume = len(evolutionary_calls)
    benchmark_before_resume = len(benchmark_calls)

    monkeypatch.setattr(calibration, "run", fake_run)
    partial_args.resume_search = True
    resumed = calibration.run_compiled_search(partial_args, partial_output)
    assert len(resumed) == 6
    assert len(evolutionary_calls) == evolutionary_before_resume + 1
    assert len(benchmark_calls) == benchmark_before_resume + 2
    resumed_sequence = [
        json.loads(command[command.index("--mapping-json") + 1])["tile_threads"]
        for command in benchmark_calls[benchmark_start:]
    ]
    resumed_new_ids = {
        candidate_id(_point(workload, json.loads(command[command.index("--mapping-json") + 1])))
        for command in benchmark_calls[benchmark_before_resume:]
    }
    assert not resumed_new_ids & prefix_ids
    assert len(prefix_ids) == 4
    assert prefix_ids <= {candidate_id(point) for point in resume_points}

    control_root = tmp_path / "uninterrupted-control"
    control_build = prepare_search_root(control_root)
    control_output = control_root / "search_coverage.json"
    control_args = copy.copy(partial_args)
    control_args.build_dir = control_build
    control_args.resume_search = False
    benchmark_control_start = len(benchmark_calls)
    evolutionary_control_start = len(evolutionary_calls)
    calibration.run_compiled_search(control_args, control_output)
    control_sequence = [
        json.loads(command[command.index("--mapping-json") + 1])["tile_threads"]
        for command in benchmark_calls[benchmark_control_start:]
    ]
    assert resumed_sequence == control_sequence
    assert len(evolutionary_calls) - evolutionary_control_start == 4


def test_actual_stage_probe_keeps_serial_full_tail_descriptors_for_scoring_and_resume(
    tmpdir, monkeypatch
):
    """The staged entry point must resolve batch-ring service views before scoring.

    This is intentionally a small wiring test.  ``stage_composition`` remains
    the implementation under test; the mock only supplies the normal probe
    documents for the original overlap point and its serial views.
    """
    tmp_path = pathlib.Path(str(tmpdir))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for binary in ("cubutterfly_bench", "cuntt_bench", "cubutterfly_plan_probe", "cubutterfly_stage_microbench"):
        (build_dir / binary).write_bytes(binary.encode())

    output = tmp_path / "search_coverage.json"
    profile = _profile()
    profile["stage_service"] = {"curves": {"mock-stage": {}}}
    (tmp_path / "calibration_device.json").write_text(json.dumps(profile) + "\n")

    workload = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": 9,
        "direction": "forward",
        "normalization": "none",
        "placement": "out-of-place",
        "element_stride": 1,
        "batch_stride": 256,
    }
    workloads_path = tmp_path / "workloads.json"
    workloads_path.write_text(json.dumps({
        "schema": "cubutterfly-install-search-v1",
        "scope": "actual-stage-probe-entry-test",
        "workloads": [workload],
    }) + "\n")

    mapping = {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "fft_core": "scalar",
        "compute_unit": "radix2",
        "stage_partition": [4, 4],
        "tile_threads": 128,
        "stage_overlap": True,
        "batch_tile_count": 4,
    }
    point = _point(workload, mapping)
    groups = [
        {
            "first_stage": 0,
            "stage_count": 4,
            "core": "scalar",
            "threads": 128,
            "grid_ctas": 2,
            "live_shared_bytes": 1024,
            "compiler_registers_per_thread": 32,
            "compiler_resources_known": True,
        },
        {
            "first_stage": 4,
            "stage_count": 4,
            "core": "scalar",
            "threads": 128,
            "grid_ctas": 2,
            "live_shared_bytes": 1024,
            "compiler_registers_per_thread": 32,
            "compiler_resources_known": True,
        },
    ]

    stage_describe_calls = []
    benchmark_calls = []
    scored_samples = []

    def stage_document(requested):
        requested = copy.deepcopy(requested)
        serial = not bool(requested.get("stage_overlap"))
        sample = dict(requested)
        sample["runtime_fingerprint"] = "mock-stage-runtime-v1"
        sample["execution_groups_json"] = json.dumps(groups, sort_keys=True, separators=(",", ":"))
        sample["stage_overlap"] = bool(requested.get("stage_overlap"))
        sample["factor_overlap"] = False
        sample["batch_tile_count"] = int(requested.get("batch_tile_count", 1))
        sample["mapping_json"] = requested["mapping_json"]
        if serial:
            sample["stage_overlap"] = False
            sample["batch_tile_count"] = 1
        return {
            "status": "resolved",
            "sample": sample,
            "groups": copy.deepcopy(groups),
            "runtime_fingerprint": "mock-stage-runtime-v1",
        }

    def fake_run(command, check=True):
        command = [str(value) for value in command]
        if command[0].endswith("cubutterfly_plan_probe") and "--hardware" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "sm_count": 2,
                    "max_blocks_per_sm": 32,
                    "max_threads_per_sm": 2048,
                    "shared_bytes_per_sm": 164 * 1024,
                    "registers_per_sm": 65536,
                }),
                stderr="",
            )
        if command[0].endswith("cubutterfly_stage_microbench"):
            requested = json.loads(command[command.index("--point-json") + 1])
            stage_describe_calls.append(requested)
            return SimpleNamespace(returncode=0, stdout=json.dumps(stage_document(requested)), stderr="")
        if command[0].endswith("cubutterfly_bench") and "--list-processing-units" in command:
            return SimpleNamespace(returncode=0, stdout="logN,threads,ept\n", stderr="")
        if command[0].endswith("cubutterfly_bench") and "--list-register-tile-mappings" in command:
            return SimpleNamespace(returncode=0, stdout="logN,local_stages\n", stderr="")
        if command[0].endswith("cubutterfly_bench") and "--csv" in command:
            benchmark_calls.append(command)
            requested = json.loads(command[command.index("--mapping-json") + 1])
            sample = {
                "operator": "fft",
                "precision": "fp32",
                "logN": "8",
                "batch": "9",
                "backend": requested["backend"],
                "fft_core": requested.get("fft_core", "scalar"),
                "compute_unit": requested.get("compute_unit", "radix2"),
                "correct": "1",
                "kernel_ms": "0.25",
                "mapping_json": json.dumps(requested, sort_keys=True, separators=(",", ":")),
                "execution_groups_json": json.dumps(groups, sort_keys=True, separators=(",", ":")),
                "runtime_fingerprint": "mock-stage-runtime-v1",
                "warmup": "3",
                "repeat": "5",
            }
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(sample))
            writer.writeheader()
            writer.writerow(sample)
            return SimpleNamespace(returncode=0, stdout=buffer.getvalue(), stderr="")
        raise AssertionError(f"unexpected mock command: {command}")

    monkeypatch.setattr(calibration, "run", fake_run)
    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: [point])
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)

    def recording_predict(sample, model, explain=False, **kwargs):
        scored_samples.append(copy.deepcopy(sample))
        if explain:
            return {"kernel_ms": 0.25, "descriptor_source": sample.get("descriptor_source")}
        return 0.25

    monkeypatch.setattr(stage_cost_model, "predict", recording_predict)

    args = SimpleNamespace(
        build_dir=build_dir,
        search_workloads=workloads_path,
        resume_search=False,
        search_budget=1,
        search_seconds=0,
        search_finalists=1,
        operator_warmup=3,
        operator_repeat=5,
        operator_trials=1,
        verify_batches=0,
        seed_budget=0,
        compile_seconds=0,
        cost_model="staged",
        stage_calibration="full",
        search_strategy="model",
    )

    result = calibration.run_compiled_search(args, output)
    assert len(result) == 1
    assert len(benchmark_calls) == 1
    assert {int(request["batch"]) for request in stage_describe_calls} >= {9, 4, 1}

    projections = [sample.get("stage_service_projection") for sample in scored_samples
                   if sample.get("stage_service_projection", {}).get("status") == "resolved"]
    assert projections
    projection = projections[-1]
    assert projection["tile_batch"] == 4
    assert projection["tail_batch"] == 1
    assert projection["tile_count"] == 3
    for view, expected_batch in ((projection["full"], 4), (projection["tail"], 1)):
        assert int(view["sample"]["batch"]) == expected_batch
        assert view["sample"]["stage_overlap"] is False
        assert json.loads(view["sample"]["execution_groups_json"]) == groups

    journal = json.loads((tmp_path / "search_measurements.json").read_text())
    measured = journal[0]
    measured_projection = measured["samples"][0]["stage_service_projection"]
    assert measured["samples"][0]["descriptor_source"] == "actual-stage-probe"
    assert measured_projection["full"]["sample"]["stage_overlap"] is False
    assert measured_projection["tail"]["sample"]["stage_overlap"] is False

    args.resume_search = True
    descriptor_count = len(stage_describe_calls)
    benchmark_count = len(benchmark_calls)
    resumed = calibration.run_compiled_search(args, output)
    assert len(resumed) == 1
    assert len(stage_describe_calls) == descriptor_count
    assert len(benchmark_calls) == benchmark_count
