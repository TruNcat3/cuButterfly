import gzip
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import capture_calibration_cpu_inputs as capture


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    build = tmp_path / "build"
    active = tmp_path / "active"
    build.mkdir()
    active.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"current-build")
    def identity(args):
        return {"binary": hashlib.sha256((args.build_dir / "cubutterfly_bench").read_bytes()).hexdigest(),
                "compile_mode": "research", "verify_batches": args.verify_batches}
    monkeypatch.setattr(capture, "measurement_identity", identity)
    measurement = identity(SimpleNamespace(build_dir=build, verify_batches=0))
    workloads = [{"operator": "fft", "precision": "fp32", "logN": log_n, "batch": 1}
                 for log_n in (10, 12, 20)]
    documents = {
        "calibration_device.json": {"hardware": {"device": "fake"}, "capabilities": {}},
        "search_measurements.json": [{"name": "historical", "correct": True}],
        "plan_descriptors.json": {"identity": {"measurement": measurement}, "descriptors": {}},
        "mapping_seed_snapshot.json": {"seeds": []},
        "search_coverage.json": {"measurement_identity": measurement,
                                 "search_protocol": {"search_stage_projection": "plan"},
                                 "workloads": [{"workload": workloads[0], "complete": True}]},
    }
    for name, document in documents.items():
        (active / name).write_text(json.dumps(document))
    specification = tmp_path / "workloads.json"
    specification.write_text(json.dumps({"schema": "cubutterfly-install-search-v1", "workloads": workloads}))
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"measurement": measurement, "scoring": "test"}))
    kwargs = dict(build_dir=build, calibration_dir=active, workloads=specification,
                  output=tmp_path / "captured" / "input.json", context_json=context)
    calls = []
    def runner(command):
        calls.append(command)
        assert "--list-design-points" in command
        assert not {"--warmup", "--repeat", "--trials"}.intersection(command)
        mappings = [{"backend": "shared-iterative", "local_stages": value} for value in (7, 5)]
        return SimpleNamespace(stdout="\n".join(json.dumps(mapping) for mapping in mappings))
    return kwargs, runner, calls, documents, workloads


def read_inventory(output, cell):
    with gzip.open(output.parent / cell["inventory"]["path"], "rt") as stream:
        return [json.loads(line) for line in stream]


def test_capture_complete_skip_full_inventory_order_and_frozen_snapshots(inputs, monkeypatch):
    kwargs, runner, calls, documents, workloads = inputs
    seed_requests = []
    def seeds(snapshot, workload):
        seed_requests.append((snapshot, workload))
        return [{"point": {"seed": 2}}, {"point": {"seed": 1}}]
    monkeypatch.setattr(capture, "candidates_for_workload", seeds)
    manifest = capture.capture(**kwargs, runner=runner)
    assert len(calls) == 2
    assert [cell["workload"] for cell in manifest["cells"]] == workloads
    assert read_inventory(kwargs["output"], manifest["cells"][0]) == []
    for cell in manifest["cells"][1:]:
        points = read_inventory(kwargs["output"], cell)
        assert [point["local_stages"] for point in points] == [7, 5]
        assert all(point["logN"] == cell["workload"]["logN"] for point in points)
        assert cell["seeds"] == [{"seed": 2}, {"seed": 1}]
    assert manifest["projection_scope"] == "plan"
    assert len(seed_requests) == 3
    for name in ("profile", "training", "descriptors"):
        reference = manifest[name]
        data = capture._read_ref(kwargs["output"].parent, reference)
        original = {"profile": "calibration_device.json", "training": "search_measurements.json",
                    "descriptors": "plan_descriptors.json"}[name]
        assert json.loads(data) == documents[original]
    assert json.loads(kwargs["output"].read_text()) == manifest
    capture.capture(**kwargs, runner=lambda command: pytest.fail("must reuse verified inventories"))


def test_interruption_resumes_without_rereading_active_state(inputs):
    kwargs, runner, calls, documents, _ = inputs
    def interrupted(command):
        if len(calls) == 1:
            raise KeyboardInterrupt()
        return runner(command)
    with pytest.raises(KeyboardInterrupt):
        capture.capture(**kwargs, runner=interrupted)
    assert not kwargs["output"].exists()
    # Active input updates must not mix into the already frozen capture.
    for name in documents:
        (kwargs["calibration_dir"] / name).write_text("{}")
    manifest = capture.capture(**kwargs, runner=runner)
    assert len(calls) == 2
    frozen = json.loads(capture._read_ref(kwargs["output"].parent, manifest["training"]))
    assert frozen == documents["search_measurements.json"]


@pytest.mark.parametrize("which", ["build", "expected", "descriptor"])
def test_reject_identity_mismatches_before_capture(inputs, which):
    kwargs, runner, calls, _, _ = inputs
    if which == "build":
        (kwargs["build_dir"] / "cubutterfly_bench").write_bytes(b"different-build")
    elif which == "expected":
        expected = kwargs["context_json"].with_name("expected.json")
        expected.write_text(json.dumps({"binary": "wrong"}))
        kwargs["expected_measurement_json"] = expected
    else:
        target = kwargs["calibration_dir"] / "plan_descriptors.json"
        target.write_text(json.dumps({"identity": {"measurement": {"binary": "wrong"}}}))
    with pytest.raises(ValueError, match="identity"):
        capture.capture(**kwargs, runner=runner)
    assert not calls
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("which", ["inventory", "profile", "environment"])
def test_resume_verifies_frozen_files_and_environment(inputs, monkeypatch, which):
    kwargs, runner, _, _, _ = inputs
    manifest = capture.capture(**kwargs, runner=runner)
    if which == "environment":
        monkeypatch.setenv("CUBUTTERFLY_PARTITION_BUDGET", "changed")
    else:
        reference = manifest["cells"][1]["inventory"] if which == "inventory" else manifest["profile"]
        (kwargs["output"].parent / reference["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        capture.capture(**kwargs, runner=lambda command: pytest.fail("cannot recapture silently"))


def test_help_requires_no_runner(capsys):
    with pytest.raises(SystemExit) as result:
        capture.main(["--help"])
    assert result.value.code == 0
    assert "not CPU-only" in capsys.readouterr().out


def test_rebind_scoring_context_preserves_capture_and_records_receipt(inputs):
    kwargs, runner, _, _, _ = inputs
    original = capture.capture(**kwargs, runner=runner)
    context = json.loads(kwargs["context_json"].read_text())
    context["scoring"] = "updated-scorer"
    context["sources"] = {"calibrate_local_hardware.py": "new-sha"}
    kwargs["context_json"].write_text(json.dumps(context))
    original_bytes = kwargs["output"].read_bytes()
    state_path = kwargs["output"].with_name(kwargs["output"].name + ".capture-state.json")
    original_state = state_path.read_bytes()
    target = kwargs["output"].with_name("active.json")
    updated = capture.rebind_score_context(kwargs["output"], kwargs["context_json"], target)
    assert updated["context"] == context
    assert updated["cells"] == original["cells"]
    assert updated["profile"] == original["profile"]
    receipt = updated["context_updates"][-1]
    assert receipt["previous_context"] == original["context"]
    assert receipt["current_context"] == context
    assert kwargs["output"].read_bytes() == original_bytes
    assert state_path.read_bytes() == original_state
    assert json.loads(target.read_text()) == updated
    with pytest.raises(ValueError, match="changed"):
        capture.capture(**kwargs, runner=lambda command: pytest.fail("strict capture cannot rebind"))


@pytest.mark.parametrize("which", ["measurement", "partition_budget", "projection_scope", "inventory"])
def test_rebind_rejects_changed_capture_identity_or_corrupt_reference(inputs, which):
    kwargs, runner, _, _, _ = inputs
    original = capture.capture(**kwargs, runner=runner)
    context = dict(original["context"], scoring="updated-scorer")
    if which == "measurement":
        context["measurement"] = {**context["measurement"], "compile_mode": "auto"}
    elif which == "inventory":
        reference = original["cells"][1]["inventory"]
        (kwargs["output"].parent / reference["path"]).write_bytes(b"corrupt")
    else:
        context[which] = "changed-policy"
    kwargs["context_json"].write_text(json.dumps(context))
    target = kwargs["output"].with_name("active.json")
    with pytest.raises(ValueError):
        capture.rebind_score_context(kwargs["output"], kwargs["context_json"], target)
    assert not target.exists()
    assert json.loads(kwargs["output"].read_text()) == original


def test_rebind_cli_and_original_overwrite_rejection(inputs):
    kwargs, runner, _, _, _ = inputs
    original = capture.capture(**kwargs, runner=runner)
    with pytest.raises(ValueError, match="different filename"):
        capture.rebind_score_context(kwargs["output"], kwargs["context_json"], kwargs["output"])
    target = kwargs["output"].with_name("active.json")
    assert capture.main(["--rebind-score-context", str(kwargs["output"]),
                         "--context-json", str(kwargs["context_json"]), "--output", str(target)]) == 0
    assert json.loads(target.read_text())["cells"] == original["cells"]


def test_capture_resume_keeps_code_identity_strict(inputs):
    kwargs, runner, _, _, _ = inputs
    capture.capture(**kwargs, runner=runner)
    state_path = kwargs["output"].with_name(kwargs["output"].name + ".capture-state.json")
    state = json.loads(state_path.read_text())
    state["invocation"]["source_sha256"]["calibration_space.py"] = "different-code"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="changed"):
        capture.capture(**kwargs, runner=lambda command: pytest.fail("strict code mismatch cannot query"))
