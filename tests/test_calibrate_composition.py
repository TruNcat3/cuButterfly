"""CPU-only composition measurement, resume and qualification contracts."""
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_composition as cc


def _csv(value):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(value))
    writer.writeheader()
    writer.writerow(value)
    return stream.getvalue()


def _inventory(slices=1):
    requests = []
    for role in ("bulk", "overlap"):
        mapping = {"backend": "shared-iterative", "stage_partition": [4, 4],
                   "stage_overlap": role == "overlap", "factor_slices": slices}
        point = dict(operator="fft", precision="fp32", batch=16, logN=8,
                     placement="out-of-place", mapping_json=json.dumps(mapping))
        requests.append(dict(id=role, pair_id="pair", role=role, family="shared",
                             workload="original", point=point))
    return dict(scope="test paired holdouts", requests=requests)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    for name in ("cubutterfly_bench", "cuntt_bench", "cubutterfly_stage_microbench", "libcuntt.so"):
        (build / name).write_bytes(name.encode())
    output = tmp_path / "output"
    profile = dict(device="test-device", compute_capability="8.0", global_memory_bytes="8192",
                   runtime_fingerprint="runtime-1")
    inventory = _inventory()
    monkeypatch.setattr(cc, "build_inventory", lambda records, **kw: copy.deepcopy(inventory))
    monkeypatch.setattr(cc.stage_cost_model, "fit", lambda records, profile: {"frozen": profile})
    events = []
    monkeypatch.setattr(cc, "resolve_batch_services", lambda *args: {"status": "supported"})

    def predict(sample, model, **kwargs):
        events.append("prediction")
        return dict(kernel_ms=2.0 if sample["stage_overlap"] else 1.0,
                    whole_plan_fallback_used=False, missing_primitives=[],
                    extrapolated_primitives=[], schedule_model="stage-composition")

    monkeypatch.setattr(cc.stage_cost_model, "predict", predict)
    state = dict(calls=[], timing_calls=0, corrupt=None, events=events, inventory=inventory,
                 described_runtime="runtime-1")

    def run(command, **kwargs):
        state["calls"].append(command)
        if "--device-identity" in command:
            return subprocess.CompletedProcess(command, 0, _csv(profile), "")
        if "--describe-only" in command:
            point = json.loads(command[command.index("--point-json") + 1])
            sample = {**point, "runtime_fingerprint": state["described_runtime"]}
            return subprocess.CompletedProcess(command, 0,
                json.dumps(dict(status="resolved", sample=sample, groups=[])), "")
        state["timing_calls"] += 1
        events.append("timing")
        report = json.loads((output / "composition_calibration.json").read_text())
        running = next(row for row in report["rows"] if row["status"] == "running")
        assert running["prediction"]["kernel_ms"] > 0
        assert "prediction_model_sha256" in report
        assert command[command.index("--verify-batches") + 1] == "0"
        mapping = command[command.index("--mapping-json") + 1]
        point = running["point"]
        sample = {**point, **profile, "mapping_json": mapping, "correct": "1",
                  "verified_batches": str(point["batch"]), "runtime_fingerprint": state["described_runtime"],
                  "kernel_ms": "2.0" if json.loads(mapping)["stage_overlap"] else "1.0"}
        if state["corrupt"]:
            sample.update(state["corrupt"])
        return subprocess.CompletedProcess(command, 0, _csv(sample), "")

    state.update(build=build, output=output, profile=profile, run=run)
    state["execute"] = lambda **kw: cc.run_calibration(build, output, profile, [],
        trials=2, warmup=1, repeat=2, run=run, guard=lambda: events.append("guard"), **kw)
    return state


def test_freezes_predictions_before_timing_and_keeps_negative_speedup(harness):
    report = harness["execute"]()
    assert report["status"] == "execution-finished"
    assert harness["timing_calls"] == 4
    assert report["summary"]["pairs"][0]["overlap_speedup_vs_bulk"] == .5
    assert report["summary"]["validation"]["status"] == "insufficient-holdout"
    assert harness["events"].index("prediction") < harness["events"].index("timing")
    assert all(len(row["samples"]) == 2 for row in report["rows"])


def test_structured_matrix_semantics_roundtrip_and_mismatch():
    point = dict(operator='structured-2x2', precision='fp64', batch=4, logN=8,
                 placement='out-of-place', mapping_json='{}',
                 stage_matrix=['1,0.1,-0.5,1', '2,0,0,2'])
    sample = {k: str(v) for k, v in point.items() if k != 'stage_matrix'}
    sample.update(stage_matrices='1:0.10000000000000001:-0.5:1x2:0:0:2',
                  device='test', compute_capability='8.0', runtime_fingerprint='r',
                  correct='1', verified_batches='4', kernel_ms='1')
    hardware = dict(device='test', compute_capability='8.0')
    cc.validate_sample(point, sample, hardware, 'r')
    with pytest.raises(ValueError, match='semantics differ: stage_matrices'):
        cc.validate_sample(point, {**sample, 'stage_matrices': '1:0:-0.5:1x2:0:0:2'}, hardware, 'r')
    with pytest.raises(ValueError, match='semantics differ: precision'):
        cc.validate_sample(point, {k: v for k, v in sample.items() if k != 'precision'}, hardware, 'r')


@pytest.mark.parametrize("corruption,match", [
    ({"verified_batches": "1"}, "full-batch"),
    ({"correct": "0"}, "full-batch"),
    ({"runtime_fingerprint": "other"}, "runtime"),
    ({"mapping_json": "{}"}, "mapping"),
    ({"batch": "1"}, "semantics"),
    ({"device": "other"}, "hardware"),
    ({"kernel_ms": "nan"}, "timing"),
])
def test_rejects_invalid_benchmark_evidence(harness, corruption, match):
    harness["corrupt"] = corruption
    with pytest.raises(ValueError, match=match):
        harness["execute"]()
    saved = json.loads((harness["output"] / "composition_calibration.json").read_text())
    assert saved["status"] == "stopped-investigation-required"
    assert saved["summary"]["validation"]["status"] != "passed"


def test_resume_does_not_repeat_completed_trials(harness):
    first = harness["execute"]()
    second = harness["execute"](resume=True)
    assert harness["timing_calls"] == 4
    assert second["rows"] == first["rows"]


def test_resumes_only_missing_trials_after_checkpoint(harness):
    harness["execute"]()
    path = harness["output"] / "composition_calibration.json"
    saved = json.loads(path.read_text())
    partial = saved["rows"][0]
    partial.update(status="running", samples=partial["samples"][:1])
    partial.pop("evidence_sha256")
    partial.pop("median_kernel_ms")
    saved["rows"] = [partial]
    path.write_text(json.dumps(saved))
    before = harness["timing_calls"]
    report = harness["execute"](resume=True)
    assert harness["timing_calls"] - before == 3
    assert all(len(row["samples"]) == 2 for row in report["rows"])


def test_fresh_described_runtime_must_match_profile(harness):
    harness["described_runtime"] = "other-runtime"
    with pytest.raises(ValueError, match="runtime"):
        harness["execute"]()
    assert harness["timing_calls"] == 0


@pytest.mark.parametrize("change", ["profile", "library"])
def test_resume_rejects_changed_measurement_identity(harness, change):
    harness["execute"]()
    if change == "profile":
        harness["profile"]["global_memory_bytes"] = "16384"
    else:
        (harness["build"] / "libcuntt.so").write_bytes(b"changed-library")
    with pytest.raises(ValueError, match="identity mismatch"):
        harness["execute"](resume=True)
    assert harness["timing_calls"] == 4


def test_resume_rejects_corrupted_completed_sample(harness):
    harness["execute"]()
    path = harness["output"] / "composition_calibration.json"
    saved = json.loads(path.read_text())
    saved["rows"][0]["samples"][0]["verified_batches"] = "1"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        harness["execute"](resume=True)


def test_requires_library_identity_for_nonempty_inventory(harness):
    (harness["build"] / "libcuntt.so").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        harness["execute"]()
    assert harness["timing_calls"] == 0


@pytest.mark.parametrize("change", ["mapping", "runtime", "median", "trial-count", "prediction"])
def test_resume_rejects_completed_evidence_changes(harness, change):
    harness["execute"]()
    path = harness["output"] / "composition_calibration.json"
    saved = json.loads(path.read_text())
    row = saved["rows"][0]
    if change == "mapping":
        row["samples"][0]["mapping_json"] = "{}"
    elif change == "runtime":
        row["samples"][0]["runtime_fingerprint"] = "other"
    elif change == "median":
        row["median_kernel_ms"] = 77.0
    elif change == "trial-count":
        row["samples"].pop()
    else:
        row["prediction"]["kernel_ms"] = 77.0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        harness["execute"](resume=True)
    assert harness["timing_calls"] == 4


def test_compile_budget_filename_is_ephemeral_but_budget_is_frozen(harness, monkeypatch):
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_FILE", "/tmp/first-budget")
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_SECONDS", "60")
    harness["execute"]()
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_FILE", "/tmp/second-budget")
    harness["execute"](resume=True)
    assert harness["timing_calls"] == 4
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_SECONDS", "120")
    with pytest.raises(ValueError, match="identity mismatch"):
        harness["execute"](resume=True)


def test_empty_overlap_population_never_qualifies(harness):
    harness["inventory"]["requests"].clear()
    report = harness["execute"]()
    assert report["status"] == "missing-overlap-population"
    assert report["summary"]["validation"]["status"] == "unvalidated"
    assert harness["calls"] == []


def test_composition_source_filter_keeps_only_current_runtime_records():
    records = [
        {"name": "old", "samples": [{"correct": "1", "runtime_fingerprint": "old"}]},
        {"name": "current", "samples": [{"correct": "1", "runtime_fingerprint": "current"}]},
        {"name": "unverified", "samples": [{"correct": "0", "runtime_fingerprint": "old"}]},
    ]
    kept, dropped = cc._current_runtime_records(records, "current")
    assert dropped == 1
    assert [row["name"] for row in kept] == ["current", "unverified"]


@pytest.mark.parametrize("patch,slices", [
    ({"missing_primitives": ["missing-stage"]}, 1),
    ({"extrapolated_primitives": ["outside-measurement"]}, 1),
    ({"whole_plan_fallback_used": True}, 1),
    ({"schedule_model": "serial-upper-estimate"}, 1),
    ({}, 2),
])
def test_uncovered_predictions_and_factor_slices_never_pass(harness, monkeypatch, patch, slices):
    harness["inventory"].update(_inventory(slices=slices))
    original = cc.stage_cost_model.predict
    monkeypatch.setattr(cc.stage_cost_model, "predict", lambda *a, **kw: {**original(*a, **kw), **patch})
    report = harness["execute"]()
    assert report["summary"]["status"] == "complete-observed"
    assert report["summary"]["missing_predictions"] == 2
    assert report["summary"]["validation"]["status"] == "uncovered"
    assert report["summary"]["pairs"][0]["overlap_speedup_vs_bulk"] == .5


def test_three_covered_holdouts_qualify_model_even_when_overlap_loses():
    requests, rows = [], []
    for index in range(3):
        for request in _inventory()["requests"]:
            request = copy.deepcopy(request)
            request.update(id=f"{index}-{request['id']}", pair_id=f"pair-{index}")
            request["point"]["batch"] += index
            requests.append(request)
            latency = 2.0 if request["role"] == "overlap" else 1.0
            rows.append({**request, "status": "measured", "prediction_covered": True,
                         "median_kernel_ms": latency, "prediction": {"kernel_ms": latency}})
    summary = cc.summarize({"requests": requests}, rows)
    assert summary["validation"]["status"] == "passed"
    assert summary["validation"]["covered_pairs"] == 3
    assert summary["validation"]["top1_accuracy"] == 1
    assert all(pair["overlap_speedup_vs_bulk"] == .5 for pair in summary["pairs"])
    rows[0]["prediction"]["kernel_ms"] = 10.0
    summary = cc.summarize({"requests": requests}, rows)
    assert summary["validation"]["status"] == "failed"
    assert summary["validation"]["top1_accuracy"] == pytest.approx(2 / 3)
