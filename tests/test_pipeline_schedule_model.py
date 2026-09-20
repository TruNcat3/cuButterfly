import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pipeline_schedule_model as model


def _raw(*, groups=(2, 3, 4), tiles=None):
    tiles = model.DEFAULT_TILES if tiles is None else tiles
    rows = []
    for group_count in groups:
        for tile_count in tiles:
            # Keep the curve deliberately simple so interpolation is exact.
            value = 0.2 + group_count * 0.01 + tile_count * 0.001
            rows.append({
                "groups": group_count,
                "tiles": tile_count,
                "trial_kernel_ms": [value, value, value],
                "correct": True,
            })
    return {
        "schema": model.SCHEMA,
        "hardware": {
            "device": "Test GPU",
            "compute_capability": "8.0",
            "global_memory_bytes": 4096,
            "sm_count": 20,
        },
        "protocol": {
            **model.DEFAULT_PROTOCOL,
            "launch_mode": "BatchPipeline::enqueue",
            "groups": list(model.DEFAULT_GROUPS),
            "tiles": list(model.DEFAULT_TILES),
            "tile_batch": 1,
        },
        "rows": rows,
    }


def _profile():
    return {
        "device": "Test GPU",
        "compute_capability": "8.0",
        "global_memory_bytes": 4096,
        "sm_count": 20,
    }


def test_piecewise_training_and_independent_holdout_are_separate():
    raw = _raw()
    holdout = next(row for row in raw["rows"] if row["groups"] == 2 and row["tiles"] == 3)
    holdout["trial_kernel_ms"] = [0.9, 0.9, 0.9]

    summary = model.summarize(raw)
    curve = next(item for item in summary["curves"] if item["groups"] == 2)
    assert [item["tiles"] for item in curve["training"]] == list(model.TRAINING_TILES)
    assert 3 not in [item["tiles"] for item in curve["training"]]
    check = next(item for item in curve["holdout"] if item["tiles"] == 3)
    assert check["independent"] is True
    assert check["relative_error"] > 0
    assert curve["validation"]["status"] == "covered"
    assert summary["validation"]["status"] == "covered"

    estimate = model.costs({"pipeline_scheduling": summary}, 2, 3)
    expected = 0.2 + 2 * 0.01 + 3 * 0.001
    assert estimate["covered"] is True
    assert estimate["measured_minimum_ms"] == pytest.approx(expected)
    assert estimate["validation"]["holdout_relative_error"] == pytest.approx(check["relative_error"])


def test_missing_and_outside_queries_are_uncovered_without_extrapolation():
    raw = _raw(groups=(2, 4), tiles=(1, 2, 4, 8, 16, 32))
    summary = model.summarize(raw)
    profile = {"pipeline_scheduling": summary}

    missing_group = model.costs(profile, 3, 4)
    assert missing_group["covered"] is False
    assert missing_group["measured_minimum_ms"] is None
    assert missing_group["validation"]["status"] == "missing"

    outside = model.costs(profile, 2, 33)
    assert outside["covered"] is False
    assert outside["measured_minimum_ms"] is None
    assert outside["validation"]["status"] == "outside"

    unsupported = model.costs(profile, 1, 4)
    assert unsupported["covered"] is False
    assert unsupported["validation"]["status"] == "missing"

    # A missing designated endpoint must not be bridged by a farther point.
    sparse = model.summarize(_raw(groups=(2,), tiles=(1, 2, 8, 16, 32)))
    gap = model.costs({"pipeline_scheduling": sparse}, 2, 3)
    assert gap["covered"] is False
    assert gap["validation"]["status"] == "missing"


def test_calibration_reuses_exact_identity_and_binary_change_remeasures(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    binary = tmp_path / "probe"
    binary.write_bytes(b"pipeline-probe-v1")
    path = tmp_path / "pipeline.json"
    raw = _raw()
    calls = []
    exclusive_calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(raw))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    first = model.calibrate(binary, path, _profile(), run, lambda: exclusive_calls.append(True))
    stale = json.loads(path.read_text())
    stale["summary"]["validation"]["status"] = "stale"
    path.write_text(json.dumps(stale))
    second = model.calibrate(binary, path, _profile(), run, lambda: exclusive_calls.append(True))
    assert first == second
    assert len(calls) == 1
    assert len(exclusive_calls) == 2

    binary.write_bytes(b"pipeline-probe-v2")
    model.calibrate(binary, path, _profile(), run, lambda: exclusive_calls.append(True))
    assert len(calls) == 2
    assert len(exclusive_calls) == 4


def test_hardware_mismatch_or_failed_measurement_keeps_previous_artifact(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    binary = tmp_path / "probe"
    binary.write_bytes(b"pipeline-probe-v1")
    path = tmp_path / "pipeline.json"
    raw = _raw()
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(raw))

    model.calibrate(binary, path, _profile(), run, lambda: None)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="device"):
        model.calibrate(binary, path, {**_profile(), "device": "Other GPU"}, run, lambda: None)
    assert path.read_bytes() == before

    def fail(_command):
        raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match="probe failed"):
        model.calibrate(binary, path, {**_profile(), "device": "Other GPU"}, fail, lambda: None)
    assert path.read_bytes() == before
    assert len(calls) == 2


def test_non_positive_trials_are_rejected():
    raw = _raw()
    raw["rows"][0]["trial_kernel_ms"][1] = 0
    with pytest.raises(ValueError, match="positive"):
        model.summarize(raw)

    raw = _raw()
    raw["rows"][0]["trial_kernel_ms"][0] = -0.001
    with pytest.raises(ValueError, match="positive"):
        model.summarize(raw)
