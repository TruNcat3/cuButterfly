"""CPU-only contracts for paper queue preparation, separate from live runs."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_prepare", ROOT / "paper/tools/prepare_experiments.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def inputs(tmp_path):
    plan = json.loads((ROOT / "paper/experiments/plan.json").read_text())
    target = json.loads((ROOT / "paper/experiments/targets/a100-80gb-gpu1.json").read_text())
    target["acceptance_dir"] = "live"
    target["profile"] = "calibration.json"
    profile = {key: target[key] for key in ("device", "compute_capability", "global_memory_bytes")}
    profile["gpu_uuid"] = target["gpu_uuid"]
    for name, value in (("plan.json", plan), ("target.json", target), ("calibration.json", profile)):
        (tmp_path / name).write_text(json.dumps(value))
    return tmp_path


def test_prepare_never_launches_process_and_preserves_live_input(inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation must not query a GPU or execute recipes")
    monkeypatch.setattr(subprocess, "run", forbidden)
    live = inputs / "live"
    live.mkdir()
    target = json.loads((inputs / "target.json").read_text())
    original = json.dumps(dict(identity=dict(device=dict(uuid=target["gpu_uuid"])), status="running",
                               cells=[dict(status="complete"), dict(status="pending")]))
    (live / "acceptance.json").write_text(original)
    output = inputs / "prepared"
    result = prepare.prepare(inputs / "plan.json", inputs / "target.json", output, root=inputs)
    assert (live / "acceptance.json").read_text() == original
    assert result["e1_snapshot"]["completed"] == 1
    assert result["control_cases"] == {"layout": 3, "partition": 3, "batch": 12}
    assert result["control_module_count"] == 9
    assert all(not c["executed"] for c in result["commands"])
    assert next(c for c in result["commands"] if c["id"] == "e1-resume")["argv"][-1] == "--resume"
    assert next(c for c in result["commands"] if c["id"] == "controls-compile")["env"]["CUDA_VISIBLE_DEVICES"] == ""
    requirements = json.loads((output / "factorial_requirements.json").read_text())
    assert len(requirements["cases"]) == 10
    assert all(c["mapping"] is None for case in requirements["cases"] for c in case["cells"])
    assert json.loads((output / "heldout_protocol.json").read_text())["training_workloads"] is None
    assert json.loads((output / "search_pool_protocol.json").read_text())["pool_manifest"] is None


def test_control_comparisons_change_only_declared_axes_and_keep_word64_exact(inputs):
    plan = json.loads((inputs / "plan.json").read_text())
    for study in prepare.controlled_studies(plan).values():
        for case in study["cases"]:
            if case["workload"]["operator"] == "ntt":
                assert case["workload"]["modulus"] == 576460756061519873
            variants = {v["id"]: v["mapping"] for v in case["variants"]}
            for pair in case["comparisons"]:
                a, b = variants[pair["baseline"]], variants[pair["treatment"]]
                changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
                assert changed == set(pair["changed_axes"])


@pytest.mark.parametrize("field,value", [("gpu_uuid", "GPU-another-card"), ("global_memory_bytes", 4096)])
def test_wrong_target_profile_is_rejected(inputs, field, value):
    path = inputs / "calibration.json"
    profile = json.loads(path.read_text())
    profile[field] = value
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="mismatch"):
        prepare.prepare(inputs / "plan.json", inputs / "target.json", inputs / "prepared", root=inputs)


def test_live_or_ancestor_output_is_rejected(inputs):
    for output in (inputs, inputs / "live", inputs / "live" / "new"):
        with pytest.raises(ValueError, match="separate"):
            prepare.prepare(inputs / "plan.json", inputs / "target.json", output, root=inputs)


def test_changed_protocol_needs_new_preparation_directory(inputs):
    output = inputs / "prepared"
    prepare.prepare(inputs / "plan.json", inputs / "target.json", output, root=inputs)
    path = inputs / "plan.json"
    plan = json.loads(path.read_text())
    plan["protocol"]["repeat"] = 10
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="inputs changed"):
        prepare.prepare(path, inputs / "target.json", output, root=inputs)
