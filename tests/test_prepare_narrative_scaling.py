"""CPU preparation contracts for the finite M1 narrative cohort."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_narrative_scaling as preparation
import static_hardware_profile


def config():
    return json.loads(preparation.DEFAULT_CONFIG.read_text())


def test_scope_precision_partitions_and_deduplication():
    study = preparation.make_study(config())
    requests, cells = preparation.module_requests(study)
    assert len(study["cases"]) == 18
    assert len(cells) == 54
    assert len(requests) == 18
    assert study["protocol"] == {"trials": 3, "warmup": 10, "repeat": 20, "verify_batches": 0}
    for case in study["cases"]:
        expected = "word64" if case["workload"]["operator"] == "ntt" else "fp64"
        assert case["workload"]["precision"] == expected
        if expected == "word64":
            assert case["workload"]["modulus"] == 576460756061519873
            assert json.loads(json.dumps(case["workload"]))["modulus"] == 576460756061519873
        assert case["variants"][0]["mapping"]["local_stage_partitions"] == []
        local4 = case["variants"][2]["mapping"]["local_stage_partitions"]
        assert local4 == ([[4, 4]] * 2 if case["workload"]["logN"] == 16 else [[4, 4, 2]] * 2)
    assert all("batch" not in request for request in requests.values())


def test_no_compile_preparation_never_invokes_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation.compile_module.subprocess, "run",
                        lambda *a, **k: pytest.fail("CPU generation must not invoke a subprocess"))
    state = preparation.prepare(output=tmp_path, build_dir=tmp_path / "build", do_compile=False)
    assert state["status"] == "prepared-not-compiled"
    assert state["gpu_queried"] is False
    assert len(state["resources"]) == 54
    prepared_cases = json.loads((tmp_path / "prepared.json").read_text())["cases"]
    assert len(prepared_cases) == 18
    for case in prepared_cases:
        if case["workload"]["operator"] == "ntt":
            command = case["variants"][0]["preflight_command"]
            assert command[command.index("--modulus") + 1] == "576460756061519873"
    assert state["numeric_policy"]["floating"] == "full-error-distribution-unfinished"


def test_capacity_failure_and_bounded_workers(tmp_path):
    model = static_hardware_profile.load(preparation.resolve(config()["hardware_model"]))
    model["hardware"]["shared_bytes_per_block"] = 1024
    with pytest.raises(ValueError, match="resource.*exceeds capacity"):
        preparation.resource_rows(preparation.make_study(config()), model)
    with pytest.raises(ValueError, match="worker count"):
        preparation.prepare(output=tmp_path, workers=5)


def test_forbids_precision_substitution():
    value = config()
    value["operators"][0]["precision"] = "fp32"
    with pytest.raises(ValueError, match="bounded M1 cohort"):
        preparation.make_study(value)


def fake_hardware_model(tmp_path, capability="8.0"):
    model = static_hardware_profile.load(preparation.resolve(config()["hardware_model"]))
    model["hardware"].update(device="Fake A100 40GB test fixture", memory_bytes=40 * 1024**3,
                             compute_capability=capability, sm_count=108)
    model["provenance"] = {"origin": "CPU-only test fixture; not live hardware"}
    path = tmp_path / "hardware_model.json"
    path.write_text(json.dumps(model))
    return path


def test_explicit_40gb_model_preserves_cohort_and_records_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation.compile_module.subprocess, "run",
                        lambda *a, **k: pytest.fail("static model must not probe GPU"))
    model_path = fake_hardware_model(tmp_path)
    state = preparation.prepare(output=tmp_path / "gpu40", build_dir=tmp_path / "build",
                                hardware_model=model_path, do_compile=False)
    assert state["status"] == "prepared-not-compiled"
    assert state["hardware_model_path"] == str(model_path.resolve())
    assert state["hardware_model_sha256"] == preparation.sha(model_path)
    assert state["hardware_model_source"] == "explicit-override"
    assert state["hardware"]["memory_bytes"] == 40 * 1024**3
    assert state["gpu_queried"] is False
    original = preparation.make_study(config())
    assert json.loads((tmp_path / "gpu40/study.json").read_text()) == original
    assert state["requests"] == preparation.module_requests(original)[0]
    assert len(state["cells"]) == 54


def test_wrong_hardware_capability_is_rejected_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation.compile_module.subprocess, "run",
                        lambda *a, **k: pytest.fail("static model must not probe GPU"))
    model_path = fake_hardware_model(tmp_path, capability="7.0")
    with pytest.raises(ValueError, match="compute capability.*sm80"):
        preparation.prepare(output=tmp_path / "wrong_gpu", hardware_model=model_path, do_compile=False)
    assert not (tmp_path / "wrong_gpu").exists()
