import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_strict_core_dataflow as strict
import run_controlled_experiment as controlled


CONFIG = ROOT / "paper/experiments/strict_core_dataflow.json"
TARGET = ROOT / "paper/experiments/targets/a100-80gb-gpu1.json"


def test_study_is_a_complete_fair_two_by_two_factorial():
    value = strict.study(json.loads(CONFIG.read_text()))
    normalized, _ = controlled._validate_study(value)
    assert len(normalized["cases"]) == 3
    assert {len(case["variants"]) for case in normalized["cases"]} == {4}
    assert {len(case["comparisons"]) for case in normalized["cases"]} == {4}
    for case in normalized["cases"]:
        mappings = {variant["id"]: variant["mapping"] for variant in case["variants"]}
        assert mappings["native-serial-factor"]["fft_core"] == "register-tile"
        assert mappings["imported-serial-factor"]["fft_core"] == "cufftdx-block"
        assert mappings["native-sliced-hybrid"]["fft_core"] == "register-tile"
        assert mappings["imported-sliced-hybrid"]["fft_core"] == "cufftdx-block"
        # Core comparisons differ only by the codelet implementation.
        assert {
            key for key in mappings["native-serial-factor"]
            if mappings["native-serial-factor"].get(key) != mappings["imported-serial-factor"].get(key)
        } == {"fft_core"}
        assert {
            key for key in mappings["native-serial-factor"]
            if mappings["native-serial-factor"].get(key) != mappings["native-sliced-hybrid"].get(key)
        } == {"factor_overlap"}
        assert mappings["native-serial-factor"]["factor_partition"] == mappings["native-sliced-hybrid"]["factor_partition"]


def test_prepare_is_cpu_only_and_deduplicates_core_modules(tmpdir, monkeypatch):
    output = Path(str(tmpdir)) / "strict"
    def forbidden(*args, **kwargs):
        raise AssertionError("strict preparation must not invoke a process or CUDA query")

    monkeypatch.setattr(controlled.subprocess, "run", forbidden)
    result = strict.prepare(CONFIG, TARGET, output)
    assert result["status"] == "prepared"
    assert result["gpu_queried"] is False
    assert result["study"]["cases"] == 3
    assert result["study"]["variants"] == 12
    assert result["study"]["comparisons"] == 12
    assert result["precompile"]["module_count"] == 6
    prepared = json.loads((output / "prepared.json").read_text())
    assert prepared["status"] == "prepared"
    assert prepared["identity"]["gpu_queried"] is False
    assert prepared["identity"]["executable_claim"] is False
    modules = json.loads((output / "precompile/modules.json").read_text())
    assert modules["complete"] is True
    assert modules["runtime_feasibility_verified"] is False
    assert all(module["request"]["backend"] == "factor-streamed" for module in modules["modules"])
    assert all(module["request"]["factor_partition"] for module in modules["modules"])
    runbook = (output / "RUNBOOK.md").read_text()
    assert "factor_slices=1" not in runbook
    assert "factor_overlap=false" in runbook
    assert "precompile_research.py" in runbook


def test_config_rejects_non_common_native_shape():
    value = json.loads(CONFIG.read_text())
    value["cases"][0]["factor_ept"] = 8
    with pytest.raises(ValueError, match="native register-tile"):
        strict.study(value)
