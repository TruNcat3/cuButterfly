import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "core_dataflow_prepare", ROOT / "paper/tools/prepare_core_dataflow_ablation.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

CONFIG = ROOT / "paper/experiments/core_dataflow_ablation.json"
TARGET = ROOT / "paper/experiments/targets/a100-80gb-gpu1.json"
APP = ROOT / "apps/cubutterfly_core_dataflow_ablation.cu"


def test_config_covers_strict_core_dataflow_cells():
    config = json.loads(CONFIG.read_text())
    target = json.loads(TARGET.read_text())
    prepare.validate(config, target)
    assert [case["workload"]["logN"] for case in config["cases"]] == [8, 12, 16]
    assert [(case["workload"]["latency_batch"], case["workload"]["throughput_batch"])
            for case in config["cases"]] == [(1, 65536), (1, 4096), (1, 256)]
    assert all(case["stage_count"] == case["workload"]["logN"] // 4 for case in config["cases"])


def test_cpu_prepare_records_direct_compile_and_all_2x2_commands(tmpdir):
    output = Path(str(tmpdir)) / "core_dataflow"
    manifest = prepare.prepare(CONFIG, TARGET, output)
    assert manifest["status"] == "prepared-not-executed"
    assert manifest["gpu_probe_performed"] is False
    assert manifest["executable_claim"] is False
    assert manifest["study"]["cells_total"] == 24
    assert manifest["compile"]["executed"] is False
    assert manifest["compile"]["argv"][0].endswith("/nvcc")
    assert "-arch=sm_80" in manifest["compile"]["argv"]
    assert all(not command["executed"] for command in manifest["measure"]["commands"])
    assert len(manifest["measure"]["commands"]) == 6
    assert all(command["argv"][command["argv"].index("--core") + 1] == "both"
               for command in manifest["measure"]["commands"])
    assert all(command["argv"][command["argv"].index("--organization") + 1] == "both"
               for command in manifest["measure"]["commands"])
    runbook = (output / "RUNBOOK.md").read_text()
    assert "stagewise-hbm" in runbook
    assert "resident-prefix-2" in runbook
    assert "CUDA events" in runbook
    assert "CUDA_VISIBLE_DEVICES=GPU-" in runbook


def test_standalone_source_has_true_organization_axis_and_no_public_plan():
    source = APP.read_text()
    assert "digit_reverse_kernel" in source
    assert "radix16_global_stage" in source
    assert "resident_prefix_kernel" in source
    assert "StagewiseHbm" in source
    assert "ResidentPrefix2" in source
    assert "NativeCodelet<4, false>" in source
    assert "DxCodelet<4, false>" in source
    assert "shared_index(unsigned row, unsigned column)" in source
    assert "column ^ row" in source
    assert "relative_l2_error" in source
    assert "ButterflyPlan" not in source
    assert "factor-streamed" not in source
    assert "factor_overlap" not in source


def test_prepare_rejects_non_radix16_or_missing_throughput():
    config = json.loads(CONFIG.read_text())
    target = json.loads(TARGET.read_text())
    config["cases"][1]["workload"]["logN"] = 10
    with pytest.raises(ValueError, match="radix-16 logN"):
        prepare.validate(config, target)
    config = json.loads(CONFIG.read_text())
    del config["cases"][0]["workload"]["throughput_batch"]
    with pytest.raises(ValueError, match="throughput_batch"):
        prepare.validate(config, target)
