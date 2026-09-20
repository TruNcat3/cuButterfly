import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "core_dataflow_analysis", ROOT / "paper/tools/analyze_core_dataflow_ablation.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

CONFIG = ROOT / "paper/experiments/core_dataflow_ablation.json"


HEADER = ("device,device_name,compute_capability,logN,batch,core,organization,trial,kernel_ms,correct,"
          "max_error,max_reference,relative_error,l2_error,l2_reference,relative_l2_error,points\n")


def row(log_n, batch, core, organization, trial, kernel_ms, correct=1):
    return "0,A100,8.0,{},{},{},{},{},{},1,0.0,1.0,0.0,0.0,1.0,0.0,{}\n".format(
        log_n, batch, core, organization, trial, kernel_ms, batch * (1 << log_n))


def complete_rows():
    config = json.loads(CONFIG.read_text())
    rows = [HEADER]
    for case in config["cases"]:
        workload = case["workload"]
        for batch in (workload["latency_batch"], workload["throughput_batch"]):
            for core, organization in analysis.VARIANTS:
                for trial in range(config["protocol"]["trials"]):
                    base = 10.0 if organization == "stagewise-hbm" else 8.0
                    if core == "cufftdx-thread":
                        base -= 1.0
                    rows.append(row(workload["logN"], batch, core, organization, trial, base))
    return "".join(rows)


def test_complete_report_computes_both_main_effects_and_interaction(tmpdir):
    input_path = Path(str(tmpdir)) / "measurements.csv"
    output = Path(str(tmpdir)) / "analysis"
    input_path.write_text(complete_rows())
    report = analysis.analyze(input_path, CONFIG, output)
    assert report["status"] == "complete"
    assert report["input_rows"] == 72
    assert report["accepted_groups"] == 6
    first = report["groups"][0]
    assert first["resident_speedup_vs_stagewise"]["native"] == pytest.approx(1.25)
    assert first["core_speedup_cufftdx_vs_native"]["stagewise-hbm"] == pytest.approx(10.0 / 9.0)
    assert first["core_dataflow_interaction"] is not None
    assert (output / "SUMMARY.md").is_file()
    saved = json.loads((output / "summary.json").read_text())
    assert saved["accepted_groups"] == 6


def test_incorrect_or_missing_cell_is_retained_as_negative(tmpdir):
    input_path = Path(str(tmpdir)) / "measurements.csv"
    output = Path(str(tmpdir)) / "analysis"
    text = complete_rows()
    text = text.replace("0,A100,8.0,8,1,native,resident-prefix-2,0,8.0,1,",
                        "0,A100,8.0,8,1,native,resident-prefix-2,0,8.0,0,", 1)
    input_path.write_text(text)
    report = analysis.analyze(input_path, CONFIG, output)
    assert report["status"] == "incomplete"
    assert report["accepted_groups"] == 5
    group = next(item for item in report["groups"] if item["logN"] == 8 and item["batch"] == 1)
    assert any(item["cell"] == "native/resident-prefix-2" for item in group["negative_cells"])
    assert "native/resident-prefix-2" in (output / "SUMMARY.md").read_text()


def test_unknown_workload_is_rejected(tmpdir):
    input_path = Path(str(tmpdir)) / "measurements.csv"
    input_path.write_text(HEADER + row(10, 1, "native", "stagewise-hbm", 0, 1.0))
    with pytest.raises(ValueError, match="outside config"):
        analysis.analyze(input_path, CONFIG, Path(str(tmpdir)) / "analysis")
