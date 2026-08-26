import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v07_instruction_attribution.py"
SPEC = importlib.util.spec_from_file_location("instruction_attribution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_source(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Address", "Source", "Source", "Instructions Executed"])
        writer.writerows(rows)


def test_source_pc_dedup_and_category_report(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    write_source(baseline_path, [
        ["0x10", "line", "IMAD R1, R2, R3, R4", "100"],
        ["0x10", "inline", "IMAD R1, R2, R3, R4", "100"],
        ["0x20", "line", "LDS R1, [R2]", "40"],
    ])
    write_source(candidate_path, [
        ["0x30", "line", "IMAD R1, R2, R3, R4", "100"],
        ["0x40", "line", "SHFL.BFLY PT, R1, R2, 1, 31", "80"],
        ["0x50", "line", "BAR.SYNC 1", "20"],
    ])
    baseline = MODULE.aggregate(MODULE.read_source_instructions(baseline_path))
    candidate = MODULE.aggregate(MODULE.read_source_instructions(candidate_path))
    assert baseline["total"] == 140
    assert baseline["pcs"] == 2
    report = MODULE.make_report(baseline, candidate)
    assert "shuffle" in report
    assert "SHFL" in report
    assert "Optimize that physical-codelet path" in report
