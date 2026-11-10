import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_packed_stage6_lane32_schedule.py"
SPEC = importlib.util.spec_from_file_location("lane32_schedule", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lane32_schedule_report_selects_best_point(tmp_path):
    raw = tmp_path / "raw.csv"
    fields = ["variant", "producer_dt", "consumer_dt", "producer_blocks",
              "consumer_blocks", "kernel_ms"]
    rows = [
        ["v06", 1, 1, 91, 69, 1.20],
        ["d7", 1, 1, 85, 75, 1.30],
        ["lane32", 1, 1, 82, 78, 1.25],
        ["lane32", 1, 2, 80, 80, 1.15],
    ]
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    summary, v06_ms, d7_ms = MODULE.summarize(raw)
    report = MODULE.report(summary, v06_ms, d7_ms)
    assert "exceeds the bracketed v0.6" in report
    assert "D_t=1:2, CTAs=80:80" in report


def test_lane32_schedule_report_rejects_unstable_controls(tmp_path):
    raw = tmp_path / "raw.csv"
    fields = ["variant", "producer_dt", "consumer_dt", "producer_blocks",
              "consumer_blocks", "kernel_ms"]
    rows = [
        ["v06", 1, 1, 91, 69, 1.2],
        ["v06", 1, 1, 91, 69, 1.5],
        ["d7", 1, 1, 85, 75, 1.3],
        ["lane32", 1, 1, 86, 74, 1.1],
    ]
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    summary, v06_ms, d7_ms = MODULE.summarize(raw)
    report = MODULE.report(summary, v06_ms, d7_ms)
    assert "exceed the 3% stability limit" in report
    assert "fixed-clock NCU is required" in report
