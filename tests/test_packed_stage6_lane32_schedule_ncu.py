import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_packed_stage6_lane32_schedule_ncu.py"
SPEC = importlib.util.spec_from_file_location("lane32_schedule_ncu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lane32_fixed_clock_report_promotes_winner(tmp_path):
    path = tmp_path / "summary.csv"
    fields = ["label"] + [field for field, _ in MODULE.METRICS]
    rows = []
    for label, time_us in (("v06", 1200), ("d7", 1250),
                           ("lane32_84_76", 1210),
                           ("lane32_85_75", 1190),
                           ("lane32_86_74", 1180)):
        row = {field: 1 for field, _ in MODULE.METRICS}
        row.update(label=label, time_us=time_us)
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = MODULE.make_report(MODULE.read_summary(path))
    assert "eligible for the V100" in report
    assert "weights=86:74" in report
    assert "throughput/v0.6=1.017x" in report
