import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_vector_stage6_distribution.py"
SPEC = importlib.util.spec_from_file_location("stage6_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage6_report_distinguishes_sector_and_time(tmp_path):
    path = tmp_path / "summary.csv"
    fields = ["label"] + [field for field, _ in MODULE.METRICS]
    rows = [
        ["v06", 1200, 280, 30_000_000, 8_000_000,
         180_000_000, 40, 48, 50, 4],
        ["vector_d6", 1300, 290, 40_000_000, 8_000_000,
         220_000_000, 106, 25, 27, 2],
        ["vector_d7", 1310, 275, 28_000_000, 8_000_000,
         225_000_000, 102, 25, 22, 3],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    report = MODULE.make_report(MODULE.read_summary(path))
    assert "0.700x" in report
    assert "does not reduce kernel time" in report
    assert "1.023x" in report
    assert "d6 excess over v0.6 removed by d7: 120.0%" in report
