import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_packed_stage6.py"
SPEC = importlib.util.spec_from_file_location("packed_stage6_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packed_stage6_report_selects_complete_win(tmp_path):
    path = tmp_path / "summary.csv"
    fields = ["label"] + [field for field, _ in MODULE.METRICS]
    def row(label, time_us, sectors, instructions):
        values = {field: 1 for field, _ in MODULE.METRICS}
        values.update(label=label, time_us=time_us,
                      global_load_sectors=sectors,
                      warp_instructions=instructions)
        return values

    rows = [
        row("v06", 1200, 30_000_000, 180_000_000),
        row("vector_d6", 1300, 40_000_000, 220_000_000),
        row("vector_d7", 1310, 31_000_000, 225_000_000),
        row("vector_packed_stage6", 1250, 31_200_000, 210_000_000),
        row("vector_packed_stage6_distributed", 1190, 30_500_000,
            211_000_000),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = MODULE.make_report(MODULE.read_summary(path))
    assert "recovered memory-level parallelism" in report
    assert "Lane32/vector16 time is 0.952x" in report
    assert "lane32/d7 warp instructions are 0.938x" in report
