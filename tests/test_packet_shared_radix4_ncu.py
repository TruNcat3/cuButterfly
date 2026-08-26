import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_packet_shared_radix4_ncu.py"
SPEC = importlib.util.spec_from_file_location("packet_shared_ncu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packet_shared_report_promotes_low_instruction_winner(tmp_path):
    path = tmp_path / "summary.csv"
    fields = ["label"] + [field for field, _ in MODULE.METRICS]
    rows = []
    for label, time_us, instructions in (
            ("v06", 1250, 180), ("d7", 1330, 235),
            ("lane32", 1335, 237), ("packet128", 1100, 175)):
        row = {field: 1 for field, _ in MODULE.METRICS}
        row.update(label=label, time_us=time_us,
                   warp_instructions=instructions)
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = MODULE.make_report(MODULE.read_summary(path))
    assert "throughput/v0.6=1.136x" in report
    assert "eligible for broader" in report
