import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_core_space(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = [
        ["1", "v06", "dataflow-radix4", "9-11", "64", "1", "0.20"],
        ["2", "v06", "dataflow-radix4", "9-11", "64", "1", "0.22"],
        ["1", "appt", "appt-online", "8-8-4", "64", "1", "0.60"],
        ["2", "appt", "appt-online", "8-8-4", "64", "1", "0.62"],
        ["1", "appt", "appt-online-radix4", "4-8-8", "64", "1", "0.40"],
        ["2", "appt", "appt-online-radix4", "4-8-8", "64", "1", "0.42"],
    ]
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "family", "physical_core", "weights", "word_bits",
            "batch", "kernel_ms"])
        writer.writerows(rows)

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/summarize_appt_core_space.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    records = list(csv.DictReader(summary.open()))
    radix4 = next(row for row in records
                  if row["physical_core"] == "appt-online-radix4")
    assert abs(float(radix4["kernel_ms"]) - 0.41) < 1.0e-12
    assert round(float(radix4["speedup_vs_v06"]), 6) == round(0.21 / 0.41, 6)
    report = markdown.read_text()
    assert "1.488x" in report
    assert "`4-8-8`" in report
