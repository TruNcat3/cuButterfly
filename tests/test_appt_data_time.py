import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_data_time_selects_role_per_td(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "variant", "scan_data_time", "roles",
                         "word_bits", "batch", "kernel_ms"])
        writer.writerows([
            [1, "v06", 0, "9-11", 64, 16, 1.0],
            [1, "writer-final", 1, "8-8-4", 64, 16, 2.0],
            [1, "data-time", 2, "8-8-4", 64, 16, 1.8],
            [1, "data-time", 2, "8-7-5", 64, 16, 1.6],
            [1, "data-time", 4, "8-8-4", 64, 16, 1.5],
        ])
    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/summarize_appt_data_time.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)
    report = markdown.read_text()
    assert "| 64 | 16 | 2 | 7 | `8-7-5` | 1.600000 | 1.250x | 0.625x |" in report
    assert "| 64 | 16 | 4 | 7 | `8-8-4` | 1.500000 | 1.333x | 0.667x |" in report
