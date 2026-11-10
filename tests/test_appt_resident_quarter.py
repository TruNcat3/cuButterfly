import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_resident_quarter_selects_best_roles(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "variant", "scan_roles", "word_bits", "batch",
            "kernel_ms",
        ])
        writer.writerows([
            [1, "v06", "9-11", 32, 16, 1.0],
            [1, "writer-final", "6-12-4", 32, 16, 2.0],
            [1, "resident-2d", "12-8-2", 32, 16, 4.0],
            [1, "resident-quarter", "10-8-4", 32, 16, 3.2],
            [1, "resident-quarter", "12-8-2", 32, 16, 3.0],
            [2, "resident-quarter", "12-8-2", 32, 16, 3.2],
        ])

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        str(root / "scripts/summarize_appt_resident_quarter.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    assert (
        "| 32 | 16 | `12-8-2` | 3.100000 | 1.290x | 0.645x | 0.323x |"
        in markdown.read_text()
    )
