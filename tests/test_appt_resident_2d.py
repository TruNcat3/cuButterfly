import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_resident_2d_selects_best_role_mapping(tmp_path):
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
            [1, "resident-2d", "10-8-4", 32, 16, 4.5],
            [1, "resident-2d", "12-8-2", 32, 16, 4.0],
            [2, "resident-2d", "12-8-2", 32, 16, 4.2],
        ])

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/summarize_appt_resident_2d.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    report = markdown.read_text()
    assert "| 32 | 16 | `12-8-2` | 4.100000 | 0.488x | 0.244x |" in report
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["roles"] == "12-8-2"]
    assert selected[0]["trials"] == "2"
