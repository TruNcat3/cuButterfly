import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_appt_static_generator(tmp_path):
    output = tmp_path / "space.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_appt_static_space.py"),
         "--word-bits", "32", "--batch", "4",
         "--fragment-widths", "16,32", "--writer-tiles", "1,2",
         "--producer-weights", "8", "--tail-weights", "12",
         "--writer-weights", "1,2", "--output", str(output)],
        check=True,
    )
    document = json.loads(output.read_text())
    assert document["schema"] == "cubutterfly-appt-static-space-v1"
    assert {point["fragment_width"] for point in document["points"]} == {16, 32}
    assert any(point["output_order"] == "natural" and
               point["writer_tiles_per_cta"] == 2 for point in document["points"])
    assert all(sum(point["stage_partition"]) == 20
               for point in document["points"])


def test_appt_static_summary(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = [
        {"word_bits": 32, "batch": 1, "variant": "v06", "kernel_ms": 1.0},
        {"word_bits": 32, "batch": 1, "variant": "static-fw32", "kernel_ms": 0.8},
        {"word_bits": 32, "batch": 1, "variant": "natural-fw32-wt1-r8-12-2", "kernel_ms": 0.9},
    ]
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_appt_static_layout.py"),
         str(raw), "--csv", str(summary), "--markdown", str(markdown)],
        check=True,
    )
    records = list(csv.DictReader(summary.open()))
    static = next(row for row in records if row["variant"] == "static-fw32")
    assert float(static["throughput_vs_v06"]) == 1.25
    assert "natural-fw32" in markdown.read_text()
