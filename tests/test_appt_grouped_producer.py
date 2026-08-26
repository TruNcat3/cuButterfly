import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_grouped_producer_selects_each_group(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "variant", "a_group", "roles", "word_bits",
            "batch", "kernel_ms",
        ])
        writer.writerows([
            [1, "v06", 0, "9-11", 32, 4, 1.0],
            [2, "v06", 0, "9-11", 32, 4, 1.2],
            [1, "radix4-matched", 1, "6-14-2", 32, 4, 2.0],
            [2, "radix4-matched", 1, "6-14-2", 32, 4, 2.2],
            [1, "grouped", 16, "5-15-2", 32, 4, 1.6],
            [2, "grouped", 16, "5-15-2", 32, 4, 1.8],
            [1, "grouped", 16, "6-14-2", 32, 4, 1.8],
            [2, "grouped", 16, "6-14-2", 32, 4, 2.0],
            [1, "grouped", 32, "5-15-2", 32, 4, 1.5],
            [2, "grouped", 32, "5-15-2", 32, 4, 1.7],
        ])

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        str(root / "scripts/summarize_appt_grouped_producer.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    report = markdown.read_text()
    assert "| 32 | 4 | 16 | `5-15-2` | 1.700000" in report
    assert "| 32 | 4 | 32 | `5-15-2` | 1.600000" in report
    rows = list(csv.DictReader(summary.open()))
    best = next(row for row in rows if row["variant"] == "grouped" and
                row["a_group"] == "32")
    assert abs(float(best["throughput_vs_radix4"]) - 1.3125) < 1.0e-12


def test_analyze_appt_grouped_producer_ncu(tmp_path):
    fields = [
        "label", "global_load_sectors", "global_store_sectors",
        "l1_hit_pct", "l2_hit_pct", "warp_instructions",
        "active_warps_pct", "barrier_stall_pct",
        "long_scoreboard_stall_pct", "mio_throttle_stall_pct",
        "registers_per_thread",
    ]
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bits in (32, 64):
            for batch in (1, 4, 16):
                for variant, scale in (("v06", 1), ("radix4", 2),
                                       ("group8", 1.5), ("group16", 1),
                                       ("group32", 0.5)):
                    row = {field: 10 for field in fields}
                    row["label"] = f"{variant}_u{bits}_b{batch}"
                    row["global_load_sectors"] = 10 * scale
                    row["global_store_sectors"] = 10 * scale
                    row["warp_instructions"] = 10 * scale
                    writer.writerow(row)

    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        str(root / "scripts/analyze_appt_grouped_producer_ncu.py"),
        str(summary), "--markdown", str(markdown),
    ], check=True)
    report = markdown.read_text()
    assert "`group16` | 0.500x | 0.500x" in report
    assert "`group32` | 0.250x | 0.250x" in report
