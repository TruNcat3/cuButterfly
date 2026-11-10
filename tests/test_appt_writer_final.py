import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_writer_final_selects_best_point(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "variant", "fragment_width", "writer_tiles", "roles",
            "word_bits", "batch", "kernel_ms",
        ])
        writer.writerows([
            [1, "v06", 0, 0, "9-11", 32, 4, 1.0],
            [2, "v06", 0, 0, "9-11", 32, 4, 1.2],
            [1, "grouped", 16, 2, "6-14-2", 32, 4, 1.6],
            [2, "grouped", 16, 2, "6-14-2", 32, 4, 1.8],
            [1, "writer-final", 16, 1, "6-12-4", 32, 4, 1.4],
            [2, "writer-final", 16, 1, "6-12-4", 32, 4, 1.6],
            [1, "writer-final", 32, 2, "6-11-5", 32, 4, 1.2],
            [2, "writer-final", 32, 2, "6-11-5", 32, 4, 1.4],
        ])

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/summarize_appt_writer_final.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    report = markdown.read_text()
    assert "| 32 | 4 | 32 | 2 | `6-11-5` | 1.300000" in report
    rows = list(csv.DictReader(summary.open()))
    selected = next(row for row in rows if row["variant"] == "writer-final" and
                    row["fragment_width"] == "32")
    assert abs(float(selected["throughput_vs_v06"]) - 1.1 / 1.3) < 1.0e-12
    assert abs(float(selected["throughput_vs_grouped"]) - 1.7 / 1.3) < 1.0e-12


def test_analyze_appt_writer_final_ncu(tmp_path):
    fields = [
        "label", "time_us", "dram_read_mib", "dram_write_mib",
        "global_load_sectors", "global_store_sectors", "l2_hit_pct",
        "warp_instructions", "integer_thread_instructions",
        "active_warps_pct", "barrier_stall_pct",
        "long_scoreboard_stall_pct", "mio_throttle_stall_pct",
        "registers_per_thread", "waves_per_sm",
    ]
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bits in (32, 64):
            for batch in (1, 4, 16):
                for variant, scale in (("v06", 0.5), ("grouped", 1.0),
                                       ("writer_final", 0.75)):
                    row = {field: 10 for field in fields}
                    row["label"] = f"{variant}_u{bits}_b{batch}"
                    row["global_load_sectors"] = 100 * scale
                    row["global_store_sectors"] = 80 * scale
                    row["warp_instructions"] = 60 * scale
                    row["integer_thread_instructions"] = 40 * scale
                    writer.writerow(row)

    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        str(root / "scripts/analyze_appt_writer_final_ncu.py"),
        str(summary), "--markdown", str(markdown),
    ], check=True)
    report = markdown.read_text()
    assert "`writer_final` | 10.0 | 0.750x | 0.750x" in report
    assert "0.750x | 0.750x | 10.0%" in report
