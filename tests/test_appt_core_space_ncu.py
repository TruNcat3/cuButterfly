import csv
import subprocess
import sys
from pathlib import Path


def test_analyze_appt_core_space_ncu(tmp_path):
    fields = [
        "label", "time_us", "dram_read_mib", "dram_write_mib",
        "global_load_sectors", "global_store_sectors", "warp_instructions",
        "integer_thread_instructions", "active_warps_pct", "barrier_stall_pct",
        "long_scoreboard_stall_pct", "dram_peak_pct", "l1_hit_pct",
        "l2_hit_pct", "registers_per_thread",
    ]
    path = tmp_path / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bits in (32, 64):
            for batch in (1, 4):
                base = {field: "10" for field in fields}
                base["label"] = f"v06_u{bits}_b{batch}"
                writer.writerow(base)
                for prefix, scale in (("warp", 3), ("cta-radix4", 2)):
                    row = {field: str(10 * scale) for field in fields}
                    row["label"] = f"{prefix}_u{bits}_b{batch}"
                    writer.writerow(row)

    output = tmp_path / "analysis.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/analyze_appt_core_space_ncu.py"),
        str(path), "--csv", str(output), "--markdown", str(markdown),
    ], check=True)
    rows = list(csv.DictReader(output.open()))
    target = next(row for row in rows if row["word_bits"] == "64" and
                  row["batch"] == "4" and row["core"] == "cta-radix4")
    assert float(target["time_vs_v06"]) == 2.0
    assert "Arithmetic instruction count" in markdown.read_text()
