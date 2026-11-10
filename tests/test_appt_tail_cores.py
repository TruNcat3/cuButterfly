import csv
import subprocess
import sys
from pathlib import Path


def test_summarize_appt_tail_cores(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "physical_core", "weights", "word_bits", "batch",
            "kernel_ms",
        ])
        writer.writerows([
            [1, "v06", "9-11", 64, 1, 0.20],
            [2, "v06", "9-11", 64, 1, 0.22],
            [1, "register-tail", "10-5-5", 64, 1, 0.34],
            [2, "register-tail", "10-5-5", 64, 1, 0.36],
        ])

    summary = tmp_path / "summary.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/summarize_appt_tail_cores.py"),
        str(raw), "--csv", str(summary), "--markdown", str(markdown),
    ], check=True)

    rows = list(csv.DictReader(summary.open()))
    register = next(row for row in rows
                    if row["physical_core"] == "register-tail")
    assert abs(float(register["kernel_ms"]) - 0.35) < 1.0e-12
    assert abs(float(register["throughput_vs_v06"]) - 0.6) < 1.0e-12
    assert "`register-tail`" in markdown.read_text()


def test_analyze_appt_tail_cores_reports_matched_ablation(tmp_path):
    fields = [
        "label", "time_us", "dram_read_mib", "dram_write_mib",
        "global_load_sectors", "global_store_sectors",
        "local_load_sectors", "local_store_sectors", "dram_peak_pct",
        "l1_hit_pct", "l2_hit_pct", "shared_load_bank_conflicts",
        "shared_store_bank_conflicts", "warp_instructions",
        "integer_thread_instructions", "active_warps_pct",
        "barrier_stall_pct", "long_scoreboard_stall_pct",
        "mio_throttle_stall_pct", "registers_per_thread",
        "shared_mem_bytes",
    ]
    cores = [
        "v06", "warp", "cta-radix4", "fused-tail", "split-tail",
        "register-tail",
    ]

    def write_summary(path, previous):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for bits in (32, 64):
                for batch in (1, 4):
                    for core in cores:
                        row = {field: 1 for field in fields}
                        row.update({
                            "label": f"{core}_u{bits}_b{batch}",
                            "time_us": 50 if core == "v06" else 100,
                            "l1_hit_pct": 30,
                            "l2_hit_pct": 90,
                            "active_warps_pct": 25 if core != "v06" else 40,
                            "barrier_stall_pct": 20,
                            "long_scoreboard_stall_pct": 40,
                            "dram_peak_pct": 25,
                            "registers_per_thread": 48 if bits == 32 else 126,
                        })
                        if previous and bits == 32 and core == "register-tail":
                            row["time_us"] = 110
                        writer.writerow(row)

    current = tmp_path / "current.csv"
    previous = tmp_path / "previous.csv"
    write_summary(current, False)
    write_summary(previous, True)
    analysis_csv = tmp_path / "analysis.csv"
    markdown = tmp_path / "analysis.md"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "scripts/analyze_appt_tail_cores_ncu.py"),
        str(current), "--ablation-baseline", str(previous),
        "--csv", str(analysis_csv), "--markdown", str(markdown),
    ], check=True)

    report = markdown.read_text()
    assert "aggregate-readiness ablation improves NCU replay" in report
    assert "change by at most 0.00%" in report
    assert "l1_hit_pct" in next(csv.DictReader(analysis_csv.open()))

    comparison = tmp_path / "comparison.md"
    subprocess.run([
        sys.executable, str(root / "scripts/compare_appt_tail_ncu.py"),
        str(previous), str(current), "--markdown", str(comparison),
    ], check=True)
    comparison_report = comparison.read_text()
    assert "APPT Coefficient-Tree Cache Attribution" in comparison_report
    assert "1.100x" in comparison_report
