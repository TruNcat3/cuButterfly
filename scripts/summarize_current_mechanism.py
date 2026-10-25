#!/usr/bin/env python3
"""Summarize the current-build controlled timing and diagnostic profile."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess


def ncu_row(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    # NCU emits a units row before the numeric sample row.
    return next((row for row in rows if row.get("ID", "").isdigit()), {})


def nsys_metrics(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv", str(path)],
        check=True, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Start (ns)"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    kernels = [r for r in rows if r.get("Name") and "CUDA memcpy" not in r["Name"]]
    return {
        "kernel_count": len(kernels),
        "kernel_duration_us": sum(float(r["Duration (ns)"]) for r in kernels) / 1000.0,
        "total_cuda_rows": len(rows),
        "kernel_names": sorted({r["Name"] for r in kernels}),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "schema": "cubutterfly-current-mechanism-report-v1",
        "timing_source": str(args.timing.resolve()),
        "timing_sha256": hashlib.sha256(args.timing.read_bytes()).hexdigest(),
        "gpu_uuid": timing["identity"]["gpu_uuid"],
        "device": timing["identity"]["device"],
        "binaries": timing["identity"]["binaries"],
        "protocol": timing["protocol"],
        "cases": [],
        "profile": {},
        "interpretation": (
            "The stagewise variant is an intentionally unfused global-scratch control. "
            "Ratios expose organization and launch costs; they are not claims about an "
            "autotuned production winner or an external-library speedup."
        ),
    }
    for case in timing["cases"]:
        values = {}
        for variant in case["variants"]:
            samples = [float(t["sample"]["kernel_ms"]) for t in variant.get("trials", [])]
            values[variant["id"]] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples) if samples else None,
                "resolved_mapping": variant.get("preflight", {}).get("resolved_mapping"),
                "resources": variant.get("preflight", {}).get("resources"),
            }
        baseline = values.get("stagewise", {}).get("median_ms")
        treatment = values.get("resident8", {}).get("median_ms")
        if baseline and treatment:
            values["speedup_stagewise_over_resident8"] = baseline / treatment
        report["cases"].append({"id": case["id"], "workload": case["workload"], "variants": values})
    for name in ("stagewise", "resident8"):
        ncu = ncu_row(args.profile_dir / f"{name}.ncu.csv")
        nsys = nsys_metrics(args.profile_dir / f"{name}.nsys.nsys-rep")
        report["profile"][name] = {
            "kernel_name": ncu.get("Kernel Name"),
            "grid_size": ncu.get("launch__grid_size"),
            "block_size": ncu.get("launch__block_size"),
            "gpu_time_ns": ncu.get("gpu__time_duration.sum"),
            "dram_read_bytes": ncu.get("dram__bytes_read.sum"),
            "dram_write_bytes": ncu.get("dram__bytes_write.sum"),
            "registers_per_thread": ncu.get("launch__registers_per_thread"),
            "shared_bytes_per_block": ncu.get("launch__shared_mem_per_block"),
            "active_warps_pct": ncu.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "barrier_stall_pct": ncu.get("smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
            "long_scoreboard_stall_pct": ncu.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
            **nsys,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": len(report["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
