#!/usr/bin/env python3
"""Summarize the bounded FWHT external comparison without accepting accuracy."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/tools"))
from summarize_mechanism_cohort_20261012 import csv_rows

OUT = ROOT / "results/paper_targeted_20261012/mechanism/external/fwht_float_report"


def profile(implementation: str) -> dict:
    rows = csv_rows((OUT / f"fwht.{implementation}.ncu.csv").read_text(), '"ID"')
    kernels = {}
    for row in rows:
        item = kernels.setdefault(row["ID"], {
            "name": row["Kernel Name"], "block": row["Block Size"],
            "grid": row["Grid Size"], "metrics": {},
        })
        item["metrics"][row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
    sums = {}
    for item in kernels.values():
        for key, value in item["metrics"].items():
            if key.endswith(".sum"):
                sums[key] = sums.get(key, 0.0) + value
    return {
        "kernel_count": len(kernels), "kernels": list(kernels.values()),
        "per_call_counter_sums": sums,
        "l2_read_write_bytes_from_32byte_sectors": 32 * (
            sums.get("lts__t_sectors_op_read.sum", 0.0)
            + sums.get("lts__t_sectors_op_write.sum", 0.0)),
        "logical_calls": 1, "warmup": 100,
        "normal_timing_accepted": False,
        "profile_accuracy_status": "accuracy-review-required",
    }


def main() -> None:
    timing = {}
    correctness = {}
    for implementation in ("cubutterfly", "external"):
        trials = [json.loads((OUT / f"fwht.{implementation}.trial{i}.json").read_text())
                  for i in range(1, 4)]
        if any(t["status"] != "accuracy-review-required" for t in trials):
            raise RuntimeError("FWHT report must remain accuracy-review-required")
        timing[implementation] = {
            "median_us": 1000 * statistics.median(t["timing"]["median_ms"] for t in trials),
            "process_median_us": [1000 * t["timing"]["median_ms"] for t in trials],
            "independent_process_trials": 3,
            "protocol": trials[0]["timing"]["protocol"],
            "status": "accuracy-review-required",
        }
        correctness[implementation] = trials[0]["correctness"]
    summary = {
        "schema": "cubutterfly-external-fwht-float-summary-v1",
        "case": "FP32 FWHT N32768 B16 forward, resident [8,7]",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "timing": timing,
        "external_over_cubutterfly": timing["external"]["median_us"] / timing["cubutterfly"]["median_us"],
        "correctness": correctness,
        "profiles": {name: profile(name) for name in ("cubutterfly", "external")},
        "numerical_acceptance": "unassessed; complete error distributions retained",
        "interpretation": "Dao does not support N65536; this bounded N32768 comparison is a diagnostic timing/profile result and not a numerical pass.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"external_over_cubutterfly": summary["external_over_cubutterfly"],
                      "cubutterfly_us": timing["cubutterfly"]["median_us"],
                      "external_us": timing["external"]["median_us"]}))


if __name__ == "__main__":
    main()
