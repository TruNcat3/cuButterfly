#!/usr/bin/env python3
"""Freeze exact CPU-normalized views from the main operator evidence.

The CPU campaign and GPU campaign are separate measurements.  This exporter
joins them only by the complete semantic case identity and keeps the CPU
implementation explicit for each operator family.  It never substitutes a
different precision, shape, batch, layout, or CPU thread count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "paper/generated/current_refresh/snapshot.json"
DEFAULT_OUTPUT = ROOT / "paper/generated/current_refresh/cpu_normalized.json"
CPU_BACKENDS = {
    "fft": "fftw-cpu-t8",
    "fwht": "fxt-cpu-t1",
    "ntt": "hexl-cpu-t1",
    "subset-zeta": "fxt-cpu-t1",
    "superset-zeta": "fxt-cpu-t1",
    "structured-2x2": "butterfly-author-cpu-t1",
}
VALID_STATUS = {"pass", "accuracy-review-required"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def positive(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


CPU_SOURCE_LABEL = "results/install_compare_20261013/cpu_external_v2/results.json (external CPU evidence)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--cpu-results", type=Path, required=True,
                        help="Completed cpu_external_v2/results.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    cpu_report = json.loads(args.cpu_results.read_text())
    requests = {r["id"]: r for r in snapshot["operators"]["requests"]
                if r["cohort"] == "main66"}
    if len(requests) != 66:
        raise ValueError(f"expected 66 main requests, found {len(requests)}")

    cpu_rows = {}
    for row in cpu_report.get("rows", []):
        key = (row.get("case_id"), row.get("backend"))
        if key in cpu_rows:
            raise ValueError(f"duplicate CPU row: {key}")
        if row.get("case_id") not in requests:
            raise ValueError(f"CPU row is outside main66: {key}")
        request = requests[row["case_id"]]
        if canonical(row.get("semantics")) != canonical(request.get("semantics")):
            raise ValueError(f"CPU semantic mismatch: {key}")
        cpu_rows[key] = row

    missing_cpu = []
    for case_id, request in requests.items():
        expected = CPU_BACKENDS.get(request["operator"])
        if expected is None or (case_id, expected) not in cpu_rows:
            missing_cpu.append({"case_id": case_id, "operator": request["operator"],
                                "precision": request["precision"],
                                "expected_cpu_backend": expected,
                                "reason": "missing exact CPU row"})
    if missing_cpu:
        raise ValueError(f"missing exact CPU rows: {len(missing_cpu)}")

    gpu_requests = requests
    gpu_pairs = {(p["case_id"], p["baseline"]): p for p in snapshot["operators"]["pairs"]
                 if p.get("cohort") == "main66"}
    records = []
    for case_id, request in sorted(gpu_requests.items()):
        cpu_backend = CPU_BACKENDS[request["operator"]]
        cpu = cpu_rows[(case_id, cpu_backend)]
        cpu_timing = cpu.get("timing", {})
        cpu_ms = cpu_timing.get("median_ms")
        cpu_samples = cpu_timing.get("samples_ms", [])
        if cpu.get("status") not in VALID_STATUS or not positive(cpu_ms):
            continue
        backends = [("cuButterfly", request.get("own_us"), request.get("status"),
                     request.get("timing_contract"), None)]
        baselines = sorted({p["baseline"] for (cid, _), p in gpu_pairs.items() if cid == case_id})
        for baseline in baselines:
            pair = gpu_pairs[(case_id, baseline)]
            backends.append((baseline, pair.get("baseline_us"), pair.get("status"),
                             pair.get("timing_contract"), pair))
        for backend, backend_us, backend_status, timing_contract, pair in backends:
            if backend_status not in VALID_STATUS or not positive(backend_us):
                continue
            cpu_us = float(cpu_ms) * 1000.0
            target_us = float(backend_us)
            records.append({
                "case_id": case_id,
                "operator": request["operator"],
                "precision": request["precision"],
                "shape": request.get("shape"),
                "batch": request.get("batch"),
                "inverse": request.get("inverse"),
                "backend": backend,
                "cpu_backend": cpu_backend,
                "cpu_us": cpu_us,
                "backend_us": target_us,
                "speedup": cpu_us / target_us,
                "cpu_status": cpu["status"],
                "backend_status": backend_status,
                "cpu_samples_us": [float(v) * 1000.0 for v in cpu_samples],
                "cpu_timing_contract": cpu.get("timing_contract", {}),
                "backend_timing_contract": timing_contract or {},
                "semantics": request["semantics"],
                "source_cpu": CPU_SOURCE_LABEL,
                "source_gpu": request.get("source"),
                "gpu_pair_source": pair.get("source") if pair else None,
            })

    if not records:
        raise ValueError("no CPU-normalized records")
    output = {
        "schema": "cpu-normalized-performance-v1",
        "qualification": "shared-host provisional CPU timings; CPU and GPU are separately measured",
        "snapshot": str(args.snapshot.relative_to(ROOT)),
        "snapshot_sha256": digest(args.snapshot),
        "cpu_results": CPU_SOURCE_LABEL,
        "cpu_results_sha256": digest(args.cpu_results),
        "cpu_identity": cpu_report.get("identity"),
        "gpu_identity": snapshot.get("identities", {}).get("main66"),
        "cpu_policy": CPU_BACKENDS,
        "timing_formula": "CPU median wall-clock microseconds / complete-call backend microseconds",
        "transfer_policy": "host-device transfers excluded on the GPU side; required operator conversions remain in both timings",
        "main_requests": len(requests),
        "records": records,
        "missing": missing_cpu,
        "coverage": {
            "cuButterfly": sum(r["backend"] == "cuButterfly" for r in records),
            "external_gpu_pairs": sum(r["backend"] != "cuButterfly" for r in records),
            "cpu_rows": len(cpu_rows),
            "gpu_external_pairs": sum(p.get("cohort") == "main66" and p.get("speedup") is not None
                                       for p in snapshot["operators"]["pairs"]),
        },
        "external_cohorts_excluded": [
            "cyclic90 has no exact rows in the 66-case CPU campaign",
            "FP16 is not in the 66-case CPU campaign",
            "CPU and GPU geometric means are not pooled across operators or hardware",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
