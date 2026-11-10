#!/usr/bin/env python3
"""Validate the A100 RC2 hardware-profile cohort and write a CPU-only runbook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_METRICS = {
    "kernel_time_us", "kernel_count", "launch_interval_us", "registers_per_thread",
    "shared_memory_bytes", "active_warps_pct", "dram_read_mib", "dram_write_mib",
    "dram_peak_pct", "l2_hit_pct", "barrier_stall_pct", "long_scoreboard_stall_pct",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/workloads/profile_rc2.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema") != "cubutterfly-profile-cohort-v1":
        errors.append("unexpected profile manifest schema")
    if manifest.get("status") != "prepared_not_run":
        errors.append("profile manifest must remain prepared_not_run until GPU execution starts")
    target = manifest.get("target")
    if target != "a100-80gb-gpu1":
        errors.append(f"unexpected target {target!r}")
    protocol_metrics = set(manifest.get("protocol", {}).get("metrics", []))
    if protocol_metrics != REQUIRED_METRICS:
        errors.append("profile metric list does not match the normalized evidence schema")
    cases = manifest.get("cases", [])
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        errors.append("profile case IDs must be unique non-empty strings")
    for case in cases:
        for field in ("operator", "workload", "axis", "baseline", "treatment", "question"):
            if field not in case:
                errors.append(f"{case.get('id', '<unknown>')}: missing {field}")
        if case.get("baseline") == case.get("treatment"):
            errors.append(f"{case.get('id')}: baseline and treatment alias each other")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness = {
        "schema": "cubutterfly-profile-readiness-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "target": target,
        "case_count": len(cases),
        "cases": [{"id": case["id"], "status": "pending_profile_run"} for case in cases],
        "errors": errors,
        "ready_for_gpu": not errors,
        "gpu_run_performed": False,
    }
    (args.output_dir / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    runbook = [
        "# A100 RC2 hardware profile runbook",
        "",
        f"Manifest: `{args.manifest.resolve()}`",
        f"Manifest SHA-256: `{readiness['manifest_sha256']}`",
        f"Target: `{target}`",
        "",
        "This runbook is CPU-only preparation. It does not launch NCU, NSYS, or a benchmark.",
        "Each case must first pass the normal full-batch correctness and timing identity check.",
        "Profile timing is diagnostic and must not replace accepted CUDA-event medians.",
        "",
        "## Cases",
    ]
    for case in cases:
        runbook.extend([
            f"- `{case['id']}`: {case['operator']} {case['workload']}; `{case['baseline']}` vs `{case['treatment']}` ({case['axis']}).",
            f"  Question: {case['question']}",
        ])
    runbook.extend([
        "",
        "## Acceptance",
        "- Match source snapshot, build hash, GPU UUID, mapping JSON, workload, and correctness record.",
        "- Store one normalized CSV row per case, variant, and metric in `profile_metrics.csv`.",
        "- Keep unavailable or failed cases in the readiness record and out of aggregates.",
    ])
    (args.output_dir / "RUNBOOK.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    print(json.dumps({"ready_for_gpu": not errors, "case_count": len(cases), "output": str(args.output_dir.resolve())}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
