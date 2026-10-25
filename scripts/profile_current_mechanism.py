#!/usr/bin/env python3
"""Collect one paired current-build mechanism profile.

The profile is intentionally a diagnostic companion to the timing journal. It
captures one representative FWHT request with the arithmetic core fixed and
the stage partition changed. NCU and NSYS outputs are kept separate from the
complete-call timing results because profiler replay changes execution cost.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_suite import require_exclusive_gpu


METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
])


def _command(variant: dict[str, object]) -> list[str]:
    command = list(variant["timed_command"])
    # The controlled runner's command is a complete semantic command. Replace
    # only timing knobs; all mapping and semantic fields remain identical.
    for flag, value in (("--warmup", "0"), ("--repeat", "1")):
        index = command.index(flag)
        command[index + 1] = value
    return command


def _visible_index(uuid: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        index, observed = [item.strip() for item in line.split(",", 1)]
        if observed.lower() == uuid.lower():
            return index
    raise RuntimeError(f"GPU UUID is not present: {uuid}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    case = next(c for c in journal["cases"] if c["id"] == "fwht-fp32-n18-b1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visible_gpu = _visible_index(args.gpu_uuid)
    manifest = {
        "schema": "cubutterfly-mechanism-profile-v1",
        "source_journal": str(args.journal.resolve()),
        "source_journal_sha256": __import__("hashlib").sha256(args.journal.read_bytes()).hexdigest(),
        "case": case["id"],
        "gpu_uuid": args.gpu_uuid,
        "metrics": METRICS.split(","),
        "profiler_timing_is_diagnostic": True,
        "variants": [],
    }
    for variant in case["variants"]:
        name = variant["id"]
        command = _command(variant)
        manifest["variants"].append({"id": name, "command": command})
        if not args.run:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        require_exclusive_gpu()
        nsys_base = ["nsys", "profile", "--trace=cuda", "--cuda-memory-usage=true",
                     "--force-overwrite=true", "-o", str(args.output_dir / f"{name}.nsys")] + command
        nsys = subprocess.run(nsys_base, env=env, text=True, capture_output=True)
        require_exclusive_gpu()
        ncu_base = ["ncu", "--target-processes", "all", "--replay-mode", "kernel",
                    "--cache-control", "none", "--clock-control", "base",
                    "--launch-count", "1", "--metrics", METRICS, "--page", "raw",
                    "--csv", "--force-overwrite", "--log-file",
                    str(args.output_dir / f"{name}.ncu.csv")] + command
        require_exclusive_gpu()
        ncu = subprocess.run(ncu_base, env=env, text=True, capture_output=True)
        require_exclusive_gpu()
        manifest["variants"][-1].update({
            "nsys_returncode": nsys.returncode,
            "ncu_returncode": ncu.returncode,
            "nsys_stdout": nsys.stdout,
            "nsys_stderr": nsys.stderr,
            "ncu_stdout": ncu.stdout,
            "ncu_stderr": ncu.stderr,
        })
        if nsys.returncode or ncu.returncode:
            manifest["status"] = "complete-with-profiler-gaps"
    manifest.setdefault("status", "prepared" if not args.run else "complete")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "variants": len(manifest["variants"]), "run": args.run}))
    return 0 if manifest["status"] in {"prepared", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
