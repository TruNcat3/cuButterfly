#!/usr/bin/env python3
"""Run the already-prepared CT precision/inverse cohort on one GPU.

GS records are intentionally never launched by this command.  The runner
records stdout/stderr and the complete CSV row so correctness and unsupported
contracts remain auditable.  CUDA-event kernel time is taken from the bench
row; process startup and compilation are reported separately by E12.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]


def run(prepared: Path, output: Path, device: Optional[str]) -> dict[str, Any]:
    document = json.loads(prepared.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(document["gpu_commands"]):
        command = [str(ROOT / item["command"][0]), *item["command"][1:]]
        command += ["--warmup", "100", "--repeat", "100"]
        env = dict(os.environ)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        samples = list(csv.DictReader(io.StringIO(process.stdout)))
        sample = samples[-1] if samples else None
        rows.append({
            "index": index,
            "identity": {key: item.get(key) for key in (
                "operator", "precision_regime", "precision", "accumulation",
                "direction", "algorithm_form", "logN", "batch", "modulus",
            )},
            "command": command,
            "returncode": process.returncode,
            "correct": bool(sample and sample.get("correct") == "1"),
            "sample": sample,
            "stdout": process.stdout,
            "stderr": process.stderr,
        })
        print(f"{index + 1}/{len(document['gpu_commands'])} {item['operator']} {item['precision_regime']} {item['direction']}: "
              f"{'PASS' if rows[-1]['correct'] else 'FAIL'}", flush=True)
    report = {
        "schema": "cubutterfly-extended-semantics-run-v1",
        "prepared": str(prepared.resolve()),
        "device": device,
        "rows": rows,
        "summary": {
            "total": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "failed": sum(not row["correct"] for row in rows),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()
    report = run(args.prepared.resolve(), args.output.resolve(), args.device)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
