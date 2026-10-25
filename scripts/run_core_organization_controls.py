#!/usr/bin/env python3
"""Run the explicit current-build core-by-organization control cohort."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in study["cases"]:
        executable = args.bench_root / case["runner"]
        command = [str(executable)] + case["args"]
        row = {"id": case["id"], "command": command, "status": "prepared"}
        if args.run:
            completed = subprocess.run(
                command,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_index,
                         CUBUTTERFLY_COMPILE_MODE="research"),
                capture_output=True,
                text=True,
            )
            raw_path = args.output_dir / (case["id"] + ".csv")
            stderr_path = args.output_dir / (case["id"] + ".stderr")
            raw_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            row["returncode"] = completed.returncode
            row["stdout"] = str(raw_path)
            row["stderr"] = str(stderr_path)
            if completed.returncode == 0:
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                parsed = list(csv.DictReader(lines))
                if len(parsed) != 1:
                    raise RuntimeError("{} produced {} CSV rows".format(case["id"], len(parsed)))
                row["measurement"] = parsed[0]
                row["status"] = "complete"
            else:
                row["status"] = "failed"
        results.append(row)
    output = {
        "schema": "cubutterfly-core-organization-results-v1",
        "study": str(args.study.resolve()),
        "gpu_index": args.gpu_index,
        "run": args.run,
        "cases": results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete" if args.run else "prepared", "cases": len(results)}))


if __name__ == "__main__":
    main()
