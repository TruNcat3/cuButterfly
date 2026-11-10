#!/usr/bin/env python3
"""Audit recorded baseline commands before an explicitly reviewed adapter update.

This performs CPU replay against saved CSV rows, never executes a GPU
benchmark, and never modifies the acceptance journal. The caller must also
review the adapter diff: command equivalence alone cannot prove equivalence
of arbitrary code changes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import research_baselines as current


def normalized(command):
    command = [str(part) for part in command]
    if "--output" in command and "benchmark_external_fht.py" in command[1]:
        command[command.index("--output") + 1] = "<temporary-dao-csv>"
    return command


def replay(adapter, workload, row, protocol, paths):
    calls = []
    raw = row["raw_sample"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(raw))
    writer.writeheader()
    writer.writerow(raw)
    stdout = buffer.getvalue()

    def fake_run(command, **kwargs):
        calls.append([str(part) for part in command])
        if len(command) > 1 and str(command[1]).endswith("benchmark_external_fht.py"):
            Path(command[command.index("--output") + 1]).write_text(stdout)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=row.get("stderr", ""))

    specs = adapter.available_baselines(workload, paths)["available"]
    spec = next(spec for spec in specs if spec["name"] == row["name"])
    with patch.object(adapter.subprocess, "run", fake_run):
        result = adapter.run_baseline(workload, spec, protocol, paths)
    if len(calls) != 1:
        raise ValueError("unexpected baseline subprocess structure")
    for field in ("sample", "raw_sample", "correct", "binary_sha256", "verification", "external"):
        if result[field] != row[field]:
            raise ValueError(f"baseline replay changed recorded {field}: {row['name']}")
    return normalized(calls[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--previous-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    document = json.loads(args.journal.read_text())
    previous = document["comparison_identity"]
    if current.sha256(args.previous_adapter) != previous["adapter_sha256"]:
        raise ValueError("previous adapter differs from frozen comparison identity")
    spec = importlib.util.spec_from_file_location("previous_research_baselines", args.previous_adapter)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.ROOT = ROOT
    paths = dict(build_dir=Path(previous["cufft_harness"]["path"]).parent,
                 vkfft_binary=Path(previous["vkfft_binary"]["path"]) if previous.get("vkfft_binary") else None,
                 gpuntt_binary=Path(previous["gpuntt_binary"]["path"]) if previous.get("gpuntt_binary") else None,
                 fht_python=Path(previous["dao"]["python"]) if previous.get("dao") else None)
    identity = current.baseline_identity(paths)
    if {k: v for k, v in identity.items() if k != "adapter_sha256"} != {k: v for k, v in previous.items() if k != "adapter_sha256"}:
        raise ValueError("baseline executable, dependency, package or external harness changed")
    rows = []
    for cell in document["cells"]:
        for row in cell["measurements"]:
            if row["name"] == "cuButterfly":
                continue
            expected = normalized(row["command"])
            for adapter in (old, current):
                if replay(adapter, cell["workload"], row, document["identity"]["protocol"], paths) != expected:
                    raise ValueError(f"command changed for recorded baseline: {cell['id']}/{row['name']}")
            rows.append(dict(cell=cell["id"], name=row["name"], trial=row["trial"], equivalent=True))
    report = dict(schema="cubutterfly-baseline-resume-audit-v1", status="passed",
                  journal=str(args.journal.resolve()), journal_sha256=current.sha256(args.journal),
                  previous_adapter=str(args.previous_adapter.resolve()),
                  previous_adapter_sha256=previous["adapter_sha256"],
                  new_adapter_sha256=identity["adapter_sha256"], baseline_dependencies_unchanged=True,
                  accepted_rows_audited=len(rows), rows=rows, gpu_benchmarks_executed=0,
                  limitation="Requires manual review of the precise adapter diff; no general identity bypass.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("audit report already exists")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "accepted_rows_audited", "gpu_benchmarks_executed", "new_adapter_sha256")}))


if __name__ == "__main__":
    main()
