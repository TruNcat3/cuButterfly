#!/usr/bin/env python3
"""Prepare an explicit controlled-study subset without executing benchmarks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_controlled_experiment as controlled


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output, build_dir, *, case_ids=(), operators=()):
    if bool(case_ids) == bool(operators):
        raise ValueError("select explicit case IDs or operators, exclusively")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source, output, build_dir = (Path(path).resolve() for path in (source, output, build_dir))
    if output.exists() and any(output.iterdir()):
        raise ValueError("subset preparation requires a fresh or empty output directory")
    parent_sha = sha(source)
    parent = json.loads(source.read_text())
    controlled._validate_study(parent)
    known_ids = {case["id"] for case in parent["cases"]}
    known_operators = {case["workload"]["operator"] for case in parent["cases"]}
    if set(case_ids) - known_ids or set(operators) - known_operators:
        raise ValueError("requested subset includes absent case IDs or operators")
    selected = [case for case in parent["cases"]
                if case["id"] in case_ids or case["workload"]["operator"] in operators]
    if not selected:
        raise ValueError("subset is empty")
    subset = copy.deepcopy(parent)
    subset["cases"] = copy.deepcopy(selected)
    controlled._validate_study(subset)
    if subset["protocol"] != parent["protocol"] or subset["cases"] != selected:
        raise ValueError("subset changed its source protocol or case contents")
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "study.json"
    controlled._atomic_write(study_path, subset)
    journal = controlled.prepare_document(subset, study_path, build_dir)
    controlled._atomic_write(output / "prepared.json", journal)
    if sha(source) != parent_sha:
        raise ValueError("parent source changed during preparation")
    manifest = dict(schema="cubutterfly-controlled-subset-preparation-v1", status="prepared",
                    gpu_queried=False, gpu_verified=False, source_path=str(source), source_sha256=parent_sha,
                    target_path=str(study_path), target_sha256=sha(study_path),
                    prepared_path=str(output / "prepared.json"), prepared_sha256=sha(output / "prepared.json"),
                    script_sha256=sha(__file__), runner_sha256=sha(controlled.__file__),
                    build_dir=str(build_dir), selected_case_ids=[case["id"] for case in selected],
                    selected_operators=sorted({case["workload"]["operator"] for case in selected}),
                    cases=len(selected), cells=sum(len(case["variants"]) for case in selected),
                    protocol_preserved=True, case_contents_preserved=True,
                    library_sha256=sha(build_dir / "libcuntt.so") if (build_dir / "libcuntt.so").is_file() else None,
                    note="Explicit source-study subset only; no benchmark execution or performance results copied.")
    controlled._atomic_write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", default=[])
    selection.add_argument("--operator", action="append", choices=sorted(controlled.SUPPORTED_OPERATORS), default=[])
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.build_dir, case_ids=args.case_id, operators=args.operator)
    print(json.dumps({key: manifest[key] for key in ("status", "cases", "cells", "gpu_queried", "source_sha256", "target_sha256")}))


if __name__ == "__main__":
    main()
