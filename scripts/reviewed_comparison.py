#!/usr/bin/env python3
"""Prepare, validate, or explicitly run the reviewed finite comparison campaign.

Preparation and validation never query a GPU. The existing queue owns execution,
isolation, checkpointing and summaries; this entry does not maintain a second runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import prepare_recalibrated_campaign as preparation
import run_recalibrated_campaign as runner


def validate(path: Path) -> dict:
    manifest = runner.load_manifest(path)
    if not manifest.get("measurement_contract"):
        raise runner.CampaignError("use a reviewed manifest with a measurement contract")
    if os.environ.get("LD_PRELOAD"):
        raise runner.CampaignError("reviewed campaign requires LD_PRELOAD unset")
    if manifest["measurement_contract"].get("execution_environment", {}) != runner.execution_environment():
        raise runner.CampaignError("execution environment changed since campaign preparation")
    runner.validate_plan_inputs(manifest, path)
    runner.fingerprint_builds(manifest)
    for _, _, step in runner._command_steps(manifest):
        runner.parse_command(step["run"], manifest["measurement_contract"]["gpu_uuid"], runner.ROOT)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare", help="CPU build (optional), then freeze a fresh campaign")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--gpu-uuid", required=True)
    prepare.add_argument("--hardware-profile", type=Path, required=True,
                         help="read device identity only; no timing or calibration success is imported")
    prepare.add_argument("--build", action="store_true", help="finish CPU compilation before freezing")
    prepare.add_argument("--jobs", type=int, default=2)
    check = commands.add_parser("validate", help="CPU-only identity and command validation")
    check.add_argument("--manifest", type=Path, required=True)
    calibrate = commands.add_parser("calibrate", help="explicit GPU confirmation of seeds for the frozen build")
    calibrate.add_argument("--manifest", type=Path, required=True)
    calibrate.add_argument("--input-profile", type=Path, required=True)
    calibrate.add_argument("--output-profile", type=Path, required=True)
    calibrate.add_argument("--additional-mappings-csv", type=Path, action="append", default=[])
    calibrate.add_argument("--resume", action="store_true")
    run = commands.add_parser("run", help="explicitly start/resume after GPU release and calibration")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--calibration-manifest", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("use a fresh output directory; existing experiments are immutable")
        hardware = json.loads(args.hardware_profile.read_text())
        if not args.gpu_uuid.startswith("GPU-") or not all(
                key in hardware for key in ("device", "compute_capability", "global_memory_bytes")):
            parser.error("an exact GPU UUID and complete hardware identity are required")
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        if args.build:
            subprocess.run(["cmake", "--build", str(preparation.CURRENT_BUILD),
                            "-j", str(args.jobs)], check=True)
        preparation.prepare(args.output, reviewed=True, gpu_uuid=args.gpu_uuid,
                            hardware_profile=args.hardware_profile)
        path = args.output.resolve() / "manifest.json"
    else:
        path = args.manifest.resolve()
    if args.action in ("calibrate", "run"):
        runner.require_campaign_active(path)
    manifest = validate(path)
    contract = manifest["measurement_contract"]
    if args.action == "calibrate":
        runner.validate_runtime_hardware(manifest, contract["gpu_uuid"])
        command = [sys.executable, str(preparation.ROOT / "scripts/finalize_recalibration.py"),
                   "--build-dir", str(preparation.CURRENT_BUILD),
                   "--input-profile", str(args.input_profile.resolve()),
                   "--output-dir", str(args.output_profile.resolve()),
                   "--revalidate-mappings", "--trials", "3", "--warmup", "100", "--repeat", "100"]
        for seed in args.additional_mappings_csv:
            command.extend(["--additional-mappings-csv", str(seed.resolve())])
        if args.resume:
            command.append("--resume")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=contract["gpu_uuid"],
                           CUBUTTERFLY_COMPILE_MODE=contract["compile_mode"])
        return subprocess.run(command, cwd=preparation.ROOT, env=environment, check=False).returncode
    if args.action == "run":
        return runner.run_campaign(path, contract["gpu_uuid"], args.calibration_manifest,
                                   resume=args.resume)
    print(json.dumps({"status": "prepared-not-run" if args.action == "prepare" else "validated-cpu-only",
                      "manifest": str(path), "revision": contract["revision"],
                      "gpu_execution": False, "rank1_cases": manifest["comparison_suite"]["cases"],
                      "xd_rows_including_alias": {k: v["semantic_rows_including_alias"]
                                                  for k, v in manifest["profiles"].items()},
                      "gap_rows": manifest["gap_population"]["row_count"],
                      "calibration_required": "matching build, hardware, compile mode and selector replay"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
