#!/usr/bin/env python3
"""Build a local cuButterfly hardware profile and mapping model table."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from statistics import median


METRIC_FIELDS = {
    "global_copy_GB_s": "global_feedback_bytes_per_second",
    "butterfly_Gbutterfly_s": "equivalent_butterflies_per_second",
    "shared_exchange_GB_s": "interstage_shared_bytes_per_second",
    "barrier_GCTA_s": "cta_barriers_per_second",
}
PIPELINE_SPACES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the current GPU and generate a mapping model table.")
    parser.add_argument("--microbench", type=pathlib.Path, help="cuntt_hardware_microbench executable")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/local_hardware_profile"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--points", type=int, default=1 << 22)
    parser.add_argument("--blocks", type=int, default=640)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--logNs", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--spatial-budgets", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--word-bytes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_microbench(requested: pathlib.Path | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if requested:
        candidates.append(requested)
    env_path = os.environ.get("CUBUTTERFLY_HARDWARE_MICROBENCH")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    found = shutil.which("cuntt_hardware_microbench")
    if found:
        candidates.append(pathlib.Path(found))
    root = pathlib.Path(__file__).resolve().parents[1]
    candidates.extend(root / build / "cuntt_hardware_microbench" for build in ("build", "build-cuda118-cufftdx2", "build-cuda118", "build-11.8"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError("cuntt_hardware_microbench was not found; build first or pass --microbench PATH")


def run_trials(args: argparse.Namespace, microbench: pathlib.Path, raw_path: pathlib.Path) -> list[dict[str, str]]:
    command = [
        str(microbench),
        "--points", str(args.points),
        "--blocks", str(args.blocks),
        "--threads", str(args.threads),
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    rows: list[dict[str, str]] = []
    for trial in range(1, args.trials + 1):
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or str(error)
            raise RuntimeError(f"hardware probe failed on trial {trial}: {detail}") from error
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"hardware probe produced no CSV row on trial {trial}")
        row = next(csv.DictReader(lines[:2]))
        row["trial"] = str(trial)
        rows.append(row)
    with raw_path.open("w", newline="") as handle:
        fields = ["trial"] + [field for field in rows[0] if field != "trial"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows: list[dict[str, str]], args: argparse.Namespace, microbench: pathlib.Path) -> dict:
    if any(row.get("pipeline_correct") != "1" for row in rows):
        raise RuntimeError("hardware probe failed stage-pipeline correctness on at least one trial")
    return {
        "schema": "cubutterfly-hardware-profile-v1",
        "status": "calibrated-local",
        "device": rows[0]["device"],
        "compute_capability": rows[0]["compute_capability"],
        "sm_count": int(rows[0]["sm_count"]),
        "trials": len(rows),
        "conditions": {name: int(getattr(args, name)) for name in ("points", "blocks", "threads", "iterations", "warmup", "repeat")},
        "capabilities": {
            output_name: median(float(row[input_name]) for row in rows) * 1.0e9
            for input_name, output_name in METRIC_FIELDS.items()
        },
        "measured_stage_pipeline": {
            f"Us{stage_space}_butterflies_per_second": median(float(row[f"pipeline_us{stage_space}_Gbutterfly_s"]) for row in rows) * 1.0e9
            for stage_space in PIPELINE_SPACES
        },
        "provenance": {
            "microbench": str(microbench),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "host": platform.node(),
            "platform": platform.platform(),
        },
    }


def limiting_rate(capabilities: dict, us: int, ud: int, word_bytes: int, barrier_model: str):
    spatial_cells = us * ud
    rates = {
        "compute": capabilities["equivalent_butterflies_per_second"] / spatial_cells,
        "boundary": capabilities["global_feedback_bytes_per_second"] / (4 * ud * word_bytes),
    }
    if us > 1:
        rates["interstage"] = capabilities["interstage_shared_bytes_per_second"] / (4 * ud * (us - 1) * word_bytes)
        if barrier_model == "cta":
            rates["synchronization"] = capabilities["cta_barriers_per_second"] / (us - 1)
    bottleneck = min(rates, key=rates.get)
    return rates[bottleneck], bottleneck, rates


def build_model_table(profile: dict, args: argparse.Namespace, output: pathlib.Path) -> int:
    records = []
    for log_n in args.logNs:
        if log_n <= 0:
            raise ValueError("logN values must be positive")
        for budget in args.spatial_budgets:
            if budget <= 0:
                raise ValueError("spatial budgets must be positive")
            for word_bytes in args.word_bytes:
                if word_bytes <= 0:
                    raise ValueError("word-byte values must be positive")
                for us in range(1, log_n + 1):
                    if budget % us:
                        continue
                    ud = budget // us
                    ts = math.ceil(log_n / us)
                    td = math.ceil(((1 << log_n) // 2) / ud)
                    for barrier_model in ("cta", "none"):
                        rate, bottleneck, rates = limiting_rate(profile["capabilities"], us, ud, word_bytes, barrier_model)
                        records.append({
                            "device": profile["device"],
                            "compute_capability": profile["compute_capability"],
                            "logN": log_n,
                            "word_bytes": word_bytes,
                            "spatial_budget_C": budget,
                            "stage_space_Us": us,
                            "data_space_Ud": ud,
                            "stage_time_Ts": ts,
                            "data_time_Td": td,
                            "stage_utilization": log_n / (us * ts),
                            "ideal_body_steps": ts * td,
                            "compute_step_rate": rates["compute"],
                            "boundary_step_rate": rates["boundary"],
                            "interstage_step_rate": rates.get("interstage", ""),
                            "synchronization_step_rate": rates.get("synchronization", ""),
                            "limiting_capability": bottleneck,
                            "calibrated_body_us": (ts * td) / rate * 1.0e6,
                            "barrier_model": barrier_model,
                        })
    records.sort(key=lambda row: (int(row["logN"]), int(row["word_bytes"]), int(row["spatial_budget_C"]), float(row["calibrated_body_us"])))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> int:
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "hardware_profile.json"
    if profile_path.exists() and not args.force:
        raise SystemExit(f"profile exists: {profile_path}; pass --force to replace it")
    microbench = find_microbench(args.microbench)
    raw_path = output_dir / "hardware_capabilities_raw.csv"
    rows = run_trials(args, microbench, raw_path)
    profile = summarize(rows, args, microbench)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    model_path = output_dir / "unfolding_model.csv"
    model_rows = build_model_table(profile, args, model_path)
    manifest = {
        "schema": "cubutterfly-hardware-initialization-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": str(profile_path),
        "raw": str(raw_path),
        "model_table": str(model_path),
        "model_rows": model_rows,
        "command": " ".join([sys.executable, *sys.argv]),
        "status": "ready-for-local-model-ranking",
        "warning": "This profile is hardware-specific; do not reuse it on another GPU fingerprint.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"device: {profile['device']} (sm_{profile['compute_capability'].replace('.', '')})")
    print(f"profile: {profile_path}")
    print(f"model table: {model_path} ({model_rows} rows)")
    print(f"manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
