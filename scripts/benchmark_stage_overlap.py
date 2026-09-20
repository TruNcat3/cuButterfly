#!/usr/bin/env python3
"""Compare bulk and streaming schedules with identical FFT units and boundaries.

Research mode can specialize during plan creation, outside execution timing.
Each configuration is verified before timing trials; CUDA event timings include
the caller-stream join and exclude plan construction. --auto-mapping replays
the common selector's complete mapping, without imposing a fixed partition.
"""
import argparse
import csv
import io
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess

from run_comprehensive_suite import require_exclusive_gpu


def summarize_paired(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault((int(row["logN"]), int(row["batch"])), {})
        samples = group.setdefault(row["mode"], {})
        round_id = int(row["round"])
        elapsed = float(row["kernel_ms"])
        if round_id in samples or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("duplicate round or invalid paired timing")
        samples[round_id] = elapsed
    summary = []
    for (log_n, batch), modes in sorted(groups.items()):
        reference_rounds = set(modes.get("cufft", {}))
        if not reference_rounds or any(set(samples) != reference_rounds for samples in modes.values()):
            raise ValueError("paired modes must have matching cuFFT rounds")
        medians = {mode: statistics.median(samples.values()) for mode, samples in modes.items()}
        best = min((mode for mode in modes if mode != "cufft"), key=medians.get)
        comparisons = {}
        for mode in modes:
            if mode.startswith("xor-"):
                linear = "linear-" + mode[4:]
                if linear in modes:
                    comparisons[mode] = {
                        "median_latency_reduction_pct": 100 * (1 - medians[mode] / medians[linear]),
                        "median_same_round_speedup": statistics.median(
                            modes[linear][r] / modes[mode][r] for r in reference_rounds),
                    }
        summary.append({"logN": log_n, "batch": batch, "rounds": len(reference_rounds),
                        "median_ms": medians, "best": best,
                        "best_over_cufft_latency": medians[best] / medians["cufft"],
                        "xor_vs_same_mapping": comparisons})
    if summary:
        # Throughput is inverse latency.  Keep this aggregate explicit so a
        # benchmark cannot accidentally report the latency ratio as a
        # "percent of cuFFT".  The mean is over workload cells, each of which
        # has the same paired-round treatment above.
        fractions = [1.0 / item["best_over_cufft_latency"] for item in summary]
        summary.append({"aggregate": "mean_best_throughput_fraction",
                        "workload_cells": len(fractions),
                        "value": statistics.fmean(fractions),
                        "percent_of_cufft": 100.0 * statistics.fmean(fractions),
                        "target_fraction": 0.80,
                        "target_met": statistics.fmean(fractions) >= 0.80})
    return summary


def export_calibration(output):
    """Link timing trials to their separate full-workload correctness preflight."""
    preflights = json.loads((output / "verification.json").read_text())
    with (output / "raw.csv").open() as source:
        rows = list(csv.DictReader(source))
    records = []
    for preflight in preflights:
        expected = preflight["sample"]
        samples = [row for row in rows if (row["logN"], row["batch"], row["mode"]) ==
                   (expected["logN"], expected["batch"], preflight["mode"])]
        if not samples or expected["correct"] != "1":
            raise ValueError("missing timing or successful preflight")
        for sample in samples:
            for key in ("backend", "fft_core", "stages_per_decomposition", "group_threads", "group_ept",
                        "boundary_layouts", "direction", "normalization", "placement", "precision",
                        "stage_overlap", "batch_tile_count", "element_stride", "batch_stride"):
                if sample[key] != expected[key]:
                    raise ValueError(f"preflight/timing mapping mismatch: {key}")
            for key in ("mapping_json","runtime_fingerprint"):
                if key in expected and sample.get(key)!=expected[key]:
                    raise ValueError(f"preflight/timing identity mismatch: {key}")
        records.append({"name": f"stage-overlap-{expected['logN']}-{expected['batch']}-{preflight['mode']}",
                        "status": "measured", "correct": True, "trials": len(samples),
                        "median_kernel_ms": statistics.median(float(row["kernel_ms"]) for row in samples),
                        "preflight": expected, "samples": samples})
    (output / "calibration.json").write_text(json.dumps(records, indent=2) + "\n")


def export_paired_calibration(output, preflights, rows):
    """Preserve full API mappings and separately paired timing evidence."""
    records = []
    for preflight in preflights:
        sample = preflight["sample"]
        timing = [r for r in rows if (r["logN"], r["batch"], r["mode"]) ==
                  (sample["logN"], sample["batch"], preflight["mode"])]
        if not timing:
            continue
        if sample["correct"] != "1" or sample["fft_core"] != "register-tile":
            raise ValueError("paired register-tile calibration requires its successful API preflight")
        samples = [{**sample, "kernel_ms": r["kernel_ms"], "warmup": "20",
                    "repeat": str(preflight["repeat"]), "measurement_exclusive_gpu": "1"} for r in timing]
        records.append({"name": f"register-tile-{sample['logN']}-{sample['batch']}-{preflight['mode']}",
                        "status": "measured", "correct": True, "trials": len(samples),
                        "median_kernel_ms": statistics.median(float(r["kernel_ms"]) for r in timing),
                        "preflight_command": preflight["command"], "samples": samples})
    (output / "paired_calibration.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--paired-binary", type=pathlib.Path,
                        help="run the same-process benchmark after the standard correctness preflight")
    parser.add_argument("--layout-sweep", action="store_true",
                        help="paired FP32 prefix linear/XOR, thread mapping and twiddle comparison")
    parser.add_argument("--production-only", action="store_true",
                        help="confirm frozen register-tile mappings through ButterflyPlan")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--logs", type=int, nargs="+", default=[18, 20])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--tiles", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--precision", choices=("fp32","fp64"), default="fp32")
    parser.add_argument("--auto-mapping", action="store_true",
                        help="resolve the public selector once per cell and compare its bulk/stream schedules")
    parser.add_argument("--verify-batches", type=int, default=0,
                        help="CPU checks within the full GPU batch; 0 checks all transforms")
    parser.add_argument("--resume", action="store_true", help="retain verified mappings and completed timing trials")
    args = parser.parse_args()
    if min(args.logs)<3 or min(args.batches+args.tiles)<1 or args.verify_batches<0:
        parser.error("sizes, batches and tiles must be positive")
    if args.paired_binary and (args.auto_mapping or args.precision!="fp32" or args.verify_batches):
        parser.error("new mapping/precision/verification options require the public benchmark route")
    if args.layout_sweep and not args.paired_binary:
        parser.error("--layout-sweep requires --paired-binary")
    if args.production_only and (not args.paired_binary or args.layout_sweep):
        parser.error("--production-only requires --paired-binary and excludes --layout-sweep")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    if args.paired_binary:
        rows = []
        preflights = []
        metadata = {"binary_sha256": hashlib.sha256(args.paired_binary.read_bytes()).hexdigest(),
                    "verification_binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    "layout_sweep": args.layout_sweep, "trials": args.trials, "repeat": args.repeat,
                    "production_only": args.production_only,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": None, "commands": []}
        for log_n in args.logs:
            for batch in args.batches:
                require_exclusive_gpu()
                device = subprocess.run([binary, "--device-identity"], capture_output=True, text=True, check=True).stdout
                if metadata["device"] is not None and metadata["device"] != device:
                    raise RuntimeError("GPU identity changed during comparison")
                metadata["device"] = device
                if args.production_only:
                    local = log_n - 10
                    for columns in (8, 16):
                        ept = 1 << (local // 2)
                        preflight_command = [binary, "--operator", "fft", "--precision", "fp32",
                            "--backend", "online-reorder", "--fft-core", "register-tile",
                            "--logN", str(log_n), "--batch", str(batch), "--stage-partition", f"{local},10",
                            "--prefix-threads", str(ept * columns), "--prefix-ept", str(ept),
                            "--suffix-threads", "256", "--suffix-ept", "16", "--cross-twiddle", "recurrence",
                            "--shared-layout", "writer-aligned", "--normalization", "none",
                            "--placement", "out-of-place", "--warmup", "0", "--repeat", "1", "--verify", "--csv"]
                        completed = subprocess.run(preflight_command, text=True, capture_output=True, check=True)
                        sample = next(csv.DictReader(io.StringIO(completed.stdout)))
                        if sample["correct"] != "1":
                            raise ValueError("register-tile failed the existing library accuracy threshold")
                        require_exclusive_gpu()
                        preflights.append({"mode": f"plan-register-c{columns}", "command": preflight_command,
                                           "sample": sample, "repeat": args.repeat})
                    (args.output / "paired_verification.json").write_text(json.dumps(preflights, indent=2) + "\n")
                command = [str(args.paired_binary.resolve()), str(log_n), str(batch),
                           str(args.trials), str(args.repeat)]
                if args.layout_sweep:
                    command.append("--layout-sweep")
                if args.production_only:
                    command.append("--production-only")
                result = subprocess.run(command,
                                        text=True, capture_output=True, check=True)
                (args.output / f"verify-log{log_n}-batch{batch}.log").write_text(result.stderr)
                require_exclusive_gpu()
                metadata["commands"].append(command)
                rows.extend(csv.DictReader(io.StringIO(result.stdout)))
                with (args.output / "paired.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (args.output / "paired_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                (args.output / "paired_summary.json").write_text(json.dumps(summarize_paired(rows), indent=2) + "\n")
                if args.production_only:
                    export_paired_calibration(args.output, preflights, rows)
                print(f"paired logN={log_n} batch={batch} complete", flush=True)
        return
    jobs = []
    metadata={"binary_sha256":hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "compile_mode":os.environ.get("CUBUTTERFLY_COMPILE_MODE","default"),
              "device":subprocess.run([binary,"--device-identity"],capture_output=True,text=True,check=True).stdout,
              "precision":args.precision,"auto_mapping":args.auto_mapping,"verify_batches":args.verify_batches,
              "logs":args.logs,"batches":args.batches,"tiles":args.tiles,"trials":args.trials,"repeat":args.repeat}
    metadata_path=args.output/"metadata.json"
    if args.resume and metadata_path.exists() and json.loads(metadata_path.read_text())!=metadata:
        raise ValueError("resume requires the same hardware, binary, compile mode and measurement protocol")
    if args.resume and not metadata_path.exists() and (args.output/"raw.csv").exists():
        raise ValueError("legacy timing has no binary/hardware resume identity")
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    verification_flags=["--verify-batches",str(args.verify_batches)] if args.verify_batches else []
    for log_n in args.logs:
        for batch in args.batches:
            mapping = [binary, "--operator", "fft", "--precision", args.precision, "--logN", str(log_n),
                       "--batch", str(batch), "--normalization", "none", "--placement", "out-of-place", "--csv"]
            unit = ["--backend", "online-reorder", "--fft-core", "cufftdx-block",
                    "--stage-partition", f"{log_n - 12},12", "--prefix-threads", "256", "--prefix-ept", "16",
                    "--suffix-threads", "256", "--suffix-ept", "16", "--direct-boundary", "tiled-transpose"]
            if args.auto_mapping:
                require_exclusive_gpu()
                selected=subprocess.run(mapping+["--auto-select","--warmup","0","--repeat","1","--verify"]+verification_flags,
                                        capture_output=True,text=True,check=True)
                require_exclusive_gpu()
                sample=next(csv.DictReader(io.StringIO(selected.stdout)))
                if sample["correct"]!="1": raise ValueError("selected mapping failed numerical verification")
                point=json.loads(sample["mapping_json"])
                eligible=point["backend"]=="shared-iterative" or (point["backend"]=="online-reorder" and
                          point["fft_core"] in ("register-tile","cufftdx-block"))
                if not eligible or int(sample["execution_group_count"])<2:
                    raise ValueError("selected mapping has no supported multi-group streaming adapter")
                point.update(stage_overlap=False,batch_tile_count=1)
                unit=["--mapping-json",json.dumps(point,sort_keys=True)]
            jobs.append((log_n, batch, "cufft", mapping + ["--backend", "cufft"]))
            jobs.append((log_n, batch, "bulk", mapping + unit))
            for tile in args.tiles:
                if tile <= batch:
                    if args.auto_mapping:
                        streamed=dict(point,stage_overlap=True,batch_tile_count=tile)
                        jobs.append((log_n,batch,f"overlap-{tile}",mapping+["--mapping-json",json.dumps(streamed,sort_keys=True)]))
                        continue
                    jobs.append((log_n, batch, f"overlap-{tile}", mapping + unit +
                                 ["--stage-overlap", "--batch-tile-count", str(tile)]))
    rng = random.Random(127)
    verification, rows = [], []
    if args.resume:
        if (args.output / "verification.json").exists():
            verification = json.loads((args.output / "verification.json").read_text())
        if (args.output / "raw.csv").exists():
            with (args.output / "raw.csv").open() as source:
                rows = list(csv.DictReader(source))
    for log_n, batch, mode, command in jobs:
        existing = [v for v in verification if (v["sample"]["logN"], v["sample"]["batch"], v["mode"]) ==
                    (str(log_n), str(batch), mode)]
        if existing:
            if len(existing) != 1 or existing[0]["command"] != command or existing[0]["sample"]["correct"] != "1":
                raise ValueError("resume verification does not match the requested mapping")
            continue
        require_exclusive_gpu()
        result = subprocess.run(command + ["--warmup", "0", "--repeat", "1", "--verify"]+verification_flags,
                                capture_output=True, text=True, check=True)
        require_exclusive_gpu()
        sample = next(csv.DictReader(io.StringIO(result.stdout)))
        if sample["correct"] != "1":
            raise RuntimeError(f"incorrect configuration: {command}")
        verification.append({"mode": mode, "command": command, "sample": sample})
        (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(f"verified logN={log_n} batch={batch} {mode}", flush=True)
    for trial in range(args.trials):
        rng.shuffle(jobs)
        for log_n, batch, mode, command in jobs:
            if any((row["logN"], row["batch"], row["mode"], str(row["trial"])) ==
                   (str(log_n), str(batch), mode, str(trial)) for row in rows):
                continue
            require_exclusive_gpu()
            result = subprocess.run(command + ["--warmup", "20", "--repeat", str(args.repeat)],
                                    capture_output=True, text=True, check=True)
            require_exclusive_gpu()
            sample = next(csv.DictReader(io.StringIO(result.stdout)))
            expected = next(v["sample"] for v in verification if v["command"] == command)
            if sample["device"] != expected["device"]:
                raise ValueError("GPU changed since preflight")
            rows.append({"trial": trial, "mode": mode, "measurement_exclusive_gpu":"1", **sample})
            with (args.output / "raw.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = []
    for log_n in args.logs:
        for batch in args.batches:
            group = [row for row in rows if int(row["logN"]) == log_n and int(row["batch"]) == batch]
            times = {mode: statistics.median(float(row["kernel_ms"]) for row in group if row["mode"] == mode)
                     for mode in sorted({row["mode"] for row in group})}
            best = min((mode for mode in times if mode != "cufft"), key=times.get)
            summary.append({"logN": log_n, "batch": batch, "median_ms": times, "best": best,
                            "bulk_over_best": times["bulk"] / times[best],
                            "cufft_over_best": times["cufft"] / times[best]})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_calibration(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
