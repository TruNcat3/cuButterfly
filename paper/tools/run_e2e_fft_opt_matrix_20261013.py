#!/usr/bin/env python3
"""Finite same-GPU application regression matrix for grouped axes / C2R scale."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02"
NEW_SHA = "89a038905a45e2d2ab072c142f9bde17ed97b42ed86106bdac656e5ec0f0f33b"
OLD_SHA = "ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--until", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/e2e_fft_opt_20261013")
    parser.add_argument("--library", type=Path, default=ROOT / "build-e2e-scale-20261013/libcuntt.so")
    parser.add_argument("--library-sha256", default=NEW_SHA)
    parser.add_argument("--skip-old-control", action="store_true",
                        help="retain existing immutable old-library measurements without rerunning")
    args = parser.parse_args()
    new = args.library.resolve()
    old = ROOT / "build-e2e-20261013/libcuntt.so"
    for library, expected in ((new, args.library_sha256), (old, OLD_SHA)):
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
            raise RuntimeError("library identity changed: " + str(library))
    jobs = []
    # Previously weak largest request first; every predeclared Poisson point
    # remains in the matrix regardless of measured speedup.
    for size, batch in ((2048, 16), (2048, 4), (1024, 16), (1024, 4)):
        for trial in (0, 1):
            jobs.append((f"poisson_{size}_b{batch}_t{trial}", "periodic_poisson", size,
                         batch, trial, new, args.library_sha256))
    for trial in (0, 1):
        jobs.append((f"registration_2048_b16_t{trial}", "phase_correlation", 2048,
                     16, trial, new, args.library_sha256))
    if not args.skip_old_control:
        jobs.append(("poisson_2048_b16_old_t0", "periodic_poisson", 2048, 16, 0, old, OLD_SHA))
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               CUBUTTERFLY_COMPILE_MODE="research", CUBUTTERFLY_XD_TUNE_MAX_BYTES="8589934592",
               CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT),
               CUBUTTERFLY_MATHDX_ROOT=str(ROOT / "external/mathdx/nvidia/mathdx"))
    # The isolated attempt owns its cache, keeping old/new artifacts separate.
    env.pop("CUBUTTERFLY_PLAN_CACHE", None)
    for name, application, size, batch, trial, library, expected in jobs:
        report_dir = args.output.resolve() / name
        release = report_dir / "release" / (application + ".json")
        if release.exists():
            record = json.loads(release.read_text())
            if record["library_sha256"] != expected or record["gpu_uuid"].removeprefix("GPU-") != GPU.removeprefix("GPU-"):
                raise RuntimeError("incompatible completed report: " + str(release))
            print(json.dumps(dict(job=name, state="reuse-completed")), flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts/run_isolated_gpu.py"),
                   "--gpu-uuid", GPU, "--report-dir", str(report_dir),
                   "--until", args.until, "--idle-seconds", "3", "--", sys.executable,
                   str(ROOT / f"paper/tools/run_{application}_graph_20261013.py"),
                   "--library", str(library), "--size", str(size), "--batch", str(batch),
                   "--calls", "20", "--warmup", "3", "--process-trial-index", str(trial),
                   "--output", application + ".json"]
        if trial == 0:
            command.append("--profile")
        print(json.dumps(dict(job=name, state="launch" if args.run else "dry-run", argv=command)), flush=True)
        if args.run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
