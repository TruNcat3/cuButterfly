"""Measured empty-CTA scheduling curves, separate from operator service costs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics

VERSION = "direct-launch-cta-curves-v1"


def summarize(raw):
    if raw.get("schema") != "cubutterfly-schedule-profile-v1":
        raise ValueError("unsupported schedule microbenchmark schema")
    grouped = {}
    for row in raw["rows"]:
        values = row["trial_kernel_ms"]
        if not values or any(not math.isfinite(float(x)) or float(x) <= 0 for x in values):
            raise ValueError("schedule measurements must be finite and positive")
        if int(row["grid_ctas"]) <= 0 or int(row["threads"]) <= 0:
            raise ValueError("schedule launch dimensions must be positive")
        grouped.setdefault(int(row["threads"]), []).append((int(row["grid_ctas"]), statistics.median(values)))
    shapes = []
    for threads, rows in sorted(grouped.items()):
        if len({grid for grid, _ in rows}) != len(rows) or len(rows) < 2:
            raise ValueError("schedule curve needs distinct grid sizes")
        if not any(grid == 1 for grid, _ in rows):
            raise ValueError("schedule curve needs a one-CTA launch")
        startup = next(value for grid, value in rows if grid == 1)
        slope = max(0.0, sum((grid - 1) * (value - startup) for grid, value in rows) /
                    sum((grid - 1)**2 for grid, _ in rows))
        errors = [abs(startup + (grid - 1) * slope - value) / value for grid, value in rows]
        shapes.append(dict(threads=threads, startup_ms=startup, dispatch_ms_per_cta=slope,
            dispatch_identified=slope > 0, measured_grid_range=[min(g for g,_ in rows), max(g for g,_ in rows)],
            fit_max_relative_error=max(errors)))
    if not shapes:
        raise ValueError("no schedule measurements")
    return dict(version=VERSION, hardware=raw["hardware"], protocol=raw["protocol"], shapes=shapes,
        source="measured direct launches of minimal CTA kernel",
        limitations=["One-CTA latency includes minimal device work and host submission gaps.",
            "Dispatch slopes are minimal-kernel estimates, not butterfly core throughput.",
            "Large grids extrapolate the measured range; resident resources remain core-specific."])


def costs(profile, threads, grid, sm_count):
    scheduling = profile.get("scheduling", {})
    shapes = scheduling.get("shapes", [])
    if not shapes:
        return dict(startup_ms=.003, dispatch_ms=grid / (sm_count * 1e5), source="unmeasured-prior")
    shape = min(shapes, key=lambda s:abs(math.log2(max(1, threads) / s["threads"])))
    identified = shape["dispatch_identified"]
    return dict(startup_ms=shape["startup_ms"],
        dispatch_ms=max(0, grid - 1) * shape["dispatch_ms_per_cta"] if identified else grid / (sm_count * 1e5),
        source="measured-minimal-CTA" if identified else "measured-startup-with-unidentified-dispatch",
        thread_shape=shape["threads"], thread_shape_matched=threads == shape["threads"],
        grid_extrapolated=not shape["measured_grid_range"][0] <= grid <= shape["measured_grid_range"][1])


def calibrate(binary, path, profile, run, exclusive):
    """Reuse compatible capability measurements; never reuse another GPU's curve."""
    identity = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        device=profile["device"], compute_capability=profile["compute_capability"],
        global_memory_bytes=int(profile["global_memory_bytes"]),
        visible_device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), version=VERSION,
        protocol=dict(warmup=10, repeat=100, trials=3))
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("identity") == identity:
            return cached["summary"]
    exclusive()
    raw = json.loads(run([str(binary), "--warmup", "10", "--repeat", "100", "--trials", "3"]).stdout)
    exclusive()
    for field in ("device", "compute_capability", "global_memory_bytes"):
        if str(raw["hardware"].get(field)) != str(identity[field]):
            raise ValueError(f"scheduling probe identity mismatch: {field}")
    summary = summarize(raw)
    artifact = dict(identity=identity, raw=raw, summary=summary, measurement_exclusive_gpu=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n")
    temporary.replace(path)
    return summary
