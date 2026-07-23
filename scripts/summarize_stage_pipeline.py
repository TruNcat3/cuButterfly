#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description="Summarize the controlled NTT256 stage-space experiment.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("stage-pipeline input is empty")

    groups = {}
    for row in rows:
        key = (row["backend"], row["stage_handoff"], int(row["stage_space"]))
        groups.setdefault(key, []).append(row)
    tile_ms = statistics.median(float(row["kernel_ms"]) for row in groups[("tile256", "atomic", 0)])
    us1_ms = {
        handoff: statistics.median(float(row["kernel_ms"]) for row in groups[("stage-pipeline", handoff, 1)])
        for handoff in ("atomic", "named-barrier")
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend", "stage_handoff", "stage_space", "data_space_warps", "stage_time_folds", "trials", "median_kernel_ms",
        "median_Gbutterfly_s", "speedup_vs_us1", "speedup_vs_tile256",
    ]
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        order = [("tile256", "atomic", 0)]
        order += [("stage-pipeline", handoff, stage_space) for handoff in ("atomic", "named-barrier") for stage_space in (1, 2, 4, 8)]
        for backend, handoff, stage_space in order:
            samples = groups[(backend, handoff, stage_space)]
            median_ms = statistics.median(float(row["kernel_ms"]) for row in samples)
            batch = int(samples[0]["batch"])
            butterflies = batch * 128 * 8
            writer.writerow({
                "backend": backend,
                "stage_handoff": handoff,
                "stage_space": stage_space,
                "data_space_warps": 0 if stage_space == 0 else 8 // stage_space,
                "stage_time_folds": 0 if stage_space == 0 else 8 // stage_space,
                "trials": len(samples),
                "median_kernel_ms": f"{median_ms:.6f}",
                "median_Gbutterfly_s": f"{butterflies / (median_ms * 1.0e6):.6f}",
                "speedup_vs_us1": f"{(tile_ms if backend == 'tile256' else us1_ms[handoff]) / median_ms:.6f}",
                "speedup_vs_tile256": f"{tile_ms / median_ms:.6f}",
            })


if __name__ == "__main__":
    main()
