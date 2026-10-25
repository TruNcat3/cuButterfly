#!/usr/bin/env python3
"""CPU-only range diagnosis for the native FP16 hierarchical DIT sequence."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def component_statistics(real, imag):
    return {name: {"nan": int(np.isnan(value).sum()),
                   "inf": int(np.isinf(value).sum()),
                   "nonfinite": int((~np.isfinite(value)).sum())}
            for name, value in (("real", real), ("imag", imag))}


def simulate_native_dit(source):
    """Separate half operations; matches this DIT sequence, not all mappings."""
    source = np.asarray(source).reshape(-1)
    n = source.size
    if n < 2 or n & (n - 1):
        raise ValueError("source length must be a power of two >= 2")
    log_n = n.bit_length() - 1
    index = np.arange(n, dtype=np.uint64)
    reverse = np.zeros(n, dtype=np.uint64)
    for _ in range(log_n):
        reverse = (reverse << 1) | (index & 1)
        index >>= 1
    real = source.real.astype(np.float16)[reverse]
    imag = source.imag.astype(np.float16)[reverse]
    stages = []
    with np.errstate(over="ignore", invalid="ignore"):
        for stage in range(log_n):
            half = 1 << stage
            angle = -np.pi * np.arange(half, dtype=np.float64) / half
            wr = np.cos(angle).astype(np.float32).astype(np.float16)
            wi = np.sin(angle).astype(np.float32).astype(np.float16)
            r = real.reshape(-1, 2 * half)
            i = imag.reshape(-1, 2 * half)
            ar, ai = r[:, :half].copy(), i[:, :half].copy()
            br, bi = r[:, half:].copy(), i[:, half:].copy()
            products = (br * wr, bi * wi, br * wi, bi * wr)
            vr, vi = products[0] - products[1], products[2] + products[3]
            outputs = (ar + vr, ai + vi, ar - vr, ai - vi)
            r[:, :half], i[:, :half] = outputs[:2]
            r[:, half:], i[:, half:] = outputs[2:]
            stages.append({
                "stage_zero_based": stage,
                "nonfinite_complex": int((~(np.isfinite(real) & np.isfinite(imag))).sum()),
                "components": component_statistics(real, imag),
                "nonfinite_real_products": [int((~np.isfinite(p)).sum()) for p in products],
                "nonfinite_weighted_components": [int((~np.isfinite(v)).sum()) for v in (vr, vi)],
                "nonfinite_butterfly_components": [int((~np.isfinite(v)).sum()) for v in outputs],
            })
    return real, imag, stages


def diagnose(row_path, error_path=None):
    row_path = Path(row_path).resolve()
    envelope = json.loads(row_path.read_text())
    row = envelope.get("row", envelope)
    semantics = row["descriptor"]["semantics"]
    mapping = row["mapping"]
    if not (semantics["operator"] == 0 and semantics["storage"] == 4
            and semantics["compute"] == 0 and semantics["inverse"] == 0
            and semantics["batch"] == 1 and semantics["fft_type"] == 0
            and semantics["fft_output_scale"] == 1
            and len(semantics["shape"]) == 1
            and semantics["input_orders"] == [0] and semantics["output_orders"] == [0]
            and mapping["backend"] == "hierarchical"
            and mapping["compute_unit"] == "radix4"
            and mapping["complex_multiply"] == "four-mul"):
        raise ValueError("requires forward native FP16 1D B=1 natural C2C hierarchical-radix4 scale=1")
    # Importing the runner is CPU-only; do not call its execution entry points.
    runner_path = Path(__file__).with_name("run_semantic_campaign.py")
    spec = importlib.util.spec_from_file_location("fp16_range_campaign_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    source = runner.make_source(semantics)
    real, imag, stages = simulate_native_dit(source)
    artifact = Path(error_path or row["error_artifact"]["path"])
    with np.load(artifact, allow_pickle=False) as arrays:
        actual = arrays["actual"].reshape(-1)
        reference = arrays["reference"].reshape(-1)
    if actual.size != real.size or reference.size != real.size:
        raise ValueError("artifact size does not match source")
    simulated_indices = np.flatnonzero(~(np.isfinite(real) & np.isfinite(imag)))
    actual_indices = np.flatnonzero(~np.isfinite(actual))
    return {
        "schema": "cubutterfly-fp16-dit-range-diagnostic-v1",
        "row_id": row["id"], "row_path": str(row_path), "error_path": str(artifact),
        "source": "unchanged runner.make_source quantized input; no scaling",
        "limitation": "CPU half DIT sequence only; not all mappings or bitwise finite-error equivalence. Actual component NaNs may include decoder effects.",
        "stages": stages,
        "first_nonfinite_stage_zero_based": next((s["stage_zero_based"] for s in stages if s["nonfinite_complex"]), None),
        "simulated_nonfinite_indices": simulated_indices.tolist(),
        "actual_nonfinite_indices": actual_indices.tolist(),
        "nonfinite_indices_exact_match": bool(np.array_equal(simulated_indices, actual_indices)),
        "simulated_components": component_statistics(real, imag),
        "actual_components": component_statistics(actual.real, actual.imag),
        "reference_components_beyond_max_finite_half": {
            "real": int((np.abs(reference.real) > 65504).sum()),
            "imag": int((np.abs(reference.imag) > 65504).sum()),
            "complex_union": int(((np.abs(reference.real) > 65504) | (np.abs(reference.imag) > 65504)).sum()),
        },
        "range_count_note": "65504 is max finite half, not the RNE overflow threshold 65520; reference is the artifact CPU oracle.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, type=Path)
    parser.add_argument("--error-npz", type=Path, help="Override row error artifact location")
    parser.add_argument("--output", type=Path, help="Write JSON here; default is stdout")
    args = parser.parse_args()
    report = json.dumps(diagnose(args.row, args.error_npz), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
