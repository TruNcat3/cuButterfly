#!/usr/bin/env python3
"""Bind a finite FFT core/schedule factorial using existing factor adapters.

The simple schedule still uses hierarchical factor decomposition. This control
isolates temporal data folding and prefetch, not the entire hybrid method.
Core names describe implementations and do not assume which one is faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "tools"))
from prepare_experiments import canonical, module_manifest, write


def study():
    cases = []
    # Both adapters require the same even-log factors and EPT=sqrt(factor N).
    # These are controls, not constraints on runtime candidate enumeration.
    for log_n, factors, ept in ((8, [4, 4], 4), (12, [6, 6], 8), (18, [6, 6, 6], 8)):
        for batch in (1, (1 << 22) // (1 << log_n)):
            variants = []
            comparisons = []
            for label, core in (("native", "register-tile"), ("imported", "cufftdx-block")):
                base = dict(schema_version=1, kind="butterfly", backend="factor-streamed",
                            fft_core=core, compute_unit="auto", local_exchange="shared",
                            shared_layout="writer-aligned", cross_twiddle="recurrence",
                            direct_boundary="direct-strided", stage_partition=[log_n],
                            factor_partition=factors, factor_ept=ept, factor_columns=8,
                            factor_slices=1, factor_overlap=False)
                for schedule, tiles, depth in (("simple", 1, 0), ("folded", 4, 0), ("prefetch", 4, 1)):
                    variants.append(dict(id=f"{label}-{schedule}",
                                         mapping=dict(base, data_tiles_per_cta=tiles, prefetch_depth=depth)))
                for before, after, axes in (("simple", "folded", ["data_tiles_per_cta"]),
                                            ("folded", "prefetch", ["prefetch_depth"]),
                                            ("simple", "prefetch", ["data_tiles_per_cta", "prefetch_depth"])):
                    comparisons.append(dict(baseline=f"{label}-{before}", treatment=f"{label}-{after}",
                                            changed_axes=axes))
            for schedule in ("simple", "folded", "prefetch"):
                comparisons.append(dict(baseline=f"native-{schedule}", treatment=f"imported-{schedule}",
                                        changed_axes=["fft_core"]))
            cases.append(dict(id=f"fft-fp32-n{log_n}-b{batch}", experiment="E2/E4",
                              workload=dict(operator="fft", precision="fp32", logN=log_n, batch=batch,
                                            placement="out-of-place", direction="forward", normalization="none"),
                              variants=variants, comparisons=comparisons))
    return dict(schema="cubutterfly-controlled-experiment-v1",
                protocol=dict(trials=3, warmup=100, repeat=100, verify_batches=0),
                scope="Native/imported factor core x simple/folded/prefetched schedule. "
                      "Simple retains hierarchical decomposition; this is not a no-hybrid baseline. "
                      "Common requested shapes still require compiled-resource and correctness validation.",
                cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = json.loads(args.target.read_text())
    value = study()
    path = args.output_dir / "factorial_study.json"
    if path.exists() and canonical(json.loads(path.read_text())) != canonical(value):
        raise ValueError("factorial study changed; select a new output directory")
    write(path, value)
    modules = module_manifest({"factorial": value}, target)
    write(args.output_dir / "precompile" / "modules.json", modules)
    print(json.dumps(dict(cases=len(value["cases"]), variants=sum(len(c["variants"]) for c in value["cases"]),
                          modules=modules["module_count"], gpu_queried=False)))


if __name__ == "__main__":
    main()
