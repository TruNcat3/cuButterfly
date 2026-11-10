#!/usr/bin/env python3
"""Bind the scalar-core portion of the planned E2/E3 organization ablation.

Both variants use the same generated radix-2 arithmetic core. The baseline
materializes every logical stage in HBM; the treatment retains up to six
dependent stages in a CTA. These fixed controls do not restrict search or
claim an equivalent imported-core, no-hybrid implementation exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_experiments import canonical, module_manifest, write


def study(requirements):
    cases = []
    for required in requirements["cases"]:
        workload = required["workload"]
        op, log_n, batch = (workload[key] for key in ("operator", "logN", "batch"))
        base = dict(schema_version=1, backend="shared-iterative", compute_unit="radix2")
        if op == "ntt":
            base.update(kind="ntt", threads_per_block=128, dataflow_layout="hermes-xor")
        else:
            base.update(kind="butterfly", tile_threads=128, fft_core="scalar",
                        local_exchange="shared", shared_layout="writer-aligned")
        resident = [6] * (log_n // 6) + ([log_n % 6] if log_n % 6 else [])
        cases.append(dict(
            id=f"{op}-n{log_n}-b{batch}-scalar-organization", experiment="E2/E3",
            workload=workload,
            variants=[dict(id="stagewise", mapping=dict(base, stage_partition=[1] * log_n)),
                      dict(id="resident6", mapping=dict(base, stage_partition=resident))],
            comparisons=[dict(baseline="stagewise", treatment="resident6",
                              changed_axes=["stage_partition"])]))
    return dict(schema="cubutterfly-controlled-experiment-v1",
                protocol=dict(trials=3, warmup=100, repeat=100, verify_batches=0),
                scope=__doc__.strip(), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = study(json.loads(args.requirements.read_text()))
    path = args.output_dir / "scalar_organization.json"
    if path.exists() and canonical(json.loads(path.read_text())) != canonical(value):
        raise ValueError("study changed; select a new output directory")
    write(path, value)
    modules = module_manifest({"scalar_organization": value}, json.loads(args.target.read_text()))
    write(args.output_dir / "precompile" / "modules.json", modules)
    print(json.dumps(dict(cases=len(value["cases"]), variants=2 * len(value["cases"]),
                          modules=modules["module_count"], gpu_queried=False)))


if __name__ == "__main__":
    main()
