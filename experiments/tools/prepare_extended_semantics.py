#!/usr/bin/env python3
"""Prepare the bounded precision/inverse/CT-GS extension cohort.

The command is intentionally CPU-only.  It emits explicit GPU commands for
the currently lowered CT form and records GS as reference-only until a
direction-aware lowering exists.  This prevents a design-space entry from
silently turning into an unsupported runtime claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/workloads/extended_semantics.json"


def _precision_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regimes = {item["id"]: item for item in config["precision_regimes"]}
    floating = [regimes[item["id"]] for item in config["precision_regimes"]]
    for family in config["operator_families"]:
        chosen = floating if family["precision_regimes"] == "floating" else [
            {"id": value, "precision": value, "accumulation": "native"}
            for value in family["precision_regimes"]
        ]
        for regime in chosen:
            for inverse in (False, True):
                for form in ("ct", "gs"):
                    row = {
                        "operator": family["operator"],
                        "precision_regime": regime["id"],
                        "precision": regime["precision"],
                        "accumulation": regime["accumulation"],
                        "direction": "inverse" if inverse else "forward",
                        "algorithm_form": form,
                        "logN": config["target"]["logN"],
                        "batch": config["target"]["batch"],
                        "placement": config["target"]["placement"],
                        "status": config["algorithm_forms"][form]["status"],
                        "normalization": family["inverse_normalization"] if inverse else "none",
                    }
                    if "stage_matrix" in family:
                        row["stage_matrix"] = family["stage_matrix"]
                    if family["operator"] == "ntt":
                        row["modulus"] = family["moduli"][regime["id"]]
                    if form == "ct":
                        if family["operator"] == "ntt":
                            command = [
                                "build-a100/cuntt_bench", "--backend", "hybrid2d",
                                "--compute-unit", config["target"]["compute_unit"],
                                "--stage-partition", "4,4,4", "--threads-per-block", "128",
                                "--logN", str(config["target"]["logN"]),
                                "--batch", str(config["target"]["batch"]),
                                "--word-bits", regime["precision"][4:],
                                "--modulus", str(row["modulus"]), "--verify", "--csv",
                            ]
                        else:
                            command = [
                                "build-a100/cubutterfly_bench",
                                "--operator", family["operator"],
                                "--backend", config["target"]["backend"],
                                "--precision", regime["precision"],
                                "--accumulation", regime["accumulation"],
                                "--compute-unit", config["target"]["compute_unit"],
                                "--tile-threads", str(config["target"]["tile_threads"]),
                                "--local-stages", str(config["target"]["local_stages"]),
                                "--logN", str(config["target"]["logN"]),
                                "--batch", str(config["target"]["batch"]),
                                "--placement", config["target"]["placement"],
                                "--verify", "--csv",
                            ]
                        if inverse:
                            command += ["--inverse"]
                            if family["operator"] != "ntt":
                                # Structured operators define their inverse by
                                # coefficient inversion and do not expose an
                                # FFT-style normalization flag.
                                normalization = row["normalization"]
                                if normalization not in ("none", "inverse"):
                                    normalization = "none"
                                command += ["--normalization", normalization]
                        if family.get("stage_matrix"):
                            command += ["--stage-matrix", family["stage_matrix"]]
                        row["command"] = command
                    rows.append(row)
    return rows


def prepare(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = _precision_rows(config)
    report = {
        "schema": "cubutterfly-extended-semantics-prepared-v1",
        "config": str(config_path.resolve()),
        "gpu_commands": [row for row in rows if row["status"] == "gpu-lowered"],
        "reference_only": [row for row in rows if row["status"] != "gpu-lowered"],
        "counts": {
            "total": len(rows),
            "gpu_commands": sum(row["status"] == "gpu-lowered" for row in rows),
            "reference_only": sum(row["status"] != "gpu-lowered" for row in rows),
        },
        "policy": config["external_baseline_policy"],
        "measurement_policy": config["measurement_policy"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.config.resolve(), args.output.resolve())
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
