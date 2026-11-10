#!/usr/bin/env python3
"""Freeze portable semantic campaign manifests without GPU or timing data.

The generated campaign is intentionally separate from the running result
directories.  It contains workload semantics, mapping candidates and a small
set of commands that existing benchmark binaries can verify.  It never copies
hardware identities, absolute paths, timing samples or selector history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_SCHEMA = "cubutterfly-install-search-v1"

SEMANTIC_FIELDS = (
    "accumulation", "batch", "batch_stride", "direction", "element_stride",
    "input_order", "logN", "modulus", "normalization", "operator",
    "output_order", "placement", "precision", "stage_matrix",
)
COMPOSED_FIELDS = (
    "application", "batch", "composition", "coset_generator", "direction",
    "logN", "mode", "moduli", "modulus_bits", "rns_channel_count",
    "rns_memberships", "rns_stream_order", "suites", "word_bits",
)
TIMING_KEYS = {
    "kernel_ms", "h2d_ms", "d2h_ms", "total_ms", "warmup", "repeat",
    "trials", "samples", "measurements", "timing", "history", "ranking",
    "median_kernel_ms", "throughput", "transforms_s", "points_s",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _rows(document: dict[str, Any], field: str = "workloads") -> list[dict[str, Any]]:
    rows = document.get(field)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"input does not contain a non-empty object list: {field}")
    return rows


def _semantic_row(row: dict[str, Any], fields: Iterable[str] = SEMANTIC_FIELDS) -> dict[str, Any]:
    return {key: row[key] for key in fields if key in row}


def _semantic_key(row: dict[str, Any]) -> str:
    """Use the registry's semantic defaults without importing runtime code."""
    value = dict(row)
    value.setdefault("operator", "fft")
    value.setdefault("precision", "word64" if value["operator"] == "ntt" else "fp32")
    value.setdefault("batch", 1)
    value.setdefault("direction", "forward")
    value.setdefault("normalization", "none")
    value.setdefault("placement", "out-of-place")
    value.setdefault("element_stride", 1)
    value.setdefault("logN", 0)
    value.setdefault("batch_stride", ((1 << int(value["logN"])) - 1) * int(value["element_stride"]) + 1
                         if int(value["logN"]) else 0)
    value.setdefault("stage_matrix", "")
    value.setdefault("modulus", "0")
    return _canonical(_semantic_row(value))


def _assert_portable(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            # Protocol knobs describe a future run; they are not historical
            # samples. Timing fields anywhere in workload/result records are
            # still rejected.
            config_path = ".protocol" in path or ".verification" in path
            if str(key).lower() in TIMING_KEYS and not config_path:
                raise ValueError(f"timing/history key leaked into portable output: {path}.{key}")
            _assert_portable(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_portable(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if "/home/" in value or value.startswith("/root/") or "\\home\\" in value:
            raise ValueError(f"absolute host path leaked into portable output: {path}")


def _write_json(path: Path, value: Any) -> tuple[str, int]:
    _assert_portable(value)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256_text(text), len(text.encode("utf-8"))


def _shape_text(shape: int | tuple[int, int]) -> str:
    return str(shape) if isinstance(shape, int) else f"{shape[0]}x{shape[1]}"


def _plan_case(case_id: str, shape: int | tuple[int, int], precision: str, batch: int) -> dict[str, Any]:
    shape_text = _shape_text(shape)
    args = [
        "--operator", "fft", "--shape", shape_text, "--batch", str(batch),
        "--precision", precision, "--length-mode", "standard", "--policy", "measure",
        "--compare-cufft", "--verify", "--trials", "3", "--warmup", "20", "--repeat", "100",
    ]
    rank = 2 if isinstance(shape, tuple) else 1
    return {
        "id": case_id,
        "operator": "fft",
        "precision": precision,
        "shape": shape_text,
        "rank": rank,
        "batch": batch,
        "length_mode": "standard",
        "direction": "forward",
        "placement": "out-of-place",
        "normalization": "none",
        "verification": {"required": True, "verify_batches": 0, "baseline": "cuFFT"},
        "args": args,
    }


def _deferred_case(case_id: str, operator: str, shape: int | tuple[int, int], precision: str,
                   batch: int, *, embedding: bool = False, inverse: bool = False,
                   matrix: str | None = None) -> dict[str, Any]:
    shape_text = _shape_text(shape)
    args = ["--operator", operator, "--shape", shape_text, "--batch", str(batch),
            "--precision", precision, "--length-mode", "embedding" if embedding else "standard"]
    if inverse:
        args.append("--inverse")
    if matrix is not None:
        args.extend(("--matrix", matrix))
    return {
        "id": case_id,
        "status": "deferred-harness",
        "operator": operator,
        "precision": precision,
        "shape": shape_text,
        "rank": 2 if isinstance(shape, tuple) else 1,
        "batch": batch,
        "length_mode": "embedding" if embedding else "standard",
        "direction": "inverse" if inverse else "forward",
        "args": args,
        "reason": "cubutterfly_plan_bench has no complete --verify contract for this semantic family",
    }


def _plan_workloads() -> dict[str, Any]:
    # These are the only plan-bench rows promoted to executable P0 points:
    # the source verifies every batch and compares the same forward C2C FFT
    # contract with cuFFT. No stride or matrix CLI is included here.
    points = [
        _plan_case("fft-r1-n256-fp32-b1", 256, "fp32", 1),
        _plan_case("fft-r1-n4096-fp32-b64", 4096, "fp32", 64),
        _plan_case("fft-r1-n65536-fp32-b16", 65536, "fp32", 16),
        _plan_case("fft-r1-n262144-fp32-b8", 262144, "fp32", 8),
        _plan_case("fft-r1-n1048576-fp32-b4", 1048576, "fp32", 4),
        _plan_case("fft-r1-n1000-fp32-b64", 1000, "fp32", 64),
        _plan_case("fft-r1-n4096-fp64-b16", 4096, "fp64", 16),
        _plan_case("fft-r1-n1000-fp64-b16", 1000, "fp64", 16),
        _plan_case("fft-r2-2048x2048-fp32-b1", (2048, 2048), "fp32", 1),
        _plan_case("fft-r2-4096x4096-fp32-b1", (4096, 4096), "fp32", 1),
        _plan_case("fft-r2-30x50-fp32-b64", (30, 50), "fp32", 64),
        _plan_case("fft-r2-64x256-fp32-b16", (64, 256), "fp32", 16),
        _plan_case("fft-r2-128x512-fp32-b32", (128, 512), "fp32", 32),
        _plan_case("fft-r2-512x128-fp32-b32", (512, 128), "fp32", 32),
        _plan_case("fft-r2-2048x2048-fp64-b1", (2048, 2048), "fp64", 1),
        _plan_case("fft-r2-30x50-fp64-b8", (30, 50), "fp64", 8),
    ]
    matrix = "1,0.25,-0.5,1"
    deferred = [
        _deferred_case("fwht-r2-64x256-fp32-fwd-b16", "fwht", (64, 256), "fp32", 16),
        _deferred_case("fwht-r2-64x256-fp32-inv-b16", "fwht", (64, 256), "fp32", 16, inverse=True),
        _deferred_case("zeta-subset-r2-64x256-u32-inv-b16", "subset-zeta", (64, 256), "uint32", 16, inverse=True),
        _deferred_case("zeta-superset-r2-64x256-u32-inv-b16", "superset-zeta", (64, 256), "uint32", 16, inverse=True),
        _deferred_case("structured-r2-64x256-fp32-inv-b16", "structured-2x2", (64, 256), "fp32", 16, inverse=True, matrix=matrix),
        _deferred_case("fwht-embed-n255-fp32-fwd-b256", "fwht", 255, "fp32", 256, embedding=True),
        _deferred_case("fwht-embed-n255-fp32-inv-b256", "fwht", 255, "fp32", 256, embedding=True, inverse=True),
        _deferred_case("zeta-subset-embed-n255-u32-inv-b256", "subset-zeta", 255, "uint32", 256, embedding=True, inverse=True),
        _deferred_case("zeta-superset-embed-n255-u32-inv-b256", "superset-zeta", 255, "uint32", 256, embedding=True, inverse=True),
        _deferred_case("structured-embed-n255-fp32-inv-b256", "structured-2x2", 255, "fp32", 256, embedding=True, inverse=True, matrix=matrix),
        _deferred_case("fft-r1-n4096-fp32-inv-b16", "fft", 4096, "fp32", 16, inverse=True),
    ]
    return {
        "schema": "cubutterfly-plan-campaign-v1",
        "scope": "finite verified FFT plan-bench points plus deferred non-FFT harness contracts",
        "runner": "cubutterfly_plan_bench",
        "verification_contract": {
            "verified_rows": "forward standard out-of-place FFT only",
            "flags": ["--verify", "--compare-cufft", "--trials", "3", "--warmup", "20", "--repeat", "100"],
            "verify_batches": 0,
            "compile_and_plan_time_in_timing": False,
        },
        "workloads": points,
        "deferred": deferred,
    }


def _operator_extensions(existing: set[str]) -> dict[str, Any]:
    matrix = "1,0.25,-0.5,1"
    candidates: list[dict[str, Any]] = []

    def add(operator: str, precision: str, log_n: int, batch: int, *, direction: str = "forward",
            placement: str = "out-of-place", element_stride: int = 1,
            batch_padding: int = 0, stage_matrix: str | None = None) -> None:
        row: dict[str, Any] = {
            "operator": operator, "precision": precision, "logN": log_n, "batch": batch,
            "placement": placement, "direction": direction,
            "normalization": "inverse" if operator == "fwht" and direction == "inverse" else "none",
        }
        if element_stride != 1:
            row["element_stride"] = element_stride
            row["batch_stride"] = ((1 << log_n) - 1) * element_stride + 1 + batch_padding
        if stage_matrix is not None:
            row["stage_matrix"] = stage_matrix
        if _semantic_key(row) not in existing and _semantic_key(row) not in {_semantic_key(x) for x in candidates}:
            candidates.append(row)

    # Inverse anchors and matching forward controls for the two integer zeta
    # directions, normalized FWHT, and fixed nonsingular structured matrices.
    for operator in ("fwht", "subset-zeta", "superset-zeta"):
        precision = "fp32" if operator == "fwht" else "uint32"
        add(operator, precision, 12, 64, direction="forward")
        add(operator, precision, 12, 64, direction="inverse")
        add(operator, precision, 15 if operator == "fwht" else 16, 128 if operator == "fwht" else 64,
            direction="inverse")
        add(operator, precision, 12, 16, direction="inverse", placement="in-place",
            element_stride=2, batch_padding=7)
    add("fwht", "fp64", 12, 64, direction="inverse")
    add("structured-2x2", "fp32", 12, 64, direction="forward", stage_matrix=matrix)
    add("structured-2x2", "fp32", 12, 64, direction="inverse", stage_matrix=matrix)
    add("structured-2x2", "fp32", 15, 128, direction="inverse", stage_matrix=matrix)
    add("structured-2x2", "fp32", 12, 16, direction="inverse", placement="in-place",
        element_stride=2, batch_padding=7, stage_matrix=matrix)
    add("structured-2x2", "fp64", 12, 64, direction="inverse", stage_matrix=matrix)

    return {
        "schema": WORKLOAD_SCHEMA,
        "scope": "bounded non-cryptographic operator extension; cubutterfly_bench full-batch verification",
        "runner": "cubutterfly_bench",
        "protocol": {
            "warmup": 20,
            "repeat": 100,
            "verify": True,
            "verify_batches": 0,
            "compile_and_plan_time_in_timing": False,
        },
        "workloads": candidates,
        "limitations": [
            "No unsupported stride or per-stage matrix CLI is emitted.",
            "Structured rows use the supported broadcast matrix contract.",
            "The finite extension is not a complete application matrix or global optimum claim.",
        ],
    }


def prepare(*, core: Path, batch_precision: Path, crypto: Path, composed: Path,
            stage_points: Path, mapping_seeds: Path, output_dir: Path) -> dict[str, Any]:
    core_doc = _load(core)
    batch_doc = _load(batch_precision)
    crypto_doc = _load(crypto)
    composed_doc = _load(composed)
    stage_doc = _load(stage_points)
    seeds_doc = _load(mapping_seeds)

    for name, document in (("core", core_doc), ("batch_precision", batch_doc), ("crypto", crypto_doc)):
        if document.get("schema") != WORKLOAD_SCHEMA:
            raise ValueError(f"{name} uses an unexpected workload schema")
    if composed_doc.get("schema") != "cubutterfly-composed-workload-v1":
        raise ValueError("composed workload schema is not supported")
    if stage_doc.get("schema") not in (None, "cubutterfly-stage-points-v1"):
        raise ValueError("unexpected stage-point schema")
    if seeds_doc.get("schema") != "cubutterfly-mapping-seeds-v1":
        raise ValueError("unexpected mapping-seed schema")

    core_rows = [_semantic_row(row) for row in _rows(core_doc)]
    batch_rows = [_semantic_row(row) for row in _rows(batch_doc)]
    crypto_rows = [_semantic_row(row) for row in _rows(crypto_doc)]
    composed_rows = [{key: row[key] for key in COMPOSED_FIELDS if key in row}
                     for row in _rows(composed_doc)]

    stage_rows = []
    for row in _rows(stage_doc, "points"):
        mapping = json.loads(row["mapping_json"]) if "mapping_json" in row else row.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("backend") != "shared-iterative":
            continue
        stage_rows.append({**_semantic_row(row), "mapping": mapping})
    if len(stage_rows) != 28:
        raise ValueError(f"expected 28 shared-iterative stage points, found {len(stage_rows)}")

    seed_rows = []
    for row in _rows(seeds_doc, "seeds"):
        if not isinstance(row.get("mapping"), dict) or not isinstance(row.get("semantics"), dict):
            raise ValueError("mapping seed must contain mapping and semantics")
        seed_rows.append({"mapping": row["mapping"], "semantics": _semantic_row(row["semantics"])})
    if len(seed_rows) != 79:
        raise ValueError(f"expected 79 mapping seeds, found {len(seed_rows)}")

    existing = {_semantic_key(row) for row in core_rows + batch_rows + crypto_rows}
    operator_doc = _operator_extensions(existing)
    plan_doc = _plan_workloads()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents: dict[str, Any] = {
        "core.json": {
            "schema": WORKLOAD_SCHEMA,
            "scope": "portable semantic snapshot of the original comprehensive workload cohort",
            "kind": "core",
            "workloads": core_rows,
        },
        "batch_precision.json": {
            "schema": WORKLOAD_SCHEMA,
            "scope": "portable semantic snapshot of the batch and precision extension",
            "kind": "batch-precision",
            "workloads": batch_rows,
        },
        "crypto.json": {
            "schema": WORKLOAD_SCHEMA,
            "scope": "portable semantic snapshot of cyclic cryptographic NTT workloads",
            "kind": "crypto",
            "workloads": crypto_rows,
        },
        "composed.json": {
            "schema": "cubutterfly-composed-workload-v1",
            "scope": "portable semantic snapshot of composed crypto contracts",
            "kind": "composed",
            "workloads": composed_rows,
        },
        "stage_points.json": {
            "schema": "cubutterfly-stage-points-v1",
            "scope": "28 shared-iterative semantic calibration points; no timing evidence",
            "points": stage_rows,
        },
        "mapping_seeds.json": {
            "schema": "cubutterfly-mapping-seeds-v1",
            "scope": "mapping-only candidates; remeasure on the target hardware",
            "seeds": seed_rows,
        },
        "plan_workloads.json": plan_doc,
        "operator_extensions.json": operator_doc,
    }

    files: dict[str, Any] = {}
    for name, document in documents.items():
        digest, byte_count = _write_json(output_dir / name, document)
        files[name] = {"sha256": digest, "bytes": byte_count}

    counts = {
        "core": len(core_rows),
        "batch_precision": len(batch_rows),
        "crypto": len(crypto_rows),
        "composed": len(composed_rows),
        "stage_points": len(stage_rows),
        "mapping_seeds": len(seed_rows),
        "plan_workloads": len(plan_doc["workloads"]),
        "plan_deferred": len(plan_doc["deferred"]),
        "operator_extensions": len(operator_doc["workloads"]),
    }
    manifest_body = {
        "schema": "cubutterfly-research-campaign-v1",
        "portable": True,
        "source_inputs": {
            "core": core.name,
            "batch_precision": batch_precision.name,
            "crypto": crypto.name,
            "composed": composed.name,
            "stage_points": stage_points.name,
            "mapping_seeds": mapping_seeds.name,
        },
        "counts": counts,
        "files": files,
        "deferred_plan_harness": {
            "count": len(plan_doc["deferred"]),
            "reason": "plan_bench --verify is complete only for forward standard out-of-place FFT",
        },
    }
    manifest_digest = _sha256_text(_canonical(manifest_body))
    manifest = {**manifest_body, "manifest_sha256": manifest_digest}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _default_paths() -> dict[str, Path]:
    return {
        "core": ROOT / "results/comprehensive_20260919/workloads.json",
        "batch_precision": ROOT / "results/batch_precision_20260920/workloads.json",
        "crypto": ROOT / "results/crypto_application_20260920/workloads.json",
        "composed": ROOT / "results/crypto_application_20260920/composed_workloads.json",
        "stage_points": ROOT / "results/comprehensive_20260919/stage_points.json",
        "mapping_seeds": ROOT / "results/comprehensive_20260919/mapping_seeds.json",
        "output_dir": ROOT / "config/research_campaign",
    }


def main(argv: list[str] | None = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "batch_precision", "crypto", "composed", "stage_points", "mapping_seeds"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=defaults[name])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    args = parser.parse_args(argv)
    manifest = prepare(
        core=args.core, batch_precision=args.batch_precision, crypto=args.crypto,
        composed=args.composed, stage_points=args.stage_points,
        mapping_seeds=args.mapping_seeds, output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "counts": manifest["counts"],
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
