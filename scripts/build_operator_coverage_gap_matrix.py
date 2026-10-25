#!/usr/bin/env python3
"""Generate the finite operator-coverage gap matrix from contract section 7.

This generator is intentionally CPU-only.  It emits a machine-readable
checklist for the next same-build acceptance run; it never loads a shared
library, probes a device, or invokes a benchmark.  Rows marked ``candidate``
are missing current-build evidence, while ``packed-only`` and ``unsupported``
rows preserve source-level boundaries instead of pretending they passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "operator-coverage-gap-v1"
CONTRACT_PATH = "docs/operator_coverage_contract_20261006.md"
CONTRACT_SECTION = "7"
RUN_MODE = "cpu-dry-run"

LOW_PRECISIONS = ("fp16", "bf16")
ACCUMULATIONS = ("native", "fp32")
FLOAT_OPERATORS = ("fft", "fwht", "structured-2x2")
REAL_SHAPES = (
    ("rank1-odd", (5,)),
    ("rank1-even", (6,)),
    ("rank2", (4, 6)),
    ("rank3", (2, 3, 5)),
)
RANK_SHAPES = ((2, (4, 8)), (3, (4, 4, 4)))
NTT_MODULI = {
    32: ("2013265921", "998244353"),
    64: ("2013265921", "998244353", "576460756061519873"),
}

SCALAR_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "word32": 4,
    "word64": 8,
    "uint32": 4,
    "uint64": 8,
}

EXPECTED_SECTION_COUNTS = {
    "low_precision_public_api": 48,
    "real_public_api": 16,
    "ntt_semantic_anchors": 21,
    "rank_layout": 12,
    "layout_boundaries": 3,
    "large_n_batch": 9,
    "unsupported_real_low_precision": 4,
}


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _packed_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * int(shape[index + 1])
    return tuple(strides)


def _span_elements(shape: tuple[int, ...], strides: tuple[int, ...],
                   batch_stride: int, batch: int) -> int:
    extent = 1 + sum((int(size) - 1) * int(stride)
                     for size, stride in zip(shape, strides))
    return (int(batch) - 1) * int(batch_stride) + extent


def _scalar_precision(precision: str) -> str:
    if precision.startswith("word"):
        return precision
    return precision


def _side_element_bytes(operator: str, precision: str, storage: str,
                        fft_type: str, side: str) -> int:
    scalar = SCALAR_BYTES[_scalar_precision(precision)]
    if operator != "fft":
        return scalar
    if fft_type == "c2c":
        return scalar * 2 if storage.startswith("complex_") else scalar
    if fft_type == "r2c":
        return scalar if side == "input" else scalar * 2
    if fft_type == "c2r":
        return scalar * 2 if side == "input" else scalar
    raise ValueError(f"unknown FFT type: {fft_type}")


def _common_row(*, section: str, row_id: str, operator: str,
                precision: str, storage: str, compute: str,
                accumulation: str, shape: tuple[int, ...], batch: int,
                direction: str, placement: str,
                input_strides: tuple[int, ...],
                output_strides: tuple[int, ...],
                input_batch_stride: int, output_batch_stride: int,
                fft_type: str = "c2c", input_shape: tuple[int, ...] | None = None,
                output_shape: tuple[int, ...] | None = None,
                normalization: str | None = None,
                coverage_state: str = "missing-public-evidence",
                expected_execution: str = "candidate",
                source_contract: str = "") -> dict[str, Any]:
    input_shape = tuple(input_shape or shape)
    output_shape = tuple(output_shape or shape)
    input_bytes = (_span_elements(input_shape, input_strides,
                                  input_batch_stride, batch) *
                   _side_element_bytes(operator, precision, storage,
                                       fft_type, "input"))
    output_bytes = (_span_elements(output_shape, output_strides,
                                   output_batch_stride, batch) *
                    _side_element_bytes(operator, precision, storage,
                                        fft_type, "output"))
    row: dict[str, Any] = {
        "id": row_id,
        "section": section,
        "run_mode": RUN_MODE,
        "operator": operator,
        "precision": precision,
        "storage": storage,
        "compute": compute,
        "accumulation": accumulation,
        "rank": len(shape),
        "shape": list(shape),
        "logical_shape": list(shape),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "batch": int(batch),
        "direction": direction,
        "normalization": normalization or ("inverse" if direction == "inverse" else "none"),
        "placement": placement,
        "input_strides": list(input_strides),
        "output_strides": list(output_strides),
        "input_batch_stride": int(input_batch_stride),
        "output_batch_stride": int(output_batch_stride),
        "length_mode": "standard",
        "fft_type": fft_type,
        "coverage_state": coverage_state,
        "expected_execution": expected_execution,
        "source_contract": source_contract,
        "input_bytes": int(input_bytes),
        "output_bytes": int(output_bytes),
    }
    if operator in FLOAT_OPERATORS or operator == "fft":
        row["execution_mode"] = "auto"
    return row


def _float_storage(operator: str, precision: str) -> str:
    return f"complex_{precision}" if operator == "fft" else precision


def _low_precision_rows() -> list[dict[str, Any]]:
    rows = []
    shapes = (("rank1", (8,), 3), ("rank2", (4, 8), 2))
    for operator in FLOAT_OPERATORS:
        for precision in LOW_PRECISIONS:
            for accumulation in ACCUMULATIONS:
                for shape_name, shape, batch in shapes:
                    strides = _packed_strides(shape)
                    for direction in ("forward", "inverse"):
                        row_id = (f"low-{operator}-{precision}-{accumulation}-"
                                  f"{shape_name}-b{batch}-{direction}")
                        row = _common_row(
                            section="low_precision_public_api",
                            row_id=row_id,
                            operator=operator,
                            precision=precision,
                            storage=_float_storage(operator, precision),
                            compute=accumulation,
                            accumulation=accumulation,
                            shape=shape,
                            batch=batch,
                            direction=direction,
                            placement="out-of-place",
                            input_strides=strides,
                            output_strides=strides,
                            input_batch_stride=_product(shape),
                            output_batch_stride=_product(shape),
                            source_contract=(
                                "low-precision public C-ABI evidence is absent; "
                                "connect this row to the same-build runner"),
                        )
                        if operator == "structured-2x2":
                            row["stage_matrix"] = "1,0.25,-0.5,1"
                        rows.append(row)
    return rows


def _real_rows() -> list[dict[str, Any]]:
    rows = []
    for precision in ("fp32", "fp64"):
        for shape_name, shape in REAL_SHAPES:
            half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
            for fft_type, direction in (("r2c", "forward"), ("c2r", "inverse")):
                input_shape = shape if fft_type == "r2c" else half_shape
                output_shape = half_shape if fft_type == "r2c" else shape
                input_strides = _packed_strides(input_shape)
                output_strides = _packed_strides(output_shape)
                row = _common_row(
                    section="real_public_api",
                    row_id=f"real-{fft_type}-{precision}-{shape_name}-b2",
                    operator="fft",
                    precision=precision,
                    storage=precision,
                    compute=precision,
                    accumulation="native",
                    shape=shape,
                    batch=2,
                    direction=direction,
                    placement="out-of-place",
                    input_strides=input_strides,
                    output_strides=output_strides,
                    input_batch_stride=_product(input_shape),
                    output_batch_stride=_product(output_shape),
                    fft_type=fft_type,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    normalization="inverse" if fft_type == "c2r" else "none",
                    source_contract=(
                        "compact real public API requires standard packed out-of-place "
                        "FP32/FP64 storage"),
                )
                row["layout"] = "packed"
                row["output_scale"] = 1.0
                rows.append(row)
    return rows


def _unsupported_real_low_precision_rows() -> list[dict[str, Any]]:
    rows = []
    shape = (4, 6)
    half_shape = (4, 4)
    for precision in LOW_PRECISIONS:
        for fft_type, direction in (("r2c", "forward"), ("c2r", "inverse")):
            input_shape = shape if fft_type == "r2c" else half_shape
            output_shape = half_shape if fft_type == "r2c" else shape
            input_strides = _packed_strides(input_shape)
            output_strides = _packed_strides(output_shape)
            row = _common_row(
                section="unsupported_real_low_precision",
                row_id=f"real-{fft_type}-{precision}-rank2-4x6-b2-unsupported",
                operator="fft",
                precision=precision,
                storage=precision,
                compute="native",
                accumulation="native",
                shape=shape,
                batch=2,
                direction=direction,
                placement="out-of-place",
                input_strides=input_strides,
                output_strides=output_strides,
                input_batch_stride=_product(input_shape),
                output_batch_stride=_product(output_shape),
                fft_type=fft_type,
                input_shape=input_shape,
                output_shape=output_shape,
                coverage_state="unsupported",
                expected_execution="unsupported",
                source_contract=(
                    "compact R2C/C2R currently requires standard FP32/FP64; "
                    "FP16/BF16 must remain an explicit unsupported row"),
            )
            row["layout"] = "packed"
            row["output_scale"] = 1.0
            rows.append(row)
    return rows


def _ntt_row(*, section: str, row_id: str, word_bits: int, log_n: int,
              batch: int, modulus: str, direction: str,
              output_order: str = "natural",
              coverage_state: str = "missing-semantic-anchor",
              expected_execution: str = "candidate",
              source_contract: str = "") -> dict[str, Any]:
    precision = f"word{word_bits}"
    shape = (1 << int(log_n),)
    strides = _packed_strides(shape)
    row = _common_row(
        section=section,
        row_id=row_id,
        operator="ntt",
        precision=precision,
        storage=f"uint{word_bits}",
        compute=f"uint{word_bits}",
        accumulation="modular",
        shape=shape,
        batch=batch,
        direction=direction,
        placement="out-of-place",
        input_strides=strides,
        output_strides=strides,
        input_batch_stride=shape[0],
        output_batch_stride=shape[0],
        normalization="inverse_n" if direction == "inverse" else "none",
        coverage_state=coverage_state,
        expected_execution=expected_execution,
        source_contract=source_contract,
    )
    row.update({
        "logN": int(log_n),
        "N": shape[0],
        "modulus": str(modulus),
        "word_bits": int(word_bits),
        "input_order": "natural",
        "output_order": output_order,
        "layout": "packed",
    })
    return row


def _ntt_rows() -> list[dict[str, Any]]:
    rows = []
    for word_bits in (32, 64):
        for log_n in (12, 16):
            for modulus in NTT_MODULI[word_bits]:
                for direction in ("forward", "inverse"):
                    rows.append(_ntt_row(
                        section="ntt_semantic_anchors",
                        row_id=(f"ntt-word{word_bits}-log{log_n}-p{modulus}-"
                                f"{direction}-natural"),
                        word_bits=word_bits,
                        log_n=log_n,
                        batch=1,
                        modulus=modulus,
                        direction=direction,
                        source_contract=(
                            "finite modulus/direction anchor; not an exhaustive "
                            "prime or root compatibility sweep"),
                    ))
    rows.append(_ntt_row(
        section="ntt_semantic_anchors",
        row_id="ntt-word32-log12-p2013265921-forward-bit-reversed",
        word_bits=32,
        log_n=12,
        batch=1,
        modulus="2013265921",
        direction="forward",
        output_order="bit-reversed",
        source_contract=(
            "legal bit-reversed output anchor: rank-1 power-of-two NTT with "
            "natural input"),
    ))
    return rows


def _rank_layout_rows() -> list[dict[str, Any]]:
    rows = []
    for operator in FLOAT_OPERATORS:
        for rank, shape in RANK_SHAPES:
            packed = _packed_strides(shape)
            if rank == 2:
                strided = (10, 1)
                strided_batch = 43
            else:
                strided = (24, 6, 1)
                strided_batch = 101
            for layout, placement, strides, batch_stride in (
                ("positive-strided-oop", "out-of-place", strided, strided_batch),
                ("packed-in-place", "in-place", packed, _product(shape)),
            ):
                row = _common_row(
                    section="rank_layout",
                    row_id=(f"layout-{operator}-rank{rank}-{_shape_text(shape)}-"
                            f"{layout}"),
                    operator=operator,
                    precision="fp32",
                    storage=_float_storage(operator, "fp32"),
                    compute="fp32",
                    accumulation="native",
                    shape=shape,
                    batch=2,
                    direction="forward",
                    placement=placement,
                    input_strides=strides,
                    output_strides=strides,
                    input_batch_stride=batch_stride,
                    output_batch_stride=batch_stride,
                    source_contract=(
                        "rank/layout row is source-legal where the public operator "
                        "accepts this placement; current matrix has no matching row"),
                )
                row["layout"] = layout
                row["execution_mode"] = "auto"
                if operator == "structured-2x2":
                    row["stage_matrix"] = "1,0.25,-0.5,1"
                rows.append(row)
    return rows


def _boundary_rows() -> list[dict[str, Any]]:
    rows = []
    for operator in ("subset-zeta", "superset-zeta"):
        shape = (8,)
        strides = _packed_strides(shape)
        row = _common_row(
            section="layout_boundaries",
            row_id=f"boundary-{operator}-rank1-packed-only",
            operator=operator,
            precision="uint32",
            storage="uint32",
            compute="uint32",
            accumulation="modular",
            shape=shape,
            batch=2,
            direction="forward",
            placement="out-of-place",
            input_strides=strides,
            output_strides=strides,
            input_batch_stride=shape[0],
            output_batch_stride=shape[0],
            coverage_state="boundary",
            expected_execution="packed-only",
            source_contract=(
                "zeta rank/layout boundary is recorded as packed-only; no "
                "general stride or layout Cartesian product is implied"),
        )
        row["layout"] = "packed"
        rows.append(row)
    rows.append(_ntt_row(
        section="layout_boundaries",
        row_id="boundary-ntt-rank1-packed-only",
        word_bits=64,
        log_n=12,
        batch=2,
        modulus="576460756061519873",
        direction="forward",
        coverage_state="boundary",
        expected_execution="packed-only",
        source_contract=(
            "rank-1 direct NTT requires packed contiguous batches; strided, "
            "in-place, and embedding variants are not silently counted"),
    ))
    return rows


def _large_rows() -> list[dict[str, Any]]:
    rows = []
    for operator, precision, log_n, modulus in (
        ("fft", "fp32", 23, None),
        ("fft", "fp64", 20, None),
        ("ntt", "word64", 20, "576460756061519873"),
    ):
        for batch in (1, 16, 256):
            shape = (1 << log_n,)
            strides = _packed_strides(shape)
            if operator == "ntt":
                row = _ntt_row(
                    section="large_n_batch",
                    row_id=f"large-ntt-word64-log{log_n}-b{batch}",
                    word_bits=64,
                    log_n=log_n,
                    batch=batch,
                    modulus=str(modulus),
                    direction="forward",
                    source_contract=(
                        "large N/B correctness anchor; current-build execution "
                        "is still pending"),
                )
            else:
                row = _common_row(
                    section="large_n_batch",
                    row_id=f"large-{operator}-{precision}-log{log_n}-b{batch}",
                    operator=operator,
                    precision=precision,
                    storage=_float_storage(operator, precision),
                    compute=precision,
                    accumulation="native",
                    shape=shape,
                    batch=batch,
                    direction="forward",
                    placement="out-of-place",
                    input_strides=strides,
                    output_strides=strides,
                    input_batch_stride=shape[0],
                    output_batch_stride=shape[0],
                    source_contract=(
                        "large N/B correctness anchor; current-build execution "
                        "is still pending"),
                )
            row.update({"logN": int(log_n), "N": shape[0], "layout": "packed"})
            if modulus is not None:
                row["modulus"] = str(modulus)
            rows.append(row)
    return rows


def _all_rows() -> list[dict[str, Any]]:
    rows = []
    rows.extend(_low_precision_rows())
    rows.extend(_real_rows())
    rows.extend(_ntt_rows())
    rows.extend(_rank_layout_rows())
    rows.extend(_boundary_rows())
    rows.extend(_large_rows())
    rows.extend(_unsupported_real_low_precision_rows())
    return rows


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_section = Counter(row["section"] for row in rows)
    by_state = Counter(row["coverage_state"] for row in rows)
    by_expected = Counter(row["expected_execution"] for row in rows)
    return {
        "rows": len(rows),
        "by_section": dict(sorted(by_section.items())),
        "by_coverage_state": dict(sorted(by_state.items())),
        "by_expected_execution": dict(sorted(by_expected.items())),
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("coverage gap matrix contains duplicate row ids")
    section_counts = Counter(row["section"] for row in rows)
    if dict(section_counts) != EXPECTED_SECTION_COUNTS:
        raise ValueError(
            f"unexpected section counts: {dict(sorted(section_counts.items()))}")

    required = {
        "id", "section", "run_mode", "operator", "precision", "storage",
        "compute", "accumulation", "rank", "shape", "logical_shape",
        "input_shape", "output_shape", "batch", "direction", "normalization",
        "placement", "input_strides", "output_strides", "input_batch_stride",
        "output_batch_stride", "length_mode", "fft_type", "coverage_state",
        "expected_execution", "source_contract", "input_bytes", "output_bytes",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"{row.get('id', '<unknown>')} missing {sorted(missing)}")
        if row["run_mode"] != RUN_MODE:
            raise ValueError(f"{row['id']} is not a CPU dry-run row")
        shape = tuple(int(value) for value in row["shape"])
        if row["rank"] != len(shape) or not shape or any(value <= 0 for value in shape):
            raise ValueError(f"{row['id']} has an invalid rank/shape")
        if row["batch"] <= 0:
            raise ValueError(f"{row['id']} has a non-positive batch")
        for field in ("input_strides", "output_strides"):
            strides = tuple(int(value) for value in row[field])
            if len(strides) != row["rank"] or any(value <= 0 for value in strides):
                raise ValueError(f"{row['id']} has invalid {field}")
        if row["coverage_state"] == "unsupported" and row["expected_execution"] != "unsupported":
            raise ValueError(f"{row['id']} unsupported state has wrong expectation")
        if row["coverage_state"] == "boundary" and row["expected_execution"] != "packed-only":
            raise ValueError(f"{row['id']} boundary state has wrong expectation")
        if row["operator"] == "ntt":
            word_bits = int(row["word_bits"])
            modulus = int(row["modulus"])
            if word_bits not in NTT_MODULI or str(modulus) not in NTT_MODULI[word_bits]:
                raise ValueError(f"{row['id']} has an incompatible NTT modulus")
            if row["input_order"] != "natural":
                raise ValueError(f"{row['id']} has a non-natural NTT input anchor")
            if row["output_order"] == "bit-reversed":
                if row["rank"] != 1 or row["direction"] != "forward":
                    raise ValueError(f"{row['id']} has an invalid bit-reversed anchor")
            elif row["output_order"] != "natural":
                raise ValueError(f"{row['id']} has an unknown NTT output order")
        if row["fft_type"] == "r2c" and row["direction"] != "forward":
            raise ValueError(f"{row['id']} has an invalid R2C direction")
        if row["fft_type"] == "c2r" and row["direction"] != "inverse":
            raise ValueError(f"{row['id']} has an invalid C2R direction")
        if row["input_bytes"] <= 0 or row["output_bytes"] <= 0:
            raise ValueError(f"{row['id']} has invalid byte accounting")


def build_manifest() -> dict[str, Any]:
    rows = _all_rows()
    validate_rows(rows)
    counts = _count_rows(rows)
    return {
        "schema": SCHEMA,
        "contract": {"path": CONTRACT_PATH, "section": CONTRACT_SECTION},
        "scope": "finite minimum next matrix; not Cartesian and not an acceptance pass",
        "execution": {
            "mode": RUN_MODE,
            "gpu_execution": False,
            "benchmark_invocations": 0,
            "library_loads": 0,
            "device_queries": 0,
        },
        "counts": counts,
        "dimensions": {
            "low_precision": {
                "precisions": list(LOW_PRECISIONS),
                "accumulations": list(ACCUMULATIONS),
                "operators": list(FLOAT_OPERATORS),
                "shapes": [{"rank": 1, "shape": [8], "batch": 3},
                           {"rank": 2, "shape": [4, 8], "batch": 2}],
            },
            "real": {
                "precisions": ["fp32", "fp64"],
                "fft_types": ["r2c", "c2r"],
                "shapes": [{"name": name, "shape": list(shape)}
                           for name, shape in REAL_SHAPES],
                "batch": 2,
                "placement": "out-of-place",
                "layout": "packed",
            },
            "ntt": {
                "word_bits": [32, 64],
                "logN": [12, 16],
                "moduli": {str(bits): list(moduli) for bits, moduli in NTT_MODULI.items()},
                "orders": ["natural->natural", "natural->bit-reversed"],
            },
            "rank_layout": {
                "operators": list(FLOAT_OPERATORS),
                "shapes": [{"rank": rank, "shape": list(shape)}
                           for rank, shape in RANK_SHAPES],
                "layouts": ["positive-strided-oop", "packed-in-place"],
            },
            "large_n_batch": {
                "anchors": [
                    {"operator": "fft", "precision": "fp32", "logN": 23},
                    {"operator": "fft", "precision": "fp64", "logN": 20},
                    {"operator": "ntt", "precision": "word64", "logN": 20},
                ],
                "batches": [1, 16, 256],
            },
        },
        "rows": rows,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="directory in which manifest.json is written")
    parser.add_argument("--dry-run", action="store_true",
                        help="required explicit marker; this generator never executes GPU work")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("this generator is CPU-only; pass --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_manifest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "dry-run", **manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
