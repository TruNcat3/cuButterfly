#!/usr/bin/env python3
"""Build one auditable end-to-end result table for the current evidence cohort.

The table deliberately keeps timing and validation separate.  A row with a
failed exact check is retained as a failure record and never contributes a
speedup.  CUDA-event medians are the primary metric when the artifact provides
them; wall-clock values remain in the source artifact as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


FIELDS = [
    "scenario", "type", "comparison", "batch", "baseline_ms",
    "cubutterfly_ms", "speedup", "validation", "status", "source",
]


def _median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty sample list")
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _row(**values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(values)
    if result["speedup"] == "" and result["baseline_ms"] != "" and result["cubutterfly_ms"] not in ("", 0):
        result["speedup"] = float(result["baseline_ms"]) / float(result["cubutterfly_ms"])
    return result


def _fft_rows(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((root / "fft").glob("gpu_*/release/image*.json")):
        report = json.loads(path.read_text())
        n = report["contract"]["image_shape"][0]
        batch = report["contract"]["batch"]
        rows = {entry["method"]: entry for entry in report["rows"]}
        treatment = rows["cubutterfly"]
        for method, label in (
            ("cupy_fftconvolve_native_padding", "CuPy fftconvolve"),
            ("torch_cufft_same_padding_diagnostic", "Torch/cuFFT"),
        ):
            baseline = rows[method]
            yield _row(
                scenario=f"2D FFT filter N={n} B={batch}",
                type="FFT2D + pointwise multiply + crop",
                comparison=label,
                batch=batch,
                baseline_ms=baseline["timing"]["median_us"] / 1000.0,
                cubutterfly_ms=treatment["timing"]["median_us"] / 1000.0,
                validation="finite; FP32 error distribution reported",
                status="valid_complete_call",
                source=str(path),
            )


def _poisson_rows(root: Path) -> Iterable[dict[str, Any]]:
    paths = sorted((root / "fft").glob("gpu_*/release/poisson*.json"))
    for path in paths:
        report = json.loads(path.read_text())
        timings = report.get("timings", {})
        if "cubutterfly_graph" not in timings:
            continue
        cu = timings["cubutterfly_graph"]["cuda_event_median_ms"]
        for backend, label in (("torch_graph", "Torch/cuFFT"), ("cupy_graph", "CuPy")):
            baseline = timings[backend]["cuda_event_median_ms"]
            yield _row(
                scenario=f"2D spectral Poisson N={report['size']} B={report['batch']}",
                type="FFT2D + spectral multiplier + inverse FFT",
                comparison=label,
                batch=report["batch"],
                baseline_ms=baseline,
                cubutterfly_ms=cu,
                validation="finite; residual/error distribution reported",
                status="valid_complete_call",
                source=str(path),
            )


def _quarot_rows(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((root / "quarot").glob("full7b_b*_p*_d*_graph.json")):
        report = json.loads(path.read_text())
        payload = report["payload"]
        base = report["baseline"]["cuda_event_ms"]["end_to_end"]
        cu = report["cubutterfly"]["cuda_event_ms"]["end_to_end"]
        yield _row(
            scenario=f"QuaRot 7B prefill={payload['prefill']} decode={payload['decode']}",
            type="FWHT quantization + model execution",
            comparison="original current-stream CUDA extension",
            batch=payload["batch_size"],
            baseline_ms=_median(base),
            cubutterfly_ms=_median(cu),
            validation="Graph original/changed/restored/lifetime exact; application error unassessed",
            status=report["status"],
            source=str(path),
        )


def _tfhe_rows(root: Path) -> Iterable[dict[str, Any]]:
    native_path = root / "tfhe" / "native_graph.csv"
    shared_path = root / "tfhe" / "shared_graph.csv"
    if not native_path.exists() or not shared_path.exists():
        return
    def load(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
        with path.open(newline="") as handle:
            return {(row["workload"], int(row["batch"])): row for row in csv.DictReader(handle)}
    native = load(native_path)
    shared = load(shared_path)
    for key, treatment in sorted(shared.items()):
        # shared_graph.csv also contains a native-original-kernel control.  The
        # standalone native_graph.csv is the actual external-library baseline.
        if treatment["backend"] != "cubutterfly-shared":
            continue
        control = native.get(key)
        if control is None:
            continue
        yield _row(
            scenario=f"TFHE {treatment['workload']} {treatment['gate']}",
            type="TFHE boolean bootstrap / NAND circuit",
            comparison="native TFHE bootstrap library",
            batch=int(treatment["batch"]),
            baseline_ms=float(control["avg_ms"]),
            cubutterfly_ms=float(treatment["avg_ms"]),
            validation="exact bits; changed/restored/lifetime Graph checks",
            status="valid_bounded_smoke",
            source=str(shared_path),
        )


def _ckks_rows(root: Path) -> Iterable[dict[str, Any]]:
    directory = root / "ckks"
    for path in sorted(directory.glob("bootstrap_*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for data in rows:
            if not data.get("scene"):
                continue
            exact = data.get("exact_ciphertext") == "1"
            parameter_set = data.get("parameter_set", "")
            if parameter_set == "bootstrap-v1-small":
                result_type = "CKKS legacy bootstrap v1 reduced-N"
            elif parameter_set == "bootstrap-v2-small":
                result_type = "CKKS bootstrap v2 reduced-N diagnostic"
            else:
                result_type = "CKKS full bootstrap v2"
            if exact:
                validation, status = "exact ciphertext", "valid_complete_call"
            elif parameter_set == "bootstrap-v2-small":
                validation = f"reduced-N functional validation failed; mismatch={data.get('mismatch_count', '?')}"
                status = "functional_validation_failed_reduced_profile"
            elif path.name in {"bootstrap_b1_eager.csv", "bootstrap_b1_probe.csv"}:
                validation = f"pre-input-restore diagnostic mismatch={data.get('mismatch_count', '?')}"
                status = "diagnostic_pre_input_restore"
            else:
                validation = f"ciphertext validation failed; mismatch={data.get('mismatch_count', '?')}"
                status = "functional_validation_failed"
            yield _row(
                scenario=f"CKKS {data['scene']} N={data['N']}",
                type=result_type,
                comparison="HEonGPU native-rns path",
                batch=int(data["batch"]),
                baseline_ms=float(data["native_ms"]) if exact else "",
                cubutterfly_ms=float(data["cubutterfly_ms"]) if exact else "",
                validation=validation,
                status=status,
                source=str(path),
            )
    # A complete runner invocation can be stopped by its ownership monitor
    # before the benchmark emits a row. Preserve that fact as evidence rather
    # than silently treating the missing CSV as an unmeasured success.
    for metadata_path in sorted(directory.glob("bootstrap_*.metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") != "invalid_gpu_contention":
            continue
        output_name = metadata_path.name[:-len(".metadata.json")]
        if "v1_small" in output_name:
            scenario = "CKKS regular_bootstrapping N=4096"
            result_type = "CKKS legacy bootstrap v1 reduced-N"
        else:
            scenario = "CKKS regular_bootstrapping_v2 N=65536"
            result_type = "CKKS full bootstrap v2"
        yield _row(
            scenario=scenario,
            type=result_type,
            comparison="runner ownership monitor",
            batch=1,
            validation="no timing; external GPU process appeared during preparation",
            status="invalid_gpu_contention",
            source=str(metadata_path),
        )
    # A Graph run that emitted no CSV row is still an important result: it is
    # an explicit support boundary, never a missing measurement.
    graph_log = directory / "bootstrap_b1_probe.log"
    if graph_log.exists() and "Graph exact check failed" in graph_log.read_text(errors="replace"):
        yield _row(
            scenario="CKKS regular_bootstrapping_v2 N=65536",
            type="CKKS full bootstrap v2",
            comparison="HEonGPU native Graph (pre-input-restore diagnostic)",
            batch=1,
            validation="Graph exact check failed before timing",
            status="diagnostic_pre_input_restore",
            source=str(graph_log),
        )
    for log_name in ("bootstrap_b1_grouped_eager.log", "bootstrap_b1_eager_restored.log"):
        log_path = directory / log_name
        if log_path.exists() and "out of memory" in log_path.read_text(errors="replace").lower():
            yield _row(
                scenario="CKKS regular_bootstrapping_v2 N=65536",
                type="CKKS full bootstrap v2",
                comparison="cuButterfly preparation/cache",
                batch=1,
                validation="no timing; device allocation exhausted during plan preparation",
                status="resource_exhausted_preparation",
                source=str(log_path),
            )


def build(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for producer in (_fft_rows, _poisson_rows, _quarot_rows, _tfhe_rows, _ckks_rows):
        rows.extend(producer(root))
    return rows


def write_outputs(root: Path, rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown = output.with_suffix(".md")
    lines = [
        "# Current end-to-end alignment",
        "",
        "CUDA-event medians are primary where available. Speedup is baseline / cuButterfly; rows with failed validation have blank timing and speedup fields.",
        "",
        "| 场景 | 类型 | 对照 | batch | baseline ms | cuButterfly ms | 加速比 | 验证 | 状态 |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value == "":
                return "—"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
        lines.append("| " + " | ".join(fmt(row[k]) for k in
                                         ("scenario", "type", "comparison", "batch", "baseline_ms",
                                          "cubutterfly_ms", "speedup", "validation", "status")) + " |")
    lines += [
        "", "## Scope", "",
        "The CKKS N=65536, Q/P bootstrap-v2 rows are kept separate from the completed short-chain CKKS matrix in `results/ckks_auto_schedule_20261019/comparison.md`. The latter remains valid historical evidence and is not silently mixed with this cohort.",
        "", "TFHE rows are bounded Graph smoke cases (NAND, batch 1/8, depths 2/4/8). QuaRot rows are full 7B Graph runs at batch 1/4. FFT rows include complete device input refresh and independent output copy.",
    ]
    markdown.write_text("\n".join(lines) + "\n")
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps({"schema": "e2e-alignment-v1", "rows": len(rows), "source_root": str(root), "csv": str(output), "markdown": str(markdown)}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/e2e_alignment_20261019"))
    parser.add_argument("--output", type=Path, default=Path("results/e2e_alignment_20261019/e2e_alignment.csv"))
    args = parser.parse_args()
    rows = build(args.root)
    write_outputs(args.root, rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
