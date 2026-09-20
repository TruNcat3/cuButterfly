#!/usr/bin/env python3
"""Fit a small target-local cost model from correctness-checked candidates.

The model is deliberately an auditable ranking aid, not a replacement for
steady-state measurements.  Complete external libraries such as cuFFT are
kept as reference rows and excluded from the cuButterfly candidate fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from typing import Any

from hardware_registry import SEMANTICS, canonical_semantics

FEATURE_VERSION = "physical-execution-groups-v2"
VALIDATION_IDENTITY_VERSION = "resolved-design-identity-v1"
SWEEP_IDENTITY_FIELDS = (
    "operator", "precision", "placement", "logN", "batch", "backend", "compute_unit",
    "complex_multiply", "cross_twiddle", "local_exchange", "shared_layout", "fft_core",
    "stage_space", "tile_threads", "prefix_threads", "suffix_threads", "prefix_ept",
    "suffix_ept", "local_stages", "reorder_columns", "warp_stages", "pipeline_warps",
)

FEATURE_NAMES = (
    "intercept",
    "work_million",
    "bytes_million",
    "decomposition_count",
    "boundary_count",
    "tile_threads_norm",
    "local_stages_norm",
    "boundary_threads_norm",
    "operator_non_fft",
    "backend_hierarchical",
    "backend_online_reorder",
    "backend_temporal_tile",
    "backend_warp_hybrid",
    "backend_stage_pipeline",
    "fft_core_scalar",
    "fft_core_cta_dft8",
    "fft_core_cufftdx_block",
    "fft_core_cufftdx_direct",
    "fft_core_cufftdx_resident",
    "fft_core_thread_dft8",
    "fft_core_turbofft_generated",
    "compute_unit_radix2",
    "compute_unit_radix4",
    "compute_unit_radix8",
    "cross_twiddle_recurrence",
    "local_exchange_warp_register",
    "shared_layout_xor_swizzle",
    "prefix_threads_norm",
    "suffix_threads_norm",
    "prefix_ept_norm",
    "suffix_ept_norm",
    "physical_execution_groups",
    "stage_overlap",
    "batch_tile_fraction",
    "launch_count",
    "boundary_event_count",
    "ring_buffer_mib",
    "schedule_tiles",
    "shared_layout_writer_aligned", "boundary_transpose", "boundary_prefix_transpose",
    "complex_gauss3", "inverse", "normalized_inverse", "in_place", "element_stride_log2",
    "batch_stride_ratio", "fft_core_register_tile", "backend_shared_iterative",
    "prefix_register_tile", "suffix_cufftdx", "precision_fp64", "precision_low", "accumulation_fp32",
    "max_group_stages", "min_group_stages", "max_group_live_kib", "sum_group_live_kib",
    "max_group_data_time", "min_group_ctas_log2",
    "operator_ntt", "operator_fwht", "operator_zeta", "operator_structured", "modulus_bits",
    "backend_factor_streamed", "cta_data_tiles", "prefetch_depth",
    "compiler_local_allocation_kib", "compiler_local_known_fraction", "max_compiler_registers",
    # Factor-streamed fields keep per-launch coverage, total work, and
    # dependency overhead separate.  They are zero for other backends.
    "factor_slices", "factor_overlap", "factor_total_launches",
    "factor_partial_ready_groups", "factor_boundary_workspace_mib",
    "factor_event_waits", "factor_total_ctas", "factor_max_grid_ctas_per_launch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a target-local cuButterfly candidate cost model.")
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--operator-calibration", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-csv", type=pathlib.Path, action="append", default=[],
                        help="verified sweep CSV; may be repeated")
    parser.add_argument("--allow-nonexclusive-csv", action="store_true",
                        help="accept sweep CSVs without the exclusive-GPU evidence marker")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    return parser.parse_args()


def _number(sample: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(sample.get(key, default) or default)
    except (OverflowError, TypeError, ValueError):
        return default


def is_external(row: dict[str, Any], sample: dict[str, Any]) -> bool:
    name = str(row.get("name", ""))
    backend = str(sample.get("backend", ""))
    return backend == "cufft" or name.endswith("-cufft")


def feature_vector(sample: dict[str, Any]) -> list[float]:
    log_n = max(_number(sample, "logN"), 1.0)
    n = max(_number(sample, "N", 2 ** log_n), 1.0)
    batch = max(_number(sample, "batch"), 1.0)
    precision = str(sample.get("precision", ""))
    # A complex FFT value contains two scalars. Real FWHT/structured values
    # must not inherit the FFT width, nor may FP64 be charged as FP32.
    word_bytes = {"fp16": 2.0, "bf16": 2.0, "fp32": 4.0, "fp64": 8.0,
                  "uint32": 4.0, "fp16-fp32": 4.0}.get(precision, 4.0)
    if sample.get("operator", "fft") == "fft":
        word_bytes *= 2.0
    if precision == "word64":
        word_bytes = 8.0
    elif precision == "word32":
        word_bytes = 4.0
    physical=sample.get("execution_groups_json", "[]")
    physical=json.loads(physical) if isinstance(physical,str) else physical
    decomposition = max(_number(sample, "decomposition_count", _number(sample,"logical_subgraphs",len(physical))), 1.0)
    # The legacy CSV count is the number of explicit config group mappings,
    # and may be zero even when the resolved plan has many physical groups.
    # Use the physical plan for both measured and projected samples. Otherwise
    # training charges one group's traffic while candidate scoring charges all
    # groups, corrupting traffic, launch, event and ring-buffer features.
    groups = float(len(physical)) if physical else max(
        _number(sample, "execution_group_count", _number(sample, "execution_groups", decomposition)), 1.0)
    overlap = bool(_number(sample,"stage_overlap"))
    tile = max(1.0,_number(sample,"batch_tile_count",1.0))
    tiles = math.ceil(batch/tile) if overlap else 1
    tile_threads = _number(sample, "tile_threads", _number(sample,"threads_per_block"))
    prefix_threads = _number(sample, "prefix_threads")
    suffix_threads = _number(sample, "suffix_threads")
    backend = str(sample.get("backend", ""))
    fft_core = str(sample.get("fft_core", ""))
    compute_unit = str(sample.get("compute_unit", ""))
    stage_counts=[float(g.get("stage_count",0)) for g in physical] or [log_n]
    live=[float(g.get("live_shared_bytes",0))/1024 for g in physical] or [0]
    mapping=sample.get("mapping_json", {})
    mapping=json.loads(mapping) if isinstance(mapping,str) and mapping else mapping
    mapping=mapping if isinstance(mapping,dict) else {}
    factor_backend = backend == "factor-streamed"

    def _group_number(group: dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(group.get(key, default) or default)
        except (OverflowError, TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def _flag(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    raw_launch_counts = [max(1.0, _group_number(group, "launch_count", 1.0)) for group in physical]
    explicit_launch_counts = [
        _group_number(group, "launch_count", 1.0)
        for group in physical if "launch_count" in group
    ]
    group_grid_ctas = [max(0.0, _group_number(group, "grid_ctas", 0.0)) for group in physical]
    factor_group_count = len(physical) if physical else int(groups)
    if factor_backend:
        mapping_slices = max(1.0, _number(mapping, "factor_slices", 1.0))
        fallback_slices = max([1.0, mapping_slices] + explicit_launch_counts[:-1])
        factor_slices = max(1.0, _number(sample, "factor_slices", fallback_slices))
        factor_overlap = _flag(sample.get("factor_overlap", mapping.get("factor_overlap", False)))
        has_partial_ready_fields = bool(physical) and all(
            "partial_dependency_ready" in group for group in physical
        )
        factor_partial_ready_groups = (
            sum(_flag(group.get("partial_dependency_ready", False)) for group in physical)
            if has_partial_ready_fields else max(factor_group_count - 2, 0)
        ) if factor_overlap else 0
        if raw_launch_counts and len(explicit_launch_counts) == len(physical):
            factor_group_launches = raw_launch_counts
        else:
            factor_group_launches = [
                factor_slices if index + 1 < factor_group_count else 1.0
                for index in range(factor_group_count)
            ]
        factor_total_launches = sum(factor_group_launches)
        factor_total_ctas = sum(
            grid * launches for grid, launches in zip(group_grid_ctas, factor_group_launches)
        )
        factor_max_grid_ctas = max(group_grid_ctas or [0.0])
        reported_workspace_bytes = _number(sample, "workspace_bytes", 0.0)
        if not math.isfinite(reported_workspace_bytes) or reported_workspace_bytes <= 0.0:
            element_stride = max(1.0, _number(
                sample, "element_stride", _number(mapping, "element_stride", 1.0)
            ))
            transform_extent = (n - 1.0) * element_stride + 1.0
            batch_stride = _number(
                sample, "batch_stride", _number(mapping, "batch_stride", transform_extent)
            )
            if batch_stride <= 0.0:
                batch_stride = transform_extent
            data_elements = (batch - 1.0) * batch_stride + transform_extent
            # Runtime uses ping-pong scratch for one-slice factor execution;
            # concurrent slices retain one full array per dependency edge.
            workspace_buffers = (
                max(factor_group_count - 1, 0)
                if factor_slices > 1.0 else min(2, max(factor_group_count - 1, 0))
            )
            reported_workspace_bytes = workspace_buffers * data_elements * word_bytes
        factor_boundary_workspace_mib = reported_workspace_bytes / (1024.0 * 1024.0)
        # FactorPipeline has one caller wait before/after enqueue, one begin
        # wait per group, one wait per interior ready slice, and one final
        # ready wait.  Serial factor scheduling does not create these events.
        factor_event_waits = (
            factor_group_count + factor_partial_ready_groups * factor_slices + 3
            if factor_overlap and factor_group_count >= 2 else 0.0
        )
    else:
        factor_slices = factor_overlap = factor_total_launches = 0.0
        factor_partial_ready_groups = factor_boundary_workspace_mib = 0.0
        factor_event_waits = factor_total_ctas = factor_max_grid_ctas = 0.0
    launch_count_feature = factor_total_launches if factor_backend else groups * tiles
    return [
        1.0,
        n * batch * log_n / 1.0e6,
        2.0 * n * batch * word_bytes * groups / 1.0e6,
        decomposition,
        max(decomposition - 1.0, 0.0),
        tile_threads / 256.0,
        _number(sample, "local_stages") / log_n,
        (prefix_threads + suffix_threads) / 512.0,
        0.0 if str(sample.get("operator", "fft")) == "fft" else 1.0,
        float(backend == "hierarchical"),
        float(backend == "online-reorder"),
        float(backend == "temporal-tile"),
        float(backend == "warp-hybrid"),
        float(backend == "stage-pipeline"),
        float(fft_core == "scalar"),
        float(fft_core == "cta-dft8"),
        float(fft_core == "cufftdx-block"),
        float(fft_core == "cufftdx-direct"),
        float(fft_core == "cufftdx-resident"),
        float(fft_core == "thread-dft8"),
        float(fft_core == "turbofft-generated"),
        float(compute_unit == "radix2"),
        float(compute_unit == "radix4"),
        float(compute_unit == "radix8"),
        float(sample.get("cross_twiddle", "") == "recurrence"),
        float(sample.get("local_exchange", "") == "warp-register"),
        float(sample.get("shared_layout", "") == "xor-swizzle"),
        prefix_threads / 256.0,
        suffix_threads / 256.0,
        _number(sample, "prefix_ept") / 8.0,
        _number(sample, "suffix_ept") / 8.0,
        groups,
        float(_number(sample, "stage_overlap") != 0),
        min(batch, max(1.0, _number(sample, "batch_tile_count", 1.0))) / batch
        if _number(sample, "stage_overlap") else 1.0,
        launch_count_feature,
        max(0,groups-1) * (4*tiles-2) if overlap else 0.0,
        2 * max(0,groups-1) * tile * _number(sample,"batch_stride",n) * word_bytes / (1024*1024) if overlap else 0.0,
        math.ceil(batch / max(1.0, _number(sample, "batch_tile_count", 1.0)))
        if _number(sample, "stage_overlap") else 1.0,
        float(sample.get("shared_layout") == "writer-aligned"),
        float(sample.get("direct_boundary") == "tiled-transpose"),
        float(sample.get("direct_boundary") == "prefix-tiled-transpose"),
        float(sample.get("complex_multiply") == "gauss3"),
        float(sample.get("direction") == "inverse" or _number(sample, "inverse") != 0),
        float(sample.get("normalization") == "inverse" and sample.get("direction") == "inverse"),
        float(sample.get("placement") == "in-place"),
        math.log2(max(1.0, _number(sample, "element_stride", 1))),
        _number(sample, "batch_stride", n) / n,
        float(fft_core == "register-tile"),
        float(backend == "shared-iterative"),
        float(str(sample.get("group_cores", sample.get("segment_cores", ""))).split(":")[0] == "register-tile"),
        float(str(sample.get("group_cores", sample.get("segment_cores", ""))).split(":")[-1].startswith("cufftdx")),
        float(precision == "fp64"), float(precision in ("fp16", "bf16")),
        float(sample.get("accumulation") == "fp32"),
        max(stage_counts), min(stage_counts), max(live), sum(live),
        max([float(g.get("data_time",1)) for g in physical] or [1]),
        math.log2(max(1,min([float(g.get("grid_ctas",1)) for g in physical] or [1]))),
        float(sample.get("operator")=="ntt"), float(sample.get("operator")=="fwht"),
        float(sample.get("operator") in ("subset-zeta","superset-zeta","xor-zeta")),
        float(sample.get("operator")=="structured-2x2"), math.log2(max(1,_number(sample,"modulus",1))),
        float(backend=="factor-streamed"),
        max([float(g.get("data_tiles_per_cta",1)) for g in physical] or [float(mapping.get("data_tiles_per_cta",1))]),
        max([float(g.get("prefetch_depth",0)) for g in physical] or [float(mapping.get("prefetch_depth",0))]),
        max([float(g.get("compiler_local_bytes_per_thread",0))/1024 for g in physical] or [0]),
        sum(bool(g.get("compiler_local_resources_known",False)) for g in physical)/max(1,len(physical)),
        max([float(g.get("compiler_registers_per_thread",0)) for g in physical] or [0]),
        factor_slices,
        float(factor_overlap),
        factor_total_launches,
        float(factor_partial_ready_groups),
        factor_boundary_workspace_mib,
        factor_event_waits,
        factor_total_ctas,
        factor_max_grid_ctas,
    ]


def _require_finite(values: list[float], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite value")


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with pivoted Gaussian elimination.

    Feature normalization should make this system well-scaled, but a complete
    workload holdout can still expose a rank-deficient or unusually sparse
    training matrix.  Keep the existing solver and ridge model while making
    the pivot fallback sign-preserving and rejecting non-finite arithmetic.
    """
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("cost model linear system has an invalid shape")
    _require_finite(vector, "linear-system rhs")
    for row in matrix:
        _require_finite(row, "linear-system matrix")
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot_value = augmented[pivot][column]
        if not math.isfinite(pivot_value):
            raise ValueError("cost model linear solve became non-finite")
        if abs(pivot_value) < 1.0e-12:
            pivot_value = math.copysign(1.0e-12, pivot_value or 1.0)
            augmented[pivot][column] = pivot_value
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= scale
        _require_finite(augmented[column], "normalized linear-system row")
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
            _require_finite(augmented[row], "eliminated linear-system row")
    result = [augmented[index][size] for index in range(size)]
    _require_finite(result, "linear-system solution")
    return result


def _column_center_scale(values: list[float]) -> tuple[float, float]:
    """Return a finite mean and RMS spread without squaring huge values."""
    if not values:
        raise ValueError("cannot standardize an empty feature column")
    _require_finite(values, "feature column")
    # Scale before summing so a large but representable column does not make
    # math.fsum overflow merely because several values share its sign.
    magnitude = max(abs(value) for value in values)
    if magnitude == 0.0:
        return 0.0, 1.0
    scaled_values = [value / magnitude for value in values]
    mean = magnitude * (math.fsum(scaled_values) / len(values))
    deviations = [value - mean for value in values]
    _require_finite(deviations, "feature deviations")
    scale = max(abs(value) for value in deviations)
    if scale <= 1.0e-12:
        return mean, 1.0
    normalized = [(value / scale) for value in deviations]
    spread = scale * math.sqrt(math.fsum(value * value for value in normalized) / len(values))
    if not math.isfinite(spread) or spread <= 1.0e-12:
        return mean, 1.0
    return mean, spread


def fit(rows: list[dict[str, Any]], ridge: float) -> tuple[list[float], list[float], list[float]]:
    if not rows:
        raise ValueError("at least one row is required to fit the cost model")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    raw_features = [feature_vector(row["sample"]) for row in rows]
    for features in raw_features:
        _require_finite(features, "raw feature vector")
    means = [0.0] * len(FEATURE_NAMES)
    scales = [1.0] * len(FEATURE_NAMES)
    for column in range(1, len(FEATURE_NAMES)):
        values = [features[column] for features in raw_features]
        means[column], scales[column] = _column_center_scale(values)
    normalized = []
    for features in raw_features:
        row = [features[0]] + [(value - means[index]) / scales[index] for index, value in enumerate(features[1:], 1)]
        _require_finite(row, "normalized feature vector")
        normalized.append(row)
    size = len(FEATURE_NAMES)
    gram = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for features, row in zip(normalized, rows):
        target = math.log(max(float(row["median_kernel_ms"]), 1.0e-9))
        if not math.isfinite(target):
            raise ValueError("median kernel latency must be finite and positive")
        for left in range(size):
            rhs[left] += features[left] * target
            for right in range(size):
                gram[left][right] += features[left] * features[right]
    for diagonal in range(1, size):
        gram[diagonal][diagonal] += ridge
    _require_finite(rhs, "cost model rhs")
    for row in gram:
        _require_finite(row, "cost model gram matrix")
    return solve_linear(gram, rhs), means, scales


def predict(sample: dict[str, Any], coefficients: list[float], means: list[float], scales: list[float]) -> float:
    raw = feature_vector(sample)
    if not len(raw) == len(coefficients) == len(means) == len(scales):
        raise ValueError("cost model feature schema changed; refit on this hardware with complete mapping records")
    _require_finite(raw, "prediction feature vector")
    _require_finite(coefficients, "prediction coefficients")
    _require_finite(means, "prediction feature means")
    normalized = [raw[0]]
    for index, value in enumerate(raw[1:], 1):
        if not math.isfinite(scales[index]) or scales[index] <= 0.0:
            raise ValueError("cost model feature scale is invalid")
        item = (value - means[index]) / scales[index]
        if not math.isfinite(item):
            raise ValueError("prediction feature normalization became non-finite")
        normalized.append(item)
    exponent = sum(coefficient * value for coefficient, value in zip(coefficients, normalized))
    if math.isnan(exponent):
        raise ValueError("cost model prediction became NaN")
    return math.exp(max(-700.0, min(700.0, exponent)))


def load_sweep_rows(paths: list[pathlib.Path], require_exclusive: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_fields = SWEEP_IDENTITY_FIELDS
    for path in paths:
        with path.open(newline="") as handle:
            grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
            for source in csv.DictReader(handle):
                if source.get("correct") != "1":
                    continue
                if require_exclusive and source.get("measurement_exclusive_gpu") != "1":
                    raise ValueError(
                        f"{path} lacks exclusive-GPU measurement evidence; rerun the sweep with "
                        "--require-exclusive-gpu or pass --allow-nonexclusive-csv for exploration"
                    )
                key = tuple(source.get(field, "") for field in identity_fields)
                grouped.setdefault(key, []).append(source)
            for key, samples in grouped.items():
                source = samples[0]
                sample = dict(source)
                sample.setdefault("N", str(1 << int(float(source.get("logN", "0")))))
                candidate = "sweep-" + "-".join(value or "0" for value in key[5:12])
                rows.append({
                    "name": candidate,
                    "median_kernel_ms": statistics.median(float(row["kernel_ms"]) for row in samples),
                    "sample": sample,
                })
    return rows


def load_rows(path: pathlib.Path, candidate_csv: list[pathlib.Path] | None = None,
              require_exclusive: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("operator calibration must be a JSON list")
    internal: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    for candidate in raw:
        if candidate.get("status") != "measured" or not candidate.get("correct"):
            continue
        samples = candidate.get("samples") or []
        if not samples:
            continue
        row = {"name": candidate.get("name", ""), "median_kernel_ms": float(candidate["median_kernel_ms"]), "sample": samples[0]}
        (external if is_external(candidate, samples[0]) else internal).append(row)
    for row in load_sweep_rows(candidate_csv or [], require_exclusive=require_exclusive):
        (external if is_external(row, row["sample"]) else internal).append(row)
    return internal, external


def workload_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the normalized complete workload semantics.

    Candidate mappings within one workload may differ freely, but direction,
    normalization, strides, ordering, modulus and stage-matrix semantics must
    remain in the same group only when the registry's canonical helper says
    they are equivalent.
    """
    canonical = canonical_semantics(row["sample"])
    # Use the registry's fixed field set so omitted optional fields and their
    # canonical defaults cannot create separate groups by dictionary shape.
    return tuple((field, str(canonical.get(field, ""))) for field in SEMANTICS)


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, str], ...], list[dict[str, Any]]]:
    grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(workload_key(row), []).append(row)
    return grouped


def design_point_key(row: dict[str, Any]) -> tuple:
    """Identify resolved designs without confusing provenance names with mappings.

    Legacy records retain name identity. Known mapping, semantic or target
    conflicts remain distinct even when their candidate names happen to agree.
    Only complete exported mapping envelopes can merge differently named rows.
    """
    sample = row["sample"]
    raw = sample.get("mapping_json")
    try:
        mapping = json.loads(raw) if isinstance(raw, str) and raw else raw
    except (TypeError, ValueError):
        mapping = None
    context = tuple(str(sample.get(k, "")) for k in
                    ("device", "compute_capability", "global_memory_bytes", "device_uuid",
                     "gpu_uuid", "uuid", "runtime_fingerprint"))
    required = {"schema_version", "kind", "backend", "compute_unit", "stage_partition",
                "execution_group_mappings"}
    if isinstance(mapping, dict):
        if mapping.get("kind") == "butterfly":
            required |= {"fft_core", "tile_threads", "boundaries", "segment_mappings"}
        elif mapping.get("kind") == "ntt":
            required |= {"threads_per_block", "boundary_mappings", "subgraph_mappings"}
        else:
            required.add("__unsupported_mapping_kind__")
        serialized = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        complete = (mapping.get("schema_version") == 1 and required <= mapping.keys()
                    and bool(sample.get("runtime_fingerprint")) and bool(sample.get("device"))
                    and bool(sample.get("compute_capability")))
        if complete:
            return ("resolved", workload_key(row), context, serialized)
    else:
        serialized = json.dumps(raw, sort_keys=True)
    # Old CSV display names omit launch axes. Preserve the full sweep grouping
    # identity as well, so equal names cannot merge different thread/EPT points.
    legacy_axes = tuple(str(sample.get(k, "")) for k in SWEEP_IDENTITY_FIELDS[5:])
    return ("legacy", workload_key(row), context, str(row.get("name", "")), serialized, legacy_axes)


def _winner_match(measured: dict[str, Any], predicted: dict[str, Any]) -> bool:
    key = design_point_key(measured)
    return (key[0] == "resolved" or bool(measured.get("name"))) and key == design_point_key(predicted)


def _distinct_candidate_count(rows: list[dict[str, Any]]) -> int:
    return len({design_point_key(row) for row in rows})


def grouped_holdout_validation(rows: list[dict[str, Any]], ridge: float,
                               minimum_training_rows: int = 3) -> dict[str, Any]:
    """Validate ranking by holding out every complete workload group.

    A group is usable only when it has at least two distinct measured internal
    designs and the remaining complete workloads provide the minimum model
    training rows.  The model is fitted afresh for each held-out group, so no
    candidate from the evaluated workload can influence its predicted ranking.
    """
    grouped = _group_rows(rows)
    checks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    all_keys = sorted(grouped)
    for key, held_out in sorted(grouped.items()):
        if len(held_out) < 2:
            continue
        distinct_count = _distinct_candidate_count(held_out)
        if distinct_count < 2:
            skipped.append({"workload": list(key), "reason": "insufficient_distinct_candidates",
                            "candidate_rows": len(held_out), "distinct_candidate_count": distinct_count})
            continue
        training = [row for other_key in all_keys if other_key != key for row in grouped[other_key]]
        if len(training) < minimum_training_rows:
            skipped.append({"workload": list(key), "reason": "insufficient_complete_workload_training_rows",
                            "training_rows": len(training), "candidate_rows": len(held_out)})
            continue
        try:
            coefficients, means, scales = fit(training, ridge)
            estimates = [(row, predict(row["sample"], coefficients, means, scales)) for row in held_out]
        except (OverflowError, ValueError) as error:
            skipped.append({"workload": list(key), "reason": "non_finite_holdout_model",
                            "training_rows": len(training), "candidate_rows": len(held_out),
                            "error": str(error)})
            continue
        measured = min(held_out, key=lambda row: row["median_kernel_ms"])
        predicted = min(estimates, key=lambda item: item[1])[0]
        best_latency = float(measured["median_kernel_ms"])
        chosen_latency = float(predicted["median_kernel_ms"])
        if not math.isfinite(best_latency) or best_latency <= 0.0 or not math.isfinite(chosen_latency) or chosen_latency <= 0.0:
            skipped.append({"workload": list(key), "reason": "invalid_holdout_latency",
                            "training_rows": len(training), "candidate_rows": len(held_out)})
            continue
        checks.append({
            "workload": list(key),
            "candidate_rows": len(held_out),
            "distinct_candidate_count": distinct_count,
            "training_rows": len(training),
            "measured_winner": measured["name"],
            "predicted_winner": predicted["name"],
            "match": _winner_match(measured, predicted),
            "latency_regret": chosen_latency / best_latency,
        })
    regrets = [float(item["latency_regret"]) for item in checks]
    top1 = (sum(bool(item["match"]) for item in checks) / len(checks)) if checks else None
    return {
        "method": "leave_one_complete_workload_out",
        "identity_version": VALIDATION_IDENTITY_VERSION,
        "workload_winner_checks": checks,
        "skipped_workloads": skipped,
        "valid_multi_candidate_workloads": len(checks),
        "top1_accuracy": top1,
        "latency_regret": statistics.mean(regrets) if regrets else None,
        "mean_latency_regret": statistics.mean(regrets) if regrets else None,
        "geomean_latency_regret": math.exp(math.fsum(math.log(value) for value in regrets) / len(regrets)) if regrets else None,
        "max_latency_regret": max(regrets) if regrets else None,
    }


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text())
    internal, external = load_rows(args.operator_calibration, args.candidate_csv,
                                   require_exclusive=not args.allow_nonexclusive_csv)
    if len(internal) < 3:
        result = {
            "schema": "cubutterfly-local-cost-model-v1",
            "status": "insufficient-candidates",
            "device": profile.get("device", ""),
            "compute_capability": profile.get("compute_capability", ""),
            "training_rows": len(internal),
            "external_reference_rows": len(external),
            "reason": "at least three correctness-checked internal candidates are required",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0

    coefficients, means, scales = fit(internal, args.ridge)
    loo_errors: list[float] = []
    for index, row in enumerate(internal):
        training = internal[:index] + internal[index + 1:]
        if len(training) < 3:
            continue
        loo_coefficients, loo_means, loo_scales = fit(training, args.ridge)
        actual = row["median_kernel_ms"]
        estimate = predict(row["sample"], loo_coefficients, loo_means, loo_scales)
        loo_errors.append(abs(estimate - actual) / max(actual, 1.0e-9))

    grouped = _group_rows(internal)
    winner_checks = []
    multi_candidate_checks = []
    for key, rows in sorted(grouped.items()):
        measured = min(rows, key=lambda row: row["median_kernel_ms"])
        predicted = min(rows, key=lambda row: predict(row["sample"], coefficients, means, scales))
        check = {"workload": list(key), "measured_winner": measured["name"],
                 "predicted_winner": predicted["name"], "match": _winner_match(measured, predicted),
                 "distinct_candidate_count": _distinct_candidate_count(rows)}
        winner_checks.append(check)
        if check["distinct_candidate_count"] >= 2:
            multi_candidate_checks.append(check)

    median_relative_error = statistics.median(loo_errors) if loo_errors else None
    winner_accuracy = (sum(item["match"] for item in multi_candidate_checks) / len(multi_candidate_checks)) if multi_candidate_checks else None
    holdout = grouped_holdout_validation(internal, args.ridge)
    holdout_accuracy = holdout["top1_accuracy"]
    holdout_count = holdout["valid_multi_candidate_workloads"]
    # Training-set winner checks remain useful diagnostics, but cannot certify
    # ranking of an unseen workload.  Admission therefore uses complete-group
    # holdout ranking plus the existing row-level error threshold.
    validation_pass = (holdout_count >= 3 and median_relative_error is not None and
                       median_relative_error <= 0.30 and holdout_accuracy is not None and holdout_accuracy >= 0.90)
    result = {
        "schema": "cubutterfly-local-cost-model-v1",
        "status": "calibrated-local" if validation_pass else "calibrated-local-warning",
        "device": profile.get("device", ""),
        "compute_capability": profile.get("compute_capability", ""),
        "hardware_profile": str(args.profile),
        "operator_calibration": str(args.operator_calibration),
        "candidate_csv": [str(path) for path in args.candidate_csv],
        "training_rows": len(internal),
        "external_reference_rows": len(external),
        "excluded_external_candidates": [row["name"] for row in external],
        "features": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "validation_identity_version": VALIDATION_IDENTITY_VERSION,
        "feature_means": means,
        "feature_scales": scales,
        "coefficients": coefficients,
        "ridge": args.ridge,
        "target_transform": "log_kernel_ms",
        "usable_for_unmeasured_ranking": validation_pass,
        "training_workload_winner_accuracy": winner_accuracy,
        "holdout_top1_accuracy": holdout_accuracy,
        "holdout_latency_regret": holdout["latency_regret"],
        "holdout_valid_multi_candidate_workloads": holdout_count,
        "validation": {
            "scope": "leave_one_complete_workload_out",
            "identity_version": VALIDATION_IDENTITY_VERSION,
            "leave_one_out_rows": len(loo_errors),
            "leave_one_out_median_relative_error": median_relative_error,
            "leave_one_out_max_relative_error": max(loo_errors) if loo_errors else None,
            # Compatibility fields: these are explicitly training-set-only
            # diagnostics and are not used by validation_pass.
            "workload_winner_checks": winner_checks,
            "multi_candidate_workloads": len(multi_candidate_checks),
            "workload_winner_accuracy": winner_accuracy,
            "training_workload_winner_checks": winner_checks,
            "training_multi_candidate_workloads": len(multi_candidate_checks),
            "training_workload_winner_accuracy": winner_accuracy,
            "holdout_workload_winner_checks": holdout["workload_winner_checks"],
            "holdout_skipped_workloads": holdout["skipped_workloads"],
            "holdout_valid_multi_candidate_workloads": holdout_count,
            "holdout_top1_accuracy": holdout_accuracy,
            "holdout_latency_regret": holdout["latency_regret"],
            "holdout_mean_latency_regret": holdout["mean_latency_regret"],
            "holdout_geomean_latency_regret": holdout["geomean_latency_regret"],
            "holdout_max_latency_regret": holdout["max_latency_regret"],
            "pass_thresholds": {"median_relative_error": 0.30, "winner_accuracy": 0.90,
                                "holdout_top1_accuracy": 0.90,
                                "holdout_valid_multi_candidate_workloads": 3},
            "training_winner_accuracy_used_for_pass": False,
        },
        "limitations": [
            "The model ranks cuButterfly candidates; it does not replace correctness-checked timing.",
            "External complete-library rows such as cuFFT are references, not training rows.",
            "Coverage is limited to the candidate shapes present in operator_calibration.json.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"cost model: {result['status']} ({len(internal)} internal rows, {len(external)} external references)")
    print(f"leave-one-out median relative error: {result['validation']['leave_one_out_median_relative_error']}")
    print(f"training workload winner accuracy: {result['validation']['training_workload_winner_accuracy']}")
    print(f"holdout top-1 accuracy: {result['validation']['holdout_top1_accuracy']}")
    print(f"holdout latency regret: {result['validation']['holdout_latency_regret']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
