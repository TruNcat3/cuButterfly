"""CPU-only, design-selected requests for native cross-stage composition.

This module emits requests, not resolved descriptors or performance evidence.
Physical segmentation is taken verbatim from verified operator samples.
"""
from __future__ import annotations

import copy
import hashlib
import json

from benchmark_semantics import normalize_benchmark_semantics
from stage_composition import _groups, _mapping, _serial_point, _truthy


SCHEMA = "cubutterfly-composition-inventory-v1"
_POINT_FIELDS = {
    "operator", "precision", "accumulation", "direction", "normalization",
    "placement", "logN", "batch", "element_stride", "batch_stride", "modulus",
    "input_order", "output_order", "backend", "compute_unit",
    "fft_core", "complex_multiply", "cross_twiddle", "direct_boundary",
    "local_exchange", "shared_layout", "stage_matrix", "stage_matrices",
}
_INTEGER_FIELDS = {
    "logN", "batch", "element_stride", "batch_stride", "modulus",
}
_BUTTERFLY_OPS = (
    "fft", "fwht", "subset-zeta", "superset-zeta", "xor-zeta", "structured-2x2", "ntt",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _id(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()[:20]


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _point(sample, mapping):
    point = {key: copy.deepcopy(value) for key, value in sample.items()
             if key in _POINT_FIELDS and value not in (None, "")}
    for key in _INTEGER_FIELDS & point.keys():
        # Parsing integers directly is essential for UINT64 NTT moduli.
        if not isinstance(point[key], (int, str)) or isinstance(point[key], bool):
            raise ValueError("integer fields require exact integers or integer strings")
        point[key] = int(point[key])
    for key in ("backend", "fft_core"):
        if key in mapping:
            point[key] = mapping[key]
    if point.get("operator") == "ntt" and "output_order" in point:
        # The NTT CLI adapts placement into --output-order as well.
        point["placement"] = point["output_order"]
    point["mapping_json"] = _json(mapping)
    return normalize_benchmark_semantics(point)


def _family(point):
    return "/".join(str(point.get(key, "native")) for key in
                    ("backend", "operator", "precision", "direction", "fft_core"))


def _support(point, mapping, groups):
    backend = point.get("backend")
    operator = point.get("operator")
    if not groups:
        return "actual-physical-groups-unavailable"
    if backend == "shared-iterative":
        if operator not in _BUTTERFLY_OPS:
            return "unsupported-shared-operator"
        return None if len(groups) >= 2 else "requires-at-least-two-physical-groups"
    if backend == "online-reorder":
        if (operator != "fft" or point.get("fft_core") not in
                {"cufftdx-block", "register-tile"}):
            return "unsupported-online-core-or-operator"
        return None if len(groups) == 2 else "online-requires-exactly-two-physical-groups"
    if backend == "factor-streamed":
        if operator != "fft" or point.get("precision") not in {"fp32", "fp64"}:
            return "factor-requires-fp32-or-fp64-fft"
        if len(groups) < 3:
            return "factor-requires-at-least-three-physical-groups"
        partition = mapping.get("factor_partition", [])
        if (len(partition) != len(groups) or not partition or
                any(not isinstance(n, int) or n < 1 or n > 14 for n in partition) or
                sum(partition) != point["logN"]):
            return "factor-physical-partition-mismatch"
        if (mapping.get("fft_core") not in {"cufftdx-block", "register-tile"} or
                point.get("accumulation", "native") != "native" or
                mapping.get("shared_layout") != "writer-aligned" or
                mapping.get("cross_twiddle") != "recurrence" or
                mapping.get("direct_boundary") != "direct-strided" or
                mapping.get("local_exchange", "shared") != "shared" or
                mapping.get("complex_multiply", "four-mul") != "four-mul"):
            return "unsupported-factor-native-configuration"
        columns = mapping.get("factor_columns", 0)
        if not isinstance(columns, int) or not _power_of_two(columns):
            return "invalid-factor-columns"
        if (1 << partition[-1]) // columns < 2:
            return "no-legal-multiple-factor-slices"
        return None
    return "backend-has-no-native-composition-lowering"


def build_inventory(records, *, max_elements=1048576, tile_counts=(2, 4, 8),
                    max_seeds_per_family=2):
    """Return a deterministic, finite inventory without executing any plan.

    ``tile_counts`` are native batch_tile_count values (transforms per tile),
    and also candidate factor_slices values. ``holdout`` freezes every entire
    pair as validation data; timings must never select seeds or fit the model.
    Original B is retained. A separate B-1 workload is added only to exercise
    a nondivisible tail and is subject to the same whole-request element cap.
    """
    tiles = sorted(set(tile_counts))
    if (not isinstance(max_elements, int) or max_elements < 1 or
            not isinstance(max_seeds_per_family, int) or max_seeds_per_family < 1 or
            any(not isinstance(t, int) or not _power_of_two(t) or t < 2 or
                t > 4294967295 for t in tiles)):
        raise ValueError("positive bounds and power-of-two tiles in [2, UINT32_MAX] required")
    requests, omissions, candidates = [], [], []
    seen = set()

    def omit(source, reason, **details):
        omissions.append({"source": source, "reason": reason, **details})

    for index, record in enumerate(records):
        source = {"record_index": index, "name": record.get("name", str(index))}
        samples = record.get("samples", [])
        if (not samples or not isinstance(samples[0], dict) or
                not _truthy(record.get("correct", samples[0].get("correct", False))) or
                not _truthy(samples[0].get("correct", record.get("correct", False)))):
            omit(source, "source-not-verified")
            continue
        sample = samples[0]
        mapping = _mapping(sample.get("mapping_json", record.get("mapping_json")))
        if not mapping:
            omit(source, "mapping-unavailable")
            continue
        missing_semantics = [key for key in ("operator", "precision", "direction")
                             if sample.get(key) in (None, "")]
        if missing_semantics:
            omit(source, "required-public-semantics-unavailable", fields=missing_semantics)
            continue
        try:
            point = _point(sample, mapping)
            if not 1 <= point["logN"] <= 30 or point["batch"] < 1:
                raise ValueError("invalid workload")
        except (KeyError, TypeError, ValueError):
            omit(source, "invalid-workload-integers")
            continue
        source["mapping_id"] = _id(mapping)
        groups = _groups(record, sample)
        family = _family(point)
        reason = _support(point, mapping, groups)
        if reason:
            omit(source, reason, family=family, physical_group_count=len(groups))
            continue
        # Strip composition policy using the same serial projection as stage
        # service resolution, retaining all other mapping axes verbatim.
        base = _serial_point(point, point["batch"])
        base.pop("factor_overlap", None)
        base_mapping = _mapping(base["mapping_json"])
        if point["operator"] == "ntt":
            # PlanConfig has no factor policy; its public serializer drops
            # those Butterfly-only fields, including the serial helper's copy.
            base_mapping = {key: value for key, value in base_mapping.items()
                            if not key.startswith("factor_")}
        if point["backend"] == "factor-streamed":
            base_mapping["factor_slices"] = 1
        base["mapping_json"] = _json(base_mapping)
        identity = _id(base)
        if identity in seen:
            omit(source, "duplicate-semantic-mapping", family=family)
            continue
        seen.add(identity)
        candidates.append((family, _json(base), source, base, base_mapping))

    counts = {}
    for family, _, source, base, mapping in sorted(candidates, key=lambda x: (x[0], x[1])):
        batch = base["batch"]
        workloads = [(batch, "original")]
        if base["backend"] != "factor-streamed" and batch > 2 and any(
                t < batch - 1 and batch % t == 0 and (batch - 1) % t for t in tiles):
            workloads.append((batch - 1, "additional-tail"))
        eligible_workloads = []
        for workload_batch, workload in workloads:
            if (1 << base["logN"]) * workload_batch > max_elements:
                omit(source, "whole-workload-element-limit", family=family,
                     batch=workload_batch, workload=workload,
                     elements=(1 << base["logN"]) * workload_batch)
                continue
            legal = [t for t in tiles if t < workload_batch]
            if base["backend"] == "factor-streamed":
                limit = (1 << mapping["factor_partition"][-1]) // mapping["factor_columns"]
                legal = [t for t in tiles if t <= limit]
            if not legal:
                omit(source, "no-requested-legal-tile-or-slices", family=family,
                     batch=workload_batch, workload=workload)
            else:
                eligible_workloads.append((workload_batch, workload, legal))
        if not eligible_workloads:
            continue
        if counts.get(family, 0) >= max_seeds_per_family:
            omit(source, "design-seed-budget", family=family)
            continue
        counts[family] = counts.get(family, 0) + 1
        for workload_batch, workload, legal in eligible_workloads:
            for tile in legal:
                pair_id = _id([base, workload_batch, tile])
                factor = base["backend"] == "factor-streamed"
                roles = ("bulk", "sliced-serial", "overlap") if factor else ("bulk", "overlap")
                for role in roles:
                    point = copy.deepcopy(base)
                    m = copy.deepcopy(mapping)
                    policy = {"stage_overlap": not factor and role == "overlap",
                              "batch_tile_count": tile if not factor and role == "overlap" else 1}
                    if base["operator"] != "ntt":
                        policy["factor_overlap"] = factor and role == "overlap"
                    if factor:
                        policy["factor_slices"] = 1 if role == "bulk" else tile
                    m.update(policy)
                    point.update({key: value for key, value in policy.items()
                                  if key not in {"factor_overlap", "factor_slices"}})
                    point.update(batch=workload_batch, mapping_json=_json(m))
                    requests.append({"id": pair_id + "-" + role, "pair_id": pair_id,
                                     "family": family, "role": role, "point": point,
                                     "holdout": True, "source": copy.deepcopy(source),
                                     "workload": workload, "tile_or_slices": tile})

    required = ["shared-iterative/" + op for op in _BUTTERFLY_OPS]
    required += ["online-reorder/fft/" + core for core in ("cufftdx-block", "register-tile")]
    required += ["factor-streamed/fft/" + precision for precision in ("fp32", "fp64")]
    covered = set()
    for request in requests:
        p = request["point"]
        suffix = ("/" + p["fft_core"] if p["backend"] == "online-reorder" else
                  "/" + p["precision"] if p["backend"] == "factor-streamed" else "")
        covered.add(p["backend"] + "/" + p["operator"] + suffix)
    gaps = sorted(set(required) - covered)
    return {"schema": SCHEMA,
            "scope": {"kind": "native-verified-mapping-composition-requests",
                      "hardware_independent": True, "measurement_status": "not-run",
                      "source_record_count": len(records),
                      "segmentation": "actual-source-groups-only"},
            "policy": {"selection": "lexicographic-design-configuration-never-timing",
                       "max_elements": max_elements, "tile_counts": tiles,
                       "max_seeds_per_family": max_seeds_per_family,
                       "batch_reduction": "none; B-1 is a separate explicit workload",
                       "holdout": "entire-pair; frozen-model-validation-only"},
            "requests": requests, "omissions": omissions,
            "coverage": {"status": "partial" if gaps or omissions else "planned",
                         "required": required, "covered": sorted(covered), "gaps": gaps,
                         "request_count": len(requests),
                         "pair_count": len({r["pair_id"] for r in requests})}}
