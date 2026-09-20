"""Portable mapping proposals, never portable performance measurements.

Source-context winners can seed another GPU/build/placement at the same
operator, precision and transform length. Every proposal remains untrusted
until the target benchmark resolves, verifies and times it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from calibration_space import candidate_id, mapping_point
from hardware_registry import SCHEMA as REGISTRY_SCHEMA, SEMANTICS, canonical_semantics, default_path, stable_id

SCHEMA = "cubutterfly-mapping-seeds-v1"


def canonical_mapping(mapping):
    """Apply the public plan's legacy column alias normalization.

    Imported online cores derive physical columns from threads and EPT.
    ButterflyPlan therefore sets reorder_columns=1 for these cores. Preserve
    every real mapping axis and leave scalar/factor columns untouched.
    """
    normalized = dict(mapping)
    if (normalized.get("backend") == "online-reorder"
            and normalized.get("fft_core") in {"register-tile", "cufftdx-block"}
            and "reorder_columns" in normalized):
        normalized["reorder_columns"] = 1
    return normalized


def _validate_strategy_axes(mapping, semantics=None):
    """Validate compiler-owned strategy axes before they enter a seed file."""
    if not isinstance(mapping, dict):
        raise ValueError("mapping seed mapping must be an object")
    backend = mapping.get("backend")
    core = mapping.get("fft_core")
    codelet = mapping.get("prefix_codelet", "native")
    layout = mapping.get("prefix_shared_layout", "linear")
    if codelet not in {"native", "cufftdx-thread"}:
        raise ValueError("prefix_codelet must be native or cufftdx-thread")
    if layout not in {"linear", "xor"}:
        raise ValueError("prefix_shared_layout must be linear or xor")
    if (codelet != "native" or layout != "linear") and not (
            backend == "online-reorder" and core == "register-tile"):
        raise ValueError("prefix codelet/layout are valid only for online-reorder register-tile")
    precision = str((semantics or {}).get("precision", "")).lower()
    if precision == "fp64" and codelet == "cufftdx-thread":
        raise ValueError("cufftdx-thread prefix is unavailable for FP64")
    policies = mapping.get("factor_io_policies", [])
    if policies in (None, ""):
        policies = []
    if not isinstance(policies, list):
        raise ValueError("factor_io_policies must be a list")
    if backend != "factor-streamed" and policies:
        raise ValueError("factor_io_policies are valid only for factor-streamed")
    factors = mapping.get("factor_partition")
    if backend == "factor-streamed" and policies and isinstance(factors, list) and len(policies) != len(factors):
        raise ValueError("factor_io_policies must match factor_partition")
    if any(value not in {"dynamic", "static-unrolled"} for value in policies):
        raise ValueError("factor_io_policies entries must be dynamic or static-unrolled")


def _seed(semantics, mapping, provenance):
    if not isinstance(mapping, dict) or mapping.get("schema_version") != 1:
        raise ValueError("mapping seeds require a schema_version=1 public mapping")
    if not mapping.get("backend") or mapping["backend"] in ("cufft", "vkfft"):
        raise ValueError("mapping seeds must use an internal backend")
    if any(key in mapping for key in ("kernel_ms", "median_kernel_ms", "runtime_fingerprint")):
        raise ValueError("mapping seed payload must not contain measurements")
    semantics = canonical_semantics(semantics)
    if any(key not in semantics for key in ("operator", "precision", "logN")):
        raise ValueError("mapping seeds require operator, precision and logN semantics")
    mapping = canonical_mapping(mapping)
    _validate_strategy_axes(mapping, semantics)
    return {"semantics": semantics, "mapping": mapping, "provenance": provenance}


def _registry_seeds(document, source):
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported seed registry schema")
    result = []
    for target in document.get("targets", []):
        winners = {}
        for record in target.get("records", []):
            mapping = record.get("mapping", {})
            if record.get("status") not in ("measured", "needs-revalidation") or mapping.get("schema_version") != 1:
                continue
            if mapping.get("backend") in ("cufft", "vkfft"):
                continue
            try:
                latency = float(record.get("kernel_ms", 0))
            except (ValueError, TypeError):
                continue
            if not math.isfinite(latency) or latency <= 0:
                continue
            # Compare historical times only within their original context to
            # identify its winner. They never rank candidates on the target.
            key = stable_id([record["key"], record.get("runtime_fingerprint", "")])
            if key not in winners or latency < float(winners[key]["kernel_ms"]):
                winners[key] = record
        for key in sorted(winners):
            record = winners[key]
            result.append(_seed(record["key"], record["mapping"], {
                "source": source, "kind": "registry-winner", "hardware": target["hardware"],
                "record_id": record["record_id"], "runtime_fingerprint": record.get("runtime_fingerprint", ""),
            }))
    return result


def seed_snapshot(args, path):
    """Freeze automatic registry inputs across resume and later promotion."""
    files = [Path(p).expanduser().resolve() for p in (getattr(args, "mapping_seeds", None) or [])]
    sources = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    registry = default_path().expanduser().resolve()
    inputs = {"files": sources, "registry_path": str(registry)}
    if getattr(args, "resume_search", False) and path.exists():
        snapshot = json.loads(path.read_text())
        if snapshot.get("schema") != SCHEMA or snapshot.get("inputs") != inputs:
            raise ValueError("mapping seed inputs changed; use a new output directory for new seeds")
        return snapshot
    seeds = []
    for p in files:
        document = json.loads(p.read_text())
        if document.get("schema") == REGISTRY_SCHEMA:
            seeds.extend(_registry_seeds(document, str(p)))
        elif document.get("schema") == SCHEMA:
            for index, record in enumerate(document["seeds"]):
                seeds.append(_seed(record["semantics"], record["mapping"], {
                    "source": str(p), "kind": "explicit", "index": index,
                    "origin": record.get("provenance", {}),
                }))
        else:
            raise ValueError(f"unsupported mapping seed schema: {p}")
    if registry.exists() and registry not in files:
        seeds.extend(_registry_seeds(json.loads(registry.read_text()), str(registry)))
    snapshot = {"schema": SCHEMA, "inputs": inputs, "seeds": seeds,
                "reuse": "mapping-only; target legality, correctness and timing required"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(path)
    return snapshot


def candidates_for_workload(snapshot, workload):
    target = canonical_semantics(workload)
    result = {}
    # Explicit seeds precede automatic historical winners. Exact semantics
    # precede transferred semantics within each source class. No target-time
    # predictions or historical cross-context latency comparisons are used.
    def priority(seed):
        semantics = seed["semantics"]
        return (seed["provenance"]["kind"] != "explicit",
                any(semantics.get(k, "") != target.get(k, "") for k in SEMANTICS),
                stable_id(seed["mapping"]))
    for seed in sorted(snapshot.get("seeds", []), key=priority):
        if any(seed["semantics"].get(k) != target.get(k) for k in ("operator", "precision", "logN")):
            continue
        point = mapping_point(workload, canonical_mapping(seed["mapping"]))
        identity = candidate_id(point)
        source = {"semantics": seed["semantics"], **seed["provenance"]}
        if identity not in result:
            result[identity] = {"point": point, "sources": []}
        result[identity]["sources"].append(source)
    return list(result.values())


def validate_mapping(requested, resolved):
    """Reject changed requested axes; allow the decoder to add default axes."""
    def contains(expected, actual):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(k in actual and contains(v, actual[k]) for k, v in expected.items())
        if isinstance(expected, list):
            # Runtime normalization may materialize the public [] default as
            # one dynamic entry per factor.  This is the only list default
            # accepted without weakening per-stage strategy identity.
            if expected == [] and isinstance(actual, list) and all(item == "dynamic" for item in actual):
                return True
            return isinstance(actual, list) and len(expected) == len(actual) and all(contains(x, y) for x, y in zip(expected, actual))
        return expected == actual
    if not contains(requested, resolved):
        raise ValueError("resolved mapping differs from requested mapping seed")
