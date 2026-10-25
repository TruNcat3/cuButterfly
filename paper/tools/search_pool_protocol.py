#!/usr/bin/env python3
"""Freeze and replay the bounded E9 independent candidate-pool protocol.

The pool is selected from the declared E2/E3 workload domain and structural
mapping rules only.  Timing rows are accepted only by ``--mode replay`` after
the frozen protocol hash has been checked.  Replay is an offline study and
does not report online compilation, calibration, or search wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from heldout_protocol import (  # noqa: E402
    DESCRIPTORS_SCHEMA,
    PREDICTIONS_SCHEMA,
    VALIDATION_SCHEMA,
    _load_descriptor,
    _load_measurement_report,
    _load_predictions,
    _load_protocol,
    _observed_ms,
    _same_workload,
    _identity_value,
    _path_ref,
    _validate_runtime_inputs,
    audit_stage_checkpoint,
    _audit_holdout_semantics,
    balanced_partition,
    canonical,
    describe as _describe_stage,
    point_signature,
    protocol_hash as heldout_protocol_hash,
    predict as _predict_stage,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
    workload_id,
)
from research_compile_requests import compile_request  # noqa: E402


SCHEMA = "cubutterfly-paper-search-pool-protocol-v2"
MANIFEST_SCHEMA = "cubutterfly-paper-search-pool-manifest-v1"
POINTS_SCHEMA = "cubutterfly-paper-search-pool-points-v1"
RESULT_SCHEMA = "cubutterfly-paper-search-pool-replay-v1"
DEFAULT_E9 = (
    ("fft", 8, 1),
    ("ntt", 12, 1024),
)
BUDGETS = (4, 8, 16)
SEEDS = (17, 23, 41, 59, 83)
POLICIES = ("stage-service", "logical-traffic", "random", "local-search")


def describe(args: argparse.Namespace) -> dict[str, Any]:
    return _describe_stage(args, SCHEMA)


def predict(args: argparse.Namespace) -> dict[str, Any]:
    return _predict_stage(args, SCHEMA)


def _hash_basis(document: dict[str, Any]) -> dict[str, Any]:
    basis = json.loads(json.dumps(document))
    basis.pop("freeze", None)
    return basis


def protocol_hash(document: dict[str, Any]) -> str:
    return sha256_bytes(canonical(_hash_basis(document)).encode())


def _manifest_hash(manifest: dict[str, Any]) -> str:
    basis = json.loads(json.dumps(manifest))
    basis.pop("manifest_sha256", None)
    return sha256_bytes(canonical(basis).encode())


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def select_workloads(domain: list[dict[str, Any]], requested: list[str] | None = None,
                     excluded: set[str] | None = None) -> list[dict[str, Any]]:
    by_id = {workload_id(item): item for item in domain}
    excluded = excluded or set()
    if requested:
        selected = []
        for identifier in requested:
            if identifier not in by_id:
                raise ValueError(f"requested workload is absent from domain: {identifier}")
            if identifier in excluded:
                raise ValueError(f"E9 workload overlaps excluded holdout workload: {identifier}")
            selected.append(dict(by_id[identifier]))
    else:
        selected = []
        for operator, log_n, batch in DEFAULT_E9:
            matches = [item for item in domain if _same_workload(item, operator, log_n, batch)
                       and workload_id(item) not in excluded]
            if not matches:
                raise ValueError(f"representative workload is absent from domain: {operator}/{log_n}/{batch}")
            selected.append(dict(sorted(matches, key=canonical)[0]))
    identifiers = [workload_id(item) for item in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("E9 workload representatives must be unique")
    return selected


def _mapping_for(workload: dict[str, Any], partition: list[int], threads: int,
                 layout: str) -> dict[str, Any]:
    mapping = {
        "schema_version": 1,
        "kind": "ntt" if workload["operator"] == "ntt" else "butterfly",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "stage_partition": partition,
    }
    if workload["operator"] == "ntt":
        mapping.update(threads_per_block=threads, dataflow_layout=layout)
    else:
        mapping.update(tile_threads=threads, fft_core="scalar",
                       local_exchange="shared", shared_layout=layout)
    return mapping


def _point(workload: dict[str, Any], groups: int, threads: int, layout: str) -> dict[str, Any]:
    point = dict(workload)
    point["mapping_json"] = canonical(_mapping_for(
        workload, balanced_partition(int(workload["logN"]), groups), threads, layout))
    return point


def _validate_point(point: dict[str, Any]) -> dict[str, Any]:
    mapping = json.loads(point["mapping_json"])
    log_n = int(point["logN"])
    partition = mapping.get("stage_partition")
    if mapping.get("backend") != "shared-iterative" or mapping.get("compute_unit") != "radix2":
        raise ValueError("E9 candidates must use shared-iterative radix2")
    if not isinstance(partition, list) or not partition or sum(partition) != log_n:
        raise ValueError("E9 partition does not cover logN")
    if any(not isinstance(value, int) or value <= 0 for value in partition):
        raise ValueError("E9 partition contains a non-positive stage")
    if point["operator"] == "ntt":
        if mapping.get("threads_per_block") not in (128, 256):
            raise ValueError("E9 NTT thread count is outside the fixed legal set")
        if mapping.get("dataflow_layout") not in ("hermes-xor", "linear"):
            raise ValueError("E9 NTT layout is outside the fixed legal set")
    else:
        if mapping.get("tile_threads") not in (128, 256):
            raise ValueError("E9 butterfly thread count is outside the fixed legal set")
        if mapping.get("shared_layout") not in ("writer-aligned", "linear"):
            raise ValueError("E9 butterfly layout is outside the fixed legal set")
    request = compile_request(point, "research")
    if request is None or request.get("backend") != "shared-iterative":
        raise ValueError("E9 candidate has no research compile request")
    return mapping


def _candidate_id(point: dict[str, Any]) -> str:
    return "pool-" + sha256_bytes(canonical(point).encode())[:20]


def candidate_pool(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate the fixed 4 x 2 x 2 structural pool (never timing-ranked)."""
    result = []
    for groups in (1, 2, 3, 4):
        for threads in (128, 256):
            for layout in (("hermes-xor", "linear") if workload["operator"] == "ntt"
                           else ("writer-aligned", "linear")):
                point = _point(workload, groups, threads, layout)
                mapping = _validate_point(point)
                identifier = _candidate_id(point)
                logical_score = (
                    float(groups) * float(1 << int(workload["logN"])) * float(workload["batch"])
                    / float(threads)
                )
                result.append({
                    "candidate_id": identifier,
                    "core": "scalar",
                    "point": point,
                    "mapping": mapping,
                    "structural": {
                        "group_count": groups,
                        "threads": threads,
                        "layout": layout,
                        "logical_traffic_score": logical_score,
                    },
                    "legality": {
                        "structural": True,
                        "compile_request": "shared-iterative-research",
                        "runtime_lowering_verified": False,
                    },
                })
    if len(result) != 16 or len({item["candidate_id"] for item in result}) != 16:
        raise AssertionError("E9 fixed pool must contain 16 unique candidates")
    return result


def _pool_points(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate["point"] for group in groups for candidate in group["candidates"]]


def _gpu_command(*, python: str, checkpoint: Path, profile: Path, build_dir: Path,
                 points: Path, output: Path, target_uuid: str,
                 compile_policy: str = "research") -> list[str]:
    runner = ROOT / "scripts" / "validate_stage_service_profile.py"
    return [
        "env", f"CUDA_VISIBLE_DEVICES={target_uuid}",
        f"CUBUTTERFLY_COMPILE_MODE={compile_policy}", python, str(runner),
        "--checkpoint", str(checkpoint.resolve()), "--profile", str(profile.resolve()),
        "--build-dir", str(build_dir.resolve()), "--points", str(points.resolve()),
        "--output", str(output.resolve()),
    ]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    domain, domain_source = read_json(args.workloads)
    if isinstance(domain, dict):
        domain = domain.get("workloads", domain.get("points", []))
    if not isinstance(domain, list) or not all(isinstance(item, dict) for item in domain):
        raise ValueError("workloads input must contain a list")
    excluded = set()
    if args.exclude_protocol:
        excluded_protocol, _ = read_json(args.exclude_protocol)
        if (not isinstance(excluded_protocol, dict)
                or excluded_protocol.get("schema") != "cubutterfly-paper-heldout-protocol-v2"
                or (excluded_protocol.get("freeze") or {}).get("protocol_sha256")
                != heldout_protocol_hash(excluded_protocol)):
            raise ValueError("excluded holdout protocol schema or hash is invalid")
        excluded = set(excluded_protocol.get("workload_ids", []))
        if not excluded:
            excluded = {
                group.get("workload_id")
                for group in excluded_protocol.get("heldout_workloads", [])
                if isinstance(group, dict) and group.get("workload_id")
            }
    selected = select_workloads(domain, args.workload_id, excluded)
    checkpoint, checkpoint_source = read_json(args.stage_checkpoint)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "cubutterfly-stage-calibration-v1":
        raise ValueError("stage checkpoint has an unsupported schema")
    audit = audit_stage_checkpoint(checkpoint, checkpoint_source)
    identity = checkpoint.get("identity", {})
    target_uuid = args.target_uuid or _identity_value(identity, "uuid")
    if not target_uuid:
        raise ValueError("stage checkpoint has no target UUID; pass --target-uuid")
    checkpoint_uuid = _identity_value(identity, "uuid")
    if checkpoint_uuid and target_uuid != checkpoint_uuid:
        raise ValueError("target UUID differs from frozen stage checkpoint")
    workload_groups = []
    prior_predictions = []
    for workload in selected:
        identifier = workload_id(workload)
        candidates = candidate_pool(workload)
        workload_groups.append({
            "workload_id": identifier,
            "workload": workload,
            "core": "scalar",
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "candidates": candidates,
        })
        for candidate in candidates:
            prior_predictions.append({
                "workload_id": identifier,
                "candidate_id": candidate["candidate_id"],
                "predicted_ms": None,
                "stage_model_predicted_ms": None,
                "fallback_predicted_ms": None,
                "prediction_usable": False,
                "prediction_source": None,
                "fallback_score": candidate["structural"]["logical_traffic_score"],
                "source": "awaiting-describe-only-stage-profile",
                "uses_whole_workload_observation": False,
                "status": "pending-describe",
                "reason": (
                    "descriptor and frozen stage-service prediction are generated before any "
                    "whole-workload measurement; structural score is not a latency prediction"
                ),
            })
    audit = _audit_holdout_semantics(audit, checkpoint, workload_groups)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen-independent-pool",
        "selection_inputs": {
            "workload_domain": domain_source,
            "generator": _source_record(Path(__file__)),
            "excluded_workload_ids": sorted(excluded),
        },
        "selection_rule": (
            "Cartesian product of group counts 1..4, threads 128/256, and the two "
            "operator-specific writer layouts; order is structural and timing-independent."
        ),
        "oracle_scope": "measured finite pool only",
        "global_optimum_available": False,
        "workloads": workload_groups,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "search_pool_protocol.json"
    manifest_path = output_dir / "search_pool_manifest.json"
    points_path = output_dir / "search_pool_points.json"
    measurements_path = output_dir / "search_pool_measurements.json"
    replay_path = output_dir / "search_pool_replay.json"
    descriptors_path = output_dir / "search_pool_descriptors.json"
    predictions_path = output_dir / "search_pool_predictions.json"
    write_json(manifest_path, manifest)
    points = _pool_points(workload_groups)
    write_json(points_path, {"schema": POINTS_SCHEMA, "points": points})
    trials = int((checkpoint.get("protocol") or {}).get("trials", 3))
    document = {
        "schema": SCHEMA,
        "status": "frozen-awaiting-gpu-pool-timing",
        "experiment": "E9",
        "target": {
            "uuid": target_uuid,
            "device": identity.get("device"),
            "compute_capability": identity.get("compute_capability"),
            "global_memory_bytes": _identity_value(identity, "global_memory_bytes"),
            "compile_policy": identity.get("compile_policy"),
        },
        "budgets": list(BUDGETS),
        "random_seeds": list(SEEDS),
        "policies": list(POLICIES),
        "sources": {
            "workload_domain": domain_source,
            "stage_checkpoint": checkpoint_source,
            "capability_profile": _path_ref(Path(args.profile)),
            "build_dir": {"path": str(Path(args.build_dir).resolve())},
            "pool_manifest": _source_record(manifest_path),
            "generator": _source_record(Path(__file__)),
        },
        "training_inventory_audit": audit,
        "stage_training_workloads": audit["stage_record_workloads_by_role"],
        "stage_training_workload_semantics": audit["stage_training_workloads"],
        "stage_validation_workloads": audit["stage_validation_workloads"],
        "stage_record_workloads_by_role": audit["stage_record_workloads_by_role"],
        "training_workloads": [],
        "validation_workloads": [],
        "pool_manifest": str(manifest_path.resolve()),
        "pool_manifest_sha256": manifest["manifest_sha256"],
        "workload_ids": [workload_id(item) for item in selected],
        "independent_pool_selection": {
            "timing_read_during_selection": False,
            "selection_uses_stage_checkpoint_timings": False,
            "fixed_core": "scalar",
            "candidate_count_per_workload": 16,
            "oracle_scope": "measured finite pool only",
            "global_optimum_claim": False,
        },
        "workloads": workload_groups,
        "prior_predictions": prior_predictions,
        "prior_contract": {
            "published_before_pool_measurement": True,
            "predictions_frozen": False,
            "prediction_artifact_required_before_measurement": True,
            "uses_whole_workload_observation": False,
            "prediction_phase": "describe-only then CPU stage-service prediction",
            "service_gap_count": None,
            "structural_fallback_is_not_latency": True,
        },
        "gpu_measurement": {
            "points_path": str(points_path.resolve()),
            "output_path": str(measurements_path.resolve()),
            "pool_point_count": len(points),
            "trials_per_point": trials,
            "expected_full_workload_trials": len(points) * trials,
            "expected_describe_only_calls": len(points),
            "requires_gpu_supplement": True,
            "reason": "The fixed pool has no whole-workload timing in the frozen checkpoint.",
            "describe_command": [
                "env", f"CUDA_VISIBLE_DEVICES={target_uuid}", args.python,
                str(Path(__file__).resolve()), "--mode", "describe",
                "--protocol", str(protocol_path.resolve()), "--build-dir", str(args.build_dir.resolve()),
                "--output", str(descriptors_path.resolve()), "--target-uuid", target_uuid,
            ],
            "predict_command": [
                "env", "CUDA_VISIBLE_DEVICES=", args.python,
                str(Path(__file__).resolve()), "--mode", "predict",
                "--protocol", str(protocol_path.resolve()), "--descriptors", str(descriptors_path.resolve()),
                "--profile", str(args.profile.resolve()), "--stage-checkpoint", str(args.stage_checkpoint.resolve()),
                "--build-dir", str(args.build_dir.resolve()),
                "--output", str(predictions_path.resolve()),
            ],
            "command": _gpu_command(
                python=args.python, checkpoint=args.stage_checkpoint,
                profile=args.profile.resolve(), build_dir=args.build_dir.resolve(),
                points=points_path, output=measurements_path, target_uuid=target_uuid,
                compile_policy=_identity_value(identity, "compile_policy") or "research"),
            "replay_command": [
                "env", "CUDA_VISIBLE_DEVICES=", args.python,
                str(Path(__file__).resolve()), "--mode", "replay",
                "--protocol", str(protocol_path.resolve()), "--descriptors", str(descriptors_path.resolve()),
                "--predictions", str(predictions_path.resolve()), "--measurements", str(measurements_path.resolve()),
                "--profile", str(args.profile.resolve()), "--build-dir", str(args.build_dir.resolve()),
                "--output", str(replay_path.resolve()),
            ],
        },
        "rules": [
            "The finite candidate pool is frozen from structural legality before any pool timing is read.",
            "All policies use the same workload, core, pool, seeds, and total budgets.",
            "A failed seed or failed candidate consumes one attempt from the requested budget.",
            "Best-so-far regret is against the measured finite-pool oracle only; no global optimum is claimed.",
            "Offline replay does not establish online compile, calibration, or search wall time.",
        ],
    }
    document["freeze"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-separators-utf8; freeze field excluded",
        "protocol_sha256": protocol_hash(document),
    }
    write_json(protocol_path, document)
    return document


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("rows", "measurements", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    if isinstance(value, list):
        return value
    raise ValueError("measurement input must contain rows")


def _measurement_index(protocol: dict[str, Any], value: Any) -> dict[tuple[str, str], dict[str, Any]]:
    by_point = {}
    for group in protocol.get("workloads", []):
        for candidate in group.get("candidates", []):
            by_point[point_signature(candidate["point"])] = (group["workload_id"], candidate["candidate_id"])
    indexed = {}
    for row in _rows(value):
        if not isinstance(row, dict):
            raise ValueError("measurement report contains a non-object row")
        point = row.get("point")
        identity = row.get("candidate_id")
        if identity:
            found = next(((group["workload_id"], identity)
                          for group in protocol.get("workloads", [])
                          if identity in group.get("candidate_ids", [])), None)
        else:
            found = point_signature(point) if isinstance(point, dict) else None
            found = by_point.get(found) if found else None
        if found is None:
            continue
        if found in indexed:
            raise ValueError(f"measurement report repeats candidate {found[1]}")
        measured = _observed_ms(row)
        status = str(row.get("status", "measured"))
        correct = row.get("correct")
        if correct is False or correct in ("0", 0, "false"):
            measured = None
            status = "incorrect"
        indexed[found] = {"observed_ms": measured if status == "measured" else None, "status": status,
                          "row": row}
    return indexed


def _local_search_order(group: dict[str, Any], seed: int,
                        observations: dict[str, dict[str, Any]]) -> list[str]:
    """Build a local-search walk with an observation ledger.

    Only the selected candidate is added to ``visible`` after each step.  The
    complete pool is still available to the offline replay, but its future
    fitness values cannot influence the historical prefix of this walk.
    """
    candidates = list(group["candidates"])
    ids = [candidate["candidate_id"] for candidate in candidates]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    visible: dict[str, dict[str, Any]] = {}
    start = random.Random(seed).randrange(len(ids))
    order = [ids[start]]
    while len(order) < len(ids):
        selected = order[-1]
        if selected in observations:
            visible[selected] = observations[selected]
        successful = [cid for cid in order
                      if visible.get(cid, {}).get("observed_ms") is not None
                      and visible.get(cid, {}).get("status")
                      not in {"incorrect", "failed", "unavailable"}]
        center_id = (min(successful, key=lambda cid: (visible[cid]["observed_ms"], cid))
                     if successful else selected)
        current = by_id[center_id]["structural"]
        remaining = [cid for cid in ids if cid not in order]
        adjacent = [cid for cid in remaining if sum(
            by_id[cid]["structural"][key] != current[key]
            for key in ("group_count", "threads", "layout")) == 1]
        if not adjacent:
            adjacent = remaining
        order.append(min(adjacent, key=lambda cid: (
            by_id[cid]["structural"]["logical_traffic_score"], cid)))
    return order


def _candidate_order(group: dict[str, Any], policy: str, seed: int,
                     predictions: dict[str, dict[str, Any]], observed: dict[str, dict[str, Any]]) -> list[str]:
    candidates = list(group["candidates"])
    ids = [candidate["candidate_id"] for candidate in candidates]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if policy == "stage-service":
        def prediction_key(cid):
            item = predictions.get(cid, {})
            usable = item.get("prediction_usable", item.get("predicted_ms") is not None)
            predicted = item.get("predicted_ms") if usable else None
            fallback = item.get("fallback_predicted_ms")
            if fallback is None:
                fallback = item.get("fallback_score")
            if fallback is None:
                fallback = by_id[cid].get("fallback_score")
            if fallback is None:
                fallback = (by_id[cid].get("structural") or {}).get(
                    "logical_traffic_score", float("inf"))
            return predicted is None, predicted if predicted is not None else fallback, cid
        return sorted(ids, key=lambda cid: (
            *prediction_key(cid),))
    if policy == "logical-traffic":
        return sorted(ids, key=lambda cid: (
            by_id[cid]["structural"]["logical_traffic_score"], cid))
    if policy == "random":
        result = list(ids)
        random.Random(seed).shuffle(result)
        return result
    if policy != "local-search":
        raise ValueError(f"unknown policy: {policy}")
    return _local_search_order(group, seed, observed)


def _replay_cell(group: dict[str, Any], policy: str, seed: int, budget: int,
                 predictions: dict[str, dict[str, Any]],
                 observations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    workload = group["workload_id"]
    prediction_by_candidate = {cid: item for cid, item in predictions.items()
                               if item.get("workload_id") == workload}
    order = _candidate_order(group, policy, seed, prediction_by_candidate,
                             {cid: observations.get((workload, cid), {}) for cid in group["candidate_ids"]})
    attempts = []
    best = None
    best_so_far = []
    for index, candidate_id in enumerate(order[:budget]):
        item = observations.get((workload, candidate_id), {"observed_ms": None, "status": "missing"})
        value = item.get("observed_ms")
        success = value is not None and item.get("status") not in {"incorrect", "failed", "unavailable"}
        if success and (best is None or value < best["observed_ms"]):
            best = {"candidate_id": candidate_id, "observed_ms": value}
        attempts.append({
            "attempt_index": index,
            "candidate_id": candidate_id,
            "is_seed": index == 0,
            "status": item.get("status", "missing"),
            "success": success,
            "observed_ms": value,
        })
        best_so_far.append(dict(best) if best is not None else None)
    pool_values = [observations.get((workload, cid), {}).get("observed_ms")
                   for cid in group["candidate_ids"]]
    pool_values = [value for value in pool_values if value is not None]
    oracle = min(pool_values) if pool_values else None
    regret_ratio = (best["observed_ms"] / oracle) if best and oracle else None
    relative_regret = (regret_ratio - 1.0) if regret_ratio is not None else None
    return {
        "workload_id": workload,
        "policy": policy,
        "seed": seed,
        "budget": budget,
        "budget_includes_seed": True,
        "attempt_count": len(attempts),
        "failed_attempt_count": sum(not item["success"] for item in attempts),
        "failed_seed": bool(attempts and not attempts[0]["success"]),
        "attempts": attempts,
        "best_so_far": best_so_far,
        "best_candidate": best,
        "pool_oracle_ms": oracle,
        "regret_ratio": regret_ratio,
        "regret_ratio_definition": "best_observed_ms / measured_finite_pool_oracle_ms",
        "pool_regret": relative_regret,
        "relative_regret": relative_regret,
        "relative_regret_definition": "regret_ratio - 1.0",
        "latency_regret_ratio": regret_ratio,
        "oracle_scope": "measured finite pool only",
        "global_optimum_claim": False,
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_source = _load_protocol(args.protocol, SCHEMA)
    expected_hash = protocol["freeze"]["protocol_sha256"]
    manifest_path = Path(protocol["pool_manifest"])
    manifest, manifest_source = read_json(manifest_path)
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("search-pool manifest hash mismatch")
    if manifest.get("manifest_sha256") != protocol.get("pool_manifest_sha256"):
        raise ValueError("protocol and pool manifest identities differ")
    descriptor_path = getattr(args, "descriptors", None)
    prediction_path = getattr(args, "predictions", None)
    strict = descriptor_path is not None or prediction_path is not None
    if strict and (descriptor_path is None or prediction_path is None):
        raise ValueError("replay requires both --descriptors and --predictions")
    descriptor = predictions_document = None
    measurement_identity = {"strict_schema": False, "uuid_verified": False}
    if strict:
        descriptor, descriptor_source = _load_descriptor(descriptor_path, protocol, SCHEMA)
        predictions_document, predictions_source = _load_predictions(
            prediction_path, protocol, descriptor, descriptor_source)
        _validate_runtime_inputs(protocol, descriptor,
                                 profile_path=getattr(args, "profile", None),
                                 build_dir=getattr(args, "build_dir", None),
                                 predictions=predictions_document)
        measurements, observations, measurement_identity = _load_measurement_report(
            args.measurements, protocol, descriptor, strict=True)
        measurements_source = measurement_identity["source"]
    else:
        measurements, measurements_source = read_json(args.measurements)
        observations = _measurement_index(protocol, measurements)
    prediction_items = (predictions_document or {}).get("prior_predictions")
    if prediction_items is None:
        prediction_items = protocol.get("prior_predictions", [])
    predictions = {(item["workload_id"], item["candidate_id"]): item
                   for item in prediction_items}
    cells = []
    for group in protocol.get("workloads", []):
        group_predictions = {cid: predictions.get((group["workload_id"], cid), {})
                             for cid in group["candidate_ids"]}
        for policy in protocol.get("policies", POLICIES):
            for seed in protocol.get("random_seeds", SEEDS):
                for budget in protocol.get("budgets", BUDGETS):
                    cells.append(_replay_cell(group, policy, seed, budget,
                                              {cid: value for cid, value in group_predictions.items()}, observations))
    measured_count = sum(value.get("observed_ms") is not None for value in observations.values())
    failed_count = sum(value.get("observed_ms") is None for value in observations.values())
    regrets = [cell["relative_regret"] for cell in cells if cell["relative_regret"] is not None]
    regret_ratios = [cell["regret_ratio"] for cell in cells
                     if cell["regret_ratio"] is not None]
    result = {
        "schema": RESULT_SCHEMA,
        "status": "replayed-with-gaps" if failed_count else "replayed",
        "protocol": protocol_source,
        "pool_manifest": manifest_source,
        "measurements": measurements_source,
        "prediction_source": (predictions_source if strict else None),
        "measurement_identity": measurement_identity,
        "protocol_sha256": expected_hash,
        "pool_manifest_sha256": manifest["manifest_sha256"],
        "budgets": protocol.get("budgets", list(BUDGETS)),
        "policies": protocol.get("policies", list(POLICIES)),
        "random_seeds": protocol.get("random_seeds", list(SEEDS)),
        "cells": cells,
        "measurement_cost": {
            "pool_candidates": sum(len(group.get("candidate_ids", [])) for group in protocol.get("workloads", [])),
            "timed_rows": measured_count,
            "failed_or_missing_rows": failed_count,
            "compile_seconds": None,
            "calibration_seconds": None,
            "online_search_seconds": None,
            "online_cost_claim": False,
        },
        "summary": {
            "cell_count": len(cells),
            "pool_regret_median": statistics.median(regrets) if regrets else None,
            "pool_regret_p90": _percentile(regrets, .90),
            "relative_regret_median": statistics.median(regrets) if regrets else None,
            "relative_regret_p90": _percentile(regrets, .90),
            "regret_ratio_median": statistics.median(regret_ratios) if regret_ratios else None,
            "regret_ratio_p90": _percentile(regret_ratios, .90),
            "latency_regret_ratio_median": statistics.median(regret_ratios) if regret_ratios else None,
            "latency_regret_ratio_p90": _percentile(regret_ratios, .90),
            "regret_ratio_definition": "best_observed_ms / measured_finite_pool_oracle_ms",
            "relative_regret_definition": "regret_ratio - 1.0",
            "oracle_scope": "measured finite pool only",
            "global_optimum_claim": False,
            "budget_includes_failed_seed": all(cell["budget_includes_seed"] for cell in cells),
        },
        "rules_checked": {
            "protocol_hash": True,
            "pool_manifest_hash": True,
            "measurement_schema": measurement_identity.get("strict_schema", False),
            "measurement_uuid": measurement_identity.get("uuid_preflight_verified",
                                                           measurement_identity.get("uuid_verified", False)),
            "measurement_build": measurement_identity.get("build_verified", False),
            "measurement_protocol": measurement_identity.get("protocol_verified", False),
            "selection_precedes_timing": True,
            "prior_precedes_timing": True,
            "fixed_core": all(group.get("core") == "scalar" for group in protocol.get("workloads", [])),
            "oracle_is_not_global": True,
        },
    }
    write_json(args.output, result)
    return result


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "describe", "predict", "replay"), default="prepare")
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--stage-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--exclude-protocol", type=Path)
    parser.add_argument("--target-uuid")
    parser.add_argument("--require-covered", action="store_true",
                        help="predict: fail before pool measurement if any calibrated prior is missing")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--profile", type=Path,
                        help="target capability profile; no platform default is assumed")
    parser.add_argument("--build-dir", type=Path,
                        help="target build directory; no platform default is assumed")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if (not args.workloads or not args.stage_checkpoint or not args.output_dir
                or not args.profile or not args.build_dir):
            parser.error("prepare requires --workloads, --stage-checkpoint, --output-dir, --profile and --build-dir")
    elif args.mode == "describe":
        if not args.protocol or not args.build_dir or not args.output:
            parser.error("describe requires --protocol, --build-dir and --output")
    elif args.mode == "predict":
        if (not args.protocol or not args.descriptors or not args.profile
                or not args.build_dir or not args.output):
            parser.error("predict requires --protocol, --descriptors, --profile, --build-dir and --output")
    elif (not args.protocol or not args.descriptors or not args.predictions
          or not args.measurements or not args.profile or not args.build_dir or not args.output):
        parser.error("replay requires --protocol, --descriptors, --predictions, --measurements, --profile, --build-dir and --output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "prepare":
        result = freeze(args)
    elif args.mode == "describe":
        result = describe(args)
    elif args.mode == "predict":
        result = predict(args)
    else:
        result = replay(args)
    print(json.dumps({"status": result["status"], "schema": result["schema"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
