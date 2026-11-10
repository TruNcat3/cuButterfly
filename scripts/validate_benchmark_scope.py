#!/usr/bin/env python3
"""Validate a declarative benchmark scope without touching GPU state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REQUIRED_AXES = {
    "operators", "precisions", "directions", "ranks", "orders", "layouts",
    "placements", "normalization", "batch_policy", "size_policy",
}
REQUIRED_CASE_IDENTITY = {
    "operator", "precision", "accumulation", "shape", "rank", "batch",
    "direction", "normalization", "input_order", "output_order", "placement",
    "layout", "strides",
}
VALID_STATUS = {"planned", "in-progress", "complete", "complete-supplement"}
VALID_KINDS = {"operator_comparison", "mechanism_profile", "end_to_end"}


def _load(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON scope {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("scope root must be an object")
    return value


def validate(path: Path, repo: Path | None = None, *, source_only: bool = False) -> dict:
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    scope = _load(path)
    if scope.get("schema") != "cubutterfly-benchmark-scope-v1":
        raise ValueError("unsupported scope schema")
    if not isinstance(scope.get("scope_id"), str) or not scope["scope_id"]:
        raise ValueError("scope_id must be a non-empty string")
    if scope.get("status") not in {"active-research", "archived"}:
        raise ValueError("scope status must be active-research or archived")
    hardware = scope.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("primary_target"):
        raise ValueError("hardware.primary_target is required")
    axes = scope.get("semantic_axes")
    if not isinstance(axes, dict) or not REQUIRED_AXES <= axes.keys():
        raise ValueError("semantic_axes is missing required axes")
    contract = scope.get("common_contract")
    if not isinstance(contract, dict):
        raise ValueError("common_contract is required")
    if not REQUIRED_CASE_IDENTITY <= set(contract.get("case_identity", [])):
        raise ValueError("case_identity must preserve every semantic identity axis")
    timing = contract.get("timing", {})
    protocols = timing.get("protocols")
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("timing.protocols must be a non-empty object")
    for protocol_id, protocol in protocols.items():
        if not isinstance(protocol, dict) or not isinstance(protocol.get("measurement"), str):
            raise ValueError(f"timing protocol is invalid: {protocol_id}")
        for field in ("warmup_calls", "graph_replays", "calls_per_graph", "repetitions"):
            if field in protocol and (not isinstance(protocol[field], int) or protocol[field] < 0):
                raise ValueError(f"timing.{protocol_id}.{field} must be a nonnegative integer")
    tracks = scope.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("tracks must be a non-empty list")
    ids = set()
    checked_refs = 0
    unchecked_results = []
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("each track must be an object")
        track_id = track.get("track_id")
        if not isinstance(track_id, str) or not track_id or track_id in ids:
            raise ValueError(f"track_id is missing or duplicated: {track_id!r}")
        ids.add(track_id)
        if track.get("kind") not in VALID_KINDS:
            raise ValueError(f"unsupported track kind for {track_id}")
        if track.get("status") not in VALID_STATUS:
            raise ValueError(f"unsupported status for {track_id}")
        protocol_ids = track.get("timing_protocol_ids", [])
        protocol_id = track.get("timing_protocol_id")
        if protocol_id is not None:
            protocol_ids = [protocol_id]
        if not protocol_ids or not all(item in protocols for item in protocol_ids):
            raise ValueError(f"timing protocol reference is missing or unknown for {track_id}")
        count = track.get("population_count")
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError(f"population_count must be nonnegative or null for {track_id}")
        ref = track.get("population_ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"population_ref is required for {track_id}")
        target = ref.split("#", 1)[0]
        if not (repo / target).exists():
            raise ValueError(f"population_ref does not exist for {track_id}: {target}")
        expected_hash = track.get("population_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"population_sha256 is required for {track_id}")
        actual_hash = hashlib.sha256((repo / target).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"population_sha256 mismatch for {track_id}: {target}")
        checked_refs += 1
        result = track.get("result_ref")
        if result is not None:
            if not isinstance(result, str) or not result:
                raise ValueError(f"result_ref must be a non-empty string or null for {track_id}")
            if source_only:
                unchecked_results.append({"track_id": track_id, "result_ref": result})
                continue
            target = result.split("#", 1)[0]
            if not (repo / target).exists():
                parent = (repo / target).parent
                if track.get("status") not in {"in-progress", "planned"} or not parent.exists():
                    raise ValueError(f"result_ref does not exist for {track_id}: {target}")
    summary = {"scope_id": scope["scope_id"], "status": scope["status"],
               "track_count": len(tracks), "checked_population_refs": checked_refs}
    if source_only:
        summary.update(validation_scope="source-only", unchecked_result_refs=unchecked_results,
                       result_evidence_verified=False, completion_verified=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    parser.add_argument("--source-only", action="store_true",
                        help="Validate source contracts and population hashes only; do not check result evidence or certify completion")
    args = parser.parse_args(argv)
    try:
        result = validate(args.scope.resolve(), source_only=args.source_only)
    except ValueError as exc:
        print(f"benchmark scope invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
