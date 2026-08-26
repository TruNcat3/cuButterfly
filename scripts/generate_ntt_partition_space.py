#!/usr/bin/env python3
"""Generate the stage-count and ordered-partition layer of the NTT space."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from generate_ntt_streaming_space import stage_partitions, task_counts


def publication_tokens(log_n: int, partition: tuple[int, ...], batch: int):
    prefix = 0
    tokens = 0
    for segment, stages in enumerate(partition[:-1]):
        next_remaining = log_n - prefix - stages - partition[segment + 1]
        tokens += batch << (prefix + next_remaining)
        prefix += stages
    return tokens


def physical_templates(partition: tuple[int, ...], cores: tuple[str, ...]):
    templates = []
    for core in cores:
        templates.append({"kind": "homogeneous", "cores": [core] * len(partition)})
    if len(partition) > 1 and "radix4" in cores and "radix8" in cores:
        templates.append({
            "kind": "tail-specialized",
            "cores": ["radix4"] * (len(partition) - 1) + ["radix8"],
        })
    return templates


def execution_templates(partition: tuple[int, ...], max_group_log: int = 10):
    """Enumerate legal materialization choices for one logical partition."""
    boundary_count = len(partition) - 1
    templates = [{
        "kind": "full-scratch",
        "boundary_storage": ["full-scratch"] * boundary_count,
        "execution_stage_partition": list(partition),
        "execution_group_count": len(partition),
        "materialized_boundary_count": boundary_count,
    }, {
        "kind": "ring",
        "boundary_storage": ["ring"] * boundary_count,
        "execution_stage_partition": list(partition),
        "execution_group_count": len(partition),
        "materialized_boundary_count": boundary_count,
    }]
    for fused_mask in range(1, 1 << boundary_count):
        groups = []
        storage = []
        stages = partition[0]
        valid = True
        for boundary in range(boundary_count):
            if fused_mask & (1 << boundary):
                stages += partition[boundary + 1]
                storage.append("resident-fused")
                if stages > max_group_log:
                    valid = False
                    break
            else:
                groups.append(stages)
                storage.append("full-scratch")
                stages = partition[boundary + 1]
        if not valid:
            continue
        groups.append(stages)
        # A fully resident transform needs a separate whole-transform core and
        # is intentionally outside the current streaming implementation.
        if len(groups) < 2 or max(groups) > max_group_log:
            continue
        templates.append({
            "kind": "resident-fused",
            "boundary_storage": storage,
            "execution_stage_partition": groups,
            "execution_group_count": len(groups),
            "materialized_boundary_count": len(groups) - 1,
        })
    templates.sort(key=lambda point: (
        point["execution_group_count"], point["execution_stage_partition"],
        point["kind"], point["boundary_storage"]))
    return templates


def make_partition_point(log_n: int, batch: int, partition: tuple[int, ...],
                         word_bits: int, preferred_logs: tuple[int, ...],
                         boundary_penalty: float, cores: tuple[str, ...]):
    mismatch = sum(min(abs(stages - preferred) for preferred in preferred_logs)
                   for stages in partition)
    mean = log_n / len(partition)
    imbalance = math.sqrt(sum((stages - mean) ** 2 for stages in partition) /
                          len(partition))
    # The final role also owns output permutation/finalization. This is only a
    # shortlist tie-breaker; measurements remain the ranking authority.
    tail_deficit = max(partition) - partition[-1]
    logical_boundaries = len(partition) - 1
    realizations = execution_templates(partition)
    minimum_groups = min(point["execution_group_count"] for point in realizations)
    minimum_boundaries = minimum_groups - 1
    minimum_realization = min(
        realizations,
        key=lambda point: (point["materialized_boundary_count"],
                           point["execution_stage_partition"]))
    n = 1 << log_n
    return {
        "segment_count": len(partition),
        "stage_partition": list(partition),
        "logical_boundary_count": logical_boundaries,
        "minimum_execution_group_count": minimum_groups,
        "minimum_materialized_boundary_count": minimum_boundaries,
        "full_scratch_value_passes": len(partition),
        "full_scratch_workspace_bytes": logical_boundaries * batch * n *
                                        (word_bits // 8),
        "minimum_full_scratch_workspace_bytes": minimum_boundaries * batch * n *
                                                 (word_bits // 8),
        "publication_tokens": publication_tokens(log_n, partition, batch),
        "minimum_publication_tokens": publication_tokens(
            log_n, tuple(minimum_realization["execution_stage_partition"]), batch),
        "task_counts": task_counts(log_n, partition, batch),
        "stage_min": min(partition),
        "stage_max": max(partition),
        "stage_stddev": imbalance,
        "default_cta_weights": list(partition),
        "physical_templates": physical_templates(partition, cores),
        "execution_templates": realizations,
        "static_score": (mismatch + boundary_penalty * minimum_boundaries +
                         0.25 * imbalance + 0.05 * tail_deficit),
    }


def generate_space(log_n: int, batch: int, word_bits: int, min_segments: int,
                   max_segments: int, min_stage_log: int, max_stage_log: int,
                   preferred_logs: tuple[int, ...], top_per_count: int,
                   boundary_penalty: float, cores: tuple[str, ...],
                   include_all: bool):
    points = []
    for partition in stage_partitions(log_n, min_stage_log, max_stage_log,
                                      min_segments, max_segments):
        if sum(partition) != log_n:
            raise AssertionError("partition generator violated the stage sum")
        points.append(make_partition_point(
            log_n, batch, partition, word_bits, preferred_logs,
            boundary_penalty, cores))
    points.sort(key=lambda point: (
        point["segment_count"], point["static_score"],
        point["minimum_full_scratch_workspace_bytes"],
        point["stage_partition"]))
    counts = Counter(point["segment_count"] for point in points)
    shortlist = []
    selected = Counter()
    for point in points:
        count = point["segment_count"]
        if selected[count] < top_per_count:
            shortlist.append(point)
            selected[count] += 1
    execution_candidates = {}
    for point in points:
        for realization in point["execution_templates"]:
            if realization["kind"] == "ring":
                continue
            key = (point["segment_count"],
                   realization["execution_group_count"])
            score = (point["static_score"] + boundary_penalty *
                     (realization["materialized_boundary_count"] -
                      point["minimum_materialized_boundary_count"]))
            candidate = {
                "segment_count": point["segment_count"],
                "stage_partition": point["stage_partition"],
                "default_cta_weights": point["default_cta_weights"],
                "execution_template": realization,
                "static_score": score,
            }
            previous = execution_candidates.get(key)
            ordering = (score, candidate["stage_partition"],
                        realization["execution_stage_partition"])
            if previous is None or ordering < previous[0]:
                execution_candidates[key] = (ordering, candidate)
    execution_shortlist = [execution_candidates[key][1]
                           for key in sorted(execution_candidates)]
    return {
        "schema": "cuntt-ntt-partition-space-v2",
        "shape": {"logN": log_n, "batch": batch, "word_bits": word_bits},
        "dimensions": {
            "segment_count": [min_segments, max_segments],
            "stage_log": [min_stage_log, max_stage_log],
            "constraint": "sum(stage_partition) == logN",
            "per_segment": ["core", "threads", "units", "data_space",
                            "data_time", "cta_weight"],
            "per_boundary": ["storage", "buffers", "readiness", "layout"],
            "execution_group_count": [2, max_segments],
            "lowering_constraint":
                "resident-fused groups adjacent logical subgraphs; each execution group has at most 10 stages",
        },
        "preferred_stage_logs": list(preferred_logs),
        "partition_count": len(points),
        "partition_count_by_segments": {str(key): counts[key]
                                         for key in sorted(counts)},
        "shortlist_policy": {
            "top_per_segment_count": top_per_count,
            "boundary_penalty": boundary_penalty,
            "note": "shortlist covers M; execution_shortlist covers every legal (M,G); event timing ranks candidates",
        },
        "shortlist": shortlist,
        "execution_shortlist": execution_shortlist,
        **({"partitions": points} if include_all else {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logN", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--word-bits", type=int, choices=(32, 64), default=32)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--min-stage-log", type=int, default=2)
    parser.add_argument("--max-stage-log", type=int, default=10)
    parser.add_argument("--preferred-stage-logs", default="6,8,10")
    parser.add_argument("--generic-cores", default="radix2,radix4,radix8")
    parser.add_argument("--top-per-count", type=int, default=4)
    parser.add_argument("--boundary-penalty", type=float, default=1.0)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.min_segments <= args.max_segments <= 8:
        raise ValueError("segment count must satisfy 2 <= min <= max <= 8")
    preferred = tuple(int(value) for value in args.preferred_stage_logs.split(','))
    cores = tuple(value for value in args.generic_cores.split(',') if value)
    if not preferred or not cores:
        raise ValueError("preferred stage logs and generic cores cannot be empty")
    document = generate_space(
        args.logN, args.batch, args.word_bits, args.min_segments,
        args.max_segments, args.min_stage_log, args.max_stage_log,
        preferred, args.top_per_count, args.boundary_penalty, cores,
        args.include_all)
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
