#!/usr/bin/env python3
"""Generate and validate architecture-level NTT subgraph streaming points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def stage_partitions(log_n: int, minimum: int = 5, maximum: int = 10,
                     min_segments: int = 2, max_segments: int = 4):
    def visit(remaining, count, prefix):
        if count == 0:
            if remaining == 0:
                yield tuple(prefix)
            return
        for stages in range(minimum, maximum + 1):
            if minimum * (count - 1) <= remaining - stages <= maximum * (count - 1):
                yield from visit(remaining - stages, count - 1, prefix + [stages])

    for count in range(min_segments, max_segments + 1):
        yield from visit(log_n, count, [])


def task_counts(log_n: int, partition: tuple[int, ...], batch: int = 1):
    prefix = 0
    counts = []
    for stages in partition:
        remaining = log_n - prefix - stages
        counts.append(batch << (prefix + remaining))
        prefix += stages
    return counts


def producer_task(log_n: int, partition: tuple[int, ...], consumer_segment: int,
                  consumer_task: int, element: int):
    prefix_before = sum(partition[:consumer_segment])
    stages = partition[consumer_segment]
    remaining = log_n - prefix_before - stages
    remaining_n = 1 << remaining
    prefix = consumer_task >> remaining
    remainder = consumer_task & (remaining_n - 1)
    logical = (prefix * (1 << stages) + element) * remaining_n + remainder

    producer = consumer_segment - 1
    producer_prefix = sum(partition[:producer])
    producer_stages = partition[producer]
    producer_remaining = log_n - producer_prefix - producer_stages
    producer_remaining_n = 1 << producer_remaining
    return ((logical >> (producer_stages + producer_remaining)) * producer_remaining_n +
            (logical & (producer_remaining_n - 1)))


def validate_dependency_graph(log_n: int, partition: tuple[int, ...]):
    if sum(partition) != log_n or len(partition) < 2:
        raise ValueError("partition must contain at least two segments and sum to logN")
    counts = task_counts(log_n, partition)
    for segment in range(1, len(partition)):
        producer_count = counts[segment - 1]
        for consumer in range(counts[segment]):
            dependencies = {
                producer_task(log_n, partition, segment, consumer, element)
                for element in range(1 << partition[segment])
            }
            if not dependencies or min(dependencies) < 0 or max(dependencies) >= producer_count:
                raise AssertionError("consumer dependency escapes the previous subgraph layer")
    return True


def make_point(log_n, batch, partition, threads, ctas_per_sm, word_bits,
               boundary_storage="full-scratch", boundary_buffers=1,
               core="dataflow-radix4", data_times=None, cta_weights=None,
               data_space=0, coefficient_reuse_stages=0):
    if "coefficient-reuse" in core and coefficient_reuse_stages == 0:
        coefficient_reuse_stages = 1 if "warp256" in core else 4
    counts = task_counts(log_n, partition, batch)
    points = batch << log_n
    slots = batch if boundary_storage == "full-scratch" else min(batch, boundary_buffers)
    boundary_words = (len(partition) - 1) * slots * (1 << log_n)
    if boundary_storage == "full-scratch":
        prefix = 0
        token_words = 0
        for segment, stages in enumerate(partition[:-1]):
            next_remaining = log_n - prefix - stages - partition[segment + 1]
            token_words += (1 << (prefix + next_remaining)) * slots
            prefix += stages
    else:
        token_words = sum((count // batch) * slots for count in counts[:-1])
    # All segments perform N*q work; q-proportional CTA weights equalize service time.
    score = sum(stages / max(stages, 1) for stages in partition)
    # The generic kernel assigns one independent subgraph to each warp. This
    # term distinguishes otherwise identical 128/256-thread candidates.
    score += 256 / threads
    score += token_words / points + max(1 << stages for stages in partition) / 4096
    if boundary_storage == "ring" and slots < batch:
        score += 1.0 / slots
    single_transform_wavefront = len(partition) >= 3
    if batch == 1 and not single_transform_wavefront:
        score += 2.0
    if cta_weights is None:
        if log_n == 20 and partition == (10, 10):
            cta_weights = (9, 11)
        elif log_n == 20 and partition == (7, 7, 6):
            cta_weights = (6, 6, 8)
        else:
            cta_weights = partition
    if data_times is None:
        data_times = (1,) * len(partition)
    if len(data_times) != len(partition) or len(cta_weights) != len(partition):
        raise ValueError("per-role data_time and CTA weights must match partition")
    generated_kernel = None
    resident_rows = None
    physical_units = None
    if log_n == 20 and partition == (10, 10) and boundary_storage == "full-scratch":
        if core == "homogeneous-radix4" and threads == 256:
            generated_kernel = "homogeneous-two-level-10x10"
            resident_rows = physical_units = 4
        elif core == "homogeneous-warp-radix2" and threads == 256:
            generated_kernel = "homogeneous-warp-10x10"
            physical_units = 8
        elif core == "homogeneous-warp256-radix2" and threads in (128, 256):
            generated_kernel = "homogeneous-warp256-10x10"
            physical_units = threads // 32
        elif (core == "homogeneous-warp256-static-radix2" and
              threads in (128, 256)):
            generated_kernel = "homogeneous-warp256-static-10x10"
            physical_units = threads // 32
        elif (core in ("homogeneous-warp256-static-io-radix2",
                       "homogeneous-warp256-coefficient-reuse-static-io-radix2") and
              threads in ((128,) if "coefficient-reuse" in core else (128, 256)) and
              ("coefficient-reuse" not in core or
               (word_bits == 32 and data_space == 4))):
            suffix = "-coefficient-reuse" if "coefficient-reuse" in core else ""
            generated_kernel = f"homogeneous-warp256{suffix}-static-io-10x10"
            physical_units = threads // 32
        elif (core in ("homogeneous-warp128-static-io-radix2",
                       "homogeneous-warp128-coefficient-reuse-static-io-radix2",
                       "homogeneous-warp128-vector-radix4-static-io",
                       "homogeneous-warp128-vector-radix4-packed-stage6-static-io",
                       "homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io",
                       "homogeneous-warp128-packet-shared-radix4-static-io",
                       "homogeneous-warp64-static-io-radix2") and
              threads in ((128, 256) if "warp128" in core else (128,)) and
              word_bits == 32 and data_space == 4):
            tile = 128 if "warp128" in core else 64
            suffix = ("-coefficient-reuse" if "coefficient-reuse" in core
                      else "-packet-shared-radix4"
                      if "packet-shared-radix4" in core
                      else "-vector-radix4-packed-stage6-distributed"
                      if "packed-stage6-distributed" in core
                      else "-vector-radix4-packed-stage6"
                      if "packed-stage6" in core
                      else "-vector-radix4" if "vector-radix4" in core else "")
            generated_kernel = f"homogeneous-warp{tile}{suffix}-static-io-10x10"
            physical_units = threads // 32
        elif (core == "homogeneous-warp128-pipeline-static-io-radix2" and
              threads == 128 and word_bits == 32 and data_space == 4):
            generated_kernel = "homogeneous-warp128-pipeline-static-io-10x10"
            physical_units = 4
        elif (core == "homogeneous-warp128-cooperative-static-io-radix2" and
              threads == 128 and word_bits == 32 and data_space == 4):
            generated_kernel = "homogeneous-warp128-cooperative-static-io-10x10"
            physical_units = 4
        elif core == "dataflow-radix4" and threads == 256:
            generated_kernel = "resident-10x10"
            resident_rows = physical_units = 4
    elif (log_n == 20 and
          partition in ((6, 7, 7), (7, 6, 7), (7, 7, 6),
                        (6, 6, 8), (6, 8, 6), (8, 6, 6)) and
          threads == 256 and boundary_storage == "full-scratch"):
        generated_kernel = ("resident-7x7x6-wave" if partition == (7, 7, 6)
                            else "resident-three-level-wave")
        resident_rows = physical_units = 32
    return {
        "stage_partition": list(partition),
        "subgraph_mappings": [
            {"core": core, "threads": threads,
             "units_per_cta": physical_units or 1,
             "data_space": data_space, "data_time": data_time,
             "coefficient_reuse_stages": coefficient_reuse_stages,
             "cta_weight": weight}
            for weight, data_time in zip(cta_weights, data_times)
        ],
        "boundaries": [{"storage": boundary_storage, "buffers": boundary_buffers}
                       for _ in partition[:-1]],
        "target_ctas_per_sm": ctas_per_sm,
        "pipeline_buffers": 1,
        "task_counts": counts,
        "publication_tokens": token_words,
        "single_transform_wavefront": single_transform_wavefront,
        "generated_kernel": generated_kernel,
        "compiled": generated_kernel is not None,
        "resident_rows": resident_rows,
        "physical_units_per_cta": physical_units,
        "workspace_bytes": (boundary_words * (word_bits // 8) + token_words * 4 +
                            ((len(partition) - 1) * slots * 12
                             if boundary_storage == "ring" and slots < batch else 0)),
        "model_score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logN", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--word-bits", type=int, choices=(32, 64), default=64)
    parser.add_argument("--threads", default="128,256")
    parser.add_argument("--ctas-per-sm", default="2,3")
    parser.add_argument("--boundary-storage", default="full-scratch,ring")
    parser.add_argument("--boundary-buffers", default="1,2,3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-dag", action="store_true")
    parser.add_argument("--homogeneous-two-level", action="store_true",
                        help="emit two-role homogeneous physical-core instances")
    parser.add_argument(
        "--homogeneous-cores",
        default="homogeneous-radix4,homogeneous-warp-radix2,homogeneous-warp256-radix2,homogeneous-warp256-static-radix2,homogeneous-warp256-static-io-radix2,homogeneous-warp256-coefficient-reuse-static-io-radix2,homogeneous-warp128-static-io-radix2,homogeneous-warp128-coefficient-reuse-static-io-radix2,homogeneous-warp128-vector-radix4-static-io,homogeneous-warp128-vector-radix4-packed-stage6-static-io,homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io,homogeneous-warp64-static-io-radix2,homogeneous-warp128-pipeline-static-io-radix2,homogeneous-warp128-cooperative-static-io-radix2",
        help="comma-separated physical cores for the homogeneous template")
    parser.add_argument("--producer-data-time", default="1")
    parser.add_argument("--consumer-data-time", default="1")
    parser.add_argument("--row-packets", default="0",
                        help="row-packet data-space values for static warp cores")
    parser.add_argument("--coefficient-reuse-stages", default="1,2,3,4,5,6,7",
                        help="low-stage reuse depths for coefficient-reuse cores")
    parser.add_argument("--cta-weight-pairs", default="9:11",
                        help="comma-separated producer:consumer role weights")
    args = parser.parse_args()

    points = []
    for partition in stage_partitions(args.logN):
        if args.homogeneous_two_level and len(partition) != 2:
            continue
        if args.validate_dag:
            validate_dependency_graph(args.logN, partition)
        for threads in map(int, args.threads.split(',')):
            for ctas in map(int, args.ctas_per_sm.split(',')):
                for storage in args.boundary_storage.split(','):
                    if args.homogeneous_two_level and storage != "full-scratch":
                        continue
                    buffers = (1,) if storage == "full-scratch" else tuple(
                        map(int, args.boundary_buffers.split(',')))
                    for depth in buffers:
                        if args.homogeneous_two_level:
                            for producer_dt in map(int, args.producer_data_time.split(',')):
                                for consumer_dt in map(int, args.consumer_data_time.split(',')):
                                    for pair in args.cta_weight_pairs.split(','):
                                        producer_weight, consumer_weight = map(int, pair.split(':'))
                                        for core in args.homogeneous_cores.split(','):
                                            packets = (map(int, args.row_packets.split(','))
                                                       if "static" in core else (0,))
                                            for packet in packets:
                                                configured_reuse_depths = tuple(
                                                    map(int, args.coefficient_reuse_stages.split(',')))
                                                if "coefficient-reuse" in core:
                                                    reuse_depths = tuple(
                                                        value for value in configured_reuse_depths
                                                        if 1 <= value <= 6)
                                                elif "packed-stage6" in core:
                                                    reuse_depths = (6,)
                                                elif "vector-radix4" in core:
                                                    reuse_depths = ((0,) + tuple(
                                                        value for value in configured_reuse_depths
                                                        if 3 <= value <= 7))
                                                else:
                                                    reuse_depths = (0,)
                                                for reuse_depth in reuse_depths:
                                                    points.append(make_point(
                                                        args.logN, args.batch, partition, threads, ctas,
                                                        args.word_bits, storage, depth, core,
                                                        (producer_dt, consumer_dt),
                                                        (producer_weight, consumer_weight), packet,
                                                        reuse_depth))
                        else:
                            points.append(make_point(
                                args.logN, args.batch, partition, threads, ctas,
                                args.word_bits, storage, depth))
    points.sort(key=lambda point: (not point["compiled"], point["model_score"],
                                   point["workspace_bytes"]))
    document = {
        "schema": ("cuntt-ntt-homogeneous-two-level-space-v1"
                   if args.homogeneous_two_level
                   else "cuntt-ntt-streaming-space-v1"),
        "shape": {"logN": args.logN, "batch": args.batch, "word_bits": args.word_bits},
        "template": ({
            "logical_axes": ["stage_space", "stage_time", "data_space", "data_time"],
            "physical_cores": args.homogeneous_cores.split(','),
            "compiled_instance": "10+10/full-scratch",
        } if args.homogeneous_two_level else None),
        "recommended": points[0] if points else None,
        "points": points,
    }
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
