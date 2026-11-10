"""Static work geometry and an allocated-lane heuristic for shared lowerings.

Utilization divides useful work by the capacity of all allocated CTA lanes
over each leaf's rounds. Inactive warps can skip the arithmetic loop body, so
that capacity is not issued arithmetic work. The hardware-rate prior does not
model resident CTA limits and can overestimate the penalty for unused lanes.
The data space/time pair is an envelope over leaves.
"""


def shared_work_geometry(stage_count, threads, local_stage_partition=(), data_tiles_per_cta=1):
    stages, threads = int(stage_count), int(threads)
    tiles = int(data_tiles_per_cta)
    if tiles != data_tiles_per_cta or tiles < 1 or tiles & (tiles - 1):
        raise ValueError("shared arithmetic geometry needs a positive power-of-two tile count")
    if tiles > 1 and local_stage_partition:
        raise ValueError("multi-tile shared arithmetic does not support resident local partitions")
    parts = list(local_stage_partition or [1] * stages)
    if stages < 1 or threads < 1 or any(type(r) is not int or r < 1 for r in parts) or sum(parts) != stages:
        raise ValueError("shared arithmetic geometry needs positive threads and a complete stage partition")
    leaves = []
    first = 0
    for extent in parts:
        work = tiles * (1 << (stages - extent))
        rounds = (work + threads - 1) // threads
        butterflies = extent * (1 << (extent - 1))
        leaves.append(dict(first_stage=first, stage_count=extent, work_items=work,
                           rounds=rounds, active_threads=min(threads, work),
                           butterflies_per_work_item=butterflies))
        first += extent
    useful = tiles * stages * (1 << (stages - 1))
    allocated_slots = sum(threads * leaf["rounds"] * leaf["butterflies_per_work_item"] for leaf in leaves)
    return dict(leaves=leaves, data_tiles_per_cta=tiles, useful_butterflies=useful, allocated_lane_butterfly_slots=allocated_slots,
                allocated_lane_utilization=useful / allocated_slots,
                data_space=max(leaf["active_threads"] for leaf in leaves),
                data_time=max(leaf["rounds"] for leaf in leaves),
                data_shape_semantics="leaf-envelope", source="static-shared-allocated-lane-heuristic",
                resident_cta_limits_modeled=False,
                limitation="Allocated slots are not issued arithmetic; ignoring resident CTA limits can overestimate the unused-lane penalty.")
