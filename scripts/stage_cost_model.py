#!/usr/bin/env python3
"""Positive, compositional costs for physical execution groups.

Whole-plan observations identify sums of startup/tail and service costs; they
do not measure isolated stage latencies. Keep that distinction in the artifact.
No coefficients, winning partitions or device names are built into the model.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
from functools import lru_cache
from schedule_cost_model import costs as scheduling_costs

try:
    import resident_mapping
except ImportError:  # pragma: no cover - package import fallback
    from . import resident_mapping

LEGACY_VERSION = "physical-stage-composition-v3"
VERSION = "physical-stage-composition-v6"
PARAMETERS = ("startup_and_tail", "service")


def number(row, key, default=0):
    value = row.get(key, default)
    return float(value) if value not in (None, "") else float(default)


def decoded(value, default):
    return json.loads(value) if isinstance(value, str) else (value if value is not None else default)


def _prefix_codelet_lanes(sample, mapping, group, group_index):
    """Return a non-default cooperative prefix lane identity, if applicable."""
    backend = mapping.get("backend", sample.get("backend"))
    core = group.get("core", sample.get("fft_core", mapping.get("fft_core")))
    if backend != "online-reorder" or core != "register-tile" or group_index != 0:
        return None
    value = mapping.get("prefix_codelet_lanes", sample.get("prefix_codelet_lanes", 1))
    if isinstance(value, bool):
        return value
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    return value if value != 1 else None


def primitive_key(sample, group):
    # Workload size and number of stages are independent variables of a cost
    # function, not a lookup key for complete kernels. Semantic contracts and
    # implementation choices must not be pooled across incompatible cores.
    mapping = decoded(sample.get("mapping_json"), {})
    if not isinstance(mapping, dict):
        mapping = {}
    try:
        group_index = int(group.get("index", 0))
    except (TypeError, ValueError):
        group_index = 0
    codelet = group.get("codelet")
    if codelet in (None, ""):
        codelet = (mapping.get("prefix_codelet", sample.get("prefix_codelet", "native"))
                   if group_index == 0 and mapping.get("backend", sample.get("backend")) == "online-reorder"
                   else "native")
    policy = group.get("io_policy")
    if policy in (None, ""):
        policies = mapping.get("factor_io_policies", sample.get("factor_io_policies", []))
        policy = policies[group_index] if (mapping.get("backend", sample.get("backend")) == "factor-streamed"
                                           and isinstance(policies, list) and group_index < len(policies)) else "dynamic"
    exchange_chunk = group.get("exchange_chunk")
    local_partition = group.get("local_stage_partition")
    if exchange_chunk in (None, ""):
        exchange_chunk = resident_mapping.group_axis(mapping, "exchange_chunk", group_index, 0)
    if local_partition in (None, ""):
        local_partition = resident_mapping.group_axis(mapping, "local_stage_partition", group_index, [])
    key = [sample.get("operator", "fft"), sample.get("precision", "fp32"),
        sample.get("accumulation", "native"), group.get("core", sample.get("fft_core", "scalar")),
        sample.get("compute_unit", "auto"), group.get("shared_layout", sample.get("shared_layout", "linear")),
        # These are physical lowering identities.  They are group-local so a
        # prefix-only strategy change does not invalidate an unchanged suffix.
        codelet,
        policy,
        # Local exchange is a physical mechanism, even when the old model did
        # not expose it in its primitive key.  Keep the v3 path available, but
        # do not pool warp-register FWHT with shared-memory FWHT in v4.
        sample.get("local_exchange", group.get("local_exchange", "unknown")),
        group.get("exchange", sample.get("exchange", "unknown")),
        ["exchange_chunk", exchange_chunk],
        ["local_stage_partition", local_partition]]
    lanes = _prefix_codelet_lanes(sample, mapping, group, group_index)
    if lanes is not None:
        key.append(["prefix_codelet_lanes", lanes])
    return json.dumps(key, separators=(",", ":"))


def stage_terms(sample, profile):
    """Return hardware-rate priors in ms per physical launch, plus provenance."""
    groups = decoded(sample.get("execution_groups_json"), [])
    if not groups:
        raise ValueError("physical execution groups are required for staged costs")
    caps = profile.get("capabilities", {})
    required = ("global_feedback_bytes_per_second", "equivalent_butterflies_per_second",
                "interstage_shared_bytes_per_second", "cta_barriers_per_second")
    if any(number(caps, key) <= 0 for key in required):
        raise ValueError("staged costs require a positive measured hardware capability profile")
    sm_count = max(1, number(profile, "sm_count", number(profile.get("hardware", {}), "sm_count", 1)))
    width = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8, "word32": 4, "word64": 8}.get(sample.get("precision"), 4)
    if sample.get("operator", "fft") == "fft":
        width *= 2
    n = number(sample, "N", 2 ** int(number(sample, "logN")))
    batch = number(sample, "batch", 1)
    tile = min(batch, max(1, number(sample, "batch_tile_count", 1))) if number(sample, "stage_overlap") else batch
    terms = []
    mapping = decoded(sample.get("mapping_json"), {})
    if not isinstance(mapping, dict):
        mapping = {}
    for group_index, group in enumerate(groups):
        launches = max(1, int(number(group, "launch_count",
                                      number(group, "kernel_launch_count", 1))))
        points = n * tile / launches
        stages = max(1, number(group, "stage_count", 1))
        grid = max(1, number(group, "grid_ctas",
                             number(group, "work_blocks", number(group, "work", 1))))
        threads = max(1, number(group, "threads", number(sample, "tile_threads", 128)))
        live = number(group, "live_shared_bytes")
        regs = number(group, "compiler_registers_per_thread")
        # Active CTA limits are conservative when compiler resources are not
        # available. Such predictions retain resource provenance below.
        resident = 1
        # Hardware-specific resource limits override the portable projection.
        hw = profile.get("hardware", {})
        if hw:
            limits = [int(hw["max_blocks_per_sm"]), int(hw["max_threads_per_sm"] // threads)]
            if live:
                limits.append(int(hw["shared_bytes_per_sm"] // live))
            if regs:
                limits.append(int(hw["registers_per_sm"] // (threads * regs)))
            resident = max(1, min(limits))
        memory = 1000 * 2 * points * width / number(caps, required[0])
        compute = 1000 * points * stages / 2 / number(caps, required[1])
        interstage_boundaries = max(0, stages - 1)
        local_partition = group.get("local_stage_partition")
        if local_partition in (None, ""):
            local_partition = resident_mapping.group_axis(
                mapping, "local_stage_partition", group_index, [])
        if (group.get("codelet") == "native-register-subgraph" and
                isinstance(local_partition, (list, tuple)) and local_partition):
            # A resident leaf fuses its internal stages in registers.  Only
            # boundaries between leaves require the shared ping-pong publish.
            interstage_boundaries = max(0, len(local_partition) - 1)
        exchange_chunk = group.get("exchange_chunk")
        if exchange_chunk in (None, ""):
            exchange_chunk = resident_mapping.group_axis(
                mapping, "exchange_chunk", group_index, 0)
        chunk = number({"value": exchange_chunk}, "value")
        ept = number(group, "elements_per_thread",
                     number(group, "EPT", number(group, "ept", 0)))
        extra_exchange_barriers = 0
        if chunk > 0 and ept > 0:
            rounds = math.ceil(ept / chunk)
            # Whole-tile exchange has one publish barrier.  Chunked exchange
            # adds one publish and one handoff barrier for every later round.
            extra_exchange_barriers = max(0, 2 * (rounds - 1))
        shared = (1000 * 2 * points * width * interstage_boundaries /
                  number(caps, required[2])) if live else 0
        barrier_count = interstage_boundaries + extra_exchange_barriers
        barriers = (1000 * grid * barrier_count / number(caps, required[3])) if live else 0
        # Independent launch/CTA probes supply this shared hardware term. Old
        # capability files retain explicitly marked priors until upgraded.
        schedule = scheduling_costs(profile, threads, grid, sm_count)
        dispatch = schedule["dispatch_ms"]
        body = max(memory, compute, shared, barriers, dispatch, 1e-9)
        terms.append(dict(key=primitive_key(sample, group), startup_and_tail=schedule["startup_ms"], scheduling=schedule,
            service=body, memory_ms=memory, compute_ms=max(compute, shared, barriers),
            occupancy=min(1.0, grid / (sm_count * resident)), launch_count=launches,
            stage_count=stages, grid_ctas=grid, points=points,
            interstage_boundaries=interstage_boundaries,
            exchange_barriers=extra_exchange_barriers,
            resource_source="compiler" if group.get("compiler_resources_known") else "projected",
            descriptor_source=sample.get("descriptor_source", "measured-plan")))
    return terms


def schedule_nodes(stages, tiles=1, factor=False, overlap=False, last_fraction=1.0, tail_stages=None):
    """The stream, ring-slot and fan-in dependencies of the current runtime."""
    nodes = []
    ids = {}
    order = [(g, k) for k in range(tiles) for g in range(len(stages) - (1 if factor else 0))]
    if factor:
        order.append((len(stages) - 1, 0))
    for g, k in order:
        deps = []
        if k:
            deps.append(ids[g, k - 1])
        if g:
            deps.append(ids[g - 1, tiles - 1 if factor and g == len(stages) - 1 else k])
        if not factor and k >= 2 and g + 1 < len(stages):
            deps.append(ids[g + 1, k - 2])
        if not overlap and nodes:
            deps.append(len(nodes) - 1)
        stage = stages[g]
        fraction = last_fraction if not factor and k == tiles - 1 else 1.0
        if tail_stages is not None and not factor and k == tiles - 1:
            stage = tail_stages[g]
            fraction = 1.0
        duration = stage["overhead_ms"] + stage["body_ms"] * fraction
        ids[g, k] = len(nodes)
        nodes.append(dict(deps=set(deps), duration=duration, group=g, tile=k,
            sm=min(1.0, max(stage.get("sm_demand", 1), 0.0)),
            hbm=min(1.0, max(stage.get("hbm_demand", 1), 0.0)),
            dispatch=min(1.0, stage["overhead_ms"] / duration)))
    return nodes


def compose(nodes):
    """Event simulation with shared SM, HBM and dispatch capacity.

    Concurrent kernels cannot each consume the full GPU. Fractional demands
    are estimates, not an assertion of measured concurrency or ideal hiding.
    """
    dependents = collections.defaultdict(list)
    pending = []
    for i, node in enumerate(nodes):
        pending.append(len(node["deps"]))
        for dependency in node["deps"]:
            dependents[dependency].append(i)
    active = {i: n["duration"] for i, n in enumerate(nodes) if not pending[i]}
    elapsed, completed = 0.0, 0
    while active:
        pressure = max(1.0, *(sum(nodes[i][axis] for i in active) for axis in ("sm", "hbm", "dispatch")))
        work = min(active.values())
        elapsed += work * pressure
        done = [i for i in active if active[i] <= work + 1e-12]
        active = {i: remaining - work for i, remaining in active.items() if i not in done}
        for i in done:
            completed += 1
            for consumer in dependents[i]:
                pending[consumer] -= 1
                if not pending[consumer]:
                    active[consumer] = nodes[consumer]["duration"]
    if completed != len(nodes):
        raise ValueError("cyclic stage schedule")
    return elapsed


_SCHEDULE_FIELDS = ("overhead_ms", "body_ms", "sm_demand", "hbm_demand")


def _schedule_signature(stages):
    # Include every stage field read by schedule_nodes, preserving the original
    # numeric values and defaults. Compiler/curve provenance is not a schedule
    # input and remains attached to the caller's fresh prediction details.
    return tuple((stage["overhead_ms"], stage["body_ms"],
                  stage.get("sm_demand", 1), stage.get("hbm_demand", 1)) for stage in stages)


@lru_cache(maxsize=512)
def _compose_stages_cached(stages, tiles, factor, overlap, last_fraction, tail_stages):
    full = [dict(zip(_SCHEDULE_FIELDS, stage)) for stage in stages]
    tail = None if tail_stages is None else [dict(zip(_SCHEDULE_FIELDS, stage)) for stage in tail_stages]
    return compose(schedule_nodes(full, tiles, factor, overlap, last_fraction, tail))


def compose_stages(stages, tiles=1, factor=False, overlap=False, last_fraction=1.0, tail_stages=None):
    """Reuse an identical simulation without retaining large node graphs.

    Cache keys are immutable values, so changing calibration, resource demands,
    remainder service or scheduling choices cannot reuse the previous result.
    The original event simulation and its arithmetic order are unchanged.
    """
    return _compose_stages_cached(_schedule_signature(stages), tiles, factor, overlap,
                                  last_fraction, None if tail_stages is None else _schedule_signature(tail_stages))


def fit(rows, profile, ridge=0.01, *, legacy=False, compatibility=None):
    """Fit nonnegative per-primitive overhead/service factors in linear time.

    Only serial plans identify these additive factors. Overlapped plan totals
    are reserved for validation until separate concurrency calibration exists.
    """
    if compatibility in ("legacy", "v3", LEGACY_VERSION):
        legacy = True
    service_profile = profile.get("stage_service") if isinstance(profile, dict) else None
    measured_service = (isinstance(service_profile, dict) and bool(service_profile.get("curves"))
                        and not legacy)
    features, targets, coverage = [], [], {}
    for row in rows:
        sample = row["sample"]
        for field in ("device", "compute_capability", "global_memory_bytes"):
            if sample.get(field) not in (None, "") and profile.get(field) not in (None, ""):
                if str(sample[field]) != str(profile[field]):
                    raise ValueError(f"stage calibration hardware identity mismatch: {field}")
        target = float(row["median_kernel_ms"])
        if not math.isfinite(target) or target <= 0:
            continue
        if number(sample, "stage_overlap") or number(sample, "factor_overlap"):
            continue
        try:
            terms = stage_terms(sample, profile)
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            if not measured_service:
                raise
            # A measured service profile is authoritative for duration.  The
            # capability-rate terms are retained only as optional fallback and
            # resource hints, so a cache without legacy rates remains usable.
            terms = _minimal_terms(sample, profile)
        feature = collections.defaultdict(float)
        for term in terms:
            for parameter in PARAMETERS:
                feature[(term["key"], parameter)] += term[parameter] * term["launch_count"]
            entry = coverage.setdefault(term["key"], dict(observations=0, stages=[], grids=[], points=[]))
            entry["observations"] += 1
            for plural, field in (("stages", "stage_count"), ("grids", "grid_ctas"), ("points", "points")):
                entry[plural].append(term[field])
        features.append(feature)
        targets.append(target)
    axes = sorted({key for row in features for key in row})
    # Relative-error weighting prevents one 30ms broken mapping from making
    # all sub-millisecond mappings irrelevant. Ridge shrinks toward priors.
    columns = [[row.get(axis, 0) / target for row, target in zip(features, targets)] for axis in axes]
    coefficients = [1.0] * len(axes)
    estimates = [sum(column[i] for column in columns) for i in range(len(targets))]
    for _ in range(120):
        largest = 0.0
        for j, column in enumerate(columns):
            old = coefficients[j]
            denominator = sum(x * x for x in column) + ridge
            new = max(0.0, (sum(x * (1.0 - y + old * x) for x, y in zip(column, estimates)) + ridge) / denominator)
            for i, x in enumerate(column):
                estimates[i] += x * (new - old)
            coefficients[j] = new
            largest = max(largest, abs(new - old))
        if largest < 1e-7:
            break
    parameters = {}
    for (key, parameter), value in zip(axes, coefficients):
        parameters.setdefault(key, {})[parameter] = value
    for key, entry in coverage.items():
        a = columns[axes.index((key, PARAMETERS[0]))]
        b = columns[axes.index((key, PARAMETERS[1]))]
        aa, bb, ab = sum(x*x for x in a), sum(x*x for x in b), sum(x*y for x,y in zip(a,b))
        entry["overhead_service_separable"] = aa * bb - ab * ab > 1e-8 * max(aa * bb, 1e-20)
        for field in ("stages", "grids", "points"):
            entry[field] = [min(entry[field]), max(entry[field])]
    model_profile = dict(profile)
    if measured_service:
        # Keep one copy of the potentially large curve table in serialized
        # audit artifacts.  predict() also accepts older caches that retain it
        # under profile.stage_service.
        model_profile.pop("stage_service", None)
    result = dict(schema="cubutterfly-stage-cost-model-v1", version=LEGACY_VERSION if legacy else VERSION,
        profile=model_profile,
        parameters=parameters, primitive_coverage=coverage, training_rows=len(rows),
        additive_training_rows=len(targets), calibration_source=(
            "independent stage probes; whole-plan sums retained as explicit fallback" if measured_service
            else "serial whole-plan sums; inferred stage costs"),
        separately_measured_stage_costs=measured_service, concurrency_calibrated=False,
        scheduling_capability_measured=bool(profile.get("scheduling")),
        service_model="measured-stage-curves" if measured_service else "legacy-whole-plan-fallback",
        legacy_fallback=("available" if measured_service else "primary"),
        legacy_compatibility_version=LEGACY_VERSION)
    if measured_service:
        result.update(stage_service=service_profile,
                      stage_service_status=service_profile.get("status"),
                      concurrency_validation="unvalidated")
    return result


def _minimal_terms(sample, profile):
    """Build only scheduling/resource provenance when measured curves are present.

    Independent service probes are authoritative for duration, so a capability
    rate profile is not required merely to compose measured groups.  The
    result is used only for resource-demand hints and explicit fallback rows.
    """
    groups = decoded(sample.get("execution_groups_json"), [])
    sm_count = max(1, number(profile, "sm_count", number(profile.get("hardware", {}), "sm_count", 1)))
    terms = []
    for group in groups:
        threads = max(1, number(group, "threads", number(sample, "tile_threads", 128)))
        grid = max(1, number(group, "grid_ctas",
                             number(group, "work_blocks", number(group, "work", 1))))
        launches = max(1, int(number(group, "launch_count",
                                      number(group, "kernel_launch_count", 1))))
        schedule = scheduling_costs(profile, threads, grid, sm_count)
        terms.append(dict(key=primitive_key(sample, group), startup_and_tail=schedule["startup_ms"],
            scheduling=schedule, service=1e-9, memory_ms=0.0, compute_ms=0.0,
            occupancy=min(1.0, grid / sm_count), launch_count=launches,
            stage_count=max(1, number(group, "stage_count", 1)), grid_ctas=grid, points=0.0,
            resource_source="unknown" if not group.get("compiler_resources_known") else "compiler",
            descriptor_source=sample.get("descriptor_source", "measured-plan")))
    return terms


def _predict_batch_composition(sample, model, projection, explain):
    """Compose actual serial tile kernels while keeping runtime dependencies."""
    batch = max(1, int(number(sample, "batch", 1)))
    tile = min(batch, max(1, int(number(sample, "batch_tile_count", 1))))
    tail_batch = batch % tile
    if (projection.get("tile_batch") != tile or projection.get("tile_count") != math.ceil(batch / tile)
            or projection.get("tail_batch") != tail_batch or bool(projection.get("tail")) != bool(tail_batch)):
        raise ValueError("stale batch service projection geometry")
    results = []
    for view in (projection["full"], projection.get("tail")):
        if view is None:
            results.append(None)
            continue
        physical_sample = {**view["sample"], "execution_groups_json": view["groups"]}
        if number(physical_sample, "stage_overlap") or number(physical_sample, "factor_overlap"):
            raise ValueError("batch service projection must contain serial descriptors")
        results.append(_predict_measured(physical_sample, model, explain=True))
    full, tail = results
    tiles = projection["tile_count"]
    stages = full["stages"]
    tail_stages = tail["stages"] if tail else None
    approximate = tiles * len(stages) > 20000
    if approximate:
        estimate = sum(stage["overhead_ms"] + stage["body_ms"] for stage in stages) * (tiles - 1)
        estimate += sum(stage["overhead_ms"] + stage["body_ms"] for stage in (tail_stages or stages))
    else:
        estimate = compose_stages(stages, tiles=tiles, overlap=True, tail_stages=tail_stages)
    resource_estimate = estimate
    pipeline = {"covered": False, "reason": "pipeline-scheduling-unmeasured"}
    if model.get("profile", {}).get("pipeline_scheduling"):
        from pipeline_schedule_model import costs as pipeline_costs
        pipeline = pipeline_costs(model["profile"], len(stages), tiles)
    if pipeline.get("covered"):
        # Submission and device execution proceed concurrently. An independent
        # minimal-work run supplies a scheduling floor, not an additive cost
        # per useful kernel (which would count launch latency twice).
        estimate = max(estimate, float(pipeline["measured_minimum_ms"]))
    predictions = [value for value in results if value is not None]
    details = dict(kernel_ms=estimate, stages=stages, tail_stages=tail_stages, schedule_tiles=tiles,
        missing_primitives=sorted({key for value in predictions for key in value["missing_primitives"]}),
        extrapolated_primitives=sorted({key for value in predictions for key in value["extrapolated_primitives"]}),
        unmeasured_groups=[group for value in predictions for group in value["unmeasured_groups"]],
        service_model="measured-stage-curves", whole_plan_fallback_used=any(
            value["whole_plan_fallback_used"] for value in predictions),
        schedule_model="serial-upper-estimate" if approximate else "dependency-and-shared-resource-simulation",
        overlap_calibration="unvalidated", concurrency_calibrated=False,
        resource_estimate_ms=resource_estimate, pipeline_scheduling=pipeline,
        service_projection="actual-bulk-tile-and-tail", tail_batch=projection["tail_batch"],
        descriptor_source=sample.get("descriptor_source", "measured-plan"))
    return details if explain else estimate


def _predict_measured(sample, model, explain=False):
    from stage_service_model import predict_group, resolve_resources, resource_regime_key

    service_profile = model.get("stage_service") or model.get("profile", {}).get("stage_service")
    projection = decoded(sample.get("stage_service_projection"), {})
    if (number(sample, "stage_overlap") and sample.get("backend") != "factor-streamed"
            and projection.get("status") == "resolved"):
        return _predict_batch_composition(sample, model, projection, explain)
    groups = decoded(sample.get("execution_groups_json"), [])
    if not groups:
        raise ValueError("physical execution groups are required for staged costs")
    try:
        terms = stage_terms(sample, model["profile"])
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        terms = _minimal_terms(sample, model["profile"])
    stages, missing, outside, unmeasured = [], [], [], []
    for index, group in enumerate(groups):
        term = terms[index] if index < len(terms) else _minimal_terms(
            dict(sample, execution_groups_json=[group]), model["profile"])[0]
        resolved_group = resolve_resources(sample, group, service_profile)
        resources_resolved = resource_regime_key(sample, resolved_group) != resource_regime_key(sample, group)
        measured = predict_group(sample, resolved_group, service_profile)
        if measured["covered"] and measured["kernel_ms"] is not None:
            # The probe duration already includes the physical launch.  Keep
            # overhead zero and launch_count one so schedule composition does
            # not add or multiply a second launch term.
            body = float(measured["kernel_ms"])
            stages.append(dict(overhead_ms=0.0, body_ms=body,
                scheduling=term["scheduling"], sm_demand=max(term["occupancy"], 0.0),
                hbm_demand=min(1.0, term["memory_ms"] / body) if body else 0.0,
                launch_count=1, service_source=measured["source"],
                curve_key=measured["curve_key"], measured_kernel_ms=body,
                resources_resolved=resources_resolved))
            continue
        key = measured["curve_key"]
        missing.append(key)
        if measured.get("extrapolated"):
            outside.append(key)
        unmeasured.append(dict(index=index, curve_key=key, reason=measured.get("reason"),
                               extrapolated=bool(measured.get("extrapolated")),
                               resources_resolved=resources_resolved))
        factors = model["parameters"].get(term["key"], {})
        overhead = term["startup_and_tail"] * factors.get("startup_and_tail", 1.0)
        body = max(1e-9, term["service"] * factors.get("service", 1.0))
        stages.append(dict(overhead_ms=overhead, body_ms=body,
            scheduling=term["scheduling"],
            sm_demand=max(term["occupancy"], min(1.0, term["compute_ms"] / body)),
            hbm_demand=min(1.0, term["memory_ms"] / body), launch_count=term["launch_count"],
            service_source="legacy-whole-plan-fallback", curve_key=key))
    factor = sample.get("backend") == "factor-streamed"
    overlap = bool(number(sample, "factor_overlap" if factor else "stage_overlap"))
    if overlap and not factor:
        marker = "unresolved-batch-service-projection"
        missing.append(marker)
        unmeasured.append(dict(index=None, curve_key=marker, reason=projection.get("reason", marker),
                               extrapolated=False, resources_resolved=False))
    batch = max(1, int(number(sample, "batch", 1)))
    tile = min(batch, max(1, int(number(sample, "batch_tile_count", 1))))
    tiles = max(t.get("launch_count", 1) for t in stages) if factor else (math.ceil(batch / tile) if overlap else 1)
    # A measured group is a full independent launch.  For a serial plan one
    # sample per group is the whole plan; overlapped plans still need a
    # conservative schedule but their concurrency has not been calibrated.
    fraction = (batch - (tiles - 1) * tile) / tile if overlap and not factor else 1.0
    approximate = tiles * len(stages) > 20000
    if approximate:
        estimate = sum(s["overhead_ms"] * tiles + s["body_ms"] * (tiles - 1 + fraction) for s in stages)
    else:
        estimate = compose_stages(stages, tiles, factor, overlap, fraction)
    details = dict(kernel_ms=estimate, stages=stages, schedule_tiles=tiles,
        missing_primitives=sorted(set(missing)), extrapolated_primitives=sorted(set(outside)),
        unmeasured_groups=unmeasured, service_model="measured-stage-curves",
        whole_plan_fallback_used=bool(unmeasured),
        schedule_model="serial-upper-estimate" if approximate else "dependency-and-shared-resource-simulation",
        overlap_calibration="unvalidated" if overlap else "not-applicable",
        concurrency_calibrated=False,
        descriptor_source=sample.get("descriptor_source", "measured-plan"))
    return details if explain else estimate


def predict(sample, model, explain=False, *, legacy=False):
    service_profile = model.get("stage_service") or model.get("profile", {}).get("stage_service")
    legacy = (legacy or model.get("version") == LEGACY_VERSION or
              model.get("service_model") == "legacy-whole-plan-fallback")
    if isinstance(service_profile, dict) and service_profile.get("curves") and not legacy:
        return _predict_measured(sample, model, explain)
    terms = stage_terms(sample, model["profile"])
    stages, missing, outside = [], [], []
    for term in terms:
        factors = model["parameters"].get(term["key"], {})
        if not factors:
            missing.append(term["key"])
        coverage = model.get("primitive_coverage", {}).get(term["key"], {})
        if coverage and any((not isinstance(coverage.get(field), (list, tuple)) or len(coverage[field]) < 2 or
                            not coverage[field][0] <= term[name] <= coverage[field][1])
                for field, name in (("stages", "stage_count"), ("grids", "grid_ctas"), ("points", "points"))):
            outside.append(term["key"])
        overhead = term["startup_and_tail"] * factors.get("startup_and_tail", 1.0)
        body = max(1e-9, term["service"] * factors.get("service", 1.0))
        stages.append(dict(overhead_ms=overhead, body_ms=body,
            scheduling=term["scheduling"],
            sm_demand=max(term["occupancy"], min(1.0, term["compute_ms"] / body)),
            hbm_demand=min(1.0, term["memory_ms"] / body)))
    factor = sample.get("backend") == "factor-streamed"
    overlap = bool(number(sample, "factor_overlap" if factor else "stage_overlap"))
    batch = max(1, int(number(sample, "batch", 1)))
    tile = min(batch, max(1, int(number(sample, "batch_tile_count", 1))))
    tiles = max(t["launch_count"] for t in terms) if factor else (math.ceil(batch / tile) if overlap else 1)
    fraction = (batch - (tiles - 1) * tile) / tile if overlap and not factor else 1.0
    # O(number of actual launches); a limit is explicit, never silently an
    # ideal pipeline extrapolation. Large schedules use a conservative serial
    # upper estimate and report that they need refined scheduling.
    approximate = tiles * len(stages) > 20000
    if approximate:
        estimate = sum(s["overhead_ms"] * tiles + s["body_ms"] * (tiles - 1 + fraction) for s in stages)
    else:
        estimate = compose_stages(stages, tiles, factor, overlap, fraction)
    details = dict(kernel_ms=estimate, stages=stages, schedule_tiles=tiles,
        missing_primitives=sorted(set(missing)), extrapolated_primitives=sorted(set(outside)),
        schedule_model="serial-upper-estimate" if approximate else "dependency-and-shared-resource-simulation",
        overlap_calibration="unmeasured" if overlap else "not-applicable",
        descriptor_source=sample.get("descriptor_source", "measured-plan"),
        service_model="legacy-whole-plan-fallback", whole_plan_fallback_used=True,
        unmeasured_groups=[])
    return details if explain else estimate


def _holdout_error_summary(checks, *, minimum_count=3, median_limit=.10, p90_limit=.20,
                           scope="complete-plan-serial-subset"):
    """Summarize independent complete-plan errors without vacuous success."""
    population = len(checks)
    covered = [item for item in checks if item.get("relative_error") is not None]
    errors = [item["relative_error"] for item in covered]
    median_error = statistics.median(errors) if errors else None
    if len(errors) == 1:
        p90_error = errors[0]
    elif errors:
        ordered = sorted(errors)
        p90_error = ordered[max(0, int(math.ceil(.90 * len(ordered))) - 1)]
    else:
        p90_error = None
    if not population:
        status = "unvalidated"
    elif len(covered) != population:
        status = "warning-uncovered"
    elif len(covered) < minimum_count:
        status = "insufficient-holdout"
    elif median_error <= median_limit and p90_error <= p90_limit:
        status = "passed"
    else:
        status = "warning-accuracy"
    return {
        "status": status, "scope": scope, "population": population,
        "count": len(covered), "covered_count": len(covered),
        "uncovered_count": population - len(covered),
        "median": median_error, "p90": p90_error,
        "pass_thresholds": {"minimum_count": minimum_count,
                            "median": median_limit, "p90": p90_limit},
        "checks": checks,
    }


def report(rows, profile):
    from fit_local_cost_model import workload_key, design_point_key, _winner_match
    model = fit(rows, profile)
    service_profile = profile.get("stage_service") if isinstance(profile, dict) else None
    measured_service = isinstance(service_profile, dict) and bool(service_profile.get("curves"))
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[workload_key(row)].append(row)
    checks, skipped, errors, serial_plan_checks, composition_checks = [], [], [], [], []
    for key, held in grouped.items():
        training = [r for r in rows if workload_key(r) != key]
        if not training or len({design_point_key(r) for r in held}) < 2:
            skipped.append(dict(workload=list(key), reason="insufficient independent candidate workloads"))
            continue
        local = fit(training, profile)
        estimates = [(r, predict(r["sample"], local, True)) for r in held]
        best = min(held, key=lambda r:r["median_kernel_ms"])
        selected = min(estimates, key=lambda x:x[1]["kernel_ms"])[0]
        workload_has_overlap = any(
            number(held_row.get("sample", {}), "stage_overlap") or
            number(held_row.get("sample", {}), "factor_overlap")
            for held_row, _ in estimates
        )
        for held_row, prediction in estimates:
            held_sample = held_row.get("sample", {})
            overlap = bool(number(held_sample, "stage_overlap") or
                           number(held_sample, "factor_overlap"))
            observed = held_row.get("median_kernel_ms")
            predicted = prediction.get("kernel_ms")
            try:
                observed = float(observed)
                predicted = float(predicted)
            except (TypeError, ValueError):
                observed = predicted = None
            error = (abs(predicted / observed - 1.0)
                     if observed is not None and observed > 0 and predicted is not None and predicted > 0
                     else None)
            errors.append(error) if error is not None else None
            prediction_covered = (error is not None and
                                  not prediction.get("missing_primitives") and
                                  not prediction.get("extrapolated_primitives"))
            if measured_service and prediction.get("whole_plan_fallback_used"):
                prediction_covered = False
            plan_check = {
                "name": held_row.get("name"), "overlap": overlap,
                "covered": prediction_covered,
                "relative_error": error if prediction_covered else None,
                "predicted_kernel_ms": predicted,
                "observed_kernel_ms": observed,
                "reason": ("covered" if prediction_covered else
                           "unmeasured-or-outside-service-curve" if measured_service else
                           "uncovered-or-invalid-plan"),
            }
            if not overlap:
                serial_plan_checks.append(plan_check)
        if measured_service:
            for held_row, prediction in estimates:
                held_sample = held_row.get("sample", {})
                if not (number(held_sample, "stage_overlap") or number(held_sample, "factor_overlap")):
                    continue
                covered = (not prediction.get("missing_primitives") and
                           not prediction.get("extrapolated_primitives"))
                error = (abs(prediction["kernel_ms"] / held_row["median_kernel_ms"] - 1)
                         if covered and prediction.get("kernel_ms") else None)
                composition_checks.append({
                    "name": held_row.get("name"), "covered": covered, "relative_error": error,
                    "predicted_kernel_ms": prediction.get("kernel_ms"),
                    "observed_kernel_ms": held_row.get("median_kernel_ms"),
                    "reason": ("covered" if covered else
                               "outside-or-unmeasured-service-curve"),
                })
        checks.append(dict(workload=list(key), match=_winner_match(best, selected),
            measured_winner=best["name"], predicted_winner=selected["name"],
            latency_regret=selected["median_kernel_ms"] / best["median_kernel_ms"],
            primitives_covered=all(not p["missing_primitives"] for _,p in estimates),
            within_calibration_range=all(not p["extrapolated_primitives"] for _,p in estimates),
            overlap=workload_has_overlap))
    accuracy = statistics.mean(c["match"] for c in checks) if checks else None
    serial_checks = [check for check in checks if not check.get("overlap")]
    serial_accuracy = statistics.mean(c["match"] for c in serial_checks) if serial_checks else None
    median_error = statistics.median(errors) if errors else None
    service_validation = service_profile.get("validation", {}) if measured_service else {}
    service_median = service_validation.get("heldout_median_relative_error")
    service_p90 = service_validation.get("heldout_p90_relative_error")
    service_holdout_count = service_validation.get("heldout_count", 0)
    service_coverage_ok = (bool(service_holdout_count) and
                           service_validation.get("heldout_covered_count", 0) == service_holdout_count and
                           service_validation.get("heldout_uncovered_count", 0) == 0)
    whole_plan_validation = _holdout_error_summary(serial_plan_checks)
    if not measured_service:
        # Legacy whole-plan fitting retains its historical thresholds.  The
        # explicit summary still records the serial subset, but does not
        # change v3 compatibility behavior.
        whole_plan_validation["pass_thresholds"] = {
            "minimum_count": 3, "median": .30, "p90": None,
        }
        whole_plan_validation["status"] = (
            "unvalidated" if not serial_plan_checks else
            "warning-uncovered" if whole_plan_validation["uncovered_count"] else
            "passed" if (len(serial_plan_checks) >= 3 and
                          whole_plan_validation["median"] is not None and
                          whole_plan_validation["median"] <= .30) else
            "insufficient-holdout")
    composition_validation = _holdout_error_summary(
        composition_checks, minimum_count=1, scope="observed-overlap-composition-holdout")
    global_validation = _holdout_error_summary(serial_plan_checks + composition_checks,
        scope="all-observed-complete-plan-holdouts")
    global_validation.update(top1_accuracy=accuracy, workload_count=len(checks))
    global_validation["passed"] = bool(global_validation["status"] == "passed" and
        len(checks) >= 3 and accuracy is not None and accuracy >= .90)
    # Composition requests are optional.  Empty overlap populations remain
    # unvalidated rather than becoming a vacuous pass.
    if measured_service:
        serial_validated = (service_coverage_ok and service_median is not None and service_p90 is not None
                            and service_median <= .10 and service_p90 <= .20
                            and len(serial_checks) >= 3 and serial_accuracy is not None and serial_accuracy >= .90
                            and whole_plan_validation["status"] == "passed"
                            and all(c["primitives_covered"] and c["within_calibration_range"]
                                    for c in serial_checks))
        overlap_rows = [row for row in rows if number(row.get("sample", {}), "stage_overlap") or
                        number(row.get("sample", {}), "factor_overlap")]
        overlap_status = composition_validation
        validated = serial_validated
        # Serial evidence has its own status. Only observed complete-plan
        # holdouts can qualify the empirical scheduler; identifying its
        # concurrent resource parameters is a separate requirement.
        status = "calibrated-local-serial" if validated else "calibrated-local-warning" if rows else "insufficient-candidates"
        if (service_coverage_ok and service_validation.get("passed") and
                global_validation["passed"] and composition_checks and
                composition_validation["status"] == "passed"):
            # Complete-plan holdouts can validate the empirical composition
            # model without claiming its concurrent resource parameters have
            # been identified independently.
            validated = True
            status = "calibrated-local"
        validation_scope = "independent-stage-holdout-plus-serial-ranking"
        median_threshold, p90_threshold = .10, .20
    else:
        validated = (len(checks) >= 3 and accuracy >= .90 and median_error <= .30
                     and all(c["primitives_covered"] for c in checks))
        overlap_rows = []
        overlap_status = composition_validation
        status = ("calibrated-local" if validated else "calibrated-local-warning") if rows else "insufficient-candidates"
        validation_scope = "leave_one_complete_workload_out"
        median_threshold, p90_threshold = .30, None
    # Keep stage-service holdout errors separate from the complete-plan errors:
    # accurate isolated groups do not prove that serial composition is accurate.
    reported_median = whole_plan_validation["median"] if measured_service else median_error
    reported_p90 = whole_plan_validation["p90"] if measured_service else None
    reported_accuracy = accuracy
    model.update(status=status, usable_for_unmeasured_ranking=validated, device=profile.get("device"),
        validation=dict(scope=validation_scope, holdout_top1_accuracy=reported_accuracy,
            holdout_valid_multi_candidate_workloads=len(checks), holdout_workload_winner_checks=checks,
            holdout_skipped_workloads=skipped, holdout_median_relative_error=reported_median,
            holdout_p90_relative_error=reported_p90,
            pass_thresholds=dict(holdout_valid_multi_candidate_workloads=3, holdout_top1_accuracy=.90,
                                 holdout_median_relative_error=median_threshold,
                                 holdout_p90_relative_error=p90_threshold,
                                 all_held_out_primitives_covered=True),
            holdout_mean_latency_regret=statistics.mean(c["latency_regret"] for c in checks) if checks else None,
            stage_service=service_validation if measured_service else None,
            stage_service_holdout_median_relative_error=(service_median if measured_service else None),
            stage_service_holdout_p90_relative_error=(service_p90 if measured_service else None),
            stage_service_holdouts_covered=(service_coverage_ok if measured_service else None),
            whole_plan_validation=whole_plan_validation,
            whole_plan_holdout_median_relative_error=(whole_plan_validation["median"]
                                                      if measured_service else None),
            whole_plan_holdout_p90_relative_error=(whole_plan_validation["p90"]
                                                   if measured_service else None),
            overlap_validation=overlap_status,
            composition_validation=composition_validation))
    model["validation"]["global_validation"] = global_validation
    model["validation"]["serial_top1_accuracy"] = serial_accuracy
    limitations = ["Unknown primitives use hardware-rate priors and require correct on-device measurement.",
                   "A bounded search can use this ranking aid; validation does not certify the full design space."]
    if measured_service:
        limitations.append("Independent group service curves are measured; whole-plan coefficients remain fallback only.")
        limitations.append("Overlap composition passed the observed holdouts; concurrent resource parameters remain unidentified."
            if composition_validation["status"] == "passed" else
            "Concurrent overlap/composition is unvalidated and must not be treated as a global calibration.")
    else:
        limitations.extend(["Startup and tail cannot be separated by whole-plan timings.",
                             "Service terms are fitted to serial plan sums, not isolated stage measurements.",
                             "Concurrent SM/HBM service sharing is an uncalibrated approximation."])
    model["limitations"] = limitations
    model["overlap_validation"] = overlap_status
    model["whole_plan_validation"] = whole_plan_validation
    model["composition_validation"] = composition_validation
    return model


def main():
    from fit_local_cost_model import load_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--operator-calibration", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows, external = load_rows(args.operator_calibration)
    model = report(rows, json.loads(args.profile.read_text()))
    model["external_reference_rows"] = len(external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n")
    print(f"staged cost model: {model['status']}; {len(rows)} rows")


if __name__ == "__main__":
    main()
