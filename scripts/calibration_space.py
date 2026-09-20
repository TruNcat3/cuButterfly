"""Compiled-unit projection used by installation calibration.

Local-unit availability comes from the linked library. It is deliberately
distinct from full-plan validation and from the theoretical architecture space.
"""
import hashlib
import itertools
import json
import resident_mapping

PROJECTION_VERSION = "physical-generic-launch-axes-v6-resident-axes"
MAPPING_PROJECTION_FIELDS = (
    "fft_core", "local_stages", "prefix_threads", "suffix_threads", "prefix_ept", "suffix_ept",
    "prefix_units_per_cta", "suffix_units_per_cta", "units_per_cta",
    "shared_layout", "cross_twiddle", "stage_partition", "stage_overlap", "batch_tile_count",
    "local_stage_partitions", "exchange_chunks", "prefix_codelet", "prefix_shared_layout",
    "prefix_codelet_lanes", "factor_io_policies",
)


def online_candidates(units, workload):
    """Full two-segment Cartesian product; no GPU/model/size-specific winners."""
    selected = [u for u in units if u["precision"] == workload["precision"]]
    for prefix in selected:
        for suffix in selected:
            if prefix["logN"] + suffix["logN"] != workload["logN"]:
                continue
            layouts = ("linear", "xor-swizzle") if (
                workload["precision"] == "fp64" or prefix["logN"] <= 10
            ) else ("linear",)
            for twiddle, layout in itertools.product(("table", "recurrence"), layouts):
                boundaries = ("direct-strided",)
                if workload["precision"] == "fp32" and suffix["logN"] in (11, 12):
                    boundaries = ("direct-strided", "tiled-transpose")
                for boundary in boundaries:
                    # Imported units resolve reorder_columns to 1. Do not
                    # benchmark aliases as distinct physical design points.
                    point = {
                        **workload, "operator": "fft", "backend": "online-reorder",
                        "fft_core": "cufftdx-block", "compute_unit": "auto",
                        "local_stages": prefix["logN"], "reorder_columns": 1,
                        "prefix_threads": prefix["threads"], "prefix_ept": prefix["ept"],
                        "suffix_threads": suffix["threads"], "suffix_ept": suffix["ept"],
                        "cross_twiddle": twiddle, "shared_layout": layout,
                        "direct_boundary": boundary,
                        "stage_overlap": 0, "batch_tile_count": 1,
                    }
                    yield point
                    # Schedule is independent of the decomposition and local
                    # unit. Keep bulk as a candidate; overlap is not always faster.
                    batch = int(workload.get("batch", 1))
                    tile = 1
                    while tile < batch:
                        yield {**point, "stage_overlap": 1, "batch_tile_count": tile}
                        tile *= 2


def butterfly_candidates(workload):
    """Enumerate the shared in-tree Butterfly realization families.

    This is intentionally a legal projection; hardware feasibility and
    correctness remain runtime checks. It keeps arithmetic, backend and
    mapping axes independent for non-FFT operators as well.
    """
    operator, precision, log_n = workload["operator"], workload["precision"], workload["logN"]
    stages = tuple(range(5, min(10, log_n - 1) + 1))
    for backend in ("temporal-tile", "hierarchical", "online-reorder"):
        if backend != "temporal-tile" and not stages:
            continue
        for unit, threads, local in itertools.product(
                ("radix2", "radix4", "radix8"), (32, 64, 128, 256),
                (0,) if backend == "temporal-tile" else stages):
                point = {**workload, "backend": backend, "compute_unit": unit,
                         "tile_threads": threads, "local_stages": local,
                         "reorder_columns": 1 if backend == "online-reorder" else 0,
                         "local_exchange": "shared", "fft_core": "scalar"}
                if operator == "structured-2x2":
                    point["stage_matrix"] = workload.get("stage_matrix", "1,0.25,-0.5,1")
                yield point
    for local, threads, layout in itertools.product(
            range(1, min(log_n, 12) + 1), (32, 64, 128, 256), ("linear", "writer-aligned")):
        point = {**workload, "backend": "shared-iterative", "compute_unit": "radix2",
                 "tile_threads": threads, "local_stages": local, "shared_layout": layout,
                 "local_exchange": "shared", "fft_core": "scalar"}
        if operator == "structured-2x2":
            point["stage_matrix"] = workload.get("stage_matrix", "1,0.25,-0.5,1")
        yield point
    if log_n == 8:
        for stage_space in (1, 2, 4, 8):
            yield {**workload, "backend": "stage-pipeline", "compute_unit": "radix2",
                   "stage_space": stage_space, "pipeline_warps": 8, "local_stages": 0,
                   "fft_core": "scalar", "local_exchange": "shared"}
    if precision == "fp32" and operator in ("fwht", "structured-2x2"):
        yield {**workload, "backend": "temporal-tile", "compute_unit": "radix2",
               "tile_threads": 256, "local_stages": 0, "reorder_columns": 0,
               "local_exchange": "warp-register", "fft_core": "scalar"}


def register_tile_candidates(mappings, workload):
    """Use the binary's compiled and resource-validated mapping inventory."""
    if workload.get("precision") != "fp32":
        return
    for mapping in mappings:
        if int(mapping["logN"]) != workload["logN"]:
            continue
        resolved = {}
        for key, value in mapping.items():
            # The binary inventory now carries string/list strategy axes in
            # addition to launch integers.  Preserve those values verbatim.
            if isinstance(value, bool):
                resolved[key] = value
            else:
                try:
                    integer = int(value)
                except (TypeError, ValueError):
                    resolved[key] = value
                else:
                    resolved[key] = integer if str(integer) == str(value).strip() else value
        resolved.setdefault("prefix_codelet", "native")
        resolved.setdefault("prefix_shared_layout", "linear")
        resolved.setdefault("prefix_codelet_lanes", 1)
        resolved.setdefault("factor_io_policies", [])
        yield {**workload, **resolved, "operator": "fft",
               "backend": "online-reorder", "fft_core": "register-tile", "compute_unit": "auto",
               "shared_layout": "writer-aligned", "cross_twiddle": "recurrence",
               "direct_boundary": "direct-strided", "reorder_columns": 1,
               "stage_overlap": 0, "batch_tile_count": 1}


def candidate_id(point):
    payload = json.dumps(point, sort_keys=True, separators=(",", ":"))
    return "search-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _prefix_lane_identity(mapping):
    value = mapping.get("prefix_codelet_lanes", 1)
    if isinstance(value, bool):
        return str(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _strategy_identity(mapping, policies):
    parts = mapping.get("local_stage_partitions", []) or []
    chunks = mapping.get("exchange_chunks", []) or []
    return (
        mapping.get("prefix_codelet", "native"),
        mapping.get("prefix_shared_layout", "linear"),
        _prefix_lane_identity(mapping),
        tuple(str(value) for value in policies),
        tuple(tuple(int(stage) for stage in part) for part in parts) if any(parts) else (),
        tuple(int(value) for value in chunks) if any(chunks) else (),
    )


def _strategy_order(strategies):
    """Cover independent strategy axes before their Cartesian combinations.

    Only the first 32 representatives need the greedy coverage calculation;
    the complete remainder stays available in deterministic order. This keeps
    an unbounded inventory export linear apart from that fixed prefix.
    """
    default = ("native", "linear", 1, (), (), ())
    remaining = set(strategies)
    result = []
    counts = [{} for _ in default]
    if default in remaining:
        result.append(default)
        remaining.remove(default)
        for index, value in enumerate(default):
            counts[index][value] = 1
    while remaining and len(result) < 32:
        key = min(remaining, key=lambda row: (
            -sum(value not in counts[index] for index, value in enumerate(row)),
            sum(counts[index].get(value, 0) for index, value in enumerate(row)), row))
        result.append(key)
        remaining.remove(key)
        for index, value in enumerate(key):
            counts[index][value] = counts[index].get(value, 0) + 1
    return result + sorted(remaining)


def stratified_candidates(points, budget):
    """Round-robin partitions and policies, spreading launch-axis values.

    This is a bounded screening order, never a proof of optimality. Budget=0
    measures the entire enumerated product. Omitted points remain in the audit.
    """
    families={}
    for point in points:
        families.setdefault((point.get("backend",""),point.get("fft_core","scalar")),[]).append(point)
    if len(families)>1:
        # Give every core/lowering a continuing share of exploration. Merely
        # seeding one point let old prefix/suffix axes consume the whole budget.
        queues=[stratified_candidates(families[key],0) for key in sorted(families)]
        ordered=[p for layer in itertools.zip_longest(*queues) for p in layer if p is not None]
        return ordered if budget==0 else ordered[:budget]
    if points and points[0].get("backend")=="factor-streamed":
        strata={}
        for point in points:
            raw_mapping = point.get("mapping_json", "{}")
            mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else (
                raw_mapping if isinstance(raw_mapping, dict) else {})
            axes={**point,**mapping}
            policies = axes.get("factor_io_policies", [])
            if not isinstance(policies, list):
                policies = [policies]
            if not policies or all(value == "dynamic" for value in policies):
                # Empty and explicit-all-dynamic are the same runtime
                # lowering; keep their public identities intact but place
                # them in one screening stratum.
                policies = ["dynamic"] * len(axes.get("factor_partition", []))
            key=(len(axes.get("factor_partition",[])),axes.get("prefetch_depth",0),axes.get("data_tiles_per_cta",1),
                 axes.get("factor_slices",1),bool(axes.get("factor_overlap",False)),
                 tuple(str(value) for value in policies))
            strata.setdefault(key,[]).append(point)
        remaining=set(strata); keys=[]; coverage=[{} for _ in range(6)]
        while remaining:
            # Cover independent axes before filling their Cartesian product;
            # lexicographic order otherwise spends the budget on one count or
            # only synchronous loading before trying any prefetch.
            key=min(remaining,key=lambda k:(-sum(v not in coverage[i] for i,v in enumerate(k)),
                                            sum(coverage[i].get(v,0) for i,v in enumerate(k)),k))
            keys.append(key); remaining.remove(key)
            for i,v in enumerate(key): coverage[i][v]=coverage[i].get(v,0)+1
        queues=[sorted(strata[key],key=candidate_id) for key in keys]
        ordered=[p for layer in itertools.zip_longest(*queues) for p in layer if p is not None]
        return ordered if budget==0 else ordered[:budget]
    strata = {}
    for point in points:
        raw_mapping = point.get("mapping_json", "{}")
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else (
            raw_mapping if isinstance(raw_mapping, dict) else {})
        policies = mapping.get("factor_io_policies", [])
        if not isinstance(policies, list):
            policies = [policies]
        # Keep implementation choices in the finite screening strata.  The
        # axes are independent of the workload and must get a representative
        # before the budget is spent on launch/layout variants.
        strategy = _strategy_identity({**point, **mapping}, policies)
        key = (point.get("local_stages", 0), point.get("cross_twiddle", "table"),
               point.get("shared_layout", "linear"), point.get("backend", ""),
               point.get("stage_overlap", 0), strategy)
        strata.setdefault(key, []).append(point)
    queues = []
    for key in sorted(strata):
        # A deterministic hash spreads both independent launch axes; avoid
        # lexicographic truncation measuring only the smallest prefix mapping.
        queues.append(iter(sorted(strata[key], key=candidate_id)))
    ordered = []
    # Reserve one representative for every implementation strategy before
    # launch/unit coverage.  A bounded screening run must see the canonical
    # native/linear lowering first when it exists; other strategies follow in
    # deterministic order.  The later strata traversal still supplies the
    # remaining structural and launch-axis coverage.
    strategy_points = {}
    for point in points:
        raw_mapping = point.get("mapping_json", "{}")
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else (
            raw_mapping if isinstance(raw_mapping, dict) else {})
        policies = mapping.get("factor_io_policies", [])
        if not isinstance(policies, list):
            policies = [policies]
        strategy = _strategy_identity({**point, **mapping}, policies)
        strategy_points.setdefault(strategy, []).append(point)
    strategy_order = _strategy_order(strategy_points)
    ordered.extend(min(strategy_points[key], key=candidate_id)
                   for key in strategy_order)
    # Prioritize distinct prefix/suffix unit pairs before extra policy variants.
    # This is still bounded by budget, and a unit-pair representative does not
    # cover every partition, layout or twiddle. Exhaustive research uses 0.
    # Cover each executable family before spreading the launch mappings.
    # A new family must not lose its entire budget to variants of an old one.
    family_points = {}
    for point in points:
        family_points.setdefault((point.get("backend", ""),point.get("fft_core", "scalar")), []).append(point)
    for family in sorted(family_points):
        ordered.append(min(family_points[family], key=candidate_id))
    mapping_queues = {}
    for point in points:
        raw_mapping = point.get("mapping_json", {})
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else raw_mapping
        mapping_key = (point.get("prefix_threads"), point.get("prefix_ept"),
                       _prefix_lane_identity({**point, **mapping}),
                       point.get("suffix_threads"), point.get("suffix_ept"))
        mapping_queues.setdefault(mapping_key, []).append(point)
    for queue in mapping_queues.values():
        queue.sort(key=candidate_id)
    for key in sorted(mapping_queues):
        ordered.append(mapping_queues[key][0])
    for layer in itertools.zip_longest(*queues):
        ordered.extend(point for point in layer if point is not None)
    # The same point appears in the mapping coverage and stratum traversal;
    # preserve uniqueness while retaining the deterministic order.
    unique = []
    seen = set()
    for point in ordered:
        identifier = candidate_id(point)
        if identifier not in seen:
            seen.add(identifier)
            unique.append(point)
    return unique if budget == 0 else unique[:budget]


def command_for(binary, point):
    command = [str(binary)]
    ntt = point.get("operator") == "ntt"
    for key, value in point.items():
        # A public mapping already owns these axes. Their top-level copies
        # support CPU scoring, not an additional CLI protocol. In particular,
        # nested partitions and units-per-CTA have no scalar CLI equivalent.
        if "mapping_json" in point and key in ("backend", *MAPPING_PROJECTION_FIELDS):
            continue
        if key == "mapping_json":
            command.extend(["--mapping-json", json.dumps(value) if isinstance(value, dict) else value])
            continue
        if ntt and key in ("operator", "accumulation", "normalization", "element_stride", "batch_stride", "stage_matrix"):
            continue
        if ntt and key == "precision":
            command.extend(["--word-bits", str(value).replace("word", "").replace("uint", "")])
            continue
        if ntt and key == "placement":
            command.extend(["--output-order", "natural" if value in ("out-of-place", "in-place") else value])
            continue
        if key == "stage_overlap":
            if value:
                command.append("--stage-overlap")
            continue
        if key == "direction":
            if value == "inverse":
                command.append("--inverse")
            elif value != "forward":
                raise ValueError("direction must be forward or inverse")
            continue
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def runtime_candidates(binary, workload, runner):
    """The library owns the candidate projection; scripts schedule its records."""
    workload = {"operator": "fft", **workload}
    completed = runner(command_for(binary, workload) + ["--list-design-points"])
    mappings = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    unique = {}
    for mapping in mappings:
        point = mapping_point(workload, mapping)
        unique[candidate_id(point)] = point
    return list(unique.values())


def mapping_point(workload, mapping):
    """Apply mapping parameters to target semantics, never source semantics."""
    point = {**workload, "mapping_json": json.dumps(mapping, sort_keys=True, separators=(",", ":")),
             "backend": mapping["backend"]}
    for field in MAPPING_PROJECTION_FIELDS:
        if field in mapping:
            point[field] = mapping[field]
    return point


def predicted_sample(point):
    # Accept both CLI-axis candidates and public mapping records. A mapping
    # record owns its physical axes when the point also carries legacy aliases.
    raw_mapping = point.get("mapping_json", {})
    mapping = {**point, **(json.loads(raw_mapping) if isinstance(raw_mapping,str) else raw_mapping)}
    sample = {**mapping, **point, "N": 1 << int(point["logN"])}
    partition = mapping.get("stage_partition", [])
    if not partition:
        local = mapping.get("local_stages", mapping.get("flow_tile_log_n",0))
        if mapping.get("backend") == "shared-iterative" and local:
            remaining = int(point["logN"])
            while remaining:
                stages = min(local, remaining); partition.append(stages); remaining -= stages
        elif mapping.get("backend") == "online-reorder" and local:
            partition = [local, int(point["logN"]) - local]
        elif mapping.get("backend") == "hierarchical" and local:
            partition = [local] + [1] * (int(point["logN"]) - local)
        else:
            partition = [int(point["logN"])]
    if mapping.get("backend") == "hierarchical" and mapping.get("local_stages"):
        # The hierarchical adapter always lowers its tail to single-stage
        # launches, even if the public config retains a logical partition.
        local = int(mapping["local_stages"])
        partition = [local] + [1] * (int(point["logN"])-local)
    boundaries = mapping.get("boundaries", [])
    groups = len(partition) - sum(b.get("residency") == "fused" for b in boundaries)
    sample["decomposition_count"] = len(partition)
    if mapping.get("backend") == "shared-iterative" and point.get("operator") != "ntt" and boundaries:
        from research_compile_requests import _shared_partition
        partition = _shared_partition(mapping, point.get("operator","fft"), int(point["logN"]))
        groups = len(partition)
    if mapping.get("backend")=="factor-streamed":
        partition=mapping.get("factor_partition") or partition
        groups=len(partition)
        # normalize_factor_mapping resolves these legacy config fields from
        # the first physical factor, rather than the inventory's defaults.
        sample["local_stages"] = partition[0]
        sample["tile_threads"] = ((1 << partition[0]) // int(mapping.get("factor_ept", 16))
                                  * int(mapping.get("factor_columns", 8)))
    sample["execution_group_count"] = groups
    sample["group_cores"] = ":".join(g["core"] for g in mapping.get("execution_group_mappings", []))
    fft = point.get("operator", "fft") == "fft"
    generic = (point.get("operator", "fft") != "ntt" and
               (not fft or mapping.get("fft_core", "scalar") == "scalar") and
               mapping.get("backend") in ("hierarchical", "online-reorder") and
               mapping.get("local_exchange", "shared") != "warp-register" and
               not sample.get("stage_overlap"))
    register = fft and mapping.get("fft_core") == "register-tile"
    register_prefix = register and mapping.get("backend") == "online-reorder"
    if mapping.get("backend") == "shared-iterative" or register_prefix:
        local_parts, exchange_chunks = resident_mapping.validate_resident_axes(
            mapping, partition, backend=mapping.get("backend"), operator=point.get("operator","fft"))
    else:
        local_parts, exchange_chunks = resident_mapping.normalize_axes(mapping)
        if local_parts or exchange_chunks:
            raise ValueError("resident axes are unsupported for this lowering")
    register_shape = resident_mapping.register_geometry(mapping, partition) if register_prefix else None
    prefix_lanes = mapping.get("prefix_codelet_lanes", 1)
    if isinstance(prefix_lanes, bool) or not isinstance(prefix_lanes, int):
        raise ValueError("prefix_codelet_lanes must be an integer")
    if prefix_lanes <= 0 or prefix_lanes & (prefix_lanes - 1):
        raise ValueError("prefix_codelet_lanes must be a positive power of two")
    if prefix_lanes != 1 and not register_prefix:
        raise ValueError("prefix_codelet_lanes are valid only for online-reorder register-tile")
    if register_prefix:
        sample["prefix_codelet_lanes"] = prefix_lanes
    if generic:
        sample["group_cores"] = ":".join(["scalar"] * groups)
    elif not sample["group_cores"]:
        if mapping.get("backend") == "factor-streamed":
            sample["group_cores"] = ":".join([mapping.get("fft_core", "cufftdx-block")] * groups)
        elif mapping.get("backend") == "online-reorder" and register:
            sample["group_cores"] = ":".join(["register-tile"] + ["cufftdx-block"] * (groups - 1))
    if mapping.get("backend") == "online-reorder" and register:
        # The register adapter canonicalizes the unused scalar arithmetic axis.
        sample["compute_unit"] = "auto"
    # Physical groups are inputs to ranking even before compiler resource
    # queries exist. Unknown register allocation remains explicitly unknown.
    physical=[]
    width={"fp32":4,"fp64":8,"fp16":2,"bf16":2,"word32":4,"word64":8,"uint32":4}.get(point.get("precision"),4)
    if point.get("operator","fft")=="fft": width*=2
    for i, stages in enumerate(partition):
        threads=int(mapping.get("tile_threads",mapping.get("threads_per_block",128))) or 128
        live=0
        if mapping.get("backend")=="shared-iterative": live=2*(1<<stages)*width
        elif register:
            threads=int(mapping.get("prefix_threads" if i==0 else "suffix_threads",threads))
            if i == 0 and register_prefix:
                columns = register_shape["prefix_columns"]
            else:
                columns=threads*int(mapping.get("suffix_ept",1))//(1<<stages) if i != 0 else threads//(1<<(stages//2))
            live=(1<<stages)*columns*width
        grid_columns=columns if register else 1
        tiles,depth=1,0
        if mapping.get("backend")=="factor-streamed":
            grid_columns=int(mapping.get("factor_columns",8))
            threads=(1<<stages)//int(mapping.get("factor_ept",16))*grid_columns
            tiles=int(mapping.get("data_tiles_per_cta",1)); depth=int(mapping.get("prefetch_depth",0))
            live=(1<<stages)*grid_columns*width*(depth+1)
        launch_batch=int(point["batch"])
        if sample.get("stage_overlap"): launch_batch=min(launch_batch,int(sample.get("batch_tile_count",1)))
        if grid_columns <= 0 or stages > int(point["logN"]):
            raise ValueError("projected local transform does not fit its thread/EPT or workload extent")
        grid=launch_batch*(1<<(int(point["logN"])-stages))//grid_columns
        launches=int(mapping.get("factor_slices",1)) if mapping.get("backend")=="factor-streamed" and i+1<len(partition) else 1
        explicit = mapping.get("execution_group_mappings", [])
        core = (sample["group_cores"].split(":")[i] if sample["group_cores"] else mapping.get("fft_core", mapping.get("subgraph_core", "scalar")))
        if mapping.get("backend") == "factor-streamed":
            ept = int(mapping.get("factor_ept", 16))
        elif register_prefix and i == 0:
            ept = register_shape["prefix_ept"]
        else:
            ept = int(mapping.get("prefix_ept" if i == 0 else "suffix_ept", 0))
        if i < len(explicit) and not generic:
            core = explicit[i].get("core", core)
            ept = int(explicit[i].get("elements_per_thread", explicit[i].get("ept", ept)))
            threads = int(explicit[i].get("threads", threads))
        data_space = min(threads, (1 << stages)//2 * grid_columns)
        data_time = max(1,((1<<stages)//2*grid_columns+threads-1)//threads)*tiles
        exchange = mapping.get("local_exchange", "shared")
        if generic:
            # Match the typed generic launchers, which ignore FFT unit/EPT
            # mappings. Hierarchical tails pack 256 butterflies into one CTA.
            core, ept = "scalar", 0
            threads = int(mapping.get("tile_threads", 128))
            butterflies = (1 << stages)//2
            live = (1 << stages)*width
            exchange = "shared"
            if mapping["backend"] == "hierarchical" and i:
                threads, live, exchange = 256, 0, "global"
                total = int(point["batch"])*(1 << (int(point["logN"])-1))
                grid = (total + threads - 1)//threads
                butterflies = min(total, threads)
            elif mapping["backend"] == "online-reorder" and i:
                columns = max(1, int(mapping.get("reorder_columns", 1)))
                grid = int(point["batch"])*(((1 << partition[0])+columns-1)//columns)
                butterflies *= columns
                live *= columns
            data_space = max(1, min(threads, butterflies))
            data_time = max(1, (butterflies+data_space-1)//data_space)
        factor_policies = mapping.get("factor_io_policies", [])
        if factor_policies in (None, ""):
            factor_policies = []
        if not isinstance(factor_policies, list):
            factor_policies = [factor_policies]
        if mapping.get("backend") == "factor-streamed":
            if factor_policies and len(factor_policies) != len(partition):
                raise ValueError("factor_io_policies must match factor_partition")
            io_policy = factor_policies[i] if factor_policies else "dynamic"
        else:
            io_policy = "dynamic"
        if register_prefix and i == 0:
            codelet = mapping.get("prefix_codelet", "native")
            group_layout = mapping.get("prefix_shared_layout", "linear")
        else:
            # ``core`` identifies the physical library kernel (for example
            # cufftdx-block); ``codelet`` is the separate code-generation
            # identity and remains the native default for suffix/generic
            # groups.
            codelet = "native"
            group_layout = mapping.get("shared_layout", "linear")
        local_part = local_parts[i] if local_parts else []
        exchange_chunk = exchange_chunks[i] if exchange_chunks else 0
        if mapping.get("backend") == "shared-iterative" and local_part:
            codelet = "native-register-subgraph"
            ept = 1 << max(local_part)
        physical.append(dict(index=i, stage_count=stages,live_shared_bytes=live,threads=threads,elements_per_thread=ept,
            core=core,codelet=codelet,io_policy=io_policy,shared_layout=group_layout,
            local_stage_partition=list(local_part),exchange_chunk=exchange_chunk,
            data_space=data_space,data_time=data_time,exchange=exchange,grid_ctas=(grid//launches+tiles-1)//tiles,
            launch_count=launches,partial_dependency_ready=bool(mapping.get("factor_overlap")) and 0<i<len(partition)-1,
            data_tiles_per_cta=tiles,prefetch_depth=depth,compiler_local_resources_known=False))
    sample["execution_groups_json"]=physical
    return sample
