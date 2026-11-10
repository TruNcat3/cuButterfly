#!/usr/bin/env python3
"""Decompose steady-state CKKS NVTX measurements in an Nsight SQLite export."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3


LABEL = re.compile(r"^ckks_measure/([^/]+)/(native_loop|cu_loop|native_stage_batch|cu_stage_batch|cu_auto)/(\d+)$")
CATEGORIES = ("ntt", "gather", "scatter", "non_ntt")


def category(name):
    if "cubutterfly_heongpu::" in name and "transfer<" in name:
        if re.search(r"transfer<(?:true|\(bool\)1)>", name):
            return "gather"
        if re.search(r"transfer<(?:false|\(bool\)0)>", name):
            return "scatter"
        raise ValueError(f"unrecognized stage transfer template: {name}")
    if ("gpuntt::" in name or "cuntt::rns_fused_detail::stage_kernel" in name
            or "hybrid2d_first_pass" in name or "hybrid2d_second_pass" in name):
        return "ntt"
    return "non_ntt"


def interval_stats(intervals):
    """Kernel sums may double-count overlap; union and span do not."""
    if not intervals:
        return {"kernel_sum_us": 0.0, "kernel_busy_us": 0.0, "gpu_span_us": 0.0}
    ordered = sorted(intervals)
    if any(end <= start for start, end in ordered):
        raise ValueError("kernel interval must have positive duration")
    begin, end = ordered[0]
    busy = 0
    for next_begin, next_end in ordered[1:]:
        if next_begin > end:
            busy += end - begin
            begin, end = next_begin, next_end
        else:
            end = max(end, next_end)
    busy += end - begin
    return {"kernel_sum_us": sum(b - a for a, b in ordered) / 1000,
            "kernel_busy_us": busy / 1000,
            "gpu_span_us": (max(b for _, b in ordered) - ordered[0][0]) / 1000}


def columns(connection, table):
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def analyze(database):
    database = Path(database).resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        required = {"NVTX_EVENTS": {"start", "end", "globalTid"},
                    "CUPTI_ACTIVITY_KIND_RUNTIME": {"start", "end", "globalTid", "correlationId", "nameId"},
                    "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "correlationId", "demangledName", "graphNodeId"},
                    "StringIds": {"id", "value"}}
        for table, fields in required.items():
            missing = fields - columns(connection, table)
            if missing:
                raise ValueError(f"{table} missing columns: {sorted(missing)}")
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        nvtx_columns = columns(connection, "NVTX_EVENTS")
        ranges = []
        identities = set()
        for row in connection.execute("SELECT * FROM NVTX_EVENTS"):
            name = row["text"] if "text" in nvtx_columns else None
            if not name and "textId" in nvtx_columns:
                name = strings.get(row["textId"])
            match = LABEL.fullmatch(name or "")
            if not match:
                continue
            scene, variant, round_text = match.groups()
            identity = (scene, variant, int(round_text))
            if identity in identities:
                raise ValueError(f"duplicate measurement identity: {identity}")
            identities.add(identity)
            if row["end"] is None or row["end"] <= row["start"] or row["globalTid"] is None:
                raise ValueError(f"incomplete measurement NVTX range: {name}")
            ranges.append(dict(scene=scene, variant=variant, round=int(round_text),
                               nvtx_start_ns=row["start"], nvtx_end_ns=row["end"],
                               global_tid=row["globalTid"], launches=[]))
        if not ranges:
            raise ValueError("no completed ckks_measure NVTX ranges")
        graph_apis = [dict(row) for row in connection.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")
            if strings.get(row["nameId"], "").startswith("cudaGraphLaunch")]
        kernels = defaultdict(list)
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            record = dict(row)
            record["name"] = strings[row["demangledName"]]
            kernels[row["correlationId"]].append(record)

    assigned = set()
    for api in graph_apis:
        owners = [item for item in ranges if item["global_tid"] == api["globalTid"]
                  and item["nvtx_start_ns"] <= api["start"]
                  and api["end"] <= item["nvtx_end_ns"]]
        if len(owners) > 1:
            raise ValueError("graph launch belongs to overlapping measurement ranges")
        if not owners:
            continue
        correlation = api["correlationId"]
        if correlation in assigned:
            raise ValueError(f"ambiguous graph-launch correlation ID: {correlation}")
        assigned.add(correlation)
        launch_kernels = kernels.get(correlation, [])
        if not launch_kernels or any(k["graphNodeId"] is None for k in launch_kernels):
            raise ValueError(f"graph launch {correlation} has missing node-level kernel activity")
        owners[0]["launches"].append((correlation, launch_kernels))

    measurements = []
    for item in sorted(ranges, key=lambda item: item["nvtx_start_ns"]):
        launches = item.pop("launches")
        if not launches:
            raise ValueError(f"measurement has no correlated graph launches: {item}")
        all_kernels = [kernel for _, records in launches for kernel in records]
        intervals = [(k["start"], k["end"]) for k in all_kernels]
        stats = interval_stats(intervals)
        grouped = defaultdict(list)
        for kernel in all_kernels:
            grouped[category(kernel["name"])].append(kernel)
        count = len(launches)
        item.update(graph_launches=count, kernel_count=len(all_kernels),
                    graph_node_count=len({k["graphNodeId"] for k in all_kernels}),
                    nvtx_host_us=(item["nvtx_end_ns"] - item["nvtx_start_ns"]) / 1000,
                    **stats)
        item["categories"] = {
            name: dict(kernel_count=len(grouped[name]),
                       kernels_per_graph=len(grouped[name]) / count,
                       kernel_sum_us=sum(k["end"] - k["start"] for k in grouped[name]) / 1000,
                       kernel_us_per_graph=sum(k["end"] - k["start"] for k in grouped[name]) / 1000 / count)
            for name in CATEGORIES}
        item["launches"] = [dict(correlation_id=correlation, kernel_count=len(records),
                                 kernels=sorted(records, key=lambda kernel: kernel["start"]),
                                 **interval_stats([(k["start"], k["end"]) for k in records]))
                            for correlation, records in launches]
        item["per_graph_mean"] = {
            name: sum(launch[name] for launch in item["launches"]) / count for name in stats}
        item["kernel_names"] = sorted({k["name"] for k in all_kernels})
        measurements.append(item)
    digest = hashlib.sha256()
    with database.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"database": str(database), "database_sha256": digest.hexdigest(),
            "timing_kind": "profiler-diagnostic",
            "boundary": "NVTX host range -> same-thread cudaGraphLaunch -> correlation ID -> GPU kernels",
            "caveat": "kernel sums include overlap; busy is interval union; span includes GPU gaps; none replaces ordinary end-to-end timing",
            "unassigned_graph_launches": len(graph_apis) - len(assigned),
            "measurements": measurements}


def loop_requests(result, n, plan_log):
    # Reuse the established CT/GS geometry and preparation-log decoders.
    from analyze_ckks_rns_workloads import fused_request, kernel_order, plan_matches_request, read_plans, signature

    plans = read_plans(plan_log)
    variants = {}
    for variant in ("native_loop", "cu_loop"):
        launches = [launch for row in result["measurements"] if row["variant"] == variant
                    for launch in row["launches"]]
        requests = []
        for launch in launches:
            kernels = [kernel for kernel in launch["kernels"] if category(kernel["name"]) == "ntt"]
            if len(kernels) % 2:
                raise ValueError("NTT kernel count is not a two-pass request sequence")
            requests.append([kernels[index:index + 2] for index in range(0, len(kernels), 2)])
        if not requests:
            raise ValueError(f"no loop requests for {variant}")
        variants[variant] = requests
    count = len(variants["cu_loop"][0])
    if any(len(requests) != count for samples in variants.values() for requests in samples):
        raise ValueError("loop NTT request counts disagree")
    total_difference = (sum(sum(k["end"] - k["start"] for pair in sample for k in pair)
                            for sample in variants["cu_loop"]) / len(variants["cu_loop"])
                        - sum(sum(k["end"] - k["start"] for pair in sample for k in pair)
                              for sample in variants["native_loop"]) / len(variants["native_loop"])) / 1000
    output = []
    geometry_fields = ("name", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                       "registersPerThread", "dynamicSharedMemory", "staticSharedMemory")
    for ordinal in range(count):
        item = {"request_ordinal": ordinal + 1}
        for variant, samples in variants.items():
            first_pair = samples[0][ordinal]
            geometry = [{key: stage.get(key) for key in geometry_fields} for stage in first_pair]
            for sample in samples[1:]:
                if [{key: stage.get(key) for key in geometry_fields} for stage in sample[ordinal]] != geometry:
                    raise ValueError("request stage geometry changes across loop replays")
            item[variant] = dict(stages=geometry,
                stage_mean_us=[sum(sample[ordinal][index]["end"] - sample[ordinal][index]["start"]
                                   for sample in samples) / len(samples) / 1000 for index in (0, 1)])
        pair = variants["cu_loop"][0][ordinal]
        request = fused_request(pair, n)
        native = variants["native_loop"][0][ordinal]
        if any(("Inverse" in kernel["name"]) != bool(request["inverse"]) for kernel in native):
            raise ValueError("native/cu request directions disagree at paired ordinal")
        item.update(B=request["B"], inverse=request["inverse"],
                    native_order=kernel_order(native[0]["name"]),
                    stage_mappings=request["stage_mappings"])
        matches = [plan for plan in plans if plan_matches_request(plan, request, n)]
        item["matching_prepared_plans"] = matches
        item["distinct_observable_plan_semantics"] = len({signature(plan) for plan in matches})
        item["excess_us"] = (sum(item["cu_loop"]["stage_mean_us"])
                             - sum(item["native_loop"]["stage_mean_us"]))
        item["ntt_excess_share"] = item["excess_us"] / total_difference if total_difference else None
        output.append(item)
    return {"N": n, "plan_log": str(plan_log), "requests": output,
            "ntt_excess_us": total_difference,
            "limit": "chronological pairing; prepared matches do not identify runtime domain/offset arrays; multiple semantic matches remain ambiguous"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, help="Write the analysis JSON to this path")
    parser.add_argument("--ntt-size", type=int, help="Decode two-pass loop requests at this N")
    parser.add_argument("--plan-log", type=Path, help="Match loop requests to preparation observations")
    args = parser.parse_args()
    if bool(args.ntt_size) != bool(args.plan_log):
        parser.error("--ntt-size and --plan-log are required together")
    result = analyze(args.database)
    if args.ntt_size:
        result["loop_ntt_requests"] = loop_requests(result, args.ntt_size, args.plan_log)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
