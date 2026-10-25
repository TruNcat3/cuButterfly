#!/usr/bin/env python3
"""CPU-only audit of prepared CKKS RNS plans and executed application graphs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from summarize_heongpu_ckks_e2e import require_performance_evidence


RUNS = ("ckks_graph_method1_native_rns", "ckks_graph_method1_specialized",
        "ckks_graph_method2_specialized", "ckks_specialized_trace_diagnostic")
MAPPING = ("n1_log", "core_family", "rows_per_block", "threads_per_block",
           "second_rows_per_block", "second_threads_per_block", "compute_unit",
           "cross_twiddle", "physical_boundary_traversal", "arithmetic")
FUSED = re.compile(r"stage_kernel<\(unsigned int\)(\d+), \(unsigned int\)(\d+), \(bool\)([01])"
                   r"(?:, \(unsigned int\)([12])(?:, \(bool\)([01]), \(bool\)([01])"
                   r"(?:, \(bool\)([01]))?)?)?>")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_plans(path):
    plans = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("ntt_adapter_plan,"):
            plan = dict(item.split("=", 1) for item in line.split(",")[1:] if "=" in item)
            plan.update(line=line_number, domain_parameters=[])
            plans.append(plan)
        elif line.startswith("ntt_adapter_domain,"):
            domain = dict(item.split("=", 1) for item in line.split(",")[1:])
            assert plans and all(domain[key] == plans[-1][key] for key in ("N", "B", "inverse"))
            plans[-1]["domain_parameters"].append({key: int(domain[key]) for key in ("domain", "q", "forward_psi")})
    for plan in plans:
        plan.setdefault("second_rows_per_block", "0")
        plan.setdefault("second_threads_per_block", "0")
        plan.setdefault("arithmetic", "shoup")
        assert len(plan["domain_parameters"]) == int(plan["domains"])
        plan["input_order"] = "bit_reversed" if plan["inverse"] == "1" else "natural"
        plan["output_order"] = "natural" if plan["inverse"] == "1" else "bit_reversed"
    return plans


def signature(plan):
    # Offsets, ordered-index arrays and pointer cache keys are absent from old logs.
    return (plan["N"], plan["B"], plan["inverse"],
            tuple((d["q"], d["forward_psi"]) for d in plan["domain_parameters"]))


def allocation_estimates(plans):
    estimates = []
    for n in sorted({int(p["N"]) for p in plans}):
        same_n = [p for p in plans if int(p["N"]) == n]
        domains = sum(int(p["domains"]) for p in same_n)
        unique = {(p["inverse"], d["q"], d["forward_psi"])
                  for p in same_n for d in p["domain_parameters"]}
        estimates.append(dict(N=n, preparation_records=len(same_n), domain_instances=domains,
                              unique_directional_domain_tables=len(unique),
                              fused_roots_shoup_bytes=domains * n * 16,
                              all_family_coefficient_bytes=domains * (n * 64 - 16),
                              scratch_bytes=sum(int(p["B"]) for p in same_n) * n * 8,
                              hypothetical_shared_fused_bytes=len(unique) * n * 16,
                              hypothetical_shared_all_family_bytes=len(unique) * (n * 64 - 16)))
    return estimates


def summarize_run(source, stem):
    metadata = json.loads((source / (stem + ".metadata.json")).read_text())
    require_performance_evidence(metadata, source / (stem + ".csv"))
    plans = read_plans(source / (stem + ".log"))
    with (source / (stem + ".csv")).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        repetitions, batch = int(row["repetitions"]), int(row["batch"])
        calls, polynomials = int(row["replacement_calls"]), int(row["polynomials"])
        assert calls % (repetitions * batch) == polynomials % (repetitions * batch) == 0
        row.update(calls_per_graph=calls // repetitions,
                   polynomials_per_graph=polynomials // repetitions,
                   calls_per_ciphertext=calls // repetitions // batch,
                   polynomials_per_ciphertext=polynomials // repetitions // batch,
                   mean_B_per_request=polynomials / calls)
    shapes = Counter((int(p["N"]), int(p["B"]), int(p["inverse"]), int(p["domains"])) for p in plans)
    mappings = Counter(tuple(p[key] for key in MAPPING) for p in plans)
    return dict(rows=rows, prepared_observations=plans,
                preparation_records=len(plans),
                observable_semantic_signatures=len({signature(p) for p in plans}),
                preparation_shape_counts=[dict(N=k[0], B=k[1], inverse=k[2], domains=k[3], preparation_records=v)
                                          for k, v in sorted(shapes.items())],
                preparation_mapping_counts=[dict(**dict(zip(MAPPING, k)), preparation_records=v)
                                            for k, v in sorted(mappings.items())],
                modeled_preparation_allocations_by_N=allocation_estimates(plans),
                slowest_rows=sorted(rows, key=lambda r: float(r["speedup"]))[:8])


def kernel_order(name):
    if "PolyOrdered" in name:
        return "polynomial"
    if "ModulusOrdered" in name:
        return "modulus"
    return "contiguous"


def fused_request(stages, n):
    if len(stages) != 2:
        raise ValueError("fused request requires exactly two stages")
    mappings = []
    for stage_index, stage in enumerate(stages):
        match = FUSED.search(stage["name"])
        if match is None:
            raise ValueError(f"unsupported fused kernel template: {stage['name']}")
        local_log, rows, inverse, radix_log, barrett, compact, first = match.groups()
        if first is not None and (first == "1") != (stage_index == 0):
            raise ValueError("fused First template role disagrees with stage execution order")
        local_log, rows, inverse = map(int, (local_log, rows, inverse))
        numerator, divisor = stage["gridX"] * rows, n >> local_log
        assert divisor and numerator % divisor == 0, "Nonintegral per-stage polynomial count"
        mappings.append(dict(stage_index=stage_index, B=numerator // divisor,
                             inverse=inverse, local_log=local_log, rows_per_block=rows,
                             compute_unit="radix4" if radix_log == "2" else "radix2",
                             arithmetic="barrett" if barrett == "1" else "shoup",
                             compact=compact == "1",
                             threads_per_block=stage["blockX"],
                             grid=[stage[k] for k in ("gridX", "gridY", "gridZ")],
                             block=[stage[k] for k in ("blockX", "blockY", "blockZ")],
                             registers_per_thread=stage["registersPerThread"],
                             dynamic_shared_bytes=stage["dynamicSharedMemory"],
                             kernel_name=stage["name"]))
        if first is not None:
            mappings[-1]["first"] = first == "1"
    a, b = mappings
    assert a["B"] == b["B"] and a["inverse"] == b["inverse"], "Passes disagree on request semantics"
    assert (1 << (a["local_log"] + b["local_log"])) == n, "Pass logs do not cover full transform"
    request = dict(B=a["B"], inverse=a["inverse"], stage_mappings=mappings,
                   n1_log=a["local_log"] if a["inverse"] else b["local_log"])
    for key in ("compute_unit", "rows_per_block", "threads_per_block"):
        request[key] = a[key] if a[key] == b[key] else "mixed"
    request["stage_mapping_key"] = json.dumps(mappings, sort_keys=True, separators=(",", ":"))
    return request


def plan_matches_request(plan, request, n):
    if (int(plan["N"]) != n or int(plan["B"]) != request["B"]
            or int(plan["inverse"]) != request["inverse"]
            or plan["core_family"] not in ("fused-ctgs", "gpuntt-merge")
            or int(plan["n1_log"]) != request["n1_log"]):
        return False
    log_n = n.bit_length() - 1
    n1_log = int(plan["n1_log"])
    local_logs = [n1_log, log_n - n1_log] if request["inverse"] else [log_n - n1_log, n1_log]
    for index, stage in enumerate(request["stage_mappings"]):
        # Overrides refer to physical execution order, including inverse NTTs.
        rows = int(plan["rows_per_block"])
        threads = int(plan["threads_per_block"])
        if index == 1:
            rows = int(plan.get("second_rows_per_block", "0")) or rows
            threads = int(plan.get("second_threads_per_block", "0")) or threads
        if (stage["local_log"] != local_logs[index] or stage["rows_per_block"] != rows
                or stage["threads_per_block"] != threads
                or stage["compute_unit"] != plan["compute_unit"]
                or stage["arithmetic"] != plan.get("arithmetic", "shoup")):
            return False
    return True


def trace_workloads(database, summary, plans, csv_rows):
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    records = connection.execute("""
        SELECT k.graphNodeId,k.correlationId,k.start,k.end,n.value AS name,
               k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
               k.dynamicSharedMemory,k.registersPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds n ON n.id=k.demangledName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        JOIN StringIds rn ON rn.id=r.nameId
        WHERE rn.value LIKE 'cudaGraphLaunch%' AND k.graphNodeId IS NOT NULL
    """).fetchall()
    connection.close()
    graphs = defaultdict(lambda: defaultdict(list))
    for row in records:
        graphs[row["graphNodeId"] >> 32][row["correlationId"]].append(dict(row))
    metadata = {g["graph_id"]: g for g in summary["application_graphs"]}
    scenes = []
    for scene in sorted({g["scene"] for g in metadata.values()}):
        pair = {g["backend"]: g for g in metadata.values() if g["scene"] == scene}
        csv_row = next(r for r in csv_rows if r["scene"] == scene)
        batch, n = int(csv_row["batch"]), int(csv_row["N"])
        all_launches = {}
        for backend, graph in pair.items():
            launch_records = sorted(graphs[graph["graph_id"]].values(), key=lambda rows: min(r["start"] for r in rows))
            assert len(launch_records) == graph["launches"]
            launches = []
            for rows in launch_records:
                ntt = [r for r in sorted(rows, key=lambda r: r["start"])
                       if (FUSED.search(r["name"]) if backend == "cubutterfly" else "gpuntt::" in r["name"])]
                assert len(ntt) == csv_row["calls_per_graph"] * 2
                requests = []
                for i in range(0, len(ntt), 2):
                    a, b = ntt[i:i + 2]
                    request = dict(kernel_ns=a["end"] - a["start"] + b["end"] - b["start"],
                                   stage_ns=[a["end"] - a["start"], b["end"] - b["start"]])
                    if backend == "cubutterfly":
                        request.update(fused_request((a, b), n))
                    else:
                        assert ("Inverse" in a["name"]) == ("Inverse" in b["name"])
                        assert kernel_order(a["name"]) == kernel_order(b["name"])
                        request.update(inverse=int("Inverse" in a["name"]), ordering=kernel_order(a["name"]))
                    requests.append(request)
                launches.append(requests)
            all_launches[backend] = launches
        launches = all_launches["cubutterfly"]
        native = all_launches["native"]
        keys = ("B", "inverse", "n1_log", "rows_per_block", "threads_per_block", "compute_unit", "stage_mapping_key")
        prototype = [tuple(r[k] for k in keys) for r in launches[0]]
        assert all([tuple(r[k] for k in keys) for r in launch] == prototype for launch in launches)
        calls_per_ct = csv_row["calls_per_ciphertext"]
        assert all(prototype[i:i + calls_per_ct] == prototype[:calls_per_ct]
                   for i in range(0, len(prototype), calls_per_ct))
        assert sum(r["B"] for r in launches[0]) == csv_row["polynomials_per_graph"]
        grouped = defaultdict(list)
        sequence = []
        for i, request in enumerate(launches[0]):
            assert all(launch[i]["inverse"] == request["inverse"] for launch in native)
            orders = {launch[i]["ordering"] for launch in native}
            assert len(orders) == 1
            ordering = next(iter(orders))
            matching = [p for p in plans if plan_matches_request(p, request, n)]
            domains = sorted({int(p["domains"]) for p in matching})
            candidates = sorted({tuple((d["q"], d["forward_psi"]) for d in p["domain_parameters"]) for p in matching})
            key = (*prototype[i], ordering)
            grouped[key].append(i)
            if i < calls_per_ct:
                sequence.append(dict(position=i, B=request["B"], inverse=request["inverse"],
                                     ordering=ordering, domain_count_candidates=domains,
                                     domain_identity_candidates=candidates,
                                     matching_preparation_lines=[p["line"] for p in matching],
                                     stage_mappings=request["stage_mappings"],
                                     n1_log=request["n1_log"],
                                     compute_unit=request["compute_unit"],
                                     rows_per_block=request["rows_per_block"],
                                     threads_per_block=request["threads_per_block"]))
        distributions = []
        total_ntt_us = pair["cubutterfly"]["ntt_sum_us"]
        for key, indices in grouped.items():
            count = len(indices)
            cu_ns = sum(launch[i]["kernel_ns"] for launch in launches for i in indices) / len(launches)
            native_ns = sum(launch[i]["kernel_ns"] for launch in native for i in indices) / len(native)
            item = dict(zip((*keys, "ordering"), key))
            item["stage_mappings"] = json.loads(item.pop("stage_mapping_key"))
            item.update(calls_per_graph=count, calls_per_ciphertext=count / batch,
                        actual_trace_calls=count * len(launches),
                        polynomials_per_graph=count * item["B"],
                        cu_us_per_graph=cu_ns / 1000, native_us_per_graph=native_ns / 1000,
                        cu_us_per_request=cu_ns / count / 1000,
                        native_us_per_request=native_ns / count / 1000,
                        cu_over_native=cu_ns / native_ns,
                        excess_us_per_graph=(cu_ns - native_ns) / 1000,
                        ntt_time_fraction=cu_ns / 1000 / total_ntt_us,
                        stage_us_per_request=[sum(launch[i]["stage_ns"][s] for launch in launches for i in indices)
                                              / len(launches) / count / 1000 for s in range(2)])
            distributions.append(item)
        assert abs(sum(r["cu_us_per_graph"] for r in distributions) - total_ntt_us) < 1e-6
        scenes.append(dict(scene=scene, N=n, ciphertext_batch=batch,
                           native_graph_id=pair["native"]["graph_id"], cu_graph_id=pair["cubutterfly"]["graph_id"],
                           observed_graph_launches=len(launches),
                           calls_per_graph=csv_row["calls_per_graph"],
                           polynomials_per_graph=csv_row["polynomials_per_graph"],
                           sequence_per_ciphertext=sequence,
                           workloads=sorted(distributions, key=lambda r: -r["excess_us_per_graph"])))
    return scenes


def report(result):
    fractions = [row["ntt_fraction_of_kernel_increase"] for row in result["trace_summary"]["scene_comparisons"]
                 if row["ntt_fraction_of_kernel_increase"] is not None and row["total_kernel_increase_us"] > 0]
    fraction_note = (f"For scenes with positive total kernel-time differences, NTT contributes "
                     f"{min(fractions)*100:.1f}-{max(fractions)*100:.1f}% of that difference in this trace."
                     if fractions else "No scene has a positive total kernel-time difference for this attribution.")
    lines = ["# CKKS RNS workload distribution audit", "",
             "CPU-only analysis of the named evidence. All trace times are profiler diagnostics.", "",
             "## Findings", "",
             fraction_note + " Request groups below localize the execution difference; they do not establish selector regret.", "",
             "Outer ciphertext batch is a scalar public-API loop. Increasing it repeats the same small RNS requests; "
             "it does not turn B=2..12 requests into a cross-ciphertext NTT batch.", "",
             "## Evidence boundaries", "",
             "`replacement_calls / repetitions` and `polynomials / repetitions` describe one captured graph. "
             "Graph replay does not execute the host counters; CSV scales capture counts by timed repetitions. "
             "Trace launch counts include validation/warmup plus timing and are separately reported.", "",
             "Plan log entries are cache preparation observations. The distinct observable semantic signatures "
             "omit cache pointers, polynomial offsets, and ordering-index contents, so they are neither actual "
             "invocation frequencies nor a certified count of unique prepared cache plans. Duplicate driver logs "
             "are excluded. Input hashes are in JSON.", "",
             f"Trace inputs: `{result['trace_inputs']['database']}`, `{result['trace_inputs']['run']}` and "
             f"`{result['trace_inputs']['summary']}`. Coverage and launch counts are listed per scene below. "
             "No per-request timing distribution is inferred outside the selected trace.", "",
             "Trace device selection recorded in metadata: "
             f"`CUDA_VISIBLE_DEVICES={result['trace_hardware']['CUDA_VISIBLE_DEVICES']}`. "
             "The recorded hardware query is retained in JSON; device identity is specific to this run.", "",
             "## Prepared observations", "",
             "| Run | CSV rows | Preparation records | Observable semantic signatures |", "|---|---:|---:|---:|"]
    for name, run in result["runs"].items():
        lines.append(f"| {name} | {len(run['rows'])} | {run['preparation_records']} | {run['observable_semantic_signatures']} |")
    lines += ["", "Full domain parameters, mapping counts and shape counts appear in [workloads.json](workloads.json).", "",
              "## Weak measured application rows", "",
              "| Run / scene | N | Ciphertext batch | Native / cu | cu minus native (us) |", "|---|---:|---:|---:|---:|"]
    for name in result["runs"]:
        for row in result["runs"][name]["slowest_rows"][:4]:
            lines.append(f"| {name} / {row['scene']} | {row['N']} | {row['batch']} | {float(row['speedup']):.4f} | {(float(row['cubutterfly_ms'])-float(row['native_ms']))*1000:.3f} |")
    lines += ["", "## Small-B residuals", "",
              "These are B=2 request groups actually executed in the selected trace. "
              "Their excess is cuButterfly minus native kernel time per graph; profiler times are diagnostic. "
              "The contribution percentage uses that scene's total NTT difference, preserving signs.", "",
              "| Scene / direction | Calls per ciphertext | Selected core | Excess us/graph | NTT difference share |",
              "|---|---:|---|---:|---:|"]
    for scene in result["trace_workloads"]:
        total_excess = sum(row["excess_us_per_graph"] for row in scene["workloads"])
        for row in scene["workloads"]:
            if row["B"] != 2:
                continue
            core = "; ".join(f"{s['compute_unit']},rows{s['rows_per_block']},t{s['threads_per_block']}"
                             for s in row["stage_mappings"])
            fraction = f"{100*row['excess_us_per_graph']/total_excess:.1f}%" if total_excess else "undefined"
            lines.append(f"| {scene['scene']} / {'inverse' if row['inverse'] else 'forward'} | "
                         f"{row['calls_per_ciphertext']:g} | {core} | {row['excess_us_per_graph']:.3f} | {fraction} |")
    lines += ["", "## Actual traced request distribution", "",
              "B below means RNS polynomials in one adapter request. F/I mean forward/inverse. "
              "Every request executes two NTT stages. B is reconstructed from "
              "`gridX * Rows / 2^(logN-LocalLog)` independently for each pass; both B values must agree "
              "and the request total must match CSV polynomials. All observed launches have the same request "
              "sequence, and the sequence repeats identically across ciphertexts. Stage mappings are preserved "
              "separately in JSON, including local log, radix, rows, threads, grid and resource counts. "
              "Preparation matching checks each execution stage separately; second_rows_per_block and "
              "second_threads_per_block inherit their first-stage value when zero or absent. Inverse NTT "
              "reverses the local-log order, while override fields continue to refer to physical execution order. "
              "Native request pairing is chronological and requires matching direction and request count. "
              "Native kernel names expose contiguous/modulus/polynomial ordering. "
              "They do not expose ordered index values.", ""]
    for scene in result["trace_workloads"]:
        lines += [f"### {scene['scene']}", "",
                  f"Per graph: {scene['calls_per_graph']} requests, {scene['polynomials_per_graph']} polynomials. "
                  f"N={scene['N']}, ciphertext batch={scene['ciphertext_batch']}; "
                  f"observed: {scene['observed_graph_launches']} launches. All recognized replacement stages are fused CT/GS.", "",
                  "| Request B/direction/ordering | Calls per ciphertext | Stages: log,rows,threads,radix | cu / native us per graph | Excess us |",
                  "|---|---:|---:|---:|---:|"]
        for row in scene["workloads"]:
            direction = "I" if row["inverse"] else "F"
            stages = "; ".join(f"{s['local_log']},{s['rows_per_block']},{s['threads_per_block']},{s['compute_unit']}"
                               for s in row["stage_mappings"])
            lines.append(f"| {row['B']}/{direction}/{row['ordering']} | {row['calls_per_ciphertext']:g} | {stages} | {row['cu_us_per_graph']:.3f} / {row['native_us_per_graph']:.3f} | {row['excess_us_per_graph']:.3f} |")
        seq = []
        for r in scene["sequence_per_ciphertext"]:
            domains = "/".join(map(str, r["domain_count_candidates"])) or "unknown"
            seq.append(f"{r['B']}{'I' if r['inverse'] else 'F'}(D={domains},{r['ordering']})")
        lines += ["", "Per-ciphertext NTT order: `" + " -> ".join(seq) + "`.", ""]
    lines += ["## Root-table allocation model", "",
              "The inspected RnsPlan constructor allocates per-domain hybrid twiddle/Shoup "
              "(2*(N-1)*8 bytes), cyclic roots/Shoup (2*N*8), negacyclic boundary/Shoup (2*N*8), "
              "and fused directed roots/Shoup (2*N*8): total 64*N-16 bytes/domain, plus metadata "
              "and B*N*8 bytes scratch/plan. All families are allocated before selecting a winner. "
              "A selected fused kernel accesses its fused roots/Shoup, not all of these allocated arrays.", "",
              "The table multiplies observed constructor log records by this allocation model. "
              "It assumes each record represents a live cached plan at that N; the inspected benchmark "
              "clears the cache at each N boundary, so rows at different N must not be added as a peak. "
              "No cudaMalloc accounting was collected. Metadata declares source/binary correspondence "
              f"{result['allocation_model_provenance']['source_binary_correspondence']}; source hashes and hash matches are retained in JSON. Thus these are modeled "
              "allocations rather than measured memory usage.", "",
              "| Run / N | Plan records / domain instances | All coefficient MiB | Fused coefficient MiB | Shared fused MiB |",
              "|---|---:|---:|---:|---:|"]
    for name in result["runs"]:
        for row in result["runs"][name]["modeled_preparation_allocations_by_N"]:
            lines.append(f"| {name} / {row['N']} | {row['preparation_records']} / {row['domain_instances']} | {row['all_family_coefficient_bytes']/2**20:.3f} | {row['fused_roots_shoup_bytes']/2**20:.3f} | {row['hypothetical_shared_fused_bytes']/2**20:.3f} |")
    lines += ["", "Shared fused is a hypothetical semantic deduplication by (N, direction, q, psi). "
              "It is not an implemented memory measurement. The native CKKS context owns shared forward/inverse "
              "uint64 root tables; it does not store cuButterfly's additional per-plan Shoup arrays. "
              "The fused table capacity can be reduced by sharing, but this does not establish a latency benefit. "
              "Static allocated bytes do not reveal accessed addresses, L2 residency, hit rate, or bandwidth. "
              "An independent NCU measurement with different moduli, mapping, or application state "
              "cannot establish L2 pressure in this trace, even when it uses the same GPU.", "",
              "## What the evidence supports next", "",
              "Compare the largest excess-time request groups before expanding a benchmark matrix. "
              "The trace can rank request groups, direction, rows, threads and stage costs. "
              "It cannot prove that another already measured candidate would win in the application, "
              "because old logs retain only the winner and confirmed time. A measured-rns-request result "
              "with selection_calibrated=0 is request tuning, not evidence that selection never measured candidates.", "",
              "Distinguish three questions: whether a faster core/mapping was executable, whether it entered "
              "the candidate set and was timed, and whether the selected candidate remains fastest in the captured "
              "application. This dataset cannot fully separate them. Standalone mapping timings or NCU "
              "measurements with different moduli, mapping, or application state do not establish "
              "selection regret for this application trace.", "",
              "Minimal instrumentation: emit a stable plan_id, full semantic hash (N, direction, ring, "
              "input/output order, domains, polynomial domain/offset list, adapter ordering/index hash), "
              "selected mapping_id, and each measured candidate's legal/compiled/measured status plus timings. "
              "During graph capture emit one request manifest with scene, ciphertext index, request ordinal, "
              "plan_id and its graph-node/stage association; retain the replay count separately. This avoids "
              "per-replay host instrumentation and enables exact per-plan frequency and selection-regret analysis.", "",
              "## Reproduce", "", "```bash",
              "python paper/experiments/analyze_ckks_rns_workloads.py \\",
              f"  --source {result['source']} \\",
              f"  --trace-stem {result['trace_inputs']['stem']} \\",
              f"  --trace-run {result['trace_inputs']['run']} \\",
              f"  --trace-summary {result['trace_inputs']['summary']} \\",
              f"  --runs {','.join(result['runs'])} \\",
              f"  --output {result['output']}", "```", "",
              "Analysis uses SQLite read-only mode. Assertions verify stage pairing, replay sequence stability, "
              "cross-ciphertext sequence repetition, CSV call/poly totals and trace-summary NTT time sums. "
              "No GPU execution is required.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-stem", default="ckks_specialized_trace")
    parser.add_argument("--trace-run", default="ckks_specialized_trace_diagnostic")
    parser.add_argument("--trace-summary", default="trace_summary.json")
    parser.add_argument("--runs", help="Comma-separated canonical run stems; must include --trace-run")
    args = parser.parse_args()
    runs = tuple(stem.strip() for stem in args.runs.split(",") if stem.strip()) if args.runs else (*RUNS[:-1], args.trace_run)
    if not runs or len(set(runs)) != len(runs) or args.trace_run not in runs:
        parser.error("--runs must contain unique stems including --trace-run")
    database = args.source / (args.trace_stem + ".sqlite")
    summary_path = args.source / args.trace_summary
    paths = [args.source / (stem + ext) for stem in runs for ext in (".log", ".csv")]
    paths += [summary_path, database]
    paths += [args.source / (stem + ".metadata.json") for stem in runs]
    result = dict(schema_version=2, source=str(args.source), output=str(args.output),
                  trace_inputs=dict(stem=args.trace_stem, database=str(database),
                                    run=args.trace_run, summary=args.trace_summary),
                  input_sha256={str(path): digest(path) for path in paths},
                  runs={stem: summarize_run(args.source, stem) for stem in runs})
    root = Path(__file__).resolve().parents[2]
    observed_sources = [root / "src/rns_ntt.cu", root / "src/rns_fused_ctgs.cuh",
                        root / "adapters/heongpu/ntt_adapter.cu",
                        root / "paper/experiments/heongpu_ckks_e2e_bench.cu"]
    metadata = json.loads((args.source / (args.trace_run + ".metadata.json")).read_text())
    result["trace_hardware"] = dict(CUDA_VISIBLE_DEVICES=metadata.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
                                     nvidia_smi=metadata.get("nvidia_smi"))
    old_sources = {r["path"]: r["sha256"] for r in metadata["source_files"]}
    result["allocation_model_provenance"] = dict(
        source_binary_correspondence=metadata["source_binary_correspondence"],
        interpretation="Static coefficient capacity model, not a measured runtime working set or peak allocation",
        sources=[dict(path=str(path), observed_sha256=digest(path),
                      trace_metadata_sha256=old_sources.get(str(path)),
                      matches_trace_metadata=digest(path) == old_sources.get(str(path))) for path in observed_sources])
    summary = json.loads(summary_path.read_text())
    result["trace_summary"] = summary
    trace_run = result["runs"][args.trace_run]
    result["trace_workloads"] = trace_workloads(database, summary,
                                               trace_run["prepared_observations"], trace_run["rows"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "workloads.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(report(result))
    print(f"Wrote {args.output / 'workloads.json'} and {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
