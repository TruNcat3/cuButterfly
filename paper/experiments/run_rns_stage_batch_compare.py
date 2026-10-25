#!/usr/bin/env python3
"""Run bounded RNS stage controls, optionally including device packing costs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading

from run_heongpu_ckks_e2e import (
    file_identity, gpu_observation, read_only_probe, stop_own_process,
    target_gpu_uuid, utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
MODES = {"cu-loop", "cu-packed", "native-loop", "native-packed"}
PACKING_MODES = {"cu-pack-transform-unpack", "native-pack-transform-unpack"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def records(text):
    parsed = {}
    for fields in csv.reader(text.splitlines()):
        require(len(fields) % 2 == 1 and len(fields) >= 3, "Malformed benchmark record")
        keys = fields[1::2]
        require(len(keys) == len(set(keys)), "Duplicate benchmark field")
        parsed.setdefault(fields[0], []).append(dict(zip(keys, fields[2::2])))
    return parsed


def validate(text, case, repetitions, include_packing=False):
    data = records(text)
    modes = MODES | PACKING_MODES if include_packing else MODES
    results, samples = data.get("result", []), data.get("sample", [])
    require(len(results) == len(modes) and {r["mode"] for r in results} == modes, "Unexpected result modes")
    require(len(samples) == 10 * len(modes), "Unexpected round sample count")
    validations = data.get("validation", [])
    require(len(validations) == 3 * len(modes) and
            {(v["mode"], v["trial"], v["exact"]) for v in validations} ==
            {(m, str(t), "1") for m in modes for t in range(3)}, "Missing exact A/B/A validation")
    fixture = data.get("fixture", [])
    require(len(fixture) == 1 and fixture[0]["moduli"] == case["moduli"].replace(",", ":") and
            fixture[0]["forward_roots"] == case["roots"].replace(",", ":") and
            fixture[0]["boundary"] == ("BR-to-natural" if case["inverse"] else "natural-to-BR") and
            fixture[0]["scope"] == ("same-stage-layout-cost-control" if include_packing else "same-stage-prepacked-control"),
            "Unexpected stage fixture")
    domains = data.get("domains", [])
    domain_count = str(1 if case["inverse"] else 2)
    require(domains == [{"cu_unique": domain_count, "native_tables": domain_count}],
            "Unexpected actual domain count")
    cu_modes = {"cu-loop", "cu-packed"} | ({"cu-pack-transform-unpack"} if include_packing else set())
    require(len(data.get("mapping", [])) == len(cu_modes) and
            {r["mode"] for r in data["mapping"]} == cu_modes, "Missing cu mappings")
    if include_packing:
        packing = data.get("packing", [])
        require(len(packing) == 1 and packing[0]["request_allocations"] == str(case["requests"]) and
                packing[0]["separate_bytes"] == str(2 * case["requests"] * (1 << 14) * 8) and
                packing[0]["arena_bytes"] == packing[0]["separate_bytes"] and
                packing[0]["scope"] == "stage-only-not-full-CKKS", "Unexpected separate allocation layout")
        by_mode = {r["mode"]: r for r in results}
        for prefix in ("cu", "native"):
            require(int(by_mode[prefix + "-pack-transform-unpack"]["kernel_nodes"]) ==
                    int(by_mode[prefix + "-packed"]["kernel_nodes"]) + 2, "Missing pack/unpack Graph nodes")
    for row in results:
        for key, value in dict(log_n=14, rows_per_request=2, request_count=case["requests"],
                               polynomials=2 * case["requests"], inverse=case["inverse"],
                               repetitions=repetitions, rounds=10, exact=1).items():
            require(row[key] == str(value), f"Unexpected {key}: {row}")
        mode_samples = [s for s in samples if s["mode"] == row["mode"]]
        require(len(mode_samples) == 10 and {s["round"] for s in mode_samples} ==
                {str(i) for i in range(10)}, "Missing mode rounds")
        require(float(row["warm_ms"]) >= 100 and int(row["kernel_nodes"]) > 0 and
                int(row["repeated_graph_nodes"]) >= int(row["kernel_nodes"]) * repetitions,
                "Invalid warmup or Graph node count")
        kernels = [r for r in data.get("kernel", []) if r["mode"] == row["mode"]]
        require(len(kernels) == int(row["kernel_nodes"]), "Missing actual kernel mappings")
        for sample in mode_samples:
            ms, perpoly = float(sample["full_stage_ms"]), float(sample["perpoly_ms"])
            require(math.isfinite(ms) and ms > 0 and
                    math.isclose(perpoly, ms / (2 * case["requests"]), rel_tol=1e-8), "Invalid sample timing")
        median = statistics.median(float(s["full_stage_ms"]) for s in mode_samples)
        require(math.isclose(median, float(row["median_ms"]), rel_tol=1e-8) and
                math.isclose(float(row["perpoly_ms"]), median / (2 * case["requests"]), rel_tol=1e-8),
                "Invalid summary timing")
    for round_index in range(10):
        order = [s for s in samples if int(s["round"]) == round_index]
        require(len(order) == len(modes) and {s["order"] for s in order} ==
                {str(i) for i in range(len(modes))}, "Invalid round order")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-static", type=Path,
                        default=ROOT / "results/e13_assets/heongpu-build-e13/thirdparty/GPU-NTT/src/libntt-1.0.a")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--include-packing", action="store_true")
    parser.add_argument("--request-counts", default="1,8,32")
    args = parser.parse_args()
    require(args.repetitions > 0, "repetitions must be positive")
    request_counts = [int(value) for value in args.request_counts.split(",")]
    require(request_counts and len(request_counts) == len(set(request_counts)) and
            set(request_counts) <= {1, 8, 32}, "request-counts must be unique values from 1,8,32")
    binary, library = args.binary.resolve(), args.library.resolve()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    gpu_uuid = target_gpu_uuid(env)
    env["LD_LIBRARY_PATH"] = str(library.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    paths = [binary, library, args.native_static.resolve(), Path(__file__).resolve(),
             ROOT / "benchmarks/rns_stage_batch_compare.cu", ROOT / "include/cuntt/rns_ntt.hpp"]
    artifacts = [file_identity(p) for p in paths]
    require(all(a.get("sha256") for a in artifacts), "Missing artifact")
    # mkdir without exist_ok reserves a fresh directory, including on failures.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = dict(started_utc=utc_now(), status="running", gpu_uuid=gpu_uuid,
                    scope=("Stage-only independent request allocations versus prepacked arena, with explicit gather-transform-scatter cost; no full CKKS claim"
                           if args.include_packing else "Stage-only prepacked independent requests: no CKKS end-to-end claim and no application packing assumption"),
                    include_packing=args.include_packing, request_counts=request_counts,
                    protocol="Same process modes; CPU exact A/B/A Graph validation at actual output buffers; resets outside timing; >=100ms per-mode joint warmup; 10 rotating/reversing rounds",
                    artifacts=artifacts, source_binary_correspondence="not_verified",
                    environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")}, executions=[])
    write_json(manifest_path, manifest)
    process = None
    try:
        hardware = read_only_probe(["nvidia-smi", f"--id={gpu_uuid}",
                                   "--query-gpu=uuid,name,memory.total,driver_version,pci.bus_id",
                                   "--format=csv,noheader,nounits"])
        linkage = read_only_probe(["env", "LD_LIBRARY_PATH=" + env["LD_LIBRARY_PATH"], "ldd", str(binary)])
        manifest.update(hardware=hardware, linkage=linkage)
        require(hardware.get("returncode") == 0 and gpu_uuid.lower() in hardware.get("stdout", "").lower(), "GPU identity query failed")
        require(linkage.get("returncode") == 0 and str(library) in linkage.get("stdout", "") and
                "not found" not in linkage.get("stdout", ""), "Requested shared library not resolved")
        for inverse in (0, 1):
            for requests in request_counts:
                case = dict(inverse=inverse, requests=requests,
                            moduli="1125899904679937,1125899904679937" if inverse else "1125899903991809,1099508121601",
                            roots="184459094098,184459094098" if inverse else "125929543876,13296178")
                label = f"n14_rows2_requests{requests}_inverse{inverse}"
                command = [str(binary), "--log-n", "14", "--rows-per-request", "2", "--request-count", str(requests),
                           "--inverse", str(inverse), "--moduli", case["moduli"], "--roots", case["roots"],
                           "--repetitions", str(args.repetitions)]
                if args.include_packing:
                    command += ["--include-packing", "1"]
                execution = dict(case=case, argv=command, started_utc=utc_now(), observations=[], status="running",
                                 stdout=label + ".stdout.log", stderr=label + ".stderr.log")
                manifest["executions"].append(execution)
                write_json(manifest_path, manifest)
                require([file_identity(p) for p in paths] == artifacts, "Artifact changed before run")
                before = gpu_observation(gpu_uuid)
                execution["observations"].append(before)
                require(before["query_ok"] and not before["foreign_processes"], "GPU busy or occupancy query failed")
                stop = threading.Event()
                violations = []

                def monitor():
                    try:
                        while not stop.wait(1.0):
                            observation = gpu_observation(gpu_uuid, process.pid)
                            execution["observations"].append(observation)
                            if not observation["query_ok"] or observation["foreign_processes"]:
                                violations.append(observation)
                                execution["child_stop"] = stop_own_process(process)
                                return
                    except Exception as error:
                        violations.append({"error": str(error)})
                        execution["child_stop"] = stop_own_process(process)

                with (output / execution["stdout"]).open("xb") as stdout, (output / execution["stderr"]).open("xb") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                    execution["pid"] = process.pid
                    thread = threading.Thread(target=monitor, daemon=True)
                    thread.start()
                    try:
                        execution["returncode"] = process.wait()
                    finally:
                        stop.set()
                        thread.join()
                        if process.poll() is None:
                            execution["child_stop"] = stop_own_process(process)
                after = gpu_observation(gpu_uuid, process.pid)
                execution["observations"].append(after)
                execution["completed_utc"] = utc_now()
                execution["output_identities"] = [file_identity(output / execution[k]) for k in ("stdout", "stderr")]
                require(not violations and after["query_ok"] and not after["foreign_processes"], "GPU contention or monitor failure; evidence invalid")
                require(execution["returncode"] == 0, "Benchmark returned nonzero")
                require([file_identity(p) for p in paths] == artifacts, "Artifact changed during run")
                parsed = validate((output / execution["stdout"]).read_text(), case, args.repetitions, args.include_packing)
                execution.update(status="exact", exact=True, performance_eligible=True, parsed=parsed)
                write_json(output / (label + ".parsed.json"), parsed)
                write_json(manifest_path, manifest)
                print(f"{label}: exact, {len(parsed['result'])} modes and {len(parsed['sample'])} samples", flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        if process is not None and process.poll() is None:
            manifest["child_stop"] = stop_own_process(process)
        manifest.update(status="failed", error=str(error))
        if manifest["executions"] and manifest["executions"][-1]["status"] == "running":
            manifest["executions"][-1].update(status="failed", exact=False, performance_eligible=False)
        raise
    finally:
        manifest["completed_utc"] = utc_now()
        write_json(manifest_path, manifest)
    print(str(manifest_path))


if __name__ == "__main__":
    main()
