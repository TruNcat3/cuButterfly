#!/usr/bin/env python3
"""Prepare and validate the bounded shared-NTT output-order repair cohort.

Preparation hides CUDA devices and only compiles modules. Run requires an idle
GPU, records exact commands/binary hashes, and never promotes a registry entry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess

from compile_module import compile_mapping
from research_compile_requests import compile_request

ROOT = Path(__file__).resolve().parents[1]


def cases():
    result = []
    for partition in ([8, 8], [6, 5, 5], [10, 6]):
        for overlap in (False, True):
            for order in ("natural", "bit-reversed"):
                result.append(dict(logN=16, batch=16, precision="word64", partition=partition,
                                   overlap=overlap, order=order, inverse=False))
    for log_n, batch, precision, partition, inverse in (
            (8, 5, "word32", [8], False), (12, 5, "word32", [6, 6], True),
            (20, 1, "word64", [10, 10], False)):
        result.append(dict(logN=log_n, batch=batch, precision=precision, partition=partition,
                           overlap=False, order="bit-reversed", inverse=inverse))
    return result


def point(case):
    return dict(operator="ntt", precision=case["precision"], logN=case["logN"],
        batch=case["batch"], output_order=case["order"],
        mapping_json=dict(schema_version=1, kind="ntt", backend="shared-iterative",
            stage_partition=case["partition"], threads_per_block=128,
            dataflow_layout="hermes-xor", stage_overlap=case["overlap"], batch_tile_count=4))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--sm", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/modules")
    args = parser.parse_args()
    if args.jobs < 1 or min(args.warmup, args.repeat, args.trials) < 1:
        parser.error("jobs, warmup, repeat and trials must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    args.build_dir = args.build_dir.resolve()
    args.cache = args.cache.resolve()
    if args.phase == "prepare":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        requests = [compile_request(point(case)) for case in cases()]
        for partition in ([6], [3, 3], [2, 2, 2], [5, 7]):
            for word in (32, 64):
                for aligned in (False, True):
                    for order in ("natural", "bit-reversed"):
                        requests.append(dict(backend="shared-iterative", operator="ntt",
                            precision=f"word{word}", logN=sum(partition), stage_partition=partition,
                            threads=128, writer_aligned=aligned, output_order=order))
        # Default portable selection returns the first feasible local=1 point.
        requests.append(dict(backend="shared-iterative", operator="ntt", precision="word64",
            logN=16, stage_partition=[1]*16, threads=32, writer_aligned=True, output_order="bit-reversed"))
        requests = list({json.dumps(r, sort_keys=True): r for r in requests}.values())
        def compile_one(request):
            module = compile_mapping(request, root=ROOT, mathdx=ROOT/"external/mathdx/nvidia/mathdx",
                nvcc=Path(os.environ.get("CUBUTTERFLY_NVCC", "/usr/local/cuda/bin/nvcc")),
                sm=args.sm, cache=args.cache)
            return dict(request=request, module=str(module))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            compiled = list(pool.map(compile_one, requests))
        write(args.output / "prepared.json", compiled)
        print(f"Prepared {len(compiled)} modules with CUDA hidden")
        return

    if not args.gpu_uuid or not args.gpu_uuid.startswith("GPU-"):
        parser.error("run requires --gpu-uuid GPU-...")
    identity = subprocess.check_output(["nvidia-smi", "-i", args.gpu_uuid,
        "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"], text=True).strip()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
        CUBUTTERFLY_TEMPLATE_ROOT=str(ROOT), CUBUTTERFLY_JIT_CACHE=str(args.cache),
        CUBUTTERFLY_REGISTRY=str(args.output.resolve() / "registry.json"))
    binaries = ("cuntt_bench", "cubutterfly_ntt_stage_probe_tests")
    summary = dict(hardware=identity, source_root=str(ROOT), full_migration_qualified=False,
        binary_sha256={n: hashlib.sha256((args.build_dir/n).read_bytes()).hexdigest() for n in binaries},
        protocol=dict(warmup=args.warmup, repeat=args.repeat, trials=args.trials, verify_batches="all",
                      compilation="outside kernel_ms"), correctness=[], measurements=[])

    def run(command, name, mode):
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                          "--format=csv,noheader"], text=True)
        if args.gpu_uuid in active:
            raise RuntimeError("target GPU has another compute process; no measurement launched")
        completed = subprocess.run([str(c) for c in command], env=dict(env, CUBUTTERFLY_COMPILE_MODE=mode),
                                   text=True, capture_output=True)
        (args.output / f"{name}.stdout").write_text(completed.stdout)
        (args.output / f"{name}.stderr").write_text(completed.stderr)
        write(args.output / f"{name}.command.json", dict(command=[str(c) for c in command], mode=mode))
        completed.check_returncode()
        return completed.stdout

    for mode in ("precompiled", "research"):
        run([args.build_dir/"cubutterfly_ntt_stage_probe_tests"], f"correctness-{mode}", mode)
        summary["correctness"].append(dict(mode=mode, status="passed"))
        for index, case in enumerate(cases()):
            command = [args.build_dir/"cuntt_bench", "--logN", str(case["logN"]), "--batch", str(case["batch"]),
                "--word-bits", case["precision"][4:], "--modulus",
                "998244353" if case["precision"] == "word32" else "576460756061519873",
                "--output-order", case["order"], "--mapping-json", json.dumps(point(case)["mapping_json"]),
                "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--verify", "--verify-batches", "0", "--csv"]
            if case["inverse"]:
                command.append("--inverse")
            rows = []
            for trial in range(args.trials):
                stdout = run(command, f"{mode}-case{index:02}-trial{trial}", mode)
                row = next(csv.DictReader(io.StringIO(stdout)))
                if row.get("correct") != "1" or int(row["verified_batches"]) != case["batch"]:
                    raise RuntimeError(f"case {index} failed full-batch verification")
                rows.append(row)
            result = dict(case=case, mode=mode, correct=True,
                          median_kernel_us=statistics.median(float(r["kernel_ms"])*1000 for r in rows), samples=rows)
            summary["measurements"].append(result)
            write(args.output / "summary.json", summary)
            print(mode, index, round(result["median_kernel_us"], 3), "us", flush=True)
    auto = run([args.build_dir/"cuntt_bench", "--logN", "16", "--batch", "16", "--word-bits", "64",
        "--modulus", "576460756061519873", "--output-order", "bit-reversed", "--auto-select",
        "--warmup", "3", "--repeat", "3", "--verify", "--verify-batches", "0", "--csv"],
        "automatic-selection", "research")
    row = next(csv.DictReader(io.StringIO(auto)))
    if row.get("correct") != "1" or int(row["verified_batches"]) != 16 or row["output_order"] != "bit-reversed":
        raise RuntimeError("automatic selection failed output-order semantics")
    summary["automatic_selection"] = row
    summary["status"] = "complete"
    write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
