#!/usr/bin/env python3
"""Replay stage-overlap preflight mappings through the public API and CUDA Graph.

This is a fixed-mapping diagnostic, not an automatic-library score. Compilation,
allocation, capture and same-precision CPU checks precede all timing. Numerical
errors are reported without inventing a new acceptance threshold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from run_comprehensive_suite import visible_gpu


def exclusive():
    rows = subprocess.check_output([
        "nvidia-smi", "-i", visible_gpu(), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    others = [row for row in rows if row.strip() and row.strip() != str(os.getpid())]
    if others:
        raise RuntimeError("target GPU has other compute processes: " + ", ".join(others))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-debug-dir", type=Path,
                        help="export the captured CUDA dependency graphs as DOT files")
    parser.add_argument("--modes", nargs="+", help="limit diagnostics to named preflight modes")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat < 1:
        parser.error("rounds and repeat must be positive")
    exclusive()
    import torch
    import cubutterfly as cb
    torch.set_num_threads(4)
    library = cb.Library(args.library.resolve())
    stream = torch.cuda.Stream()
    records = json.loads(args.verification.read_text())
    if args.modes:
        missing = set(args.modes) - {row["mode"] for row in records}
        if missing:
            parser.error("unknown preflight modes: " + ", ".join(sorted(missing)))
        records = [row for row in records if row["mode"] in args.modes]
    report = {
        "kind": "fixed-mapping-cuda-graph-diagnostic",
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "verification_sha256": hashlib.sha256(args.verification.read_bytes()).hexdigest(),
        "device": subprocess.check_output([
            "nvidia-smi", "-i", visible_gpu(), "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader"], text=True).strip(),
        "compile_mode": os.environ.get("CUBUTTERFLY_COMPILE_MODE", "default"),
        "captured_executions": 20, "rounds": args.rounds, "repeat": args.repeat,
        "modes": args.modes,
        "numerical_policy": "finite outputs; report same-precision CPU errors; no new tolerance",
        "results": [],
    }
    runtime_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    report["cuda_runtime_uuid"] = str(runtime_uuid) if runtime_uuid is not None else None
    if runtime_uuid is not None:
        requested_uuid = report["device"].split(",", 1)[0].strip()
        if str(runtime_uuid).replace("GPU-", "", 1) != requested_uuid.replace("GPU-", "", 1):
            raise RuntimeError("CUDA runtime device differs from the requested measurement GPU")
    plans, tensors, graphs = [], [], []
    try:
        for record in records:
            exclusive()
            sample = record["sample"]
            print("preparing graph", record["mode"], flush=True)
            n, batch = int(sample["N"]), int(sample["batch"])
            if (sample["operator"] != "fft" or sample["precision"] not in ("fp32", "fp64") or
                    sample["placement"] != "out-of-place" or int(sample["element_stride"]) != 1 or
                    int(sample["batch_stride"]) != n):
                raise ValueError("diagnostic requires packed out-of-place FP32/FP64 complex FFT")
            fp64 = sample["precision"] == "fp64"
            dtype = torch.complex128 if fp64 else torch.complex64
            generator = torch.Generator().manual_seed(127)
            cpu_input = torch.randn((batch, n), dtype=dtype, generator=generator)
            inverse = sample["direction"] == "inverse"
            normalized = sample["normalization"] == "inverse"
            reference = (torch.fft.ifft(cpu_input, norm="backward" if normalized else "forward")
                         if inverse else torch.fft.fft(cpu_input))
            transform = cb.Transform(extents=[n], batch=batch, inverse=inverse,
                normalize_inverse=normalized,
                storage=cb.DataType.COMPLEX_FP64 if fp64 else cb.DataType.COMPLEX_FP32)
            mapping = sample["mapping_json"]
            if isinstance(mapping, dict):
                mapping = json.dumps(mapping)
            plan = cb.Plan(library, transform, cb.PlanOptions(mapping=mapping),
                           stream=stream.cuda_stream)
            plans.append(plan)
            with torch.cuda.stream(stream):
                source = cpu_input.cuda()
                output = torch.empty_like(source)
                workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
                if plan.workspace_size():
                    plan.set_workspace(workspace.data_ptr(), plan.workspace_size())
                execute = lambda: plan.execute_steady(source.data_ptr(), output.data_ptr())
                for _ in range(3):
                    execute()
                stream.synchronize()
                direct = output.cpu()
                graph = torch.cuda.CUDAGraph()
                if args.graph_debug_dir:
                    graph.enable_debug_mode()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(report["captured_executions"]):
                        execute()
                if args.graph_debug_dir:
                    args.graph_debug_dir.mkdir(parents=True, exist_ok=True)
                    graph.debug_dump(str(args.graph_debug_dir /
                        f"{record['mode']}-n{n}-b{batch}.dot"))
                graph.replay()
                stream.synchronize()
                actual = output.cpu()
            if not torch.isfinite(actual).all() or not torch.equal(actual, direct):
                raise RuntimeError("capture changed output or produced non-finite values")
            delta = (actual - reference).abs().double()
            report["results"].append({
                "mode": record["mode"], "N": n, "batch": batch, "precision": sample["precision"],
                "mapping": json.loads(plan.mapping_json()), "workspace_bytes": plan.workspace_size(),
                "max_abs_error": delta.max().item(),
                "relative_l2_error": (torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(reference.abs().double())).item(),
                "graph_matches_direct": True, "trials_ms": [],
            })
            tensors.append((source, output, workspace))
            graphs.append(graph)
        order = list(range(len(plans)))
        randomizer = random.Random(127)
        for trial in range(args.rounds):
            randomizer.shuffle(order)
            for index in order:
                exclusive()
                graph = graphs[index]
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        graph.replay()
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    for _ in range(args.repeat):
                        graph.replay()
                    stop.record(stream)
                    stop.synchronize()
                elapsed = start.elapsed_time(stop) / (args.repeat * report["captured_executions"])
                exclusive()
                row = report["results"][index]
                row["trials_ms"].append(elapsed)
                row["median_ms"] = statistics.median(row["trials_ms"])
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("completed graph round", trial + 1, flush=True)
        print(json.dumps([{key: row[key] for key in ("mode", "N", "batch", "median_ms", "relative_l2_error")}
                          for row in report["results"]], indent=2))
    finally:
        stream.synchronize()
        graphs.clear()
        for plan in plans:
            plan.close()


if __name__ == "__main__":
    main()
