#!/usr/bin/env python3
"""Finite real-request candidate benchmark; preparation never uses a GPU."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from cubutterfly.fused_short_fwht import FusedShortFWHT, prepare

CASES = [(32, 128, 4096), (32, 65536, 4096), (64, 172, 11008), (64, 88064, 11008)]
OUTPUT = ROOT / "results/paper_targeted_20261012/fused_short_fwht_candidate"


def difference(actual, reference):
    import numpy as np
    error = actual.double().cpu().numpy() - reference.double().cpu().numpy()
    absolute = np.abs(error)
    return dict(zero_diff=bool(np.all(error == 0)), mismatched=int(np.count_nonzero(error)),
                max_abs=float(absolute.max()), mean_abs=float(absolute.mean()),
                quantiles={str(q): float(np.quantile(absolute, q)) for q in (0, .5, .95, .99, 1)},
                nonfinite=int(np.count_nonzero(~np.isfinite(error)))), error


def timing(torch, fn, graph=False):
    # Compile and allocate before capture or measurement, on the same stream.
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(100):
                fn()
        stream.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            retained = fn()
        fn = capture.replay
    else:
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
    rows = []
    for _ in range(3):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(100):
            fn()
        end.record()
        end.synchronize()
        rows.append(begin.elapsed_time(end) * 1000 / 100)
    return dict(trial_us=rows, median_us=statistics.median(rows), calls_per_trial=100,
                process_trials=False, mode="cuda-graph" if graph else "eager", setup_excluded=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--guard-child", action="store_true")
    parser.add_argument("--library", type=Path, default=ROOT / "results/search_budget_20261011/frozen_runtime/libcuntt.so")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    preparation_dir = args.output_dir.resolve()
    output = Path(os.environ["XD_REPORT_DIR"]).resolve() if args.guard_child else preparation_dir
    output.mkdir(parents=True, exist_ok=True)
    module = preparation_dir / "fused_short_fwht.so"
    if args.prepare:
        spec = prepare(module)
        frozen = json.loads((ROOT / "results/paper_targeted_20261012/medium_large_mechanism/fwht/launch.json").read_text())
        command = [frozen["python"], str(ROOT / "scripts/run_isolated_gpu.py"), "--gpu-uuid", frozen["gpu_uuid"],
                   "--report-dir", str(output / "guard"), "--until", "2026-10-12T16:30:00+08:00", "--", "env"]
        command += [f"{key}={value}" for key, value in sorted(frozen["environment"].items())]
        command += [frozen["python"], str(Path(__file__).resolve()), "--guard-child", "--output-dir", str(output), "--library", str(args.library.resolve())]
        spec.update(cases=[dict(n=n, batch=batch, application_full_n=full_n) for n, batch, full_n in CASES],
                    launch_argv=command, benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    status="prepared-not-gpu-validated", required_zero_diff=True,
                    integration_gap="Add standalone provider as public local-core processing unit with descriptor scale/FP32 accumulation and search identity; not modifying frozen library or QuaRot adapter.")
        (output / "prepared.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps(spec, indent=2))
        return 0
    prepared = json.loads((preparation_dir / "prepared.json").read_text())
    if hashlib.sha256(module.read_bytes()).hexdigest() != prepared["module_sha256"]:
        raise ValueError("candidate module changed since preparation")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != prepared["benchmark_sha256"] or hashlib.sha256(args.library.read_bytes()).hexdigest() != prepared["library_sha256"]:
        raise ValueError("benchmark or frozen library changed since preparation")
    import numpy as np
    import torch
    from fast_hadamard_transform import hadamard_transform
    path = ROOT / "experiments/workloads/e13/quarot_adapter.py"
    location = importlib.util.spec_from_file_location("candidate_unchanged_quarot_adapter", path)
    adapter_module = importlib.util.module_from_spec(location)
    sys.modules[location.name] = adapter_module
    location.loader.exec_module(adapter_module)
    candidate = FusedShortFWHT(module)
    generator = torch.Generator(device="cuda").manual_seed(20261012)
    result = dict(schema="cubutterfly-short-fwht-candidate-v1", provider=prepared, cases=[],
                  scope="Standalone FWHT subpath only; excludes common hadK/application; not public search integrated",
                  acceptance="Exact zero diff against Dao and existing fp32_composed; no tolerance relaxation")
    failed = False
    with adapter_module.QuaRotFWHTAdapter(args.library, device="cuda:0", compute_contract="fp32_composed") as adapter:
        for n, batch, full_n in CASES:
            row = dict(n=n, batch=batch, full_n=full_n)
            try:
                scale = float(1.0 / torch.tensor(full_n).sqrt())
                x = torch.randn(batch, n, device="cuda", dtype=torch.float16, generator=generator) * .1
                funcs = dict(dao=lambda: hadamard_transform(x, scale),
                             fp32_composed=lambda: adapter._transform_vectors(x, n, full_n),
                             candidate=lambda: candidate(x, scale))
                outputs = {name: fn() for name, fn in funcs.items()}
                torch.cuda.synchronize()
                checks = {}
                payload = dict(input=x.cpu().numpy())
                for name in ("fp32_composed", "candidate"):
                    checks[name + "_vs_dao"], error = difference(outputs[name], outputs["dao"])
                    payload[name + "_vs_dao"] = error
                checks["candidate_vs_fp32_composed"], payload["candidate_vs_fp32_composed"] = difference(outputs["candidate"], outputs["fp32_composed"])
                np.savez_compressed(output / f"n{n}_b{batch}.errors.npz", **payload)
                row.update(scale=scale, input_seed=20261012, error_checks=checks,
                           exact_accepted=all(check["zero_diff"] for check in checks.values()))
                failed |= not row["exact_accepted"]
                row["eager"] = {name: timing(torch, fn) for name, fn in funcs.items()}
                row["cuda_graph"] = {name: timing(torch, fn, graph=True) for name, fn in funcs.items()}
            except Exception as error:
                row.update(status="failed", error=str(error))
                failed = True
            result["cases"].append(row)
            result["status"] = "failed" if failed else "accepted-so-far"
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result["adapter_setup"] = adapter.plan_statistics()
    result["status"] = "rejected" if failed else "accepted"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
