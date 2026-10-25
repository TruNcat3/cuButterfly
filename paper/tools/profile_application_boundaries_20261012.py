#!/usr/bin/env python3
"""Bounded application diagnostics; instrumentation is outside headline timing.

FNO invokes the existing end-to-end runner unchanged and profiles one EXTRA call
per backend. QuaRot replays the four measured B1 FWHT mappings as diagnostics,
not as a newly tuned GPU2 library or a complete model benchmark.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/tools"))
from run_application_batch_20261012 import LIB, TARGETS, digest, effective_environment


COMPONENT_SCHEMA = "cubutterfly-application-breakdown-v1"
COMPONENT_CATEGORIES = ("compile_plan", "adapter_conversion", "operator_gpu", "application_rest")


def component_breakdown(*, application, backend, headline_scope, profiler=False,
                        isolated=False, notes=()):
    """Return a comparable component contract without inventing timings.

    Missing measurements are explicit.  ``headline_scope`` describes the
    timing that can be compared across backends; profiler and isolated stage
    observations are never implicitly summed into it.
    """
    if headline_scope not in ("eager_interval", "cuda_graph_replay", "unknown"):
        raise ValueError("unknown headline timing scope")
    scopes = {
        "compile_plan": "excluded_from_headline; no independent sample in this profiler",
        "adapter_conversion": "included_in_eager_interval; not separable from wrapper unless isolated",
        "operator_gpu": "included_in_eager_interval; profiler intervals may be nested",
        "application_rest": "included_in_eager_interval; profiler intervals may be nested",
    }
    return {
        "schema": COMPONENT_SCHEMA,
        "application": application,
        "backend": backend,
        "headline": {"scope": headline_scope, "unit": "us",
                      "includes": ["adapter_conversion", "operator_gpu", "application_rest"],
                      "excludes": ["compile_plan"]},
        "components": [
            {"name": category, "category": category, "status": "unmeasured",
             "timing_scope": scopes[category], "median_us": None,
             "additive": False}
            for category in COMPONENT_CATEGORIES
        ],
        "profiler": {"available": bool(profiler),
                      "nested_totals_are_not_additive": bool(profiler)},
        "isolated_samples": bool(isolated),
        "notes": list(notes),
    }


def classify_profiler_event(name):
    """Classify an event conservatively; unknown events stay application rest."""
    text = str(name).lower()
    if "compile" in text or "plan" in text:
        return "compile_plan"
    if "cubutterfly/" in text or "torch/fft" in text or "fft" in text and "application/" not in text:
        return "operator_gpu"
    if text.startswith("application/"):
        return "application_rest"
    return "application_rest"


def normalize_profiler_report(*, application, backend, profile):
    """Attach categories to an existing profiler report, preserving old keys."""
    result = component_breakdown(application=application, backend=backend,
                                  headline_scope="eager_interval", profiler=True,
                                  notes=("Profiler event totals are nested and non-additive.",
                                         "No compile/plan sample was collected."))
    result["events"] = [dict(event, component=classify_profiler_event(event.get("name", "")),
                              timing_relation="nested_or_independent")
                        for event in profile.get("events", [])]
    return result


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def errors(torch, actual, reference):
    delta = (actual.double() - reference.double()).abs().flatten()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return dict(finite=False, accepted=None)
    q = torch.tensor([0.0, .5, .9, .99, 1.0], device=delta.device, dtype=delta.dtype)
    return dict(finite=True, accepted=None, exact=bool((delta == 0).all()),
                max_abs=float(delta.max()), relative_l2=float(torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(reference.double()).clamp_min(1e-300)),
                abs_quantiles=dict(zip(("min", "p50", "p90", "p99", "max"),
                                       torch.quantile(delta, q).cpu().tolist())))


def timing(torch, function, warmup=10, repeats=20, samples=7):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    result = []
    for _ in range(samples):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        result.append(begin.elapsed_time(end) * 1000 / repeats)
    return dict(median_us=statistics.median(result), samples_us=result,
                repeats=repeats, warmup=warmup, mode="eager CUDA interval; includes dispatch gaps")


def profile_call(torch, function, output, name):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        with torch.profiler.record_function("application/" + name):
            function()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / (name + ".trace.json")))
    rows = [dict(name=event.key, count=event.count, cpu_total_us=event.cpu_time_total,
                 self_cpu_us=event.self_cpu_time_total,
                 device_total_us=getattr(event, "device_time_total", 0),
                 self_device_us=getattr(event, "self_device_time_total", 0))
            for event in prof.key_averages()]
    report = dict(
        profiler_timing_is_headline=False, nested_totals_are_not_additive=True, events=rows)
    backend = "cubutterfly" if "cubutterfly" in name else "torch"
    report["component_breakdown"] = normalize_profiler_report(
        application=name.split("_", 1)[0], backend=backend, profile=report)
    save(output / (name + ".profile.json"), report)


def graph_timing(torch, function, calls_per_graph=20, samples=7):
    """Separate device-throughput control; not application eager timing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = function()  # prepare stream-specific plans before capture
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls_per_graph):
            result = function()
    torch.cuda.current_stream().wait_stream(stream)
    values = timing(torch, graph.replay, warmup=3, repeats=10, samples=samples)
    values["median_us"] /= calls_per_graph
    values["samples_us"] = [x / calls_per_graph for x in values["samples_us"]]
    values.update(mode="CUDA graph replay divided by captured calls; separate throughput control",
                  calls_per_graph=calls_per_graph)
    return values


def fno(output, library):
    import torch
    runner_path = ROOT / "results/e13_assets/run_fno_end_to_end.py"
    runner = load_module("fno_boundary_runner", runner_path)
    original = runner.synchronized_forward
    seen = set()
    active = [False]
    intervals = []

    def instrument(function, label):
        def wrapped(*args, **kwargs):
            if not active[0]:
                return function(*args, **kwargs)
            tensor = args[0]
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            with torch.profiler.record_function(label):
                begin.record()
                result = function(*args, **kwargs)
                end.record()
            intervals.append((label, list(tensor.shape), tensor.is_contiguous(), begin, end))
            return result
        return wrapped

    torch.fft.rfftn = instrument(torch.fft.rfftn, "torch/rfftn")
    torch.fft.irfftn = instrument(torch.fft.irfftn, "torch/irfftn")
    for name in ("rfftn", "irfftn"):
        original_method = getattr(runner.AdapterPatch, name)
        def method(self, *args, _fn=original_method, _name=name, **kwargs):
            if not active[0]:
                return _fn(self, *args, **kwargs)
            return instrument(lambda *a, **kw: _fn(self, *a, **kw), "cubutterfly/" + _name)(*args, **kwargs)
        setattr(runner.AdapterPatch, name, method)

    def measured(model, x, device):
        result = original(model, x, device)
        label = "cubutterfly" if getattr(torch.fft.rfftn, "__self__", None) is not None else "torch"
        if label not in seen:
            seen.add(label)
            intervals.clear()
            active[0] = True
            try:
                profile_call(torch, lambda: model(x), output, "fno_" + label)
            finally:
                active[0] = False
            save(output / ("fno_" + label + ".intervals.json"), [dict(
                scope=n, input_shape=s, contiguous=c, instrumented_cuda_interval_us=b.elapsed_time(e)*1000)
                for n, s, c, b, e in intervals])
        return result

    runner.synchronized_forward = measured
    sys.argv = [str(runner_path), "--data-root", str(ROOT / "results/e13_assets/data/extracted64"),
                "--resolution", "64", "--n-test", "4", "--batch-size", "4", "--trials", "1",
                "--calls", "5", "--warmup", "5", "--library", str(library),
                "--output", str(output / "fno_end_to_end.json")]
    code = runner.main()
    if code or seen != {"torch", "cubutterfly"}:
        raise RuntimeError(f"FNO diagnostic incomplete: exit={code}, profiles={seen}")
    save(output / "fno_component_breakdown.json", {
        "schema": COMPONENT_SCHEMA,
        "application": "fno",
        "headline": "fno_end_to_end.json timed calls",
        "backends": [component_breakdown(
            application="fno", backend=backend, headline_scope="eager_interval",
            profiler=True, isolated=True,
            notes=("FFT interval annotations include wrapper and dispatch gaps",
                   "profiler event totals are nested and non-additive",
                   "compile and plan setup are excluded from timed calls"))
                    for backend in ("torch", "cubutterfly")],
        "gap": "No independent compile/plan, adapter-only, or application-only GPU samples were collected.",
    })


def quarot(output, library):
    import torch
    from fast_hadamard_transform import hadamard_transform
    source = ROOT / "results/paper_targeted_20261012/application_batch/quarot_b1_p1/release/result.json"
    prior = json.loads(source.read_text())
    module = load_module("quarot_boundary_adapter", ROOT / "paper/experiments/e13/quarot_adapter.py")
    torch.manual_seed(20261012)
    results = []
    for metadata in prior["adapter_setup"]["plans"]:
        mapping = metadata["mapping"]
        n = mapping["contract"]["shape"][0]
        batch = mapping["contract"]["batch"]
        full_n = {32: 4096, 64: 11008}[n]
        scale = float(1.0 / torch.tensor(full_n).sqrt())
        os.environ["CUBUTTERFLY_E13_ALGORITHM_POLICY"] = json.dumps(mapping)
        with module.QuaRotFWHTAdapter(library, device="cuda:0", compute_contract="fp32_composed") as adapter:
            x = torch.randn(batch, n, device="cuda", dtype=torch.float16) * .1
            x32 = x.float()
            plan = adapter._plan_for(x32, n, batch)
            raw = plan.execute(x32)
            scaled = raw * scale
            scale_buffer = torch.empty_like(raw)
            funcs = dict(dao_fp16_scaled=lambda: hadamard_transform(x, scale),
                         cubutterfly_fp16_composed=lambda: adapter._transform_vectors(x, n, full_n),
                         cubutterfly_fp32_core=lambda: plan.execute(x32),
                         dao_fp32_core=lambda: hadamard_transform(x32, 1.0),
                         cast_input=lambda: x.float(),
                         scale_only_out_buffer=lambda: torch.mul(raw, scale, out=scale_buffer),
                         cast_output=lambda: scaled.half())
            reference = funcs["dao_fp16_scaled"]()
            actual = funcs["cubutterfly_fp16_composed"]()
            error = errors(torch, actual, reference)
            core_error = errors(torch, plan.execute(x32), hadamard_transform(x32, 1.0))
            # Preserve every per-element error separately; timing never includes this export.
            import numpy as np
            np.savez_compressed(output / f"quarot_n{n}_b{batch}.errors.npz",
                difference=(actual.double()-reference.double()).cpu().numpy())
            # Interleave the two full boundaries; isolated stages are diagnostic only.
            times = {key: timing(torch, fn) for key, fn in funcs.items()}
            graphs = {key: graph_timing(torch, funcs[key]) for key in (
                "dao_fp16_scaled", "cubutterfly_fp16_composed", "cubutterfly_fp32_core", "dao_fp32_core")}
            for key in ("dao_fp16_scaled", "cubutterfly_fp16_composed"):
                profile_call(torch, funcs[key], output, f"quarot_n{n}_b{batch}_{key}")
            results.append(dict(n=n, batch=batch, full_n=full_n, scale=scale, error=error,
                                core_error=core_error, plan=plan.metadata, timings=times, graph_timings=graphs,
                                component_breakdown=component_breakdown(
                                    application="quarot", backend="cubutterfly",
                                    headline_scope="eager_interval", isolated=True,
                                    notes=("composed timing includes adapter casts, FWHT, and scale",
                                           "isolated component timings are diagnostic and non-additive",
                                           "plan creation and compilation were excluded")),
                                ratio=times["dao_fp16_scaled"]["median_us"] /
                                      times["cubutterfly_fp16_composed"]["median_us"]))
            save(output / "quarot_components.json", dict(
                source=str(source), source_sha256=digest(source), cases=results,
                scope="fixed measured B1 mappings replayed on GPU2; FWHT subpath only; no common hadK matmul",
                semantic_validation="errors reported; this does not replace full-model zero-tolerance checks",
                isolated_times_are_additive=False, library_performance_claim=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("fno", "quarot"), required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--until", default="2026-10-12T18:00:00+08:00")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    uuid, calibration_path = TARGETS["fft_filter"]
    calibration = json.loads(calibration_path.read_text())
    if digest(LIB) != calibration["measurement_identity"]["library_sha256"]:
        raise RuntimeError("library/calibration mismatch")
    env, identity = effective_environment(calibration, uuid, 512*1024**2)
    if not args.child:
        command = [sys.executable, str(ROOT / "scripts/run_isolated_gpu.py"),
                   "--gpu-uuid", uuid, "--report-dir", str(args.report_dir.resolve()),
                   "--until", args.until, "--", sys.executable, str(Path(__file__).resolve()),
                   "--lane", args.lane, "--report-dir", str(args.report_dir.resolve()), "--child"]
        print(json.dumps(dict(command=command, environment=identity), indent=2), flush=True)
        return subprocess.call(command, env=env, cwd=ROOT) if args.run else 0
    output = Path(os.environ["XD_REPORT_DIR"])
    save(output / "identity.json", dict(library_sha256=digest(LIB), runner_sha256=digest(__file__),
        environment=identity, calibration_sha256=digest(calibration_path), gpu_uuid=uuid,
        setup_excluded=True, headline_benchmark=False))
    os.environ["CUBUTTERFLY_PLAN_CACHE"] = str(output / "diagnostic.plans")
    os.environ["CUBUTTERFLY_E13_ALGORITHM_POLICY"] = "default"
    if args.lane == "fno":
        fno(output, LIB)
    else:
        quarot(output, LIB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
