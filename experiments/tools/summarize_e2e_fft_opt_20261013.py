#!/usr/bin/env python3
"""Matched FFT optimization cohort; complete releases only, no numerical gate."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
NEW_SHA = "89a038905a45e2d2ab072c142f9bde17ed97b42ed86106bdac656e5ec0f0f33b"
OLD_SHA = "ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b"
GPU = "GPU-00000000-0000-0000-0000-000000000001"
METRICS = ("wall_median_ms", "cuda_event_median_ms")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(data):
    result = dict(data.get("identity", data))
    uuid = result.get("gpu_uuid", "")
    result["gpu_uuid"] = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    return result


def validate_identity(data, expected_sha):
    ident = identity(data)
    if ident.get("library_sha256") != expected_sha or ident.get("gpu_uuid") != GPU:
        raise ValueError("mixed/unexpected library or GPU identity")


def selected_plans(data):
    plans = data.get("plan_metadata", data.get("setup", {}).get("cubutterfly", {}).get("plans", []))
    return [{"direction": p.get("direction"), "algorithm": p.get("algorithm"),
             "selection_source": p.get("selection_source"), "mapping": p.get("mapping"),
             "selection_scope": p.get("selection_scope")}
            for p in plans]


def normalizations(data):
    plans = data.get("plan_metadata", data.get("setup", {}).get("cubutterfly", {}).get("plans", []))
    return [{"direction": p.get("direction"), "norm": p.get("norm"),
             "scale": p.get("mapping", {}).get("contract", {}).get("fft_output_scale"),
             "normalize_inverse": p.get("mapping", {}).get("contract", {}).get("normalize_inverse")}
            for p in plans]


def protocol(data, filter_case):
    ident = identity(data)
    keys = ("runner_sha256", "graph_timing_helpers_sha256", "public_real_fft_sha256",
            "adapter_sha256", "torch", "cupy", "cuda", "torch_cuda", "compile_mode")
    record = {key: ident.get(key) for key in keys}
    if not record["runner_sha256"]:
        raise ValueError("missing runner identity")
    record.update(norm=data.get("norm"), size=data.get("size"), batch=data.get("batch"),
        seed=data.get("seed"), input_sha256=data.get("input_sha256"),
        timing_scope=data.get("timing_scope", data.get("graph_call_contract")),
        input_roles=data.get("input_roles"), timed_input_roles=data.get("timed_input_roles"),
        multiplier_sha256=data.get("multiplier_sha256"), reference_sha256=data.get("reference_sha256"),
        epsilon=data.get("epsilon"), zero_mode=data.get("zero_mode"),
        normalization=normalizations(data), error_acceptance=data.get("error_acceptance"))
    if filter_case:
        record.update(contract=data.get("contract"), padding=data.get("actual_cubutterfly_padding"),
            filter_cached=data.get("filter_spectrum_cached"),
            compensation=data.get("cached_filter_normalization_compensation"),
            reference=data.get("validation_reference"), execution_mode=data.get("execution_mode"),
            tolerance=data.get("tolerance"),
            graph_protocol=[r.get("timing", {}).get("protocol") for r in data.get("rows", [])])
        if not data.get("contract", {}).get("sha256"):
            raise ValueError("missing filter input identity")
    elif not record["input_sha256"] or record["norm"] is None:
        raise ValueError("missing input or normalization identity")
    return record


def timing_records(data, filter_case):
    if not filter_case:
        result = {key[:-6]: value for key, value in data.get("timings", {}).items()
                  if key.endswith("_graph")}
        for backend in ("cubutterfly", "torch", "cupy"):
            if backend not in result:
                raise ValueError("missing complete graph timing: " + backend)
            for metric, raw in (("wall_median_ms", "wall_ms"),
                                ("cuda_event_median_ms", "cuda_event_ms")):
                values = result[backend].get(raw)
                if not values or statistics.median(values) != result[backend].get(metric):
                    raise ValueError("missing/inconsistent raw graph timing: " + backend + "/" + raw)
        return result
    if data.get("execution_mode") != "graph" or not data.get("complete_filter_call"):
        raise ValueError("filter is not a complete graph call")
    result = {}
    for row in data.get("rows", []):
        timing = row["timing"]
        samples = timing.get("samples_us", [])
        if not samples or statistics.median(samples) != timing["median_us"]:
            raise ValueError("filter missing/inconsistent raw event timing")
        result[row["method"]] = {"cuda_event_median_ms": timing["median_us"] / 1000,
            "cuda_event_ms": [value / 1000 for value in samples], "source_timing": timing}
    if "cubutterfly" not in result:
        raise ValueError("filter missing cubutterfly timing")
    return result


def ratios(numerator, denominator):
    return {metric: numerator[metric] / denominator[metric] for metric in METRICS
            if metric in numerator and metric in denominator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/e2e_fft_opt_20261013")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/e2e_graph_20261013")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    output = args.output or args.results_root / "summary.json"
    markdown = args.markdown or output.with_suffix(".md")
    missing, invalid, processes = [], [], []

    def load(directory, filename, expected_sha, case, trial, optional=False):
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else None
        if filename == "*.json":
            candidates = [p for p in (directory / "release").glob("*.json") if ".trace." not in p.name]
            path = candidates[0] if len(candidates) == 1 else None
        else:
            path = directory / "release" / filename
        if path is None or not path.is_file():
            missing.append({"case": case, "trial": trial, "optional": optional,
                "expected_path": str(directory / "release" / filename), "guard_status": status})
            return None
        try:
            if not status or status.get("state") != "complete":
                raise ValueError("release lacks complete isolated guard status")
            data = json.loads(path.read_text())
            if data.get("error") or data.get("cpu_only") or data.get("status") not in (
                    None, "complete", "completed", "pass", "passed", "ok", "success"):
                raise ValueError("report is failed or not a GPU measurement")
            validate_identity(data, expected_sha)
            filter_case = case.startswith("filter")
            timings = timing_records(data, filter_case)
            return {"case": case, "trial": trial, "path": str(path), "report_sha256": sha(path),
                "library_sha256": expected_sha, "gpu_uuid": GPU,
                "protocol": protocol(data, filter_case), "timings": timings,
                "all_reported_timings": data.get("timings"),
                "quality": data.get("quality", [{"method": r["method"], "error": r.get("error"),
                    "changing_input": r.get("graph_changing_input_validation"),
                    "formal_performance_eligible": r.get("formal_performance_eligible")}
                    for r in data.get("rows", [])]),
                "error_acceptance": data.get("error_acceptance", data.get("tolerance")),
                "selected_plans": selected_plans(data), "gpu_traces": data.get("gpu_traces"),
                "environment": data.get("environment"), "source_status": data.get("status")}
        except (ValueError, KeyError, TypeError) as error:
            invalid.append({"case": case, "trial": trial, "path": str(path),
                            "reason": str(error), "guard_status": status})
            return None

    def pair(case, trial, filename, group="primary", new_directory=None, old_directory=None,
             expected_sha=NEW_SHA, optional=False, old_filename=None):
        new = load(new_directory or args.results_root / (case + "_t" + str(trial)),
                   filename, expected_sha, case, trial, optional)
        old = load(old_directory or args.baseline_root / (case + "_t" + str(trial)),
                   old_filename or filename, OLD_SHA, case, trial, optional)
        if new is None:
            return
        new["group"] = group
        new["baseline"] = old
        if old:
            differences = [key for key in new["protocol"] if new["protocol"][key] != old["protocol"].get(key)]
            if differences:
                invalid.append({"case": case, "trial": trial, "path": new["path"],
                    "baseline_path": old["path"], "reason": "unmatched runner/input/norm/protocol",
                    "differing_fields": differences})
                return
            new["baseline_over_new_ratios"] = {backend: ratios(old["timings"][backend], timing)
                for backend, timing in new["timings"].items() if backend in old["timings"]}
        new["current_process_backend_over_cubutterfly"] = {
            backend: ratios(timing, new["timings"]["cubutterfly"])
            for backend, timing in new["timings"].items() if backend != "cubutterfly"}
        processes.append(new)

    for trial in (0, 1):
        for size in (1024, 2048):
            for batch in (4, 16):
                pair("poisson_%d_b%d" % (size, batch), trial, "periodic_poisson.json")
        pair("registration_2048_b16", trial, "phase_correlation.json", group="targeted_registration", old_filename="registration.json")
        pair("filter_b16", trial, "*.json", group="optional_filter_event_only", optional=True,
             old_directory=args.baseline_root / ("filter_b16_r2_t" + str(trial)))
    control = args.results_root / "poisson_2048_b16_old_t0"
    pair("poisson_2048_b16", 0, "periodic_poisson.json", group="optional_old_binary_control",
         new_directory=control, expected_sha=OLD_SHA, optional=True)
    report = {"schema_version": 1, "summarizer_sha256": sha(Path(__file__)),
        "expected_library_sha256": NEW_SHA, "baseline_library_sha256": OLD_SHA,
        "gpu_uuid": GPU, "status": "rejected_mixed_identity_or_invalid" if invalid else
            ("partial_missing_reports" if any(not m["optional"] for m in missing) else "complete"),
        "ratio_definition": "baseline latency / new latency and same-process external backend / cuButterfly; >1 favors new/cuButterfly",
        "aggregation": "matched per-trial ratios only; no heterogeneous geometric mean; wall and events separate; filter event-only protocol separate",
        "accuracy": "preserve source distributions and structural checks; no new tolerance or numerical acceptance",
        "processes": processes, "missing": missing, "invalid": invalid}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# FFT Optimization Cohort", "", "Status: `%s`. Wall and CUDA-event ratios remain separate; no geometric mean." % report["status"], "",
             "| 场景 | 类型 | 对照 | 加速比 |", "| --- | --- | --- | --- |"]
    def formatted(values):
        return "; ".join(("wall" if key.startswith("wall") else "event") + " %.4fx" % value
                         for key, value in values.items()) or "unavailable"
    for process in processes:
        label = "%s t%d (%s)" % (process["case"], process["trial"], process["group"])
        for backend, ratio in process["current_process_backend_over_cubutterfly"].items():
            lines.append("| %s | FFT FP32 Graph | %s current process | %s |" % (label, backend, formatted(ratio)))
        ratio = process.get("baseline_over_new_ratios", {}).get("cubutterfly")
        if ratio:
            lines.append("| %s | FFT FP32 Graph | old cuButterfly matched trial | %s |" % (label, formatted(ratio)))
    lines.extend(["", "## Raw Per-Trial Latencies", "",
        "Milliseconds; full raw sample arrays and error distributions are retained in the adjacent JSON.", "",
        "| Trial | Backend | Wall Median ms | CUDA Event Median ms |", "| --- | --- | --- | --- |"])
    for process in processes:
        for cohort, source in (("new", process), ("baseline", process.get("baseline"))):
            if source:
                for backend, timing in source["timings"].items():
                    lines.append("| %s t%d %s | %s | %s | %s |" % (process["case"], process["trial"], cohort,
                        backend, timing.get("wall_median_ms", "not collected"), timing.get("cuda_event_median_ms", "not collected")))
    lines.extend(["", "## Missing And Invalid", "", "Missing reports: %d (including optional); rejected reports: %d." % (len(missing), len(invalid))])
    lines.extend("\n- `%s` t%s%s: %s" % (m["case"], m["trial"], " optional" if m["optional"] else "", m["expected_path"]) for m in missing)
    lines.extend("\n- Rejected `%s`: %s" % (m["path"], m["reason"]) for m in invalid)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "markdown": str(markdown), "processes": len(processes),
                      "missing": len(missing), "invalid": len(invalid), "status": report["status"]}))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
