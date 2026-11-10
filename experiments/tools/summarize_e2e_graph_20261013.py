#!/usr/bin/env python3
"""Transparent dated aggregation of released, matched complete-call graph reports."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread(values):
    return dict(values=values, median=statistics.median(values), minimum=min(values), maximum=max(values))


def norms(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "norm":
                found.add(str(child))
            else:
                found.update(norms(child))
    elif isinstance(value, list):
        for child in value:
            found.update(norms(child))
    return found


def load_process(path, case, trial, origin=None):
    data = json.loads(path.read_text())
    identity = data.get("identity", data)
    uuid = identity.get("gpu_uuid")
    gpu = "GPU-" + uuid.removeprefix("GPU-") if uuid else None
    identity_keys = ("library_sha256", "runner_sha256", "adapter_sha256", "public_real_fft_sha256",
                     "graph_timing_helpers_sha256", "checks_sha256", "quarot_extension_sha256",
                     "quarot_extension_protocol", "attention_backend", "algorithm_policy", "seed",
                     "state_sha256", "torch", "transformers", "cupy", "cuda", "torch_cuda")
    compatible = {key: identity[key] for key in identity_keys if key in identity}
    compatible["gpu_uuid"] = gpu
    compatible["norm"] = sorted(norms(data))
    compatible["timing_scope"] = data.get("timing_scope", data.get("graph_call_contract"))
    compatible["input_identity"] = data.get("input_sha256", data.get("contract", {}).get("sha256"))
    compatible["compute_contract"] = data.get("compute_contract")
    compatible["payload"] = data.get("payload")
    if case == "filter_1024_b16":
        compatible["normalization_scales"] = [dict(direction=p.get("direction"),
            scale=p.get("mapping", {}).get("contract", {}).get("fft_output_scale"))
            for p in data.get("plan_metadata", [])]
        compatible["padding"] = data.get("actual_cubutterfly_padding")
        compatible["cached_filter"] = data.get("filter_spectrum_cached")
    timings, quality, eligibility = {}, {}, {}
    for backend, timing in data.get("timings", {}).items():
        if backend.endswith("_graph"):
            name = backend.removesuffix("_graph")
            timings[name] = {key: timing[key] for key in ("wall_median_ms", "cuda_event_median_ms", "wall_ms", "cuda_event_ms") if key in timing}
            quality[name] = data.get("quality", {}).get(backend)
    for row in data.get("rows", []):
        if data.get("execution_mode") != "graph":
            continue
        name = row["method"]
        timings[name] = dict(cuda_event_median_ms=row["timing"]["median_us"] / 1000,
                            cuda_event_ms=[x / 1000 for x in row["timing"].get("samples_us", [])],
                            protocol=row["timing"].get("protocol"))
        quality[name] = dict(error=row.get("error"), changing_input=row.get("graph_changing_input_validation"))
        eligibility[name] = row.get("formal_performance_eligible")
    if case == "quarot" and data.get("execution_mode") == "graph":
        case = f"quarot_b{data.get('payload', {}).get('batch_size', 'unknown')}"
        for name in ("baseline", "cubutterfly"):
            timing = data.get(name, {})
            converted = {}
            for raw, median in (("wall_ms", "wall_median_ms"), ("cuda_event_ms", "cuda_event_median_ms")):
                values = timing.get(raw, {}).get("end_to_end", [])
                if values:
                    converted.update({raw: values, median: statistics.median(values)})
            if converted:
                timings[name] = converted
            quality[name] = {key: data[key] for key in ("baseline_repeat", "cubutterfly_repeat", "comparison", "numerical_checks", "graph_changed_token_validation") if key in data}
    state = data.get("status")
    failed = bool(data.get("error")) or (state is not None and state not in ("complete", "completed", "pass", "passed", "ok", "success"))
    return dict(case=case, trial=trial, path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        report_sha256=sha(path), origin=origin, release_report="release" in path.parts,
        source_status=state, completed_release=not failed and bool(timings) and "release" in path.parts,
        failure=data.get("error"), identity=compatible, timings=timings, quality=quality,
        formal_performance_eligible=eligibility,
        provenance={key: data[key] for key in ("cache", "environment", "gpu_traces", "error_acceptance", "validation_reference", "summary", "timing_contract") if key in data})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/e2e_graph_20261013/summary.json")
    parser.add_argument("--quarot-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    directory = ROOT / "results/e2e_graph_20261013"
    processes, missing, invalid = [], [], []

    def ingest(path, case, trial, origin=None):
        if not path.is_file():
            missing.append(dict(case=case, trial=trial, expected_path=str(path)))
            return
        try:
            process = load_process(path, case, trial, origin)
            processes.append(process)
            if not process["completed_release"]:
                invalid.append(dict(path=str(path), source_status=process["source_status"], reason="not a successful released graph measurement"))
        except (ValueError, KeyError, TypeError) as error:
            invalid.append(dict(path=str(path), reason=str(error)))

    for trial in (0, 1):
        candidates = sorted((directory / f"filter_b16_r2_t{trial}/release").glob("*.json"))
        candidates = [path for path in candidates if ".trace." not in path.name]
        if not candidates:
            missing.append(dict(case="filter_1024_b16", trial=trial, expected_path=str(directory / f"filter_b16_r2_t{trial}/release/*.json")))
        for path in candidates:
            ingest(path, "filter_1024_b16", trial)
        for application, filename in (("registration", "registration.json"), ("poisson", "periodic_poisson.json")):
            for size in (1024, 2048):
                for batch in (4, 16):
                    case = f"{application}_{size}_b{batch}"
                    ingest(directory / f"{case}_t{trial}/release/{filename}", case, trial)
    fno_path = ROOT / "results/e2e_unified_20261013/fno_hotpath_summary.json"
    reused = None
    if fno_path.is_file():
        fno = json.loads(fno_path.read_text())
        reused = dict(path=str(fno_path.relative_to(ROOT)), sha256=sha(fno_path), summary=fno)
        for result in fno["results"]:
            for trial, relative in enumerate(result["reports"]):
                ingest(fno_path.parent / relative, f"fno_{result['size']}_b{fno['batch']}", trial, reused["path"])
    else:
        missing.append(dict(case="fno", expected_path=str(fno_path)))
    quarot_reports = args.quarot_report or [ROOT / "results/e2e_unified_20261013" /
        f"quarot_graph_b4_current_stream_t{trial}/release/quarot_graph.json" for trial in (0, 1)]
    for trial, path in enumerate(quarot_reports):
        ingest(path.resolve(), "quarot", trial)
    groups = defaultdict(list)
    for process in processes:
        if process["completed_release"]:
            groups[(process["case"], json.dumps(process["identity"], sort_keys=True))].append(process)
    cases = []
    for (case, serialized), members in sorted(groups.items()):
        backend_rows = {}
        for backend in sorted({backend for p in members for backend in p["timings"]}):
            metrics, ratios = {}, {}
            for metric in ("wall_median_ms", "cuda_event_median_ms"):
                values = [p["timings"][backend][metric] for p in members if metric in p["timings"].get(backend, {})]
                paired = [p["timings"][backend][metric] / p["timings"]["cubutterfly"][metric] for p in members
                          if metric in p["timings"].get(backend, {}) and metric in p["timings"].get("cubutterfly", {})]
                if values:
                    metrics[metric] = spread(values)
                if paired:
                    ratios[metric] = spread(paired)
            backend_rows[backend] = dict(process_count=sum(backend in p["timings"] for p in members),
                process_medians=metrics, paired_backend_over_cubutterfly_ratios=ratios,
                formal_performance_eligible=[p["formal_performance_eligible"].get(backend) for p in members])
        cases.append(dict(case=case, identity=json.loads(serialized), process_count=len(members),
                          backends=backend_rows, processes=members))
    report = dict(test="dated_complete_call_cuda_graph_aggregation", summarizer_sha256=sha(Path(__file__)),
        aggregation="per-case compatible identity groups only; median/min/max of process medians and paired process ratios; no heterogeneous geometric mean",
        ratio_definition="matched graph backend latency / matched graph cuButterfly latency within each process; >1 favors cuButterfly",
        timing_metrics="wall and CUDA events remain separate; filter reports CUDA-event graph samples only",
        completion_scope="successful release measurements only; missing/failing/unreleased reports are not completed",
        cases=cases, missing=missing, invalid=invalid, fno_reused_summary=reused)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), compatible_case_groups=len(cases), released_processes=sum(c["process_count"] for c in cases), missing=len(missing), invalid=len(invalid))))


if __name__ == "__main__":
    main()
