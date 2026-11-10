#!/usr/bin/env python3
"""Matched VkFFT application trials; preserve raw data and reject mixed protocols."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from summarize_e2e_fft_opt_20261013 import ROOT, GPU, METRICS, sha, identity, protocol, timing_records

NEW_SHA = "4f25ddd1fca9f5cfeb34db589980d7ffae8935d36be0e0b3fc39c4ff9d6eebbd"
OLD_SHA = "89a038905a45e2d2ab072c142f9bde17ed97b42ed86106bdac656e5ec0f0f33b"


def contains_vkfft(value):
    if isinstance(value, dict):
        return any(contains_vkfft(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_vkfft(item) for item in value)
    return isinstance(value, str) and "vkfft" in value.lower()


def plan_evidence(data):
    plans = data.get("plan_metadata", data.get("setup", {}).get("cubutterfly", {}).get("plans", []))
    records = []
    for plan in plans:
        mapping = plan.get("mapping", {})
        axes = mapping.get("physical", {}).get("axes")
        counts = Counter(axis.get("core", "unknown") for axis in axes) if isinstance(axes, list) else None
        reason = plan.get("selection_reason", plan.get("reason", ""))
        parsed, parse_error = None, None
        if isinstance(reason, str) and "; candidates=" in reason:
            try:
                candidates, end = json.JSONDecoder().raw_decode(reason.split("; candidates=", 1)[1])
                if not isinstance(candidates, list) or not all(isinstance(row, dict) for row in candidates):
                    raise ValueError("candidate payload is not a list of records")
                parsed = [row for row in candidates if contains_vkfft(row)]
            except (ValueError, TypeError) as error:
                parse_error = str(error)
        records.append({"direction": plan.get("direction"), "algorithm": plan.get("algorithm"),
            "selection_source": plan.get("selection_source"), "selection_scope": plan.get("selection_scope"),
            "actual_axis_provider_counts": dict(counts) if counts is not None else None,
            "actual_vkfft_axis_count": sum(count for core, count in counts.items() if "vkfft" in core.lower())
                if counts is not None else None,
            "axis_count_note": "physical.axes only; resource configuration entries are not counted",
            "mapping": mapping, "selection_reason": reason,
            "vkfft_candidate_measurements": parsed, "candidate_parse_error": parse_error,
            "full_plan_metadata": plan})
    return records


def ratios(numerator, denominator):
    result = {}
    for metric in METRICS:
        if metric in numerator and metric in denominator:
            first, second = numerator[metric], denominator[metric]
            if not math.isfinite(first) or not math.isfinite(second) or first <= 0 or second <= 0:
                raise ValueError("nonpositive/nonfinite timing: " + metric)
            result[metric] = first / second
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/e2e_vkfft_axis_20261013")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/e2e_fft_opt_20261013")
    parser.add_argument("--library-sha256", default=NEW_SHA)
    parser.add_argument("--baseline-library-sha256", default=OLD_SHA)
    parser.add_argument("--without-filter", action="store_true",
                        help="summarize the 10 Poisson and phase-correlation trials; exclude filter")
    args = parser.parse_args()
    missing, invalid, processes = [], [], []

    def load(root, cohort, case, trial, filename, expected_sha):
        directory = root / (case + "_t" + str(trial))
        path = directory / "release" / filename
        filter_case = case.startswith("filter")
        if filename == "*.json":
            matches = [p for p in (directory / "release").glob("image*.json")
                       if ".trace." not in p.name]
            if len(matches) == 1:
                path = matches[0]
        status_path = directory / "status.json"
        record = dict(case=case, trial=trial, cohort=cohort, path=str(path))
        try:
            status = json.loads(status_path.read_text()) if status_path.is_file() else None
        except (ValueError, OSError) as error:
            invalid.append(dict(record, reason="unreadable guard status: " + str(error)))
            return None
        record["guard_status"] = status
        if not path.is_file():
            missing.append(dict(record, expected_path=str(path)))
            if status and status.get("state") in ("failed", "error", "rejected", "aborted"):
                invalid.append(dict(record, reason="guard records failure without a released report"))
            return None
        data = None
        try:
            data = json.loads(path.read_text())
            record.update(report_sha256=sha(path), source_report=data)
            if not status or status.get("state") != "complete":
                raise ValueError("release lacks complete guard status")
            if data.get("error") or data.get("cpu_only") or data.get("status") not in (
                    None, "complete", "completed", "pass", "passed", "ok", "success"):
                raise ValueError("failed report or not a GPU measurement")
            ident = identity(data)
            if ident.get("library_sha256") != expected_sha or ident.get("gpu_uuid") != GPU:
                raise ValueError("unexpected library SHA or GPU1 UUID")
            record.update(identity=ident, library_sha256=expected_sha, gpu_uuid=GPU,
                protocol=protocol(data, filter_case), timings=timing_records(data, filter_case),
                all_reported_timings=data.get("timings"), quality=data.get("quality"),
                error_acceptance=data.get("error_acceptance"), selected_plans=plan_evidence(data))
            if not record["selected_plans"]:
                raise ValueError("missing actual public plan metadata")
            for timing in record["timings"].values():
                ratios(timing, timing)
            return record
        except (ValueError, KeyError, TypeError, OSError) as error:
            invalid.append(dict(record, reason=str(error)))
            return None

    for trial in (0, 1):
        cohort = [("poisson_%d_b%d" % (size, batch), "periodic_poisson.json")
                  for size in (1024, 2048) for batch in (4, 16)]
        cohort.append(("registration_2048_b16", "phase_correlation.json"))
        if not args.without_filter:
            cohort.append(("filter_b16", "*.json"))
        for case, filename in cohort:
            new = load(args.results_root, "new", case, trial, filename, args.library_sha256)
            old = load(args.baseline_root, "baseline", case, trial, filename, args.baseline_library_sha256)
            process = dict(case=case, trial=trial, new=new, baseline=old,
                           protocol_matched=None, comparisons={})
            processes.append(process)
            if new:
                process["comparisons"]["current_backend_over_new_public"] = {
                    backend: ratios(timing, new["timings"]["cubutterfly"])
                    for backend, timing in new["timings"].items() if backend != "cubutterfly"}
            if new and old:
                differences = [key for key in new["protocol"]
                               if new["protocol"][key] != old["protocol"].get(key)]
                process["protocol_matched"] = not differences
                if differences:
                    invalid.append(dict(case=case, trial=trial, path=new["path"], baseline_path=old["path"],
                        reason="unmatched runner/input/norm/protocol", differing_fields=differences,
                        differences={key: {"new": new["protocol"][key], "baseline": old["protocol"].get(key)}
                                     for key in differences}))
                else:
                    process["comparisons"]["matched_baseline_over_new"] = {
                        backend: ratios(old["timings"][backend], timing)
                        for backend, timing in new["timings"].items() if backend in old["timings"]}

    performance_changes = {}
    for metric in METRICS:
        values = [row["comparisons"]["matched_baseline_over_new"]["cubutterfly"][metric]
                  for row in processes if row["protocol_matched"] is True and
                  metric in row["comparisons"]["matched_baseline_over_new"]["cubutterfly"]]
        if values:
            performance_changes[metric] = dict(matched_trials=len(values),
                baseline_over_current_min=min(values), baseline_over_current_max=max(values),
                current_slower_trials=sum(value < 1 for value in values))
    report = dict(schema_version=1, summarizer_sha256=sha(Path(__file__)),
        expected_library_sha256=args.library_sha256, baseline_library_sha256=args.baseline_library_sha256, gpu_uuid=GPU,
        expected_cases=10 if args.without_filter else 12,
        scope="Poisson 8 trials + phase correlation 2 trials" +
            ("; filter excluded (not rerun)" if args.without_filter else "; filter 2 trials"),
        status="rejected_mixed_identity_or_invalid" if invalid else
            "partial_missing_reports" if missing else "complete",
        ratio_definition="same-process external backend / new public cuButterfly, or matched old / new latency; >1 favors new",
        aggregation="per-case per-trial only; wall and CUDA events separate; no heterogeneous geometric mean",
        accuracy="full source errors and structural checks retained; no new numerical threshold",
        matched_public_performance_changes=performance_changes,
        processes=processes, missing=missing, invalid=invalid)
    output = args.results_root / "summary.json"
    markdown = args.results_root / "summary.md"
    args.results_root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# VkFFT Application Cohort", "", "Status: `%s`. GPU1 `%s`." % (report["status"], GPU), "",
        "Scope: %s. Protocol matched: %d/%d." % (report["scope"],
            sum(row["protocol_matched"] is True for row in processes), report["expected_cases"]), "",
        "Current library SHA256: `%s`. Baseline library SHA256: `%s`." %
            (args.library_sha256, args.baseline_library_sha256), "",
        "Execution protocol: complete application CUDA Graph calls. Wall and CUDA events are reported separately. "
        "No geometric mean or numerical threshold is introduced.", "",
        "Baseline / current latency ratios above 1 favor current; ratios below 1 retain regressions.", "",
        "These are observed trial differences; no statistical significance claim is made.", ""]
    for metric, change in performance_changes.items():
        lines.append("%s: baseline / current %.4f–%.4fx; current slower in %d/%d matched trials." %
            (metric, change["baseline_over_current_min"], change["baseline_over_current_max"],
             change["current_slower_trials"], change["matched_trials"]))
    lines.extend(["", "| 场景 | 类型 | 对照 | 加速比 |", "| --- | --- | --- | --- |"])
    def formatted(values):
        return "; ".join(("wall" if metric.startswith("wall") else "event") + " %.4fx" % value
                         for metric, value in values.items())
    for process in processes:
        label = "%s t%d" % (process["case"], process["trial"])
        operator_type = "2D 实数 FFT（FP32，正向 R2C / 逆向 C2R）"
        for backend, ratio in process["comparisons"].get("current_backend_over_new_public", {}).items():
            lines.append("| %s | %s | %s same process / new public | %s |" %
                         (label, operator_type, backend, formatted(ratio)))
        ratio = process["comparisons"].get("matched_baseline_over_new", {}).get("cubutterfly")
        if ratio:
            lines.append("| %s | %s | baseline public / current public | %s |" %
                         (label, operator_type, formatted(ratio)))
        if not process["new"]:
            lines.append("| %s | %s | new report missing or invalid | unavailable |" % (label, operator_type))
        elif not process["baseline"] or process["protocol_matched"] is False:
            lines.append("| %s | %s | baseline missing, invalid, or unmatched | unavailable |" % (label, operator_type))
    lines.extend(["", "## Per-Trial Latencies", "", "Full wall/event sample arrays and errors are retained in summary.json.", "",
        "| Trial | Cohort | Backend | Wall Median ms | CUDA Event Median ms |",
        "| --- | --- | --- | --- | --- |"])
    for process in processes:
        for cohort in ("new", "baseline"):
            record = process[cohort]
            if record:
                for backend, timing in record["timings"].items():
                    lines.append("| %s t%d | %s | %s | %s | %s |" % (process["case"], process["trial"],
                        cohort, backend, timing.get("wall_median_ms", "unavailable"),
                        timing.get("cuda_event_median_ms", "unavailable")))
    lines.extend(["", "## Actual Axis Providers", "",
        "Counts come from physical.axes. Complete mappings, selection reasons, and parsed VkFFT candidate records are in summary.json.", "",
        "| Trial | Cohort | Direction | Actual Provider Counts | VkFFT Axes |",
        "| --- | --- | --- | --- | --- |"])
    for process in processes:
        for cohort in ("new", "baseline"):
            record = process[cohort]
            if record:
                for plan in record["selected_plans"]:
                    lines.append("| %s t%d | %s | %s | %s | %s |" % (process["case"], process["trial"], cohort,
                        plan["direction"], json.dumps(plan["actual_axis_provider_counts"], sort_keys=True),
                        plan["actual_vkfft_axis_count"]))
    lines.extend(["", "## Missing And Invalid", "", "Missing: %d; invalid: %d." % (len(missing), len(invalid)), ""])
    lines.extend("- Missing %s `%s` t%d: `%s`; guard=%s" % (row["cohort"], row["case"], row["trial"],
        row["path"], json.dumps(row.get("guard_status"), sort_keys=True)) for row in missing)
    lines.extend("- Invalid `%s` t%d: %s%s" % (row["case"], row["trial"], row["reason"],
        "; fields=" + ", ".join(row["differing_fields"]) if row.get("differing_fields") else "") for row in invalid)
    markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=report["status"], cases=len(processes),
        available_new=sum(row["new"] is not None for row in processes),
        matched=sum(row["protocol_matched"] is True for row in processes), missing=len(missing),
        invalid=len(invalid), output=str(output), markdown=str(markdown))))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
