#!/usr/bin/env python3
"""Export the finite current campaign, using only persisted CPU-readable evidence.

Ratios use reported complete-call medians. Repeated NTT trials use the median
of paired trial ratios; group summaries give every case one vote. No historical
results, tuning populations, selection traces, or raw samples are imported.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


OPERATORS = {0: "fft", 1: "ntt", 2: "fwht", 3: "subset-zeta",
             4: "superset-zeta", 5: "structured-2x2"}
STORAGE = {0: "fp16", 1: "bf16", 2: "fp32", 3: "fp64", 4: "fp16",
           5: "bf16", 6: "fp32", 7: "fp64", 8: "word32", 9: "word64"}
CKKS_FILES = ("method1_n14", "method2_n14", "method1_n16_b1",
              "method1_n16_b8", "method1_n16_b32_cooled")


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def ratio(baseline, own):
    return baseline / own if number(baseline) and number(own) else None


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_summary(pairs):
    values = [p["speedup"] for p in pairs if number(p.get("speedup"))]
    return dict(count=len(values), geometric_mean=math.exp(sum(map(math.log, values)) / len(values)) if values else None,
                median=median(values) if values else None, minimum=min(values) if values else None,
                maximum=max(values) if values else None, wins=sum(v > 1 for v in values),
                losses=sum(v < 1 for v in values), ties=sum(v == 1 for v in values))


def speedup_allowed(row, floating=False):
    statuses = ("pass", "accuracy-review-required") if floating else ("pass",)
    return (row.get("status", "accuracy-review-required" if floating else "unreported") in statuses
            and row.get("performance_eligible") is not False
            and row.get("formal_performance_eligible") is not False)


def filter_report_eligible(row):
    """Use explicit failures and graph checks, not legacy tolerance-only flags."""
    graph = row.get("graph_changing_input_validation", {})
    return (row.get("accepted") is not False and row.get("error", {}).get("finite") is True
            and number(row.get("timing", {}).get("median_us")) is not None
            and all(graph.get(key) is True for key in
                    ("distinct_input_outputs", "previous_output_preserved", "restored_input_exact"))
            and len(graph.get("same_precision_cpu_errors", [])) == 2
            and all(error.get("finite") is True for error in graph["same_precision_cpu_errors"]))


def summarize(rows, keys):
    """Never combine cohorts, baselines, precisions, or categories implicitly."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    result = []
    for group, members in sorted(groups.items(), key=lambda x: str(x[0])):
        values = [r["speedup"] for r in members if number(r.get("speedup"))]
        result.append(dict(zip(keys, group), requests=len(members), measured=len(values),
                           missing=len(members) - len(values),
                           geometric_mean=math.exp(sum(map(math.log, values)) / len(values)) if values else None,
                           nominal_faster=sum(v > 1 for v in values),
                           minimum=min(values) if values else None,
                           maximum=max(values) if values else None))
    return result


def mapping_summary(mapping):
    mapping = mapping or {}
    keys = ("kind", "lowering", "backend", "fft_core", "hierarchical_core", "compute_unit",
            "stage_partition", "factor_partition", "tile_threads", "data_tiles_per_cta",
            "batch_tile_count", "prefetch_depth", "pipeline_buffers", "stage_overlap",
            "factor_overlap", "execution_group_mappings")
    result = {k: mapping[k] for k in keys if k in mapping}
    physical = mapping.get("physical", {})
    if physical:
        result["physical"] = {k: physical[k] for k in
                              ("axis_core", "axis_order", "layout_policy", "cost", "axes") if k in physical}
    return result


def semantic_fields(semantics):
    op = OPERATORS.get(semantics.get("operator"), "unknown")
    storage = semantics.get("storage")
    precision = STORAGE.get(storage, "unknown")
    if storage == 8 and op in ("subset-zeta", "superset-zeta"):
        precision = "uint32"
    return dict(operator=op, precision=precision, storage=storage,
                shape=semantics.get("shape"), batch=semantics.get("batch"),
                inverse=bool(semantics.get("inverse")))


class Exporter:
    def __init__(self, campaign):
        self.campaign = Path(campaign).resolve()
        self.sources = {}
        self.missing_sources = []
        self.identities = {}

    def read(self, relative, csv_file=False):
        path = self.campaign / relative
        if not path.is_file():
            self.missing_sources.append(relative)
            return [] if csv_file else {}
        raw = path.read_bytes()
        self.sources[relative] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
        if csv_file:
            return list(csv.DictReader(raw.decode().splitlines()))
        return json.loads(raw)

    def identity(self, name, data):
        identity = data.get("identity", data)
        keys = ("library_sha256", "library", "gpu_uuid", "gpu_name", "gpu_total_memory",
                "gpu_memory_bytes", "memory_bytes", "compile_mode", "runner_revision", "runner_sha256")
        self.identities[name] = {k: identity[k] for k in keys if k in identity}
        for key in ("binaries", "device", "artifacts", "source_binary_correspondence",
                    "CUBUTTERFLY_COMPILE_MODE", "CUDA_VISIBLE_DEVICES", "performance_eligible"):
            if key in identity:
                self.identities[name][key] = identity[key]
        protocol = identity.get("protocol", data.get("protocol"))
        if protocol:
            self.identities[name]["protocol"] = {k: v for k, v in protocol.items() if k not in
                                                  ("tolerances", "error_distribution")}

    def operator_row(self, row, cohort, source):
        sem = row.get("semantics", row.get("descriptor", {}).get("semantics", {}))
        fields = semantic_fields(sem)
        timing = number(row.get("timing", {}).get("median_ms"))
        own = timing * 1000 if timing else None
        correctness = "error-report-only" if fields["precision"].startswith(("fp", "bf")) else row.get("status", "unreported")
        floating = correctness == "error-report-only"
        request = dict(fields, id=row.get("id", row.get("case_id")), cohort=cohort,
                       own_us=own, status=row.get("status", "accuracy-review-required"),
                       algorithm=row.get("algorithm"), category=row.get("category"),
                       mapping_summary=mapping_summary(row.get("actual_mapping") or row.get("mapping")),
                       correctness=correctness, source=source)
        pairs = []
        for b in row.get("baselines", row.get("baseline_results", [])):
            baseline = number(b.get("timing", {}).get("median_ms"))
            baseline = baseline * 1000 if baseline else None
            pairs.append(dict(fields, case_id=request["id"], cohort=cohort, baseline=b["name"],
                              category=b.get("category"), own_us=own, baseline_us=baseline,
                              speedup=ratio(baseline, own) if speedup_allowed(row, floating) and speedup_allowed(b, floating) else None,
                              status=b.get("status", "unreported"),
                              correctness=correctness, source=source))
        return request, pairs

    def operators(self):
        index = self.read("measurement/results.json")
        manifest = self.read("manifest.json")
        self.identity("main66", index)
        requests, pairs, seen = [], [], set()
        for entry in index.get("rows", []):
            source = "measurement/" + entry["path"]
            envelope = self.read(source)
            row = envelope.get("row")
            if row is None:
                continue
            if entry.get("sha256") and envelope.get("sha256") != entry["sha256"]:
                raise ValueError("Row envelope/index hash mismatch: " + source)
            digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if envelope.get("sha256") and digest != envelope["sha256"]:
                raise ValueError("Row payload hash mismatch: " + source)
            request, row_pairs = self.operator_row(row, "main66", source)
            requests.append(request)
            pairs.extend(row_pairs)
            seen.add(request["id"])
        for case in manifest.get("cases", []):
            if case["id"] not in seen:
                request, _ = self.operator_row(dict(case, status="missing", timing={}), "main66", "manifest.json")
                requests.append(request)
        for cohort, source in (("fft_grouped2", "fft_xd_grouped_baseline/supplement.json"),
                               ("ntt5", "ntt_search_repair_5/results.json")):
            data = self.read(source)
            self.identity(cohort, data)
            for row in data.get("rows", []):
                request, row_pairs = self.operator_row(row, cohort, source)
                if cohort == "ntt5":
                    trials = row.get("trials", [])
                    own = [t["cubutterfly"]["median_ms"] for t in trials if t.get("cubutterfly")]
                    base = [t["gpu-ntt"]["median_ms"] for t in trials if t.get("gpu-ntt")]
                    ratios = [ratio(t.get("gpu-ntt", {}).get("median_ms"), t.get("cubutterfly", {}).get("median_ms")) for t in trials]
                    ratios = [v for v in ratios if v is not None]
                    request["own_us"] = median(own) * 1000 if own else None
                    row_pairs = [dict(semantic_fields(row["semantics"]), case_id=request["id"], cohort=cohort,
                                      baseline="gpu-ntt", category="adapter", own_us=request["own_us"],
                                      baseline_us=median(base) * 1000 if base else None,
                                      speedup=median(ratios) if ratios and speedup_allowed(row) else None, trials=len(trials),
                                      status=row.get("status"), correctness=request["correctness"], source=source)]
                requests.append(request)
                pairs.extend(row_pairs)
        return dict(requests=requests, pairs=pairs,
                    groups=summarize(pairs, ("cohort", "operator", "precision", "baseline", "category")))

    def cyclic(self):
        source = "recovery_followups_20261029/cyclic_mapping/measurement/results.json"
        supplement_source = "recovery_gpuntt_baseline_20261102/results.json"
        index, supplement = self.read(source), self.read(supplement_source)
        empty = dict(status="missing", requests=[], pairs=[], summary=paired_summary([]),
                     measured_requests=0, remaining_requests=None)
        if not index:
            return empty

        def checked_rows(report, report_source):
            if digest(report.get("rows", [])) != report.get("rows_sha256"):
                raise ValueError("Cyclic row index hash mismatch: " + report_source)
            result = {}
            for entry in report["rows"]:
                if entry["path"] != "rows/" + entry["id"] + ".json":
                    raise ValueError("Invalid cyclic row path")
                row_source = str(Path(report_source).parent / entry["path"])
                envelope = self.read(row_source)
                row = envelope.get("row", {})
                if (digest(row) != envelope.get("sha256") or envelope.get("sha256") != entry.get("sha256")
                        or row.get("id") != entry["id"] or row.get("status") != entry.get("status")):
                    raise ValueError("Cyclic row checksum or index mismatch: " + row_source)
                if row["id"] in result:
                    raise ValueError("Duplicate cyclic row")
                result[row["id"]] = (row, row_source, entry["sha256"])
            return result

        originals = checked_rows(index, source)
        additions = checked_rows(supplement, supplement_source) if supplement else {}
        source_identity = index.get("identity", {})
        if supplement:
            identity = supplement.get("identity", {})
            if identity.get("source_report_sha256") != self.sources[source]["sha256"]:
                raise ValueError("Cyclic source report hash mismatch")
            for key in ("gpu_uuid", "protocol"):
                if not source_identity.get(key) or identity.get(key) != source_identity[key]:
                    raise ValueError("Cyclic supplement identity mismatch: " + key)
            manifest_source = str(Path(source).parent.parent / "manifest.json")
            manifest = self.read(manifest_source)
            if (not manifest or digest(manifest) != source_identity.get("manifest_sha256")
                    or self.sources[manifest_source]["sha256"] != identity.get("manifest_sha256")):
                raise ValueError("Cyclic manifest hash mismatch")
            cases = {case["id"]: case for case in manifest.get("cases", [])}
            if any(case_id not in cases or row["descriptor"] != cases[case_id].get("descriptor")
                   for case_id, (row, _, _) in originals.items()):
                raise ValueError("Cyclic manifest semantics mismatch")
            root_hash = identity.get("backend", {}).get("root_helper_library", {}).get("sha256")
            if root_hash != source_identity.get("library_sha256"):
                raise ValueError("Cyclic supplement library mismatch")
            for name, value in identity.get("shared_sources", {}).items():
                if source_identity.get("sources", {}).get(name) != value:
                    raise ValueError("Cyclic shared runner mismatch: " + name)

        requests, pairs = [], []
        for case_id, (row, row_source, row_hash) in originals.items():
            baseline_records = [b for b in row.get("baselines", []) if b.get("name") == "gpu-ntt"]
            if len(baseline_records) != 1 or row.get("descriptor", {}).get("semantics", {}).get("operator") != 1:
                raise ValueError("Expected one GPU-NTT baseline per cyclic NTT request")
            baseline = baseline_records[0]
            added_source = None
            if case_id in additions:
                added, added_source, _ = additions[case_id]
                if baseline.get("status") != "missing-baseline" or baseline.get("support_status") != "missing-external-binding":
                    raise ValueError("Cyclic supplement would overwrite an existing baseline")
                if (added.get("source_row_sha256") != row_hash or added.get("descriptor") != row.get("descriptor")
                        or added.get("source_cubutterfly_timing") != row.get("timing")
                        or added.get("source_mapping") != row.get("mapping")):
                    raise ValueError("Cyclic supplement source row mismatch")
                replacements = added.get("baselines", [])
                if (added.get("status") != "pass" or row.get("status") != "pass" or len(replacements) != 1
                        or replacements[0].get("name") != "gpu-ntt" or replacements[0].get("status") != "pass"
                        or replacements[0].get("support_status") != "supported"):
                    raise ValueError("Invalid cyclic baseline supplement")
                baseline = replacements[0]
            if baseline.get("status") == "pass":
                if (not number(baseline.get("timing", {}).get("median_ms"))
                        or not number(row.get("timing", {}).get("median_ms"))
                        or not row.get("timing", {}).get("protocol")
                        or baseline.get("timing", {}).get("protocol") != row["timing"]["protocol"]):
                    raise ValueError("Cyclic baseline timing protocol mismatch")
            effective = dict(row, baselines=[baseline])
            request, row_pairs = self.operator_row(effective, "cyclic90", row_source)
            request.update(descriptor=row["descriptor"], mapping=row.get("mapping"), timing=row.get("timing"),
                           original_baselines=row.get("baselines"), baselines=[baseline], source_row_sha256=row_hash,
                           supplement_source=added_source)
            requests.append(request)
            for pair in row_pairs:
                pair.update(supplemented=added_source is not None, supplement_source=added_source,
                            support_status=baseline.get("support_status"), reason=baseline.get("reason"),
                            source_row_sha256=row_hash, baseline_record=baseline,
                            semantics=row["descriptor"]["semantics"])
            pairs.extend(row_pairs)
        if additions.keys() - originals.keys():
            raise ValueError("Unknown cyclic supplement request")
        valid = [p for p in pairs if number(p.get("speedup"))]
        status = "complete-with-gaps" if len(valid) < len(requests) else "complete"
        if index.get("remaining_case_ids") or not index.get("status", "").startswith("complete"):
            status = "incomplete"
        return dict(status=status, requests=requests, pairs=pairs, summary=paired_summary(pairs),
                    original_pairs=paired_summary([p for p in pairs if not p["supplemented"]]),
                    supplemented_pairs=paired_summary([p for p in pairs if p["supplemented"]]),
                    word32=paired_summary([p for p in pairs if p["precision"] == "word32"]),
                    word64=paired_summary([p for p in pairs if p["precision"] == "word64"]),
                    measured_requests=len(requests), remaining_requests=len(index.get("remaining_case_ids", [])),
                    status_summary=dict(case_status_counts=dict(Counter(r["status"] for r in requests)),
                                        baseline_status_counts=dict(Counter(p["status"] for p in pairs)),
                                        baseline_support_counts=dict(Counter(p["support_status"] for p in pairs))),
                    source=source, supplement_source=supplement_source if supplement else None,
                    source_status=index.get("status"), supplement_status=supplement.get("status", "missing"),
                    identities=dict(original=source_identity, supplement=supplement.get("identity")),
                    scope="Independent cyclic NTT cohort; original cuButterfly timings with separately measured missing GPU-NTT baselines; not an interleaved rerun")

    def applications(self):
        pairs = []
        for name in CKKS_FILES + ("bootstrap_v1_small_b1_b8",):
            bootstrap = name.startswith("bootstrap")
            source = "ckks/" + ("bootstrap_prepared/" if bootstrap else "auto_schedule_36/") + name + ".csv"
            metadata = self.read(source[:-4] + ".metadata.json")
            self.identity(name, metadata)
            for row in self.read(source, csv_file=True):
                if not bootstrap and row["batch_submission"] != "measured_auto_loop_or_stage_batch":
                    continue
                own, baseline = number(row["cubutterfly_ms"]), number(row["native_ms"])
                exact = row["exact_ciphertext"] == "1" and int(row["mismatch_count"]) == 0
                metadata_valid = bool(metadata) and metadata.get("performance_eligible") is True and metadata.get("status") == "completed"
                eligible = exact and metadata_valid
                pairs.append(dict(scenario=row["scene"], type="NTT", N=int(row["N"]), batch=int(row["batch"]),
                                  baseline=row["baseline"], own_ms=own, baseline_ms=baseline,
                                  speedup=ratio(baseline, own) if eligible else None, cohort="ckks_bootstrap" if bootstrap else "ckks_auto36",
                                  eligibility="exact-ciphertext-validated" if eligible else "pending-unvalidated",
                                  correctness=dict(exact_ciphertext=row["exact_ciphertext"] == "1", mismatch_count=int(row["mismatch_count"])),
                                  method=row["keyswitch_method"], mode=row["mode"], source=source))
        for name in ("poisson-1024-b4", "poisson-2048-b4", "poisson-2048-b16"):
            source = "poisson/" + name + "/report.json"
            data = self.read(source)
            if not data:
                continue
            self.identity(name, data)
            own = number(data.get("timings", {}).get("cubutterfly_graph", {}).get("cuda_event_median_ms"))
            for backend in ("torch_graph", "cupy_graph"):
                baseline = number(data.get("timings", {}).get(backend, {}).get("cuda_event_median_ms"))
                pairs.append(dict(scenario="periodic-poisson", type="FFT", shape=[data["size"]] * 2,
                                  batch=data["batch"], baseline=backend, own_ms=own, baseline_ms=baseline,
                                  speedup=ratio(baseline, own), cohort="poisson_graph", mode="cuda_graph",
                                  eligibility="error-report-only", correctness="error-report-only", source=source))
        for size in (1024, 2048):
            name = "image%d_filter31_batch16" % size
            source = "fft_filters/" + name + "/" + name + ".current_graph_fast.json"
            data = self.read(source)
            if not data:
                continue
            self.identity(name, data)
            rows = {r["method"]: r for r in data.get("rows", [])}
            own = number(rows.get("cubutterfly", {}).get("timing", {}).get("median_us"))
            baseline = number(rows.get("cupy_fftconvolve_native_padding", {}).get("timing", {}).get("median_us"))
            compared = {name: rows.get(name, {}) for name in
                        ("cubutterfly", "cupy_fftconvolve_native_padding")}
            eligible = all(filter_report_eligible(row) for row in compared.values())
            pairs.append(dict(scenario="fft-filter31", type="FFT", shape=data["contract"]["image_shape"],
                              batch=data["contract"]["batch"], baseline="cupy_fftconvolve_native_padding",
                              own_ms=own / 1000 if own else None, baseline_ms=baseline / 1000 if baseline else None,
                              speedup=ratio(baseline, own) if eligible else None,
                              cohort="fft_filter_graph", mode="cuda_graph",
                              eligibility="error-report-only" if eligible else "invalid-or-incomplete",
                              formal_performance_eligible=eligible,
                              correctness="error-report-only", independent_processes=1,
                              statistical_limitations=["single independent process; no between-process uncertainty"],
                              source_numerical_reports={name: {key: row.get(key) for key in
                                  ("accepted", "formal_performance_eligible", "numerical_acceptance", "error",
                                   "graph_changing_input_validation")} for name, row in compared.items()},
                              source=source))
        strongest = {}
        for row in pairs:
            if row["cohort"] != "ckks_auto36":
                continue
            key = (row["scenario"], row["N"], row["batch"], row["method"])
            if key not in strongest or (number(row["baseline_ms"]) is not None and
                                       (number(strongest[key]["baseline_ms"]) is None or
                                        row["baseline_ms"] < strongest[key]["baseline_ms"])):
                strongest[key] = row
        strongest = [dict(r, baseline="strongest-native", selected_baseline=r["baseline"]) for r in strongest.values()]
        return dict(pairs=pairs, strongest_native_pairs=strongest,
                    groups=summarize(pairs, ("cohort", "scenario", "baseline")))

    def mechanisms(self):
        source = "mechanism/timing.json"
        data = self.read(source)
        self.identity("mechanism", data)
        pairs = []
        for case in data.get("cases", []):
            w = case["workload"]
            for comp in case.get("comparisons", []):
                paired = [p["baseline_kernel_ms"] / p["treatment_kernel_ms"] for p in comp.get("pairs", [])]
                pairs.append(dict(case_id=case["id"], operator=w["operator"], precision=w["precision"],
                                  shape=[2 ** w["logN"]], batch=w["batch"], baseline=comp["baseline"],
                                  treatment=comp["treatment"], baseline_ms=comp.get("baseline_median_kernel_ms"),
                                  own_ms=comp.get("treatment_median_kernel_ms"), speedup=median(paired) if paired else None,
                                  trial_speedups=paired, speedup_min=min(paired) if paired else None,
                                  speedup_max=max(paired) if paired else None,
                                  trials=len(paired), status=comp["status"],
                                  baseline_stage_partition=comp.get("baseline_resolved_mapping", {}).get("stage_partition"),
                                  treatment_stage_partition=comp.get("treatment_resolved_mapping", {}).get("stage_partition"),
                                  changed_axes=comp.get("actual_changed_axes"), source=source))
        return dict(pairs=pairs, scope="fixed-core stage organization; internal comparison, not external library speedups")

    def export(self):
        operators = self.operators()
        cyclic = self.cyclic()
        applications = self.applications()
        mechanisms = self.mechanisms()
        status = self.read("status.json")
        composed = self.read("composed_138/graph/summary.json")
        coverage = dict(campaign={k: status[k] for k in ("status", "remaining", "scope", "scope_note", "full_refresh_complete", "operator_track_complete", "measured_rows", "population", "phase") if k in status},
                        composed138={k: composed[k] for k in ("status", "complete_cells", "declared_cells", "failed_attempts") if k in composed},
                        cyclic={k: cyclic.get(k) for k in ("status", "measured_requests", "remaining_requests", "status_summary",
                                                         "source", "supplement_source", "source_status", "supplement_status")},
                        operator_status_counts=dict(Counter(r["status"] for r in operators["requests"] if r["cohort"] == "main66")))
        return dict(schema="current-paper-refresh-v1", campaign=self.campaign.name,
                    aggregation=dict(ratio="baseline complete-call median / own complete-call median; NTT5 median of paired trial ratios",
                                     mechanisms="median of paired trial baseline/treatment ratios, which can differ from the ratio of median times; trial_speedups and min/max retain the observed trial range",
                                     groups="geometric mean of per-case ratios within cohort/operator/precision/baseline/category; missing excluded, never zero",
                                     interpretation="nominal faster counts only; no statistical significance; all floating point results error-report-only",
                                     ckks="only named measured_auto_loop_or_stage_batch rows; bootstrap separate; strongest-native uses minimum baseline latency per request",
                                     exclusions=["historical results", "population CSV", "selection CSV", "sample CSV", "torch same-padding filter diagnostic"]),
                    operators=operators, cyclic=cyclic, applications=applications, mechanisms=mechanisms, coverage=coverage,
                    identities=self.identities, sources=[self.sources[k] for k in sorted(self.sources)],
                    missing_sources=sorted(set(self.missing_sources)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("results/full_current_20261025"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated/current_refresh/snapshot.json"))
    args = parser.parse_args()
    snapshot = Exporter(args.campaign).export()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Exported %d operator requests and %d application pairs to %s" %
          (len(snapshot["operators"]["requests"]), len(snapshot["applications"]["pairs"]), args.output))


if __name__ == "__main__":
    main()
