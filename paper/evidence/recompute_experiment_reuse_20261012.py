"""CPU-only evidence extraction. Never pools cohorts or applies error thresholds."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sources = {}


def source(path, field):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        key = str(p.relative_to(ROOT))
    except ValueError:
        key = str(p)
    if p.exists():
        sources[key] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "bytes": p.stat().st_size, "verification": "machine-read; human acceptance not recorded"}
    else:
        sources[key] = {"status": "missing"}
    return {"path": key, "field": field, **sources[key]}


def read(path):
    source(path, "/")
    return json.loads((ROOT / path).read_text())


def stats(values):
    return {"pairs": len(values), "wins": sum(v > 1 for v in values),
            "losses": sum(v < 1 for v in values), "parity": sum(v == 1 for v in values),
            "geomean": math.exp(sum(map(math.log, values))/len(values)),
            "min": min(values), "max": max(values)} if values else None


report_path = "results/paper_targeted_20261012/unified_report/summary.json"
report = read(report_path)
operators = report["tables"]["operators"]
assert len(operators) == 48
assert len({(r["cohort"], r["id"]) for r in operators}) == len(operators)
groups = defaultdict(list)
for i, row in enumerate(operators):
    groups[(row["cohort"], row["operator"], row["precision"])].append((i, row))
operator_groups = []
for (cohort, operator, precision), rows in sorted(groups.items()):
    record = {"cohort": cohort, "operator": operator, "precision": precision,
              "rows": len(rows), "trials": sorted({r["trials"] for _, r in rows}),
              "acceptance": sorted({r["acceptance"] for _, r in rows}), "comparators": {},
              "evidence": [], "derived_from": source(report_path, "/tables/operators")}
    for baseline in ("cufft", "vkfft", "daofht", "gpu-ntt"):
        pairs = [r[baseline + "_us"]/r["cubutterfly_us"] for _, r in rows
                 if r.get(baseline + "_us") is not None and r.get("cubutterfly_us") is not None]
        record["comparators"][baseline] = stats(pairs)
    for i, r in rows:
        record["evidence"].append({"id": r["id"], "locator": f"/tables/operators/{i}",
           "measurements": [source(p, "backend latency/error fields; see snapshot locator") for p in r["source_paths"]],
           "identities": [source(p, "/identity") for p in r["identity_source_paths"]]})
    operator_groups.append(record)

filter_rows = []
for i, r in enumerate(report["tables"]["applications"]):
    if r["cohort"] != "applications-20261012-filter-v2":
        continue
    filter_rows.append({**r, "speedup_vs_cupy": r["cupy_native_ms"]/r["cubutterfly_ms"],
        "cubutterfly_per_image_ms": r["cubutterfly_ms"]/r["B"],
        "evidence": [source(p, "/rows; /contract; /identity; /tolerance; /plan_metadata") for p in r["source_paths"]],
        "derived_from": source(report_path, f"/tables/applications/{i}")})
assert len(filter_rows) == 9
filter_groups = [{"B": b, **stats([r["speedup_vs_cupy"] for r in filter_rows if r["B"] == b])}
                 for b in sorted({r["B"] for r in filter_rows})]

core_records = []
for device, directory in (("A10080", "results/paper_completion_20260917/core_dataflow_ablation"),
                          ("A10040", "results/paper_completion_20260917/core_dataflow_ablation/a100-40gb")):
    path = directory + "/measurements.csv"
    with (ROOT / path).open() as f:
        raw = list(csv.DictReader(f))
    cells = defaultdict(list)
    for r in raw:
        assert r["correct"] == "1"
        cells[(int(r["logN"]), int(r["batch"]), r["core"], r["organization"])].append(float(r["kernel_ms"]))
    medians = {k: median(v) for k, v in cells.items()}
    org, core = defaultdict(list), defaultdict(list)
    cases = []
    for n, b in sorted({(k[0], k[1]) for k in cells}):
        c = {f"{a}/{o}": medians[n,b,a,o] for a in ("native", "cufftdx-thread")
             for o in ("stagewise-hbm", "resident-prefix-2")}
        for a in ("native", "cufftdx-thread"):
            org[a].append(c[a+"/stagewise-hbm"]/c[a+"/resident-prefix-2"])
        for o in ("stagewise-hbm", "resident-prefix-2"):
            core[o].append(c["native/"+o]/c["cufftdx-thread/"+o])
        cases.append({"logN": n, "B": b, "median_ms": c})
    core_records.append({"device": device, "input_rows": len(raw), "cases": cases,
        "same_core_organization": {k: stats(v) for k,v in org.items()},
        "same_organization_core_native_over_cufftdx": {k: stats(v) for k,v in core.items()},
        "evidence": source(path, "kernel_ms grouped by logN,batch,core,organization; median over trial"),
        "summary_evidence": source(directory+"/summary.json", "/geomean or /geomeans"),
        "caveat": "Historical controlled fixed mappings; no current-build speed or equal-budget organization search claim."})

e3path = "results/paper_completion_20260918/e3_cross_operator/summary.json"
e3 = read(e3path)
compath = "results/paper_completion_20260918/e3_cross_operator/compilation.json"
compilation = read(compath)
module_records = []
for m in compilation["modules"]:
    module_records.append({"id": m["id"], "request": m["request"],
        "module": source(m["module"], "binary"), "recorded_sha256": m["sha256"],
        "manifest": source(str(Path(m["module"]).with_name("manifest.json")), "/mapping; /module_sha256")})
e3_record = {"results": e3, "evidence": source(e3path, "/rows; /identity; /accepted_units"),
    "journal": source(e3["journal"], "accepted trial records"), "modules": module_records,
    "actual_requested_changed_axes": ["local_stage_partitions: absent/[] -> [[2,2,2,2],[2,2]]"],
    "fixed_axes": ["shared-iterative backend", "stage_partition=[8,4]", "threads=128", "writer_aligned=true", "operator/precision/semantics within each pair"],
    "caveat": "One core per arithmetic family; local schedule and generated module change. This is not a complete two-core native-organization/template transfer or equal-budget re-search study. Generated temporary CUDA source is absent; exact compiled arithmetic identity needs historical source audit before stronger attribution."}
e3_record["lowering_audit"] = {
    "sources": [source("scripts/compile_module.py", "render_shared: group kernel choice, launch/grid/shared storage"),
                source("src/shared_resident.cuh", "run_leaf; shared_resident_kernel")],
    "source_identity_scope": "inspected current source; not asserted identical to historical compiler/source",
    "actual_lowering_changes": ["shared_iterative_kernel -> shared_resident_kernel",
        "stagewise shared exchange -> two-stage register leaves (four values per leaf thread)",
        "barrier/shared access schedule, thread work assignment and compiled wrapper"],
    "preserved_in_inspected_lowering": ["one launch per physical group", "group grids", "128 threads",
        "two shared buffers sized by physical group", "inter-group HBM boundaries"],
    "interpretation": "Do not describe leaf0 as HBM writeback after every logical stage; both variants keep physical groups."}

rcpath = "paper/evidence/rc2_snapshot_20261002.json"
rc = read(rcpath)
rc_sources = {k: source(p, "raw frozen suite summary") for k,p in rc["sources"].items()}
old = read("paper/generated/current_results_summary.json")
composition_path = "results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json"
composition = read(composition_path)
composition_sources = [source(composition_path, "/rows; /model_report"),
    source(composition["public_trial_source"], "/rows public correctness and timings"),
    source(composition["pipeline_calibration"], "independent scheduling floor"),
    source("results/staged_migration_20260916/a100-80gb-composition/report.md", "Complete-plan validation; serial-case diagnostic")]
calibration_path = "../cuButterfly-v0.9.1-rc2/artifacts/revision_acceptance_20261001_replay_fix1/targets/nvidia-a100-80gb-pcie-80gb-2971e500-d3f/calibration/stage_calibration.json"
calibration = read(calibration_path)
validation = {k:v for k,v in calibration["model"]["validation"].items() if k != "checks"}
search_path = "results/paper_experiments_20260917/analysis/search_protocol_summary.json"
search = read(search_path)
search_sources = [source(search_path, "/e9; /limitations")]
search_sources.extend(source(s["path"], "frozen protocol/predictions/measurements/replay") for s in search["sources"])
protocols = {b: sorted({tuple(r[b+"_protocol"]) for r in operators if r[b+"_us"] is not None})
             for b in ("cubutterfly", "cufft", "vkfft", "daofht", "gpu-ntt")}
protocol_evidence = source("scripts/semantic_gpu_pipeline.py", "timing_steps (178-225), baseline_steps (232-322), public plan (425-428)")
for p in ("paper/evidence/application_snapshot_20261009.json", "paper/evidence/APPLICATION_RECORDS_20261009.md",
          "results/paper_experiments_20260917/analysis/REPORT.md", "paper/tools/run_e3_cross_operator.py",
          "paper/generated/rc2_unified_absolute_throughput.csv", "paper/generated/rc2_unified_absolute_throughput_summary.json",
          "paper/tools/render_rc2_unified_boxplot.py", "paper/sections/evaluation.tex", "docs/research_principles.md"):
    source(p, "/ or named subsection")

evidence_groups = [
 {"group": "operator comparison", "historical_reuse": "RC2 named build; retain each competitor independently and losses", "current_recorded_build": "48 rows in two cohorts; floating accuracy review pending", "must_remeasure": "If claiming latest working-tree build, incompatible binary/semantic/hardware/timing identity requires affected rerun", "missing": "matched GPU baselines for zeta/structured; word32 GPU-NTT; FWHT FP64 DAO; broader ranks/layouts/precision matrix"},
 {"group": "ablation 1: fixed component and disabled-component re-search", "historical_reuse": "fixed layout/depth/pipeline controls and current normal-timing/profile diagnostics", "current_recorded_build": "component diagnostics retained with actual_changed_axes and underfilled status", "must_remeasure": "equal-budget full versus disabled-component actual search and independent winner timing", "missing": "cross-operator re-search envelope, actual avoidance of harmful mechanism, occupancy/L2 hit rate"},
 {"group": "ablation 2: core x organization", "historical_reuse": "FFT native/cuFFTDx four cells on A10080 and A10040; E3 one-core cross-arithmetic local organization witness", "current_recorded_build": "FFT core x overlap only", "must_remeasure": "same-core original/matched well-filled versus template organizations, controlled plus equal-budget searched comparisons", "missing": "second replaceable core for FWHT/NTT/zeta/structured and complete cross table; retained generated historical E3 CUDA source"},
 {"group": "applications", "historical_reuse": "FNO complete-call loss; QuaRot random-weight B1 near parity; separately pinned builds", "current_recorded_build": "filter v2 nine matched real-batch points; exploratory tolerance unassessed", "must_remeasure": "affected application build/contract changes only; repaired QuaRot pending full calls", "missing": "CKKS complete RNS bootstrap; accepted floating policy for filter; this-round FNO batch campaign"},
 {"group": "selection and costs", "historical_reuse": "finite-pool E9 offline replay and planning/first-use costs with separate protocol", "current_recorded_build": "cache receipts and selected mapping identities in filter; do not imply rank1 calibration consumption", "must_remeasure": "actual online equal-budget selection and consumed profile/cache compatible with current binary", "missing": "E6 eligible prediction priors; calibrated E9 stage-service (zero covered priors); full install/tuning/amortization evidence"},
]
data = {"schema": "experiment-reuse-20261012-v1", "submission_ready": False,
 "human_verification": "not recorded", "policy": "baseline/cuButterfly; unique cohort,id; no cross-cohort deduplication or pooling; no new floating threshold",
 "operator_groups": operator_groups, "filter_v2_rows": filter_rows, "filter_v2_by_batch": filter_groups,
 "core_factorial_recomputed": core_records, "e3_audit": e3_record, "rc2": {"snapshot": rc, "raw_sources": rc_sources},
 "historical_selection": {"values": old["search_evidence"], "evidence": source("paper/generated/current_results_summary.json", "/search_evidence; /limitations")},
 "coverage": evidence_groups, "current_missing": report["missing"], "sources": sources}
data["primary_protocol"] = {"by_backend_recorded": protocols, "evidence": protocol_evidence,
    "scope": "all measured rows in this out-of-place matrix; 10 warmup executes, 20 calls captured, 7 CUDA-event graph replays divided by 20, median; adapters inside execute. In-place and historical cohorts use different protocols."}
data["historical_reuse_checks"] = {
    "search_16_candidate_pool": {"reuse": "yes, bounded offline replay; not calibrated stage-service or current online tuning cost", "evidence": search_sources},
    "composition_27_plan": {"reuse": "yes, historical diagnostic; preserve FWHT largest error and unresolved overlap assumption", "plan_rows": len(composition["rows"]), "evidence": composition_sources},
    "local_561_point": {"reuse": "yes, frozen RC2 local holdout validation; different task from whole-plan composition", "values": validation,
        "evidence": source(calibration_path, "/model/validation/checks; /model/validation/heldout_count; /model/validation/heldout_median_relative_error; /model/validation/heldout_p90_relative_error")}}
(OUT / "EXPERIMENT_REUSE_20261012.json").write_text(json.dumps(data, indent=2)+"\n")

lines = ["# Experiment evidence reuse (2026-10-12)", "", "Machine-audited draft; human scientific acceptance is not recorded. All numerical evidence, raw file SHA256, field locators, build identities and transformations are in [the JSON ledger](EXPERIMENT_REUSE_20261012.json). Recompute with `CUDA_VISIBLE_DEVICES='' python3 paper/evidence/recompute_experiment_reuse_20261012.py`. No GPU measurement is performed.", "", "## Reuse boundaries", "", "Historical method evidence remains usable under its named build. Recorded latest-campaign binaries are not the current working-tree build. Each raw identity is retained; binary compatibility must be checked before calling a measurement current. Do not merge E1 and NTT-extension observations, even where semantic requests coincide. Float error distributions/reference errors remain descriptive with accuracy review pending; integer/NTT exact-pass requires exact equality. No new threshold is imposed.", "", "| Evidence group | Historical reuse | Recorded campaign | Remeasure / missing |", "| --- | --- | --- | --- |"]
for g in evidence_groups:
    lines.append(f"| {g['group']} | {g['historical_reuse']} | {g['current_recorded_build']} | {g['must_remeasure']}; {g['missing']} |")
lines += ["", "## Operator groups", "", "All 48 rows retained. Ratios below are descriptive for floating rows, not accepted accuracy-qualified wins. Cohort, operator and precision stay separate; competitors never share a mean.", "", "| Cohort | Operator / precision | Rows / process trials | Comparator | Pairs | Wins | GM | Range |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for g in operator_groups:
    for b,s in g["comparators"].items():
        if s:
            lines.append(f"| {g['cohort']} | {g['operator']} / {g['precision']} | {g['rows']} / {g['trials']} | {b} | {s['pairs']} | {s['wins']} | {s['geomean']:.6f} | {s['min']:.6f}--{s['max']:.6f} |")
    if not any(g["comparators"].values()):
        lines.append(f"| {g['cohort']} | {g['operator']} / {g['precision']} | {g['rows']} / {g['trials']} | no matched external time | NA | NA | NA | NA |")
lines += ["", "## Batch FFT filtering: new nine-point cohort", "", "V2 is a new application cohort, not nine additional operator rows. Each point uses independent images in a real batch and a shared filter recomputed once per complete call. Native competitor padding differs as explicitly recorded. Planning/selection are outside warm call timing. Tolerance remains unassessed, so retain exploratory status and losses.", "", "| Input | B | cuB ms/batch | CuPy ms/batch | CuPy/cuB | cuB ms/image |", "| --- | --- | --- | --- | --- | --- |"]
for r in filter_rows:
    lines.append(f"| {r['input']} | {r['B']} | {r['cubutterfly_ms']:.6f} | {r['cupy_native_ms']:.6f} | {r['speedup_vs_cupy']:.6f} | {r['cubutterfly_per_image_ms']:.6f} |")
for g in filter_groups:
    lines.append(f"\nBatch {g['B']}: {g['wins']}/{g['pairs']} descriptive wins; GM {g['geomean']:.6f}, range {g['min']:.6f}--{g['max']:.6f}.")
lines += ["", "## Core and organization audit", "", "Raw factorial CSV medians reproduce the organization gains. A10040 summary core-speedup fields use cuFFTDx/native, opposite to A10080 and the manuscript definition. Consequently the manuscript's claim that imported-core execution slows on A10040 is reversed. Use the recomputed native/cuFFTDx ratios below, not the inverse summary values.", "", "| Historical GPU | Native stage/resident GM | cuFFTDx stage/resident GM | Native/cuFFTDx stage GM | Native/cuFFTDx resident GM |", "| --- | --- | --- | --- | --- |"]
for r in core_records:
    o=r["same_core_organization"];c=r["same_organization_core_native_over_cufftdx"]
    lines.append(f"| {r['device']} | {o['native']['geomean']:.6f} | {o['cufftdx-thread']['geomean']:.6f} | {c['stagewise-hbm']['geomean']:.6f} | {c['resident-prefix-2']['geomean']:.6f} |")
lines += ["", "E3 fixes the shared-iterative backend, physical [8,4] groups, 128 threads, writer-aligned mapping and paired user semantics. The requested change is local_stage_partitions from absent/empty to [[2,2,2,2],[2,2]], which changes local execution schedule and compiled module. It is one core in each arithmetic family and a local organization witness, not a two-core transfer cross table or a re-search experiment. Source temporary CUDA files were removed; compiled modules and manifests remain hash-bound. Avoid claiming isolated HBM traffic or a strictly identical compiled kernel from these records.", ""]
for r in e3["rows"]:
    lines.append(f"- Historical E3 {r['operator']} N=4096 B={r['batch']}: leaf0/leaf2 {r['leaf2_speedup']:.6f} ({r['leaf0_median_ms']:.9f}/{r['leaf2_median_ms']:.9f} ms); JSON /e3_audit/results/rows.")
lines += ["", "## Existing manuscript number sources", "", "- RC2 operator main figure: frozen sibling repository integrated_repairs_20261001_replay_fix1/a100-80gb suite summaries; generated rc2_unified_absolute_throughput.csv and renderer are presentation derivatives. Existing speedup_summary mixes competitors within FFT-FP32 and FWHT groups; use per-competitor RC2 snapshot/raw suites for numerical claims. Old RC2 speed is not the latest library speed.", "- FFT fixed-core table and GM: core_dataflow_ablation/measurements.csv; A10040 replication has its own CSV and protocol. Correct the ratio direction before reusing its core claim.", "- E3 cross-arithmetic ratios: e3_cross_operator/summary.json and a100-80gb/journal.json, with compilation.json/module manifests for actual axes.", "- Layout, depth, pipeline and finite-pool selection claims: paper/generated/current_results_summary.json, its provenance control report and named frozen source. E9 stage-service has zero covered priors and is structural ordering; offline replay is not online tuning cost or calibrated prediction validation.", "- Historical FNO/QuaRot and first-use costs: application_snapshot_20261009.json and APPLICATION_RECORDS_20261009.md. FNO slowdown and QuaRot random-weight near-parity must remain. CKKS compatibility probe is not complete bootstrap timing.", "", "## Safe prose conclusions", "", "The historical FFT factorial demonstrates organization gains with both tested cores, with replication on the two named A100 devices. The bounded FWHT/word64 NTT control extends local organization evidence to two arithmetic families, while a second core per family and equal-budget organization search remain open. Current operator observations cover all six families and retain unsupported external contracts and negative cells; floating correctness remains under review. The new batch-filter records show strong workload dependence and losses at larger batches, so operator gains cannot establish end-to-end benefit. Fixed-component diagnostics and finite-pool replay do not replace a cross-operator disabled-component re-search experiment.", ""]
lines += ["## Protocol and older model evidence", "", "The primary out-of-place matrix records the same graph protocol for every measured backend: ten warmup executes, twenty complete executes per captured graph, seven CUDA-event replays normalized by twenty, then the sample median. Required baseline scaling and order/layout adapters are inside execute. E1 has one process observation; the NTT extension has three. This statement is bounded to the actual measured rows; in-place reset timing, old RC2 and profiler studies have their own protocols.", "", "The historical sixteen-candidate E9 replay remains reusable as finite-pool policy evidence. Its stage-service policy has zero covered priors and uses structural fallback; invocation wall time is not online tuning cost. The historical twenty-seven-plan composition study remains reusable as a diagnostic with all correctness trials retained, including the large FWHT prediction error and unqualified overlap assumption. The RC2 local holdout study retains 561 covered checks, median relative error 1.1733% and P90 6.0294%. These numbers are /model/validation fields in the hash-bound RC2 calibration checkpoint. Local interpolation and full-plan composition are distinct prediction tasks; neither result is a current-build blanket model claim.", "", "E3 lowering audit: inspected current render_shared selects shared_iterative_kernel for leaf0 and shared_resident_kernel for leaf2. The latter keeps four values per two-stage leaf in registers, changing shared access, barriers, thread work and wrapper code. Both retain one launch per physical group, the group grid, threads and inter-group HBM boundaries. Do not claim per-stage HBM writeback in leaf0. Current source explains the implementation pattern but is not asserted to be the missing historical generated CUDA source.", ""]
(OUT / "EXPERIMENT_REUSE_20261012.md").write_text("\n".join(lines))
print(json.dumps({"operator_rows":len(operators), "operator_groups":len(operator_groups),
                  "filter_rows":len(filter_rows), "filter_by_batch":filter_groups,
                  "core_records": [{"device":r["device"],"core":r["same_organization_core_native_over_cufftdx"]} for r in core_records],
                  "sources":len(sources)}, indent=2))
