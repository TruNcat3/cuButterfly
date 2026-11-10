#!/usr/bin/env python3
"""CPU-only finite repair; reuse the frozen v1 guard child, not its failed journal."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import statistics

import prepare_principle_gap_campaign as base

ROOT = base.ROOT
CONFIG = ROOT / "experiments/workloads/principle_gap_campaign_20261012_v2.json"
REQUESTS = ("8e4cef433105ee43585397cc0d767512797a4f8ba14c64d6612dad93faebea93",
            "f7e9a834026e1bd24dc93fadc9d07e0a379e37aa68c98df61a8e3ee5737ef6e4")


def ready_modules():
    records = []
    for request in REQUESTS:
        path = ROOT / "results/plan_batch_trace_20261010/shared_compile/records" / (request + ".json")
        record = json.loads(path.read_text())
        if record.get("status") != "ready" or base.sha(record["path"]) != record["module_sha256"]:
            raise ValueError("repair dependency is not ready: " + request)
        records.append(dict(record, record_path=str(path), record_sha256=base.sha(path)))
    return records


def repair_study(config):
    study = base.layout_study(json.loads((ROOT / config["extends"]).read_text()))
    expected = ["fwht-n12-b1-layout", "fwht-n12-b16-layout", "ntt-n12-b1-layout"]
    if config["cases"] != expected or config["retained_v1_case"] != "ntt-n12-b16-layout":
        raise ValueError("repair must contain exactly the three incomplete pairs")
    study["cases"] = [case for case in study["cases"] if case["id"] in expected]
    study["scope"] += " Repair only; floating verification is internal sanity, not user numerical acceptance."
    base.controlled._validate_study(study)
    return study


def prepare(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    output = ROOT / config["output"]
    original = ROOT / config["v1_output"]
    spec = copy.deepcopy(json.loads((original / "launch.json").read_text()))
    # Keep the already launched wrapper byte-for-byte unchanged.
    if base.sha(base.__file__) != spec["launcher_sha256"]:
        raise ValueError("frozen guard child changed")
    modules = ready_modules()
    study = repair_study(config)
    base.save(output / "study.json", study)
    base.save(output / "prepared.json", base.controlled.prepare_document(study, output / "study.json", Path(spec["build_dir"])))
    spec.update(config_path=str(Path(config_path).resolve()), config_sha256=base.sha(config_path),
                study_path=str(output / "study.json"), study_sha256=base.sha(output / "study.json"),
                cells=6, timing_processes=18, status="prepared-not-executed",
                repair_generator_sha256=base.sha(__file__), ready_dependencies=modules,
                retained_v1_case=config["retained_v1_case"], numeric_policy=config["numeric_policy"],
                acceptance="Separate execution, preparation, mapping and numerical gates; floating performance exploratory")
    spec["launch_argv"] = [str(item).replace(str(original), str(output)) for item in spec["launch_argv"]]
    spec["launch_command"] = shlex.join(spec["launch_argv"])
    spec["audit_command"] = shlex.join([spec["python"], str(Path(__file__).resolve()), "--audit-journal",
                                        str(output / "guard/release/timing.json")])
    base.save(output / "launch.json", spec)
    return spec


def audit(journal, study, numeric_policy):
    result = base.audit_journal(journal, study)
    result["status_meaning"] = "mapping and internal sanity only; not whole-run or user floating numerical acceptance"
    result["numeric_policy"] = numeric_policy
    for row, case in zip(result["cases"], study["cases"]):
        row["numerical_gate"] = "unassessed-user-acceptance" if case["workload"]["operator"] == "fwht" else "exact-full-batch" if row["status"] == "accepted" else "incomplete"
    return result


def v1_evidence():
    root = ROOT / "results/paper_targeted_20261012/principle_gap_attempt_safe"
    release = root / "guard/release"
    journal = json.loads((release / "timing.json").read_text())
    study = json.loads((root / "study.json").read_text())
    mapping = base.audit_journal(journal, study)
    telemetry = json.loads((release / "telemetry.telemetry.json").read_text())
    events = [json.loads(line) for line in (release / "preparation_events.jsonl").read_text().splitlines() if line]
    rows = []
    for case, checks in zip(journal["cases"], mapping["cases"]):
        variants = []
        for variant in case["variants"]:
            trials = [float(row["kernel_ms"]) * 1000 for row in variant.get("trials", [])]
            preflight = variant["preflight"]
            variants.append(dict(id=variant["id"], preflight_status=preflight["status"],
                                 failure=preflight.get("reason"), trial_us=trials,
                                 median_us=statistics.median(trials) if trials else None,
                                 verified_batches=preflight.get("sample", {}).get("verified_batches"),
                                 correct=preflight.get("sample", {}).get("correct")))
        row = dict(id=case["id"], variants=variants, mapping_audit=checks,
                   numerical_gate="unassessed-user-acceptance; internal sanity only where executed; error distribution missing" if case["workload"]["operator"] == "fwht" else "exact-full-batch" if checks["status"] == "accepted" else "incomplete-pair")
        if checks["status"] == "accepted":
            row["baseline_over_treatment_median"] = variants[0]["median_us"] / variants[1]["median_us"]
        rows.append(row)
    sources = {name: dict(path=str(release / name), sha256=base.sha(release / name)) for name in
               ("timing.json", "telemetry.telemetry.json", "preparation_events.jsonl", "owned_processes.json")}
    return dict(schema="cubutterfly-principle-gap-run-audit-v1", status="partial-not-whole-run-accepted",
                sources=sources, cases=rows, retained_subset="ntt-n12-b16-layout",
                execution_gate=dict(guard=json.loads((root / "guard/status.json").read_text())["state"],
                                    telemetry_status=telemetry["status"], returncode=telemetry["returncode"],
                                    telemetry_scope=telemetry["sampling_scope"]),
                preparation_gate=dict(status="pending-events-recorded", events=events,
                                      note="Later ready modules do not retroactively clean this attempt. NTT B16 is archived as a complete conditional subset, not whole-run acceptance."),
                ready_now=ready_modules(), limitations=["Three trials do not establish significance.",
                "Kernel launch sequence, not full C API end-to-end.", "Layout-only mechanism ablation I, not fixed-core organization transfer (ablation II).",
                "NTT B1 treatment has three trials but its linear baseline is missing; no pair ratio."])


def combined_evidence():
    historical = v1_evidence()
    root = ROOT / "results/paper_targeted_20261012/principle_gap_v2"
    release = root / "guard/release"
    config = json.loads(CONFIG.read_text())
    journal = json.loads((release / "timing.json").read_text())
    mapping = audit(journal, repair_study(config), config["numeric_policy"])
    telemetry = json.loads((release / "telemetry.telemetry.json").read_text())
    event_path = release / "preparation_events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line] if event_path.exists() else []
    rows = []
    for case, checks in zip(journal["cases"], mapping["cases"]):
        variants = []
        for variant in case["variants"]:
            trials = [float(row["kernel_ms"]) * 1000 for row in variant.get("trials", [])]
            preflight = variant["preflight"]
            variants.append(dict(id=variant["id"], preflight_status=preflight["status"], trial_us=trials,
                                 median_us=statistics.median(trials) if trials else None,
                                 verified_batches=preflight.get("sample", {}).get("verified_batches"),
                                 correct=preflight.get("sample", {}).get("correct")))
        row = dict(id=case["id"], source="v2", variants=variants, mapping_audit=checks,
                   numerical_gate=checks["numerical_gate"])
        if checks["status"] == "accepted":
            row["baseline_over_treatment_median"] = variants[0]["median_us"] / variants[1]["median_us"]
        rows.append(row)
    rows.append(dict(historical["cases"][-1], source="v1-conditional-subset"))
    thermal = telemetry["thermal_samples"]
    return dict(schema="cubutterfly-principle-gap-run-audit-v2", status="performance-recorded-floating-numerical-acceptance-unassessed",
                cases=rows, v1=historical, v2_mapping_audit=mapping,
                execution_gate=dict(guard=json.loads((root / "guard/status.json").read_text())["state"],
                                    telemetry_status=telemetry["status"], returncode=telemetry["returncode"],
                                    thermal_samples=len(thermal), thermal_slowdown_observed=any(int(row[2]) & 0x60 for row in thermal),
                                    telemetry_scope=telemetry["sampling_scope"]),
                preparation_gate=dict(status="clean" if not events else "events-require-review", events=events,
                                      event_file_present=event_path.exists()),
                sources={name: dict(path=str(release / name), sha256=base.sha(release / name)) for name in
                         ("timing.json", "telemetry.telemetry.json", "owned_processes.json")},
                missing_dependency=dict(id="fwht-fixed-mapping-full-error-report", status="blocked_implementation",
                    completion="Expose full-batch error distribution for the same resolved fixed mapping/input/reference; report errors without inventing a user threshold."),
                limitations=historical["limitations"][:3] + ["FWHT performance is exploratory; internal sanity is not user numerical acceptance."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-journal", type=Path)
    parser.add_argument("--archive-v1", action="store_true")
    parser.add_argument("--archive-combined", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    if args.audit_journal:
        result = audit(json.loads(args.audit_journal.read_text()), repair_study(config), config["numeric_policy"])
        result["journal_sha256"] = base.sha(args.audit_journal)
        base.controlled._atomic_write(ROOT / config["output"] / "mapping_audit.json", result)
    elif args.archive_v1:
        result = v1_evidence()
        base.controlled._atomic_write(ROOT / "experiments/provenance/PRINCIPLE_GAP_RUN_20261012.json", result)
    elif args.archive_combined:
        result = combined_evidence()
        base.controlled._atomic_write(ROOT / "experiments/provenance/PRINCIPLE_GAP_RUN_20261012.json", result)
    else:
        result = prepare()
    print(json.dumps(result, indent=2))
    return 1 if args.audit_journal and result["status"] != "accepted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
