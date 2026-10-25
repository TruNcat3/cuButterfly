#!/usr/bin/env python3
"""Independent CPU preparation of one FWHT local-residency pair."""
import argparse
import copy
import json
from pathlib import Path
import shlex

import prepare_medium_large_mechanism_20261012 as medium
import prepare_principle_gap_campaign as base

ROOT = base.ROOT
CONFIG = ROOT / "paper/experiments/medium_large_residency_20261012_v2.json"


def study_and_points(config, source):
    study = copy.deepcopy(json.loads(Path(source["study_path"]).read_text()))
    case = study["cases"][0]
    if case["workload"]["operator"] != "fwht" or (case["workload"]["logN"], case["workload"]["batch"]) != (20, 4):
        raise ValueError("residency must retain FWHT N2^20 B4")
    case["id"] = "fwht-n20-b4-local-residency"
    case["experiment"] = "mechanism-ablation-I-local-residency"
    for variant, name, parts in zip(case["variants"], ("stage-publication", "resident2"),
                                    ([], [[2] * 5, [2] * 5])):
        mapping = variant["mapping"]
        if mapping["stage_partition"] != [10, 10] or mapping["tile_threads"] != 256:
            raise ValueError("physical partition or launch threads changed")
        variant["id"] = name
        mapping.update(shared_layout="linear", local_stage_partitions=parts)
    case["comparisons"] = [dict(baseline="stage-publication", treatment="resident2", changed_axes=["local_stage_partitions"])]
    study["scope"] = config["claim"]
    base.controlled._validate_study(study)
    points = medium.module_points(study)
    for point, variant in zip(points, case["variants"]):
        point.update(writer_aligned=False, local_stage_partitions=variant["mapping"]["local_stage_partitions"])
    return study, points


def audit(journal, study):
    normalized = []
    for document in (journal, study):
        value = copy.deepcopy(document)
        for case in value["cases"]:
            for variant in case["variants"]:
                variant["id"] = {"stage-publication": "linear", "resident2": "writer-aligned"}[variant["id"]]
            case["comparisons"] = [dict(baseline="linear", treatment="writer-aligned", changed_axes=["local_stage_partitions"])]
        normalized.append(value)
    result = base.audit_journal(*normalized)
    result.update(label_normalization="Audit helper labels linear/writer-aligned mean stage-publication/resident2 here; shared layout is linear in both.",
                  numerical_gate="unassessed-user-acceptance; internal sanity only",
                  claim="One public parameter with induced ownership/register/publication changes, not strict machine-level single variable.")
    return result


def prepare(do_compile=False):
    config = json.loads(CONFIG.read_text())
    source = json.loads((ROOT / config["source_launch"]).read_text())
    output = ROOT / config["output"]
    if (output / "guard").exists():
        raise ValueError("refusing to replace a launched specification")
    study, points = study_and_points(config, source)
    output.mkdir(parents=True, exist_ok=True)
    dependencies = medium.compile_dependencies(points, source, output, do_compile)
    base.save(output / "study.json", study)
    base.save(output / "prepared.json", base.controlled.prepare_document(study, output / "study.json", Path(source["build_dir"])))
    spec = copy.deepcopy(source)
    spec.update(config_path=str(CONFIG), config_sha256=base.sha(CONFIG), entrypoint_sha256=base.sha(__file__),
                study_path=str(output / "study.json"), study_sha256=base.sha(output / "study.json"),
                ready_dependencies=dependencies, changed_implementation_axes=config["changed_implementation_axes"],
                status="ready-for-guarded-preflight" if all(row["status"] == "ready" for row in dependencies) else "blocked_module_preparation",
                source_launch_sha256=base.sha(ROOT / config["source_launch"]),
                task_statuses={"fwht-local-residency": "prepared"})
    child = [spec["python"], str(Path(__file__).resolve()), "--guard-child"]
    argv = [spec["python"], str(ROOT / "scripts/run_isolated_gpu.py"), "--gpu-uuid", spec["gpu_uuid"],
            "--report-dir", str(output / "guard"), "--until", config["deadline"], "--", "env"]
    spec["launch_argv"] = argv + [f"{key}={value}" for key, value in sorted(spec["environment"].items())] + child
    spec["launch_command"] = shlex.join(spec["launch_argv"])
    spec["audit_command"] = shlex.join([spec["python"], str(Path(__file__).resolve()), "--audit-journal", str(output / "guard/release/timing.json")])
    base.controlled._atomic_write(output / "launch.json", spec)
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--guard-child", action="store_true")
    parser.add_argument("--audit-journal", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    output = ROOT / config["output"]
    if args.guard_child:
        spec = json.loads((output / "launch.json").read_text())
        if base.sha(__file__) != spec["entrypoint_sha256"] or spec["status"] != "ready-for-guarded-preflight":
            raise ValueError("entrypoint changed or modules not ready")
        for dependency in spec["ready_dependencies"]:
            if base.sha(dependency["path"]) != dependency["module_sha256"]:
                raise ValueError("module changed")
        return base.guard_child(output)
    if args.audit_journal:
        result = audit(json.loads(args.audit_journal.read_text()), json.loads((output / "study.json").read_text()))
        result["journal_sha256"] = base.sha(args.audit_journal)
        base.controlled._atomic_write(output / "mapping_audit.json", result)
    else:
        result = prepare(args.compile)
    print(json.dumps(result, indent=2))
    return 1 if args.audit_journal and result["status"] != "accepted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
