#!/usr/bin/env python3
"""Prepare finite stage-training inventories for the paper's missing E6/E9 tests.

Only independent groups are training observations. Whole-plan timings remain
held out. E9 keeps its original 16-point structural pools; calibration is
explicitly targeted to these pools, not advertised as structural cold start.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/tools"))
from prepare_experiments import canonical, module_manifest, read, write, workload
from heldout_protocol import candidate_set, workload_id
from search_pool_protocol import candidate_pool


def inventories():
    modulus = 576460756061519873
    e6_domain = [workload("fft", 18, 16, modulus), workload("fwht", 12, 1, modulus)]
    e9_domain = [workload("fft", 8, 1, modulus), workload("ntt", 12, 1024, modulus)]
    e6, e9 = [], []
    for w in e6_domain:
        for c in candidate_set(w):
            # Select an unobserved interior integer after stage acquisition,
            # using load presence only, before any full-plan measurements.
            e6.append(dict(c["point"], batch=32, min_batch=8))
    for w in e9_domain:
        for c in candidate_pool(w):
            lower, upper = (1, 3) if w["operator"] == "fft" else (1008, 1040)
            e9.append(dict(c["point"], batch=upper, min_batch=lower))
    return e6_domain, e9_domain, e6, e9


def prepare(target_path, output):
    target, source = read(target_path)
    e6_domain, e9_domain, e6, e9 = inventories()
    specification = dict(schema="cubutterfly-paper-model-completion-v1", target=source,
        protocol=dict(warmup=10000, repeat=100, trials=3, verify_batches=0),
        e6=dict(configurations=len(e6), requested_load_interval=[8, 32],
                whole_plan_holdout="smallest common unobserved interior batch; presence only, no timing ranking"),
        e9=dict(configurations=len(e9), pool_size_per_workload=16, budgets=[4, 8, 16],
                calibration_scope="targeted physical-stage acquisition before frozen whole-plan predictions",
                structural_cold_start_claimed=False),
        limits=["New warmed timing cohort; never append to the old 100-warmup checkpoint.",
                "No whole-plan timing enters the stage prior.",
                "A failed coverage gate remains a result, never silently a structural fallback.",
                "Requested intervals do not certify a complete hardware-migration domain."])
    manifest = output / "preparation.json"
    if manifest.exists() and canonical(json.loads(manifest.read_text())) != canonical(specification):
        raise ValueError("preparation changed; choose a new output directory")
    write(manifest, specification)
    write(output / "e6_domain.json", {"workloads": e6_domain})
    write(output / "e9_domain.json", {"workloads": e9_domain})
    write(output / "e6_stage_points.json", {"points": e6})
    write(output / "e9_stage_points.json", {"points": e9})
    cases = []
    for index, point in enumerate(e6 + e9):
        semantics = {k: v for k, v in point.items() if k not in {"mapping_json", "min_batch"}}
        cases.append(dict(id=f"stage-config-{index}", workload=semantics,
                          variants=[dict(id="training", mapping=json.loads(point["mapping_json"]))]))
    modules = module_manifest({"stage-training": {"cases": cases}}, target)
    write(output / "precompile/modules.json", modules)
    return dict(e6_configurations=len(e6), e9_configurations=len(e9), modules=modules["module_count"])


def prepare_repairs(target_path, audit_path, output):
    """Reacquire only explicitly incomplete configurations in a new cohort."""
    target, target_source = read(target_path)
    audit, audit_source = read(audit_path)
    snapshot = audit["snapshots"][-1]
    checkpoint, checkpoint_source = read(Path(snapshot["source"]))
    if checkpoint_source["sha256"] != snapshot["sha256"]:
        raise ValueError("audited checkpoint changed")
    identity = checkpoint["identity"]
    if identity.get("gpu_uuid", identity.get("uuid")) != target["gpu_uuid"]:
        raise ValueError("audit and target are different devices")
    missing = [row for row in snapshot["configuration_rows"] if not row["complete"]]
    points = []
    for row in missing:
        found = []
        for candidate in checkpoint["candidates"]:
            point = candidate["point"]
            mapping = json.loads(point["mapping_json"])
            threads = mapping.get("tile_threads", mapping.get("threads_per_block"))
            if all(point.get(k) == row[k] for k in ("operator", "precision", "logN", "placement")) and threads == row["threads"]:
                found.append(point)
        if len(found) != 1:
            raise ValueError("incomplete configuration does not uniquely resolve to a physical request")
        points.extend(found)
    write(output / "preparation.json", dict(schema="cubutterfly-paper-calibration-repair-v1",
        target=target_source, audit=audit_source, checkpoint=checkpoint_source,
        protocol=dict(warmup=10000, repeat=100, trials=3), configurations=len(points),
        scope="new warmed cohort for the audited incomplete configurations only; old checkpoint preserved"))
    write(output / "stage_points.json", {"points": points})
    cases = [dict(id=f"repair-{i}", workload={k: v for k, v in p.items() if k != "mapping_json"},
                  variants=[dict(id="repair", mapping=json.loads(p["mapping_json"]))])
             for i, p in enumerate(points)]
    write(output / "precompile/modules.json", module_manifest({"repairs": {"cases": cases}}, target))
    return dict(configurations=len(points))


def freeze_e6(target_path, checkpoint_path, output, build_dir):
    """Choose new batch semantics using presence only, then freeze the public protocol."""
    import argparse
    from heldout_protocol import freeze
    target, target_source = read(target_path)
    checkpoint, source = read(checkpoint_path)
    domain, _, templates, _ = inventories()
    workloads, points, selections = [], [], []
    for original in domain:
        matches = [p for p in templates if p["operator"] == original["operator"]]
        observed = {int(r["sample"]["batch"]) for r in checkpoint.get("records", [])
                    if isinstance(r.get("sample"), dict)
                    and r["sample"].get("operator") == original["operator"]
                    and int(r["sample"].get("logN", -1)) == original["logN"]}
        unused = [b for b in range(9, 32) if b not in observed]
        # When adaptive refinement observes every load, the valid remaining
        # question is composition of known stage services. Never relabel it
        # as unseen-batch interpolation or delete stage observations.
        selected_batch = unused[0] if unused else 16
        selected = dict(original, batch=selected_batch)
        workloads.append(selected)
        points.extend({**selected, "mapping_json": p["mapping_json"]} for p in matches)
        selections.append(dict(operator=original["operator"], logN=original["logN"],
                               observed_batches=sorted(observed), selected_batch=selected_batch,
                               mode="unseen-batch" if unused else "known-stage-composition"))
    write(output / "selection.json", dict(stage_checkpoint=source, target=target_source,
        rule="smallest unobserved interior batch 9..31; if none, composition-only at predefined batch16",
        full_plan_observations_read=False, selections=selections))
    write(output / "workloads.json", {"workloads": workloads})
    write(output / "points.json", {"points": points})
    result = freeze(argparse.Namespace(workloads=output / "workloads.json",
        stage_checkpoint=checkpoint_path, output_dir=output,
        composition_holdout=any(s["mode"] == "known-stage-composition" for s in selections),
        candidate_points=output / "points.json", workload_id=[workload_id(w) for w in workloads],
        target_uuid=target["gpu_uuid"], python=target["python"],
        profile=ROOT / target["profile"], build_dir=build_dir))
    write_execution_plan(result, target, output)
    return dict(status=result["status"], workloads=len(workloads), candidates=len(points))


def write_execution_plan(protocol, target, output):
    commands = []
    experiment = protocol["experiment"].lower()
    for key in ("describe_command", "predict_command", "command",
                "evaluate_command" if experiment == "e6" else "replay_command"):
        argv = list(protocol["gpu_measurement"][key])
        environment = dict(CUBUTTERFLY_COMPILE_MODE="research", PYTHONUNBUFFERED="1")
        if argv and argv[0] == "env":
            argv.pop(0)
            while argv and "=" in argv[0] and not argv[0].startswith("--"):
                name, value = argv.pop(0).split("=", 1)
                environment[name] = value
        if key == "predict_command":
            argv.append("--require-covered")
        requires_gpu = key in {"describe_command", "command"}
        if requires_gpu:
            argv = [target["python"], str(ROOT / "paper/tools/run_with_gpu_telemetry.py"),
                    "--gpu-uuid", target["gpu_uuid"], "--stop-on-thermal-throttle", "--output-prefix",
                    str(output / (experiment + "_" + key)), "--", *argv]
        commands.append(dict(id=f"{experiment}-{key}", argv=argv, env=environment,
                             cwd=str(ROOT), requires_gpu=requires_gpu))
    write(output / "execution_plan.json", dict(schema="cubutterfly-prepared-experiment-commands-v1",
                                               commands=commands))


def freeze_e9(target_path, checkpoint_path, output, build_dir):
    from search_pool_protocol import freeze
    target, _ = read(target_path)
    _, workloads, _, _ = inventories()
    write(output / "workloads.json", {"workloads": workloads})
    result = freeze(argparse.Namespace(workloads=output / "workloads.json",
        stage_checkpoint=checkpoint_path, output_dir=output, workload_id=None,
        exclude_protocol=None, target_uuid=target["gpu_uuid"], python=target["python"],
        profile=ROOT / target["profile"], build_dir=build_dir))
    write_execution_plan(result, target, output)
    return dict(status=result["status"], workloads=len(workloads),
                candidates=result["gpu_measurement"]["pool_point_count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repair-audit", type=Path)
    parser.add_argument("--freeze-e6", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--freeze-e9", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-a100-cufftdx")
    args = parser.parse_args()
    result = (freeze_e9(args.target, args.freeze_e9, args.output_dir.resolve(), args.build_dir)
              if args.freeze_e9 else
              freeze_e6(args.target, args.freeze_e6, args.output_dir.resolve(), args.build_dir)
              if args.freeze_e6 else
              prepare_repairs(args.target, args.repair_audit, args.output_dir.resolve())
              if args.repair_audit else prepare(args.target, args.output_dir.resolve()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
