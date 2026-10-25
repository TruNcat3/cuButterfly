#!/usr/bin/env python3
"""Prepare the paper's bounded experiment queue without acquiring a GPU.

This generates inputs and command recipes. It never executes those commands,
queries hardware, changes an acceptance journal, or updates frozen paper tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from research_compile_requests import compile_request


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve(path, root=ROOT):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def workload(operator, log_n, batch, modulus):
    point = dict(operator=operator, precision="word64" if operator == "ntt" else "fp32",
                 logN=log_n, batch=batch, placement="out-of-place", direction="forward",
                 normalization="none")
    if operator == "ntt":
        point.update(modulus=modulus, input_order="natural", output_order="natural")
    return point


def factorial_workloads(plan):
    specification = plan["factorial"]
    fft = [workload("fft", log_n, batch, specification["ntt_modulus"])
           for log_n in specification["fft_logN"]
           for batch in sorted({1, max(1, specification["total_elements"] // (1 << log_n))})]
    other = [workload(op, specification["other_logN"], batch, specification["ntt_modulus"])
             for op in ("fwht", "ntt") for batch in specification["other_batches"]]
    return fft, other


def shared_mapping(operator, partition=(6, 6)):
    mapping = dict(schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
                   backend="shared-iterative", compute_unit="radix2", stage_partition=list(partition))
    if operator == "ntt":
        mapping.update(threads_per_block=128, dataflow_layout="hermes-xor")
    else:
        mapping.update(tile_threads=128, fft_core="scalar", local_exchange="shared",
                       shared_layout="writer-aligned")
    return mapping


def controlled_studies(plan):
    """Explicit fixed-core controls, not an invented optimized-core factorial."""
    studies = {name: dict(schema="cubutterfly-controlled-experiment-v1", protocol=plan["protocol"],
                         scope="Fixed-core organization controls; target feasibility and timing pending.", cases=[])
               for name in ("layout", "partition", "batch")}
    for operator in ("fft", "fwht", "ntt"):
        point = workload(operator, 12, 32, plan["factorial"]["ntt_modulus"])
        base = shared_mapping(operator)
        field = "dataflow_layout" if operator == "ntt" else "shared_layout"
        studies["layout"]["cases"].append(dict(
            id=f"{operator}-layout", experiment="E4", workload=point,
            variants=[dict(id="linear", mapping={**base, field: "linear"}),
                      dict(id="writer-aligned", mapping=base)],
            comparisons=[dict(baseline="linear", treatment="writer-aligned", changed_axes=[field])]))
        studies["partition"]["cases"].append(dict(
            id=f"{operator}-partition", experiment="E4", workload=point,
            variants=[dict(id="depth3", mapping=shared_mapping(operator, (3, 3, 3, 3))),
                      dict(id="depth6", mapping=base)],
            comparisons=[dict(baseline="depth3", treatment="depth6", changed_axes=["stage_partition"])]))
        for batch in (1, 4, 16, 64):
            variants = [dict(id="bulk", mapping={**base, "stage_overlap": False, "batch_tile_count": 1})]
            pairs = []
            for tile in (1, 4, 16):
                if tile > batch:
                    continue
                name = f"pipeline-tile{tile}"
                variants.append(dict(id=name, mapping={**base, "stage_overlap": True, "batch_tile_count": tile}))
                pairs.append(dict(baseline="bulk", treatment=name,
                                  changed_axes=["stage_overlap"] + (["batch_tile_count"] if tile != 1 else [])))
            studies["batch"]["cases"].append(dict(
                id=f"{operator}-batch{batch}", experiment="E5",
                workload=workload(operator, 12, batch, plan["factorial"]["ntt_modulus"]),
                variants=variants, comparisons=pairs))
    return studies


def module_manifest(studies, target):
    modules = {}
    for study in studies.values():
        for case in study["cases"]:
            for variant in case["variants"]:
                point = {**case["workload"], "mapping_json": canonical(variant["mapping"])}
                request = compile_request(point, target["compile_mode"])
                if request is None:
                    continue
                key = hashlib.sha256(canonical(request).encode()).hexdigest()
                entry = modules.setdefault(key, dict(id=key, request=request, representative_point=point, uses=[]))
                entry["uses"].append(f"{case['id']}/{variant['id']}")
    capability = str(target["compute_capability"]).split(".")
    if len(capability) != 2 or not all(part.isdigit() for part in capability):
        raise ValueError("target compute_capability must be major.minor")
    return dict(schema="cubutterfly-research-precompile-v1", status="ready", complete=True,
                policy=target["compile_mode"], profile=dict(sm=int(capability[0]) * 10 + int(capability[1])),
                source="explicit-paper-controls-without-device-probe", target_declared=target["gpu_uuid"],
                runtime_feasibility_verified=False, module_count=len(modules), modules=list(modules.values()))


def prepare(plan_path, target_path, output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan, plan_source = read(plan_path)
    target, target_source = read(target_path)
    if plan.get("schema") != "cubutterfly-paper-plan-v1" or target.get("schema") != "cubutterfly-paper-target-v1":
        raise ValueError("unsupported plan/target schema")
    experiment_ids = [x["id"] for x in plan["experiments"]]
    required_ids = {f"E{i}" for i in range(1, 11)}
    supported_ids = required_ids | {"E11", "E12", "E13"}
    if (len(set(experiment_ids)) != len(experiment_ids)
            or not required_ids.issubset(experiment_ids)
            or not set(experiment_ids).issubset(supported_ids)):
        raise ValueError("plan must contain E1 through E10 once, with optional E11/E12/E13 entries")
    if target.get("compile_mode") not in ("research", "auto", "precompiled") or not target.get("gpu_uuid"):
        raise ValueError("target must specify UUID and compile mode")
    for name in ("trials", "repeat"):
        if type(plan["protocol"][name]) is not int or plan["protocol"][name] < 1:
            raise ValueError("invalid timing protocol")
    acceptance = resolve(target["acceptance_dir"], root)
    if output == root or output == acceptance or output in acceptance.parents or acceptance in output.parents:
        raise ValueError("preparation output must be separate from repository root and live acceptance")
    if (output / "acceptance.json").exists():
        raise ValueError("refusing to write into an acceptance directory")
    run_identity = dict(plan_sha256=plan_source["sha256"], target_sha256=target_source["sha256"],
                        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    previous = output / "execution_plan.json"
    if previous.exists() and read(previous)[0].get("preparation_identity") != run_identity:
        raise ValueError("preparation inputs changed; choose a new output directory")
    profile_path = resolve(target["profile"], root)
    provenance = [plan_source, target_source]
    warnings = []
    if profile_path.exists():
        profile, source = read(profile_path)
        provenance.append(source)
        for field in ("device", "compute_capability", "global_memory_bytes"):
            if str(profile.get(field)) != str(target[field]):
                raise ValueError(f"target/profile mismatch: {field}")
        uuid = profile.get("gpu_uuid", profile.get("uuid"))
        if uuid and uuid != target["gpu_uuid"]:
            raise ValueError("target/profile UUID mismatch")
    else:
        warnings.append("Target capability profile is missing; GPU recipes require calibration first.")
    e1_snapshot = None
    if (acceptance / "acceptance.json").exists():
        journal, source = read(acceptance / "acceptance.json")
        if journal.get("identity", {}).get("device", {}).get("uuid") != target["gpu_uuid"]:
            raise ValueError("acceptance belongs to another physical GPU")
        provenance.append(source)
        e1_snapshot = dict(status=journal.get("status"), completed=sum(c.get("status") == "complete" for c in journal["cells"]),
                           expected=len(journal["cells"]), source=source)
    fft, others = factorial_workloads(plan)
    for name, workloads in (("e2", fft), ("e3", others), ("e7", fft + others)):
        write(output / f"workloads_{name}.json", dict(schema="cubutterfly-install-search-v1",
              scope="Paper semantic domains; workload enumeration is not measured acceptance.", workloads=workloads))
    requirements = dict(schema="cubutterfly-paper-factorial-requirements-v1", status="needs-matched-mappings",
                        cases=[dict(workload=w, cells=[dict(label=label, mapping=None, status="unbound")
                              for label in plan["factorial"]["required_cells"]]) for w in fft + others])
    write(output / "factorial_requirements.json", requirements)
    write(output / "heldout_protocol.json", dict(
        schema="cubutterfly-paper-heldout-protocol-v1", status="needs-frozen-training-inventory",
        target_uuid=target["gpu_uuid"], metric_population="whole-workload-heldout",
        candidate_workloads=fft + others, training_workloads=None, heldout_workloads=None,
        frozen_training_checkpoint=None, frozen_training_checkpoint_sha256=None,
        rules=["Assign disjoint whole-workload IDs and freeze the independent service inventory before timing held-out plans.",
               "Publish zero-new-service predictions and gaps before any target-driven service extension.",
               "Report post-extension ranking and acquisition cost separately; do not relabel it as cold-start holdout.",
               "Never fit a held-out whole-plan time; finalist-only ranking is a separate population."]))
    write(output / "search_pool_protocol.json", dict(
        schema="cubutterfly-paper-search-pool-protocol-v1", status="needs-independent-pool-and-prior-scores",
        target_uuid=target["gpu_uuid"], budgets=[4, 8, 16], random_seeds=[17, 23, 41, 59, 83],
        policies=["stage-service", "logical-traffic", "random", "local-search"],
        workload_domain=fft + others, pool_manifest=None, pool_manifest_sha256=None,
        prior_predictions=None, prepared_timing_identity=None,
        rules=["Use one independently defined finite pool and fixed core per workload for every policy.",
               "Seeds, failed attempts and exploration all consume the same total candidate budget.",
               "Freeze predictions before reading measured pool latencies; no outcome-based pool selection.",
               "Report best-so-far and regret against the measured pool optimum, not a global optimum.",
               "Offline replay cannot establish online compilation/calibration/search wall time."]))
    write(output / "adaptation_audit.json", dict(
        schema="cubutterfly-paper-adaptation-audit-v1", status="needs-audited-extension-case",
        categories=["mathematically-legal", "lowerable", "compiled", "correctly-executed", "measured"],
        case_kind=None, heldout_shape_or_new_semantics=None, before_revision=None, after_revision=None,
        reused_paths=[], modified_paths=[], handwritten_schedule_changes=None,
        rules=["Separate a new shape from a new semantic adapter.",
               "Audit actual code changes and preparation costs; do not infer reuse from similar operator names.",
               "Reuse E1 semantic variants where their contracts match; do not duplicate those GPU trials."]))
    studies = controlled_studies(plan)
    for name, study in studies.items():
        write(output / f"controlled_{name}.json", study)
    modules = module_manifest(studies, target)
    write(output / "control_precompile/modules.json", modules)
    py = target["python"]
    environment = dict(CUDA_VISIBLE_DEVICES=target["gpu_uuid"], CUBUTTERFLY_COMPILE_MODE=target["compile_mode"])
    commands = []

    def command(identifier, experiment, argv, *, gpu=False, prerequisites=(), env=None):
        commands.append(dict(id=identifier, experiment=experiment, argv=[str(x) for x in argv], cwd=str(root),
                             env=(environment if gpu else {"CUDA_VISIBLE_DEVICES": "", "CUBUTTERFLY_COMPILE_MODE": target["compile_mode"]}) if env is None else env,
                             requires_gpu=gpu, prerequisites=list(prerequisites), executed=False))

    common = [py, root / "scripts/run_research_acceptance.py", "--build-dir", resolve(target["build_dir"], root),
              "--output-dir", acceptance, "--workloads", root / next(e for e in plan["experiments"] if e["id"] == "E1")["workloads"]]
    for key in ("profile", "stage_checkpoint", "pipeline_calibration", "schedule_calibration"):
        common += ["--" + key.replace("_", "-"), resolve(target[key], root)]
    for key, value in {**plan["search"], **plan["protocol"]}.items():
        common += ["--" + key.replace("_", "-"), value]
    for path in target.get("mapping_seeds", []):
        common += ["--mapping-seeds", resolve(path, root)]
    for key in ("vkfft_binary", "fht_python", "gpuntt_binary"):
        if target.get(key):
            common += ["--" + key.replace("_", "-"), resolve(target[key], root)]
    command("e1-resume", "E1", common + ["--mode", "all"] + (["--resume"] if e1_snapshot else []), gpu=True,
            prerequisites=["Existing writer must have exited; do not start a second acceptance process.",
                           "Target is exclusive and calibration/build/protocol identities match."])
    for name in studies:
        argv = [py, root / "scripts/run_controlled_experiment.py", "--study", output / f"controlled_{name}.json",
                "--build-dir", resolve(target["build_dir"], root), "--output", output / "measurements" / f"{name}.json",
                "--gpu-uuid", target["gpu_uuid"]]
        experiment = "E5" if name == "batch" else "E4"
        command(f"{name}-prepare", experiment, argv + ["--mode", "prepare"])
        command(f"{name}-run", experiment, argv + ["--mode", "run", "--resume"], gpu=True,
                prerequisites=["Run CPU prepare first, then check target lowering and resolved comparison axes.",
                               "Run after E1 releases the GPU; incompatible/alias mappings are missing controls, not wins."])
    command("controls-compile", "E4/E5", [py, root / "scripts/precompile_research.py", "--mode", "compile",
            "--output-dir", output / "control_precompile", "--jobs", 2],
            prerequisites=["Only the explicit control modules are compiled; no full-matrix export or GPU query."])
    command("evidence-status", "E1/E6/E9/E10", [py, root / "paper/tools/summarize_experiments.py",
            "--acceptance-dir", acceptance, "--output", output / "evidence_status.json"])
    for name in ("theory", "construction"):
        command(f"check-{name}", "E8/E10", [py, root / f"paper/tools/check_{name}.py",
                "--output", output / f"{name}_checks.json"])
    migration_dir = output / "migration"
    command("e7-migration-preview", "E7", [py, root / "scripts/calibrate_hardware.py",
            "--build-dir", resolve(target["build_dir"], root), "--profile-dir", migration_dir,
            "--output-dir", migration_dir, "--search-workloads", output / "workloads_e7.json",
            "--stage-calibration", "full", "--search-only", "--search-seconds", 0, "--mode", "dry-run"], gpu=True,
            prerequisites=["Preview still queries hardware. Reuse a matching existing migration if one exists.",
                           "Create one target configuration per hardware cohort; do not copy timings between UUIDs."])
    queue = dict(schema="cubutterfly-paper-execution-plan-v1", status="prepared-not-executed",
                 preparation_identity=run_identity, target=target, sources=provenance, e1_snapshot=e1_snapshot,
                 control_cases={k: len(v["cases"]) for k, v in studies.items()}, control_module_count=modules["module_count"],
                 warnings=warnings, commands=commands, experiments=plan["experiments"],
                 remaining_protocols={
                     "E2/E3": "The bounded A100 80GB strict FFT factorial and FWHT/word64 NTT cross-arithmetic control are archived; broader core/operator coverage remains open.",
                     "E6": "Freeze a disjoint workload split and training-service inventory before evaluating held-out workloads.",
                     "E7": "Matched current-version V100/A100 execution and preparation-cost accounting remain pending.",
                     "E9": "Freeze an independent candidate pool and prior predictions for equal-budget policy comparison.",
                     "E10": "Audit adapter/code reuse and handwritten changes; prepared or compiled counts do not establish GPU correctness."})
    write(output / "execution_plan.json", queue)
    lines = ["# Generated paper experiment queue", "", "Commands are recipes, not executed jobs. GPU recipes wait for the existing E1 writer to exit.",
             "CPU prepare/compile does not certify target execution. Frozen paper evidence is unchanged.", ""]
    for item in commands:
        lines += [f"## {item['id']} ({item['experiment']}; {'GPU' if item['requires_gpu'] else 'CPU'})", ""]
        lines += [f"- {condition}" for condition in item["prerequisites"]]
        lines += ["", "```bash", "cd " + shlex.quote(item["cwd"]),
                  shlex.join(["env"] + [f"{k}={v}" for k, v in item["env"].items()] + item["argv"]), "```", ""]
    (output / "RUNBOOK.md").write_text("\n".join(lines))
    return queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "paper/experiments/plan.json")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.plan, args.target, args.output_dir)
    print(json.dumps(dict(status=result["status"], control_cases=result["control_cases"],
                         control_modules=result["control_module_count"], output=str(args.output_dir.resolve())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
