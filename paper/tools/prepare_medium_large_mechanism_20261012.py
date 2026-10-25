#!/usr/bin/env python3
"""Prepare one finite medium/large controlled pair at a time, without GPU use."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex

import prepare_principle_gap_campaign as base
import semantic_compile_broker as broker

ROOT = base.ROOT
CONFIG = ROOT / "paper/experiments/medium_large_mechanism_20261012.json"


def make_study(config, operator):
    shape = config["first_round"]
    if shape != dict(logN=20, batch=4, stage_partition=[10, 10], threads=256,
                     batch_tile_count=1, stage_overlap=False) or operator not in config["operators"]:
        raise ValueError("first round must remain the finite N2^20 B4 parallel contract")
    workload = dict(operator=operator, precision="word64" if operator == "ntt" else "fp32",
                    logN=20, batch=4, direction="forward", normalization="none", placement="out-of-place")
    mapping = dict(schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
                   backend="shared-iterative", compute_unit="radix2", stage_partition=[10, 10],
                   stage_overlap=False, batch_tile_count=1, local_stage_partitions=[], exchange_chunks=[])
    if operator == "ntt":
        workload.update(modulus=config["semantics"]["ntt_modulus"], input_order="natural", output_order="natural")
        mapping.update(threads_per_block=256, modular_multiply="shoup")
        axis, layouts = "dataflow_layout", ("linear", "hermes-xor")
    else:
        workload["accumulation"] = "native"
        mapping.update(tile_threads=256, fft_core="scalar", local_exchange="shared",
                       complex_multiply="four-mul", cross_twiddle="table")
        axis, layouts = "shared_layout", ("linear", "writer-aligned")
    study = dict(schema=base.controlled.SCHEMA, protocol=config["protocol"],
                 scope="Mechanism ablation I: layout-only fixed-core replay; not automatic-library E1, core transfer or re-search.",
                 cases=[dict(id=f"{operator}-n20-b4-layout", experiment="mechanism-ablation-I-layout",
                             workload=workload,
                             variants=[dict(id=name, mapping=dict(mapping, **{axis: layout}))
                                       for name, layout in zip(("linear", "writer-aligned"), layouts)],
                             comparisons=[dict(baseline="linear", treatment="writer-aligned", changed_axes=[axis])])])
    base.controlled._validate_study(study)
    return study


def module_points(study):
    case = study["cases"][0]
    workload = case["workload"]
    points = []
    for variant in case["variants"]:
        mapping = variant["mapping"]
        point = dict(backend="shared-iterative", operator=workload["operator"], precision=workload["precision"],
                     logN=workload["logN"], stage_partition=mapping["stage_partition"],
                     threads=mapping.get("tile_threads", mapping.get("threads_per_block")),
                     local_stage_partitions=[], exchange_chunks=[], writer_aligned=variant["id"] != "linear")
        if workload["operator"] == "ntt":
            point["output_order"] = "natural"
        else:
            point.update(accumulation="native", complex_multiply="four-mul", batch_tile=1)
        points.append(point)
    return points


def resources(study, config):
    workload = study["cases"][0]["workload"]
    width = 4 if workload["operator"] == "fwht" else 8
    n, batch = 1 << workload["logN"], workload["batch"]
    result = dict(value_bytes=width, dynamic_shared_bytes=2 * 1024 * width,
                  device_memory_planning_envelope_bytes=(8 * batch + 4) * n * width,
                  physical_groups=2, logical_stages_per_group=10, values_per_cta=1024,
                  threads_per_cta=256, butterfly_pairs_per_thread_per_stage=2,
                  ctas_per_group=batch * n // 1024, measured_allocation=False)
    if result["dynamic_shared_bytes"] > config["resource_limits"]["dynamic_shared_ceiling_bytes"] or result["device_memory_planning_envelope_bytes"] > config["resource_limits"]["device_estimate_ceiling_bytes"]:
        raise ValueError("first round exceeds its resource envelope")
    return result


def compile_dependencies(points, spec, output, do_compile=False):
    directory = Path(spec["environment"]["CUBUTTERFLY_PREPARE_BROKER"])
    document = broker._document(directory)
    if document["config"] != spec["jit_identity"]["config"]:
        raise ValueError("broker and frozen launch JIT configuration disagree")
    compiler = broker._validated_worker(directory)[1] if do_compile else None
    records = []
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        for point in points:
            identifier = broker._id(dict(broker_id=document["id"], mapping=point))
            path = directory / "records" / (identifier + ".json")
            if do_compile:
                try:
                    broker.runtime_request(directory, document["config"], point, output / "cpu_preparation_events.jsonl", timeout=120)
                except RuntimeError:
                    request = directory / "requests" / (identifier + ".json")
                    if request.exists():
                        broker._compile_one(directory, document, request, compiler, False)
            record = json.loads(path.read_text()) if path.exists() else dict(id=identifier, status="not_prepared")
            if record.get("status") == "ready" and base.sha(record["path"]) != record["module_sha256"]:
                raise ValueError("ready module hash mismatch")
            records.append(dict(record, mapping=point, record_path=str(path)))
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
    return records


def prepare(operator="fft", do_compile=False, config_path=CONFIG, output_root=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    output = (Path(output_root).resolve() if output_root else ROOT / config["output"]) / operator
    original = ROOT / config["identity_source"]
    spec = copy.deepcopy(json.loads(original.read_text()))
    for path, expected in ((base.__file__, spec["launcher_sha256"]),
                           (spec["library_path"], spec["library_sha256"])):
        if base.sha(path) != expected:
            raise ValueError("frozen identity changed")
    for name, expected in spec["binaries_sha256"].items():
        if base.sha(Path(spec["build_dir"]) / name) != expected:
            raise ValueError("frozen benchmark identity changed")
    study = make_study(config, operator)
    resource = resources(study, config)
    output.mkdir(parents=True, exist_ok=True)
    dependencies = compile_dependencies(module_points(study), spec, output, do_compile)
    base.save(output / "study.json", study)
    base.save(output / "prepared.json", base.controlled.prepare_document(study, output / "study.json", Path(spec["build_dir"])))
    spec.update(schema="cubutterfly-medium-large-mechanism-launch-v1", config_path=str(config_path),
                config_sha256=base.sha(config_path), entrypoint_sha256=base.sha(__file__),
                study_path=str(output / "study.json"), study_sha256=base.sha(output / "study.json"),
                cells=2, timing_processes=6, operator=operator, resource_contract=resource,
                task_statuses={"layout-fixed-core": "prepared", **{task["id"]: task["status"] for task in config["separate_tasks"]}},
                ready_dependencies=dependencies, numeric_policy=config["numeric_policy"],
                core_identity=config["core_identities"][operator], gpu_queried=False,
                status="ready-for-guarded-preflight" if all(row["status"] == "ready" for row in dependencies) else "blocked_module_preparation",
                acceptance="Execution, preparation, mapping, internal sanity and user numerical gates recorded separately; floating exploratory.")
    probe = Path(spec["build_dir"]) / "cubutterfly_stage_microbench"
    spec["binaries_sha256"][probe.name] = base.sha(probe)
    child = [spec["python"], str(Path(__file__).resolve()), "--guard-child", "--operator", operator]
    if output_root:
        child += ["--output-root", str(output_root.resolve())]
    argv = [spec["python"], str(ROOT / "scripts/run_isolated_gpu.py"), "--gpu-uuid", spec["gpu_uuid"],
            "--report-dir", str(output / "guard"), "--until", config["deadline"], "--", "env"]
    spec["launch_argv"] = argv + [f"{key}={value}" for key, value in sorted(spec["environment"].items())] + child
    spec["launch_command"] = shlex.join(spec["launch_argv"])
    spec["audit_command"] = shlex.join([spec["python"], str(Path(__file__).resolve()), "--operator", operator, "--audit-journal", str(output / "guard/release/timing.json")])
    # Preparation may transition to ready before launch; never rewrite an executed spec.
    launch = output / "launch.json"
    if (output / "guard").exists():
        raise ValueError("cannot replace an already launched specification")
    base.controlled._atomic_write(launch, spec)
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("fft", "fwht", "ntt"), default="fft")
    parser.add_argument("--compile", action="store_true", help="CPU-only selective preparation of this pair's two modules")
    parser.add_argument("--guard-child", action="store_true")
    parser.add_argument("--output-root", type=Path, help="fresh result root for a new immutable cohort")
    parser.add_argument("--audit-journal", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    output = (args.output_root.resolve() if args.output_root else ROOT / config["output"]) / args.operator
    if args.guard_child:
        spec = json.loads((output / "launch.json").read_text())
        if base.sha(__file__) != spec["entrypoint_sha256"] or spec["status"] != "ready-for-guarded-preflight":
            raise ValueError("guard entrypoint changed or modules not prepared")
        for dependency in spec["ready_dependencies"]:
            if dependency["status"] != "ready" or base.sha(dependency["path"]) != dependency["module_sha256"]:
                raise ValueError("prepared module identity changed")
        return base.guard_child(output)
    if args.audit_journal:
        result = base.audit_journal(json.loads(args.audit_journal.read_text()), make_study(config, args.operator))
        result.update(status_meaning="mapping/internal sanity only, not whole-run numerical acceptance",
                      numerical_gate="unassessed-user-acceptance; error distribution missing" if args.operator != "ntt" else "exact-full-batch" if result["status"] == "accepted" else "incomplete",
                      journal_sha256=base.sha(args.audit_journal))
        base.controlled._atomic_write(output / "mapping_audit.json", result)
    else:
        result = prepare(args.operator, args.compile, output_root=args.output_root)
    print(json.dumps(result, indent=2))
    return 1 if args.audit_journal and result["status"] != "accepted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
