#!/usr/bin/env python3
"""CPU-only preparation and post-run identity audit for the finite principle gaps."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_controlled_experiment as controlled

CONFIG = ROOT / "experiments/workloads/principle_gap_campaign_20261012.json"
OUTPUT = ROOT / "results/paper_targeted_20261012/principle_gap_attempt_safe"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing changed preparation at {path}; select a fresh output directory")
        return
    controlled._atomic_write(path, value)


def layout_study(config):
    if config.get("schema") != "cubutterfly-principle-gap-campaign-v1":
        raise ValueError("unsupported campaign schema")
    task = next(item for item in config["tasks"] if item["id"] == "layout-fixed-core")
    if (task["operators"], task["logN"], task["batches"], task["stage_partition"], task["threads"]) != (
        ["fwht", "ntt"], 12, [1, 16], [6, 6], 32
    ):
        raise ValueError("layout campaign exceeds its frozen finite parallel contract")
    cases = []
    for operator in task["operators"]:
        for batch in task["batches"]:
            workload = dict(operator=operator, precision="word64" if operator == "ntt" else "fp32",
                            logN=12, batch=batch, placement="out-of-place", direction="forward",
                            normalization="none")
            mapping = dict(schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
                           backend="shared-iterative", compute_unit="radix2", stage_partition=[6, 6],
                           stage_overlap=False, batch_tile_count=1)
            if operator == "ntt":
                workload.update(modulus=task["ntt_modulus"], input_order="natural", output_order="natural")
                mapping.update(threads_per_block=32, modular_multiply="shoup")
                axis, layouts = "dataflow_layout", ("linear", "hermes-xor")
            else:
                mapping.update(tile_threads=32, fft_core="scalar", local_exchange="shared")
                axis, layouts = "shared_layout", ("linear", "writer-aligned")
            cases.append(dict(id=f"{operator}-n12-b{batch}-layout", experiment="E1-fixed-core-layout",
                              workload=workload,
                              variants=[dict(id=name, mapping=dict(mapping, **{axis: layout}))
                                        for name, layout in zip(("linear", "writer-aligned"), layouts)],
                              comparisons=[dict(baseline="linear", treatment="writer-aligned", changed_axes=[axis])]))
    value = dict(schema=controlled.SCHEMA, protocol=dict(config["protocol"]),
                 scope=task["claim"], cases=cases)
    controlled._validate_study(value)
    return value


def audit_journal(journal, study):
    """Check resolved controls without ignoring derived core/group identities."""
    rows = []
    expected_cases = {case["id"]: case for case in study["cases"]}
    observed_cases = {case["id"]: case for case in journal.get("cases", [])}
    for case_id, expected in expected_cases.items():
        record = dict(case=case_id, status="rejected", reasons=[])
        rows.append(record)
        case = observed_cases.get(case_id, {})
        if case.get("workload") != expected["workload"]:
            record["reasons"].append("workload differs from the frozen semantic contract")
        variants = {variant["id"]: variant for variant in case.get("variants", [])}
        resolved = {}
        for requested in expected["variants"]:
            variant = variants.get(requested["id"], {})
            preflight = variant.get("preflight", {})
            mapping = preflight.get("resolved_mapping", {})
            resolved[requested["id"]] = mapping
            if preflight.get("status") != "accepted" or not mapping:
                record["reasons"].append(requested["id"] + ": missing accepted full-batch preflight")
            sample = preflight.get("sample", {})
            if str(sample.get("verified_batches")) != str(expected["workload"]["batch"]):
                record["reasons"].append(requested["id"] + ": incomplete correctness count")
            fingerprint = preflight.get("runtime_fingerprint")
            if not fingerprint or not controlled._sample_correct(sample):
                record["reasons"].append(requested["id"] + ": missing runtime identity or correctness verdict")
            for key, value in requested["mapping"].items():
                if key in {"kind", "schema_version"}:
                    continue
                if mapping.get(key) != value:
                    record["reasons"].append(requested["id"] + ": resolved control mismatch " + key)
            trials = variant.get("trials", [])
            if len(trials) != study["protocol"]["trials"] or not all(row.get("performance_accepted") for row in trials):
                record["reasons"].append(requested["id"] + ": missing accepted timing trials")
            if {row.get("trial") for row in trials} != set(range(study["protocol"]["trials"])):
                record["reasons"].append(requested["id"] + ": trial identity changed")
            for trial in trials:
                try:
                    latency = float(trial["kernel_ms"])
                    finite = math.isfinite(latency) and latency > 0
                except (KeyError, TypeError, ValueError):
                    finite = False
                if (not finite or trial.get("resolved_mapping") != mapping or
                    trial.get("sample", {}).get("runtime_fingerprint") != fingerprint):
                    record["reasons"].append(requested["id"] + ": trial mapping, fingerprint, or latency changed")
        baseline, treatment = resolved.get("linear", {}), resolved.get("writer-aligned", {})
        # Resource values may follow layout; core and execution-group mappings must match.
        excluded = {"runtime_fingerprint", "workspace_bytes", "selection"}
        difference = sorted(key for key in set(baseline) | set(treatment)
                            if key not in excluded and baseline.get(key) != treatment.get(key))
        expected_axes = expected["comparisons"][0]["changed_axes"]
        record.update(actual_changed_axes=difference, declared_changed_axes=expected_axes,
                      baseline_resolved_mapping=baseline, treatment_resolved_mapping=treatment)
        if difference != expected_axes:
            record["reasons"].append("actual mapping difference is not exactly the declared layout axis")
        if not record["reasons"]:
            record["status"] = "accepted"
    if set(observed_cases) != set(expected_cases):
        rows.append(dict(case="case-inventory", status="rejected", reasons=["journal case inventory changed"]))
    return dict(schema="cubutterfly-principle-gap-audit-v1", status="accepted" if all(
        row["status"] == "accepted" for row in rows) else "rejected", cases=rows,
        limitations=["Acceptance also requires the exclusive guard release and clean telemetry.",
                     "Kernel launch-sequence timings are not full C API end-to-end timings."])


def prepare(config_path=CONFIG, output_dir=OUTPUT):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    config = json.loads(config_path.read_text())
    study = layout_study(config)
    target = config["target"]
    library = ROOT / target["library"]
    calibration_path = ROOT / target["calibration"]
    calibration = json.loads(calibration_path.read_text())
    measured = calibration["measurement_identity"]
    if sha(library) != target["library_sha256"] or measured["library_sha256"] != target["library_sha256"]:
        raise ValueError("frozen runtime and calibration library identities disagree")
    hardware_path = Path(calibration["hardware_rates_source"])
    hardware = json.loads(hardware_path.read_text())
    if hardware["provenance"]["cuda_visible_devices"] != target["gpu_uuid"]:
        raise ValueError("calibration does not belong to the target GPU UUID")
    build = ROOT / target["build_dir"]
    binaries = {}
    for name in ("cubutterfly_bench", "cuntt_bench"):
        actual = sha(build / name)
        if actual != measured["binary_sha256"][name]:
            raise ValueError("calibrated benchmark binary identity changed: " + name)
        binaries[name] = actual
    study_path = output_dir / "study.json"
    save(study_path, study)
    save(output_dir / "prepared.json", controlled.prepare_document(study, study_path, build))
    runtime = measured["jit_runtime"]
    jit = runtime["config"]
    primitive_sources = {}
    for relative in ("src/operator_traits.cuh", "src/ntt_traits.cuh", "src/shared_iterative.cuh"):
        source = Path(jit["root"]) / relative
        primitive_sources[relative] = dict(path=str(source), sha256=sha(source))
    save(output_dir / "primitive_sources.json", dict(
        schema="cubutterfly-principle-gap-primitive-sources-v1",
        source_root=jit["root"], sources=primitive_sources,
        note="Frozen compile snapshot; current working-tree sources are not substituted for this runtime identity."))
    broker = ROOT / "results/plan_batch_trace_20261010/shared_compile"
    environment = dict(CUDA_VISIBLE_DEVICES=target["gpu_uuid"], CUBUTTERFLY_COMPILE_MODE="research",
                       LD_LIBRARY_PATH=str(library.parent) + runtime["ld_library_path"],
                       CUBUTTERFLY_TEMPLATE_ROOT=jit["root"], CUBUTTERFLY_JIT_CACHE=jit["cache"],
                       CUBUTTERFLY_MATHDX_ROOT=jit["mathdx"], CUBUTTERFLY_NVCC=jit["nvcc"],
                       CUBUTTERFLY_CALIBRATION_BROKER=str(broker), CUBUTTERFLY_PREPARE_BROKER=str(broker),
                       CUBUTTERFLY_JIT_PYTHON=str(ROOT / "results/async_fast_dispatch_20261010_v0934_3/bin/semantic_compile_client"),
                       CUBUTTERFLY_COMPILE_SOCKET=target["compile_socket"])
    child = [target["python"], str(Path(__file__).resolve()), "--guard-child", "--output-dir", str(output_dir)]
    guarded = [target["python"], str(ROOT / "scripts/run_isolated_gpu.py"), "--gpu-uuid", target["gpu_uuid"],
               "--report-dir", str(output_dir / "guard"), "--until", target["deadline"], "--", "env"]
    guarded += [f"{key}={value}" for key, value in sorted(environment.items())] + child
    manifest = dict(schema="cubutterfly-principle-gap-prepared-v1", status="prepared-not-executed",
                    gpu_queried=False, executable_claim=False, config_path=str(config_path), config_sha256=sha(config_path),
                    launcher_sha256=sha(__file__), telemetry_sha256=sha(ROOT / "experiments/tools/run_with_gpu_telemetry.py"),
                    gpu_uuid=target["gpu_uuid"], library_sha256=sha(library), binaries_sha256=binaries,
                    calibration_sha256=sha(calibration_path), hardware_profile_sha256=sha(hardware_path),
                    jit_identity=runtime, task_statuses={task["id"]: task["status"] for task in config["tasks"]},
                    study_path=str(study_path), build_dir=str(build), python=target["python"],
                    library_path=str(library), study_sha256=sha(study_path), cells=8, timing_processes=24, environment=environment,
                    launch_argv=guarded, launch_command=shlex.join(guarded),
                    audit_command=shlex.join([target["python"], str(Path(__file__).resolve()), "--audit-journal",
                                             str(output_dir / "guard/release/timing.json"), "--output-dir", str(output_dir)]),
                    acceptance="Resolved mapping audit + accepted full batch + all 3 trials + guard release + clean telemetry + preparation event gate")
    save(output_dir / "launch.json", manifest)
    return manifest


def guard_context(output_dir, environment=None):
    environment = os.environ if environment is None else environment
    directory = environment.get("XD_REPORT_DIR")
    if not directory:
        raise ValueError("--guard-child requires XD_REPORT_DIR from the isolation guard")
    attempt = Path(directory).resolve()
    if attempt.parent != Path(output_dir).resolve() / "guard/attempts" or not attempt.is_dir():
        raise ValueError("guard attempt directory is outside this immutable launch specification")
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise ValueError("guard child must own the session and process group created by the guard")
    parent_command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    if os.fsencode(str(ROOT / "scripts/run_isolated_gpu.py")) not in parent_command:
        raise ValueError("guard child parent is not the expected isolation guard")
    return attempt


def child_commands(spec, attempt):
    attempt = Path(attempt)
    environment = dict(spec["environment"], CUBUTTERFLY_PREPARE_EVENTS=str(attempt / "preparation_events.jsonl"))
    timed = [spec["python"], str(ROOT / "scripts/run_controlled_experiment.py"), "--mode", "run",
             "--study", spec["study_path"], "--build-dir", spec["build_dir"], "--gpu-uuid", spec["gpu_uuid"],
             "--output", str(attempt / "timing.json")]
    arguments = ["run_with_gpu_telemetry.py", "--gpu-uuid", spec["gpu_uuid"],
                 "--stop-on-thermal-throttle", "--output-prefix", str(attempt / "telemetry"), "--"] + timed
    return environment, arguments


def interrupted(signum, frame):
    raise KeyboardInterrupt("isolation guard interrupted its owned child")


def guard_child(output_dir):
    output_dir = Path(output_dir).resolve()
    attempt = guard_context(output_dir)
    spec = json.loads((output_dir / "launch.json").read_text())
    telemetry_path = ROOT / "experiments/tools/run_with_gpu_telemetry.py"
    for path, expected in ((Path(spec["config_path"]), spec["config_sha256"]),
                           (Path(spec["study_path"]), spec["study_sha256"]),
                           (Path(__file__), spec["launcher_sha256"]),
                           (telemetry_path, spec["telemetry_sha256"]),
                           (Path(spec["library_path"]), spec["library_sha256"])):
        if sha(path) != expected:
            raise ValueError("immutable launch identity changed: " + str(path))
    for name, expected in spec["binaries_sha256"].items():
        if sha(Path(spec["build_dir"]) / name) != expected:
            raise ValueError("benchmark identity changed: " + name)
    environment, arguments = child_commands(spec, attempt)
    for name in ("timing.json", "telemetry.telemetry.json", "preparation_events.jsonl"):
        if (attempt / name).exists():
            raise ValueError("guard child refuses to reuse an existing attempt artifact")
    os.environ.update(environment)
    ownership = dict(schema="cubutterfly-guard-child-ownership-v1", pid=os.getpid(), parent_pid=os.getppid(),
                     process_group=os.getpgrp(), session=os.getsid(0), attempt=str(attempt),
                     telemetry_child_policy="owned child session; existing telemetry BaseException cleanup terminates its child group",
                     telemetry_sha256=spec["telemetry_sha256"])
    controlled._atomic_write(attempt / "owned_processes.json", ownership)
    location = importlib.util.spec_from_file_location("principle_gap_telemetry", telemetry_path)
    telemetry = importlib.util.module_from_spec(location)
    location.loader.exec_module(telemetry)
    previous_argv = sys.argv
    previous_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    sys.argv = arguments
    for number in previous_handlers:
        signal.signal(number, interrupted)
    try:
        return telemetry.main()
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        sys.argv = previous_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--audit-journal", type=Path)
    parser.add_argument("--guard-child", action="store_true")
    args = parser.parse_args(argv)
    if args.guard_child:
        return guard_child(args.output_dir)
    if args.audit_journal:
        report = audit_journal(json.loads(args.audit_journal.read_text()),
                               layout_study(json.loads(args.config.read_text())))
        report["journal_sha256"] = sha(args.audit_journal)
        controlled._atomic_write(args.output_dir / "mapping_audit.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "accepted" else 1
    manifest = prepare(args.config, args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
