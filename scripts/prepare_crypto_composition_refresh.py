#!/usr/bin/env python3
"""Prepare the frozen crypto composition cohort without CUDA execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_crypto_composition as composition

ROOT = Path(__file__).resolve().parents[1]


def calibration_workloads(rows: list[dict], protocol: dict) -> dict:
    contracts = {}
    compositions = []
    for row in rows:
        channels = []
        for index, modulus in enumerate(row["moduli"]):
            identifier = composition.cyclic_cell_key(row, modulus)
            semantics = composition._cyclic_workload(row, modulus)
            if identifier in contracts and contracts[identifier]["semantics"] != semantics:
                raise ValueError("cyclic calibration contract identity collision")
            contracts[identifier] = {"id": identifier, "semantics": semantics,
                                     "mode": "cyclic", "word_bits": row["word_bits"]}
            channels.append({"channel_index": index, "cyclic_contract_id": identifier,
                             "cyclic_semantics": semantics, "modulus": int(modulus)})
        compositions.append({"id": row["id"], "semantics": composition.composition_semantics(row),
                             "channels": channels})
    if len(contracts) != 90 or len(compositions) != 138:
        raise ValueError("frozen calibration population must be 90 cyclic contracts for 138 compositions")
    return {
        "schema": "cubutterfly-install-search-v1", "scope": "90 exact cyclic channel contracts for the frozen 138 crypto compositions",
        "count": 90, "composition_count": 138,
        "workloads": [record["semantics"] for record in contracts.values()],
        "contracts": list(contracts.values()), "composition_channels": compositions,
        "required_confirmation_protocol": protocol,
        "execution": {
            "status": "blocked-interface-gap", "next_gpu_argv": None,
            "dependencies": ["exclusive-target-gpu", "current-build-stage-profile-and-service-calibration",
                             "Graph-protocol-search-confirmation-and-acceptance-export-interface"],
            "inspected_interfaces": [
                {"path": "scripts/run_research_acceptance.py", "entry": "arguments/prepare/search",
                 "semantic_admission": "install-search-v1 supports these word32/word64 cyclic NTT workload rows through calibration_space.command_for",
                 "gap": "only trials/warmup/repeat are accepted and stored; no timing-mode/graph-replays/calls-per-graph; search invokes event-based cuntt_bench"},
                {"path": "scripts/calibrate_local_hardware.py", "entry": "run_operator_calibration",
                 "gap": "operator-trials/operator-warmup/operator-repeat have no Graph replay/calls contract"},
                {"path": "apps/cuntt_bench.cpp", "entry": "CLI and benchmark timing",
                 "gap": "no CUDA Graph capture/replay CLI for acceptance candidate confirmation"},
                {"path": "scripts/run_current_recalibration.py", "entry": "phase command construction",
                 "gap": "hardcoded 3 trials/20 warmups/50 event repeats; cannot request Graph 10/7/20"},
                {"path": "scripts/run_semantic_campaign.py", "entry": "public descriptor MEASURE and Graph timing",
                 "gap": "Graph 10/7/20 exists, but consumes semantic descriptor manifests and exports mapping/selection rows, not exact cyclic acceptance cells with selected.samples.mapping_json required by composition mapping_sources"}],
            "required_interface_work": "Reuse the existing search only after Graph candidate confirmation and compatible exact-contract acceptance export are available; then measure these same 90 contracts and explicitly bind their new acceptance.json as --mapping-sources. No historical mappings or new search matrix are substituted.",
        },
    }


def validate_results(preparation: dict, document: dict, rows: list[dict]) -> None:
    if document.get("status") != "complete" or document.get("failures"):
        raise ValueError("composition result is incomplete or contains failures")
    if document.get("identity") != composition._identity(preparation["manifest"]):
        raise ValueError("composition result identity differs from CPU preparation")
    cells = document.get("cells", [])
    if len(cells) != len(rows) or {cell.get("id") for cell in cells} != {row["id"] for row in rows}:
        raise ValueError("composition result population differs")
    protocol = preparation["protocol"]
    for row in rows:
        cell = next(cell for cell in cells if cell["id"] == row["id"])
        if cell.get("status") != "complete" or cell.get("workload") != row:
            raise ValueError("composition cell status or contract differs")
        measurements = cell.get("measurements", [])
        names = preparation.get("expected_portfolios", ["public-default"])
        expected = {(pattern, trial, name) for pattern in protocol["patterns"]
                    for trial in range(protocol["trials"]) for name in names}
        actual = {(sample.get("pattern"), sample.get("trial"), sample.get("portfolio")) for sample in measurements}
        if len(measurements) != len(expected) or actual != expected:
            raise ValueError("composition samples missing, duplicated or unexpected")
        for sample in measurements:
            mappings = preparation.get("expected_mappings", {}).get(row["id"], {})
            composition.validate_sample(sample["sample"], row, preparation["manifest"]["gpu_uuid"],
                                        sample["pattern"], sample["trial"], protocol, mappings.get(sample["portfolio"]))


def rebind_static_inputs(output: Path, *, reason: str, source_paths: list[Path]) -> dict:
    """Refresh exactly the named Python sources; retain binary/data guards."""
    if not reason.strip():
        raise ValueError("static-input rebind reason must be non-empty")
    output = output.expanduser().resolve()
    preparation_path = output / "preparation.json"
    preparation = composition.read_json(preparation_path)
    records = preparation.get("static_inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("preparation has no static input records")
    changes = []
    updated = []
    allowed = {str(Path(path).resolve()) for path in source_paths}
    for name in allowed:
        path = Path(name)
        if not path.is_relative_to(ROOT / "scripts") or path.suffix != ".py":
            raise ValueError("static-input rebind requires named Python source paths under scripts")
    for record in records:
        path = Path(record["path"]).resolve()
        if str(path) == "/sdb" or str(path).startswith("/sdb/"):
            raise ValueError("static inputs must not resolve under /sdb")
        if not path.is_file():
            raise FileNotFoundError(path)
        current = composition.sha256(path)
        previous = record.get("sha256")
        if current != previous:
            if str(path) not in allowed:
                raise ValueError("static input changed outside named source paths: " + str(path))
            changes.append({"path": str(path), "old": previous, "new": current})
        item = dict(record)
        item["sha256"] = current
        updated.append(item)
    if {item["path"] for item in changes} != allowed:
        raise ValueError("named source paths must exactly match changed static inputs")
    if not changes:
        return preparation
    document = dict(preparation)
    document["static_inputs"] = updated
    provenance = dict(document.get("provenance", {}))
    history = list(provenance.get("static_input_rebindings", []))
    history.append({"reason": reason, "changes": changes})
    provenance["static_input_rebindings"] = history
    document["provenance"] = provenance
    composition.write_json(preparation_path, document)
    return document


def rebind_job_manifest(output: Path, *, cohort: str = "public-default", reason: str,
                        build_files: list[str]) -> dict:
    """Adopt named build inventory changes before composition evidence exists.

    The consumer binary and its dynamic dependencies must be unchanged. A
    rebuilt, unused comparison bridge can change the broader build inventory;
    explicitly naming that delta permits a prepared experiment to proceed.
    """
    if not reason.strip():
        raise ValueError("job-manifest rebind reason must be non-empty")
    output = output.expanduser().resolve()
    preparation_path = output / "preparation.json"
    preparation = composition.read_json(preparation_path)
    for directory in (output / "graph", output / "searched_graph"):
        if directory.exists() and any(directory.rglob("*")):
            raise ValueError("job-manifest rebind requires no graph/searched result evidence")
    jobs = preparation.get("jobs", [])
    job_id = "composed-138-public-default-graph" if cohort == "public-default" else "composed-138-searched-graph"
    job = next((item for item in jobs if item.get("id") == job_id), None)
    if not isinstance(job, dict):
        raise ValueError("prepared job is missing: " + job_id)
    argv = list(job.get("argv", []))
    if len(argv) < 3 or Path(argv[1]).name != "run_crypto_composition.py":
        raise ValueError("prepared job does not use the composition runner")
    command_args = composition.parser().parse_args(argv[2:])
    command_args.patterns = tuple(command_args.patterns.split(","))
    document, rows = composition._workloads_document(command_args.workloads)
    current = composition._manifest(command_args, document, rows)
    old = preparation.get("manifest")
    if not isinstance(old, dict):
        raise ValueError("preparation manifest is missing")
    old_without_binary = dict(old)
    current_without_binary = dict(current)
    old_binary = old_without_binary.pop("binary", None)
    current_binary = current_without_binary.pop("binary", None)
    if old_without_binary != current_without_binary:
        raise ValueError("job-manifest rebind would change non-binary composition identity")
    if not isinstance(old_binary, dict) or not isinstance(current_binary, dict):
        raise ValueError("consumer binary identity is missing")
    old_consumer, new_consumer = dict(old_binary), dict(current_binary)
    old_build, new_build = old_consumer.pop("build", {}), new_consumer.pop("build", {})
    if old_consumer != new_consumer:
        raise ValueError("consumer binary or dynamic dependency identity changed")
    if {k: v for k, v in old_build.items() if k not in {"files", "sha256"}} != {
            k: v for k, v in new_build.items() if k not in {"files", "sha256"}}:
        raise ValueError("build identity outside inventory changed")
    old_files, new_files = old_build.get("files", []), new_build.get("files", [])
    if (old_build.get("sha256") != composition.json_sha256(old_files) or
            new_build.get("sha256") != composition.json_sha256(new_files)):
        raise ValueError("build inventory checksum differs")
    old_map = {row["path"]: row for row in old_files}
    new_map = {row["path"]: row for row in new_files}
    if len(old_map) != len(old_files) or len(new_map) != len(new_files) or old_map.keys() != new_map.keys():
        raise ValueError("build inventory paths changed")
    changes = [{"path": path, "old": old_map[path], "new": new_map[path]}
               for path in old_map if old_map[path] != new_map[path]]
    allowed = set(build_files)
    if not allowed or allowed != {item["path"] for item in changes}:
        raise ValueError("named build files must exactly match build inventory changes")
    archive_root = output / "recovery_manifest_rebinds"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / "preparation.before.json"
    if not archive.exists():
        shutil.copy2(preparation_path, archive)
    document = dict(preparation)
    document["manifest"] = current
    provenance = dict(document.get("provenance", {}))
    history = list(provenance.get("job_manifest_rebindings", []))
    history.append({"cohort": cohort, "job_id": job_id, "reason": reason,
                    "archive": str(archive), "old_binary": old_binary,
                    "new_binary": current_binary, "build_file_changes": changes})
    provenance["job_manifest_rebindings"] = history
    document["provenance"] = provenance
    composition.write_json(preparation_path, document)
    return document


def run_prepared(output: Path, gpu_uuid: str, cohort: str = "public-default",
                 cyclic_report: Path | None = None) -> None:
    if cyclic_report is not None and cohort != "searched":
        raise ValueError("a cyclic source report requires the searched cohort")
    preparation = composition.read_json(output / "preparation.json")
    if preparation["manifest"]["gpu_uuid"] != gpu_uuid:
        raise ValueError("prepared GPU UUID differs")
    for record in preparation["static_inputs"]:
        if composition.sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("prepared static input identity changed: " + record["path"])
    job = (preparation["jobs"][0] if cohort == "public-default" else
           next(job for job in preparation["jobs"] if job["id"] == "composed-138-searched-graph"))
    if cohort == "searched":
        import crypto_semantic_mapping_refresh as adapter
        adapter.adapt(output, cyclic_report or output / "cyclic_mapping/measurement/results.json",
                      Path(preparation["library_path"]), gpu_uuid)
    # Recompute the same runtime/build identity before allowing any GPU work.
    command_args = composition.parser().parse_args(job["argv"][2:])
    command_args.patterns = tuple(command_args.patterns.split(","))
    document, rows = composition._workloads_document(command_args.workloads)
    current_manifest = composition._manifest(command_args, document, rows)
    comparable = {**current_manifest, "mapping_sources": []} if cohort == "searched" else current_manifest
    composition._validate_manifest(comparable, preparation["manifest"])
    preparation["manifest"] = current_manifest
    if cohort == "searched":
        preparation["expected_portfolios"] = ["public-default", "searched-cyclic-cores"]
        preparation["expected_mappings"] = {}
        mapping_context = composition.prepare_mapping_context(command_args.build_dir, command_args.mapping_sources, gpu_uuid)
        for row in rows:
            portfolios = composition.mapping_portfolios(row, gpu_uuid, preparation["protocol"],
                                                        command_args.build_dir, command_args.mapping_sources, context=mapping_context)
            if len(portfolios) != 2 or any(mapping is None for mapping in portfolios[1]["mappings"]):
                raise ValueError("searched composition requires every cyclic channel mapping")
            preparation["expected_mappings"][row["id"]] = {item["name"]: item["mappings"] for item in portfolios}
        composition.validate_mapping_context(mapping_context, full_digest=True)
    manifest_path = command_args.output_dir / "manifest.json"
    if manifest_path.exists():
        composition._validate_manifest(preparation["manifest"], composition.read_json(manifest_path))
    argv = job["resume_argv"] if manifest_path.exists() else job["argv"]
    completed = subprocess.run(argv, env={**os.environ, **job["environment"]}, check=False)
    if completed.returncode:
        raise RuntimeError(f"composition prepared job failed ({completed.returncode})")
    if cohort == "searched":
        composition.validate_mapping_context(mapping_context, full_digest=True)
    validate_results(preparation, composition.read_json(Path(job["result"]["path"])), rows)


def prepare(workloads: Path, binary: Path, build_dir: Path, output: Path,
            gpu_uuid: str, cpu_preparation_pid: int | None = None) -> dict:
    document, rows = composition._workloads_document(workloads)
    if len(rows) != 138:
        raise ValueError("frozen composition population must contain exactly 138 contracts")
    for path in (workloads, binary, build_dir, output):
        if str(path.resolve()).startswith("/sdb/") or path.resolve() == Path("/sdb"):
            raise ValueError("refresh assets must not resolve under /sdb")
    protocol = dict(trials=1, warmup=10, repeat=20, timing_mode="graph",
                    graph_replays=7, calls_per_graph=20,
                    patterns=list(composition.DEFAULT_PATTERNS), seed=composition.DEFAULT_SEED)
    args = argparse.Namespace(binary=binary, build_dir=build_dir, workloads=workloads,
                              output_dir=output, gpu_uuid=gpu_uuid, compile_mode="research",
                              template_root=ROOT, mapping_sources=[], **protocol)
    manifest = composition._manifest(args, document, rows)
    calibration_path = output / "calibration_workloads.json"
    calibration = calibration_workloads(rows, protocol)
    import crypto_semantic_mapping_refresh as adapter
    semantic_preparation = adapter.prepare(output, build_dir / "libcuntt.so", calibration)
    semantic_dir = output / "cyclic_mapping"
    python = str(Path(sys.executable).resolve())
    adapter_path = ROOT / "scripts/crypto_semantic_mapping_refresh.py"
    calibration_manifest = output.parent / "calibration/finalized/calibration_manifest.json"
    broker = output.parent / "compile"
    semantic_argv = [python, str(adapter_path), "--run", "--output-dir", str(output.resolve()),
                     "--library", str((build_dir / "libcuntt.so").resolve()), "--gpu-uuid", gpu_uuid,
                     "--compile-broker", str(broker.resolve()), "--calibration-manifest", str(calibration_manifest.resolve())]
    calibration["execution"] = dict(status="ready-entrypoint-cpu-and-gpu-pending", next_gpu_argv=semantic_argv,
        dependencies=["all-90-cpu-preparations-ready", "current-device-calibration", "exclusive-target-gpu", "compile-producer"],
        semantic_preparation=semantic_preparation,
        selection_protocol="existing semantic public MEASURE, not evolutionary acceptance search; source identity and raw Graph timing preserved by adapter")
    composition.write_json(calibration_path, calibration)
    input_paths = [workloads.resolve(), binary.resolve(), (build_dir / "libcuntt.so").resolve(),
                   ROOT / "scripts/run_crypto_composition.py", Path(__file__).resolve(),
                   ROOT / "benchmarks/crypto_ntt/crypto_ntt_bench.cu",
                   ROOT / "benchmarks/crypto_ntt/CMakeLists.txt", calibration_path.resolve()]
    input_paths += [adapter_path, semantic_dir.resolve() / "config.json", semantic_dir.resolve() / "manifest.json",
                    semantic_dir.resolve() / "cpu_budget.json", semantic_dir.resolve() / "prepared/config.json"]
    input_paths += [ROOT / relative for relative in (
        "scripts/run_research_acceptance.py", "scripts/calibration_space.py",
        "scripts/calibrate_local_hardware.py", "apps/cuntt_bench.cpp",
        "scripts/run_current_recalibration.py", "scripts/run_semantic_campaign.py")]
    input_paths += [ROOT / "scripts" / name for name in (
        "semantic_campaign_manifest.py", "semantic_preparation_queue.py", "semantic_cpu_pipeline.py",
        "semantic_gpu_pipeline.py", "run_async_semantic_campaign.py", "semantic_ntt_reference.py")]
    argv = [str(Path(sys.executable).resolve()), str(ROOT / "scripts/run_crypto_composition.py"),
            "--binary", str(binary.resolve()), "--build-dir", str(build_dir.resolve()),
            "--workloads", str(workloads.resolve()), "--output-dir", str((output / "graph").resolve()),
            "--gpu-uuid", gpu_uuid, "--compile-mode", "research", "--timing-mode", "graph",
            "--trials", "1", "--warmup", "10", "--repeat", "20", "--graph-replays", "7",
            "--calls-per-graph", "20"]
    result = {
        "schema": "cubutterfly-composition-refresh-preparation-v1", "status": "cpu-ready-gpu-pending",
        "population": 138, "manifest": manifest, "protocol": protocol,
        "library_path": str((build_dir / "libcuntt.so").resolve()),
        "cyclic_mapping_preparation": semantic_preparation,
        "cpu_preparation_argv": [python, str(adapter_path), "--prepare-cpu", "--output-dir", str(output.resolve()), "--jobs", "4"],
        "cpu_preparation": {"contracts_and_metadata": "complete", "all_cpu_references_precomputed": False,
                            "reference_generation": "consumer generates full CPU input/reference online before each measurement; excluded from device timing",
                            "full_cpu_preparation_complete": False},
        "static_inputs": [{"path": str(path), "sha256": composition.sha256(path)} for path in input_paths],
        "contracts": [{"id": row["id"], "semantics": composition.composition_semantics(row),
                       "status": "ready", "cpu_reference": "generated by consumer before timing"} for row in rows],
        "jobs": [{"id": "composed-138-public-default-graph", "argv": argv,
                  "resume_argv": argv + ["--resume"],
                  "input_paths": [str(path) for path in input_paths] + [str((output / "preparation.json").resolve())],
                  "selection": "fixed public-default diagnostic; not current calibrated selection",
                  "dependencies": ["exclusive-target-gpu", "current-library-binary-identity"],
                  "environment": {"CUBUTTERFLY_COMPILE_MODE": "research", "CUDA_VISIBLE_DEVICES": gpu_uuid},
                  "result": {"path": str((output / "graph/results.json").resolve()),
                             "status": "complete", "expected_cells": 138,
                             "expected_samples_per_cell": 2, "require_sample_validation": True,
                             "validation_argv": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
                                                 "--validate-result", str((output / "graph/results.json").resolve()),
                                                 "--output-dir", str(output.resolve())]}}],
        "pending_dependencies": [{"id": "current-calibrated-exact-cyclic-mappings",
                                  "requires_gpu": True,
                                  "unique_cyclic_contracts": 90,
                                  "workloads_path": str(calibration_path.resolve()),
                                  "execution": calibration["execution"],
                                  "reason": "90 current exact cyclic selections and strict source adapter are queued; all 90 CPU preparations and GPU measurements must complete before searched composition"}],
        "claims": ["138 modular composition witnesses, including twists; not complete CKKS or bootstrap",
                   "public-default is fixed diagnostic; searched portfolio requires the explicit 90-point source receipt; no external baseline win"],
        "historical_timing_rows_adopted": 0,
    }
    result["jobs"].append(dict(id="crypto-cyclic-90-public-measure-graph", argv=semantic_argv,
        input_paths=[str(path) for path in input_paths] + [str((output / "preparation.json").resolve())],
        compile_producer=True, dependencies=calibration["execution"]["dependencies"],
        compile_priority_report=str((semantic_dir / "measurement/results.json").resolve()),
        worker_recycle_exit=75,
        environment={"CUBUTTERFLY_COMPILE_MODE": "research", "CUBUTTERFLY_SEARCH_BUDGET": "512"},
        compile_producer_argv=[python, str(ROOT / "scripts/semantic_compile_producer.py"), "--broker", str(broker.resolve()),
                               "--jobs", "4", "--watch", "--priority-report", str((semantic_dir / "measurement/results.json").resolve())],
        result=dict(path=str((semantic_dir / "mapping_source.json").resolve()), status="complete", expected_cells=90)))
    searched_argv = list(argv)
    searched_argv[searched_argv.index("--output-dir") + 1] = str((output / "searched_graph").resolve())
    searched_argv += ["--mapping-sources", str((semantic_dir / "mapping_source.json").resolve())]
    result["jobs"].append(dict(id="composed-138-searched-graph", argv=searched_argv,
        resume_argv=searched_argv + ["--resume"],
        input_paths=[str(path) for path in input_paths] + [str((output / "preparation.json").resolve())],
        depends_on=["crypto-cyclic-90-public-measure-graph"],
        dependencies=["verified-all-90-current-semantic-mappings", "exclusive-target-gpu"],
        environment={"CUBUTTERFLY_COMPILE_MODE": "research", "CUDA_VISIBLE_DEVICES": gpu_uuid},
        selection="public MEASURE cyclic cores replayed inside compositions; no joint RNS search",
        result=dict(path=str((output / "searched_graph/results.json").resolve()), status="complete", expected_cells=138,
                    expected_samples_per_cell=4)))
    previous_path = output / "preparation.json"
    prior_pid = (composition.read_json(previous_path).get("cpu_preparation_background", {}).get("pid")
                 if previous_path.exists() else None)
    pid = cpu_preparation_pid or prior_pid
    counts = {}
    for case in composition.read_json(semantic_dir / "manifest.json")["cases"]:
        status = (adapter.preparation.record_for(semantic_dir / "prepared", case["id"]) or {}).get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    result["cpu_preparation_background"] = dict(pid=pid, counts_at_receipt=counts,
        status_at_receipt="ready" if counts.get("ready") == 90 else "background-running-or-pending",
        log_path=str((semantic_dir / "cpu_preparation.log").resolve()),
        terminal_status_path=str((semantic_dir / "cpu_preparation_status.json").resolve()),
        argv=result["cpu_preparation_argv"], cuda_visible_devices="",
        note="snapshot only; mutable worker records are excluded from static identity; GPU launch requires all 90 current records ready")
    audit_path = semantic_dir / "historical_row_shape_audit.json"
    if audit_path.exists():
        result["historical_row_shape_audit"] = dict(path=str(audit_path.resolve()), sha256=composition.sha256(audit_path),
            scope="real historical NTT receipt shape audit only; no prior timings or mappings adopted")
    normalized = dict(schema="cubutterfly-refresh-followups-v1",
        input_paths=[str(path) for path in input_paths] + [str((output / "preparation.json").resolve())],
        jobs=[dict(id="crypto-cyclic-90-public-measure-graph", argv=semantic_argv[1:],
                   compile_producer=True, compile_priority_report=str((semantic_dir / "measurement/results.json").resolve()),
                   worker_recycle_exit=75),
              dict(id="composed-138-searched-graph", depends_on=["crypto-cyclic-90-public-measure-graph"],
                   argv=[str(Path(__file__).resolve()), "--run-prepared", "--cohort", "searched",
                         "--output-dir", str(output.resolve()), "--gpu-uuid", gpu_uuid])])
    composition.write_json(output / "normalized_followups.json", normalized)
    composition.write_json(output / "preparation.json", result)
    composition.write_json(output / "contracts.json", {"population": 138, "contracts": result["contracts"]})
    composition.write_json(output / "queue.json", {"jobs": result["jobs"], "pending_dependencies": result["pending_dependencies"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=ROOT / "config/research_campaign/composed.json")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--validate-result", type=Path)
    parser.add_argument("--run-prepared", action="store_true")
    parser.add_argument("--rebind-static-inputs", action="store_true",
                        help="explicitly refresh reviewed static input digests")
    parser.add_argument("--rebind-job-manifest", action="store_true",
                        help="adopt named build inventory changes before measurement")
    parser.add_argument("--rebind-source-path", action="append", type=Path, default=[])
    parser.add_argument("--rebind-build-file", action="append", default=[])
    parser.add_argument("--rebind-reason", default="",
                        help="audit reason required with --rebind-static-inputs")
    parser.add_argument("--cohort", choices=("public-default", "searched"), default="public-default")
    parser.add_argument("--cyclic-report", type=Path,
                        help="verified completed cyclic report, including an independent recovery directory")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-e13-cufftdx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/full_current_20261025/composed_138")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--cpu-preparation-pid", type=int)
    args = parser.parse_args()
    if args.rebind_static_inputs:
        rebind_static_inputs(args.output_dir, reason=args.rebind_reason, source_paths=args.rebind_source_path)
        print(json.dumps({"status": "rebound", "output": str(args.output_dir.resolve())}, sort_keys=True))
        return 0
    if args.rebind_job_manifest:
        rebind_job_manifest(args.output_dir, cohort=args.cohort, reason=args.rebind_reason,
                            build_files=args.rebind_build_file)
        print(json.dumps({"status": "rebound", "output": str(args.output_dir.resolve()),
                          "cohort": args.cohort}, sort_keys=True))
        return 0
    if args.run_prepared:
        if not args.gpu_uuid:
            parser.error("--run-prepared requires --gpu-uuid")
        run_prepared(args.output_dir, args.gpu_uuid, args.cohort, args.cyclic_report)
        return 0
    if args.validate_result:
        preparation = composition.read_json(args.output_dir / "preparation.json")
        _, rows = composition._workloads_document(args.workloads)
        validate_results(preparation, composition.read_json(args.validate_result), rows)
        return 0
    if not args.binary or not args.gpu_uuid:
        parser.error("preparation requires --binary and --gpu-uuid")
    prepare(args.workloads, args.binary, args.build_dir, args.output_dir, args.gpu_uuid, args.cpu_preparation_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
