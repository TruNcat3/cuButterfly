#!/usr/bin/env python3
"""Bind the 90 crypto cyclic contracts to existing semantic Graph machinery."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import run_crypto_composition as composition
import semantic_campaign_manifest as manifest_api
import semantic_preparation_queue as preparation
import run_semantic_campaign as semantic

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCHEMA = "cubutterfly-semantic-cyclic-mapping-source-v1"
GRAPH_PROTOCOL = "20 complete calls per CUDA graph; 7 replays; 10 warmup"
# Comparator binding changes do not alter the selected cuButterfly mapping.
# Keep their original identities and record the delta instead of invalidating
# all verified own-library rows when an external comparison is repaired.
COMPARATOR_ONLY_SOURCES = frozenset({"semantic_gpuntt_baseline.py"})


def mapping_source_updates(identity: dict, current: dict) -> dict:
    previous = identity.get("sources", {})
    if previous.keys() != current.keys():
        raise ValueError("cyclic runner source population differs")
    changes = {name: dict(original=previous[name], current=current[name])
               for name in previous if previous[name] != current[name]}
    if set(changes) - COMPARATOR_ONLY_SOURCES:
        raise ValueError("cyclic mapping execution source identity differs")
    return changes


def semantic_config(calibration: dict) -> dict:
    populations = []
    for contract in calibration["contracts"]:
        row = contract["semantics"]
        bits = contract["word_bits"]
        populations.append(dict(name=contract["id"], shapes=[[1 << row["logN"]]], batches=[row["batch"]],
            directions=[row["direction"]], normalizations=[row["normalization"]], placements=["out-of-place"],
            order_patterns=["natural"], layouts=["packed"],
            regimes=[dict(name="exact-cyclic", transform=dict(op=1, storage=8 if bits == 32 else 9,
                compute=3 if bits == 32 else 4, modulus=row["modulus"], word_bits=bits))]))
    config = dict(schema=manifest_api.CONFIG_SCHEMA, populations=populations)
    manifest_api.validate_config(config)
    return config


def budgets(calibration: dict) -> dict:
    rows = calibration["workloads"]
    points = [row["batch"] * (1 << row["logN"]) for row in rows]
    work = [count * row["logN"] // 2 for count, row in zip(points, rows)]
    return dict(reference_max_elements=max(points), ntt_max_operations=max(work),
                total_elements=sum(points), total_radix2_butterflies=sum(work),
                cpu_workers=4, estimated_peak_bytes_per_worker=max(points) * 256,
                estimate_basis="conservative 256 bytes/element for arbitrary-width integer oracle temporaries; four workers; not a measured RSS claim")


def verify_case(case: dict, contract: dict) -> None:
    s = case["descriptor"]["semantics"]
    row = contract["semantics"]
    bits = contract["word_bits"]
    expected = dict(operator=1, storage=8 if bits == 32 else 9, compute=3 if bits == 32 else 4,
        shape=[1 << row["logN"]], batch=row["batch"], inverse=row["direction"] == "inverse",
        normalize_inverse=row["normalization"] == "inverse", placement=0,
        input_orders=[0], output_orders=[0], input_strides=[1], output_strides=[1],
        input_batch_stride=1 << row["logN"], output_batch_stride=1 << row["logN"],
        modulus=row["modulus"], word_bits=bits, length_mode=0)
    for field, value in expected.items():
        if s.get(field) != value:
            raise ValueError(f"cyclic descriptor {field} differs: {s.get(field)!r} != {value!r}")
    if case["id"] != semantic.digest(s) or case["populations"] != [contract["id"]]:
        raise ValueError("cyclic semantic case identity/population differs")


def prepare(output: Path, library: Path, calibration: dict) -> dict:
    directory = output / "cyclic_mapping"
    directory.mkdir(parents=True, exist_ok=True)
    config_path, manifest_path = directory / "config.json", directory / "manifest.json"
    config = semantic_config(calibration)
    composition.write_json(config_path, config)
    manifest = manifest_api.generate_manifest(config, library, config_path=config_path)
    if manifest["invalid_requests"] or len(manifest["cases"]) != 90:
        raise ValueError("90 cyclic contracts did not produce 90 valid public descriptors")
    contracts = {row["id"]: row for row in calibration["contracts"]}
    for case in manifest["cases"]:
        verify_case(case, contracts[case["populations"][0]])
        if case["capabilities"]["status"] != "eligible":
            raise ValueError("cyclic contract lacks executable public lowering")
    if manifest_path.exists() and composition.read_json(manifest_path) != manifest:
        raise ValueError("cyclic descriptor manifest identity changed; use a separate preparation")
    composition.write_json(manifest_path, manifest)
    limits = budgets(calibration)
    preparation.initialize(directory / "prepared", manifest, limits["reference_max_elements"], limits["ntt_max_operations"])
    composition.write_json(directory / "cpu_budget.json", limits)
    ready = sum((preparation.record_for(directory / "prepared", case["id"]) or {}).get("status") == "ready"
                for case in manifest["cases"])
    return dict(directory=str(directory.resolve()), manifest=str(manifest_path.resolve()),
                config=str(config_path.resolve()), budget=str((directory / "cpu_budget.json").resolve()),
                preparation=str((directory / "prepared").resolve()), count=90,
                cpu_reference_status="ready" if ready == 90 else "pending; readiness is checked from all 90 preparation records at GPU launch",
                ready_count=ready,
                selection="existing public MEASURE; original internal selection protocol retained; not acceptance evolutionary search",
                graph_protocol=GRAPH_PROTOCOL, warmup_unit="10 eager complete calls in existing semantic runner",
                budget_limits=limits)


def require_ready(output: Path) -> tuple[dict, dict]:
    directory = output / "cyclic_mapping"
    manifest = composition.read_json(directory / "manifest.json")
    limits = composition.read_json(directory / "cpu_budget.json")
    config, _ = preparation._config(directory / "prepared")
    if config["manifest"] != manifest or any(config[key] != limits[key] for key in ("reference_max_elements", "ntt_max_operations")):
        raise ValueError("cyclic CPU preparation identity/bounds differ")
    pending = [case["id"] for case in manifest["cases"]
               if (preparation.record_for(directory / "prepared", case["id"]) or {}).get("status") != "ready"]
    if pending:
        raise ValueError(f"cyclic CPU preparation requires all 90 ready; {len(pending)} not ready")
    return manifest, limits


def verify_selected_row(case: dict, contract: dict, row: dict) -> None:
    if row.get("status") != "pass" or row.get("descriptor") != case["descriptor"]:
        raise ValueError("cyclic row contract/status differs")
    if row.get("plan_descriptor", {}).get("semantics") != case["descriptor"]["semantics"]:
        raise ValueError("selected cyclic plan semantics differ")
    if row.get("plan_descriptor", {}).get("selection", {}).get("algorithm_policy") != 1:
        raise ValueError("cyclic selection was not public MEASURE")
    count = contract["semantics"]["batch"] * (1 << contract["semantics"]["logN"])
    for field in ("correctness", "mapping_replay_correctness"):
        verdict = row.get(field, {})
        if verdict.get("exact") is not True or verdict.get("mismatches") != 0 or verdict.get("elements") != count:
            raise ValueError("cyclic exact integer correctness differs: " + field)
    mapping = row.get("mapping", {})
    if mapping.get("kind") != "ntt" or mapping.get("schema_version") != 1 or mapping.get("contract") != case["descriptor"]["semantics"]:
        raise ValueError("selected mapping is not an exact native NTT contract")
    timing = row.get("timing", {})
    samples = timing.get("samples_ms", [])
    if (timing.get("protocol") != GRAPH_PROTOCOL or len(samples) != 7 or
        any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples) or
        timing.get("median_ms") != sorted(samples)[3]):
        raise ValueError("cyclic Graph timing evidence differs")


def audit_historical_shape(report_path: Path, output: Path) -> dict:
    """Audit receipt structure only; never export mappings/timings from a prior build."""
    report = composition.read_json(report_path)
    audited = []
    for index in report["rows"]:
        envelope = composition.read_json(report_path.parent / index["path"])
        row = envelope["row"]
        s = row.get("descriptor", {}).get("semantics", {})
        if s.get("operator") != 1 or len(s.get("shape", [])) != 1 or row.get("status") != "pass":
            continue
        if semantic.digest(row) != envelope["sha256"] or envelope["sha256"] != index["sha256"]:
            raise ValueError("historical shape audit row checksum differs")
        log_n = s["shape"][0].bit_length() - 1
        contract = dict(id="shape-audit", word_bits=s["word_bits"], semantics=dict(logN=log_n,
            batch=s["batch"], direction="inverse" if s["inverse"] else "forward",
            normalization="inverse" if s["normalize_inverse"] else "none", modulus=s["modulus"]))
        case = dict(id=row["id"], descriptor=row["descriptor"], populations=["shape-audit"])
        verify_case(case, contract)
        verify_selected_row(case, contract, row)
        audited.append(dict(id=row["id"], row_sha256=envelope["sha256"], word_bits=s["word_bits"],
                            correctness_fields=sorted(row["correctness"]), mapping_kind=row["mapping"]["kind"]))
    if not audited:
        raise ValueError("historical shape audit found no passing rank-one NTT receipt")
    receipt = dict(status="shape-audit-passed", source_path=str(report_path.resolve()),
        source_sha256=composition.sha256(report_path), audited_rows=audited,
        historical_timing_rows_adopted=0, historical_mapping_rows_adopted=0,
        scope="CPU schema/field audit only; source build and hardware are historical, not current evidence")
    composition.write_json(output / "cyclic_mapping/historical_row_shape_audit.json", receipt)
    return receipt


def adapt(output: Path, report_path: Path, library: Path, gpu_uuid: str, publish: bool = True) -> dict:
    directory = output / "cyclic_mapping"
    manifest = composition.read_json(directory / "manifest.json")
    if not manifest_api.validate_manifest_hash(manifest):
        raise ValueError("cyclic manifest checksum differs")
    semantic.validate_fingerprints(manifest, library)
    calibration = composition.read_json(output / "calibration_workloads.json")
    limits = composition.read_json(directory / "cpu_budget.json")
    report = composition.read_json(report_path)
    identity = report.get("identity", {})
    if (not str(report.get("status", "")).startswith("complete") or report.get("remaining_case_ids") != [] or
        report.get("invalid_request_count") != 0 or identity.get("manifest_sha256") != semantic.digest(manifest) or
        identity.get("library_sha256") != composition.sha256(library) or identity.get("gpu_uuid") != gpu_uuid or
        identity.get("compile_mode") != "research" or identity.get("timing") is not True):
        raise ValueError("cyclic report population/build/GPU/timing identity differs or is incomplete")
    protocol = identity.get("protocol", {})
    import run_async_semantic_campaign as asynchronous
    if identity.get("runner_revision") != asynchronous.REVISION:
        raise ValueError("cyclic runner source identity differs")
    source_updates = mapping_source_updates(identity, asynchronous.source_identity())
    for field, value in dict(warmup_calls=10, graph_calls=20, replays=7,
                             reference_max_elements=limits["reference_max_elements"], ntt_max_operations=limits["ntt_max_operations"]).items():
        if protocol.get(field) != value:
            raise ValueError("cyclic source protocol differs: " + field)
    expected_ids = {case["id"] for case in manifest["cases"]}
    indices = report.get("rows", [])
    if len(indices) != 90 or {row.get("id") for row in indices} != expected_ids:
        raise ValueError("cyclic report must contain all 90 unique cases")
    semantic.validate_resume(report, identity, report_path.parent, expected_ids)
    contracts = {row["id"]: row for row in calibration["contracts"]}
    indices = {row["id"]: row for row in indices}
    cells = []
    for case in manifest["cases"]:
        contract = contracts[case["populations"][0]]
        verify_case(case, contract)
        envelope = composition.read_json(report_path.parent / indices[case["id"]]["path"])
        row = envelope["row"]
        verify_selected_row(case, contract, row)
        mapping = row.get("mapping", {})
        timing = row.get("timing", {})
        cells.append(dict(id=contract["id"], workload=contract["semantics"], status="complete",
            selected=dict(samples=[dict(correct=True, mapping_json=mapping, gpu_uuid=gpu_uuid)]),
            provenance=dict(semantic_case_id=case["id"], row_sha256=envelope["sha256"],
                            selection_scope=row.get("selection_scope"), selection_reason=row.get("selection_reason"),
                            source_timing=timing)))
    exported = dict(schema=EXPORT_SCHEMA, status="complete", cells=cells,
        identity=dict(gpu_uuid=gpu_uuid, library_sha256=composition.sha256(library), compile_mode="research"),
        source=dict(report_path=str(report_path.resolve()), report_sha256=composition.sha256(report_path),
                    manifest_path=str((directory / "manifest.json").resolve()), manifest_sha256=composition.sha256(directory / "manifest.json"),
                    original_measurement_identity=identity,
                    comparator_only_source_updates=source_updates,
                    reuse_scope="selected cuButterfly mappings only; comparator results are not imported",
                    search_protocol={"policy": "existing public MEASURE", "candidate_timing": "host-event-loop",
                                     "candidate_warmup": 3, "candidate_repeat": 20,
                                     "search_budget": identity.get("runtime_environment", {}).get("CUBUTTERFLY_SEARCH_BUDGET"),
                                     "scope": "existing NTT selector; retained actual selection_reason; no evolutionary acceptance search or timing synthesis"},
                    composition_mapping_use="fixed selected cyclic cores replayed in composed semantics; not a joint RNS search"))
    if publish:
        composition.write_json(directory / "mapping_source.json", exported)
    return exported


def validate_source(document: dict, library: Path, gpu_uuid: str) -> None:
    identity = document.get("identity", {})
    if (document.get("schema") != EXPORT_SCHEMA or document.get("status") != "complete" or
        identity != dict(gpu_uuid=gpu_uuid, library_sha256=composition.sha256(library), compile_mode="research")):
        raise ValueError("semantic mapping source build/GPU identity differs")
    cells = document.get("cells", [])
    if len(cells) != 90 or len({cell.get("id") for cell in cells}) != 90:
        raise ValueError("semantic mapping source requires all 90 unique cells")
    source = document["source"]
    for kind in ("report", "manifest"):
        if composition.sha256(Path(source[kind + "_path"])) != source[kind + "_sha256"]:
            raise ValueError("semantic mapping source changed: " + kind)
    expected = adapt(Path(source["manifest_path"]).parent.parent, Path(source["report_path"]), library, gpu_uuid, publish=False)
    if document != expected:
        raise ValueError("semantic mapping export differs from its verified source rows")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--compile-broker", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--gpuntt-library", type=Path)
    parser.add_argument(
        "--worker-max-rss-bytes", type=int,
        default=8 * 1024**3,
        help="RSS ceiling for the measurement/search process (default: 8 GiB)",
    )
    parser.add_argument("--prepare-cpu", action="store_true")
    parser.add_argument("--adapt", type=Path)
    parser.add_argument("--audit-historical-shape", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.worker_max_rss_bytes <= 0:
        parser.error("--worker-max-rss-bytes must be positive")
    directory = args.output_dir / "cyclic_mapping"
    if args.audit_historical_shape:
        audit_historical_shape(args.audit_historical_shape, args.output_dir)
        return 0
    if args.prepare_cpu:
        manifest = composition.read_json(directory / "manifest.json")
        limits = composition.read_json(directory / "cpu_budget.json")
        preparation.initialize(directory / "prepared", manifest, limits["reference_max_elements"], limits["ntt_max_operations"])
        result = preparation.prepare_manifest(manifest, directory / "prepared", args.jobs)
        composition.write_json(directory / "cpu_preparation_status.json", result)
        return 0 if result["status"] == "complete" and result["counts"]["ready"] == 90 else 2
    if args.adapt:
        adapt(args.output_dir, args.adapt, args.library, args.gpu_uuid)
        return 0
    if not args.run or not all((args.library, args.gpu_uuid, args.compile_broker, args.calibration_manifest)):
        parser.error("--run requires library, GPU UUID, compile broker and calibration manifest")
    manifest, limits = require_ready(args.output_dir)
    os.environ["CUBUTTERFLY_SEARCH_BUDGET"] = "512"
    # Only increase the existing oracle's finite CPU admission bounds. Search,
    # plan selection, Graph capture and correctness are owned by the existing runner.
    import run_async_semantic_campaign as asynchronous
    asynchronous.old.PROTOCOL.update({key: limits[key] for key in ("reference_max_elements", "ntt_max_operations")})
    argv = ["--manifest", str(directory / "manifest.json"), "--library", str(args.library),
            "--output", str(directory / "measurement"), "--preparation", str(directory / "prepared"),
            "--compile-broker", str(args.compile_broker), "--calibration-manifest", str(args.calibration_manifest),
            "--gpu-uuid", args.gpu_uuid, "--timing", "--cpu-workers", "4", "--in-flight", "1",
            "--worker-max-rss-bytes", str(args.worker_max_rss_bytes)]
    if args.gpuntt_library is not None:
        argv.extend(["--gpuntt-library", str(args.gpuntt_library)])
    report_path = directory / "measurement/results.json"
    if report_path.exists():
        argv.append("--resume")
    code = asynchronous.main(argv)
    if code:
        return code
    adapt(args.output_dir, report_path, args.library, args.gpu_uuid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
