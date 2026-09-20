#!/usr/bin/env python3
"""Hardware calibration and model-guided search of library-exported mappings.

All six mathematical operators use the linked library's candidate inventory.
Coverage records retain budget omissions, failures and confirmed finalists.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import io
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time
import os
import tempfile
import math
from typing import Any
from hardware_registry import promote, default_path, canonical_semantics

from calibration_space import stratified_candidates, candidate_id, command_for, runtime_candidates, predicted_sample, PROJECTION_VERSION
from calibration_seeds import seed_snapshot, candidates_for_workload, validate_mapping
from fit_local_cost_model import fit, predict, design_point_key, FEATURE_VERSION
import stage_cost_model
from schedule_cost_model import calibrate as calibrate_scheduling
from pipeline_schedule_model import calibrate as calibrate_pipeline_scheduling
from stage_service_calibration import calibrate as calibrate_stage_services
from local_selector_data import write_header as write_local_selector_header, semantic_key
from run_comprehensive_suite import require_exclusive_gpu


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT / "share/cuButterfly"
PROFILE_SCRIPT = SCRIPT_DIR / "initialize_hardware_profile.py"
CHECK_SCRIPT = SCRIPT_DIR / "check_hardware_profile.py"
COST_MODEL_SCRIPT = SCRIPT_DIR / "fit_local_cost_model.py"
MODEL_UPDATE_POLICY_VERSION = "correct-training-change-v1"
SEARCH_SELECTION_VERSION = "resolved-feasibility-and-distinct-finalists-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate cuButterfly for the current GPU during installation.")
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--trials", type=int, default=5, help="hardware capability probe trials")
    parser.add_argument("--operator-trials", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=100)
    parser.add_argument("--operator-repeat", type=int, default=100)
    parser.add_argument("--verify-batches", type=int, default=0,
                        help="check this many evenly spaced transforms within the actual timed batch; 0 checks all")
    parser.add_argument("--skip-operator-calibration", action="store_true")
    parser.add_argument("--search-only", action="store_true",
                        help="omit legacy smoke cells and calibrate only the configured search workloads")
    parser.add_argument("--search-workloads", type=pathlib.Path, default=ROOT / "config/install_search_workloads.json")
    parser.add_argument("--search-budget", type=int, default=64,
                        help="screened library mappings per workload; 0 exhausts the exported inventory (64 by default)")
    parser.add_argument("--search-finalists", type=int, default=3)
    parser.add_argument("--cost-model", choices=("staged", "legacy"), default="staged")
    parser.add_argument("--stage-calibration", choices=("full", "bounded", "skip"), default="full",
                        help="independent stage service calibration; full is required for complete migration")
    parser.add_argument("--stage-import-checkpoint", type=pathlib.Path, action="append", default=[],
                        help="reuse strictly compatible independent-stage measurements")
    parser.add_argument("--search-strategy", choices=("model", "evolutionary"), default="model")
    parser.add_argument("--mapping-seeds", type=pathlib.Path, action="append", default=[],
                        help="mapping-only seed file or registry; repeat to add sources alongside the local registry")
    parser.add_argument("--seed-budget", type=int, default=8,
                        help="historical mappings revalidated per workload before the exploration budget; 0 disables seeds")
    parser.add_argument("--search-seconds", type=float,
                        help="total search wall-time budget; 0 for unrestricted research")
    parser.add_argument("--compile-seconds", type=float,
                        help="additional specialization compilation budget, independent of search (0 = unrestricted)")
    parser.add_argument("--resume-search", action="store_true",
                        help="reuse matching search measurements only when the benchmark binary hash is unchanged")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embed-legacy-selector", action="store_true",
                        help="also generate the legacy build-time header; the dynamic registry needs no rebuild")
    args = parser.parse_args()
    if args.search_seconds is None:
        args.search_seconds = 0 if args.stage_calibration == "full" else 300
    if args.compile_seconds is None:
        args.compile_seconds = 0 if args.stage_calibration == "full" else 600
    return args


def build_python(build_dir: pathlib.Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        match = re.search(r"^Python3_EXECUTABLE:[^=]*=(.*)$", cache.read_text(), re.MULTILINE)
        if match and pathlib.Path(match.group(1)).is_file():
            return match.group(1)
    return sys.executable


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed


def write_checkpoint(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def measurement_identity(args):
    return dict(binary_sha256={name:hashlib.sha256((args.build_dir/name).read_bytes()).hexdigest()
                for name in ("cubutterfly_bench","cuntt_bench")},
                compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE","auto"),
                verify_batches=getattr(args,"verify_batches",0))


def rank_model(profile: dict[str, Any], output: pathlib.Path) -> int:
    capabilities = profile["capabilities"]
    records: list[dict[str, Any]] = []
    for log_n in (8, 12, 16, 20):
        for budget in (32, 64, 128, 256):
            for word_bytes in (4, 8):
                for us in range(1, log_n + 1):
                    if budget % us:
                        continue
                    ud = budget // us
                    ts = (log_n + us - 1) // us
                    td = (((1 << log_n) // 2) + ud - 1) // ud
                    cells = us * ud
                    rates = {
                        "compute": capabilities["equivalent_butterflies_per_second"] / cells,
                        "boundary": capabilities["global_feedback_bytes_per_second"] / (4 * ud * word_bytes),
                    }
                    if us > 1:
                        rates["interstage"] = capabilities["interstage_shared_bytes_per_second"] / (4 * ud * (us - 1) * word_bytes)
                        rates["synchronization"] = capabilities["cta_barriers_per_second"] / (us - 1)
                    bottleneck = min(rates, key=rates.get)
                    records.append({
                        "device": profile["device"],
                        "logN": log_n,
                        "word_bytes": word_bytes,
                        "spatial_budget_C": budget,
                        "stage_space_Us": us,
                        "data_space_Ud": ud,
                        "stage_time_Ts": ts,
                        "data_time_Td": td,
                        "limiting_capability": bottleneck,
                        "calibrated_body_us": (ts * td) / rates[bottleneck] * 1.0e6,
                    })
    records.sort(key=lambda row: (row["logN"], row["word_bytes"], row["spatial_budget_C"], row["calibrated_body_us"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def target_batch(log_n: int) -> int:
    return max(1, (1 << 22) // (1 << log_n))


def candidate_commands(build_dir: pathlib.Path, cufftdx: bool) -> list[tuple[str, list[str]]]:
    binary = build_dir / "cubutterfly_bench"
    common = ["--precision", "fp32", "--placement", "out-of-place", "--normalization", "none"]
    candidates: list[tuple[str, list[str]]] = []

    def add(name: str, args: list[str]) -> None:
        candidates.append((name, [str(binary), *args]))

    for log_n in (8, 12, 16):
        batch = target_batch(log_n)
        add(f"fft-log{log_n}-cufft", ["--operator", "fft", "--backend", "cufft", "--logN", str(log_n), "--batch", str(batch), *common])
    add("fft-log8-cta-dft8", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cta-dft8", "--compute-unit", "radix8", "--tile-threads", "128", "--logN", "8", "--batch", str(target_batch(8)), *common])
    add("fft-log12-scalar-hierarchical", ["--operator", "fft", "--backend", "hierarchical", "--fft-core", "scalar", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("fft-log12-shared-temporal-radix4", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "scalar", "--compute-unit", "radix4", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    if cufftdx:
        add("fft-log12-cufftdx-direct", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cufftdx-direct", "--tile-threads", "1024", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-direct512", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cufftdx-direct", "--tile-threads", "512", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-online", ["--operator", "fft", "--backend", "online-reorder", "--fft-core", "cufftdx-block", "--local-stages", "6", "--prefix-threads", "512", "--suffix-threads", "512", "--prefix-ept", "8", "--suffix-ept", "8", "--reorder-columns", "1", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-resident", ["--operator", "fft", "--backend", "online-reorder", "--fft-core", "cufftdx-resident", "--local-stages", "6", "--tile-threads", "512", "--reorder-columns", "1", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("fwht-log15-warp-register", ["--operator", "fwht", "--backend", "temporal-tile", "--local-exchange", "warp-register", "--compute-unit", "radix2", "--tile-threads", "256", "--logN", "15", "--batch", str(target_batch(15)), *common])
    add("subset-zeta-log12-hierarchical", ["--operator", "subset-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("superset-zeta-log12-hierarchical", ["--operator", "superset-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("xor-zeta-log12-hierarchical", ["--operator", "xor-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("structured-2x2-log12-hierarchical", ["--operator", "structured-2x2", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--stage-matrix", "0.9238795,-0.3826834,0.3826834,0.9238795", "--logN", "12", "--batch", str(target_batch(12)), *common])
    return candidates


def ntt_candidate_commands(build_dir: pathlib.Path) -> list[tuple[str, list[str]]]:
    binary = build_dir / "cuntt_bench"
    return [
        ("ntt-log12-hybrid2d-radix4-word32", [str(binary), "--backend", "hybrid2d", "--compute-unit", "radix4", "--cross-twiddle", "fused", "--word-bits", "32", "--output-order", "natural", "--modulus", "998244353", "--logN", "12", "--batch", str(target_batch(12))]),
        ("ntt-log16-hybrid2d-radix4-word64", [str(binary), "--backend", "hybrid2d", "--compute-unit", "radix4", "--cross-twiddle", "fused", "--word-bits", "64", "--output-order", "natural", "--modulus", "576460756061519873", "--logN", "16", "--batch", str(target_batch(16))]),
    ]


def _search_workload_key(workload: dict[str, Any]) -> str:
    return json.dumps(workload, sort_keys=True)


def _resume_selection_prefix(cell: dict[str, Any], retained: dict[str, dict[str, Any]]):
    """Recover only a fully persisted, contiguous selection prefix.

    ``selection_index`` is written before a candidate is measured, while
    ``screened`` advances only after the measurement is persisted.  Requiring
    both the contiguous audit entries and their journal rows prevents a
    checkpoint taken mid-selection from being treated as completed work.
    """
    if not isinstance(cell, dict) or cell.get("complete"):
        return None
    screened_count = cell.get("screened")
    if type(screened_count) is not int or screened_count <= 0:
        return None
    candidates = cell.get("candidates")
    if not isinstance(candidates, list):
        return None
    indexed = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        selection_index = candidate.get("selection_index")
        if type(selection_index) is not int or selection_index < 0:
            continue
        if selection_index in indexed:
            return None
        indexed[selection_index] = candidate
    prefix = []
    for selection_index in range(screened_count):
        candidate = indexed.get(selection_index)
        if not isinstance(candidate, dict):
            return None
        candidate_id_value = candidate.get("id")
        configuration = candidate.get("configuration")
        if not isinstance(candidate_id_value, str) or not isinstance(configuration, dict):
            return None
        try:
            if candidate_id(configuration) != candidate_id_value:
                return None
        except (TypeError, ValueError):
            return None
        row = retained.get(candidate_id_value)
        if not isinstance(row, dict) or row.get("name") != candidate_id_value:
            return None
        row_configuration = row.get("configuration")
        if not isinstance(row_configuration, dict):
            return None
        try:
            if candidate_id(row_configuration) != candidate_id_value:
                return None
        except (TypeError, ValueError):
            return None
        prefix.append((copy.deepcopy(configuration), copy.deepcopy(row)))
    return prefix


def run_compiled_search(args, output):
    binary = args.build_dir / "cubutterfly_bench"
    staged = getattr(args, "cost_model", "legacy") == "staged"
    evolutionary = getattr(args, "search_strategy", "model") == "evolutionary"
    profile = json.loads(output.with_name("calibration_device.json").read_text()) if staged else None
    probe = args.build_dir / "cubutterfly_plan_probe"
    if staged and not probe.is_file():
        raise ValueError("staged costs require cubutterfly_plan_probe; build/install that target first")
    if staged:
        profile["hardware"] = json.loads(run([str(probe), "--hardware"]).stdout)
        write_checkpoint(output.with_name("calibration_device.json"), profile)
    units = list(csv.DictReader(io.StringIO(run([str(binary), "--list-processing-units"]).stdout)))
    for unit in units:
        for field in ("logN", "threads", "ept"):
            unit[field] = int(unit[field])
    register_mappings = list(csv.DictReader(io.StringIO(run([str(binary), "--list-register-tile-mappings"]).stdout)))
    specification = json.loads(args.search_workloads.read_text())
    if specification.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported install search workload schema")
    seed_budget = getattr(args, "seed_budget", 0)
    snapshot = seed_snapshot(args, output.with_name("mapping_seed_snapshot.json")) if seed_budget else {"seeds": []}
    audit = {"schema": "cubutterfly-search-coverage-v1", "scope": specification["scope"],
             "measurement_identity": measurement_identity(args),
             "search_protocol": {key:getattr(args,key,None) for key in
                ("search_budget","search_finalists","search_seconds","compile_seconds",
                 "operator_trials","operator_warmup","operator_repeat")},
             "partition_projection_budget": int(os.environ.get("CUBUTTERFLY_PARTITION_BUDGET", "128")),
             "partition_projection": "library cursor spans all resource-feasible group counts; 0 removes enumeration budget",
             "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
             "compiled_local_units": units, "search_budget_per_workload": args.search_budget,
             "compiled_register_tile_mappings": register_mappings,
             "optimality": "best confirmed measured candidate; no global-optimum claim",
             "workloads": []}
    audit["search_protocol"]["seed_budget"] = seed_budget
    audit["search_protocol"]["cost_feature_version"] = FEATURE_VERSION
    audit["search_protocol"]["cost_projection_version"] = PROJECTION_VERSION
    audit["search_protocol"]["model_update_policy_version"] = MODEL_UPDATE_POLICY_VERSION
    audit["search_protocol"]["selection_policy_version"] = SEARCH_SELECTION_VERSION
    audit["search_protocol"]["cost_model"] = "staged" if staged else "legacy"
    audit["search_protocol"]["search_strategy"] = "evolutionary" if evolutionary else "model"
    audit["search_protocol"]["stage_calibration"] = getattr(args, "stage_calibration", "skip")
    if staged:
        audit["search_protocol"]["stage_model_version"] = stage_cost_model.VERSION
        audit["search_protocol"]["plan_probe_sha256"] = hashlib.sha256(probe.read_bytes()).hexdigest()
    audit["search_protocol"]["seed_snapshot_sha256"] = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    result = []
    search_start = time.monotonic()
    def compile_spent():
        budget_file=os.environ.get("CUBUTTERFLY_COMPILE_BUDGET_FILE")
        return float(pathlib.Path(budget_file).read_text() or "0") if budget_file and pathlib.Path(budget_file).exists() else 0
    compile_start=compile_spent()
    def search_elapsed():
        return time.monotonic()-search_start-(compile_spent()-compile_start)
    cached = {}
    retained = {}
    completed_cells = {}
    resumable_cells = {}
    journal = output.with_name("search_measurements.json")
    if args.resume_search and output.exists() and journal.exists():
        previous = json.loads(output.read_text())
        if previous.get("measurement_identity") != audit["measurement_identity"]:
            raise ValueError("cannot resume search measurements from different binaries, compile policy or verification coverage")
        for row in json.loads(journal.read_text()):
            retained[row["name"]] = row
            if row.get("correct") and row.get("samples"):
                cached[row["name"]] = row
        if previous.get("search_protocol")==audit["search_protocol"]:
            previous_cells = previous.get("workloads", [])
            completed_cells={_search_workload_key(cell["workload"]):cell for cell in previous_cells
                             if isinstance(cell, dict) and cell.get("workload") is not None and cell.get("complete")}
            resumable_cells={_search_workload_key(cell["workload"]):cell for cell in previous_cells
                             if isinstance(cell, dict) and cell.get("workload") is not None and not cell.get("complete")}
    def save_measurements(rows):
        # Resuming an earlier cell must not erase completed later cells if
        # another interruption occurs before the traversal reaches them.
        retained.update((row["name"],row) for row in rows)
        write_checkpoint(journal,list(retained.values()))

    descriptors = {}
    descriptor_path = output.with_name("plan_descriptors.json")
    descriptor_identity = {"measurement": audit["measurement_identity"],
        "service_projection_version": "actual-batch-tile-v1",
        "probe": audit["search_protocol"].get("plan_probe_sha256"),
        "stage_probe": hashlib.sha256((args.build_dir / "cubutterfly_stage_microbench").read_bytes()).hexdigest()
            if staged and profile.get("stage_service") else None,
        "profile": {key:value for key,value in (profile or {}).items() if key != "stage_service"}}
    if staged and args.resume_search and descriptor_path.exists():
        previous = json.loads(descriptor_path.read_text())
        if previous.get("identity") == descriptor_identity:
            descriptors = previous.get("descriptors", {})

    def projected(point):
        resolved = descriptors.get(candidate_id(point), {})
        if resolved.get("status") == "resolved":
            return {**point, **resolved.get("sample", {}), **json.loads(resolved["mapping_json"]), "N": 1 << int(point["logN"]),
                "stage_service_projection": resolved.get("stage_service_projection", {}),
                "execution_groups_json": resolved["execution_groups_json"], "descriptor_source": "resolved-plan"}
        return {**predicted_sample(point), "descriptor_source": "projected"}

    def resolve(point):
        key = candidate_id(point)
        if key not in descriptors:
            actual_stage = staged and profile.get("stage_service")
            command = ([str(args.build_dir / "cubutterfly_stage_microbench"), "--describe-only"]
                       if actual_stage else [str(probe)])
            completed = run(command + ["--point-json", json.dumps(point)], check=False)
            try:
                value = json.loads(completed.stdout)
                if actual_stage and value.get("status") == "resolved":
                    from stage_composition import resolve_batch_services
                    def describe_tile(tile_point):
                        return resolve(tile_point)
                    value["stage_service_projection"] = resolve_batch_services(point, value, describe_tile)
                    sample = value["sample"]
                    value = {**value, "mapping_json":sample["mapping_json"],
                        "execution_groups_json":value["groups"],
                        "runtime_fingerprint":sample["runtime_fingerprint"], "descriptor_source":"actual-stage-probe"}
            except ValueError:
                value = dict(status="unavailable", reason=completed.stderr or "plan probe returned no JSON")
            # Compilation exhaustion may be retried with a later round's budget.
            if "compilation budget exhausted" not in value.get("reason", ""):
                descriptors[key] = value
                write_checkpoint(descriptor_path, dict(identity=descriptor_identity, descriptors=descriptors))
            return value
        return descriptors[key]

    def measure(point, trials, warmup, repeat, *, seed=False):
        candidate_binary = args.build_dir / ("cuntt_bench" if point.get("operator") == "ntt" else "cubutterfly_bench")
        command = command_for(candidate_binary, point)
        old = cached.get(candidate_id(point))
        reusable=bool(old and old.get("correct") and old.get("samples") and all(
            int(s["warmup"])>=warmup and int(s["repeat"])>=repeat for s in old["samples"]))
        descriptor = descriptors.get(candidate_id(point), {})
        if reusable and descriptor.get("descriptor_source") == "actual-stage-probe":
            # A compatible earlier run may predate independent stage
            # calibration. Reuse its timings while upgrading only the
            # descriptor, after checking the actual code/mapping identity.
            old = copy.deepcopy(old)
            try:
                for sample in old["samples"]:
                    if sample.get("runtime_fingerprint") != descriptor.get("runtime_fingerprint"):
                        raise ValueError("cached timing and stage descriptor build mismatch")
                    validate_mapping(json.loads(descriptor["mapping_json"]), json.loads(sample["mapping_json"]))
                    sample.setdefault("benchmark_execution_groups_json",sample.get("execution_groups_json","[]"))
                    sample["execution_groups_json"] = json.dumps(descriptor["execution_groups_json"])
                    sample["stage_service_projection"] = descriptor.get("stage_service_projection", {})
                    sample["descriptor_source"] = "actual-stage-probe"
            except (ValueError, TypeError, KeyError):
                reusable = False
        if reusable and seed:
            try:
                for sample in old["samples"]:
                    validate_mapping(json.loads(point["mapping_json"]), json.loads(sample.get("mapping_json", "{}")))
            except (ValueError, TypeError):
                reusable = False
        if reusable and len(old["samples"])>=trials:
            return dict(copy.deepcopy(old),status="measured")
        record = {"name": candidate_id(point), "command": command, "configuration": point,
                  "samples": copy.deepcopy(old["samples"]) if reusable else [],
                  "errors": [], "correct": False, "status": "unavailable"}
        for _ in range(len(record["samples"]),trials):
            require_exclusive_gpu()
            verification = ["--verify-batches",str(args.verify_batches)] if getattr(args,"verify_batches",0) else []
            completed = run(command + ["--warmup", str(warmup), "--repeat", str(repeat), "--verify", "--csv"] + verification, check=False)
            require_exclusive_gpu()
            if completed.returncode:
                record["correct"]=False
                record["errors"].append(completed.stderr.strip() or completed.stdout.strip())
                if "compilation budget exhausted" in record["errors"][-1]:
                    record["status"]="needs-compilation"
                return record
            samples = list(csv.DictReader(io.StringIO(completed.stdout)))
            if len(samples) != 1 or samples[0].get("correct") != "1":
                record["correct"]=False
                record["status"] = "incorrect"
                return record
            sample = samples[0]
            descriptor = descriptors.get(candidate_id(point), {})
            if staged and descriptor.get("status") == "resolved" and sample.get("runtime_fingerprint") != descriptor.get("runtime_fingerprint"):
                record["errors"].append("plan probe and timing binary resolved different runtime fingerprints")
                return record
            if staged and descriptor.get("descriptor_source") == "actual-stage-probe":
                try:
                    validate_mapping(json.loads(descriptor["mapping_json"]), json.loads(sample["mapping_json"]))
                except (ValueError, TypeError, KeyError) as error:
                    record.update(correct=False, status="mapping-mismatch")
                    record["errors"].append(str(error))
                    return record
                sample["benchmark_execution_groups_json"] = sample.get("execution_groups_json", "[]")
                sample["execution_groups_json"] = json.dumps(descriptor["execution_groups_json"])
                sample["stage_service_projection"] = descriptor.get("stage_service_projection", {})
                sample["descriptor_source"] = "actual-stage-probe"
            if point.get("operator") == "ntt":
                sample.update(operator="ntt", precision="word"+sample["word_bits"],
                              placement=sample["output_order"], direction="inverse" if sample["inverse"] == "1" else "forward")
            sample["measurement_exclusive_gpu"] = "1"
            sample.setdefault("modulus", "0")
            if seed:
                try:
                    validate_mapping(json.loads(point["mapping_json"]), json.loads(sample.get("mapping_json", "{}")))
                    # Match target semantics, including placement and strides;
                    # source semantics are never copied into the command.
                    expected_semantics, resolved_semantics = canonical_semantics(point), canonical_semantics(sample)
                    for key in ("placement", "direction", "normalization", "element_stride", "batch_stride",
                                "accumulation", "modulus", "input_order", "output_order"):
                        if key in point and resolved_semantics.get(key) != expected_semantics.get(key):
                            raise ValueError(f"resolved seed semantics mismatch for {key}")
                except (ValueError, TypeError) as error:
                    record.update(correct=False, status="mapping-mismatch")
                    record["errors"].append(str(error))
                    return record
            if record["samples"] and any(sample.get(key) != record["samples"][0].get(key)
                                          for key in ("mapping_json", "runtime_fingerprint")):
                raise ValueError("mapping or code changed during search confirmation")
            # A timed fallback is not evidence for the requested design point.
            for key, value in point.items():
                if key == "stage_matrix" or ("mapping_json" in point and key not in ("backend", "fft_core", "logN", "batch", "precision", "operator")):
                    continue
                if key not in sample or str(sample[key]) != str(value):
                    raise ValueError(f"resolved mapping mismatch for {key}: {sample.get(key)} != {value}")
            record["samples"].append(sample)
            record.update(status="screened",correct=True,trials=len(record["samples"]),
                median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in record["samples"]))
            cached[record["name"]]=copy.deepcopy(record)
            save_measurements([record])
        record.update(status="measured", correct=True, trials=trials,
                      median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in record["samples"]))
        return record

    for workload_index, workload in enumerate(specification["workloads"]):
        remaining_cells=len(specification["workloads"])-workload_index
        cell_start=search_elapsed()
        cell_budget=max(0,args.search_seconds-cell_start)/remaining_cells if args.search_seconds else 0
        workload = {"operator": "fft", **workload}
        completed=completed_cells.get(_search_workload_key(workload))
        if completed:
            audit["workloads"].append(completed)
            result.extend(retained[p["id"]] for p in completed["candidates"] if p["id"] in retained)
            print(f"resume completed {workload['operator']} {workload['precision']} logN={workload['logN']} batch={workload['batch']}",flush=True)
            continue
        candidate_binary = args.build_dir / ("cuntt_bench" if workload.get("operator") == "ntt" else "cubutterfly_bench")
        points = runtime_candidates(candidate_binary, workload, run)
        seeds = candidates_for_workload(snapshot, workload)
        chosen_seeds = seeds[:seed_budget]
        chosen_ids = {candidate_id(s["point"]) for s in chosen_seeds}
        resume_cell = resumable_cells.get(_search_workload_key(workload))
        resume_prefix = _resume_selection_prefix(resume_cell, retained) if resume_cell else None
        resume_fast = bool(resume_prefix)
        remaining = stratified_candidates([p for p in points if candidate_id(p) not in chosen_ids], 0)
        exploration_budget = (args.search_budget or len(remaining)) if evolutionary else min(args.search_budget or len(remaining), len(remaining))
        budget = len(chosen_seeds) + exploration_budget
        if resume_fast:
            previous_budget = resume_cell.get("requested_screened_count")
            if (type(previous_budget) is int and previous_budget != budget) or len(resume_prefix) > budget:
                resume_fast = False
                resume_prefix = None
            else:
                # Keep the original stratified order and full budget. Only
                # already screened prefix IDs are removed from the frontier.
                resume_prefix_ids = {candidate_id(item[0]) for item in resume_prefix}
                remaining = [point for point in remaining if candidate_id(point) not in resume_prefix_ids]
        unique = {candidate_id(p): p for p in points}
        unique.update((candidate_id(s["point"]), s["point"]) for s in seeds)
        points = list(unique.values())
        fresh_cell = {"workload": workload, "enumerated": len(points), "screened": 0,
                "candidate_source": "linked-library+mapping-seeds" if seeds else "linked-library", "model_updates": [],
                "requested_screened_count": budget,
                "seed_coverage": {"available": len(seeds), "requested": len(chosen_seeds),
                                  "attempted": 0, "confirmed": 0, "budget_omitted": len(seeds)-len(chosen_seeds)},
                "budget_omitted": len(points),
                "partitions": sorted({tuple(json.loads(p["mapping_json"]).get("stage_partition", [])) for p in points}),
                "candidates": [{"id": candidate_id(p), "configuration": p,
                                "status": "budget-omitted"} for p in points]}
        if resume_fast:
            cell = copy.deepcopy(resume_cell)
            cell["workload"] = workload
            cell["screened"] = len(resume_prefix)
            cell["requested_screened_count"] = budget
            cell["enumerated"] = max(int(cell.get("enumerated", 0)), len(points), len(cell.get("candidates", [])))
            cell["budget_omitted"] = len(cell.get("candidates", [])) - len(resume_prefix)
            cell.pop("complete", None)
            cell.pop("stop_reason", None)
            by_id = {candidate["id"]: candidate for candidate in cell.get("candidates", [])
                     if isinstance(candidate, dict) and isinstance(candidate.get("id"), str)}
            for point in points:
                identifier = candidate_id(point)
                if identifier in by_id:
                    continue
                candidate = {"id": identifier, "configuration": point, "status": "budget-omitted"}
                cell.setdefault("candidates", []).append(candidate)
                by_id[identifier] = candidate
            cell["enumerated"] = max(cell["enumerated"], len(cell["candidates"]))
        else:
            cell = fresh_cell
        audit["workloads"].append(cell)
        by_id = {p["id"]: p for p in cell["candidates"]}
        for seed in seeds:
            by_id[candidate_id(seed["point"])]["seed_sources"] = seed["sources"]
        screened = [row for _, row in resume_prefix] if resume_fast else []
        if resume_fast:
            print(f"resume screened prefix {workload['operator']} {workload['precision']} "
                  f"logN={workload['logN']} batch={workload['batch']} {len(screened)}/{budget}", flush=True)
            for row in screened:
                by_id[row["name"]].update(status=row.get("status", "screened"),
                                           errors=row.get("errors", []),
                                           median_kernel_ms=row.get("median_kernel_ms"))
        model = stage_cost_model.fit([], profile) if staged else None
        model_training_ids = None
        for i in range(len(screened), budget):
            if args.search_seconds and search_elapsed()-cell_start >= cell_budget:
                cell["stop_reason"] = "fair-share-search-time-budget"
                break
            pool = {r["name"]: r for r in ([*retained.values(), *screened] if staged else screened)
                    if r["correct"] and r.get("samples")}
            training = [{"sample": r["samples"][0], "median_kernel_ms": r["median_kernel_ms"]} for r in pool.values()]
            training_ids = tuple((r["name"], r["median_kernel_ms"]) for r in pool.values())
            if len(training) >= (1 if staged else 8) and training_ids != model_training_ids:
                model = stage_cost_model.fit(training, profile) if staged else fit(training, 1.0e-3)
                model_training_ids = training_ids
                update = {"screened": i, "training_rows": len(training),
                    "training_candidate_ids": list(pool),
                    "feature_version": FEATURE_VERSION,
                    "projection_version": PROJECTION_VERSION,
                    "use": "target-local search ranking; every fourth selection explores a stratum"}
                audit_model = dict(model) if staged else None
                if staged and profile.get("stage_service"):
                    audit_model.pop("stage_service", None)
                    audit_model["profile"] = {key:value for key,value in model.get("profile", {}).items() if key != "stage_service"}
                    audit_model["stage_service_artifact"] = str(output.with_name("stage_service_profile.json"))
                    audit_model["stage_service_sha256"] = hashlib.sha256(output.with_name("stage_service_profile.json").read_bytes()).hexdigest()
                update.update({"stage_model": audit_model} if staged else
                    {"coefficients": model[0], "feature_means": model[1], "feature_scales": model[2]})
                cell["model_updates"].append(update)
            is_seed = i < len(chosen_seeds)
            exploit = not is_seed and model and (i-len(chosen_seeds))%4
            def score(point):
                # A runtime rejection overrides a cheap symbolic estimate.
                # Keep its descriptor in the audit, but never rank it ahead of
                # a resolved candidate because its hypothetical cost is lower.
                if staged and descriptors.get(candidate_id(point), {}).get("status") == "unavailable":
                    return float("inf")
                try:
                    return stage_cost_model.predict(projected(point), model) if staged else predict(predicted_sample(point), *model)
                except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                    # A syntactic neighbor may need a different lowering or
                    # violate a local shape constraint. The runtime probe and
                    # verifier decide feasibility; it is not a zero-cost point.
                    return float("inf")
            proposals = []
            # Exploration selects the next stratum directly. Building and
            # ranking a neighborhood here cannot affect that choice.
            if evolutionary and exploit and screened:
                from evolutionary_search import evolutionary_candidates
                parents = [r["configuration"] for r in sorted(screened, key=lambda r:r.get("median_kernel_ms", float("inf"))) if r["correct"]]
                proposals = evolutionary_candidates(parents, remaining, {r["name"] for r in screened}, score, limit=32)
            choices = proposals if proposals and exploit else remaining
            if not is_seed and not choices:
                cell["stop_reason"] = "candidate-neighborhood-exhausted"
                break
            if is_seed:
                point = chosen_seeds[i]["point"]
            elif exploit:
                shortlist = sorted(choices, key=score)[:4] if staged else choices
                if staged:
                    for proposed in shortlist:
                        resolve(proposed)
                    # Retain unavailable proposals as explicit failures; don't
                    # pretend that a projected neighbor is an executable kernel.
                point = min(shortlist, key=score)
            else:
                point = choices[0]
            remaining = [p for p in remaining if candidate_id(p) != candidate_id(point)]
            if candidate_id(point) not in by_id:
                record = dict(id=candidate_id(point), configuration=point, status="budget-omitted", source="evolutionary-neighbor")
                cell["candidates"].append(record); by_id[record["id"]] = record
                points.append(point); cell["enumerated"] = len(points)
                cell["generated_neighbors"] = cell.get("generated_neighbors", 0) + 1
            selection = by_id[candidate_id(point)]
            selection["selection_reason"] = "historical-seed" if is_seed else ("evolutionary-model" if proposals and exploit else "model" if exploit else "stratified-exploration")
            selection["selection_index"] = i
            if exploit:
                estimate = score(point)
                selection["predicted_kernel_ms"] = estimate if math.isfinite(estimate) else None
            if staged:
                description = resolve(point)
                selection["plan_probe"] = description
                try:
                    selection["cost_breakdown"] = stage_cost_model.predict(projected(point), model, True)
                except (ValueError, ZeroDivisionError, OverflowError, IndexError) as error:
                    selection["cost_breakdown"] = dict(status="unestimated", reason=str(error))
            print(f"search {workload['operator']} {workload['precision']} logN={workload['logN']} batch={workload['batch']} {i + 1}/{budget}", flush=True)
            row = measure(point, 1, max(3, min(args.operator_warmup, 10)), max(5, min(args.operator_repeat, 10)), seed=is_seed)
            if is_seed:
                cell["seed_coverage"]["attempted"] += 1
                row["seed_sources"] = selection["seed_sources"]
            if row["correct"]: row["status"]="screened"
            by_id[row["name"]]["status"] = row["status"]
            screened.append(row)
            cell["screened"] = len(screened)
            cell["budget_omitted"] = len(points)-len(screened)
            write_checkpoint(output,audit)
            # Persist completed screens immediately. An unrelated GPU job may
            # arrive before this cell reaches finalist confirmation.
            save_measurements(result+[dict(r,status="screened") if r["correct"] else r for r in screened])
        valid = sorted((r for r in screened if r["correct"]), key=lambda r: r["median_kernel_ms"])
        # A candidate name is provenance, not a resolved design identity. Seed
        # aliases and library aliases can resolve to the same mapping. Use the
        # first correct screen sample so its complete mapping, workload
        # semantics, and hardware context participate in finalist selection.
        # design_point_key deliberately falls back to conservative name/axis
        # identity when a resolved mapping envelope is incomplete.
        finalist_keys = set()
        finalists = set()
        for row in valid:
            samples = row.get("samples")
            if isinstance(samples, list) and samples and isinstance(samples[0], dict):
                key = design_point_key({"name": row.get("name", ""), "sample": samples[0]})
            else:
                key = ("incomplete-screen", row.get("name", ""))
            if key in finalist_keys:
                continue
            finalist_keys.add(key)
            finalists.add(row["name"])
            if len(finalists) >= args.search_finalists:
                break
        for screened_index, row in enumerate(screened):
            if row["name"] in finalists:
                row = measure(row["configuration"], args.operator_trials, args.operator_warmup, args.operator_repeat,
                              seed=row["name"] in chosen_ids)
            elif row["correct"]:
                row["status"] = "screened"
            by_id[row["name"]].update(status=row["status"], errors=row.get("errors", []),
                                     median_kernel_ms=row.get("median_kernel_ms"))
            if row["name"] in chosen_ids:
                row["seed_sources"] = by_id[row["name"]]["seed_sources"]
                if row["correct"] and row["status"] == "measured":
                    cell["seed_coverage"]["confirmed"] += 1
            result.append(row)
            save_measurements(result+screened[screened_index+1:])
        # Preserve completed cells if calibration is interrupted later.
        save_measurements(result)
        cell["complete"]=True
        write_checkpoint(output,audit)
    write_checkpoint(output,audit)
    return result


def incumbent_commands(args):
    """Remeasure the installed selector so a bounded search keeps its incumbent.

    No winning parameters are encoded here. The resolved mapping returned by
    the binary is the evidence; unavailable workloads simply have no incumbent.
    """
    document = json.loads(args.search_workloads.read_text())
    for workload in document["workloads"]:
        point = {"operator": "fft", **workload}
        binary = args.build_dir / ("cuntt_bench" if point.get("operator") == "ntt" else "cubutterfly_bench")
        yield "incumbent-" + candidate_id(point), command_for(binary, point) + ["--auto-select"]


def run_operator_calibration(args: argparse.Namespace, output: pathlib.Path, cufftdx: bool) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    identity=measurement_identity(args)
    identity_path=output.with_name("operator_measurement_identity.json")
    cached={}
    if args.resume_search and output.exists() and identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("cannot resume incumbent measurements from different binaries, compile policy or verification coverage")
        cached={row["name"]:row for row in json.loads(output.read_text())}
    write_checkpoint(identity_path,identity)
    retained=dict(cached)
    def save_incumbents():
        retained.update((row["name"],row) for row in raw)
        write_checkpoint(output,list(retained.values()))
    commands = list(incumbent_commands(args))
    if not args.search_only:
        commands = candidate_commands(args.build_dir, cufftdx) + ntt_candidate_commands(args.build_dir) + commands
    for name, command in commands:
        print(f"incumbent {name}",flush=True)
        previous=cached.get(name)
        reusable=bool(previous and previous.get("correct") and previous.get("samples") and all(
            int(s["warmup"])>=args.operator_warmup and int(s["repeat"])>=args.operator_repeat for s in previous["samples"]))
        if reusable and len(previous["samples"])>=args.operator_trials:
            raw.append(dict(previous,status="measured"))
            save_incumbents()
            continue
        samples: list[dict[str, str]] = copy.deepcopy(previous["samples"]) if reusable else []
        errors: list[str] = []
        record={"name":name,"command":command,"samples":samples,"errors":errors,
                "correct":False,"status":"unavailable","trials":0}
        raw.append(record)
        for trial in range(len(samples),args.operator_trials):
            require_exclusive_gpu()
            verification = ["--verify-batches",str(args.verify_batches)] if getattr(args,"verify_batches",0) else []
            completed = run(command + ["--warmup", str(args.operator_warmup), "--repeat", str(args.operator_repeat), "--verify", "--csv"] + verification, check=False)
            require_exclusive_gpu()
            if completed.returncode:
                errors.append(completed.stderr.strip() or completed.stdout.strip())
                break
            try:
                rows = list(csv.DictReader(io.StringIO(completed.stdout)))
                if len(rows) != 1:
                    raise ValueError("benchmark did not return exactly one CSV row")
                sample = rows[0]
                sample["measurement_exclusive_gpu"]="1"
                if "word_bits" in sample:
                    sample["operator"] = "ntt"
                    sample["precision"] = f"word{sample['word_bits']}"
                    sample["placement"] = sample["output_order"]
                    sample["direction"] = "inverse" if sample["inverse"] == "1" else "forward"
                else:
                    sample.setdefault("modulus", "0")
                if samples and any(sample.get(key)!=samples[0].get(key) for key in ("mapping_json","runtime_fingerprint")):
                    raise ValueError("incumbent mapping or code changed during confirmation")
                samples.append(sample)
                record.update(status="screened",correct=all(s.get("correct")=="1" for s in samples),trials=len(samples),
                    median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in samples))
                save_incumbents()
            except (ValueError, csv.Error) as error:
                errors.append(str(error))
                break
        if samples and len(samples) == args.operator_trials and not errors:
            values = [float(row["kernel_ms"]) for row in samples]
            record.update({
                "name": name,
                "status": "measured" if all(row.get("correct") == "1" for row in samples) else "incorrect",
                "trials": len(samples),
                "median_kernel_ms": statistics.median(values),
                "correct": all(row.get("correct") == "1" for row in samples),
                "command": command,
                "samples": samples,
            })
        else:
            record.update(status="unavailable",correct=False,errors=errors)
        save_incumbents()
    raw.extend(run_compiled_search(args, output.with_name("search_coverage.json")))
    output.write_text(json.dumps(raw, indent=2) + "\n")
    summary_path = output.with_suffix(".csv")
    with summary_path.open("w", newline="") as handle:
        fields = ["name", "status", "trials", "median_kernel_ms", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in raw:
            writer.writerow({field: row.get(field, "") for field in fields})
    recommendations: dict[str, dict[str, Any]] = {}
    operator_families: set[str] = set()
    for row in raw:
        if row.get("status") != "measured" or not row.get("correct"):
            continue
        operator_families.update(sample.get("operator", "") for sample in row.get("samples", []) if sample.get("operator"))
        key = json.dumps(semantic_key(row["samples"][0]))
        current = recommendations.get(key)
        if current is None or row["median_kernel_ms"] < current["median_kernel_ms"]:
            recommendations[key] = {
                "candidate": row["name"],
                "median_kernel_ms": row["median_kernel_ms"],
            }
    recommendation_path = output.with_name("operator_recommendations.json")
    recommendation_path.write_text(json.dumps(recommendations, indent=2) + "\n")
    return {
        "status": "complete",
        "cufftdx_enabled": cufftdx,
        "candidates": raw,
        "operator_families": sorted(operator_families),
        "summary": str(summary_path),
        "recommendations": str(recommendation_path),
    }


def _compute_capability_parts(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"cannot parse compute capability: {value}")
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    args = parse_args()
    if args.trials <= 0 or args.operator_trials <= 0 or args.operator_warmup < 0 or args.operator_repeat <= 0:
        raise SystemExit("trials and repeat values must be positive")
    if args.search_budget < 0 or args.seed_budget < 0 or args.search_finalists < 1 or args.search_seconds < 0 or args.compile_seconds < 0:
        raise SystemExit("search budget must be nonnegative and finalists positive")
    build_dir = args.build_dir.expanduser().resolve()
    profile_dir = args.profile_dir.expanduser().resolve()
    output_dir = (args.output_dir or profile_dir).expanduser().resolve()
    args.build_dir = build_dir
    python = build_python(build_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_budget = tempfile.TemporaryDirectory(prefix="cubutterfly-install-budget-")
    os.environ["CUBUTTERFLY_COMPILE_BUDGET_FILE"] = str(pathlib.Path(compile_budget.name)/"seconds")
    os.environ["CUBUTTERFLY_COMPILE_BUDGET_SECONDS"] = str(args.compile_seconds)

    profile_command = [python, str(PROFILE_SCRIPT), "--microbench", str(build_dir / "cuntt_hardware_microbench"), "--output-dir", str(profile_dir), "--trials", str(args.trials)]
    if args.force:
        profile_command.append("--force")
    require_exclusive_gpu()
    if args.force or not (profile_dir / "hardware_profile.json").exists():
        run(profile_command)
    run([python, str(CHECK_SCRIPT), str(profile_dir / "hardware_profile.json")])
    profile = json.loads((profile_dir / "hardware_profile.json").read_text())
    identity = next(csv.DictReader(io.StringIO(run([str(build_dir / "cubutterfly_bench"), "--device-identity"]).stdout)))
    if identity["device"] != profile["device"] or identity["compute_capability"] != profile["compute_capability"]:
        raise ValueError("hardware probe and benchmark device identities disagree")
    if "global_memory_bytes" in profile and int(profile["global_memory_bytes"]) != int(identity["global_memory_bytes"]):
        raise ValueError("cached hardware profile belongs to a different memory capacity; use a separate profile directory")
    profile["global_memory_bytes"] = int(identity["global_memory_bytes"])
    if args.cost_model == "staged" and getattr(args, "stage_calibration", "skip") != "skip" and not args.skip_operator_calibration:
        profile["hardware"] = json.loads(run([str(build_dir / "cubutterfly_plan_probe"), "--hardware"]).stdout)
    identity_path = output_dir / "calibration_device.json"
    if args.resume_search and identity_path.exists():
        old = json.loads(identity_path.read_text())
        if any(old.get(key) != profile.get(key) for key in ("device", "compute_capability", "global_memory_bytes")):
            raise ValueError("cannot resume search measurements from a different device model/SM/memory")
    (output_dir / "calibration_device.json").write_text(json.dumps(profile, indent=2) + "\n")
    if args.cost_model == "staged" and not args.skip_operator_calibration:
        schedule_probe = build_dir / "cubutterfly_schedule_microbench"
        if not schedule_probe.is_file():
            raise ValueError("staged calibration requires cubutterfly_schedule_microbench; build/install that target first")
        profile["scheduling"] = calibrate_scheduling(schedule_probe, output_dir / "schedule_calibration.json",
                                                   profile, run, require_exclusive_gpu)
        stage_mode = getattr(args, "stage_calibration", "skip")
        if stage_mode != "skip":
            pipeline_probe = build_dir / "cubutterfly_pipeline_microbench"
            if not pipeline_probe.is_file():
                raise ValueError("stage calibration requires cubutterfly_pipeline_microbench; build/install that target first")
            profile["pipeline_scheduling"] = calibrate_pipeline_scheduling(
                pipeline_probe, output_dir / "pipeline_schedule_calibration.json",
                profile, run, require_exclusive_gpu)
            stage_probe = build_dir / "cubutterfly_stage_microbench"
            if not stage_probe.is_file():
                raise ValueError("independent stage calibration requires cubutterfly_stage_microbench")
            specification = json.loads(args.search_workloads.read_text())
            stage_points = []
            for workload in specification["workloads"]:
                workload = {"operator": "fft", **workload}
                candidate_binary = build_dir / ("cuntt_bench" if workload["operator"] == "ntt" else "cubutterfly_bench")
                stage_points.extend(runtime_candidates(candidate_binary, workload, run))
            profile["stage_service"] = calibrate_stage_services(stage_probe, output_dir / "stage_calibration.json",
                profile, stage_points, lambda command: run(command, check=False), require_exclusive_gpu, mode=stage_mode,
                warmup=args.operator_warmup, repeat=args.operator_repeat, trials=args.operator_trials,
                time_budget=args.search_seconds if stage_mode == "bounded" else 0,
                import_checkpoints=getattr(args, "stage_import_checkpoint", []))
            write_checkpoint(output_dir / "stage_service_profile.json", profile["stage_service"])
        write_checkpoint(output_dir / "calibration_device.json", profile)
    rank_rows = rank_model(profile, output_dir / "unfolding_rank.csv")

    cache_text = (build_dir / "CMakeCache.txt").read_text() if (build_dir / "CMakeCache.txt").exists() else ""
    inventory = run([str(build_dir / "cubutterfly_bench"), "--list-processing-units"], check=False)
    cufftdx = inventory.returncode == 0 and len(list(csv.DictReader(io.StringIO(inventory.stdout)))) > 0
    operator_result: dict[str, Any]
    if args.skip_operator_calibration:
        operator_result = {"status": "skipped", "cufftdx_enabled": cufftdx, "candidates": []}
    else:
        operator_result = run_operator_calibration(args, output_dir / "operator_calibration.json", cufftdx)

    cost_model_path = output_dir / "cost_model.json"
    if args.skip_operator_calibration:
        cost_model_path.write_text(json.dumps({
            "schema": "cubutterfly-local-cost-model-v1",
            "status": "skipped",
            "device": profile["device"],
            "compute_capability": profile["compute_capability"],
            "reason": "operator calibration was skipped",
        }, indent=2) + "\n")
    else:
        run([
            python, str(SCRIPT_DIR / "stage_cost_model.py" if args.cost_model == "staged" else COST_MODEL_SCRIPT),
            "--profile", str(output_dir / "calibration_device.json" if args.cost_model == "staged" else profile_dir / "hardware_profile.json"),
            "--operator-calibration", str(output_dir / "operator_calibration.json"),
            "--output", str(cost_model_path),
        ])

    local_selector_header = build_dir / "generated" / "generated_local_selector.hpp"
    local_selector_points = write_local_selector_header(profile, operator_result, local_selector_header) if args.embed_legacy_selector else 0
    # Dynamic registry is cumulative across semantic cells and GPU capacities.
    # A skipped calibration does not erase any previously verified records.
    registry = default_path()
    registry_points = 0 if args.skip_operator_calibration else promote(registry, profile, operator_result)

    manifest = {
        "schema": "cubutterfly-local-calibration-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": profile["device"],
        "compute_capability": profile["compute_capability"],
        "build_dir": str(build_dir),
        "profile": str(profile_dir / "hardware_profile.json"),
        "unfolding_rank": str(output_dir / "unfolding_rank.csv"),
        "unfolding_rank_rows": rank_rows,
        "operator_calibration": str(output_dir / "operator_calibration.json") if not args.skip_operator_calibration else "",
        "operator_calibration_summary": str(output_dir / "operator_calibration.csv") if not args.skip_operator_calibration else "",
        "operator_recommendations": str(output_dir / "operator_recommendations.json") if not args.skip_operator_calibration else "",
        "cost_model": str(cost_model_path),
        "schedule_calibration": str(output_dir / "schedule_calibration.json") if (output_dir / "schedule_calibration.json").exists() else "",
        "stage_calibration_mode": getattr(args, "stage_calibration", "skip"),
        "stage_service_profile": str(output_dir / "stage_service_profile.json") if profile.get("stage_service") else "",
        "search_coverage": str(output_dir / "search_coverage.json") if not args.skip_operator_calibration else "",
        "global_memory_bytes": profile["global_memory_bytes"],
        "theory_alignment": "partial; see docs/theory_alignment_audit.md",
        "operator_result": operator_result,
        "operator_families": operator_result.get("operator_families", []),
        "local_selector_header": str(local_selector_header),
        "local_selector_points": local_selector_points,
        "registry_path": str(registry),
        "registry_promoted_records": registry_points,
        "runtime_auto_select": "hardware-registry-with-portable-fallback",
        "status": "ready-with-measured-mappings" if registry_points else "ready-with-unmeasured-fallback",
    }
    (output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"local calibration: {manifest['status']}")
    print(f"profile: {manifest['profile']}")
    print(f"unfolding rank: {manifest['unfolding_rank']} ({rank_rows} rows)")
    print(f"calibration manifest: {output_dir / 'calibration_manifest.json'}")
    print(f"cost model: {cost_model_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
