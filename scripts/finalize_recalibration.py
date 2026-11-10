#!/usr/bin/env python3
"""Confirm a finite search population and calibrate its actual stage services.

This reuses the acceptance engine, rather than repeating search. Outputs live
in a fresh directory. Full-inventory qualification is deliberately separate
from the readiness of the measured mappings for comparative experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import SimpleNamespace

import calibrate_hardware as migration
import calibrate_local_hardware as calibration
from calibration_space import command_for
from hardware_registry import promote
from run_comprehensive_suite import require_exclusive_gpu
from run_research_acceptance import cell_key, confirmed_services, ranking, verify_sample
from calibrate_composition import run_calibration as calibrate_composition
from pipeline_schedule_model import calibrate as calibrate_pipeline
from schedule_cost_model import calibrate as calibrate_schedule
from run_current_recalibration import require_release_ready


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    calibration.write_checkpoint(Path(path), value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_from_sample(sample):
    fields = ("operator", "precision", "logN", "batch", "placement", "direction",
              "normalization", "accumulation", "element_stride", "batch_stride",
              "modulus", "input_order", "output_order", "mapping_json")
    point = {key: sample[key] for key in fields if key in sample and sample[key] != ""}
    for field in ("logN", "batch", "element_stride", "batch_stride", "modulus"):
        if field in point:
            point[field] = int(point[field])
    if sample.get("stage_matrices"):
        point["stage_matrices"] = sample["stage_matrices"]
    return point


def validate_fingerprints(records, expected):
    selected = [row for row in records if row.get("status") == "measured" and row.get("correct")]
    if not selected:
        raise ValueError("there are no confirmed search/incumbent mappings")
    for row in selected:
        if not row.get("samples") or any(s.get("runtime_fingerprint") != expected for s in row["samples"]):
            raise ValueError("confirmation input belongs to a different runtime: " + row["name"])
    return selected


def mapping_seeds(records, expected, revalidate=False):
    if not revalidate:
        return validate_fingerprints(records, expected)
    selected = [row for row in records if row.get("status") == "measured" and row.get("correct")]
    if not selected:
        raise ValueError("there are no verified mappings to revalidate")
    # Import only a description, never old timings, verification or services.
    return [dict(name=row["name"], samples=[point_from_sample(row["samples"][0])],
                 status="seed", correct=False) for row in selected]


def load_retained(journal, identity_path, identity, resume):
    if journal.exists() and not identity_path.exists():
        raise ValueError("confirmation journal has no measurement identity; refusing reuse")
    if identity_path.exists() and (not resume or read(identity_path) != identity):
        raise ValueError("confirmation directory exists with a different identity, or --resume is missing")
    return {row["name"]: row for row in read(journal)} if resume and journal.exists() else {}


def select_hardware_profile(source_profile, current, fresh_profile=None):
    """Use explicitly supplied fresh rates, keeping old timing data out of them."""
    for field in ("device", "compute_capability", "global_memory_bytes"):
        if str(source_profile.get(field)) != str(current.get(field)):
            raise ValueError("source profile belongs to a different target: " + field)
    if fresh_profile is None:
        profile = copy.deepcopy(source_profile)
        profile.pop("stage_service", None)
        return profile
    for field in ("device", "compute_capability"):
        if str(fresh_profile.get(field)) != str(current.get(field)):
            raise ValueError("fresh hardware profile belongs to a different target: " + field)
    if ("global_memory_bytes" in fresh_profile and
            int(fresh_profile["global_memory_bytes"]) != int(current["global_memory_bytes"])):
        raise ValueError("fresh hardware profile belongs to a different memory capacity")
    if not fresh_profile.get("capabilities") or fresh_profile.get("stage_service"):
        raise ValueError("fresh rates must contain capabilities and no imported stage services")
    profile = copy.deepcopy(fresh_profile)
    profile["global_memory_bytes"] = int(current["global_memory_bytes"])
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--input-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-profile", type=Path,
                        help="fresh hardware-rate profile; never import rates or services from the seed profile")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revalidate-mappings", action="store_true",
                        help="import mapping descriptions across builds; remeasure every plan and physical service")
    parser.add_argument("--additional-mappings-csv", type=Path, action="append", default=[],
                        help="verified benchmark CSV supplying extra mapping seeds; timings are always measured anew")
    args = parser.parse_args(argv)
    if min(args.trials, args.repeat) < 1 or args.warmup < 0:
        parser.error("invalid timing protocol")
    args.build_dir = args.build_dir.resolve()
    source, output = args.input_profile.resolve(), args.output_dir.resolve()
    if source == output:
        parser.error("use a separate output directory to preserve source measurements")
    source_records = source / "operator_calibration.json"
    coverage = read(source / "search_coverage.json")
    if not coverage.get("workloads") or not all(cell.get("complete") for cell in coverage["workloads"]):
        raise ValueError("source search has incomplete workload cells")
    header = (args.build_dir / "generated/build_fingerprint.hpp").read_text()
    fingerprint = re.search(r'CUBUTTERFLY_RUNTIME_FINGERPRINT\s+"([0-9a-f]+)"', header).group(1)
    mode = calibration.effective_compile_mode(args.build_dir)
    if mode == "build-default-unknown":
        raise ValueError("set CUBUTTERFLY_COMPILE_MODE explicitly when the build default is unavailable")
    fingerprint += ":" + mode
    records = mapping_seeds(read(source_records), fingerprint, args.revalidate_mappings)
    for path in args.additional_mappings_csv:
        for sample in csv.DictReader(path.open()):
            if sample.get("correct") != "1" or not sample.get("mapping_json"):
                raise ValueError("additional mapping is missing verification/mapping: " + str(path))
            if "word_bits" in sample:
                sample.update(operator="ntt", precision="word"+sample["word_bits"],
                              placement=sample["output_order"],
                              direction="inverse" if sample["inverse"] == "1" else "forward")
            point=point_from_sample(sample)
            name="additional-"+hashlib.sha256(json.dumps(point,sort_keys=True).encode()).hexdigest()[:20]
            records.append(dict(name=name,samples=[point],status="seed",correct=False))
    identity_args = SimpleNamespace(build_dir=args.build_dir, verify_batches=0)
    library_guard = calibration._SharedLibraryGuard(args.build_dir)
    build_identity = calibration.measurement_identity(identity_args, library_guard)
    identity = dict(source_sha256=sha(source_records), build=build_identity,
                    revalidate_mappings=args.revalidate_mappings,
                    additional_mapping_sha256=[sha(path) for path in args.additional_mappings_csv],
                    trials=args.trials, warmup=args.warmup, repeat=args.repeat)
    identity['composition_protocol_version'] = '0.9.3.2'
    identity['probe_sha256'] = {name: sha(args.build_dir / name) for name in
        ('cubutterfly_stage_microbench', 'cubutterfly_schedule_microbench', 'cubutterfly_pipeline_microbench')}
    if args.hardware_profile:
        identity["hardware_profile_sha256"] = sha(args.hardware_profile)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "confirmation_identity.json"
    journal = output / "operator_calibration.json"
    retained = load_retained(journal, identity_path, identity, args.resume)
    write(identity_path, identity)
    def guard():
        require_exclusive_gpu()
        if calibration.measurement_identity(identity_args, library_guard) != build_identity:
            raise ValueError("the executable/shared library changed during confirmation")
        if any(sha(args.build_dir / name) != value for name, value in identity['probe_sha256'].items()):
            raise ValueError('calibration probe changed during confirmation')
    current = next(csv.DictReader(io.StringIO(subprocess.run(
        [str(args.build_dir / "cubutterfly_bench"), "--device-identity"],
        check=True, capture_output=True, text=True).stdout)))
    profile = select_hardware_profile(read(source / "calibration_device.json"), current,
        read(args.hardware_profile) if args.hardware_profile else None)
    if args.hardware_profile:
        guard()
        profile["hardware"] = json.loads(subprocess.run(
            [str(args.build_dir / "cubutterfly_plan_probe"), "--hardware"],
            check=True, capture_output=True, text=True).stdout)
        guard()
    profile["runtime_fingerprint"] = fingerprint
    def run_probe(command, check=True):
        return subprocess.run(command, text=True, capture_output=True, check=check)
    profile['scheduling'] = calibrate_schedule(args.build_dir / 'cubutterfly_schedule_microbench',
        output / 'schedule_calibration.json', profile, run_probe, guard)
    profile['pipeline_scheduling'] = calibrate_pipeline(args.build_dir / 'cubutterfly_pipeline_microbench',
        output / 'pipeline_schedule_calibration.json', profile, run_probe, guard)
    confirmed = []
    for index, original in enumerate(records):
        row = retained.get(original["name"])
        if row and row.get("correct") and len(row.get("samples", [])) == args.trials:
            confirmed.append(row)
            continue
        point = point_from_sample(original["samples"][0])
        binary = args.build_dir / ("cuntt_bench" if point["operator"] == "ntt" else "cubutterfly_bench")
        command = command_for(binary, point) + ["--verify", "--csv", "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
        samples = []
        for _ in range(args.trials):
            guard()
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            guard()
            rows = list(csv.DictReader(io.StringIO(completed.stdout)))
            if len(rows) != 1:
                raise ValueError("benchmark did not return one sample")
            sample = rows[0]
            if point["operator"] == "ntt":
                sample.update(operator="ntt", precision="word" + sample["word_bits"],
                              placement=sample["output_order"],
                              direction="inverse" if sample["inverse"] == "1" else "forward")
            verify_sample(point, sample, profile)
            if sample.get("runtime_fingerprint") != fingerprint or json.loads(sample["mapping_json"]) != json.loads(point["mapping_json"]):
                raise ValueError("confirmation resolved a different runtime or mapping")
            sample["measurement_exclusive_gpu"] = "1"
            samples.append(sample)
        row = dict(copy.deepcopy(original), samples=samples, trials=args.trials, status="measured", correct=True,
                   median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in samples))
        retained[row["name"]] = row
        confirmed.append(row)
        write(journal, list(retained.values()))
        print(f"confirmed mapping {index + 1}/{len(records)} {row['name']}", flush=True)
    # The helper preserves the source checkpoint's protocol, so provide the
    # requested current protocol separately without importing old raw services.
    protocol_path = output / "stage_protocol.json"
    write(protocol_path, dict(protocol=dict(warmup=args.warmup, repeat=args.repeat, trials=args.trials)))
    service_args = SimpleNamespace(build_dir=args.build_dir, output_dir=output, stage_checkpoint=protocol_path)
    guard()
    confirmed, gaps = confirmed_services(confirmed, service_args, profile, output)
    guard()
    write(journal, confirmed)
    write(output / "calibration_device.json", profile)
    if profile.get("stage_service"):
        write(output / "stage_service_profile.json", profile["stage_service"])
    composition_seeds = calibration.composition_seed_records(
        confirmed, output / 'stage_calibration.json', profile)
    composition = calibrate_composition(args.build_dir, output / 'composition_calibration', profile,
        composition_seeds, trials=args.trials, warmup=args.warmup, repeat=args.repeat,
        resume=args.resume, run=run_probe, guard=guard)
    # Selector replay verifies execution, but cannot certify missing model services.
    require_release_ready(profile, composition['summary'], gaps)
    subprocess.run([sys.executable, str(Path(__file__).with_name("stage_cost_model.py")),
                    "--profile", str(output / "calibration_device.json"), "--operator-calibration", str(journal),
                    "--output", str(output / "cost_model.json")], check=True)
    registry = output / "registry.json"
    pending_registry = output / "registry.pending.json"
    operator_result = dict(status="complete", candidates=confirmed,
                           operator_families=sorted({r["samples"][0]["operator"] for r in confirmed}))
    promoted = promote(pending_registry, profile, operator_result)
    released = False
    try:
        guard()
        selector = migration._selector_replay(args, output, {"CUBUTTERFLY_REGISTRY": str(pending_registry)})
        guard()
        if selector.get("status") == "passed":
            pending_registry.replace(registry)
            released = True
    finally:
        if not released:
            promote(pending_registry, profile, operator_result, verified=False)
    populations = {}
    for row in confirmed:
        populations.setdefault(cell_key(row["samples"][0]), []).append(row)
    ranked = {key: ranking(rows, profile) for key, rows in populations.items()}
    write(output / "ranking.json", ranked)
    model = read(output / "cost_model.json")
    result = dict(schema="cubutterfly-local-calibration-v1", status="ready-with-measured-mappings"
                  if selector.get("status") == "passed" else "selector-replay-failed",
                  source_profile=str(source), build_dir=str(args.build_dir), measurement_identity=build_identity,
                  device=profile["device"], compute_capability=profile["compute_capability"],
                  global_memory_bytes=profile["global_memory_bytes"], registry_path=str(registry),
                  registry_promoted_records=promoted, operator_calibration=str(journal),
                  operator_families=operator_result["operator_families"],
                  cost_model=str(output / "cost_model.json"), cost_model_status=model.get("status"),
                  composition_calibration=dict(
                      status=composition['summary']['status'], scope=composition['scope'],
                      path=str(output / 'composition_calibration/composition_calibration.json'),
                      summary=composition['summary']),
                  scheduling_calibration=str(output / 'schedule_calibration.json'),
                  pipeline_calibration=str(output / 'pipeline_schedule_calibration.json'),
                  stage_calibration_mode="confirmed-mappings", service_gaps=gaps, selector_replay=selector,
                  inventory_complete=False, scope="finite confirmed mapping population; not full design-space qualification",
                  hardware_rates_source=str(args.hardware_profile.resolve()) if args.hardware_profile else str(source / "calibration_device.json"),
                  hardware_rates_remeasured=bool(args.hardware_profile),
                  kernel_timing_excludes="compilation, plan construction, transfers and reference checking")
    write(output / "calibration_manifest.json", result)
    print(json.dumps({key: result[key] for key in ("status", "registry_promoted_records", "cost_model_status", "inventory_complete")}))
    return 0 if selector.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
