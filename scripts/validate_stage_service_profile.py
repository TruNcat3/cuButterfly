#!/usr/bin/env python3
"""Compare independent stage costs with freshly measured public plan execution.

This diagnostic consumes an explicit stage checkpoint, never a wider candidate
inventory. It preserves the checkpoint's numeric semantics, mapping and timing
counts. It does not promote mappings or certify an entire hardware migration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import statistics
import subprocess

from calibration_space import command_for
from run_comprehensive_suite import require_exclusive_gpu
from stage_service_calibration import _load_checkpoint, _normal_point, _point_static_key
import stage_cost_model
import stage_service_model
from stage_composition import resolve_batch_services


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--points", type=pathlib.Path,
                        help="optional original requests, including smaller batches merged into a curve")
    parser.add_argument("--reuse-public-trials", type=pathlib.Path,
                        help="refit against existing public trials with identical binaries/commands/protocol")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed compatible rows in --output and measure remaining requests")
    parser.add_argument("--pipeline-calibration", type=pathlib.Path,
                        help="calibrate/reuse the independent BatchPipeline scheduling floor at this path")
    args = parser.parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise ValueError("stage checkpoint does not exist")
    probe = args.build_dir / "cubutterfly_stage_microbench"
    if hashlib.sha256(probe.read_bytes()).hexdigest() != checkpoint["identity"]["binary_sha256"]:
        raise ValueError("stage probe binary changed since calibration")
    if os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto") != checkpoint["identity"].get("compile_policy"):
        raise ValueError("stage checkpoint compile policy differs")
    # Raw compatible trials remain useful when curve-key semantics change.
    # Never validate against a cached profile produced by an older model.
    service = stage_service_model.build_profile(checkpoint.get("records", []), checkpoint.get("identity"))
    if not service or not service.get("curves"):
        raise ValueError("checkpoint has no measured stage curves")
    profile = json.loads(args.profile.read_text())
    identity_result = subprocess.run([str(args.build_dir / "cubutterfly_bench"), "--device-identity"],
                                     text=True, capture_output=True, check=True)
    identity = next(csv.DictReader(io.StringIO(identity_result.stdout)))
    for key in ("device", "compute_capability", "global_memory_bytes"):
        if str(identity.get(key)) != str(checkpoint["identity"].get(key)):
            raise ValueError(f"stage checkpoint belongs to a different target: {key}")
        if str(identity.get(key)) != str(profile.get(key)):
            raise ValueError(f"capability profile belongs to a different target: {key}")
    profile["stage_service"] = service
    if args.pipeline_calibration:
        from pipeline_schedule_model import calibrate
        profile["pipeline_scheduling"] = calibrate(
            args.build_dir / "cubutterfly_pipeline_microbench", args.pipeline_calibration, profile,
            lambda command: subprocess.run(command, text=True, capture_output=True, check=True),
            require_exclusive_gpu)
    model = stage_cost_model.fit([], profile)
    protocol = checkpoint["protocol"]
    warmup, repeat, trials = (int(protocol[key]) for key in ("warmup", "repeat", "trials"))
    rows, training = [], []
    cached_rows = {}
    cache_source = args.reuse_public_trials or (args.output if args.resume and args.output.exists() else None)
    if cache_source:
        cached_report = json.loads(cache_source.read_text())
        allowed_status = {"complete-diagnostic", "running"} if args.resume else {"complete-diagnostic"}
        if cached_report.get("hardware") != identity or cached_report.get("status") not in allowed_status:
            raise ValueError("public trial report is incomplete or belongs to a different target")
        cached_rows = {json.dumps(row["point"], sort_keys=True): row for row in cached_report["rows"]}
    report = dict(schema="cubutterfly-stage-plan-validation-v1", scope="explicit requests against measured stage curves",
                  hardware=identity,
                  compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto"),
                  checkpoint=str(args.checkpoint.resolve()), status="running", rows=rows)
    if cache_source:
        report["public_trial_source"] = str(cache_source.resolve())
    if args.pipeline_calibration:
        report["pipeline_calibration"] = str(args.pipeline_calibration.resolve())
        report["pipeline_scheduling"] = profile["pipeline_scheduling"]

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    descriptions = {}

    def describe(point):
        key = json.dumps(point, sort_keys=True)
        if key not in descriptions:
            require_exclusive_gpu()
            result = subprocess.run([str(probe), "--describe-only", "--point-json", key],
                                    text=True, capture_output=True, check=True)
            require_exclusive_gpu()
            descriptions[key] = json.loads(result.stdout)
        return descriptions[key]

    candidates = checkpoint.get("candidates", [])
    requests = [(candidate, candidate["point"]) for candidate in candidates]
    if args.points:
        requested = json.loads(args.points.read_text())
        requested = requested.get("points", []) if isinstance(requested, dict) else requested
        by_alias = {alias: candidate for candidate in candidates for alias in candidate.get("aliases", [])}
        requests = []
        for point in requested:
            candidate = by_alias.get(_point_static_key(_normal_point(point)))
            requests.append((candidate, point))
    for candidate, requested_point in requests:
        requested_batch = int(requested_point["batch"])
        records = [r for r in checkpoint["records"] if candidate and r.get("status") == "measured"
                   and r.get("candidate_key") == candidate["key"]
                   and int(r["batch"]) == requested_batch]
        if records:
            record = records[-1]
        else:
            # A merged request can be an interpolated load. Resolve its real
            # grid/resources without inventing a scaled descriptor or adding
            # a stage training point during whole-plan validation.
            record = describe(requested_point)
            if record.get("status") != "resolved":
                raise ValueError("requested load has no resolved physical descriptor")
        sample = record["sample"]
        service_projection = resolve_batch_services(requested_point, record, describe)
        point = {key: sample[key] for key in
                 ("operator", "precision", "logN", "batch", "mapping_json", "placement",
                  "direction", "normalization", "accumulation", "element_stride", "batch_stride",
                  "modulus", "input_order", "output_order", "stage_matrix") if key in sample}
        binary = args.build_dir / ("cuntt_bench" if point["operator"] == "ntt" else "cubutterfly_bench")
        command = command_for(binary, point) + ["--warmup", str(warmup), "--repeat", str(repeat),
                                                "--verify", "--verify-batches", "0", "--csv"]
        cached = cached_rows.get(json.dumps(point, sort_keys=True))
        samples = []
        binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if cached or (args.reuse_public_trials and not args.resume):
            if not cached or cached.get("command") != command or cached.get("binary_sha256") != binary_digest:
                raise ValueError("cached public trial mapping, protocol or binary differs")
            samples = cached.get("samples", [])
            if len(samples) != trials or any(s.get("correct") != "1" or
                    s.get("runtime_fingerprint") != sample.get("runtime_fingerprint") or
                    json.loads(s["mapping_json"]) != json.loads(sample["mapping_json"]) for s in samples):
                raise ValueError("cached public trials are incomplete or incompatible")
        for _ in range(trials - len(samples)):
            require_exclusive_gpu()
            result = subprocess.run(command, text=True, capture_output=True)
            require_exclusive_gpu()
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            measured = list(csv.DictReader(io.StringIO(result.stdout)))
            if len(measured) != 1 or measured[0].get("correct") != "1":
                raise ValueError("public benchmark failed full-batch correctness")
            actual = measured[0]
            if actual.get("runtime_fingerprint") != sample.get("runtime_fingerprint"):
                raise ValueError("benchmark and stage probe builds/policies differ")
            if json.loads(actual["mapping_json"]) != json.loads(sample["mapping_json"]):
                raise ValueError("benchmark and stage probe resolved different mappings")
            samples.append(actual)
        # The NTT CSV uses word_bits/inverse and omits the generic operator and
        # precision columns. Preserve resolved semantics when joining formats.
        actual = {**point, **samples[0], "execution_groups_json": json.dumps(record["groups"])}
        if service_projection.get("status") != "not-applicable":
            actual["stage_service_projection"] = service_projection
        prediction = stage_cost_model.predict(actual, model, explain=True)
        observed = statistics.median(float(s["kernel_ms"]) for s in samples)
        stage_sum = (sum(statistics.median(g["trial_kernel_ms"]) for g in record["groups"])
                     if all(g.get("trial_kernel_ms") for g in record["groups"]) else None)
        row = dict(point=point, status="measured", command=command,
                   binary_sha256=binary_digest,
                   physical_groups=record["groups"], public_trials_reused=bool(cached),
                   samples=samples, public_plan_ms=observed,
                   probe_plan_ms=(statistics.median(record["plan_trial_kernel_ms"])
                                  if record.get("plan_trial_kernel_ms") else None),
                   stage_sum_ms=stage_sum, predicted_ms=prediction["kernel_ms"],
                   relative_error=abs(prediction["kernel_ms"] / observed - 1),
                   covered=not prediction["whole_plan_fallback_used"],
                   missing_group_count=len(prediction.get("unmeasured_groups", [])))
        if service_projection.get("status") != "not-applicable":
            row["stage_service_projection"] = service_projection
        row["prediction"] = prediction
        rows.append(row)
        training.append(dict(name=f"candidate-{len(rows)}", sample=actual, median_kernel_ms=observed))
        save()
        print(json.dumps({key: row[key] for key in
                          ("public_plan_ms", "probe_plan_ms", "stage_sum_ms", "predicted_ms",
                           "relative_error", "covered")}), flush=True)
    report["model_report"] = stage_cost_model.report(training, profile)
    report["status"] = "complete-diagnostic"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
