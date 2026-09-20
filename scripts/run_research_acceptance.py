#!/usr/bin/env python3
"""Finite cross-operator acceptance using the existing search and registry.

This is a measured workload-matrix acceptance, not a full hardware-migration
certificate. Model warnings and unavailable baselines remain visible. Each
cell checkpoints search, finalist service calibration, replay and paired trials.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import calibrate_local_hardware as calibration
from calibration_space import command_for
from hardware_registry import canonical_semantics, promote, stable_id
from local_selector_data import select_points
from run_comprehensive_suite import require_exclusive_gpu, visible_gpu
from stage_composition import resolve_batch_services
import stage_cost_model
import stage_service_calibration
import stage_service_model
from verify_local_selector import verify

SCHEMA = "cubutterfly-research-acceptance-v1"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    calibration.write_checkpoint(Path(path), value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_code_revision(document):
    # Preserve analysis provenance without discarding valid raw kernel timings
    # after a numerically equivalent CPU optimization. Numerical model versions
    # and measurement/baseline contracts remain separately guarded identities.
    names = ("run_research_acceptance.py", "calibrate_local_hardware.py", "calibration_space.py",
             "evolutionary_search.py", "stage_cost_model.py", "stage_service_model.py",
             "stage_service_calibration.py", "stage_composition.py", "verify_local_selector.py")
    revision = {name: digest(Path(__file__).with_name(name)) for name in names}
    history = document.setdefault("code_revisions", [])
    if not history or history[-1] != revision:
        history.append(revision)


def run(command, check=True):
    return subprocess.run([str(x) for x in command], text=True, capture_output=True, check=check)


def cell_key(workload):
    semantic = canonical_semantics(workload)
    if workload.get("stage_matrix"):
        semantic["stage_matrices"] = ":".join(format(float(v), ".17g") for v in workload["stage_matrix"].split(","))
    return stable_id(semantic)[:20]


def ranking(rows, profile):
    """Ranking of repeated-confirmed designs, with no in-sample plan-time fit."""
    model = stage_cost_model.fit([], profile)
    candidates, gaps = [], []
    for row in rows:
        if row.get("status") != "measured" or not row.get("correct") or not row.get("samples"):
            continue
        sample = row["samples"][0]
        if sample.get("backend") == "cufft":
            continue
        try:
            estimate = stage_cost_model.predict(sample, model, True)
            predicted = float(estimate["kernel_ms"])
            if not math.isfinite(predicted) or predicted <= 0:
                raise ValueError("nonpositive prediction")
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as error:
            gaps.append({"name": row["name"], "reason": str(error)})
            continue
        candidates.append(dict(name=row["name"], observed_ms=row["median_kernel_ms"],
            predicted_ms=predicted, covered=not estimate.get("whole_plan_fallback_used", True),
            mapping_json=sample.get("mapping_json"), prediction=estimate))
    # Aliases must not count twice as independent competing designs.
    unique = {}
    for row in candidates:
        key = json.dumps(json.loads(row["mapping_json"]), sort_keys=True)
        if key not in unique or row["observed_ms"] < unique[key]["observed_ms"]:
            unique[key] = row
    candidates = list(unique.values())
    result = dict(scope="confirmed candidates only; not exhaustive design-space regret",
                  candidates=candidates, unestimated=gaps, candidate_count=len(candidates))
    if len(candidates) < 2:
        return dict(result, status="insufficient-candidates")
    observed = sorted(candidates, key=lambda r: r["observed_ms"])
    predicted = sorted(candidates, key=lambda r: r["predicted_ms"])
    pairs = list(itertools.combinations(candidates, 2))
    correct = sum((a["observed_ms"] - b["observed_ms"]) *
                  (a["predicted_ms"] - b["predicted_ms"]) > 0 for a, b in pairs)
    return dict(result, status="evaluated", top1=observed[0]["name"] == predicted[0]["name"],
        top2=observed[0]["name"] in {r["name"] for r in predicted[:2]},
        latency_regret=predicted[0]["observed_ms"] / observed[0]["observed_ms"],
        pairwise_correct=correct, pairwise_count=len(pairs),
        observed_order=[r["name"] for r in observed], predicted_order=[r["name"] for r in predicted],
        all_services_covered=all(r["covered"] for r in candidates))


def verify_sample(workload, sample, device):
    if str(sample.get("correct")) != "1":
        raise ValueError("public acceptance correctness failed")
    latency = float(sample.get("kernel_ms", 0))
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("public acceptance timing is invalid")
    actual = canonical_semantics(sample)
    wanted = canonical_semantics(workload)
    if workload.get("stage_matrix"):
        wanted["stage_matrices"] = ":".join(format(float(v), ".17g") for v in workload["stage_matrix"].split(","))
    for field in ("operator", "precision", "logN", "batch", "placement", "direction", "modulus",
                  "normalization", "accumulation", "element_stride", "batch_stride", "stage_matrices",
                  "input_order", "output_order"):
        if field in wanted and actual.get(field) != wanted[field]:
            raise ValueError(f"public semantics differ: {field}: {actual.get(field)} != {wanted[field]}")
    for field in ("device", "compute_capability"):
        if str(sample.get(field)) != str(device[field]):
            raise ValueError(f"public hardware differs: {field}")


_PROBE_INTEGER_FIELDS = ("logN", "batch", "element_stride", "batch_stride", "modulus")


def _normalize_probe_point(point):
    """Serialize benchmark CSV integer fields as exact JSON numbers."""
    normalized = dict(point)
    for field in _PROBE_INTEGER_FIELDS:
        if field in normalized:
            # CSV samples carry these values as strings.  Python's int keeps
            # word64 moduli exact and leaves omitted optional fields omitted.
            normalized[field] = int(normalized[field])
    return normalized


def stage_probe_protocol(args):
    """Preserve the source checkpoint's timing contract during extension.

    Public-plan confirmation counts can differ from independent-stage counts.
    Hard-coded stage warmups would reject a warmed source checkpoint or mix
    incompatible timing states when filling a finalist's missing service.
    """
    protocol = read(args.stage_checkpoint).get("protocol", {})
    result = {}
    for name in ("warmup", "repeat", "trials"):
        value = protocol.get(name)
        if type(value) is not int or value < (0 if name == "warmup" else 1):
            raise ValueError(f"invalid source stage timing protocol: {name}")
        result[name] = value
    return result


def confirmed_services(records, args, profile, cell_dir):
    """Fill only missing physical services of the finite confirmed population."""
    points, gaps = {}, []
    descriptions = {}

    def describe(point):
        key = json.dumps(point, sort_keys=True)
        if key not in descriptions:
            require_exclusive_gpu()
            response = run([args.build_dir / "cubutterfly_stage_microbench", "--describe-only", "--point-json", key], check=False)
            require_exclusive_gpu()
            try:
                descriptions[key] = json.loads(response.stdout)
            except json.JSONDecodeError:
                descriptions[key] = dict(status="unavailable", reason=response.stderr or "stage probe returned no descriptor")
        return descriptions[key]

    model = stage_cost_model.fit([], profile)
    upgraded = copy.deepcopy(records)
    for row in upgraded:
        if row.get("status") != "measured" or not row.get("correct") or not row.get("samples"):
            continue
        original = row["samples"][0]
        if original.get("backend") == "cufft":
            continue
        point = _normalize_probe_point({k: original[k] for k in ("operator", "precision", "logN", "batch", "placement",
            "direction", "normalization", "accumulation", "element_stride", "batch_stride", "modulus",
            "input_order", "output_order", "mapping_json") if k in original})
        if original.get("stage_matrices"):
            point["stage_matrix"] = original["stage_matrices"].replace(":", ",")
        description = describe(point)
        if description.get("status") != "resolved":
            gaps.append(dict(name=row["name"], reason=description.get("reason", "unresolved-stage-descriptor")))
            continue
        actual = description["sample"]
        if actual.get("runtime_fingerprint") != original.get("runtime_fingerprint") or json.loads(actual["mapping_json"]) != json.loads(original["mapping_json"]):
            raise ValueError("confirmed timing and current descriptor differ")
        projection = resolve_batch_services(point, description, describe)
        for sample in row["samples"]:
            sample["execution_groups_json"] = json.dumps(description["groups"])
            sample["stage_service_projection"] = projection
            sample["descriptor_source"] = "actual-stage-probe"
        views = ([projection["full"], projection.get("tail")] if projection.get("status") == "resolved"
                 else [{"sample": actual, "groups": description["groups"]}])
        for view in views:
            if not view:
                continue
            sample, groups = view["sample"], view["groups"]
            if not groups or not all(g.get("independent") for g in groups):
                gaps.append(dict(name=row["name"], reason="independent-stage-adapter-unavailable"))
                continue
            prediction = stage_cost_model.predict({**sample, "execution_groups_json": groups}, model, True)
            if not prediction.get("whole_plan_fallback_used"):
                continue
            service_point = {**point, "batch": int(sample["batch"]), "mapping_json": sample["mapping_json"]}
            points[json.dumps(service_point, sort_keys=True)] = service_point
    write(cell_dir / "service_requests.json", dict(points=list(points.values()), gaps=gaps))
    if points:
        profile["stage_service"] = stage_service_calibration.calibrate(
            args.build_dir / "cubutterfly_stage_microbench", args.output_dir / "stage_calibration.json",
            profile, list(points.values()), lambda command: run(command, check=False), require_exclusive_gpu,
            mode="full", **stage_probe_protocol(args))
        write(args.output_dir / "calibration_device.json", profile)
        write(args.output_dir / "stage_service_profile.json", profile["stage_service"])
    write(cell_dir / "confirmed_records.json", upgraded)
    return upgraded, gaps


def summarize(document):
    completed = [c for c in document["cells"] if c.get("status") == "complete"]
    unavailable = [dict(id=c.get("id"), workload=c.get("workload"), evidence=c.get("unavailability"))
                   for c in document["cells"] if c.get("status") == "unavailable"]
    ranked = [c["ranking"] for c in document["cells"] if c.get("ranking", {}).get("status") == "evaluated"]
    ratios, groups, cells = {}, {}, []
    for cell in completed:
        measurements = cell.get("measurements", [])
        names = {r["name"] for r in measurements}
        latencies = {name: statistics.median(float(r["sample"]["kernel_ms"]) for r in measurements if r["name"] == name) for name in names}
        cell_ratios = {}
        for name, latency in latencies.items():
            if name != "cuButterfly":
                speedup = latency / latencies["cuButterfly"]
                ratios.setdefault(name, []).append(speedup)
                workload = cell.get("workload", {})
                group = "/".join((workload.get("operator", "unknown"), workload.get("precision", "unknown"), name))
                groups.setdefault(group, []).append(speedup)
                cell_ratios[name] = speedup
        cells.append(dict(id=cell.get("id"), workload=cell.get("workload"), median_ms=latencies,
                          throughput_vs_baseline=cell_ratios, unavailable=cell.get("baseline_plan", {}).get("unavailable", [])))
    def statistics_for(values):
        return dict(paired_cells=len(values), geometric_mean_speedup=math.exp(statistics.mean(math.log(v) for v in values)),
                    minimum_speedup=min(values), faster_cells=sum(v >= 1 for v in values))
    return dict(schema=SCHEMA, status=document["status"], expected_cells=len(document["cells"]),
        completed_cells=len(completed), full_migration_qualified=False,
        all_workloads_measured=len(completed) == len(document["cells"]),
        unavailable_cells=unavailable,
        ranking=dict(cells=len(ranked), top1=sum(r["top1"] for r in ranked),
            top2=sum(r["top2"] for r in ranked),
            pairwise_correct=sum(r.get("pairwise_correct", 0) for r in ranked),
            pairwise_count=sum(r.get("pairwise_count", 0) for r in ranked),
            all_services_covered_cells=sum(r.get("all_services_covered", False) for r in ranked),
            mean_latency_regret=statistics.mean(r["latency_regret"] for r in ranked) if ranked else None,
            worst_latency_regret=max((r["latency_regret"] for r in ranked), default=None)),
        baselines={name: statistics_for(values) for name, values in ratios.items()},
        operator_baselines={name: statistics_for(values) for name, values in groups.items()}, cells=cells,
        limitations=document["limitations"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--pipeline-calibration", type=Path, required=True)
    parser.add_argument("--schedule-calibration", type=Path, required=True)
    parser.add_argument("--mapping-seeds", type=Path, action="append", default=[])
    parser.add_argument("--search-budget", type=int, default=16)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--seed-budget", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--verify-batches", type=int, default=0)
    parser.add_argument("--mode", choices=("prepare", "search", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-unavailable", action="store_true",
                        help="record explicit capability rejections and visit later cells; never count them as measured")
    parser.add_argument("--vkfft-binary", type=Path)
    parser.add_argument("--fht-python", type=Path)
    parser.add_argument("--gpuntt-binary", type=Path)
    args = parser.parse_args(argv)
    if min(args.search_budget, args.finalists, args.trials, args.repeat) < 1 or min(args.seed_budget, args.warmup, args.verify_batches) < 0:
        parser.error("invalid search or timing budget")
    for field in ("build_dir", "output_dir", "workloads", "profile", "stage_checkpoint", "pipeline_calibration", "schedule_calibration",
                  "vkfft_binary", "fht_python", "gpuntt_binary"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).expanduser().resolve())
    args.mapping_seeds = [p.expanduser().resolve() for p in args.mapping_seeds]
    return args


def prepare(args):
    specification = read(args.workloads)
    if specification.get("schema") != "cubutterfly-install-search-v1" or not specification.get("workloads"):
        raise ValueError("invalid acceptance matrix")
    workloads = specification["workloads"]
    if len({cell_key(w) for w in workloads}) != len(workloads):
        raise ValueError("duplicate semantic cells")
    device = next(csv.DictReader(io.StringIO(run([args.build_dir / "cubutterfly_bench", "--device-identity"]).stdout)))
    gpu = next(csv.reader(io.StringIO(run(["nvidia-smi", "-i", visible_gpu(), "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"]).stdout), skipinitialspace=True))
    device.update(uuid=gpu[0], driver=gpu[1])
    identity = dict(device=device, compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto"),
        workloads_sha256=digest(args.workloads), profile_sha256=digest(args.profile),
        source_stage_sha256=digest(args.stage_checkpoint),
        cost_model_version=stage_cost_model.VERSION,
        service_model_version=stage_service_model.VERSION,
        binaries={name:digest(args.build_dir / name) for name in ("cubutterfly_bench", "cuntt_bench", "cubutterfly_stage_microbench", "cubutterfly_plan_probe")},
        mapping_seeds=[dict(path=str(p),sha256=digest(p)) for p in args.mapping_seeds],
        protocol={key:getattr(args,key) for key in ("search_budget", "finalists", "seed_budget", "trials", "warmup", "repeat", "verify_batches")})
    output = args.output_dir / "acceptance.json"
    if output.exists():
        if not args.resume:
            raise ValueError("acceptance journal exists; use --resume")
        document = read(output)
        if document.get("identity") != identity:
            raise ValueError("acceptance target/build/matrix/protocol changed")
        return document
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = read(args.profile)
    for field in ("device", "compute_capability", "global_memory_bytes"):
        if str(profile.get(field)) != str(device.get(field)):
            raise ValueError(f"calibration profile differs from target: {field}")
    from schedule_cost_model import calibrate as schedule
    from pipeline_schedule_model import calibrate as pipeline
    profile["scheduling"] = schedule(args.build_dir / "cubutterfly_schedule_microbench", args.schedule_calibration, profile, run, require_exclusive_gpu)
    profile["pipeline_scheduling"] = pipeline(args.build_dir / "cubutterfly_pipeline_microbench", args.pipeline_calibration, profile, run, require_exclusive_gpu)
    profile["stage_service"] = stage_service_calibration.calibrate(args.build_dir / "cubutterfly_stage_microbench",
        args.output_dir / "stage_calibration.json", profile, [], run, require_exclusive_gpu,
        mode="full", **stage_probe_protocol(args), import_checkpoints=[args.stage_checkpoint])
    write(args.output_dir / "calibration_device.json", profile)
    write(args.output_dir / "stage_service_profile.json", profile["stage_service"])
    document = dict(schema=SCHEMA, identity=identity, scope=specification.get("scope"), status="prepared",
        registry_path=str(args.output_dir / "registry.json"),
        limitations=["Finite candidate screening, not exhaustive design-space or full migration qualification.",
            "Ranking is evaluated over confirmed candidates; stage and composition coverage remain explicit.",
            *specification.get("limitations", [])],
        cells=[dict(id=cell_key(w),workload=w,status="pending",measurements=[]) for w in workloads])
    write(output, document)
    return document


def capability_rejections(records):
    """Recognize explicit capability errors, preserving crashes/incorrectness as failures."""
    if not records:
        return None
    evidence = []
    for row in records:
        errors = row.get("errors", [])
        if row.get("status") != "unavailable" or row.get("correct") or row.get("samples") or not errors:
            return None
        if not all(isinstance(error, str) and any(phrase in error.lower() for phrase in
                   ("currently supported only", "currently requires", "currently supports only"))
                   for error in errors):
            return None
        evidence.append(dict(name=row.get("name"), errors=errors))
    return dict(scope="attempted candidates only; not proof of theoretical infeasibility",
                attempted_candidates=len(records), capability_rejections=evidence)


def main(argv=None):
    args = parse_args(argv)
    document = prepare(args)
    os.environ["CUBUTTERFLY_REGISTRY"] = document["registry_path"]
    record_code_revision(document)
    continue_unavailable = getattr(args, "continue_unavailable", False)
    if continue_unavailable:
        document["continuation_policy"] = "continue explicit capability rejections; unavailable cells remain unmeasured"

    def save():
        write(args.output_dir / "acceptance.json", document)
        write(args.output_dir / "summary.json", summarize(document))

    if args.mode == "prepare":
        save()
        print(f"prepared {len(document['cells'])} cells: {args.output_dir}", flush=True)
        return 0
    # Older external harnesses have their own linkage and package environment.
    # Freeze them separately so a completed search can acquire baseline paths
    # later, while an interrupted comparison cannot mix changed libraries.
    if args.mode == "all":
        from research_baselines import baseline_identity
        identity = baseline_identity(dict(build_dir=args.build_dir, vkfft_binary=args.vkfft_binary,
            fht_python=args.fht_python, gpuntt_binary=args.gpuntt_binary))
        bind_comparison_identity(document, identity)
    document["status"] = "running"
    document.pop("last_error", None)
    save()
    try:
        for index, cell in enumerate(document["cells"]):
            if cell["status"] == "complete" or (args.mode == "search" and cell["status"] == "search-complete"):
                continue
            if continue_unavailable and cell["status"] == "unavailable":
                continue
            require_exclusive_gpu()
            cell_dir = args.output_dir / "cells" / cell["id"]
            cell_dir.mkdir(parents=True, exist_ok=True)
            profile = read(args.output_dir / "calibration_device.json")
            if cell["status"] not in ("search-complete", "comparing"):
                print(f"[{index + 1}/{len(document['cells'])}] search {cell['workload']}", flush=True)
                write(cell_dir / "workloads.json", dict(schema="cubutterfly-install-search-v1",scope=document["scope"],workloads=[cell["workload"]]))
                # The proposal model stays fixed across resumes of this cell.
                # New curves affect the next cell and the finalist audit.
                if not (cell_dir / "calibration_device.json").exists():
                    write(cell_dir / "calibration_device.json", profile)
                    write(cell_dir / "stage_service_profile.json", profile["stage_service"])
                search_args = SimpleNamespace(build_dir=args.build_dir, search_workloads=cell_dir / "workloads.json",
                    resume_search=True, search_budget=args.search_budget, search_finalists=args.finalists,
                    search_seconds=0, compile_seconds=0, operator_trials=args.trials, operator_warmup=args.warmup,
                    operator_repeat=args.repeat, verify_batches=args.verify_batches, seed_budget=args.seed_budget,
                    mapping_seeds=args.mapping_seeds, cost_model="staged", search_strategy="evolutionary",
                    stage_calibration="bounded", search_only=True)
                result = calibration.run_operator_calibration(search_args, cell_dir / "operator_calibration.json", True)
                records = result["candidates"]
                cell["ranking_before_service_extension"] = ranking(records, read(cell_dir / "calibration_device.json"))
                records, gaps = confirmed_services(records, args, profile, cell_dir)
                cell["ranking"] = ranking(records, profile)
                cell["service_gaps"] = gaps
                winners = select_points({"candidates": records})
                if len(winners) != 1:
                    rejection = capability_rejections(records) if not winners and continue_unavailable else None
                    if rejection is not None:
                        cell.update(status="unavailable", unavailability=rejection)
                        save()
                        print(f"acceptance {index+1}/{len(document['cells'])}: unavailable; capability errors retained", flush=True)
                        continue
                    raise ValueError(f"expected one confirmed semantic winner, got {len(winners)}")
                promote(Path(document["registry_path"]), profile, {"candidates": winners})
                replay = verify(args.build_dir, winners, cell_dir / "selector_replay.json")
                cell.update(selected=winners[0], selector_replay=replay, status="search-complete")
                save()
            if args.mode == "all":
                compare_cell(cell, args, document, save)
            print(f"acceptance {index+1}/{len(document['cells'])} {cell['workload']}: {cell['status']}", flush=True)
        document["status"] = ("complete-with-unavailable-cells"
                              if any(c["status"] == "unavailable" for c in document["cells"])
                              else "complete-measured-matrix" if args.mode == "all" else "search-complete")
        save()
    except (Exception, KeyboardInterrupt) as error:
        document["status"] = "interrupted"
        document["last_error"] = str(error) or type(error).__name__
        save()
        raise
    print(json.dumps(summarize(document), indent=2), flush=True)
    return 0


def bind_comparison_identity(document, identity):
    previous = document.get("comparison_identity")
    if previous is not None and previous != identity:
        raise ValueError("baseline binaries/packages/adapters changed; use a separate comparison output")
    document["comparison_identity"] = identity


def compare_cell(cell, args, document, save):
    # External adapters own library-specific semantics; the orchestrator owns
    # exclusive timing, repeated trials, identities and the resumable journal.
    from research_baselines import available_baselines, run_baseline
    paths = dict(build_dir=args.build_dir, vkfft_binary=args.vkfft_binary,
                 fht_python=args.fht_python, gpuntt_binary=args.gpuntt_binary)
    protocol = {key:getattr(args,key) for key in ("trials", "warmup", "repeat", "verify_batches")}
    baseline_plan = available_baselines(cell["workload"], paths)
    cell["baseline_plan"] = baseline_plan
    specs = baseline_plan["available"]
    cell["status"] = "comparing"
    complete = {(r["name"],r["trial"]) for r in cell["measurements"]}
    for trial in range(args.trials):
        choices = [dict(name="cuButterfly"), *specs]
        if trial % 2:
            choices.reverse()
        for spec in choices:
            name = spec["name"]
            if (name,trial) in complete:
                continue
            require_exclusive_gpu()
            if name == "cuButterfly":
                workload = cell["workload"]
                binary = args.build_dir / ("cuntt_bench" if workload["operator"] == "ntt" else "cubutterfly_bench")
                command = command_for(binary, workload) + ["--auto-select", "--warmup",str(args.warmup),
                    "--repeat",str(args.repeat),"--verify","--verify-batches",str(args.verify_batches),"--csv"]
                sample = next(csv.DictReader(io.StringIO(run(command).stdout)))
                verify_sample(workload, sample, document["identity"]["device"])
                selected = cell["selected"]
                expected = selected["samples"][0]
                if sample.get("selected_implementation") != selected["name"] or json.loads(sample["mapping_json"]) != json.loads(expected["mapping_json"]) or sample.get("runtime_fingerprint") != expected.get("runtime_fingerprint"):
                    raise ValueError("public selector differs from confirmed winner")
                measurement = dict(sample=sample,command=command,binary_sha256=digest(binary),correct=True)
            else:
                measurement = run_baseline(cell["workload"], spec, protocol, paths)
                if not measurement.get("correct"):
                    raise ValueError(f"baseline correctness failed: {name}")
                # The GPU-NTT legacy CSV has no device columns; the target UUID
                # and inherited CUDA_VISIBLE_DEVICES are recorded by the run.
                for field in ("device", "compute_capability"):
                    observed = measurement["sample"].get(field)
                    if observed is not None and str(observed) != str(document["identity"]["device"][field]):
                        raise ValueError(f"baseline hardware differs: {name}/{field}")
            require_exclusive_gpu()
            cell["measurements"].append(dict(name=name,trial=trial,**measurement))
            save()
    cell["status"] = "complete"
    save()


if __name__ == "__main__":
    raise SystemExit(main())
