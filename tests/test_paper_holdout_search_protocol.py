"""CPU-only contracts for the bounded E6/E9 paper protocols."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heldout = _load("heldout_protocol", "paper/tools/heldout_protocol.py")
pool = _load("search_pool_protocol", "paper/tools/search_pool_protocol.py")


@pytest.fixture
def tmp_path(tmpdir):
    """Keep this suite compatible with the repository's older pytest."""
    return Path(str(tmpdir))


def _workload_domain():
    fft = {
        "operator": "fft", "precision": "fp32", "placement": "out-of-place",
        "direction": "forward", "normalization": "none",
    }
    fwht = {**fft, "operator": "fwht"}
    ntt = {
        "operator": "ntt", "precision": "word64", "placement": "out-of-place",
        "direction": "forward", "normalization": "none",
        "modulus": 576460756061519873, "input_order": "natural", "output_order": "natural",
    }
    return [
        {**fft, "logN": 18, "batch": 16},
        {**fwht, "logN": 12, "batch": 1},
        {**fft, "logN": 8, "batch": 1},
        {**ntt, "logN": 12, "batch": 1024},
    ]


def _checkpoint(tmp_path):
    checkpoint = {
        "schema": "cubutterfly-stage-calibration-v1",
        "status": "complete",
        "calibration_status": "complete",
        "identity": {
            "gpu_uuid": "GPU-fixture", "device": "fixture-A100",
            "compute_capability": "8.0", "compile_policy": "research",
        },
        "protocol": {"trials": 3, "warmup": 100, "repeat": 100},
        "model": {"curve_count": 1, "training_groups": 2, "validation_groups": 1},
        "candidates": [{"point": {"operator": "fft", "precision": "fp32", "logN": 18,
                                     "batch": 16, "placement": "out-of-place",
                                     "mapping_json": "{}"}}],
        "records": [
            {"role": "train", "plan_trial_kernel_ms": [1.0],
             "groups": [{"role": "train", "independent": True}]},
            {"role": "validation", "plan_trial_kernel_ms": [1.1],
             "groups": [{"role": "validation", "independent": True}]},
        ],
    }
    path = tmp_path / "stage_calibration.json"
    path.write_text(json.dumps(checkpoint))
    return path


def _prepare(tmp_path):
    domain = tmp_path / "workloads.json"
    domain.write_text(json.dumps({"workloads": _workload_domain()}))
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "protocols"
    heldout.freeze(__import__("argparse").Namespace(
        workloads=domain, stage_checkpoint=checkpoint, output_dir=output,
        workload_id=None, target_uuid="GPU-fixture", python="python3",
        profile=tmp_path / "profile.json", build_dir=tmp_path / "build",
    ))
    pool.freeze(__import__("argparse").Namespace(
        workloads=domain, stage_checkpoint=checkpoint, output_dir=output / "pool",
        workload_id=None, exclude_protocol=output / "heldout_protocol.json",
        target_uuid="GPU-fixture", python="python3", profile=tmp_path / "profile.json",
        build_dir=tmp_path / "build",
    ))
    return output


def test_freeze_audits_stage_roles_and_excludes_probe_plan_times(tmp_path):
    output = _prepare(tmp_path)
    document = json.loads((output / "heldout_protocol.json").read_text())
    assert document["freeze"]["protocol_sha256"] == heldout.protocol_hash(document)
    audit = document["training_inventory_audit"]
    assert audit["record_role_counts"] == {"train": 1, "validation": 1}
    assert audit["probe_plan_timing_rows"] == 2
    assert audit["whole_workload_training_eligible"] is False
    assert audit["training_scope"] == "physical-stage-groups-only"
    assert all(not row["uses_whole_workload_observation"]
               for row in document["prior_predictions"])
    assert all(row["prediction_usable"] is False for row in document["prior_predictions"])
    assert all(row["fallback_score"] > 0 for row in document["prior_predictions"])
    assert document["gpu_measurement"]["expected_full_workload_trials"] == 24
    assert document["gpu_measurement"]["output_path"].endswith("heldout_measurements.json")
    evaluate_command = document["gpu_measurement"]["evaluate_command"]
    assert evaluate_command[evaluate_command.index("--measurements") + 1] == document["gpu_measurement"]["output_path"]
    assert evaluate_command[evaluate_command.index("--profile") + 1].endswith("profile.json")
    assert evaluate_command[evaluate_command.index("--build-dir") + 1].endswith("/build")


def test_holdout_semantics_use_record_samples_and_keep_roles_separate(tmp_path):
    checkpoint = json.loads(_checkpoint(tmp_path).read_text())
    checkpoint["records"][0]["sample"] = {
        "operator": "fft", "precision": "fp32", "logN": 16, "N": 65536,
        "batch": 1, "placement": "out-of-place", "direction": "forward",
        "normalization": "none", "accumulation": "native", "element_stride": 1,
        "batch_stride": 65536,
    }
    checkpoint["records"][1]["sample"] = {
        "operator": "fwht", "precision": "fp32", "logN": 15, "N": 32768,
        "batch": 1, "placement": "out-of-place", "direction": "forward",
        "normalization": "none", "accumulation": "native", "element_stride": 1,
        "batch_stride": 32768,
    }
    checkpoint_path = tmp_path / "sampled_checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    domain = tmp_path / "workloads.json"
    domain.write_text(json.dumps({"workloads": _workload_domain()}))
    output = tmp_path / "protocols"
    heldout.freeze(__import__("argparse").Namespace(
        workloads=domain, stage_checkpoint=checkpoint_path, output_dir=output,
        workload_id=None, target_uuid="GPU-fixture", python="python3",
        profile=tmp_path / "profile.json", build_dir=tmp_path / "build",
    ))
    audit = json.loads((output / "heldout_protocol.json").read_text())["training_inventory_audit"]
    assert audit["stage_training_workloads"] == [{
        "N": 65536, "accumulation": "native", "batch": 1,
        "batch_stride": 65536, "direction": "forward", "element_stride": 1,
        "logN": 16, "normalization": "none", "operator": "fft",
        "placement": "out-of-place", "precision": "fp32",
    }]
    assert audit["stage_validation_workloads"][0]["operator"] == "fwht"
    assert audit["whole_workload_training_workloads"] == []
    assert audit["whole_workload_validation_workloads"] == []


def test_holdout_rejects_sample_semantic_overlap(tmp_path):
    checkpoint = json.loads(_checkpoint(tmp_path).read_text())
    checkpoint["records"][0]["sample"] = {
        "operator": "fft", "precision": "fp32", "logN": 18, "N": 1 << 18,
        "batch": 16, "placement": "out-of-place", "direction": "forward",
        "normalization": "none", "accumulation": "native", "element_stride": 1,
        "batch_stride": 1 << 18,
    }
    checkpoint_path = tmp_path / "overlap_checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    domain = tmp_path / "workloads.json"
    domain.write_text(json.dumps({"workloads": _workload_domain()}))
    with pytest.raises(ValueError, match="overlap checkpoint.records.sample"):
        heldout.freeze(__import__("argparse").Namespace(
            workloads=domain, stage_checkpoint=checkpoint_path,
            output_dir=tmp_path / "protocols", workload_id=None,
            target_uuid="GPU-fixture", python="python3",
            profile=tmp_path / "profile.json", build_dir=tmp_path / "build",
        ))


def test_semantic_key_normalizes_ntt_word_bits_from_runner_sample():
    request = _workload_domain()[3]
    resolved = {**request, "word_bits": 64, "N": 4096, "batch_stride": 4096}
    assert heldout._semantic_key(request) == heldout._semantic_key(resolved)


def test_composition_holdout_keeps_stage_overlap_explicit(tmp_path):
    checkpoint_path = _checkpoint(tmp_path)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["records"][0]["sample"] = _workload_domain()[0]
    checkpoint_path.write_text(json.dumps(checkpoint))
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps({"workloads": _workload_domain()}))
    result = heldout.freeze(__import__("argparse").Namespace(
        workloads=domain, stage_checkpoint=checkpoint_path, output_dir=tmp_path / "composition",
        workload_id=None, target_uuid="GPU-fixture", python="python3", composition_holdout=True,
        profile=tmp_path / "profile.json", build_dir=tmp_path / "build"))
    assert result["metric_population"] == "whole-workload-composition-holdout"
    assert result["composition_holdout_declared"] is True
    assert result["training_inventory_audit"]["heldout_stage_semantic_overlap"]
    assert result["training_inventory_audit"]["whole_workload_holdout_eligible"] is False
    assert result["training_inventory_audit"]["whole_workload_training_workloads"] == []
    assert all(not p["uses_whole_workload_observation"] for p in result["prior_predictions"])


def test_holdout_evaluation_keeps_service_gaps_and_does_not_overwrite_measurements(tmp_path):
    output = _prepare(tmp_path)
    protocol_path = output / "heldout_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    measurements_path = output / "heldout_measurements.json"
    rows = []
    for index, group in enumerate(protocol["heldout_workloads"]):
        for candidate_index, candidate in enumerate(group["candidates"]):
            rows.append({"point": candidate["point"], "status": "measured",
                         "public_plan_ms": float(10 * (index + 1) + candidate_index + 1)})
    measurements_path.write_text(json.dumps({"rows": rows}))
    result_path = output / "heldout_validation.json"
    result = heldout.evaluate(__import__("argparse").Namespace(
        protocol=protocol_path, measurements=measurements_path, output=result_path))
    assert result["status"] == "evaluated-with-gaps"
    assert result["metrics"]["service_gap_count"] == 8
    assert result["metrics"]["top1"] is None
    assert result["used_whole_workload_observations_for_predictions"] is False
    assert json.loads(measurements_path.read_text())["rows"] == rows


def test_search_pool_is_independent_and_replay_counts_failed_seed(tmp_path):
    output = _prepare(tmp_path)
    protocol_path = output / "pool" / "search_pool_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((output / "pool" / "search_pool_manifest.json").read_text())
    assert "stage_checkpoint" not in manifest["selection_inputs"]
    assert manifest["global_optimum_available"] is False
    assert all(len(group["candidates"]) == 16 for group in protocol["workloads"])
    measurements = []
    for group in protocol["workloads"]:
        for index, candidate in enumerate(group["candidates"]):
            measurements.append({
                "point": candidate["point"],
                "status": "failed",
                "public_plan_ms": None,
            })
    measurements_path = output / "pool" / "measurements.json"
    measurements_path.write_text(json.dumps({"rows": measurements}))
    result_path = output / "pool" / "replay.json"
    result = pool.replay(__import__("argparse").Namespace(
        protocol=protocol_path, measurements=measurements_path, output=result_path))
    assert result["status"] == "replayed-with-gaps"
    assert len(result["cells"]) == 2 * 4 * 5 * 3
    assert result["summary"]["budget_includes_failed_seed"] is True
    assert result["summary"]["global_optimum_claim"] is False
    replay_command = protocol["gpu_measurement"]["replay_command"]
    assert replay_command[replay_command.index("--measurements") + 1] == protocol["gpu_measurement"]["output_path"]
    assert replay_command[replay_command.index("--profile") + 1].endswith("profile.json")
    assert replay_command[replay_command.index("--build-dir") + 1].endswith("/build")
    first_budget = [cell for cell in result["cells"] if cell["budget"] == 4 and cell["seed"] == 17]
    assert first_budget and any(not attempt["success"] for cell in first_budget for attempt in cell["attempts"])
    assert any(cell["failed_seed"] == (not cell["attempts"][0]["success"]) for cell in first_budget)
    assert all(cell["attempt_count"] == cell["budget"] for cell in result["cells"])


def test_tampered_protocol_is_rejected_before_replay(tmp_path):
    output = _prepare(tmp_path)
    protocol_path = output / "pool" / "search_pool_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["budgets"] = [4, 8, 32]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(protocol))
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        pool.replay(__import__("argparse").Namespace(
            protocol=tampered, measurements=measurements, output=tmp_path / "out.json"))


def test_local_search_future_fitness_cannot_change_history_prefix():
    workload = _workload_domain()[2]
    candidates = pool.candidate_pool(workload)
    group = {
        "workload_id": "fixture-workload",
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "candidates": candidates,
    }
    observations = {
        item["candidate_id"]: {"observed_ms": float(index + 1), "status": "measured"}
        for index, item in enumerate(candidates)
    }
    baseline = pool._candidate_order(group, "local-search", 17, {}, observations)
    future_id = baseline[5]
    changed = dict(observations)
    changed[future_id] = {"observed_ms": 0.000001, "status": "measured"}
    updated = pool._candidate_order(group, "local-search", 17, {}, changed)
    assert baseline[:5] == updated[:5]


def test_replay_reports_regret_ratio_separately_from_relative_regret():
    workload = _workload_domain()[2]
    candidates = pool.candidate_pool(workload)
    group = {
        "workload_id": "fixture-workload",
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "candidates": candidates,
    }
    order = pool._candidate_order(group, "random", 17, {}, {})
    observations = {
        (group["workload_id"], candidate["candidate_id"]): {
            "observed_ms": float(index + 2), "status": "measured",
        }
        for index, candidate in enumerate(candidates)
    }
    cell = pool._replay_cell(group, "random", 17, 4, {}, observations)
    assert cell["regret_ratio"] >= 1.0
    assert cell["relative_regret"] == pytest.approx(cell["regret_ratio"] - 1.0)
    assert cell["pool_regret"] == cell["relative_regret"]
    assert cell["regret_ratio"] != cell["relative_regret"]


def test_holdout_metrics_do_not_compute_regret_for_unmeasured_prediction():
    workload = _workload_domain()[0]
    candidates = heldout.candidate_set(workload)
    group = {
        "workload_id": "fixture-workload",
        "candidate_ids": [item["candidate_id"] for item in candidates],
    }
    predictions = {
        candidate["candidate_id"]: {
            "predicted_ms": float(index + 1), "prediction_usable": True,
        }
        for index, candidate in enumerate(candidates)
    }
    observed = {
        candidate["candidate_id"]: {
            "observed_ms": None if index == 0 else float(index + 2),
            "status": "missing" if index == 0 else "measured",
        }
        for index, candidate in enumerate(candidates)
    }
    report = heldout._rank_metrics(group, predictions, observed)
    assert report["predicted_order"][0] == candidates[0]["candidate_id"]
    assert report["top1"] is None
    assert report["regret_ratio"] is None
    assert report["relative_regret"] is None


def test_point_signature_preserves_csv_word64_modulus():
    point = {"operator": "ntt", "precision": "word64", "logN": 12,
             "batch": 1024, "modulus": 576460756061519873}
    assert heldout.point_signature(point) == heldout.point_signature(
        {**point, "modulus": "576460756061519873"})
    assert heldout.point_signature(point) != heldout.point_signature(
        {**point, "modulus": "576460756061519872"})


def test_explicit_pool_freezes_two_mappings_and_rejects_semantic_changes(tmp_path):
    workload = {"operator": "fft", "precision": "fp32", "logN": 16,
                "batch": 9, "placement": "out-of-place", "direction": "forward",
                "normalization": "none"}
    points = [heldout.make_point(workload, [8, 8], threads) for threads in (128, 256)]
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"points": points}))
    candidates, source = heldout.explicit_candidate_sets(path, [workload])
    assert len(candidates[heldout.workload_id(workload)]) == 2
    assert source["sha256"] == heldout.sha256_file(path)

    path.write_text(json.dumps({"points": [points[0], points[0]]}))
    with pytest.raises(ValueError, match="repeats"):
        heldout.explicit_candidate_sets(path, [workload])
    path.write_text(json.dumps({"points": [points[0]]}))
    with pytest.raises(ValueError, match="at least two"):
        heldout.explicit_candidate_sets(path, [workload])
    path.write_text(json.dumps({"points": [points[0], {**points[1], "batch": 10}]}))
    with pytest.raises(ValueError, match="outside"):
        heldout.explicit_candidate_sets(path, [workload])


def test_descriptor_identity_queries_only_cubutterfly_bench(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "cubutterfly_bench").write_bytes(b"cubutterfly-fixture")
    cuntt_binary = build_dir / "cuntt_bench"
    cuntt_binary.write_bytes(b"cuntt-fixture")

    checkpoint = {
        "identity": {
            "gpu_uuid": "GPU-fixture",
            "device": "fixture-A100",
            "compute_capability": "8.0",
            "compile_policy": "research",
        }
    }
    protocol = {
        "target": {
            "uuid": "GPU-fixture",
            "device": "fixture-A100",
            "compute_capability": "8.0",
            "compile_policy": "research",
        }
    }
    queried = []

    monkeypatch.setattr(heldout, "_query_gpu_uuid", lambda requested: requested)

    def query_identity(binary, environment=None):
        queried.append(binary.name)
        return {
            "device": "fixture-A100",
            "compute_capability": "8.0",
            "global_memory_bytes": 80 * (1 << 30),
        }

    monkeypatch.setattr(heldout, "_query_benchmark_identity", query_identity)

    descriptor_identity = heldout._descriptor_identity(
        protocol, checkpoint, build_dir, "GPU-fixture"
    )

    assert queried == ["cubutterfly_bench"]
    binaries = descriptor_identity["benchmark_binaries"]
    assert set(binaries) == {"cubutterfly_bench", "cuntt_bench"}
    assert "identity" not in binaries["cuntt_bench"]
    assert binaries["cuntt_bench"]["path"] == str(cuntt_binary.resolve())
    assert binaries["cuntt_bench"]["sha256"] == heldout.sha256_file(cuntt_binary)
