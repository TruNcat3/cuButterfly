"""Synthetic CPU fixtures for the runtime static-ranking bridge protocol."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rank_static_candidates as ranking
import stage_cost_model as costs
from calibrate_local_hardware import bind_static_model_identity


HARDWARE = dict(device="synthetic CPU fixture", compute_capability="8.0", memory_bytes=1024,
    sm_count=10, max_threads_per_block=1024, max_threads_per_sm=2048, max_blocks_per_sm=32,
    registers_per_sm=65536, registers_per_block=65536, shared_bytes_per_sm=163840,
    shared_bytes_per_block=163840, cooperative_launch=True)
PROFILE = dict(schema="cubutterfly-hardware-profile-v1", device=HARDWARE["device"],
    compute_capability="8.0", global_memory_bytes=1024, sm_count=10,
    provenance={"fixture": True}, capabilities=dict(global_feedback_bytes_per_second=1e12,
        equivalent_butterflies_per_second=1e11, interstage_shared_bytes_per_second=1e13,
        cta_barriers_per_second=1e9))


def sample(stages=(4,), compiled=False):
    return dict(operator="fft", precision="fp32", accumulation="native", logN=12, batch=4,
        direction="forward", normalization="none", mapping_json=json.dumps({
            "backend": "shared-iterative", "stage_overlap": False}),
        execution_groups_json=json.dumps([dict(stage_count=s, grid_ctas=1024,
            threads=64, core="scalar", live_shared_bytes=2**s*16,
            compiler_resources_known=compiled, compiler_registers_per_thread=32)
            for s in stages]))


def write(path, document):
    path.write_text(json.dumps(document))


def request(tmp_path, samples=None):
    write(tmp_path / "hardware_profile.json", PROFILE)
    return dict(schema=ranking.REQUEST_SCHEMA, hardware=copy.deepcopy(HARDWARE),
        runtime_fingerprint="fixture-build:research", profile_dir=str(tmp_path),
        candidates=[dict(index=i, sample=s) for i, s in enumerate(samples or [sample()])])


def model_fixture(value):
    profile = dict(PROFILE, hardware=HARDWARE)
    rows, records = [], []
    prior = costs.fit([], profile)
    for batch in (1, 4, 8):
        for alternative in (0, 1):
            observed = dict(value, batch=batch, backend="shared-iterative",
                runtime_fingerprint="fixture-build:research", warmup="3", repeat="20")
            name = f"synthetic-fixture-{batch}-{alternative}"
            latency = costs.predict(observed, prior)
            rows.append(dict(name=name, sample=observed, median_kernel_ms=latency))
            records.append(dict(name=name, status="measured", correct=True,
                                samples=[observed], median_kernel_ms=latency))
    model = costs.report(rows, profile)
    assert model["usable_for_unmeasured_ranking"] is True
    assert "status" not in model["validation"]
    return bind_static_model_identity(model, {"candidates": records})


@pytest.mark.parametrize("compiled", [False, True])
def test_prior_matches_existing_predictor_and_orders_stably(tmp_path, compiled):
    samples = [sample((4, 8), compiled), sample((4,), compiled), sample((4,), compiled)]
    req = request(tmp_path, samples)
    result = ranking.rank(req)
    profile = dict(PROFILE, hardware=HARDWARE)
    expected = [costs.predict(dict(s, backend="shared-iterative"), costs.fit([], profile)) for s in samples]
    assert [item["index"] for item in result["ranked"]] == [1, 2, 0]
    for item in result["ranked"]:
        assert item["predicted_kernel_ms"] == expected[item["index"]]
        assert item["confidence"] == "hardware-rate-prior"
    assert result["mode"] == "hardware-rate-prior"
    assert "static cost model file is absent:" in result["reason"]
    assert str(tmp_path / "cost_model.json") in result["reason"]
    assert not result["rejected"] and not result["unranked"]


def test_calibration_device_precedence_and_request_caps_override(tmp_path):
    req = request(tmp_path)
    write(tmp_path / "calibration_device.json", dict(PROFILE, sm_count=1,
                                                   hardware={"max_threads_per_block": 1}))
    result = ranking.rank(req)
    assert result["profile_source"].endswith("calibration_device.json")
    assert result["ranked"]


@pytest.mark.parametrize("filename", ["hardware_profile.json", "calibration_device.json"])
def test_profile_mismatch_is_explicit_error(tmp_path, filename):
    req = request(tmp_path)
    write(tmp_path / filename, dict(PROFILE, global_memory_bytes=2048))
    with pytest.raises(ValueError, match="identity mismatch"):
        ranking.rank(req)


def test_missing_profile_keeps_infeasible_reasons(tmp_path):
    req = request(tmp_path, [sample(), sample()])
    req["profile_dir"] = None
    groups = json.loads(req["candidates"][1]["sample"]["execution_groups_json"])
    groups[0]["threads"] = 2048
    req["candidates"][1]["sample"]["execution_groups_json"] = groups
    result = ranking.rank(req)
    assert result["mode"] == "unavailable"
    assert result["unranked"][0]["index"] == 0
    assert result["rejected"][0]["index"] == 1
    assert "max_threads_per_block" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("change,reason", [
    ({"version": "physical-stage-composition-v6"}, "stale"),
    ({"usable_for_unmeasured_ranking": False}, "not validated"),
    ({"status": "calibrated-local-warning"}, "not validated"),
    ({"runtime_fingerprint": "wrong-build"}, "fingerprint"),
    ({"protocol": None}, "protocol"),
    ({"parameters": {"bad": {"service": -1}}}, "coefficients"),
])
def test_ineligible_model_never_uses_coefficients(tmp_path, change, reason):
    req = request(tmp_path)
    model = model_fixture(sample())
    req["timing_protocol"] = "direct-launch"
    model.update(change)
    write(tmp_path / "cost_model.json", model)
    result = ranking.rank(req)
    assert result["mode"] == "hardware-rate-prior"
    assert reason in result["reason"]
    (tmp_path / "cost_model.json").unlink()
    assert result["ranked"][0]["predicted_kernel_ms"] == ranking.rank(req)["ranked"][0]["predicted_kernel_ms"]


def test_eligible_model_and_unknown_protocol_fallback(tmp_path):
    value = sample()
    req = request(tmp_path, [value])
    model = model_fixture(value)
    write(tmp_path / "cost_model.json", model)
    assert "unspecified" in ranking.rank(req)["reason"]
    req["timing_protocol"] = {"timing": "direct-launch"}
    result = ranking.rank(req)
    assert result["mode"] == "validated-static-model"
    assert result["ranked"][0]["predicted_kernel_ms"] == costs.predict(
        dict(value, backend="shared-iterative"), model)


def test_outside_model_coverage_uses_uncalibrated_prior(tmp_path):
    req = request(tmp_path, [sample((8,))])
    model = model_fixture(sample())
    req["timing_protocol"] = "direct-launch"
    write(tmp_path / "cost_model.json", model)
    result = ranking.rank(req)
    assert result["mode"] == "hardware-rate-prior"
    assert "coverage" in result["ranked"][0]["reason"]


@pytest.mark.parametrize("operator", ["fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2"])
def test_one_chunk_reuses_serial_cost_without_requiring_concurrency_validation(tmp_path, operator):
    serial = sample((4, 8))
    serial["operator"] = operator
    serial["precision"] = "word64" if operator == "ntt" else "fp32"
    serial["mapping_json"] = dict(backend="shared-iterative", stage_overlap=False, batch_tile_count=1)
    overlapped = copy.deepcopy(serial)
    overlapped["mapping_json"].update(stage_overlap=True, batch_tile_count=4)
    req = request(tmp_path, [serial, overlapped])
    req["timing_protocol"] = "direct-launch"
    fitted = model_fixture(serial)
    fitted.pop("composition_validation", None)
    write(tmp_path / "cost_model.json", fitted)
    frozen = copy.deepcopy(req)
    result = ranking.rank(req)
    assert req == frozen
    assert not result["rejected"] and not result["unranked"]
    rows = {row["index"]: row for row in result["ranked"]}
    assert rows[0]["confidence"] == rows[1]["confidence"] == "validated-static-model"
    assert rows[1]["predicted_kernel_ms"] == rows[0]["predicted_kernel_ms"]
    assert rows[1]["service_view"] == "equivalent-serial-single-chunk"
    assert rows[1]["stage_overlap_effective"] is False


def test_multiple_chunks_still_require_concurrency_validation(tmp_path):
    serial = sample((4, 8))
    req = request(tmp_path, [dict(serial, stage_overlap=True, batch_tile_count=2)])
    req["timing_protocol"] = "direct-launch"
    fitted = model_fixture(serial)
    fitted.pop("composition_validation", None)
    write(tmp_path / "cost_model.json", fitted)
    result = ranking.rank(req)
    row = result["ranked"][0]
    assert row["confidence"] == "hardware-rate-prior"
    assert "overlap composition is unvalidated" in row["reason"]
    assert "service_view" not in row


def test_single_chunk_service_view_preserves_group_local_resident_identity():
    original = ranking._sample(dict(sample((8, 8, 4)), batch=16,
        mapping_json=dict(backend="shared-iterative", stage_overlap=True, batch_tile_count=16,
                          local_stage_partitions=[[2, 2, 2, 2], [2, 2, 2, 2], []], tile_threads=128)))
    view, serial = ranking._single_chunk_service_view(original)
    assert serial
    assert view["execution_groups_json"] == original["execution_groups_json"]
    assert view["mapping_json"]["local_stage_partitions"] == original["mapping_json"]["local_stage_partitions"]
    assert view["mapping_json"]["tile_threads"] == 128
    assert original["stage_overlap"] is True and original["batch_tile_count"] == 16


@pytest.fixture
def measured_shared_service_model():
    from calibration_space import predicted_sample
    import stage_service_model as service

    serial = predicted_sample(dict(operator="fwht", precision="fp32", logN=20, batch=16,
        runtime_fingerprint="fixture-build:research", mapping_json=dict(
            backend="shared-iterative", stage_partition=[8, 8, 4], tile_threads=128,
            stage_overlap=False, batch_tile_count=1, local_stage_partitions=[[4, 4], [], [2, 2]])))
    first_stage = 0
    for index, group in enumerate(serial["execution_groups_json"]):
        group.update(first_stage=first_stage, compiler_resources_known=True,
                     compiler_registers_per_thread=32)
        first_stage += group["stage_count"]
    durations = [.003, .007, .002]
    records = [dict(schema=service.SCHEMA, status="measured", correct=True, sample=serial,
        groups=[dict(group, independent=True, role=role, trial_kernel_ms=[duration, duration])
                for group, duration in zip(serial["execution_groups_json"], durations)])
        for role in ("train", "validation")]
    measured = service.build_profile(records)
    assert measured["validation"]["passed"]
    measured.update(validated=True, calibration_status="complete-bounded")
    model = costs.fit([], dict(PROFILE, hardware=HARDWARE, stage_service=measured))
    model.update(status="calibrated-local-serial", usable_for_unmeasured_ranking=True,
        validation={"scope": "independent-stage-holdout-plus-serial-ranking"},
        whole_plan_validation={"status": "passed"}, runtime_fingerprint="fixture-build:research",
        protocol={"timing": "direct-launch"})
    return serial, measured, model, sum(durations)


def test_single_chunk_ranker_reuses_measured_curves_with_mixed_resident_groups(
        tmp_path, measured_shared_service_model):
    import stage_service_model as service

    serial, measured, model, expected = measured_shared_service_model
    overlapped = copy.deepcopy(serial)
    overlapped.update(stage_overlap=True, batch_tile_count=16)
    overlapped["mapping_json"].update(stage_overlap=True, batch_tile_count=16)
    multiple = copy.deepcopy(overlapped)
    multiple.update(batch_tile_count=8)
    multiple["mapping_json"].update(batch_tile_count=8)
    req = request(tmp_path, [serial, overlapped, multiple])
    req["timing_protocol"] = "direct-launch"
    write(tmp_path / "cost_model.json", model)
    frozen = copy.deepcopy(req)

    view, converted = ranking._single_chunk_service_view(ranking._sample(overlapped))
    assert converted
    assert view["mapping_json"]["local_stage_partitions"] == [[4, 4], [], [2, 2]]
    assert view["execution_groups_json"] == serial["execution_groups_json"]
    serial_predictions = costs.predict(ranking._sample(serial), model, explain=True)
    overlap_predictions = costs.predict(view, model, explain=True)
    serial_keys = [stage["curve_key"] for stage in serial_predictions["stages"]]
    assert serial_keys == [stage["curve_key"] for stage in overlap_predictions["stages"]]
    assert len(set(serial_keys)) == 3
    assert set(serial_keys) == set(measured["curves"])
    shapes = [json.loads(key)["group_shape"] for key in serial_keys]
    assert [shape["local_stage_partition"] for shape in shapes] == [[4, 4], [], [2, 2]]
    assert [shape["codelet"] for shape in shapes] == [
        "native-register-subgraph", "native", "native-register-subgraph"]
    for sample_view in (ranking._sample(serial), view):
        for group, key in zip(sample_view["execution_groups_json"], serial_keys):
            prediction = service.predict_group(sample_view, group, measured)
            assert prediction["covered"] and prediction["source"] == "measured-exact"
            assert prediction["curve_key"] == key
    assert not serial_predictions["whole_plan_fallback_used"]
    assert not overlap_predictions["whole_plan_fallback_used"]
    assert serial_predictions["kernel_ms"] == overlap_predictions["kernel_ms"] == expected

    result = ranking.rank(req)
    rows = {row["index"]: row for row in result["ranked"]}
    assert rows[0]["confidence"] == rows[1]["confidence"] == "validated-static-model"
    assert rows[0]["predicted_kernel_ms"] == rows[1]["predicted_kernel_ms"] == expected
    assert rows[1]["service_view"] == "equivalent-serial-single-chunk"
    assert rows[2]["confidence"] == "hardware-rate-prior"
    assert "overlap composition is unvalidated" in rows[2]["reason"]
    assert "service_view" not in rows[2]
    assert not result["unranked"] and not result["rejected"]
    assert req == frozen


def test_bad_rates_and_sample_are_unranked(tmp_path):
    req = request(tmp_path)
    write(tmp_path / "hardware_profile.json", dict(PROFILE, capabilities={}))
    assert ranking.rank(req)["unranked"]
    req["candidates"][0]["sample"]["execution_groups_json"] = "[]"
    assert "groups" in ranking.rank(req)["unranked"][0]["reason"]


def test_cli_batch_does_not_write_profiles_or_spawn_compiler(tmp_path, monkeypatch):
    req = request(tmp_path, [sample((4, 8)), sample()])
    for item in req["candidates"]:
        item["sample"]["stage_overlap"] = True
        item["sample"]["batch_tile_count"] = 2
    before = (tmp_path / "hardware_profile.json").read_bytes()
    request_file, output = tmp_path / "request.json", tmp_path / "output.json"
    write(request_file, req)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/rank_static_candidates.py"),
        "--request", str(request_file), "--output", str(output)], text=True, capture_output=True,
        env={**os.environ, "PATH": "", "CUDA_VISIBLE_DEVICES": ""})
    assert completed.returncode == 0, completed.stderr
    assert len(json.loads(output.read_text())["ranked"]) == 2
    assert (tmp_path / "hardware_profile.json").read_bytes() == before
    assert not (tmp_path / "cost_model.json").exists()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected process launch"))
    assert ranking.rank(req)["ranked"]


def test_malformed_profile_and_duplicate_indices_fail(tmp_path):
    req = request(tmp_path)
    (tmp_path / "hardware_profile.json").write_text("{")
    with pytest.raises(ValueError):
        ranking.rank(req)
    req["candidates"] *= 2
    with pytest.raises(ValueError, match="distinct"):
        ranking.rank(req)


def test_producer_identity_requires_actual_consistent_samples():
    empty = bind_static_model_identity({}, {"candidates": []})
    assert empty["static_ranking_identity"]["status"] == "unavailable"
    assert "runtime_fingerprint" not in empty
    record = dict(status="measured", correct=True,
                  samples=[dict(runtime_fingerprint="a", warmup=3, repeat=20),
                           dict(runtime_fingerprint="b", warmup=3, repeat=20)])
    mixed = bind_static_model_identity({}, {"candidates": [record]})
    assert "protocol" not in mixed
    record["samples"].pop()
    del record["samples"][0]["repeat"]
    missing = bind_static_model_identity({}, {"candidates": [record]})
    assert "counts" in missing["static_ranking_identity"]["reason"]


@pytest.mark.parametrize("target", ["graph", {"timing": "graph"}])
def test_direct_launch_producer_is_not_a_graph_model(tmp_path, target):
    req = request(tmp_path)
    write(tmp_path / "cost_model.json", model_fixture(sample()))
    req["timing_protocol"] = target
    result = ranking.rank(req)
    assert result["mode"] == "hardware-rate-prior"
    assert "protocol is incompatible" in result["reason"]


@pytest.mark.parametrize("validated_model", [False, True])
def test_unavailable_descriptor_remains_executable_fallback_not_rejected(
        tmp_path, monkeypatch, validated_model):
    value = sample()
    req = request(tmp_path, [value])
    if validated_model:
        write(tmp_path / "cost_model.json", model_fixture(value))
        req["timing_protocol"] = "direct-launch"
    value.update(physical_descriptor_status="unavailable",
                 physical_descriptor_reason="lowering has no faithful physical projection",
                 execution_groups_json="[]")
    monkeypatch.setattr(costs.PreparedPredictor, "prepare",
                        lambda *args: pytest.fail("unavailable descriptor reached scoring"))
    result = ranking.rank(req)
    assert result["ranked"] == [] and result["rejected"] == []
    assert result["unranked"] == [dict(index=0, reason=value["physical_descriptor_reason"])]


def test_descriptor_status_preserves_complete_candidate_partition(tmp_path):
    values = [sample() for _ in range(6)]
    values[1]["physical_descriptor_status"] = "projected"
    values[2]["physical_descriptor_status"] = "resolved"
    values[3]["physical_descriptor_status"] = "unavailable"
    values[4]["physical_descriptor_status"] = "unknown-status"
    groups = json.loads(values[5]["execution_groups_json"])
    groups[0]["threads"] = 2048
    values[5]["execution_groups_json"] = groups
    req = request(tmp_path, values)
    before = copy.deepcopy(req)
    result = ranking.rank(req)
    assert [item["index"] for item in result["ranked"]] == [0, 1, 2]
    assert [item["index"] for item in result["unranked"]] == [3, 4]
    assert [item["index"] for item in result["rejected"]] == [5]
    assert result["unranked"][0]["reason"] == "physical descriptor unavailable"
    assert "unknown physical_descriptor_status" in result["unranked"][1]["reason"]
    partitions = [item["index"] for key in ("ranked", "unranked", "rejected") for item in result[key]]
    assert sorted(partitions) == list(range(6))
    assert req == before


def test_ranker_reports_load_only_backend_and_preserves_user_disable(tmp_path, monkeypatch):
    import stage_schedule_native as native
    req = request(tmp_path)
    observed = []
    def backend():
        observed.append(os.environ.get("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY"))
        return "python-event-simulation"
    monkeypatch.setattr(native, "backend", backend)
    monkeypatch.setenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", "1")
    monkeypatch.delenv("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", raising=False)
    result = ranking.rank(req)
    assert result["schedule_backend"] == "python-event-simulation"
    assert observed == ["1"]
    assert os.environ["CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE"] == "1"
    assert "CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY" not in os.environ
