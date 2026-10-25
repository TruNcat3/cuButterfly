import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_hardware as migration
import calibrate_local_hardware as local
from stage_service_calibration import _profile_identity


def _summary(**updates):
    return {"status": "complete-observed", "measured_requests": 2,
            "overlap_requests": 1, "measured_overlap_requests": 1,
            "failed_requests": 0, "missing_predictions": 0,
            "validation": {"status": "passed"},
            "inventory_complete": False, "concurrency_parameters_identified": False,
            **updates}


def _args(tmp_path, **updates):
    return SimpleNamespace(build_dir=tmp_path / "build", cost_model="staged",
                           stage_calibration="full", skip_operator_calibration=False,
                           operator_trials=3, operator_warmup=11, operator_repeat=17,
                           resume_search=True, **updates)


def _stage_seed_checkpoint(tmp_path, *, identity_updates=None, record_updates=None):
    profile = {"device": "Test GPU", "compute_capability": "8.0",
               "global_memory_bytes": 4096, "runtime_fingerprint": "runtime-current"}
    sample = {"operator": "fft", "precision": "fp32", "direction": "forward",
              "placement": "out-of-place", "logN": 8, "batch": 16,
              "runtime_fingerprint": "runtime-current",
              "mapping_json": json.dumps({"backend": "shared-iterative", "stage_partition": [4, 4]})}
    groups = [{"stage_begin": 0, "stage_count": 4, "threads": 128},
              {"stage_begin": 4, "stage_count": 4, "threads": 64}]
    record = {"status": "measured", "correct": True, "sample": sample, "groups": groups,
              **(record_updates or {})}
    identity = {**_profile_identity(profile), **(identity_updates or {})}
    path = tmp_path / "stage_calibration.json"
    path.write_text(json.dumps({"identity": identity, "records": [record]}))
    return path, profile, record


def test_stage_seed_preserves_actual_groups_and_is_independent_of_operator_data(tmp_path):
    path, profile, record = _stage_seed_checkpoint(tmp_path)
    operator = [{"name": "single-stage-winner", "samples": []}]
    original = copy.deepcopy(operator)
    seeds = local.composition_seed_records(operator, path, profile)
    assert operator == original
    assert seeds[0] == operator[0]
    seed = seeds[1]
    assert seed["correct"] is True and seed["status"] == "measured"
    assert seed["name"].startswith("stage-service-")
    assert seed["samples"][0]["correct"] == "1"
    assert json.loads(seed["samples"][0]["execution_groups_json"]) == record["groups"]
    assert seed["source"]["identity"] == _profile_identity(profile)
    assert seed["source"]["checkpoint"] == str(path)
    from composition_inventory import build_inventory
    assert any(request["role"] == "overlap" for request in build_inventory(seeds)["requests"])


@pytest.mark.parametrize("identity_updates", [
    {"device": "Other GPU"}, {"runtime_fingerprint": "old-runtime"},
    {"compile_policy": "old-policy"},
])
def test_stage_seed_rejects_checkpoint_identity_mismatch(tmp_path, identity_updates):
    path, profile, _ = _stage_seed_checkpoint(tmp_path, identity_updates=identity_updates)
    with pytest.raises(ValueError, match="identity mismatch"):
        local.composition_seed_records([], path, profile)


@pytest.mark.parametrize("record_updates", [
    {"correct": None}, {"correct": False}, {"status": "described"}, {"groups": []},
])
def test_stage_seed_skips_unverified_records(tmp_path, record_updates):
    path, profile, _ = _stage_seed_checkpoint(tmp_path, record_updates=record_updates)
    assert local.composition_seed_records([], path, profile) == []


def test_stage_seed_deduplicates_full_semantics_mapping_and_prefers_actual_groups(tmp_path):
    path, profile, record = _stage_seed_checkpoint(tmp_path)
    sample = {**record["sample"], "correct": "1"}
    operator = [{"name": "same-design", "samples": [sample]}]
    operator.append(copy.deepcopy(operator[0]))
    seeds = local.composition_seed_records(operator, path, profile)
    assert len(seeds) == 1
    assert seeds[0]["name"].startswith("stage-service-")
    assert "execution_groups_json" not in sample
    checkpoint = json.loads(path.read_text())
    checkpoint["records"].append(copy.deepcopy(record))
    checkpoint["records"].append({**record, "sample": {**sample, "direction": "inverse"}})
    path.write_text(json.dumps(checkpoint))
    seeds = local.composition_seed_records(operator, path, profile)
    assert len(seeds) == 2
    assert {seed["samples"][0]["direction"] for seed in seeds} == {"forward", "inverse"}


def test_stage_seed_rejects_record_runtime_mismatch(tmp_path):
    path, profile, record = _stage_seed_checkpoint(tmp_path)
    record["sample"]["runtime_fingerprint"] = "old-runtime"
    path.write_text(json.dumps({"identity": _profile_identity(profile), "records": [record]}))
    with pytest.raises(ValueError, match="identity mismatch"):
        local.composition_seed_records([], path, profile)


@pytest.mark.parametrize("summary", [
    _summary(),
    _summary(status="missing-overlap-population", overlap_requests=0, measured_overlap_requests=0),
    _summary(status="failed", failed_requests=1),
])
def test_composition_entry_preserves_operator_data_and_forwards_protocol(tmp_path, monkeypatch, summary):
    profile = {"stage_service": {"curves": {}}, "scheduling": {}, "pipeline_scheduling": {}}
    records = [{"name": "existing", "samples": [{"kernel_ms": "1.0"}]}]
    original = copy.deepcopy(records)
    operator_result = {"status": "complete", "candidates": records}
    guarded_run = object()
    exclusive = object()
    calls = []

    def calibration(build_dir, output_dir, target_profile, candidates, **kwargs):
        calls.append((build_dir, output_dir, candidates, kwargs))
        assert target_profile == profile
        assert target_profile is not profile
        target_profile["stage_service"]["probe_only"] = True
        return {"summary": summary}

    monkeypatch.setitem(sys.modules, "calibrate_composition", SimpleNamespace(run_calibration=calibration))
    monkeypatch.setattr(local, "require_exclusive_gpu", exclusive)
    monkeypatch.setattr(local, "run_operator_calibration", lambda *args: pytest.fail("candidate timings must be reused"))
    result = local._composition_calibration(_args(tmp_path), tmp_path, profile, operator_result, guarded_run)
    assert len(calls) == 1
    assert calls[0][0] == tmp_path / "build"
    assert calls[0][1] == tmp_path / "composition_calibration"
    assert calls[0][2] is records
    assert calls[0][3] == dict(trials=3, warmup=11, repeat=17, resume=True,
                             run=guarded_run, guard=exclusive)
    assert result["status"] == summary["status"]
    assert result["scope"] == "independent-heldout-cross-group-overlap"
    assert result["path"] == str(tmp_path / "composition_calibration/composition_calibration.json")
    assert ("gap" in result) == (summary["overlap_requests"] == 0)
    assert records == original
    assert "probe_only" not in profile["stage_service"]


@pytest.mark.parametrize("changes", [
    {"cost_model": "legacy"}, {"stage_calibration": "skip"},
    {"skip_operator_calibration": True},
])
def test_composition_entry_skip_and_legacy_do_not_invoke_probe(tmp_path, monkeypatch, changes):
    args = _args(tmp_path)
    vars(args).update(changes)
    monkeypatch.setitem(sys.modules, "calibrate_composition", SimpleNamespace(
        run_calibration=lambda *args, **kwargs: pytest.fail("probe must be skipped")))
    result = local._composition_calibration(args, tmp_path, {}, {"status": "complete"}, object())
    assert result["status"] == "skipped"
    assert result["path"] == ""


def test_incomplete_operator_sampling_does_not_invoke_probe(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "calibrate_composition", SimpleNamespace(
        run_calibration=lambda *args, **kwargs: pytest.fail("sampling has not completed")))
    result = local._composition_calibration(_args(tmp_path), tmp_path, {}, {"status": "failed"}, object())
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("summary,inventory,observed,expected", [
    (_summary(), {"coverage": {"gaps": []}, "omissions": []}, True, False),
    (_summary(inventory_complete=True), {"coverage": {"gaps": []}, "omissions": []}, True, True),
    (_summary(inventory_complete=True), {"coverage": {"gaps": ["ntt"]}, "omissions": []}, True, False),
    (_summary(inventory_complete=True), {"coverage": {"gaps": []}, "omissions": [{"reason": "budget"}]}, True, False),
    (_summary(inventory_complete=True), {}, True, False),
    (_summary(inventory_complete=True), {"coverage": {"gaps": None}, "omissions": []}, True, False),
    ({key: value for key, value in _summary().items() if key != "inventory_complete"},
     {"coverage": {"gaps": []}, "omissions": []}, True, False),
    (_summary(status="missing-overlap-population", overlap_requests=0, measured_overlap_requests=0), {}, False, False),
    (_summary(overlap_requests=0, measured_overlap_requests=0), {}, False, False),
    (_summary(status="failed", failed_requests=1), {}, False, False),
    (_summary(failed_requests=1), {}, False, False),
    (_summary(measured_overlap_requests=0), {}, False, False),
    (_summary(validation={"status": "failed"}), {}, False, False),
    (_summary(validation={"status": "uncovered"}), {}, False, False),
    (None, {}, False, False),
])
def test_full_migration_requires_observed_overlap_and_prediction_validation(
        tmp_path, monkeypatch, summary, inventory, observed, expected):
    monkeypatch.setattr(migration, "_workload_coverage", lambda *args: {
        "complete": True, "search_protocol_complete": True, "covered": []})
    monkeypatch.setattr(migration, "_selector_replay_summary", lambda *args: {"status": "passed"})
    (tmp_path / "cost_model.json").write_text(json.dumps({"status": "calibrated-local"}))
    (tmp_path / "stage_service_profile.json").write_text(json.dumps({
        "validated": True, "calibration_status": "complete", "inventory_complete": True}))
    manifest = {"status": "ready-with-measured-mappings", "stage_calibration_mode": "full"}
    if summary is not None:
        artifact = tmp_path / "composition_calibration.json"
        artifact.write_text(json.dumps({"summary": summary, "inventory": inventory}))
        manifest["composition_calibration"] = {"path": str(artifact), "status": summary["status"]}
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    result = migration._execution_summary(tmp_path, [])
    assert result["complete"] is expected
    assert result["composition_calibration_observed"] is observed
    assert result["composition_full_qualified"] is expected
    assert result["composition_inventory_complete"] is bool(summary and summary.get("inventory_complete") is True)
    if observed and not expected:
        assert result["composition_calibration_gap"]
    assert result["stage_service_validated"] is True
    assert result["status"] == "ready-with-measured-mappings"
    if summary is None:
        assert result["composition_calibration_status"] == "unknown"
    manifest["stage_calibration_mode"] = "skip"
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    assert migration._execution_summary(tmp_path, [])["complete"] is True
