"""CPU-only preflight and provenance checks for fresh library continuations."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_library_rebound_campaign as rebound
from semantic_campaign_manifest import MANIFEST_SCHEMA, manifest_hash, validate_manifest_hash

GPU = "GPU-00000000-0000-0000-0000-000000000002"


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    library = tmp_path / "library.so"
    library.write_bytes(b"new build")
    source = tmp_path / "source.py"
    source.write_text("immutable source")
    config = tmp_path / "population.json"
    config.write_text("{}")
    manifest = {"schema": MANIFEST_SCHEMA, "cases": [
        {"id": "case-one", "descriptor": {"semantics": {"batch": 17}}, "populations": ["full"]}],
        "fingerprints": {"library": {"path": str(library), "sha256": "old-library"},
                         "sources": [rebound.evidence(source)], "config": rebound.evidence(config)},
        "continuation_provenance": {"historical_reports": [{"path": "old-result", "sha256": "old-hash"}]}}
    manifest["manifest_sha256"] = manifest_hash(manifest)
    paths = [tmp_path / "remaining.json", tmp_path / "refresh.json"]
    for path in paths:
        write(path, manifest)
    profile = tmp_path / "rates.json"
    write(profile, {"device": "CPU fixture GPU", "global_memory_bytes": 123,
                    "compute_capability": "8.0", "provenance": {"cuda_visible_devices": GPU}})
    for name in ("registry.json", "model.json"):
        write(tmp_path / name, {})
    calibration = {"schema": "cubutterfly-local-calibration-v1", "status": "ready-with-measured-mappings",
        "measurement_identity": {"library_sha256": rebound.runner.sha256_file(library),
                                 "compile_mode": "research", "verify_batches": 0, "jit_runtime": {"id": "current"}},
        "device": "CPU fixture GPU", "global_memory_bytes": 123, "compute_capability": "8.0",
        "hardware_rates_remeasured": True, "hardware_rates_source": str(profile),
        "registry_path": str(tmp_path / "registry.json"), "cost_model": str(tmp_path / "model.json"),
        "selector_replay": {"status": "passed"}}
    calibration_path = tmp_path / "calibration.json"
    write(calibration_path, calibration)
    plan = {"gpu_uuid": GPU, "library": str(library), "compile_broker": str(tmp_path),
            "historical_provenance": {"path": "historical-handoff", "sha256": "preserved"},
            "phases": [{"manifest": str(path), "name": name, "preparation": f"{name}-prepared",
                        "timing": timing, "extra_phase_metadata": name}
                       for path, name, timing in zip(paths, ("remaining", "refresh"), (True, False))]}
    plan_path = tmp_path / "gpu1_plan.json"
    write(plan_path, plan)
    monkeypatch.setattr(rebound.broker, "_document", lambda path: {"id": "broker"})
    def timing(value, document):
        if value["measurement_identity"].get("jit_runtime") != {"id": "current"}:
            raise ValueError("calibration JIT runtime identity mismatch")
        return {"id": "current"}
    monkeypatch.setattr(rebound.jit, "validate_timing", timing)
    output = tmp_path / "out"
    args = ["--source-plan", str(plan_path), "--source-manifest", str(paths[0]),
            "--library", str(library), "--calibration", str(calibration_path),
            "--output", str(output), "--compile-socket", "/tmp/rebound-cpu-test.sock"]
    return dict(args=args, output=output, calibration=calibration, calibration_path=calibration_path,
                manifest=manifest, paths=paths, source=source, plan=plan, plan_path=plan_path, profile=profile)


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("library_sha256", "wrong"), ("compile_mode", "production"),
    ("verify_batches", 1), ("jit_runtime", {"id": "stale"}), ("selector_replay", {"status": "failed"}),
    ("hardware_rates_remeasured", False), ("device", "wrong GPU"),
    ("compute_capability", "9.0"), ("global_memory_bytes", 456)])
def test_calibration_mismatch_leaves_no_output(campaign, field, value):
    calibration = campaign["calibration"]
    target = calibration["measurement_identity"] if field in (
        "library_sha256", "compile_mode", "verify_batches", "jit_runtime") else calibration
    target[field] = value
    write(campaign["calibration_path"], calibration)
    with pytest.raises(ValueError):
        rebound.main(campaign["args"])
    assert not campaign["output"].exists()


def test_profile_uuid_mismatch_leaves_no_output(campaign):
    profile = json.loads(campaign["profile"].read_text())
    profile["provenance"]["cuda_visible_devices"] = "GPU-00000000-0000-0000-0000-000000000001"
    write(campaign["profile"], profile)
    with pytest.raises(ValueError, match="UUID"):
        rebound.main(campaign["args"])
    assert not campaign["output"].exists()


@pytest.mark.parametrize("selected", [0, 1])
def test_phase_metadata_cases_and_history_preserved(campaign, selected, capsys):
    args = campaign["args"]
    args[args.index("--source-manifest") + 1] = str(campaign["paths"][selected])
    assert rebound.main(args) == 0
    output = campaign["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert validate_manifest_hash(manifest)
    assert manifest["cases"] == campaign["manifest"]["cases"]
    assert manifest["continuation_provenance"]["parent_continuation_provenance"] == campaign["manifest"]["continuation_provenance"]
    plan = json.loads((output / "gpu1_plan.json").read_text())
    expected = copy.deepcopy(campaign["plan"]["phases"][selected])
    expected["manifest"] = str(output / "manifest.json")
    assert plan["phases"] == [expected]
    assert plan["historical_provenance"] == campaign["plan"]["historical_provenance"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["omitted_phases"] == [campaign["plan"]["phases"][1 - selected]]
    assert provenance["selected_phase"] == campaign["plan"]["phases"][selected]
    assert provenance["historical_rows_adopted"] == 0
    assert not provenance["full_source_plan"]
    assert json.loads(capsys.readouterr().out)["omitted_phases"] == ["refresh" if selected == 0 else "remaining"]


@pytest.mark.parametrize("failure", ["manifest", "source", "config", "unrelated", "socket", "broker"])
def test_bad_preflight_leaves_no_output(campaign, failure, monkeypatch):
    if failure == "manifest":
        value = copy.deepcopy(campaign["manifest"])
        value["cases"] = []
        write(campaign["paths"][0], value)
    elif failure == "source":
        campaign["source"].write_text("changed source")
    elif failure == "config":
        (campaign["source"].parent / "population.json").write_text("changed config")
    elif failure == "unrelated":
        unrelated = campaign["source"].parent / "unrelated.json"
        write(unrelated, campaign["manifest"])
        campaign["args"][campaign["args"].index("--source-manifest") + 1] = str(unrelated)
    elif failure == "socket":
        campaign["args"][-1] = "/tmp/" + "x" * 103
    else:
        def corrupt(path):
            raise ValueError("corrupt broker identity")
        monkeypatch.setattr(rebound.broker, "_document", corrupt)
    with pytest.raises(ValueError):
        rebound.main(campaign["args"])
    assert not campaign["output"].exists()


def test_explicit_c_api_rebind_records_historical_and_current_sources(campaign):
    historical = campaign["source"].parent / "historical_c_api.cpp"
    historical.write_text("historical C API")
    old = rebound.evidence(historical)
    value = copy.deepcopy(campaign["manifest"])
    value["fingerprints"]["sources"].append(old)
    value["manifest_sha256"] = manifest_hash(value)
    write(campaign["paths"][0], value)
    current = ROOT / "src/c_api.cpp"
    with pytest.raises(ValueError, match="explicit source rebound"):
        rebound.main(campaign["args"])
    assert not campaign["output"].exists()
    assert rebound.main(campaign["args"] + ["--rebind-c-api-source", str(current)]) == 0
    provenance = json.loads((campaign["output"] / "provenance.json").read_text())
    change = provenance["source_rebindings"][0]
    assert change["historical_source"] == old
    assert change["current_source"]["path"] == str(current)
    assert change["current_source"]["sha256"] == rebound.runner.sha256_file(current)
    assert json.loads(campaign["paths"][0].read_text()) == value


def test_c_api_rebind_rejects_unrelated_source(campaign):
    with pytest.raises(ValueError, match="current repository"):
        rebound.main(campaign["args"] + ["--rebind-c-api-source", str(campaign["source"])])
    assert not campaign["output"].exists()


def test_c_api_rebind_rejects_corrupt_historical_file(campaign):
    historical = campaign["source"].parent / "historical_c_api.cpp"
    historical.write_text("historical C API")
    value = copy.deepcopy(campaign["manifest"])
    value["fingerprints"]["sources"].append(rebound.evidence(historical))
    value["manifest_sha256"] = manifest_hash(value)
    write(campaign["paths"][0], value)
    historical.write_text("corrupt historical evidence")
    with pytest.raises(ValueError, match="historical C API source fingerprint"):
        rebound.main(campaign["args"] + ["--rebind-c-api-source", str(ROOT / "src/c_api.cpp")])
    assert not campaign["output"].exists()


def test_profile_without_memory_reports_validation_limit(campaign):
    profile = json.loads(campaign["profile"].read_text())
    profile.pop("global_memory_bytes")
    write(campaign["profile"], profile)
    assert rebound.main(campaign["args"]) == 0
    provenance = json.loads((campaign["output"] / "provenance.json").read_text())
    assert provenance["calibration_validation"]["memory_identity_validation"].startswith("recorded-calibration-only")


def test_missing_calibration_profile_is_readable_and_leaves_no_output(campaign):
    calibration = campaign["calibration"]
    calibration.pop("hardware_rates_source")
    write(campaign["calibration_path"], calibration)
    with pytest.raises(ValueError, match="calibration missing required fields: hardware_rates_source"):
        rebound.main(campaign["args"])
    assert not campaign["output"].exists()
