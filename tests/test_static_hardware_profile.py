import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import static_hardware_profile as static


def model():
    return dict(schema=static.SCHEMA, hardware=dict(
        device="NVIDIA A100 80GB PCIe", compute_capability="8.0", memory_bytes=84974239744,
        sm_count=108, max_threads_per_block=1024, max_threads_per_sm=2048,
        max_blocks_per_sm=32, registers_per_sm=65536, registers_per_block=65536,
        shared_bytes_per_sm=167936, shared_bytes_per_block=166912, cooperative_launch=True),
        provenance={"uuid": "GPU-test"})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def preflight_artifacts(tmp_path):
    from hardware_registry import SCHEMA, canonical_semantics
    document = model()
    expected = static.identity(document)
    root = tmp_path / "profiles"
    hardware = write(root / static._tag(expected) / static.FILENAME, document)
    key = canonical_semantics(dict(operator="fft", precision="fp32", logN=10,
                                   batch=1, placement="out-of-place"))
    registry = write(tmp_path / "registry.json", dict(schema=SCHEMA, targets=[dict(
        hardware=expected, records=[dict(status="measured", runtime_fingerprint="build-test", key=key)])]))
    return root, expected, hardware, registry, key


def test_preflight_exact_artifacts_and_optional_workloads(tmp_path):
    root, expected, hardware, registry, key = preflight_artifacts(tmp_path)
    result = static.preflight(root, expected, registry, "build-test", workload_keys=[key])
    assert result["hardware_path"] == str(hardware)
    assert result["measured_records"] == 1
    assert result["calibration_completion_checked"] is False
    assert static.main(["preflight", "--profile-root", str(root), "--expected-identity",
                        str(write(tmp_path / "identity.json", expected)), "--registry", str(registry),
                        "--runtime-fingerprint", "build-test"]) == 0
    with pytest.raises(ValueError, match="exact measured workload"):
        static.preflight(root, expected, registry, "build-test", workload_keys=[dict(key, batch="2")])


@pytest.mark.parametrize("failure", ["leaf", "hardware", "registry-hardware", "fingerprint", "unmeasured"])
def test_preflight_rejects_incompatible_artifacts(tmp_path, failure):
    root, expected, hardware, registry, _ = preflight_artifacts(tmp_path)
    match = "identity mismatch"
    if failure == "leaf":
        root = hardware.parent
        match = "legacy leaf"
    elif failure == "hardware":
        document = json.loads(hardware.read_text())
        document["hardware"]["memory_bytes"] = 40 << 30
        write(hardware, document)
    elif failure in ("registry-hardware", "unmeasured"):
        document = json.loads(registry.read_text())
        target = document["targets"][0]
        if failure == "registry-hardware":
            target["hardware"]["global_memory_bytes"] = 40 << 30
        else:
            target["records"][0]["status"] = "needs-revalidation"
            match = "runtime_fingerprint"
        write(registry, document)
    else:
        match = "runtime_fingerprint"
    with pytest.raises(ValueError, match=match):
        static.preflight(root, expected, registry, "other-build" if failure == "fingerprint" else "build-test")


@pytest.mark.parametrize("field,value", [("device", "A100"), ("compute_capability", "9.0"),
                                        ("global_memory_bytes", 40 << 30)])
def test_identity_is_exact(field, value):
    document = model()
    expected = dict(static.identity(document), **{field: value})
    with pytest.raises(ValueError, match="identity mismatch"):
        static.validate(document, expected)


@pytest.mark.parametrize("field", sorted(static.FIELDS))
def test_missing_capability_rejected(field):
    document = model()
    del document["hardware"][field]
    with pytest.raises(ValueError, match="full exact capability"):
        static.validate(document)


@pytest.mark.parametrize("field,value", [("shared_bytes_per_block", True),
                                        ("max_threads_per_block", 0),
                                        ("cooperative_launch", 1),
                                        ("registers_per_block", 65537)])
def test_invalid_capacity_types_and_bounds(field, value):
    document = model()
    document["hardware"][field] = value
    with pytest.raises(ValueError):
        static.validate(document)


def test_selection_precedence_tag_and_explicit_mismatch(tmp_path):
    document = model()
    expected = static.identity(document)
    local = tmp_path / "local"
    bundled = tmp_path / "bundled"
    path = write(bundled / static._tag(expected) / static.FILENAME, document)
    result = static.select(expected, local_dir=local, bundled_root=bundled)
    assert result["selected"]["path"] == str(path)
    assert result["action"] == "reuse" and not result["recalibration_requested"]
    write(local / static.FILENAME, document)
    assert static.select(expected, local_dir=local, bundled_root=bundled)["selected"]["path"] == str(local / static.FILENAME)
    wrong = copy.deepcopy(document)
    wrong["hardware"]["memory_bytes"] = 40 << 30
    explicit = write(tmp_path / "wrong.json", wrong)
    assert static.select(expected, explicit=explicit, local_dir=local)["selected"] is None
    from calibrate_hardware import _tag
    assert static._tag(expected) == _tag(expected)


def calibration(tmp_path):
    document = model()
    profile = tmp_path / "calibration"
    hardware = dict(document["hardware"])
    del hardware["shared_bytes_per_block"]
    write(profile / "calibration_device.json", dict(
        schema="cubutterfly-hardware-profile-v1",
        device=document["hardware"]["device"], compute_capability="8.0",
        global_memory_bytes=document["hardware"]["memory_bytes"], hardware=hardware,
        provenance={"cuda_visible_devices": "GPU-test"}))
    evidence = write(tmp_path / "verified_caps.json", {"shared_bytes_per_block": 166912})
    supplemental = dict(hardware={"shared_bytes_per_block": 166912},
                        provenance=dict(verified=True, sources=[str(evidence)]))
    return profile, supplemental


def test_export_requires_verified_supplement_and_hashes_sources(tmp_path):
    profile, supplemental = calibration(tmp_path)
    output = tmp_path / "out" / static.FILENAME
    with pytest.raises(ValueError, match="full exact capability"):
        static.export_from_calibration(profile, output)
    unverified = copy.deepcopy(supplemental)
    unverified["provenance"]["verified"] = False
    with pytest.raises(ValueError, match="verified source"):
        static.export_from_calibration(profile, output, unverified)
    document = static.export_from_calibration(profile, output, supplemental)
    assert len(document["provenance"]["source_sha256"]) == 2
    assert document["provenance"]["calibration_provenance"]["cuda_visible_devices"] == "GPU-test"
    assert static.load(output) == document


def test_default_reuse_preserves_bytes_and_refresh_is_explicit(tmp_path):
    profile, supplemental = calibration(tmp_path)
    output = tmp_path / static.FILENAME
    first = static.export_from_calibration(profile, output, supplemental)
    before = output.read_bytes()
    # Default reuse does not even require historical source files to remain present.
    assert static.export_from_calibration(tmp_path / "missing", output) == first
    assert output.read_bytes() == before
    evidence = Path(supplemental["provenance"]["sources"][0])
    evidence.write_text("updated verified evidence")
    static.export_from_calibration(profile, output, supplemental, refresh=True)
    assert output.read_bytes() != before


def test_warning_cost_presence_never_means_validated(tmp_path):
    profile, supplemental = calibration(tmp_path)
    write(profile / "cost_model.json", dict(schema="cost", version="historical-v6",
        status="calibrated-local-warning", usable_for_unmeasured_ranking=False,
        validation={"status": "passed"}))
    write(profile / "stage_service_profile.json", dict(version="v5", validated=True,
        calibration_status="complete"))
    output = profile / static.FILENAME
    document = static.export_from_calibration(profile, output, supplemental)
    layers = static.inspect(output)["layers"]
    assert layers["cost_model"]["present"] and not layers["cost_model"]["validated"]
    assert layers["cost_model"]["version"] == "historical-v6"
    assert layers["stage_service"]["validated"]
    assert document["provenance"]["performance_layers"] == layers


def reported_cost_model():
    """Produce a validated model from synthetic CPU-only calibration rows."""
    import stage_cost_model as costs
    profile = dict(static.identity(model()), hardware=model()["hardware"], sm_count=108,
        capabilities=dict(global_feedback_bytes_per_second=1e12,
            equivalent_butterflies_per_second=1e11, interstage_shared_bytes_per_second=1e13,
            cta_barriers_per_second=1e9))
    prior = costs.fit([], profile)
    rows = []
    for batch in (1, 4, 8):
        for alternative in (0, 1):
            sample = dict(operator="fft", precision="fp32", logN=8, batch=batch,
                execution_groups_json=[dict(stage_count=4, grid_ctas=16, threads=64,
                    core="scalar", live_shared_bytes=256)])
            rows.append(dict(name=f"synthetic-{batch}-{alternative}", sample=sample,
                             median_kernel_ms=costs.predict(sample, prior)))
    return costs.report(rows, profile)


def test_real_report_validation_is_distinct_from_request_compatibility(tmp_path):
    import rank_static_candidates as ranking
    cost = reported_cost_model()
    assert cost["usable_for_unmeasured_ranking"] is True
    assert "status" not in cost["validation"]
    write(tmp_path / "cost_model.json", cost)
    layer = static._performance_layers(tmp_path)["cost_model"]
    assert layer["validated"] is True
    assert layer["compatibility"] == "requires-request-check"
    assert layer["validation_reason"] == ""
    reason = ranking._model_reason(cost, dict(runtime_fingerprint="unproven-build"))
    assert "fingerprint" in reason


@pytest.mark.parametrize("change", [
    {"status": "calibrated-local-warning"},
    {"usable_for_unmeasured_ranking": False},
    {"version": "physical-stage-composition-v6"},
    {"parameters": {"bad": {"service": -1}}},
    {"validation": {"status": "passed"}},
])
def test_layer_and_ranker_share_model_validation_gate(tmp_path, change):
    import rank_static_candidates as ranking
    cost = dict(reported_cost_model(), **change)
    write(tmp_path / "cost_model.json", cost)
    layer = static._performance_layers(tmp_path)["cost_model"]
    assert layer["validated"] is False
    assert layer["validation_reason"] == ranking._model_reason(cost, {})


def test_cli_is_json_and_cpu_only(tmp_path, capsys):
    output = write(tmp_path / static.FILENAME, model())
    assert static.main(["inspect", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["hardware_complete"]
    assert static.main(["inspect", str(tmp_path / "absent")]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_legacy_partial_reused_only_after_complete_candidates(tmp_path):
    local, _ = calibration(tmp_path)
    expected = static.identity(model())
    performance = write(local / "cost_model.json", dict(status="calibrated-local-warning"))
    before = {path: path.read_bytes() for path in (local / "calibration_device.json", performance)}
    result = static.select(expected, local_dir=local)
    assert result["action"] == "reuse-partial"
    assert not result["selected"]["hardware_complete"]
    assert "shared_bytes_per_block" in result["selected"]["missing_capabilities"]
    assert not result["recalibration_requested"]
    assert result["selected"]["layers"]["cost_model"]["present"]
    assert not result["selected"]["layers"]["cost_model"]["validated"]
    assert all(path.read_bytes() == content for path, content in before.items())
    bundled = tmp_path / "bundled"
    write(bundled / static._tag(expected) / static.FILENAME, model())
    result = static.select(expected, local_dir=local, bundled_root=bundled)
    assert result["action"] == "reuse"
    assert result["selected"]["hardware_complete"]
    assert result["selected"]["path"].startswith(str(bundled))
    assert any(source["source"] == "local-legacy" and source["compatible"]
               for source in result["sources"])


@pytest.mark.parametrize("field,value", [("device", "A100"), ("compute_capability", "9.0"),
                                        ("global_memory_bytes", 40 << 30)])
def test_legacy_partial_rejects_identity_mismatch(tmp_path, field, value):
    local, _ = calibration(tmp_path)
    expected = dict(static.identity(model()), **{field: value})
    result = static.select(expected, local_dir=local)
    assert result["selected"] is None
    assert result["action"] == "hardware-artifact-required"
    assert not result["recalibration_requested"]


def test_legacy_explicit_is_inspectable_but_not_loadable(tmp_path):
    local, _ = calibration(tmp_path)
    path = local / "calibration_device.json"
    expected = static.identity(model())
    assert not static.inspect_legacy(path, expected)["hardware_complete"]
    assert static.select(expected, explicit=path)["action"] == "error"
    with pytest.raises(ValueError, match="exactly schema"):
        static.load(path, expected)


@pytest.mark.parametrize("field", static.INTEGER_CAPS)
def test_integer_capacities_match_cpp_unsigned_ranges(field):
    document = model()
    document["hardware"][field] = 1 << (64 if field == "memory_bytes" else 32)
    with pytest.raises(ValueError, match="positive uint"):
        static.validate(document)


def test_rates_are_reported_without_implying_validated_model(tmp_path):
    output = write(tmp_path / static.FILENAME, model())
    write(tmp_path / "hardware_profile.json", dict(schema="cubutterfly-hardware-profile-v1",
        status="calibrated-local", validated=True))
    layer = static.inspect(output)["layers"]["hardware_profile"]
    assert layer["present"] and layer["status"] == "calibrated-local"
    assert not layer["validated"]


@pytest.mark.parametrize("contents", ["not JSON", "{}"])
def test_broken_local_static_artifact_blocks_all_fallbacks(tmp_path, contents):
    local, _ = calibration(tmp_path)
    (local / static.FILENAME).write_text(contents)
    expected = static.identity(model())
    bundled = tmp_path / "bundled"
    write(bundled / static._tag(expected) / static.FILENAME, model())
    result = static.select(expected, local_dir=local, bundled_root=bundled)
    assert result["selected"] is None
    assert result["action"] == "error" and "invalid hardware artifact" in result["error"]
    assert not result["recalibration_requested"]


def test_missing_explicit_is_error_even_when_bundle_matches(tmp_path):
    expected = static.identity(model())
    bundled = tmp_path / "bundled"
    write(bundled / static._tag(expected) / static.FILENAME, model())
    result = static.select(expected, explicit=tmp_path / "absent.json", bundled_root=bundled)
    assert result["action"] == "error" and result["selected"] is None
    assert not result["recalibration_requested"]


def test_broken_local_symlink_blocks_legacy_and_bundle(tmp_path):
    local, _ = calibration(tmp_path)
    (local / static.FILENAME).symlink_to(tmp_path / "absent.json")
    expected = static.identity(model())
    bundled = tmp_path / "bundled"
    write(bundled / static._tag(expected) / static.FILENAME, model())
    result = static.select(expected, local_dir=local, bundled_root=bundled)
    assert result["action"] == "error" and result["selected"] is None
    assert not result["recalibration_requested"]
