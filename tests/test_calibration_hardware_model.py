import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_local_hardware as calibration
import static_hardware_profile as static


@pytest.fixture
def setup(tmp_path, monkeypatch):
    build, profile_dir, output_dir = (tmp_path / name for name in ("build", "profile", "output"))
    for path in (build, profile_dir, output_dir):
        path.mkdir()
    (build / "cubutterfly_plan_probe").write_bytes(b"mock CPU-only probe identity")
    hardware = dict(device="NVIDIA A100 80GB PCIe", compute_capability="8.0", memory_bytes=84974239744,
        sm_count=108, max_threads_per_block=1024, max_threads_per_sm=2048, max_blocks_per_sm=32,
        registers_per_sm=65536, registers_per_block=65536, shared_bytes_per_sm=167936,
        shared_bytes_per_block=166912, cooperative_launch=True)
    identity = dict(device=hardware["device"], compute_capability="8.0", global_memory_bytes=str(hardware["memory_bytes"]))
    profile = dict(schema="cubutterfly-hardware-profile-v1", device=hardware["device"],
        compute_capability="8.0", global_memory_bytes=hardware["memory_bytes"], provenance={"microbench": "existing"})
    raw = dict(hardware, resource_source="live-cuda-properties:explicit-hardware-probe", auxiliary="not a cap")
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(raw))

    monkeypatch.setattr(calibration, "run", run)
    return SimpleNamespace(build=build, directory=profile_dir, output=output_dir,
        hardware=hardware, identity=identity, profile=profile, raw=raw, calls=calls)


def collect(state, *, refresh=False):
    return calibration.ensure_static_hardware_model(state.profile, state.identity, state.build,
        state.directory, state.output, refresh=refresh)


def test_initialization_exports_normalized_caps_and_hashed_live_evidence(setup):
    result = collect(setup)
    artifact = setup.directory / static.FILENAME
    assert artifact.is_file()
    assert not (setup.output / static.FILENAME).exists()
    assert result["hardware"] == setup.hardware
    assert set(result["hardware"]) == static.FIELDS
    assert setup.calls == [[str(setup.build / "cubutterfly_plan_probe"), "--hardware-live"]]
    snapshot = setup.directory / "hardware_probe.json"
    provenance = result["provenance"]
    assert provenance["source_sha256"][str(snapshot.resolve())] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert provenance["supplemental_provenance"]["resource_source"] == setup.raw["resource_source"]
    assert provenance["calibration_provenance"]["microbench"] == "existing"
    assert json.loads((setup.output / "calibration_device.json").read_text())["hardware"] == setup.hardware


def test_default_reuse_preserves_artifact_and_snapshot_without_probe(setup):
    collect(setup)
    artifact = setup.directory / static.FILENAME
    before = artifact.read_bytes()
    snapshot_before = (setup.directory / "hardware_probe.json").read_bytes()
    setup.calls.clear()
    setup.profile["hardware"] = {"stale": "ignored"}
    collect(setup)
    assert setup.calls == []
    assert artifact.read_bytes() == before
    assert (setup.directory / "hardware_probe.json").read_bytes() == snapshot_before
    assert setup.profile["hardware"] == setup.hardware


def test_explicit_refresh_replaces_old_caps_with_live_not_cached_data(setup):
    collect(setup)
    setup.calls.clear()
    setup.raw["sm_count"] = 107
    refreshed = collect(setup, refresh=True)
    assert len(setup.calls) == 1
    assert refreshed["hardware"]["sm_count"] == 107
    assert static.load(setup.directory / static.FILENAME)["hardware"]["sm_count"] == 107
    assert setup.profile["hardware"]["sm_count"] == 107


@pytest.mark.parametrize("change", ["static-source", "identity", "missing-cap", "bad-bool"])
def test_bad_live_capture_does_not_overwrite_previous_model(setup, change):
    collect(setup)
    artifact = setup.directory / static.FILENAME
    before = artifact.read_bytes()
    if change == "static-source":
        setup.raw["resource_source"] = "static-hardware-model:old.json"
    elif change == "identity":
        setup.raw["device"] = "another GPU"
    elif change == "missing-cap":
        del setup.raw["shared_bytes_per_block"]
    else:
        setup.raw["cooperative_launch"] = 1
    with pytest.raises((ValueError, KeyError)):
        collect(setup, refresh=True)
    assert artifact.read_bytes() == before


def test_invalid_existing_model_fails_instead_of_refreshing_by_default(setup):
    collect(setup)
    artifact = setup.directory / static.FILENAME
    value = json.loads(artifact.read_text())
    value["hardware"]["device"] = "another GPU"
    artifact.write_text(json.dumps(value))
    setup.calls.clear()
    with pytest.raises(ValueError, match="identity mismatch"):
        collect(setup)
    assert setup.calls == []


def test_probe_failure_preserves_existing_artifact(setup, monkeypatch):
    collect(setup)
    artifact = setup.directory / static.FILENAME
    before = artifact.read_bytes()

    def fail(command):
        raise RuntimeError("mock probe failed")

    monkeypatch.setattr(calibration, "run", fail)
    with pytest.raises(RuntimeError, match="mock probe failed"):
        collect(setup, refresh=True)
    assert artifact.read_bytes() == before


def test_main_binds_selected_artifact_before_later_calibration(setup, monkeypatch):
    collect(setup)
    (setup.directory / "hardware_profile.json").write_text(json.dumps(setup.profile))
    monkeypatch.setenv("CUBUTTERFLY_HARDWARE_MODEL", "/old/hardware_model.json")
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_FILE", "test-before")
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_BUDGET_SECONDS", "test-before")
    args = SimpleNamespace(trials=1, operator_trials=1, operator_warmup=0, operator_repeat=1,
        search_budget=0, seed_budget=0, search_finalists=1, search_seconds=0, compile_seconds=0,
        stage_point_budget=0, build_dir=setup.build, profile_dir=setup.directory, output_dir=setup.output,
        skip_operator_calibration=True, force=False, resume_search=False)
    monkeypatch.setattr(calibration, "parse_args", lambda: args)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)

    def run(command):
        if command[-1] == "--device-identity":
            return SimpleNamespace(stdout="device,compute_capability,global_memory_bytes\n"
                f"{setup.identity['device']},8.0,{setup.identity['global_memory_bytes']}\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(calibration, "run", run)

    class ReachedLaterCalibration(Exception):
        pass

    def guard(build_dir):
        assert calibration.os.environ["CUBUTTERFLY_HARDWARE_MODEL"] == str(setup.directory / static.FILENAME)
        assert static.load(calibration.os.environ["CUBUTTERFLY_HARDWARE_MODEL"])["hardware"] == setup.hardware
        raise ReachedLaterCalibration

    monkeypatch.setattr(calibration, "_SharedLibraryGuard", guard)
    with pytest.raises(ReachedLaterCalibration):
        calibration.main()
