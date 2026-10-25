"""Installer profile policy exercised with CPU-only mocked executables."""
import json
import os
import pathlib
import shlex
import subprocess
import sys

from test_install_migration_shell import _mock_tools


ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts/install_with_hardware_profile.sh"


def _hardware(memory=42949672960):
    return {"schema": "cubutterfly-hardware-model-v1", "hardware": {
        "device": "Test GPU", "compute_capability": "8.0", "memory_bytes": memory,
        "sm_count": 108, "max_threads_per_block": 1024, "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32, "registers_per_sm": 65536, "registers_per_block": 65536,
        "shared_bytes_per_sm": 167936, "shared_bytes_per_block": 166912,
        "cooperative_launch": True}, "provenance": {"source": "CPU test fixture"}}


def _profile(tmp_path, *, legacy=False, memory=42949672960):
    directory = tmp_path / "profiles" / "Test-GPU-sm80-42949672960B"
    directory.mkdir(parents=True)
    if legacy:
        document = {"schema": "cubutterfly-hardware-profile-v1", "device": "Test GPU",
                    "compute_capability": "8.0", "global_memory_bytes": memory}
        name = "calibration_device.json"
    else:
        document, name = _hardware(memory), "hardware_model.json"
    (directory / name).write_text(json.dumps(document))
    return directory


def _run(tmp_path, *options, bundled_root=None, environment=None, use_environment_root=False):
    bin_dir, log = _mock_tools(tmp_path)
    wrapper = bin_dir / "mock-python"
    original = wrapper.read_text()
    marker = 'if [[ "${1:-}" == *"/scripts/calibrate_hardware.py" ]]; then'
    wrapper.write_text(original.replace(marker, marker + '\n    printf "calibration-env model=%s root=%s\\n" "${CUBUTTERFLY_HARDWARE_MODEL:-unset}" "${CUBUTTERFLY_PROFILE_ROOT:-unset}" >> "$MOCK_LOG"'))
    if bundled_root is not None:
        fake_root = tmp_path / "source"
        fake_root.mkdir()
        (fake_root / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
        (fake_root / "config").mkdir()
        (fake_root / "config/hardware").symlink_to(bundled_root, target_is_directory=True)
        wrapper = bin_dir / "mock-python"
        original = wrapper.read_text()
        dispatch = (f'if [[ "${{1:-}}" == "-" && "${{2:-}}" == {shlex.quote(str(ROOT))} ]]; then\n'
                    f'    shift 2\n    exec {shlex.quote(sys.executable)} - {shlex.quote(str(fake_root))} "$@"\nfi\n')
        wrapper.write_text(original.replace(f'exec {pathlib.Path(sys.executable).resolve()} "$@"',
                                            dispatch + f'exec {shlex.quote(sys.executable)} "$@"'))
    env = os.environ.copy()
    env.update(PATH=f"{bin_dir}:{env['PATH']}", PYTHON=str(bin_dir / "mock-python"), MOCK_LOG=str(log))
    for name in ("PROFILE_DIR", "PROFILE_ROOT", "BUILD_DIR", "PREFIX",
                 "CUBUTTERFLY_HARDWARE_MODEL", "CUBUTTERFLY_PROFILE_ROOT"):
        env.pop(name, None)
    env.update(environment or {})
    root_options = [] if use_environment_root else ["--profile-root", str(tmp_path / "profiles")]
    completed = subprocess.run([str(INSTALLER), "--build-dir", str(tmp_path / "build"),
        "--prefix", str(tmp_path / "prefix"), *root_options,
        "--skip-tests", *options], cwd=ROOT, env=env, text=True, capture_output=True)
    return completed, log.read_text() if log.exists() else ""


def test_matching_local_static_profile_reused_without_calibration(tmp_path):
    directory = _profile(tmp_path)
    cost = {"status": "calibrated-local-warning", "usable_for_unmeasured_ranking": True,
            "validation": {"status": "passed"}}
    (directory / "cost_model.json").write_text(json.dumps(cost))
    original = (directory / "cost_model.json").read_bytes()
    completed, commands = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands and "--force" not in commands
    assert "reusing profile artifacts" in completed.stderr
    assert "cost_model: present, unvalidated" in completed.stderr
    assert "stage_service: missing" in completed.stderr
    assert (directory / "cost_model.json").read_bytes() == original
    decision = json.loads((directory / "installation_profile_selection.json").read_text())
    assert decision["action"] == "reuse"


def test_explicit_refresh_measures_with_force(tmp_path):
    _profile(tmp_path)
    completed, commands = _run(tmp_path, "--refresh-profile")
    assert completed.returncode == 0, completed.stderr
    calibration = next(line for line in commands.splitlines() if line.startswith("calibrate "))
    assert "--force" in calibration and "--resume-search" not in calibration


def test_missing_profile_initializes_without_force(tmp_path):
    completed, commands = _run(tmp_path, "--skip-operator-calibration")
    assert completed.returncode == 0, completed.stderr
    calibration = next(line for line in commands.splitlines() if line.startswith("calibrate "))
    assert "--force" not in calibration
    assert "--skip-operator-calibration" in calibration


def test_resume_search_is_an_explicit_measurement_request(tmp_path):
    _profile(tmp_path)
    completed, commands = _run(tmp_path, "--resume-search")
    assert completed.returncode == 0, completed.stderr
    calibration = next(line for line in commands.splitlines() if line.startswith("calibrate "))
    assert "--resume-search" in calibration and "--force" not in calibration


def test_refresh_and_resume_are_mutually_exclusive(tmp_path):
    completed, commands = _run(tmp_path, "--refresh-profile", "--resume-search")
    assert completed.returncode == 2
    assert "mutually exclusive" in completed.stderr and not commands


def test_incompatible_existing_local_profile_is_not_implicitly_replaced(tmp_path):
    directory = _profile(tmp_path, memory=85899345920)
    original = (directory / "hardware_model.json").read_bytes()
    completed, commands = _run(tmp_path)
    assert completed.returncode != 0
    assert "invalid hardware artifact" in completed.stderr
    assert "calibrate " not in commands
    assert (directory / "hardware_model.json").read_bytes() == original


def test_matching_legacy_partial_profile_is_preserved_without_refresh(tmp_path):
    directory = _profile(tmp_path, legacy=True)
    completed, commands = _run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands
    assert "hardware capabilities: partial" in completed.stderr
    assert "--refresh-profile" in completed.stderr
    assert json.loads((directory / "installation_profile_selection.json").read_text())["action"] == "reuse"


def test_complete_bundle_fills_caps_and_reports_preserved_local_performance(tmp_path):
    local = _profile(tmp_path, legacy=True)
    cost = {"status": "calibrated-local-warning", "usable_for_unmeasured_ranking": False}
    (local / "cost_model.json").write_text(json.dumps(cost))
    bundled = tmp_path / "bundled"
    target = bundled / local.name
    target.mkdir(parents=True)
    (target / "hardware_model.json").write_text(json.dumps(_hardware()))
    (target / "hardware_profile.json").write_text(json.dumps({"status": "hardware-rates-only"}))
    completed, commands = _run(tmp_path, bundled_root=bundled)
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands
    assert json.loads((local / "cost_model.json").read_text()) == cost
    assert (local / "hardware_model.json").is_file()
    assert "cost_model: present, unvalidated" in completed.stderr
    assert "stage_service: missing" in completed.stderr
    decision = json.loads((local / "installation_profile_selection.json").read_text())
    assert decision["selection"]["published"]["layers"]["cost_model"]["present"]


def test_custom_profile_has_explicit_runtime_environment_and_install_prefix(tmp_path):
    auto = _profile(tmp_path)
    custom = tmp_path / "custom-profile"
    auto.rename(custom)
    completed, commands = _run(tmp_path, "--profile-dir", str(custom))
    assert completed.returncode == 0, completed.stderr
    configure = next(line for line in commands.splitlines() if line.startswith("cmake ") and " -S " in line)
    assert f"-DCMAKE_INSTALL_PREFIX={tmp_path / 'prefix'}" in configure
    activation = custom / "profile_environment.sh"
    assert f"runtime profile environment: source {activation}" in completed.stdout
    activated = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "$CUBUTTERFLY_HARDWARE_MODEL"',
                                "test", str(activation)], text=True, capture_output=True, check=True)
    assert activated.stdout.strip() == str(custom / "hardware_model.json")


def test_explicit_hardware_file_wins_and_is_published_under_canonical_name(tmp_path):
    local = tmp_path / "profiles" / "Test-GPU-sm80-42949672960B"
    explicit = tmp_path / "requested-caps.json"
    document = _hardware()
    document["provenance"] = {"source": "explicit selection"}
    explicit.write_text(json.dumps(document))
    completed, commands = _run(tmp_path, environment={"CUBUTTERFLY_HARDWARE_MODEL": str(explicit)})
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands
    assert json.loads((local / "hardware_model.json").read_text()) == document
    decision = json.loads((local / "installation_profile_selection.json").read_text())
    assert decision["selection"]["selected"]["path"] == str(explicit)
    assert decision["selection"]["published"]["path"] == str(local / "hardware_model.json")


def test_explicit_selection_preserves_existing_local_bytes_and_drives_runtime(tmp_path):
    local = _profile(tmp_path)
    original = (local / "hardware_model.json").read_bytes()
    explicit = tmp_path / "chosen.json"
    document = _hardware()
    document["hardware"]["sm_count"] = 80
    document["provenance"] = {"source": "explicit distinct caps"}
    explicit.write_text(json.dumps(document))
    completed, commands = _run(tmp_path, environment={"CUBUTTERFLY_HARDWARE_MODEL": str(explicit)})
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands
    assert (local / "hardware_model.json").read_bytes() == original
    decision = json.loads((local / "installation_profile_selection.json").read_text())
    assert decision["runtime_hardware_model"] == str(explicit)
    assert decision["selection"]["selected"]["path"] == str(explicit)
    assert decision["selection"]["selected"]["provenance"] == document["provenance"]
    activated = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "$CUBUTTERFLY_HARDWARE_MODEL"',
        "test", str(local / "profile_environment.sh")], text=True, capture_output=True, check=True)
    assert activated.stdout.strip() == str(explicit)


def test_invalid_explicit_file_cannot_fall_back_to_complete_bundle(tmp_path):
    bundle = tmp_path / "bundled"
    target = bundle / "Test-GPU-sm80-42949672960B"
    target.mkdir(parents=True)
    (target / "hardware_model.json").write_text(json.dumps(_hardware()))
    explicit = tmp_path / "wrong-caps.json"
    explicit.write_text(json.dumps(_hardware(memory=85899345920)))
    completed, commands = _run(tmp_path, bundled_root=bundle,
                               environment={"CUBUTTERFLY_HARDWARE_MODEL": str(explicit)})
    assert completed.returncode != 0
    assert "invalid hardware artifact" in completed.stderr
    assert "calibrate " not in commands
    assert not (tmp_path / "profiles" / target.name / "hardware_model.json").exists()


def test_explicit_file_cannot_overwrite_incompatible_existing_local_artifact(tmp_path):
    local = _profile(tmp_path, memory=85899345920)
    original = (local / "hardware_model.json").read_bytes()
    explicit = tmp_path / "requested.json"
    explicit.write_text(json.dumps(_hardware()))
    completed, commands = _run(tmp_path, environment={"CUBUTTERFLY_HARDWARE_MODEL": str(explicit)})
    assert completed.returncode != 0
    assert "cannot replace an existing incompatible local profile" in completed.stderr
    assert "calibrate " not in commands
    assert (local / "hardware_model.json").read_bytes() == original


def test_runtime_profile_root_environment_is_used_when_cli_root_is_absent(tmp_path):
    local = _profile(tmp_path)
    completed, commands = _run(tmp_path, use_environment_root=True,
                               environment={"CUBUTTERFLY_PROFILE_ROOT": str(local.parent)})
    assert completed.returncode == 0, completed.stderr
    assert "calibrate " not in commands
    assert (local / "installation_profile_selection.json").exists()


def test_explicit_refresh_isolates_inherited_profile_environment_from_measurements(tmp_path):
    completed, commands = _run(tmp_path, "--refresh-profile", environment={
        "CUBUTTERFLY_HARDWARE_MODEL": str(tmp_path / "stale-missing.json"),
        "CUBUTTERFLY_PROFILE_ROOT": str(tmp_path / "unrequested-root")})
    assert completed.returncode == 0, completed.stderr
    assert "calibration-env model=unset root=unset" in commands
    assert "--force" in commands
    assert "inherited static profile environment is isolated" in completed.stdout
