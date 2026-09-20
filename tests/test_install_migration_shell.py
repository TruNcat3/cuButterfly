import os
import pathlib
import stat
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_with_hardware_profile.sh"


def _executable(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _mock_tools(tmp_path: pathlib.Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"

    _executable(bin_dir / "nvidia-smi", """#!/usr/bin/env bash
if [[ "$*" == *memory.total* ]]; then
    printf 'Test GPU, 8.0, 40960\\n'
else
    printf '8.0\\n'
fi
""")
    _executable(bin_dir / "cmake", """#!/usr/bin/env bash
set -euo pipefail
{ printf 'cmake'; printf ' %q' "$@"; printf '\\n'; } >> "$MOCK_LOG"
build=''
is_build=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == '-B' && $((i + 1)) -lt ${#args[@]} ]]; then
        build="${args[i + 1]}"
    elif [[ "${args[i]}" == '--build' && $((i + 1)) -lt ${#args[@]} ]]; then
        build="${args[i + 1]}"
        is_build=1
    fi
done
if [[ -n "$build" ]]; then
    mkdir -p "$build"
    if [[ ! -f "$build/CMakeCache.txt" ]]; then
        printf 'CMAKE_CUDA_ARCHITECTURES:STRING=80\\n' > "$build/CMakeCache.txt"
    fi
    if (( is_build )); then
        cat > "$build/cubutterfly_bench" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--device-identity" ]]; then
    printf 'device,compute_capability,global_memory_bytes\\nTest GPU,8.0,42949672960\\n'
    exit 0
fi
exit 0
EOF
        chmod +x "$build/cubutterfly_bench"
    fi
fi
""")
    _executable(bin_dir / "ctest", """#!/usr/bin/env bash
set -euo pipefail
{ printf 'ctest'; printf ' %q' "$@"; printf '\\n'; } >> "$MOCK_LOG"
""")

    real_python = pathlib.Path(sys.executable).resolve()
    _executable(bin_dir / "mock-python", f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == *"/scripts/calibrate_hardware.py" ]]; then
    {{ printf 'calibrate'; printf ' %q' "$@"; printf '\\n'; }} >> "$MOCK_LOG"
    profile=''
    args=("$@")
    for ((i=0; i<${{#args[@]}}; i++)); do
        if [[ "${{args[i]}}" == '--profile-dir' && $((i + 1)) -lt ${{#args[@]}} ]]; then
            profile="${{args[i + 1]}}"
        fi
    done
    mkdir -p "$profile"
    printf '{{"status":"complete"}}\\n' > "$profile/migration_manifest.json"
    exit 0
fi
exec {real_python} "$@"
""")
    return bin_dir, log


def _run_installer(tmp_path, args):
    bin_dir, log = _mock_tools(tmp_path)
    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "PYTHON": str(bin_dir / "mock-python"),
        "MOCK_LOG": str(log),
    })
    result = subprocess.run(
        [str(INSTALLER), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return log.read_text()


def test_fresh_install_resolves_identity_and_forwards_verification(tmp_path):
    profile_root = tmp_path / "profiles"
    log = _run_installer(tmp_path, [
        "--build-dir", str(tmp_path / "build"),
        "--prefix", str(tmp_path / "prefix"),
        "--profile-root", str(profile_root),
        "--verify-batches", "2",
        "--skip-tests",
    ])

    profile = profile_root / "Test-GPU-sm80-42949672960B"
    assert (profile / "migration_manifest.json").is_file()
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in log
    assert "-DCUBUTTERFLY_FFT_CODEGEN_MODE=precompiled" in log
    calibration = next(line for line in log.splitlines() if line.startswith("calibrate "))
    assert f"--profile-dir {profile}" in calibration
    assert "--verify-batches 2" in calibration
    assert "--seed-budget 8" in calibration
    assert "--force" in calibration
    assert "verify_local_selector.py" not in log


def test_existing_build_only_reconfigures_explicit_overrides_and_resumes(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_ARCHITECTURES:STRING=70\n"
        "CUBUTTERFLY_FFT_CODEGEN_MODE:STRING=precompiled\n"
    )
    log = _run_installer(tmp_path, [
        "--build-dir", str(build),
        "--profile-root", str(tmp_path / "profiles"),
        "--cuda-architectures", "90",
        "--fft-codegen-mode", "on-demand",
        "--verify-batches", "3",
        "--resume-search",
        "--skip-tests",
    ])

    configure = next(line for line in log.splitlines() if line.startswith("cmake ") and " -S " in line)
    assert "-DCMAKE_CUDA_ARCHITECTURES=90" in configure
    assert "-DCUBUTTERFLY_FFT_CODEGEN_MODE=on-demand" in configure
    calibration = next(line for line in log.splitlines() if line.startswith("calibrate "))
    assert "--resume-search" in calibration
    assert "--verify-batches 3" in calibration
    assert "--force" not in calibration


def test_existing_build_without_overrides_preserves_cache(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_ARCHITECTURES:STRING=86\n"
        "CUBUTTERFLY_FFT_CODEGEN_MODE:STRING=on-demand\n"
    )
    log = _run_installer(tmp_path, [
        "--build-dir", str(build),
        "--profile-root", str(tmp_path / "profiles"),
        "--skip-tests",
    ])

    configure_lines = [line for line in log.splitlines() if line.startswith("cmake ") and " -S " in line]
    assert configure_lines == []
    assert "-DCMAKE_CUDA_ARCHITECTURES=86" not in log
    assert "-DCUBUTTERFLY_FFT_CODEGEN_MODE=on-demand" not in log


def test_seed_budget_and_repeated_mapping_seeds_reach_calibrator(tmp_path):
    first = tmp_path / "first-seeds.json"
    second = tmp_path / "second-seeds.json"
    first.write_text("{}\n")
    second.write_text("{}\n")
    log = _run_installer(tmp_path, [
        "--build-dir", str(tmp_path / "build"),
        "--prefix", str(tmp_path / "prefix"),
        "--profile-root", str(tmp_path / "profiles"),
        "--seed-budget", "5",
        "--mapping-seeds", str(first),
        "--mapping-seeds", str(second),
        "--stage-import-checkpoint", str(first),
        "--stage-import-checkpoint", str(second),
        "--skip-tests",
    ])

    calibration = next(line for line in log.splitlines() if line.startswith("calibrate "))
    assert "--seed-budget 5" in calibration
    assert calibration.count("--mapping-seeds") == 2
    assert calibration.count("--stage-import-checkpoint") == 2
    assert str(first) in calibration
    assert str(second) in calibration


def test_stage_modes_and_default_search_budgets_reach_calibrator(tmp_path):
    cases = (
        ("full", "0", "0"),
        ("bounded", "300", "600"),
        ("skip", "300", "600"),
    )
    for mode, seconds, compile_seconds in cases:
        case_root = tmp_path / mode
        case_root.mkdir()
        log = _run_installer(case_root, [
            "--build-dir", str(case_root / "build"),
            "--prefix", str(case_root / "prefix"),
            "--profile-root", str(case_root / "profiles"),
            "--stage-calibration", mode,
            "--skip-tests",
        ])
        calibration = next(line for line in log.splitlines()
                           if line.startswith("calibrate "))
        assert f"--stage-calibration {mode}" in calibration
        assert f"--search-seconds {seconds}" in calibration
        assert f"--compile-seconds {compile_seconds}" in calibration
