import json
import os
import pathlib
import stat
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_research_target.sh"


def _executable(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run(args, *, env=None):
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_help_describes_cpu_only_v100_entry():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "--cuda-arch70" in result.stdout
    assert "ctest -N" in result.stdout
    assert "--install" in result.stdout
    assert "--json-root" in result.stdout
    assert "BF16" in result.stdout


def test_dry_run_is_portable_and_does_not_create_target_or_install(tmp_path):
    target = tmp_path / "research target with spaces"
    prefix = tmp_path / "current environment must remain untouched"
    mathdx = tmp_path / "mathdx checkout with spaces"
    json_root = tmp_path / "json checkout with spaces"
    result = _run([
        "--repo-dir", str(ROOT),
        "--build-dir", str(target),
        "--prefix", str(prefix),
        "--mathdx-root", str(mathdx),
        "--json-root", str(json_root),
        "--nvcc", str(tmp_path / "cuda toolkit" / "bin" / "nvcc"),
        "--python", str(tmp_path / "conda env" / "bin" / "python"),
        "--jobs", "3",
        "--dry-run",
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists()
    assert not prefix.exists()
    assert "-DCMAKE_CUDA_ARCHITECTURES=70" in result.stdout
    assert "-DCUBUTTERFLY_ENABLE_CUFFTDX=ON" in result.stdout
    assert "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=" in result.stdout
    assert "crypto-ntt-sm70" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=" in result.stdout
    assert "--parallel 3" in result.stdout
    assert f"-DCMAKE_INSTALL_PREFIX={prefix}".replace(" ", "\\ ") in result.stdout
    assert "BF16: emulated storage/rounding only" in result.stdout
    assert "install" in result.stdout
    assert "A100" not in result.stdout


def test_dry_run_with_vkfft_and_explicit_create_env_is_non_mutating(tmp_path):
    env_prefix = tmp_path / "env with spaces"
    result = _run([
        "--repo-dir", str(ROOT),
        "--build-dir", str(tmp_path / "target"),
        "--env-prefix", str(env_prefix),
        "--create-env",
        "--with-vkfft",
        "--cuda-arch70",
        "--dry-run",
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    assert not env_prefix.exists()
    assert "create -p" in result.stdout
    assert str(env_prefix).replace(" ", "\\ ") in result.stdout
    assert f"-DPython3_EXECUTABLE={env_prefix}/bin/python".replace(" ", "\\ ") in result.stdout
    assert "-DCUBUTTERFLY_ENABLE_VKFFT=ON" in result.stdout
    assert "-DCUBUTTERFLY_VKFFT_ROOT=" in result.stdout


def test_invalid_architecture_and_jobs_fail_before_build(tmp_path):
    bad_arch = _run(["--build-dir", str(tmp_path / "bad arch"), "--cuda-arch", "sm70"])
    assert bad_arch.returncode == 2
    assert "invalid CUDA architecture" in bad_arch.stderr
    assert not (tmp_path / "bad arch").exists()

    bad_jobs = _run(["--build-dir", str(tmp_path / "bad jobs"), "--jobs", "0"])
    assert bad_jobs.returncode == 2
    assert "invalid jobs" in bad_jobs.stderr
    assert not (tmp_path / "bad jobs").exists()


def test_create_env_requires_an_explicit_prefix(tmp_path):
    result = _run(["--build-dir", str(tmp_path / "target"), "--create-env"])
    assert result.returncode == 2
    assert "--create-env requires --env-prefix" in result.stderr
    assert not (tmp_path / "target").exists()


def test_existing_non_sm70_cache_is_rejected_without_reconfigure(tmp_path):
    build = tmp_path / "existing target"
    build.mkdir()
    (build / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=80\n")
    result = _run(["--build-dir", str(build), "--dry-run"])
    assert result.returncode == 2
    assert "configured for CUDA architecture '80'" in result.stderr
    assert "sm70" in result.stderr


def test_missing_archived_selector_is_a_build_dependency(tmp_path):
    source = tmp_path / "reduced source checkout"
    source.mkdir()
    target = tmp_path / "target"
    result = _run([
        "--repo-dir", str(source),
        "--build-dir", str(target),
        "--dry-run",
    ])
    assert result.returncode == 2
    assert "missing required legacy runtime-selector CSV" in result.stderr
    assert "results/v100_scaling_full_summary.csv" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("arch", [70, 120])
def test_mocked_cpu_prepare_builds_crypto_bench_lists_tests_and_records_bf16(tmp_path, arch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    mathdx = tmp_path / "mathdx"
    (mathdx / "include").mkdir(parents=True)
    (mathdx / "include" / "cufftdx.hpp").write_text("// test header\n")
    json_root = tmp_path / "json root"
    (json_root / "include" / "nlohmann").mkdir(parents=True)
    (json_root / "include" / "nlohmann" / "json.hpp").write_text("// test header\n")
    nvcc = _executable(bin_dir / "nvcc", "#!/usr/bin/env bash\nexit 0\n")
    _executable(
        bin_dir / "cmake",
        """#!/usr/bin/env bash
set -euo pipefail
{ printf 'cmake'; printf ' %q' "$@"; printf '\n'; } >> "$MOCK_LOG"
mkdir -p "$MOCK_BUILD_DIR"
printf 'CMAKE_CUDA_ARCHITECTURES:STRING=70\\n' > "$MOCK_BUILD_DIR/CMakeCache.txt"
""",
    )
    _executable(
        bin_dir / "ctest",
        """#!/usr/bin/env bash
set -euo pipefail
{ printf 'ctest'; printf ' %q' "$@"; printf '\n'; } >> "$MOCK_LOG"
""",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "MOCK_LOG": str(log),
        "MOCK_BUILD_DIR": str(tmp_path / "v100 build with spaces"),
        "PYTHON": os.environ.get("PYTHON", "python3"),
    })
    build = tmp_path / "v100 build with spaces"
    prefix = tmp_path / "conda prefix"
    (prefix / "conda-meta").mkdir(parents=True)
    result = _run([
        "--build-dir", str(build),
        "--prefix", str(prefix), "--install", "--shared",
        "--fft-codegen-mode", "on-demand", "--cuda-arch", str(arch),
        "--mathdx-root", str(mathdx),
        "--json-root", str(json_root),
        "--nvcc", str(nvcc),
        "--jobs", "2",
    ], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((build / "research_target_manifest.json").read_text())
    portable = json.loads((build / "portable_architecture_manifest.json").read_text())
    source_manifest = json.loads((ROOT / "config/v09_architecture_manifest.json").read_text())
    missing_result_paths = [
        path for path in source_manifest["layers"]["runtime_selection"]["paths"]
        if path.startswith("results/")
    ]
    assert missing_result_paths
    assert all(not path.startswith("results/") for path in portable["layers"]["runtime_selection"]["paths"])
    assert portable["generated_artifact_paths"]["runtime_selection"] == missing_result_paths
    assert portable["portable_source_manifest"]["generated_artifacts_are_informational"] is True
    assert manifest["target"] == "CUDA"
    assert manifest["cuda_architecture"] == arch
    assert manifest["cufftdx_enabled"] is True
    assert manifest["runtime_selector"]["source"] == "legacy-build-input"
    assert manifest["runtime_selector"]["calibration_required"] is True
    assert manifest["architecture_manifest"]["portable"] == str((build / "portable_architecture_manifest.json").resolve())
    assert manifest["nlohmann_json_root"] == str(json_root.resolve())
    assert manifest["bf16"]["native_arithmetic"] is (False if arch == 70 else None)
    assert manifest["bf16"]["measurement_label"] == ("emulated_native" if arch == 70 else "unmeasured")
    for phase in ("activate", "deactivate"):
        hook = prefix / f"etc/conda/{phase}.d/cubutterfly_profile.sh"
        assert hook.read_bytes() == (ROOT / f"scripts/conda_{phase}_profile.sh").read_bytes()
    assert manifest["preparation"]["gpu_execution"] is False
    commands = log.read_text()
    assert f"-DCMAKE_CUDA_ARCHITECTURES={arch}" in commands
    assert "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=" in commands
    assert "-DCUBUTTERFLY_NLOHMANN_INCLUDE=" in commands
    assert "-DCUBUTTERFLY_ARCHITECTURE_MANIFEST=" in commands
    assert f"crypto-ntt-sm{arch}" in commands
    assert "ctest" in commands and "-N" in commands
    assert "--install" in commands
    assert "-DCUBUTTERFLY_BUILD_SHARED=ON" in commands
    assert "-DCUBUTTERFLY_FFT_CODEGEN_MODE=on-demand" in commands
    assert f"-DCMAKE_INSTALL_PREFIX={prefix}".replace(" ", "\\ ") in commands


def test_sm120_dry_run_plans_actual_support_probe_without_gpu(tmp_path):
    build = tmp_path / "blackwell"
    result = _run(["--cuda-arch", "120", "--build-dir", str(build), "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert not build.exists()
    assert "target: CUDA sm120" in result.stdout
    assert "-DCUBUTTERFLY_PROBE_ARCHITECTURES=120" in result.stdout
    assert "CuFFTDxTarget.cmake" in result.stdout
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in result.stdout
    assert "V100-compatible" not in result.stdout


def test_codegen_mode_rejects_unknown_before_build(tmp_path):
    target = tmp_path / "invalid"
    result = _run(["--build-dir", str(target), "--fft-codegen-mode", "unknown"])
    assert result.returncode == 2
    assert "invalid --fft-codegen-mode" in result.stderr
    assert not target.exists()
