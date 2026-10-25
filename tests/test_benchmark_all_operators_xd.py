import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_all_operators_xd as benchmark  # noqa: E402


@pytest.fixture
def tmp_path(tmpdir):
    return pathlib.Path(str(tmpdir))


def test_gpu_info_uses_requested_selector(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="GPU-actual,Test GPU,40960,8.0\n",
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    info = benchmark._gpu_info("GPU-cli")

    assert commands == [[
        "nvidia-smi", "--id=GPU-cli",
        "--query-gpu=uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]]
    assert info["uuid"] == "GPU-actual"
    assert info["selector"] == "GPU-cli"


def test_parse_output_keeps_commas_inside_algorithm_and_parses_correctness():
    output = (
        "algorithm=xd[axis,cooperative],logical_shape=4x8,"
        "physical_shape=4x8,batch=2,workspace_bytes=128,trials=3,"
        "kernel_ms=1.25,verified_batches=2,correct=1\n"
    )

    parsed = benchmark._parse_output(output)

    assert parsed["algorithm"] == "xd[axis,cooperative]"
    assert parsed["kernel_ms"] == 1.25
    assert parsed["trials"] == 3
    assert parsed["verified_batches"] == 2
    assert parsed["correct"] is True


def test_parse_output_preserves_resolved_mapping_json():
    mapping = '{"kind":"xd","lowering":"axis-tiled","physical":{"axes":[]}}'
    output = (
        "algorithm=xd[axis-tiled],logical_shape=4x8,physical_shape=4x8,"
        "batch=1,workspace_bytes=0,trials=1,kernel_ms=2.0,"
        f"mapping_json={mapping}\n"
    )

    parsed = benchmark._parse_output(output)

    assert parsed["mapping_json"] == mapping
    assert json.loads(parsed["mapping_json"])["lowering"] == "axis-tiled"


def test_timing_protocol_reflects_cli_values():
    protocol = benchmark._timing_protocol(warmup=7, repeat=11, trials=3)

    assert "warmup=7" in protocol
    assert "repeat=11" in protocol
    assert "trials=3" in protocol


def _row(configuration, kernel_ms, *, status="measured", operator="fft"):
    return {
        "id": f"{configuration}-{kernel_ms}",
        "operator": operator,
        "precision": "fp32",
        "shape": [4, 8],
        "batch": 1,
        "inverse": False,
        "configuration": configuration,
        "status": status,
        "kernel_ms": kernel_ms,
    }


def test_summary_uses_true_even_medians_and_separates_measured_rows():
    rows = [
        _row("selected", 1.0),
        _row("selected", 2.0),
        _row("resident-t128", 2.0),
        _row("resident-t128", 4.0),
        _row("selected", 99.0, status="failed"),
    ]

    summary = benchmark._summary(rows)

    assert summary["requested"] == 5
    assert summary["completed"] == 4
    assert summary["passed"] == 0
    assert summary["measured"] == 4
    assert summary["failed"] == 1
    assert summary["operators"]["fft"]["median_kernel_ms"] == 2.0
    assert summary["configurations"]["selected"]["median_kernel_ms"] == 1.5
    assert summary["configurations"]["resident-t128"]["median_kernel_ms"] == 3.0
    assert summary["selector_vs_resident_t128"]["median_speedup"] == 2.0


def test_runtime_environment_injects_requested_library_and_gpu(tmp_path, monkeypatch):
    library = tmp_path / "libcuntt.so"
    library.write_bytes(b"library")
    args = SimpleNamespace(
        library=library,
        gpu_uuid="GPU-test",
        compile_mode="research",
    )
    monkeypatch.setenv("LD_PRELOAD", "/previous.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/previous/lib")

    environment = benchmark._runtime_environment(args)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert environment["CUBUTTERFLY_COMPILE_MODE"] == "research"
    assert "LD_PRELOAD" not in environment
    assert environment["LD_LIBRARY_PATH"] == f"{library.resolve().parent}:/previous/lib"


def test_runtime_library_resolution_accepts_versioned_same_file(tmp_path, monkeypatch):
    bench = tmp_path / "cubutterfly_plan_bench"
    versioned = tmp_path / "libcuntt.so.1"
    soname = tmp_path / "libcuntt.so"
    bench.write_bytes(b"bench")
    versioned.write_bytes(b"library")
    soname.symlink_to(versioned.name)

    def fake_tags(path):
        if pathlib.Path(path) == bench:
            return {"needed": ["libcuntt.so"], "soname": None,
                    "runpath": "$ORIGIN", "rpath": ""}
        return {"needed": [], "soname": "libcuntt.so",
                "runpath": "", "rpath": ""}

    monkeypatch.setattr(benchmark, "_read_dynamic_tags", fake_tags)
    resolution = benchmark._verify_runtime_library(
        bench, versioned, {"LD_LIBRARY_PATH": str(tmp_path)}
    )

    assert resolution["soname"] == "libcuntt.so"
    assert pathlib.Path(resolution["resolved"]) == versioned.resolve()


def test_runtime_library_resolution_rejects_wrong_soname(tmp_path, monkeypatch):
    bench = tmp_path / "cubutterfly_plan_bench"
    library = tmp_path / "libcuntt.so"
    bench.write_bytes(b"bench")
    library.write_bytes(b"library")

    def fake_tags(path):
        if pathlib.Path(path) == bench:
            return {"needed": ["libcuntt.so"], "soname": None,
                    "runpath": "$ORIGIN", "rpath": ""}
        return {"needed": [], "soname": "libcuntt.so.other",
                "runpath": "", "rpath": ""}

    monkeypatch.setattr(benchmark, "_read_dynamic_tags", fake_tags)
    with pytest.raises(ValueError, match="SONAME"):
        benchmark._verify_runtime_library(
            bench, library, {"LD_LIBRARY_PATH": str(tmp_path)}
        )


def test_runtime_library_resolution_honors_legacy_rpath_precedence(tmp_path, monkeypatch):
    bench_dir = tmp_path / "bench"
    requested_dir = tmp_path / "requested"
    bench_dir.mkdir()
    requested_dir.mkdir()
    bench = bench_dir / "cubutterfly_plan_bench"
    rpath_library = bench_dir / "libcuntt.so"
    requested_library = requested_dir / "libcuntt.so"
    bench.write_bytes(b"bench")
    rpath_library.write_bytes(b"rpath-library")
    requested_library.write_bytes(b"requested-library")

    def fake_tags(path):
        if pathlib.Path(path) == bench:
            return {"needed": ["libcuntt.so"], "soname": None,
                    "runpath": "", "rpath": "$ORIGIN"}
        return {"needed": [], "soname": "libcuntt.so",
                "runpath": "", "rpath": ""}

    monkeypatch.setattr(benchmark, "_read_dynamic_tags", fake_tags)
    with pytest.raises(ValueError, match="not requested library"):
        benchmark._verify_runtime_library(
            bench, requested_library,
            {"LD_LIBRARY_PATH": str(requested_dir)},
        )


def test_parse_args_enforces_runtime_library_identity(tmp_path, monkeypatch):
    bench = tmp_path / "cubutterfly_plan_bench"
    library = tmp_path / "libcuntt.so"
    bench.write_bytes(b"bench")
    library.write_bytes(b"library")

    def reject(*_args):
        raise ValueError("runtime loader resolves a different library")

    monkeypatch.setattr(benchmark, "_verify_runtime_library", reject)
    with pytest.raises(SystemExit):
        benchmark._parse_args([
            "--bench", str(bench), "--library", str(library),
            "--output", str(tmp_path / "report"),
        ])


def test_run_case_without_correctness_is_measured(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "algorithm=xd[axis],logical_shape=4x8,physical_shape=4x8,"
                "batch=1,workspace_bytes=0,trials=1,kernel_ms=2.0\n"
            ),
            stderr="",
        )

    library = tmp_path / "libcuntt.so"
    library.write_bytes(b"library")
    args = SimpleNamespace(
        bench=tmp_path / "bench",
        library=library,
        gpu_uuid="GPU-test",
        compile_mode="research",
        warmup=2,
        repeat=5,
        trials=1,
        policy="measure",
        modulus=97,
        matrix="1,0.25,-0.5,1",
        compare_cufft=False,
        keep_logs=False,
        output=tmp_path,
    )
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    row = benchmark._run_case(args, "fft", "fp32", (4, 8), 1, False, "selected")

    assert row["status"] == "measured"
    assert "correct" not in row
    assert row["mapping_evidence"]["status"] == "unavailable"
    assert row["mapping_evidence"]["bound"] is False
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert "LD_PRELOAD" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["LD_LIBRARY_PATH"].startswith(
        str(library.resolve().parent)
    )


def test_run_case_failed_correctness_is_not_passed(monkeypatch, tmp_path):
    def fake_run(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "algorithm=xd[axis],logical_shape=4x8,physical_shape=4x8,"
                "batch=1,workspace_bytes=0,trials=1,kernel_ms=2.0,"
                "verified_batches=1,correct=0\n"
            ),
            stderr="",
        )

    library = tmp_path / "libcuntt.so"
    library.write_bytes(b"library")
    args = SimpleNamespace(
        bench=tmp_path / "bench",
        library=library,
        gpu_uuid="GPU-test",
        compile_mode="research",
        warmup=0,
        repeat=1,
        trials=1,
        policy="measure",
        modulus=97,
        matrix="1,0.25,-0.5,1",
        compare_cufft=False,
        keep_logs=False,
        output=tmp_path,
    )
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    row = benchmark._run_case(args, "fft", "fp32", (4, 8), 1, False, "selected")

    assert row["status"] == "failed"
    assert row["correct"] is False
    assert "correctness" in row["error"]


def test_resident_fallback_is_unavailable_not_a_control(monkeypatch, tmp_path):
    def fake_run(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "algorithm=xd[flattened],logical_shape=64x64,physical_shape=4096x0,"
                "batch=1,workspace_bytes=0,trials=1,kernel_ms=2.0\n"
            ),
            stderr="",
        )

    library = tmp_path / "libcuntt.so"
    library.write_bytes(b"library")
    args = SimpleNamespace(
        bench=tmp_path / "bench",
        library=library,
        gpu_uuid="GPU-test",
        warmup=0,
        repeat=1,
        trials=1,
        policy="measure",
        modulus=97,
        matrix="1,0.25,-0.5,1",
        compare_cufft=False,
        verify=False,
        keep_logs=False,
        output=tmp_path,
    )
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    row = benchmark._run_case(args, "fwht", "fp32", (64, 64), 1, False,
                              "resident-t128")

    assert row["status"] == "unavailable"
    assert "flattened" in row["error"]


def test_coverage_profile_is_rank2_rank3_matrix(tmp_path, monkeypatch):
    library = tmp_path / "libcuntt.so"
    bench = tmp_path / "cubutterfly_plan_bench"
    library.write_bytes(b"library")
    bench.write_bytes(b"bench")
    monkeypatch.setattr(
        benchmark, "_gpu_info", lambda selector: {"uuid": selector, "model": "cpu-test"}
    )
    output = tmp_path / "report"
    assert benchmark.main([
        "--bench", str(bench), "--library", str(library),
        "--output", str(output), "--profile", "coverage", "--dry-run",
    ]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["profile"] == "coverage"
    assert {len(shape) for shape in manifest["shapes"]} == {2, 3}
    assert manifest["batches"] == [1, 4, 16]
    rows = json.loads((output / "rows.json").read_text())
    # 10 precision/operator pairs * 4 shapes * 3 batches * 2 directions
    # * 2 configurations.
    assert len(rows) == 480


def test_dry_run_manifest_records_actual_timing_and_provenance(tmp_path, monkeypatch):
    library = tmp_path / "libcuntt.so"
    bench = tmp_path / "cubutterfly_plan_bench"
    library.write_bytes(b"library")
    bench.write_bytes(b"bench")
    monkeypatch.setattr(
        benchmark, "_gpu_info", lambda selector: {"uuid": selector, "model": "cpu-test"}
    )

    output = tmp_path / "report"
    assert benchmark.main([
        "--bench", str(bench), "--library", str(library),
        "--output", str(output), "--gpu-uuid", "GPU-cli",
        "--warmup", "7", "--repeat", "11", "--trials", "3", "--dry-run",
    ]) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["gpu_selector"] == "GPU-cli"
    assert manifest["gpu"]["uuid"] == "GPU-cli"
    assert manifest["bench_sha256"] == benchmark._sha256(bench)
    assert manifest["library_sha256"] == benchmark._sha256(library)
    assert manifest["timing"] == {"warmup": 7, "repeat": 11, "trials": 3}
    assert "warmup=7" in manifest["timing_protocol"]
    assert "repeat=11" in manifest["timing_protocol"]
    assert "trials=3" in manifest["timing_protocol"]
