import json
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_xd_expanded as benchmark  # noqa: E402


RUNTIME = {
    "gpu_uuid": "gpu-a",
    "cuda_visible_devices": "GPU-a",
    "cuda_visible_uuid": "gpu-a",
    "compile_mode": "research",
}
CUTOFF_TEXT = "2026-10-05T00:45:24+08:00"


@pytest.fixture
def tmp_path(tmpdir):
    # The repository's supported pytest is old enough to expose tmpdir only.
    return pathlib.Path(str(tmpdir))


def _args(library, vkfft_library):
    return SimpleNamespace(library=library, vkfft_library=vkfft_library)


def _manifest(tmp_path):
    library = tmp_path / "libcuntt.so"
    vkfft = tmp_path / "libvkfft_xd.so"
    library.write_bytes(b"library-a")
    vkfft.write_bytes(b"vkfft-a")
    manifest = benchmark._make_manifest(
        _args(library, vkfft), benchmark.cases(), RUNTIME
    )
    return library, vkfft, manifest


def _row(name, vkfft_sha):
    row = {
        "requested": name,
        "correctness": {"max_abs": 0.0},
        "graph_correctness": {"max_abs": 0.0},
        "graph": {"graph_used": True, "graph_capture_calls": 20,
                  "graph_replays": 7, "samples_ms": [1.0] * 7, "median_ms": 1.0},
        "input_preserved": True,
        "status": "passed",
    }
    if name == "vkfft":
        row["baseline_metadata"] = {"library_sha256": vkfft_sha}
    return row


def _report(case, manifest, *, gpu="gpu-a", compile_mode="research",
            status="passed"):
    return {
        "shape": case["shape"],
        "batch": case["batch"],
        "precision": case["precision"],
        "inverse": case["inverse"],
        "gpu": {"uuid": gpu},
        "library_sha256": manifest["library_sha256"],
        "compile_mode": compile_mode,
        "protocol": benchmark.REPORT_PROTOCOL,
        "rows": [_row(name, manifest["vkfft_library_sha256"])
                  for name in ("selected", "cufft-native", "vkfft")],
        "status": status,
    }


def test_reuses_only_complete_old_report_and_records_provenance(tmp_path, monkeypatch):
    library, vkfft, manifest = _manifest(tmp_path)
    source = tmp_path / "source"
    fresh = tmp_path / "fresh"
    source.mkdir()
    # Model the pre-recovery manifest emitted by the interrupted run.
    old_manifest = dict(manifest)
    for key in ("gpu_uuid", "cuda_visible_devices", "cuda_visible_uuid", "compile_mode",
                "timing_protocol"):
        old_manifest.pop(key, None)
    (source / "manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
    first = benchmark.cases()[0]
    old_report = source / f"{first['id']}.json"
    old_report.write_text(json.dumps(_report(first, manifest)), encoding="utf-8")
    cutoff, _ = benchmark._parse_cutoff(CUTOFF_TEXT)
    os.utime(old_report, (cutoff - 10.0, cutoff - 10.0))

    monkeypatch.setattr(benchmark, "_runtime_identity", lambda **_: dict(RUNTIME))
    rerun_calls = []

    def fake_run(command, **_kwargs):
        rerun_calls.append(command)
        output = pathlib.Path(command[command.index("--output") + 1])
        shape_start = command.index("--shape") + 1
        batch_index = command.index("--batch")
        shape = [int(value) for value in command[shape_start:batch_index]]
        batch = int(command[batch_index + 1])
        precision = command[command.index("--precision") + 1]
        inverse = "--inverse" in command
        case = next(case for case in benchmark.cases()
                    if case["shape"] == shape and case["batch"] == batch
                    and case["precision"] == precision and case["inverse"] == inverse)
        output.write_text(json.dumps(_report(case, manifest)), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    assert benchmark.main([
        "--library", str(library), "--vkfft-library", str(vkfft),
        "--output", str(fresh), "--resume-from", str(source),
        "--resume-before", CUTOFF_TEXT,
    ]) == 0

    assert len(rerun_calls) == 63
    completion = json.loads((fresh / "completion.json").read_text())
    assert completion["completed"] == 64
    assert completion["failures"] == []
    assert completion["reused_ids"] == [first["id"]]
    resume = json.loads((fresh / "resume.json").read_text())
    assert resume["source"] == str(source.resolve())
    assert resume["cutoff"] == CUTOFF_TEXT
    assert resume["reused_ids"] == [first["id"]]
    assert resume["reused"][0]["source"] == str(old_report.resolve())
    assert json.loads((fresh / old_report.name).read_text()) == json.loads(old_report.read_text())


def test_cutoff_and_incomplete_reports_are_rerun_candidates(tmp_path):
    library, vkfft, manifest = _manifest(tmp_path)
    case = benchmark.cases()[0]
    report_path = tmp_path / "case.json"
    report_path.write_text(json.dumps(_report(case, manifest)), encoding="utf-8")
    cutoff, _ = benchmark._parse_cutoff(CUTOFF_TEXT)

    os.utime(report_path, (cutoff + 1.0, cutoff + 1.0))
    assert not benchmark._reusable_case(report_path, case, manifest, RUNTIME, cutoff)

    os.utime(report_path, (cutoff - 1.0, cutoff - 1.0))
    incomplete = _report(case, manifest)
    incomplete["rows"][1].pop("graph")
    report_path.write_text(json.dumps(incomplete), encoding="utf-8")
    os.utime(report_path, (cutoff - 1.0, cutoff - 1.0))
    assert not benchmark._reusable_case(report_path, case, manifest, RUNTIME, cutoff)

    report_path.write_text("{not-json", encoding="utf-8")
    os.utime(report_path, (cutoff - 1.0, cutoff - 1.0))
    assert not benchmark._reusable_case(report_path, case, manifest, RUNTIME, cutoff)


def test_hidden_or_invalid_device_does_not_fall_back_to_gpu_zero():
    physical = ["gpu-a", "gpu-b"]
    for visible in ("", "-1", "99"):
        assert benchmark._select_gpu_uuid(visible, physical) is None
    assert benchmark._select_gpu_uuid("1", physical) == "gpu-b"
    assert benchmark._select_gpu_uuid("GPU-b", physical) == "gpu-b"


def test_incomplete_shape_and_nonfinite_timing_are_not_reused(tmp_path):
    _, _, manifest = _manifest(tmp_path)
    case = benchmark.cases()[0]
    report = _report(case, manifest)
    report["shape"] = None
    assert not benchmark._complete_report(report, case, manifest, RUNTIME)
    report = _report(case, manifest)
    report["rows"][0]["graph"]["samples_ms"][0] = float("nan")
    assert not benchmark._complete_report(report, case, manifest, RUNTIME)


def test_bad_library_and_gpu_identity_are_rejected(tmp_path):
    library, vkfft, manifest = _manifest(tmp_path)
    case = benchmark.cases()[0]
    cutoff, _ = benchmark._parse_cutoff(CUTOFF_TEXT)
    report_path = tmp_path / "case.json"
    report_path.write_text(json.dumps(_report(case, manifest)), encoding="utf-8")
    os.utime(report_path, (cutoff - 1.0, cutoff - 1.0))

    bad_library = dict(manifest, library_sha256="wrong-library")
    with pytest.raises(benchmark.ResumeError, match="library SHA mismatch"):
        benchmark._reusable_case(report_path, case, bad_library, RUNTIME, cutoff)

    bad_gpu_report = _report(case, manifest, gpu="gpu-b")
    report_path.write_text(json.dumps(bad_gpu_report), encoding="utf-8")
    os.utime(report_path, (cutoff - 1.0, cutoff - 1.0))
    with pytest.raises(benchmark.ResumeError, match="gpu UUID mismatch"):
        benchmark._reusable_case(report_path, case, manifest, RUNTIME, cutoff)


@pytest.mark.parametrize("field", ["population", "cases", "library_sha256", "baselines",
                                    "vkfft_library_sha256"])
def test_manifest_identity_mismatch_is_rejected(tmp_path, field):
    library, vkfft, manifest = _manifest(tmp_path)
    source = dict(manifest)
    if field == "population":
        source[field] = "different-population"
    elif field == "cases":
        source[field] = source[field][:-1]
    elif field == "baselines":
        source[field] = ["cufft-native"]
    else:
        source[field] = "different-sha"
    with pytest.raises(benchmark.ResumeError, match=f"manifest mismatch for {field}"):
        benchmark._validate_source_manifest(source, manifest)


def test_resume_arguments_must_be_paired(tmp_path):
    library = tmp_path / "libcuntt.so"
    library.write_bytes(b"library-a")
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--library", str(library), "--output", str(tmp_path / "out"),
                               "--resume-from", str(tmp_path / "source")])
