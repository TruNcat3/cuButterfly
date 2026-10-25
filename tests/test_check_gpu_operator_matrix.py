"""CPU-only contract tests for the GPU correctness matrix planner."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_gpu_operator_matrix as matrix  # noqa: E402


SOURCE_MAPPING = (
    '{"contract":{"batch":1},"kind":"xd",'
    '"lowering":"axis-tiled","physical":{"axes":[]},'
    '"schema_version":1}'
)


def test_coverage_and_scale_reuse_the_240_cell_benchmark_inventory():
    for profile in ("coverage", "scale"):
        cases = matrix._matrix_cases(profile)
        assert len(cases) == 240
        assert {case.configuration for case in cases} == {"selected"}
        assert {case.inverse for case in cases} == {False, True}
        assert {case.batch for case in cases} == {1, 4, 16}
        assert {case.shape for case in cases} == set(
            matrix.PROFILE_SHAPES[profile])


def test_resident_control_adds_a_distinct_checked_cell_set():
    cases = matrix._matrix_cases("coverage", include_resident=True)
    assert len(cases) == 480
    assert sum(case.configuration == "selected" for case in cases) == 240
    assert sum(case.configuration == "resident-t128" for case in cases) == 240
    assert len({case.name for case in cases}) == 480


def test_dry_run_does_not_initialize_cuda(monkeypatch, tmpdir, capsys):
    output = pathlib.Path(str(tmpdir)) / "report.json"
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(matrix, "_sha256", lambda _path: "test-hash")
    assert matrix.main([
        "--dry-run", "--max-cases", "1", "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "dry-run"
    assert report["gpu_used"] is False
    assert report["correctness"]["actual_gpu_results_checked"] is False
    assert report["summary"]["passed"] == 0
    assert json.loads(capsys.readouterr().out)["cases"][0]["status"] == "dry-run"


def test_require_source_mapping_rejects_legacy_rows_without_gpu_use(tmpdir, capsys):
    rows = [{
        "operator": "fft", "precision": "fp32", "shape": [8, 8],
        "batch": 1, "inverse": False, "configuration": "selected",
    }]
    source = pathlib.Path(str(tmpdir)) / "rows.json"
    source.write_text(json.dumps(rows))
    output = pathlib.Path(str(tmpdir)) / "report.json"

    assert matrix.main([
        "--input-run", str(source), "--require-source-mapping",
        "--output", str(output),
    ]) == 2

    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["mapping_evidence"]["status"] == "unavailable"
    assert "mapping_json" in report["error"]
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_input_run_preserves_only_explicit_workloads(tmpdir):
    rows = [
        {
            "operator": "fft", "precision": "fp32", "shape": [8, 8],
            "batch": 4, "inverse": False, "configuration": "selected",
        },
        {
            "operator": "fft", "precision": "fp32", "shape": [8, 8],
            "batch": 4, "inverse": True, "configuration": "resident-t128",
        },
    ]
    path = pathlib.Path(str(tmpdir)) / "rows.json"
    path.write_text(json.dumps(rows))

    selected = matrix._load_input_cases(path, include_resident=False)
    all_cases = matrix._load_input_cases(path, include_resident=True)
    assert len(selected) == 1
    assert selected[0].configuration == "selected"
    assert len(all_cases) == 2
    assert all_cases[1].configuration == "resident-t128"
    assert all_cases[0].source_row == rows[0]


def test_input_run_accepts_legacy_shape_text_without_mapping(tmpdir):
    rows = [{
        "operator": "fft", "precision": "fp32", "shape": "8x8",
        "batch": 1, "inverse": False, "configuration": "selected",
    }]
    path = pathlib.Path(str(tmpdir)) / "rows.json"
    path.write_text(json.dumps(rows))

    cases = matrix._load_input_cases(path, include_resident=False)

    assert cases[0].shape == (8, 8)
    record = matrix._record_base(cases[0], "hash")
    assert record["mapping_replay_verified"] is None
    assert record["mapping_evidence"]["status"] == "unavailable"
    assert record["mapping_evidence"]["bound"] is False


def _run_mock_gpu_case(monkeypatch, source_row, observed_mapping):
    captured = {}

    class FakeTensor:
        def clone(self):
            return self

        def numel(self):
            return 1

    class FakeCuda:
        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def empty_like(_source):
            return FakeTensor()

        @staticmethod
        def equal(_left, _right):
            return True

    class FakePlan:
        metadata = {
            "algorithm": "xd[mock]",
            "mapping_json": observed_mapping,
            "selection_reason": "mock",
            "workspace_bytes": 0,
            "input_bytes": 4,
            "output_bytes": 4,
        }

        def execute(self, _source, _output):
            return None

        def close(self):
            return None

    def fake_make_plan(_api, _handle, _torch, **spec):
        captured["spec"] = spec
        return FakePlan()

    monkeypatch.setattr(matrix, "_precision_types",
                        lambda *_args: (1, 2, object()))
    monkeypatch.setattr(matrix, "_gpu_random",
                        lambda *_args: FakeTensor())
    monkeypatch.setattr(matrix, "make_plan", fake_make_plan)
    monkeypatch.setattr(matrix, "_run_fft",
                        lambda *_args: {"reference": "mock", "error": {}})

    case = matrix.GpuCase(
        "fft", "fp32", (4, 8), 1, False, "selected", source_row=source_row,
    )
    record = matrix._run_gpu_case(
        None, None, FakeTorch(), None, case, "hash", 1, 257,
        matrix.DEFAULT_MATRIX,
    )
    return captured, record


def test_source_mapping_is_passed_as_explicit_plan_mapping(monkeypatch):
    captured, record = _run_mock_gpu_case(
        monkeypatch, {"mapping_json": SOURCE_MAPPING}, SOURCE_MAPPING,
    )

    assert captured["spec"]["mapping"] == SOURCE_MAPPING
    assert record["mapping_json"] == SOURCE_MAPPING
    assert record["source_mapping_json"] == SOURCE_MAPPING
    assert record["mapping_replay_verified"] is True
    assert record["mapping_evidence"]["status"] == "verified"
    assert record["actual_result_checked"] is True
    assert record["status"] == "passed"


def test_source_mapping_mismatch_fails_before_claiming_checked_result(monkeypatch):
    different_mapping = SOURCE_MAPPING.replace("axis-tiled", "pair-tiled")
    _captured, record = _run_mock_gpu_case(
        monkeypatch, {"mapping_json": SOURCE_MAPPING}, different_mapping,
    )

    assert record["mapping_replay_verified"] is False
    assert record["mapping_evidence"]["status"] == "mismatch"
    assert record["actual_result_checked"] is False
    assert record["status"] == "failed"
    assert "mapping replay mismatch" in record["error"]


def test_legacy_source_without_mapping_is_explicitly_unbound(monkeypatch):
    captured, record = _run_mock_gpu_case(
        monkeypatch, {"status": "measured"}, SOURCE_MAPPING,
    )

    assert captured["spec"]["mapping"] is None
    assert record["mapping_replay_verified"] is None
    assert record["mapping_evidence"]["status"] == "unavailable"
    assert record["mapping_evidence"]["bound"] is False
    assert record["status"] == "passed"


def test_vectorized_ntt_reference_matches_existing_reference():
    case = matrix.GpuCase("ntt", "uint64", (8, 8), 2, False)
    source = np.arange(2 * 8 * 8, dtype=np.uint64)
    actual = matrix._ntt_numpy_reference(np, source, case.shape, case.batch,
                                          257, False)
    from check_xd import _ntt_reference
    expected = np.asarray(_ntt_reference(source, case.shape, case.batch, 257, False),
                          dtype=np.uint64)
    np.testing.assert_array_equal(actual, expected)


def test_structured_matrix_uses_benchmark_single_matrix_contract():
    matrices = matrix._structured_matrices((32, 32, 8))
    assert matrices == tuple((tuple(matrix.DEFAULT_MATRIX),) for _ in range(3))


def test_skip_is_incomplete_and_nonfinite_values_are_rejected():
    summary = matrix._summary(
        [matrix.GpuCase("fft", "fp32", (8, 8), 1, False)],
        [{"id": "x", "status": "skipped"}],
    )
    assert matrix._status_for_summary(summary) == "incomplete"
    with __import__("pytest").raises(AssertionError, match="non-finite"):
        matrix._assert_finite(torch, torch.tensor([float("nan")]), "test")


def test_checkpoint_round_trip_keeps_only_terminal_case_records(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "checkpoint.json"
    case = matrix.GpuCase("fft", "fp32", (8, 8), 1, False)
    record = {"id": case.name, "status": "passed", "actual_result_checked": True}
    matrix._write_checkpoint(path, [case], {case.name: record}, "hash", "coverage",
                             257, matrix.DEFAULT_MATRIX)
    loaded = matrix._load_checkpoint(path, [case], "hash", "coverage", 257,
                                     matrix.DEFAULT_MATRIX)
    assert loaded[case.name]["status"] == "passed"
    assert json.loads(path.read_text())["status"] == "complete"
    assert matrix._load_checkpoint(path, [case], "other", "coverage", 257,
                                   matrix.DEFAULT_MATRIX) == {}


def test_checkpoint_reuse_requires_the_same_timed_mapping(tmp_path):
    path = tmp_path / "checkpoint.json"
    case = matrix.GpuCase("fft", "fp32", (8, 8), 1, False,
                          source_row={"mapping_json": SOURCE_MAPPING})
    record = {"id": case.name, "status": "passed", "actual_result_checked": True,
              "mapping_replay_verified": True, "source_mapping_json": SOURCE_MAPPING,
              "mapping_json": SOURCE_MAPPING}
    matrix._write_checkpoint(path, [case], {case.name: record}, "hash", "coverage",
                             257, matrix.DEFAULT_MATRIX)
    assert matrix._load_checkpoint(path, [case], "hash", "coverage", 257,
                                   matrix.DEFAULT_MATRIX)
    changed = matrix.GpuCase("fft", "fp32", (8, 8), 1, False,
                             source_row={"mapping_json": SOURCE_MAPPING.replace("axis-tiled", "pair-tiled")})
    assert matrix._load_checkpoint(path, [changed], "hash", "coverage", 257,
                                   matrix.DEFAULT_MATRIX) == {}
