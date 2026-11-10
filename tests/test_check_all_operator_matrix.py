"""CPU contract tests for scripts/check_all_operator_matrix.py."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_all_operator_matrix as matrix  # noqa: E402


def test_coverage_profile_is_the_requested_rank_batch_direction_matrix():
    cases = matrix._case_matrix("coverage", None, None, None, None, None)

    assert len(cases) == 240
    assert {case.operator for case in cases} == set(matrix.OPERATORS)
    assert {case.shape for case in cases} == {
        (8, 8), (32, 32), (4, 4, 4), (8, 8, 8),
    }
    assert {case.batch for case in cases} == {1, 4, 16}
    assert {case.inverse for case in cases} == {False, True}
    assert all(len(case.shape) in (2, 3) for case in cases)
    for operator in matrix.OPERATORS:
        observed = {case.precision for case in cases if case.operator == operator}
        assert observed == set(matrix.PRECISIONS[operator])


def test_cpu_cells_compare_actual_outputs_with_check_xd_references():
    np, torch, error = matrix._import_runtime()
    assert error is None
    cases = matrix._case_matrix(
        "coverage", None, None, ((4, 4),), (1,), None)
    records = [matrix._run_case(np, torch, case, 17) for case in cases]

    assert len(records) == 20
    assert {record["status"] for record in records} == {"passed"}
    assert all(record["actual_result_checked"] for record in records)
    assert all("reference" in record for record in records)


def test_scale_profile_reports_quadratic_ntt_limit_as_skipped():
    cases = matrix._case_matrix("scale", ("ntt",), None,
                                ((64, 64, 64),), (1,), None)
    np, torch, error = matrix._import_runtime()
    assert error is None
    records = [matrix._run_case(np, torch, case, 19) for case in cases]

    assert {record["status"] for record in records} == {"skipped"}
    assert all("quadratic" in record["reason"] for record in records)


def test_extended_checks_identify_the_existing_cuda_entrypoints():
    np, torch, error = matrix._import_runtime()
    assert error is None
    records = matrix._extended_cpu_checks(np, torch, 23)

    assert {record["name"] for record in records} == {
        "layout", "in-place", "embedding", "real",
    }
    assert {record["status"] for record in records} == {"passed"}
    assert all(record["cpu_equivalent"] for record in records)
    assert all(record["gpu_entrypoint"] for record in records)
