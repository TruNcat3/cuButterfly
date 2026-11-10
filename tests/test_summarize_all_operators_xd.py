import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_all_operators_xd as summarizer  # noqa: E402


@pytest.fixture
def tmp_path(tmpdir):
    return Path(str(tmpdir))


def _write_run(tmp_path, manifest, rows):
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    return summarizer.summarize(tmp_path)[0]


def _row(
    operator="fft",
    precision="fp32",
    shape=(4, 8),
    batch=1,
    inverse=False,
    configuration="selected",
    status="passed",
    correct=True,
    kernel_ms=1.0,
    **fields,
):
    row = {
        "operator": operator,
        "precision": precision,
        "shape": list(shape),
        "batch": batch,
        "inverse": inverse,
        "configuration": configuration,
        "status": status,
        "kernel_ms": kernel_ms,
    }
    if correct is not None:
        row["correct"] = correct
    row.update(fields)
    return row


def test_expected_population_is_cartesian_product_and_passes_only_verified_rows(tmp_path):
    manifest = {
        "operators": [
            {"name": "fft", "precisions": ["fp32"]},
            {"name": "fwht", "precisions": ["fp32"]},
        ],
        "shapes": [[4, 8]],
        "batches": [1, 2],
        "directions": ["forward", "inverse"],
        "configurations": ["selected", "resident-t128"],
    }
    rows = []
    for operator in ("fft", "fwht"):
        for batch in (1, 2):
            for inverse in (False, True):
                for configuration in ("selected", "resident-t128"):
                    rows.append(
                        _row(
                            operator=operator,
                            batch=batch,
                            inverse=inverse,
                            configuration=configuration,
                        )
                    )

    report = _write_run(tmp_path, manifest, rows)

    assert report["requested"] == 16
    assert report["expected_keys"] == 16
    assert report["observed_rows"] == 16
    assert report["coverage"]["missing_count"] == 0
    assert report["coverage"]["duplicate_count"] == 0
    assert report["coverage"]["unexpected_count"] == 0
    assert report["coverage"]["failed_count"] == 0
    assert report["coverage"]["unverified_count"] == 0
    assert report["coverage"]["verified_complete"] is True
    assert report["coverage"]["status"] == "verified-complete"


def test_missing_duplicate_unexpected_failed_and_unverified_are_reported(tmp_path):
    manifest = {
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]],
        "batches": [1],
        "directions": ["forward", "inverse"],
        "configurations": ["selected", "resident-t128"],
    }
    rows = [
        _row(),
        _row(status="measured", correct=None, kernel_ms=2.0),
        _row(inverse=True, status="failed", correct=False),
        _row(
            inverse=True,
            configuration="resident-t128",
            status="backend_error",
            correct=False,
        ),
        _row(configuration="debug", status="measured", correct=None),
    ]

    report = _write_run(tmp_path, manifest, rows)
    coverage = report["coverage"]

    assert report["requested"] == 4
    assert report["observed_rows"] == 5
    assert coverage["missing_count"] == 1
    assert coverage["duplicate_count"] == 1
    assert coverage["unexpected_count"] == 1
    assert coverage["failed_count"] == 2
    assert coverage["unverified_count"] == 1
    assert coverage["verified_complete"] is False
    assert coverage["status"] == "incomplete-key-population"
    assert any(item["configuration"] == "resident-t128" for item in coverage["failed"])


def test_missing_manifest_configurations_are_inferred_from_rows_without_external_claim(tmp_path):
    manifest = {
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]],
        "batches": [1],
        "directions": ["forward", "inverse"],
    }
    rows = [
        _row(status="measured", correct=None),
        _row(inverse=True, status="measured", correct=None),
    ]

    report = _write_run(tmp_path, manifest, rows)

    assert report["configuration_inferred"] is True
    assert report["configuration_source"] == "rows.configuration"
    assert report["requested_configurations"] == ["selected"]
    assert report["requested"] == 2
    assert report["coverage"]["missing_count"] == 0
    assert report["coverage"]["unverified_count"] == 2
    assert report["coverage"]["status"] == "unverified"
    assert "external completeness is unknown" in report["configuration_contract"]["claim"]


def test_even_control_median_uses_true_median_and_pass_requires_correctness(tmp_path):
    manifest = {
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]],
        "batches": [1, 2],
        "directions": ["forward"],
        "configurations": ["selected", "resident-t128"],
    }
    rows = [
        _row(batch=1, configuration="selected", kernel_ms=1.0),
        _row(batch=1, configuration="resident-t128", kernel_ms=3.0),
        _row(batch=2, configuration="selected", kernel_ms=1.0),
        _row(batch=2, configuration="resident-t128", kernel_ms=1.0),
    ]

    report = _write_run(tmp_path, manifest, rows)

    assert report["controls"]["selector_vs_resident_median_speedup"] == 2.0
    assert report["controls"]["selector_vs_resident_t128"]["median_speedup"] == 2.0

    rows[0]["status"] = "passed"
    rows[0].pop("correct")
    unverified_report = _write_run(tmp_path, manifest, rows)
    assert unverified_report["coverage"]["unverified_count"] == 1
    assert unverified_report["coverage"]["verified_complete"] is False


def test_timing_only_rows_stay_in_coverage_but_do_not_enter_performance_stats(tmp_path):
    manifest = {
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
        "configurations": ["selected"],
    }
    report = _write_run(tmp_path, manifest, [_row(status="measured", correct=None)])

    assert report["coverage"]["unverified_count"] == 1
    assert report["selected_rows"] == 0
    assert report["controls"]["paired"] == 0
    assert report["fft_forward_vs_cufft"]["rows"] == 0


def test_declared_aliases_are_canonicalized_and_duplicate_cells_are_not_counted(tmp_path):
    manifest = {
        "operators": [
            {"name": "subset-zeta", "precisions": ["uint32"]},
            {"name": "xor-zeta", "precisions": ["uint32"]},
        ],
        "semantic_aliases": {"xor-zeta": "subset-zeta"},
        "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
        "configurations": ["selected"],
    }
    rows = [
        _row(operator="subset-zeta", precision="uint32"),
        _row(operator="xor-zeta", precision="uint32"),
    ]
    report = _write_run(tmp_path, manifest, rows)

    assert report["expected_keys"] == 1
    assert report["coverage"]["duplicate_count"] == 1
    assert report["selected_rows"] == 0
    assert list(report["operators"]) == ["subset-zeta"]


def test_runtime_and_protocol_mismatches_cannot_satisfy_the_population(tmp_path):
    manifest = {
        "bench": "/current/plan_bench", "library_sha256": "current",
        "timing": {"warmup": 2, "repeat": 25, "trials": 1},
        "policy": "measure",
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
        "configurations": ["selected"],
    }
    stale = _row(command=["/old/plan_bench", "--execution-mode", "xd",
                          "--warmup", "1", "--repeat", "25", "--trials", "1",
                          "--policy", "measure"], library_sha256="old")
    report = _write_run(tmp_path, manifest, [stale])

    assert report["coverage"]["runtime_mismatch_count"] == 1
    assert report["coverage"]["status"] == "runtime-mismatch"
    assert report["selected_rows"] == 0

    semantic_rows = [
        _row(normalization="none"),
        _row(normalization="inverse"),
    ]
    semantic_report = _write_run(tmp_path, {**manifest, "bench": None}, semantic_rows)
    assert semantic_report["coverage"]["semantic_conflict_count"] == 1
    assert semantic_report["coverage"]["status"] == "semantic-mismatch"
    assert semantic_report["selected_rows"] == 0


def test_mixed_runtime_fingerprints_are_not_pooled(tmp_path):
    manifest = {
        "operators": [{"name": "fft", "precisions": ["fp32"]}],
        "shapes": [[4, 8]], "batches": [1, 2], "directions": ["forward"],
        "configurations": ["selected"],
    }
    rows = [
        _row(batch=1, runtime_fingerprint="old"),
        _row(batch=2, runtime_fingerprint="current"),
    ]
    report = _write_run(tmp_path, manifest, rows)
    assert report["coverage"]["runtime_mismatch_count"] == 2
    assert report["selected_rows"] == 0


def test_separate_correctness_artifact_can_verify_timing_only_operator_rows(tmp_path):
    run = tmp_path / "xd_coverage"
    run.mkdir()
    manifest = {
        "operators": [{"name": "fwht", "precisions": ["fp32"]}],
        "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
        "configurations": ["selected"], "compile_mode": "research",
    }
    rows = [_row(operator="fwht", status="measured", correct=None, id="fwht-case")]
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "xd_coverage_correctness.json").write_text(
        json.dumps({"compile_mode": "research", "cases": [{"id": "fwht-case", "status": "passed"}]}),
        encoding="utf-8",
    )

    report = summarizer.summarize(run)[0]

    assert report["coverage"]["unverified_count"] == 1
    assert report["correctness"]["verified"] == 1
    assert report["correctness"]["trusted"] is True
    assert report["selected_rows"] == 0


@pytest.mark.parametrize("fault", [None, "mapping", "identity", "unchecked", "duplicate", "semantics"])
def test_timed_mapping_requires_matching_value_check(tmp_path, fault):
    run = tmp_path / "xd_coverage"
    run.mkdir()
    manifest = {
        "operators": [{"name": "fwht", "precisions": ["fp32"]}],
        "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
        "configurations": ["selected"], "compile_mode": "research", "library_sha256": "current",
    }
    mapping = {"execution_mode": "xd", "strategy": "resident", "tile_threads": 128}
    row = _row(operator="fwht", status="measured", correct=None, id="fwht-case", mapping_json=mapping)
    check = dict(row, status="passed", actual_result_checked=True, mapping_replay_verified=True,
                 source_mapping_json=json.dumps(mapping))
    evidence = {"compile_mode": "research", "library_sha256": "current", "cases": [check]}
    if fault == "mapping":
        check["mapping_json"] = dict(mapping, tile_threads=256)
    elif fault == "identity":
        evidence["library_sha256"] = "old"
    elif fault == "unchecked":
        check["actual_result_checked"] = False
    elif fault == "duplicate":
        evidence["cases"].append(dict(check))
    elif fault == "semantics":
        check["inverse"] = True
    (tmp_path / "xd_coverage_correctness.json").write_text(json.dumps(evidence))
    report = _write_run(run, manifest, [row])
    assert report["mapping_bound_verified"] == (1 if fault is None else 0)
    assert report["coverage"]["verified_complete"] == (fault is None)
    assert summarizer.main([str(run), "--require-verified-complete"]) == (0 if fault is None else 1)


def test_conflicting_value_check_invalidates_inline_pass(tmp_path):
    mapping = {"strategy": "resident"}
    row = _row(id="case", mapping_json=mapping)
    manifest = {"operators": [{"name": "fft", "precisions": ["fp32"]}],
                "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
                "configurations": ["selected"], "compile_mode": "research", "library_sha256": "current"}
    check = dict(row, status="failed", source_mapping_json=mapping, actual_result_checked=True)
    (tmp_path / "correctness.json").write_text(json.dumps({
        "compile_mode": "research", "library_sha256": "current", "cases": [check]}))
    report = _write_run(tmp_path, manifest, [row])
    assert report["selected_rows"] == 0
    assert report["coverage"]["failed_count"] == 1


def test_unverified_duplicate_cannot_disappear_before_statistics(tmp_path):
    manifest = {"operators": [{"name": "fft", "precisions": ["fp32"]}],
                "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
                "configurations": ["selected"]}
    report = _write_run(tmp_path, manifest, [_row(), _row(status="failed", correct=False)])
    assert report["coverage"]["duplicate_count"] == 1
    assert report["coverage"]["verified_count"] == 0
    assert report["selected_rows"] == 0


@pytest.mark.parametrize("latency", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_latency_cannot_be_verified_performance(tmp_path, latency):
    manifest = {"operators": [{"name": "fft", "precisions": ["fp32"]}],
                "shapes": [[4, 8]], "batches": [1], "directions": ["forward"],
                "configurations": ["selected"]}
    report = _write_run(tmp_path, manifest, [_row(kernel_ms=latency)])
    assert report["selected_rows"] == 0
    assert report["coverage"]["verified_complete"] is False
