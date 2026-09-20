import copy
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "tools"))

import summarize_experiments as summary


@pytest.fixture
def tmp_path(tmpdir):
    """Keep this suite compatible with the repository's older pytest."""
    return pathlib.Path(str(tmpdir))


def _identity():
    return {
        "device": {
            "device": "fixture-A100",
            "compute_capability": "8.0",
            "global_memory_bytes": "1234",
        },
        "compile_mode": "research",
        "workloads_sha256": "workloads",
        "profile_sha256": "profile",
        "source_stage_sha256": "stage",
        "binaries": {
            "cubutterfly_bench": "bench",
            "cuntt_bench": "ntt",
        },
        "protocol": {
            "search_budget": 16,
            "finalists": 3,
            "seed_budget": 4,
            "trials": 3,
            "warmup": 100,
            "repeat": 100,
            "verify_batches": 0,
        },
    }


def _sample(latency, *, correct=True, device="fixture-A100"):
    return {
        "device": device,
        "compute_capability": "8.0",
        "operator": "fft",
        "precision": "fp32",
        "logN": "8",
        "batch": "1",
        "placement": "out-of-place",
        "kernel_ms": latency,
        "correct": "1" if correct else "0",
        "mapping_json": '{"schema_version":1,"backend":"temporal-tile"}',
    }


def _ranking(*, before=False, bad_candidate=False):
    candidates = [
        {"name": "a", "observed_ms": 2.0, "predicted_ms": -1.0 if bad_candidate else 2.0},
        {"name": "b", "observed_ms": 3.0, "predicted_ms": 3.0},
    ]
    return {
        "status": "evaluated",
        "candidate_count": 2,
        "top1": bool(before),
        "top2": True,
        "latency_regret": 1.0,
        "pairwise_correct": 1,
        "pairwise_count": 1,
        "all_services_covered": True,
        "candidates": candidates,
        "unestimated": [],
        "observed_order": ["a", "b"],
        "predicted_order": ["a", "b"],
    }


def _good_cell(cell_id="good"):
    workload = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": 1,
        "placement": "out-of-place",
    }
    ours = _sample(2.0)
    baseline = _sample(4.0)
    return {
        "id": cell_id,
        "workload": workload,
        "status": "complete",
        "measurements": [
            {"name": "cuButterfly", "trial": 0, "correct": True, "sample": ours},
            {"name": "cuFFT", "trial": 0, "correct": True, "sample": baseline},
        ],
        "ranking_before_service_extension": _ranking(before=False),
        "ranking": _ranking(before=True),
        "service_gaps": [],
        "selector_replay": [{
            "name": "a",
            "correct": True,
            "mapping_matches": True,
            "sample": ours,
        }],
    }


def _search_fixture(workload, *, mismatch=False):
    identity = {
        "compile_mode": "wrong" if mismatch else "research",
        "verify_batches": 0,
        "binary_sha256": {"cubutterfly_bench": "other" if mismatch else "bench", "cuntt_bench": "ntt"},
    }
    return {
        "schema": "cubutterfly-search-coverage-v1",
        "scope": "fixture scope",
        "measurement_identity": identity,
        "binary_sha256": "other" if mismatch else "bench",
        "search_protocol": {"search_budget": 16, "seed_budget": 4, "search_strategy": "evolutionary"},
        "workloads": [{
            "workload": workload,
            "enumerated": 10,
            "screened": 4,
            "requested_screened_count": 4,
            "budget_omitted": 6,
            "generated_neighbors": 2,
            "candidate_source": "fixture",
            "seed_coverage": {"available": 1, "requested": 1, "attempted": 1, "confirmed": 1, "budget_omitted": 0},
            "candidates": [
                {"id": "a", "selection_index": 0, "selection_reason": "historical-seed", "status": "measured"},
                {"id": "b", "selection_index": 1, "selection_reason": "stratified-exploration", "status": "screened"},
                {"id": "bad", "selection_index": 2, "selection_reason": "evolutionary-model", "status": "unavailable"},
                {"id": "omitted", "status": "budget-omitted"},
            ],
            "complete": True,
        }],
    }


def _write_fixture(tmp_path, *, mismatch=False, bad_measurement=False, incomplete=False):
    root = tmp_path / "acceptance"
    cell = _good_cell()
    if bad_measurement:
        cell["measurements"][0]["sample"] = _sample(float("nan"))
    cells = [cell]
    if incomplete:
        pending = _good_cell("pending")
        pending["status"] = "pending"
        cells.append(pending)
    document = {
        "schema": "cubutterfly-research-acceptance-v1",
        "identity": _identity(),
        "scope": "fixture scope",
        "status": "interrupted",
        "cells": cells,
        "limitations": [],
    }
    cell_workload = cell["workload"]
    cell_dir = root / "cells" / "good"
    cell_dir.mkdir(parents=True)
    (root / "acceptance.json").write_text(json.dumps(document))
    (cell_dir / "search_coverage.json").write_text(json.dumps(_search_fixture(cell_workload, mismatch=mismatch)))
    (cell_dir / "confirmed_records.json").write_text(json.dumps([
        {"name": "a", "status": "measured", "correct": True, "median_kernel_ms": 2.0,
         "samples": [{"warmup": 10, "repeat": 10, "trials": 1}]},
        {"name": "b", "status": "screened", "correct": True, "median_kernel_ms": 3.0,
         "samples": [{"warmup": 10, "repeat": 10, "trials": 1}]},
    ]))
    return root


def test_summary_excludes_incomplete_cells_and_keeps_phases_separate(tmp_path):
    root = _write_fixture(tmp_path, incomplete=True)
    result = summary.summarize_acceptance(root)

    assert result["snapshot"]["expected_cells"] == 2
    assert result["snapshot"]["completed_cells"] == 1
    assert result["e1"]["completed_cells"] == 1
    assert result["e1"]["baselines"]["cuFFT"]["paired_cells"] == 1
    assert result["e6"]["before_service_extension"]["evaluated_cells"] == 1
    assert result["e6"]["after_service_extension"]["evaluated_cells"] == 1
    assert result["e6"]["whole_workload_holdout"]["complete"] is False
    assert result["provenance"]["summary_does_not_read_summary_json"] is True


def test_bad_latency_and_correctness_are_rejected_from_baselines(tmp_path):
    root = _write_fixture(tmp_path, bad_measurement=True)
    document = json.loads((root / "acceptance.json").read_text())
    document["cells"][0]["measurements"][1]["correct"] = False
    (root / "acceptance.json").write_text(json.dumps(document))
    result = summary.summarize_acceptance(root)

    assert result["e1"]["baselines"] == {}
    reasons = {item["reason"] for item in result["e1"]["invalid_measurements"]}
    assert "kernel-ms-not-positive-finite" in reasons
    assert "row-not-correct" in reasons


def test_mixed_search_cohort_is_rejected_but_acceptance_identity_is_retained(tmp_path):
    root = _write_fixture(tmp_path, mismatch=True)
    result = summary.summarize_acceptance(root)

    assert result["snapshot"]["identity"] == _identity()
    assert result["e9"]["trace_count"] == 1
    assert result["e9"]["traces"][0]["artifact_identity_status"] == "mismatch"
    assert result["provenance"]["identity_checks"]["mismatches"]
    assert result["e10"]["replay"]["verified_records"] == 0


def test_e9_trace_counts_failures_and_does_not_claim_algorithm_comparison(tmp_path):
    root = _write_fixture(tmp_path)
    result = summary.summarize_acceptance(root)
    trace = result["e9"]["traces"][0]

    assert trace["failure_count"] == 1
    assert trace["seed_coverage"]["attempted"] == 1
    assert trace["budget_omitted"] == 6
    assert result["e9"]["status"] == "not-ready"
    assert result["e9"]["costs"]["online_search_elapsed_seconds"] is None
    assert result["e9"]["algorithm_comparison"]["status"] == "not-ready"


def test_e10_does_not_infer_correctness_from_generated_candidates(tmp_path):
    root = _write_fixture(tmp_path, incomplete=True)
    document = json.loads((root / "acceptance.json").read_text())
    document["cells"][0].pop("selector_replay")
    (root / "acceptance.json").write_text(json.dumps(document))
    result = summary.summarize_acceptance(root)

    assert result["e10"]["replay"]["verified_records"] == 0
    assert result["e10"]["generated_inventory"]["reported_neighbor_total"] == 2
    assert result["e10"]["generated_inventory"]["correctness_inferred"] is False


def test_cli_rejects_overwriting_acceptance(tmp_path):
    root = _write_fixture(tmp_path)
    for output in (root, root / "acceptance.json", root / "summary.json", root / "cells" / "good" / "replacement.json"):
        with pytest.raises(SystemExit):
            summary.main(["--acceptance-dir", str(root), "--output", str(output)])


@pytest.mark.parametrize("field,value", [
    ("uuid", "different-gpu"),
    ("gpu_uuid", "different-gpu"),
    ("global_memory_bytes", "9999"),
    ("compile_mode", "release"),
    ("runtime_fingerprint", "different-runtime"),
])
def test_optional_sample_identity_mismatch_is_not_counted(tmp_path, field, value):
    root = _write_fixture(tmp_path)
    document = json.loads((root / "acceptance.json").read_text())
    document["identity"]["device"].update(uuid="fixture-gpu", global_memory_bytes="1234")
    document["identity"]["runtime_fingerprint"] = "fixture-runtime"
    sample = document["cells"][0]["measurements"][0]["sample"]
    sample[field] = value
    (root / "acceptance.json").write_text(json.dumps(document))

    result = summary.summarize_acceptance(root)

    assert result["e1"]["baselines"] == {}
    assert any(item["reason"] == "cohort-identity-mismatch" for item in result["e1"]["gaps"])
    assert result["e6"]["before_service_extension"]["evaluated_cells"] == 0


def test_cell_id_cannot_escape_acceptance_cells_root(tmp_path):
    root = _write_fixture(tmp_path)
    document = json.loads((root / "acceptance.json").read_text())
    document["cells"][0]["id"] = "../outside"
    (root / "acceptance.json").write_text(json.dumps(document))
    outside = root.parent / "outside"
    outside.mkdir()
    source = root / "cells" / "good" / "search_coverage.json"
    (outside / "search_coverage.json").write_text(source.read_text())

    result = summary.summarize_acceptance(root)

    assert result["e9"]["trace_count"] == 0
