"""Guard against string ordering in CSV-based mechanism winner selection."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a100_confirmed_winner", ROOT / "results/a100_closure_20261102/run_tail_queue.py")
tail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tail)
CORRECTION_SPEC = importlib.util.spec_from_file_location(
    "a100_winner_correction", ROOT / "results/a100_closure_20261102/mechanism/correct_confirmed_winners.py")
correction = importlib.util.module_from_spec(CORRECTION_SPEC)
CORRECTION_SPEC.loader.exec_module(correction)


def candidate(name, samples):
    return {"id": name, "preflight": {"status": "accepted"},
            "trials": [{"performance_accepted": True, "sample": {"kernel_ms": x}}
                       for x in samples]}


def test_cross_digit_csv_winner_uses_numeric_order_without_mutating_input():
    fast = candidate("conventional-fast", ["4.344893", "4.346092", "4.344248"])
    slow = candidate("conventional-slow", ["12.470016", "12.471798", "12.465582"])
    rows = [slow, fast]
    before = copy.deepcopy(rows)
    assert tail.confirmed_winner(rows, "conventional") is fast
    assert tail.confirmed_median_ms(fast) == 4.344893
    assert rows == before


def test_numeric_median_mixed_encodings_and_deterministic_ties():
    a = candidate("hybrid-a", ["2e0", "10", 3.0])
    b = candidate("hybrid-b", [3, "3.0", "3e0"])
    assert tail.confirmed_median_ms(a) == 3.0
    assert tail.confirmed_winner([b, a], "hybrid") is a


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", None, True, "bad"])
def test_invalid_sample_cannot_be_selected_or_silently_dropped(value):
    with pytest.raises(ValueError):
        tail.confirmed_winner([candidate("hybrid-invalid", [1, value, 2])], "hybrid")


def test_requires_confirmed_trials_and_correct_side():
    partial = candidate("hybrid-partial", ["0.1"])
    rejected = candidate("hybrid-rejected", ["0.1"] * 3)
    rejected["trials"][0]["performance_accepted"] = False
    wrong = candidate("conventional-wrong", ["0.01"] * 3)
    assert tail.confirmed_winner([partial, rejected, wrong], "hybrid") is None


def test_captured_fft_confirmation_selects_true_fastest_finalist():
    path = ROOT / ("results/a100_closure_20261102/tail/bounded_recovery/release/"
                   "screen/fft-fp64-n20-b16/confirmation.json")
    if not path.exists():
        pytest.skip("captured A100 confirmation is not installed")
    rows = json.loads(path.read_text())["cases"][0]["variants"]
    selected = tail.confirmed_winner(rows, "conventional")
    assert selected["id"] == "conventional-10b1ab5487d034e9"
    assert tail.confirmed_median_ms(selected) == 4.344893


def test_real_correction_preserves_source_and_schedules_only_changed_winner(tmp_path):
    source = correction.DEFAULT_SOURCE
    if not source.exists():
        pytest.skip("captured A100 confirmation is not installed")
    original = tail.sha(source / "receipt.json")
    result = correction.prepare(source, tmp_path)
    assert tail.sha(source / "receipt.json") == original
    assert len(result["cases"]) == 3
    for job in result["jobs"]:
        assert job["input_file"] in result["identities"]
        receipt = json.loads((source / "receipt.json").read_text())
        assert result["identities"][job["input_file"]] == receipt["sources"][job["input_file"]]
    assert [(j["case_id"], j["side"]) for j in result["jobs"]] == [
        ("fft-fp64-n20-b16", "conventional")]
    statuses = [s["numerical_status"] for c in result["cases"] for s in c["sides"].values()]
    assert statuses.count("reused-finite-error-report-only") == 3
    assert statuses.count("existing-full-batch-exact") == 2
    with pytest.raises(ValueError, match="already exists"):
        correction.prepare(source, tmp_path)


def test_reused_error_report_requires_same_input_mapping_and_precision():
    workload = {"operator": "fft", "precision": "fp64", "logN": 10, "batch": 2}
    mapping = {"stage_partition": [5, 5], "tile_threads": 128}
    record = dict(gpu_uuid=tail.UUID, output_finite=True, status="finite-error-report-only",
                  input_sha256="input", resolved_mapping=mapping,
                  request=dict(operator="fft", precision="fp64", N=1024, batch=2,
                               direction="forward", normalization="none",
                               input_order="natural", output_order="natural"))
    correction.validate_error(record, mapping, workload, {"source_sha256": "input"})
    with pytest.raises(ValueError, match="request/input"):
        correction.validate_error(record, mapping, workload, {"source_sha256": "other"})
    with pytest.raises(ValueError, match="mapping"):
        correction.validate_error(record, dict(mapping, tile_threads=64), workload, {"source_sha256": "input"})
    with pytest.raises(ValueError, match="request/input"):
        correction.validate_error(record, mapping, dict(workload, precision="fp32"), {"source_sha256": "input"})


def test_report_rejects_missing_changed_arrays_and_wrong_jobs(tmp_path):
    selection_path = tmp_path / "selection.json"
    tail.write(selection_path, {"identities": {}, "cases": [], "jobs": [{"case_id": "fft", "side": "conventional"}]})
    release = tmp_path / "validation/release"
    release.mkdir(parents=True)
    tail.write(tmp_path / "validation/status.json", {"state": "complete", "gpu_uuid": tail.UUID})
    tail.write(release / "error.json", {"status": "finite-error-report-only"})
    arrays = release / "error.npz"
    arrays.write_bytes(b"fixture-array")
    receipt = {"selection_sha256": tail.sha(selection_path), "status": "changed-winner-error-report-complete",
               "jobs": [{"case_id": "fft", "side": "conventional", "report": "error.json",
                         "report_sha256": tail.sha(release / "error.json"), "arrays_sha256": tail.sha(arrays)}]}
    tail.write(release / "receipt.json", receipt)
    correction.report(tmp_path)
    arrays.unlink()
    with pytest.raises(FileNotFoundError):
        correction.report(tmp_path)
    arrays.write_bytes(b"modified")
    with pytest.raises(ValueError, match="arrays changed"):
        correction.report(tmp_path)
    receipt["jobs"][0]["side"] = "hybrid"
    tail.write(release / "receipt.json", receipt)
    with pytest.raises(ValueError, match="invalid validation release"):
        correction.report(tmp_path)
