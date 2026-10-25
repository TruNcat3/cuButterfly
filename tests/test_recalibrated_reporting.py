import csv
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from finalize_recalibration import load_retained, point_from_sample, validate_fingerprints, mapping_seeds
from summarize_recalibrated_campaign import (
    external_population,
    paired_external,
    paired_rank1,
    paired_xd,
    report,
    write_report,
)


def test_final_confirmation_rejects_old_runtime_and_preserves_large_modulus():
    record = {"name": "x", "status": "measured", "correct": True,
              "samples": [{"runtime_fingerprint": "old"}]}
    with pytest.raises(ValueError, match="different runtime"):
        validate_fingerprints([record], "current")
    point = point_from_sample({"operator": "ntt", "precision": "word64",
                              "modulus": "576460756061519873", "batch": "64",
                              "logN": "16", "direction": "inverse",
                              "output_order": "bit-reversed", "mapping_json": "{}"})
    assert point["modulus"] == 576460756061519873
    assert point["direction"] == "inverse" and point["output_order"] == "bit-reversed"


def test_final_confirmation_never_reuses_journal_without_matching_identity(tmp_path):
    journal = tmp_path / "operator_calibration.json"
    identity = tmp_path / "confirmation_identity.json"
    journal.write_text('[{"name":"old"}]')
    with pytest.raises(ValueError, match="no measurement identity"):
        load_retained(journal, identity, {"build": "current"}, True)
    identity.write_text('{"build":"old"}')
    with pytest.raises(ValueError, match="different identity"):
        load_retained(journal, identity, {"build": "current"}, True)
    identity.write_text('{"build":"current"}')
    assert list(load_retained(journal, identity, {"build": "current"}, True)) == ["old"]


def test_cross_build_revalidation_imports_only_mapping_not_measurement():
    old = dict(name="old", status="measured", correct=True, median_kernel_ms=1,
               samples=[dict(operator="ntt", mapping_json='{"backend":"compact-stage"}',
                             runtime_fingerprint="old", kernel_ms=1, correct="1")],
               stage_services={"old": 1})
    with pytest.raises(ValueError, match="different runtime"):
        mapping_seeds([old], "new")
    seed = mapping_seeds([old], "new", True)[0]
    assert seed["status"] == "seed" and not seed["correct"]
    assert set(seed) == {"name", "status", "correct", "samples"}
    assert seed["samples"] == [{"operator": "ntt", "mapping_json": '{"backend":"compact-stage"}'}]


def raw_sample(name, implementation, time, **fields):
    return dict(suite_case_id=name, suite_group="same-name", implementation=implementation,
                reference="1" if implementation == "cuFFT" else "0", kernel_ms=str(time),
                preflight_correct="1", correct="1", performance_batch="16", N="256", operator="fft",
                precision="fp32", direction="forward", placement="in-place", **fields)


def test_report_requires_both_sides_verified_and_same_workload():
    ours = raw_sample("cub", "cuButterfly", 1)
    reference = raw_sample("ref", "cuFFT", 2)
    assert paired_rank1([ours, reference], "new")[0]["speedup"] == 2
    assert paired_rank1([ours, dict(reference, preflight_correct="-1")], "new") == []
    assert paired_rank1([ours, dict(reference, performance_batch="32")], "new") == []
    assert paired_rank1([ours, dict(reference, direction="inverse")], "new") == []
    assert paired_rank1([dict(ours, correct="0"), reference], "new") == []
    assert paired_rank1([dict(ours, warmup="5"), dict(reference, warmup="10")], "new") == []
    timed = paired_rank1([dict(ours, correct="1", verified_batches="2"), reference], "new")[0]
    assert timed["verification"].startswith("timed cuButterfly mapping: 2 sampled")


def test_rank1_missing_correct_and_duplicate_trial_identity_are_not_verified():
    ours = raw_sample("cub", "cuButterfly", 1)
    reference = raw_sample("ref", "cuFFT", 2)
    missing = dict(ours)
    missing.pop("correct")
    assert paired_rank1([missing, reference], "new") == []

    duplicate = dict(ours, trial="1")
    first = dict(ours, trial="1")
    assert paired_rank1([first, duplicate, reference], "new") == []
    assert paired_rank1([ours, dict(ours, kernel_ms="2"), reference], "new") == []


def test_report_does_not_promote_unverified_xd_or_aliases_to_wins():
    row = dict(operator="fft", precision="fp32", configuration="selected", shape=[8, 8],
               batch=1, kernel_ms=1, cufft_ms=2, status="measured")
    assert paired_xd([row], "xd") == []
    assert paired_xd([dict(row, correct=True, status="passed")], "xd")[0]["speedup"] == 2
    ours = dict(raw_sample("alias", "cuButterfly", 1), operator="xor-zeta")
    reference = dict(raw_sample("ref", "external", 2), operator="xor-zeta")
    assert paired_rank1([ours, reference], "new") == []


def external_sample(group, runner, implementation, time, *, operator="fwht", trial="1", correct="1", **fields):
    defaults = dict(
        case_id=f"{group}_{runner}_{trial}", group=group, trial=trial,
        implementation=implementation, reference="1" if runner in ("butterfly", "ntt") else "0",
        runner=runner, operator=operator, precision="fp32", direction="forward",
        normalization="none", placement="out-of-place", output_order="natural",
        logN="8", N="256", batch="16", modulus="", modulus_bits="",
        warmup="1000", repeat="100",
        kernel_ms=str(time), correct=correct,
    )
    defaults.update(fields)
    return defaults


def test_external_pairs_use_external_over_ours_and_keep_exact_modulus():
    rows = [
        external_sample("fwht8-batch16", "butterfly", "cuButterfly-warp-register", 1, trial="1"),
        external_sample("fwht8-batch16", "butterfly", "cuButterfly-warp-register", 3, trial="2"),
        external_sample("fwht8-batch16", "dao-fht", "Dao-AILab-FHT", 2, trial="1"),
        external_sample("fwht8-batch16", "dao-fht", "Dao-AILab-FHT", 4, trial="2"),
        external_sample("ntt16-batch4", "ntt", "cuNTT-Hybrid2D-radix4", 5,
                        operator="ntt", precision="uint64", logN="16", N="65536", batch="4",
                        modulus="576460756061519873", modulus_bits="60"),
        external_sample("ntt16-batch4", "gpuntt", "GPU-NTT-Merge-natural", 10,
                        operator="ntt", precision="uint64", logN="16", N="65536", batch="4",
                        modulus="576460756061519873", modulus_bits="60"),
        external_sample("fwht8-batch16", "cufft", "cuFFT", 0.5),
    ]

    pairs = paired_external(rows, "comparison/external_raw.csv")
    assert len(pairs) == 2
    fwht = next(row for row in pairs if row["operator"] == "fwht")
    ntt = next(row for row in pairs if row["operator"] == "ntt")
    assert fwht["ours_ms"] == 2 and fwht["external_ms"] == 3
    assert fwht["speedup"] == 1.5
    assert fwht["ratio_direction"] == "external_ms/cuButterfly_ms"
    assert ntt["modulus"] == "576460756061519873"
    assert ntt["modulus_bits"] == "60"


def test_external_pairs_require_matching_semantics_and_all_correct_trials():
    ours = external_sample("fwht8-batch16", "butterfly", "current", 1)
    external = external_sample("fwht8-batch16", "dao-fht", "Dao", 2, correct="0")
    assert paired_external([ours, external]) == []
    pending = external_population([ours, external])
    assert pending["measured_cases"] == 0
    assert pending["pending_cases"] == 1
    assert pending["statuses"] == {"correct": 1, "incorrect": 1}

    mismatched = external_sample("fwht8-batch16", "dao-fht", "Dao", 2, batch="32")
    assert paired_external([ours, mismatched]) == []
    assert external_population([ours, mismatched])["pending_cases"] == 2
    protocol_mismatch = external_sample("fwht8-batch16", "dao-fht", "Dao", 2, warmup="10")
    assert paired_external([ours, protocol_mismatch]) == []
    assert external_population([ours, protocol_mismatch])["pending_cases"] == 2


def test_external_pairs_reject_duplicate_trials_and_implementation_cartesian_products():
    ours = external_sample("fwht8-batch16", "butterfly", "current", 1, trial="1")
    external = external_sample("fwht8-batch16", "dao-fht", "Dao", 2, trial="1")
    duplicate = dict(ours, kernel_ms="3")
    pending = external_population([ours, duplicate, external])
    assert paired_external([ours, duplicate, external]) == []
    assert pending["duplicate_trial_groups"] == 1
    assert pending["pending_cases"] == 1

    exact_duplicate = dict(ours)
    exact_pending = external_population([ours, exact_duplicate, external])
    assert paired_external([ours, exact_duplicate, external]) == []
    assert exact_pending["duplicate_trial_groups"] == 1

    alias = dict(ours, implementation="current-alias")
    pending = external_population([ours, alias, external])
    assert paired_external([ours, alias, external]) == []
    assert any("multiple implementations" in item["reason"] for item in pending["pending"])


def test_external_rows_without_implementation_do_not_split_trials_into_aliases():
    ours1 = external_sample("fwht8-batch16", "butterfly", "", 1, trial="1")
    ours2 = external_sample("fwht8-batch16", "butterfly", "", 3, trial="2")
    external1 = external_sample("fwht8-batch16", "dao-fht", "Dao", 2, trial="1")
    external2 = external_sample("fwht8-batch16", "dao-fht", "Dao", 4, trial="2")
    assert len(paired_external([ours1, ours2, external1, external2])) == 1


def test_report_exposes_external_missing_side_as_pending(tmp_path):
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    rows = [external_sample("fwht8-batch16", "butterfly", "current", 1)]
    fields = list(rows[0])
    with (comparison / "external_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "external_suite.json").write_text(
        '{"workloads": [{"id": "fwht8"}]}'
    )

    result = report(tmp_path)
    population = next(item for item in result["populations"]
                      if item["population"] == "comparison/external_raw.csv")
    assert population["measured_cases"] == 0
    assert population["pending_cases"] == 1
    assert population["pending"][0]["external_runner"] == "dao-fht"
    assert result["aggregates"] == []


def test_write_report_unions_case_columns_across_populations(tmp_path):
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    rank1_rows = [raw_sample("cub", "cuButterfly", 1), raw_sample("ref", "cuFFT", 2)]
    rank1_fields = list(rank1_rows[0])
    with (comparison / "rank1_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rank1_fields)
        writer.writeheader()
        writer.writerows(rank1_rows)
    external_rows = [
        external_sample("fwht8-batch16", "butterfly", "current", 1),
        external_sample("fwht8-batch16", "dao-fht", "Dao", 2),
    ]
    external_fields = list(external_rows[0])
    with (comparison / "external_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=external_fields)
        writer.writeheader()
        writer.writerows(external_rows)

    write_report(tmp_path)
    with (tmp_path / "tables/performance_cases.csv").open(newline="") as stream:
        fields = next(csv.reader(stream))
    assert "external_ms" in fields and "cuButterfly_ms" in fields


def test_xd_duplicate_cells_and_vendor_fallbacks_do_not_create_wins():
    row = dict(operator="fft", precision="fp32", configuration="selected", shape=[8, 8],
               batch=1, kernel_ms=1, cufft_ms=2, status="passed", correct=True,
               inverse=False)
    assert paired_xd([row, dict(row, id="duplicate")], "xd") == []
    pairs = [
        dict(population="p", operator="fft", precision="fp32", direction="forward",
             reference="cuFFT", reference_kind="external-library", modulus="",
             modulus_bits="", shape="8x8", batch=1, normalization="", placement="",
             output_order="", logN="", N="", speedup=2.0, vendor_fallback=True,
             selector_status="measured"),
        dict(population="p", operator="fft", precision="fp32", direction="forward",
             reference="cuFFT", reference_kind="external-library", modulus="",
             modulus_bits="", shape="16x16", batch=1, normalization="", placement="",
             output_order="", logN="", N="", speedup=1.5, vendor_fallback=False,
             selector_status="xD-complete-plan-measurement"),
    ]
    summary = __import__("summarize_recalibrated_campaign").aggregate(pairs)
    assert len(summary) == 1 and summary[0]["cases"] == 1


def test_population_geomean_spans_sizes_without_double_counting_workload():
    from summarize_recalibrated_campaign import aggregate
    common = dict(population="fixed", operator="fft", precision="fp32", direction="forward",
                  reference="cuFFT", reference_kind="external-library", vendor_fallback=False,
                  selector_status="measured", batch=1)
    result = aggregate([dict(common, shape="256", speedup=2),
                        dict(common, shape="256", speedup=2),
                        dict(common, shape="4096", speedup=0.5)])
    assert len(result) == 1
    assert result[0]["cases"] == 2 and result[0]["duplicate_observations"] == 1
    assert result[0]["geomean"] == pytest.approx(1)


def test_external_fixed_reference_can_use_explicit_preflight_verification():
    ours = raw_sample("ours", "cuButterfly", 1)
    reference = dict(raw_sample("ref", "VkFFT", 2, suite_runner="fft-library"), correct="-1")
    assert paired_rank1([ours, reference], "new")[0]["speedup"] == 2
    assert paired_rank1([dict(ours, correct="-1"), reference], "new") == []
    assert paired_rank1([ours, dict(reference, correct="0")], "new") == []


def test_completed_operator_without_baseline_is_not_unfinished_execution():
    from summarize_recalibrated_campaign import _rank1_population
    row = raw_sample("ours", "cuButterfly", 1, trial="1")
    result = _rank1_population([row], expected=1)
    assert result["verified_cases"] == 1 and result["pending_cases"] == 0
    assert result["matched_pairs"] == 0 and result["verified_without_reference"] == 1


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_strict_failed_queue_output_stays_observed_but_cannot_be_performance(tmp_path):
    output = tmp_path / "comparison" / "rank1_raw.csv"
    rows = [raw_sample("cub", "cuButterfly", 1), raw_sample("ref", "cuFFT", 2)]
    _write_rows(output, rows)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "measurement_contract": {"schema": "reviewed"},
        "comparison_suite": {"cohorts": {"standard_rank1": 1}},
        "phases": {"comparison": [{"id": "rank1", "outputs": [str(output)]}]},
    }))
    (tmp_path / "campaign_queue_state.json").write_text(json.dumps({
        "status": "gpu-conflict",
        "phases": {"comparison": {"steps": [{
            "id": "rank1", "status": "gpu-conflict", "gate_passed": False,
            "returncode": 1,
        }]}},
    }))

    result = report(tmp_path)
    population = next(item for item in result["populations"]
                       if item["population"] == "comparison/rank1_raw.csv")
    assert population["observed_trial_rows"] == 2
    assert population["eligible_trial_rows"] == 0
    assert population["verified_cases"] == 0 and population["pending_cases"] == 1
    assert population["pending_case_ids"] == ["cub"]
    assert population["measurement_gate"]["reason"] == "queue-status-gpu-conflict"
    assert result["pairs"] == []


def test_strict_passed_queue_output_is_eligible_for_pairing(tmp_path):
    output = tmp_path / "comparison" / "rank1_raw.csv"
    summary = tmp_path / "comparison" / "rank1_summary.md"
    _write_rows(output, [raw_sample("cub", "cuButterfly", 1), raw_sample("ref", "cuFFT", 2)])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "measurement_contract": {"schema": "reviewed"},
        "comparison_suite": {"cohorts": {"standard_rank1": 1}},
        "phases": {"comparison": [
            {"id": "rank1", "outputs": [str(output)]},
            {"id": "summary", "run": f"python summarize.py {output}",
             "outputs": [str(summary)]},
        ]},
    }))
    (tmp_path / "campaign_queue_state.json").write_text(json.dumps({
        "status": "completed",
        "phases": {"comparison": {"steps": [
            {"id": "rank1", "status": "passed", "gate_passed": True,
             "returncode": 0},
            {"id": "summary", "status": "running", "gate_passed": False},
        ]}},
    }))

    result = report(tmp_path)
    population = next(item for item in result["populations"]
                       if item["population"] == "comparison/rank1_raw.csv")
    assert population["verified_cases"] == 1
    assert population["measurement_gate"]["eligible"] is True
    assert len(result["pairs"]) == 1


def test_strict_profile_directory_output_gates_rows_json(tmp_path):
    output_dir = tmp_path / "profile" / "xd_coverage"
    output_dir.mkdir(parents=True)
    (output_dir / "rows.json").write_text(json.dumps([{
        "operator": "fft", "precision": "fp32", "configuration": "selected",
        "shape": [8, 8], "batch": 1, "kernel_ms": 1, "cufft_ms": 2,
        "status": "passed", "correct": True,
    }]))
    (output_dir / "coverage_summary.json").write_text(json.dumps({
        "coverage": {"expected_keys": 1, "verified_count": 1, "status": "verified-complete"},
    }))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "measurement_contract": {"schema": "reviewed"},
        "phases": {"profile_gate": [
            {"id": "profile", "outputs": [str(output_dir)]},
            {"id": "summary", "run": f"python summarize.py {output_dir / 'rows.json'}",
             "outputs": [str(output_dir / "coverage_summary.json")]},
        ]},
    }))
    (tmp_path / "campaign_queue_state.json").write_text(json.dumps({
        "status": "completed",
        "phases": {"profile_gate": {"steps": [
            {"id": "profile", "status": "passed", "gate_passed": True,
             "returncode": 0},
            {"id": "summary", "status": "passed", "gate_passed": True,
             "returncode": 0},
        ]}},
    }))

    coverage = next(item for item in report(tmp_path)["populations"]
                    if item["population"] == "profile/xd_coverage")
    assert coverage["verified_cases"] == 1
    assert coverage["measurement_gate"]["eligible"] is True
    assert coverage["coverage_gate"]["step_id"] == "summary"


def test_external_single_declared_partition_wins_over_stale_combined_file(tmp_path):
    comparison = tmp_path / "comparison"
    split_rows = [
        external_sample("fwht8", "butterfly", "current", 1),
        external_sample("fwht8", "dao-fht", "Dao", 2),
    ]
    _write_rows(comparison / "external_dao_raw.csv", split_rows)
    _write_rows(comparison / "external_raw.csv", [
        external_sample("fwht8", "butterfly", "current", 1),
        external_sample("fwht8", "dao-fht", "Dao", 9),
    ])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "external_comparison": {"fresh_artifacts": ["comparison/external_dao_raw.csv"]},
    }))
    (tmp_path / "external_suite.json").write_text(json.dumps({
        "workloads": [{"id": "fwht8", "batches": [16]}],
    }))

    result = report(tmp_path)
    population = next(item for item in result["populations"]
                       if item["population"] == "comparison/external_raw.csv")
    assert population["measured_cases"] == 1
    assert population["source_files"] == ["comparison/external_dao_raw.csv"]
    assert result["pairs"][0]["external_ms"] == 2.0


def test_xd_nested_coverage_and_aliases_do_not_inflate_verified_count(tmp_path):
    profile = tmp_path / "profile" / "xd_coverage"
    profile.mkdir(parents=True)
    rows = [
        {"operator": "subset-zeta", "precision": "uint32", "shape": [8, 8],
         "batch": 1, "inverse": False, "configuration": "selected",
         "status": "passed", "correct": True},
        {"operator": "xor-zeta", "precision": "uint32", "shape": [8, 8],
         "batch": 1, "inverse": False, "configuration": "selected",
         "status": "passed", "correct": True},
    ]
    (profile / "rows.json").write_text(json.dumps(rows))
    (profile / "manifest.json").write_text(json.dumps({"semantic_aliases": {"xor-zeta": "subset-zeta"}}))
    (profile / "coverage_summary.json").write_text(json.dumps({
        "coverage": {"expected_keys": 1,
                      "duplicate_count": 1, "status": "verified-complete"},
    }))

    populations = report(tmp_path)["populations"]
    coverage = next(item for item in populations if item["population"] == "profile/xd_coverage")
    assert coverage["requested"] == 1
    assert coverage["verified_cases"] == 1
    assert coverage["duplicate_cases"] == 1
    assert coverage["coverage_status"] == "verified-complete"


def test_markdown_summary_labels_protocol_without_shape_or_batch(tmp_path):
    comparison = tmp_path / "comparison"
    rank1_rows = [raw_sample("cub", "cuButterfly", 1), raw_sample("ref", "cuFFT", 2)]
    _write_rows(comparison / "rank1_raw.csv", rank1_rows)

    write_report(tmp_path)
    markdown = (tmp_path / "tables/performance.md").read_text()
    assert "| Protocol |" in markdown
    assert "Shape / batch / protocol" not in markdown
