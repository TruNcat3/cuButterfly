"""CPU contracts for explicitly controlled nested physical mappings."""
import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import run_controlled_experiment as runner


def nested_study():
    mapping = {
        "backend": "hierarchical-dataflow", "stage_partition": [10, 10],
        "subgraph_mappings": [
            {"core": "homogeneous-radix4", "threads_per_block": 256,
             "units_per_cta": 4, "data_time": 1} for _ in range(2)],
    }
    treatment = copy.deepcopy(mapping)
    for item in treatment["subgraph_mappings"]:
        item["data_time"] = 2
    return {
        "schema": runner.SCHEMA,
        "protocol": {"trials": 1, "warmup": 1, "repeat": 2, "verify_batches": 0},
        "cases": [{
            "id": "ntt-data-time", "experiment": "nested-control",
            "workload": {"operator": "ntt", "precision": "word64", "logN": 20,
                         "batch": 4, "modulus": 576460756061519873},
            "variants": [{"id": "dt1", "mapping": mapping},
                         {"id": "dt2", "mapping": treatment}],
            "comparisons": [{"baseline": "dt1", "treatment": "dt2", "changed_axes": [
                "subgraph_mappings.0.data_time", "subgraph_mappings.1.data_time"]}],
        }],
    }


def measured_case():
    case = nested_study()["cases"][0]
    for variant in case["variants"]:
        variant["preflight"] = {
            "status": "accepted", "resolved_mapping": copy.deepcopy(variant["mapping"]),
        }
        variant["trials"] = [{"trial": 0, "performance_accepted": True, "kernel_ms": 1.0}]
    return case


def test_nested_data_time_prepares_and_compares(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("GPU access"))
    path = tmp_path / "study.json"
    path.write_text(json.dumps(nested_study()))
    document = runner.prepare_document(nested_study(), path, tmp_path / "build")
    assert document["identity"]["gpu_queried"] is False
    case = measured_case()
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["status"] == "measured"
    assert result["actual_changed_axes"] == [
        "subgraph_mappings.0.data_time", "subgraph_mappings.1.data_time"]


@pytest.mark.parametrize("field,value", [("core", "homogeneous-warp-radix2"),
                                         ("threads_per_block", 128), ("units_per_cta", 8)])
def test_prepare_rejects_undeclared_nested_change(field, value):
    study = nested_study()
    study["cases"][0]["variants"][1]["mapping"]["subgraph_mappings"][0][field] = value
    with pytest.raises(runner.StudyError, match="undeclared mapping axes.*" + field):
        runner._validate_study(study)


def test_runtime_nested_alias_is_blocked():
    case = measured_case()
    case["variants"][1]["preflight"]["resolved_mapping"] = copy.deepcopy(
        case["variants"][0]["preflight"]["resolved_mapping"])
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["performance_accepted"] is False
    assert "data_time" in result["reason"]


@pytest.mark.parametrize("field,value", [("core", "homogeneous-warp-radix2"),
                                         ("threads_per_block", 128)])
def test_runtime_rejects_changed_explicit_nested_leaf(field, value):
    case = measured_case()
    case["variants"][1]["preflight"]["resolved_mapping"]["subgraph_mappings"][0][field] = value
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["performance_accepted"] is False
    assert field in result["reason"]


def test_runtime_rejects_both_variants_silently_changing_core():
    case = measured_case()
    for variant in case["variants"]:
        variant["preflight"]["resolved_mapping"]["subgraph_mappings"][0]["core"] = "radix2"
    with pytest.raises(RuntimeError, match="requested nested mapping.*core"):
        runner._validate_nested_replay(case, case["variants"][0],
                                       case["variants"][0]["preflight"]["resolved_mapping"])
    assert runner._comparison_result(case, case["comparisons"][0])["performance_accepted"] is False


def test_runtime_checks_unrequested_fields_in_controlled_root():
    case = measured_case()
    for index, variant in enumerate(case["variants"]):
        variant["preflight"]["resolved_mapping"]["subgraph_mappings"][0]["cta_weight"] = index + 1
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["performance_accepted"] is False
    assert "undeclared axes: subgraph_mappings.0.cta_weight" in result["reason"]


def test_runtime_rejects_both_variants_growing_mapping_array():
    case = measured_case()
    for variant in case["variants"]:
        values = variant["preflight"]["resolved_mapping"]["subgraph_mappings"]
        values.append(copy.deepcopy(values[0]))
    result = runner._comparison_result(case, case["comparisons"][0])
    assert result["performance_accepted"] is False
    assert "array shape changed" in result["reason"]


def test_preflight_rejects_nested_replay_mismatch_before_resource_probe(tmp_path, monkeypatch):
    case = nested_study()["cases"][0]
    variant = case["variants"][0]
    variant["preflight_command"] = ["mock"]
    resolved = copy.deepcopy(variant["mapping"])
    resolved["subgraph_mappings"][0]["threads_per_block"] = 128
    monkeypatch.setattr(runner, "_invoke", lambda command: ({}, None))
    monkeypatch.setattr(runner, "_validate_sample", lambda *a, **k: (resolved, {}))
    monkeypatch.setattr(runner, "_probe_resources", lambda *a: pytest.fail("rejected mapping probed"))
    runner._preflight_variant(case, variant, {}, {}, tmp_path)
    assert variant["preflight"]["status"] == "rejected"
    assert "threads_per_block" in variant["preflight"]["reason"]


@pytest.mark.parametrize("axis", ["subgraph_mappings.0.missing", "subgraph_mappings",
                                  "subgraph_mappings.01.data_time", "subgraph_mappings.*.data_time"])
def test_no_blanket_or_missing_nested_axis(axis):
    study = nested_study()
    study["cases"][0]["comparisons"][0]["changed_axes"] = [axis]
    with pytest.raises(runner.StudyError):
        runner._validate_study(study)


def test_legacy_uncontrolled_nested_mapping_remains_derived():
    left = {"shared_layout": "linear", "subgraph_mappings": [{"core": "radix2"}]}
    right = {"shared_layout": "writer-aligned", "subgraph_mappings": [{"core": "radix4"}]}
    assert runner._mapping_diff(left, right, include_derived=False) == {"shared_layout"}
    assert runner._mapping_diff(runner._control_mapping(left), runner._control_mapping(right)) == {"shared_layout"}
