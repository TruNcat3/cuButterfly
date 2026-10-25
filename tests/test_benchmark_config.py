import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.manifest import (  # noqa: E402
    BenchmarkManifestError,
    expand_cases,
    load_manifest,
    write_case_manifest,
)


def test_quick_manifest_expands_all_scopes_and_is_bounded():
    document = load_manifest(ROOT / "config" / "benchmark_quick.json")
    cases = expand_cases(document, "quick")
    assert {case["scope"] for case in cases} == {"operator", "e2e", "profile"}
    assert len(cases) == 14
    assert all(case["mode"] == "quick" for case in cases)
    assert all("execution" in case for case in cases)
    assert all(not (case["provider"] == "cpu" and case["execution"]["gpu"]) for case in cases)


def test_filters_are_exact_and_reject_empty_selection():
    document = load_manifest(ROOT / "config" / "benchmark_full.json")
    cases = expand_cases(document, "full", {
        "scope": "operator", "operators": ["ntt"], "sizes": [32768],
        "batches": [4], "precisions": ["word64"], "backend": "gpuntt",
    })
    assert cases
    assert {case["scope"] for case in cases} == {"operator"}
    assert {case["operator"] for case in cases} == {"ntt"}
    assert {case["size"] for case in cases} == {32768}
    assert {case["backend"] for case in cases} == {"gpuntt"}
    with pytest.raises(BenchmarkManifestError, match="no cases"):
        expand_cases(document, "full", {"operators": ["does-not-exist"]})


def test_unknown_fields_and_unknown_mode_case_are_rejected(tmp_path):
    source = json.loads((ROOT / "config" / "benchmark_quick.json").read_text())
    source["cases"][0]["typo"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(BenchmarkManifestError, match="unknown field"):
        load_manifest(path)
    source = json.loads((ROOT / "config" / "benchmark_quick.json").read_text())
    source["modes"]["quick"]["case_ids"] = ["missing"]
    path.write_text(json.dumps(source))
    with pytest.raises(BenchmarkManifestError, match="unknown case"):
        load_manifest(path)

    source = json.loads((ROOT / "config" / "benchmark_quick.json").read_text())
    source["cases"][0]["targets"][0]["provider"] = ""
    path.write_text(json.dumps(source))
    with pytest.raises(BenchmarkManifestError, match="non-empty string"):
        load_manifest(path)


def test_case_manifest_is_auditable(tmp_path):
    document = load_manifest(ROOT / "config" / "benchmark_quick.json")
    cases = expand_cases(document, "quick", {"scope": "profile"})
    output = tmp_path / "cases.json"
    write_case_manifest(document, cases, output)
    result = json.loads(output.read_text())
    assert result["schema"] == "cubutterfly-benchmark-case-manifest-v1"
    assert result["source"] == "quick-default"
    assert result["case_count"] == len(cases)
    assert all("case_id" in case and "scope" in case for case in result["cases"])


def test_execution_filter_keeps_explicit_target_cohort():
    document = load_manifest(ROOT / "config" / "benchmark_quick.json")
    gpu_cases = expand_cases(document, "quick", {"scope": "operator", "gpu": True})
    cpu_cases = expand_cases(document, "quick", {"scope": "operator", "cpu": True})
    assert gpu_cases and cpu_cases
    assert {case["provider"] for case in gpu_cases} == {"cuda"}
    assert {case["provider"] for case in cpu_cases} == {"cpu"}


def test_plural_batch_axis_uses_batch_field_and_creates_parent(tmp_path):
    document = load_manifest(ROOT / "config" / "benchmark_full.json")
    cases = expand_cases(document, "full", {
        "scope": "operator", "operators": ["ntt"], "sizes": [32768],
        "batches": [4], "precisions": ["word64"], "backend": "gpuntt",
    })
    assert {case["batch"] for case in cases} == {4}
    output = tmp_path / "nested" / "cases.json"
    write_case_manifest(document, cases, output)
    assert output.is_file()
    assert "batche" not in output.read_text()


def test_cli_defaults_can_be_consumed_without_gpu(tmp_path):
    document = load_manifest(ROOT / "config" / "benchmark_full.json")
    cases = expand_cases(document, "full", {"scope": "profile", "operators": ["fft"]})
    assert cases
    assert all(case["execution"]["gpu"] for case in cases)


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_fft_aliases_and_legacy_filter_preserve_operator(mode):
    document = load_manifest(ROOT / "config" / f"benchmark_{mode}.json")
    legacy = expand_cases(document, mode, {"operators": ["fft"]})
    high = expand_cases(document, mode, {"operators": ["fft-h"]})
    low = expand_cases(document, mode, {"compute_types": ["fft-l"]})
    assert len(legacy) == len(high) + len(low)
    assert {case["operator"] for case in legacy} == {"fft"}
    assert {case["compute_type"] for case in high} == {"fft-h"}
    assert {case["compute_type"] for case in low} == {"fft-l"}
    assert {case["precision"] for case in low} == {"fp16", "bf16"}


def test_fft_semantic_contract_and_rejections():
    from benchmark.compute_types import fft_semantics

    assert fft_semantics("fp16", accumulation="fp32") == {
        "storage_precision": "fp16", "compute_precision": "fp32",
        "accumulation": "fp32", "accumulation_precision": "fp32", "compute_type": "fft-l",
    }
    mixed = fft_semantics("fp16-fp32")
    assert mixed["storage_precision"] == "fp32"
    assert mixed["compute_precision"] == "fp16"
    assert mixed["accumulation"] == "native"
    assert mixed["accumulation_precision"] == "fp32"
    assert mixed["compute_type"] == "fft-l"
    for precision in ("tf32", "word32", "word64"):
        with pytest.raises(ValueError, match="unsupported"):
            fft_semantics(precision)
    with pytest.raises(ValueError, match="compute_precision"):
        fft_semantics("fp32", compute_precision="tf32")
    document = load_manifest(ROOT / "config" / "benchmark_quick.json")
    document["cases"] = [dict(document["cases"][0], operators=["fft-h"])]
    with pytest.raises(BenchmarkManifestError, match="conflicts"):
        expand_cases(document)
    with pytest.raises(BenchmarkManifestError, match="fft-h/fft-l"):
        expand_cases(document, filters={"compute_types": ["fft-medium"]})


def test_fft_compute_types_are_derived_allowed_set_not_cartesian_axis():
    document = load_manifest(ROOT / "config" / "benchmark_quick.json")
    case = dict(document["cases"][0], operators=["fft"], precisions=["fp16", "fp32"],
                compute_types=["fft-h", "fft-l"])
    document["cases"] = [case]
    rows = expand_cases(document)
    assert [(row["precision"], row["compute_type"]) for row in rows] == [
        ("fp16", "fft-l"), ("fp32", "fft-h"),
    ]
    assert {row["operator"] for row in rows} == {"fft"}
    case.update(precisions=["fp16"], compute_types=["fft-h"])
    with pytest.raises(BenchmarkManifestError, match="conflicts"):
        expand_cases(document)
    case.update(operators=["fft-l"], precisions=["fp16"], compute_types=["fft-h"])
    with pytest.raises(BenchmarkManifestError, match="alias conflicts"):
        expand_cases(document)


def test_full_fft_e2e_only_declares_fft_scenes():
    document = load_manifest(ROOT / "config" / "benchmark_full.json")
    rows = expand_cases(document, "full", {"scope": "e2e", "operators": ["fft"]})
    assert {row["scene"] for row in rows} == {"image_filter", "poisson"}
