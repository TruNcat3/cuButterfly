from scripts.prepare_core_organization_controls import build_study
from scripts.summarize_core_organization_controls import main as summarize_main

import json


def test_core_organization_cohort_has_explicit_legal_families():
    study = build_study()
    assert len(study["cases"]) == 6
    assert all("--verify" in case["args"] for case in study["cases"])
    assert {case["operator"] for case in study["cases"]} == {"fwht", "ntt"}
    ntt = [case for case in study["cases"] if case["operator"] == "ntt"]
    assert len(ntt) == 2
    assert {case["core"] for case in ntt} == {
        "homogeneous-warp128-static-io-radix2",
        "homogeneous-warp128-vector-radix4-static-io",
    }
    assert {case["organization"]["stage_partition"] for case in ntt} == {"10,10"}
    assert {case["semantic_contract"]["N"] for case in ntt} == {1048576}


def test_core_organization_summary_keeps_correctness_and_mapping(tmp_path, monkeypatch):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "cases": [{
            "id": "ntt-test", "operator": "ntt", "status": "complete",
            "semantic_contract": {"N": 16, "batch": 1},
            "core": "core-a",
            "measurement": {"mapping_json": json.dumps({"backend": "x", "stage_partition": [8, 8]}),
                            "kernel_ms": 1.0, "correct": "1"},
        }],
    }))
    output = tmp_path / "summary.csv"
    monkeypatch.setattr("sys.argv", ["summarize", str(results), "--output", str(output)])
    summarize_main()
    text = output.read_text()
    assert "core-a" in text and "8+8" in text and ",1,complete" in text
