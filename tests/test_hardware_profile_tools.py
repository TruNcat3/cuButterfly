import csv
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import initialize_hardware_profile as profile_tools


def test_model_table_contains_both_barrier_models(tmp_path):
    profile = {
        "device": "test-gpu",
        "compute_capability": "9.0",
        "capabilities": {
            "global_feedback_bytes_per_second": 1.0e12,
            "equivalent_butterflies_per_second": 2.0e12,
            "interstage_shared_bytes_per_second": 3.0e12,
            "cta_barriers_per_second": 4.0e9,
        },
    }
    args = type(
        "Args",
        (),
        {"logNs": [8], "spatial_budgets": [32], "word_bytes": [4]},
    )()
    output = tmp_path / "model.csv"
    count = profile_tools.build_model_table(profile, args, output)
    assert count > 0
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["barrier_model"] for row in rows} == {"cta", "none"}
    assert all(row["device"] == "test-gpu" for row in rows)
    assert all(float(row["calibrated_body_us"]) > 0 for row in rows)


def test_profile_schema_is_compatible_with_ranker(tmp_path):
    profile = {
        "schema": "cubutterfly-hardware-profile-v1",
        "status": "calibrated-local",
        "device": "test-gpu",
        "compute_capability": "9.0",
        "sm_count": 1,
        "trials": 3,
        "capabilities": {},
    }
    path = tmp_path / "hardware_profile.json"
    path.write_text(json.dumps(profile))
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "calibrated-local"
    assert "capabilities" in loaded
