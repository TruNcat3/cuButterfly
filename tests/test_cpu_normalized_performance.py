import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cpu_normalized_rows_are_exact_and_complete():
    data = json.loads((ROOT / "paper/generated/current_refresh/cpu_normalized.json").read_text())
    assert data["schema"] == "cpu-normalized-performance-v1"
    assert data["coverage"] == {
        "cpu_rows": 83,
        "cuButterfly": 66,
        "external_gpu_pairs": 49,
        "gpu_external_pairs": 49,
    }
    keys = set()
    for row in data["records"]:
        key = (row["case_id"], row["backend"])
        assert key not in keys
        keys.add(key)
        expected = row["cpu_us"] / row["backend_us"]
        assert math.isclose(row["speedup"], expected, rel_tol=1e-12, abs_tol=1e-12)
        assert row["cpu_backend"]
        assert row["semantics"]["shape"] == row["shape"]
    assert sum(row["backend"] == "cuButterfly" for row in data["records"]) == 66
