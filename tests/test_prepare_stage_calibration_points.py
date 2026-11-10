import json
from pathlib import Path

import pytest

from scripts.prepare_stage_calibration_points import normalize


def test_normalizes_mapping_without_inference():
    points, summary = normalize([{"operator": "fft", "precision": "fp64", "logN": 20,
                                  "batch": 16, "mapping": {"backend": "factor-streamed",
                                  "factor_columns": 2, "prefetch_depth": 2}}])
    assert len(points) == 1
    assert json.loads(points[0]["mapping_json"])["factor_columns"] == 2
    assert "mapping" not in points[0]
    assert summary["omitted_count"] == 0


def test_filters_are_explicit_and_duplicates_are_visible():
    source = [{"operator": "fft", "logN": 16, "mapping": {"backend": "shared-iterative"}},
              {"operator": "ntt", "logN": 16, "mapping": {"backend": "shared-iterative"}},
              {"operator": "fft", "logN": 16, "mapping": {"backend": "shared-iterative"}}]
    points, summary = normalize(source, operators={"fft"})
    assert len(points) == 1
    assert summary["omitted_count"] == 2
    assert {item["reason"] for item in summary["omitted"]} == {"operator-filter", "duplicate"}


def test_missing_mapping_is_rejected():
    with pytest.raises(ValueError, match="mapping"):
        normalize([{"operator": "fft", "logN": 16}])
