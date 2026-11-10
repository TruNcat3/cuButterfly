import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/tools"))
import prepare_medium_large_mechanism_20261012 as medium
import prepare_medium_large_residency_20261012_v2 as residency
import validate_fixed_mapping_error_20261012 as errors


def test_residency_keeps_physical_core_and_declares_induced_axes():
    config = json.loads(residency.CONFIG.read_text())
    source = json.loads((ROOT / config["source_launch"]).read_text())
    study, points = residency.study_and_points(config, source)
    baseline, treatment = [case["variants"][i]["mapping"] for i in (0, 1) for case in study["cases"]]
    assert baseline["stage_partition"] == treatment["stage_partition"] == [10, 10]
    assert baseline["tile_threads"] == treatment["tile_threads"] == 256
    assert baseline["local_stage_partitions"] == []
    assert treatment["local_stage_partitions"] == [[2] * 5, [2] * 5]
    assert points[0]["writer_aligned"] is False and points[1]["writer_aligned"] is False
    assert "register pressure" in config["changed_implementation_axes"][-1]


def test_fwht_reference_report_has_distribution_without_threshold():
    values = np.arange(16, dtype=np.float32).reshape(1, 16)
    expected = errors.fwht_reference(values)
    assert expected.shape == values.shape
    assert expected[0, 0] == 120
    assert expected[0, -1] == 0
    config = json.loads(residency.CONFIG.read_text())
    assert "unassessed" in config["numerical_policy"]


def test_residency_cpu_preparation_does_not_expand_matrix():
    config = json.loads(medium.CONFIG.read_text())
    assert config["candidate_matrix"]["status"] == "planned_not_queued"
    assert json.loads(residency.CONFIG.read_text())["operator"] == "fwht"
