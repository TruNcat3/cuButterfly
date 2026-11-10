from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts")

from fit_local_cost_model import FEATURE_NAMES, feature_vector


def test_xd_physical_cost_features_include_boundaries_and_prefixes():
    mapping = {"kind": "xd", "physical": {
        "complex_only": False, "inverse": True, "output_scale": 0.5,
        "workspace_bytes": 2 * 1024 * 1024,
        "cost": {"prefix_launches": 5, "permutation_launches": 2,
                  "scale_kernel_launches": 0, "c2r_d2d_copy_launches": 0,
                  "real_boundary_launches": 1, "total_launches": 8,
                  "workspace_bytes": 3 * 1024 * 1024},
        "axes": [{"prefix_loops": 3, "boundary": "permute_nd", "permutation": [1, 0]},
                  {"prefix_loops": 1, "boundary": "none", "permutation": [0, 1]}]}}
    row = {"operator": "fft", "precision": "fp32", "N": 64, "batch": 2,
           "mapping_json": json.dumps(mapping), "input_bytes": 1024, "output_bytes": 2048}
    features = dict(zip(FEATURE_NAMES, feature_vector(row)))
    assert features["xd_axis_count"] == 2
    assert features["xd_prefix_loops"] == 5
    assert features["xd_permutation_count"] == 2
    assert features["xd_workspace_mib"] == 3
    assert features["xd_boundary_bytes_mib"] > 0
    assert features["xd_scale_kernel_count"] == 0
    assert features["xd_d2d_copy_count"] == 0
    assert features["xd_real_boundary_launches"] == 1
    assert features["xd_launch_count"] == 8


def test_xd_rank_one_c2r_retains_d2d_boundary_cost():
    mapping = {"kind": "xd", "physical": {"complex_only": False,
               "inverse": True, "axes": [], "workspace_bytes": 4096}}
    row = {"operator": "fft", "precision": "fp32", "N": 64, "batch": 1,
           "mapping_json": mapping}
    features = dict(zip(FEATURE_NAMES, feature_vector(row)))
    assert features["xd_d2d_copy_count"] == 1
    assert features["xd_real_boundary_launches"] == 1
    assert features["xd_launch_count"] == 2
