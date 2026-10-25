"""CPU checks for explicit bounded populations, independent of GPU support."""
import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_bounded_semantic_retest import prepare_configuration, RANK_POINT_CAPS
from prepare_semantic_retest import configuration as original_configuration
from semantic_campaign_manifest import semantic_id, validate_config


def scaling(config):
    return [p for p in config["populations"] if p["name"].startswith("size-batch-scaling-")]


def test_default_bounds_and_short_n_big_batch():
    config, policy = prepare_configuration()
    pairs = {(tuple(shape), batch) for p in scaling(config)
             for shape in p["shapes"] for batch in p["batches"]}
    assert ((256,), 4096) in pairs
    assert ((4096,), 256) in pairs
    assert ((65536,), 16) in pairs
    assert ((1048576,), 1) in pairs
    assert ((1024, 1024), 1) in pairs
    assert ((64, 64, 64), 1) in pairs
    assert ((1048576,), 16) not in pairs
    assert all(math.prod(shape) * batch <= 1 << 20 for shape, batch in pairs)
    assert all(math.prod(shape) <= RANK_POINT_CAPS[len(shape)] for shape, batch in pairs)
    assert policy["retained_shape_batch_combinations"] == len(pairs) == 20
    assert policy["excluded_shape_batch_combinations"] == 24


def test_all_functional_populations_and_missing_feature_regimes_retained():
    config, policy = prepare_configuration()
    old = original_configuration()
    for population in old["populations"]:
        if population["name"] == "size-batch-scaling":
            continue
        assert population == next(p for p in config["populations"] if p["name"] == population["name"])
    original_scaling = next(p for p in old["populations"] if p["name"] == "size-batch-scaling")
    for population in scaling(config):
        for dimension in ("regimes", "directions", "normalizations", "placements", "order_patterns", "layouts"):
            assert population[dimension] == original_scaling[dimension]
    names = {regime["name"] for regime in scaling(config)[0]["regimes"]}
    assert {"subset-zeta-uint64", "superset-zeta-uint64", "fft-r2c-fp16",
            "fft-c2r-bf16", "fft-fp32-compute-fp64"} <= names
    assert policy["non_performance_populations"] == ["rank-boundary"]


def test_complete_exclusion_audit_and_no_batch_substitution():
    config, policy = prepare_configuration()
    old_scaling = next(p for p in original_configuration()["populations"] if p["name"] == "size-batch-scaling")
    expected = {(tuple(shape), batch) for shape in old_scaling["shapes"] for batch in old_scaling["batches"]}
    assert {(tuple(row["shape"]), row["batch"]) for row in policy["shape_batch_audit"]} == expected
    assert len(policy["shape_batch_audit"]) == 44
    for population in scaling(config):
        assert len(population["shapes"]) == 1
        shape = population["shapes"][0]
        retained = [row["batch"] for row in policy["shape_batch_audit"]
                    if row["shape"] == shape and row["decision"] == "retained"]
        assert population["batches"] == retained
    extreme = next(row for row in policy["shape_batch_audit"] if row["shape"] == [16777216] and row["batch"] == 4096)
    assert extreme["reasons"] == ["rank-point-cap", "total-logical-element-cap"]


def test_custom_element_budget_and_fixed_dimension_caps():
    config, policy = prepare_configuration(1 << 24)
    assert policy["max_scaling_logical_elements"] == 1 << 24
    assert all(math.prod(p["shapes"][0]) <= RANK_POINT_CAPS[len(p["shapes"][0])] for p in scaling(config))
    assert all(math.prod(p["shapes"][0]) * batch <= 1 << 24 for p in scaling(config) for batch in p["batches"])
    tiny, _ = prepare_configuration(1)
    assert not scaling(tiny)
    assert len(tiny["populations"]) == 3


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "1048576"])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_configuration(budget)


def test_deterministic_configuration_and_hashes():
    first, policy = prepare_configuration()
    saved = copy.deepcopy(first)
    validate_config(first)
    assert first == saved
    assert (first, policy) == prepare_configuration()
    assert policy["config_sha256"] == semantic_id(first)
    assert policy["source_config_sha256"] == semantic_id(original_configuration())
