"""CPU contracts for the register-prefix diagonal twiddle supply axis."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evolutionary_search as evolution
import calibration_space
import calibration_seeds
from calibrate_local_hardware import _stage_template_key
import resident_mapping
import stage_cost_model as cost
import stage_service_model as service
from calibration_space import candidate_id
from research_compile_requests import compile_request


def mapping():
    return dict(schema_version=1, kind="butterfly", backend="online-reorder",
                fft_core="register-tile", logN=12, local_stages=6,
                stage_partition=[6, 6], prefix_threads=128, prefix_ept=8,
                suffix_threads=128, suffix_ept=8)


def sample(value):
    return dict(operator="fft", precision="fp64", backend="online-reorder",
                fft_core="register-tile", logN=12, batch=1, mapping_json=value)


def test_prefix_twiddle_validation_and_default():
    assert resident_mapping.validate_prefix_twiddle({}) == "runtime"
    assert resident_mapping.register_geometry(dict(mapping(), prefix_twiddle="table"))["prefix_twiddle"] == "table"
    with pytest.raises(ValueError, match="runtime or table"):
        resident_mapping.validate_prefix_twiddle(dict(mapping(), prefix_twiddle="recurrence"))
    with pytest.raises(ValueError, match="only to online-reorder"):
        resident_mapping.validate_prefix_twiddle(dict(prefix_twiddle="table", backend="shared-iterative"))


def test_prefix_twiddle_mutation_is_bidirectional_and_has_distinct_identity():
    base = mapping()
    for source, target in (("runtime", "table"), ("table", "runtime")):
        point = sample(json.dumps(dict(base, prefix_twiddle=source)))
        proposals = evolution.neighbors(point)
        changed = [p for p in proposals if json.loads(p["mapping_json"]).get("prefix_twiddle") == target]
        assert changed
        assert all(candidate_id(p) != candidate_id(point) for p in changed)
    assert any(json.loads(p["mapping_json"]).get("prefix_twiddle") == "table"
               for p in evolution.neighbors(sample(json.dumps(base))))


def test_prefix_twiddle_survives_candidate_projection_and_canonical_compile_request():
    base = mapping()
    workload = dict(operator="fft", precision="fp64", logN=12, batch=1)
    projected = calibration_space.mapping_point(workload, dict(base, prefix_twiddle="table"))
    assert projected["prefix_twiddle"] == "table"
    assert calibration_space.predicted_sample(projected)["prefix_twiddle"] == "table"
    request = compile_request(projected)
    assert request["prefix_twiddle"] == "table"
    omitted = calibration_space.mapping_point(workload, base)
    runtime = calibration_space.mapping_point(workload, dict(base, prefix_twiddle="runtime"))
    assert compile_request(omitted) == compile_request(runtime)
    assert "prefix_twiddle" not in compile_request(runtime)
    assert _stage_template_key(omitted) == _stage_template_key(runtime)
    assert _stage_template_key(projected) != _stage_template_key(runtime)
    with pytest.raises(ValueError, match="requires on-demand compilation"):
        compile_request(projected, "precompiled")


def test_table_radix4_never_aliases_linked_inventory():
    from test_research_compile_requests import radix4_point
    request = compile_request(radix4_point(prefix_twiddle="table"))
    assert request["prefix_twiddle"] == "table"
    assert request["prefix_codelet"] == "radix4"


def test_seed_and_precompile_reject_unconsumed_prefix_twiddle():
    invalid = dict(mapping(), backend="shared-iterative", prefix_twiddle="table")
    with pytest.raises(ValueError, match="only to online-reorder"):
        compile_request(sample(invalid))
    with pytest.raises(ValueError, match="only to online-reorder"):
        calibration_seeds._validate_strategy_axes(invalid)


@pytest.mark.parametrize("key", [cost.primitive_key, service.curve_key])
def test_prefix_twiddle_keys_are_group_local_and_preserve_runtime_default(key):
    base = mapping()
    omitted = sample(base)
    runtime = sample(dict(base, prefix_twiddle="runtime"))
    table = sample(dict(base, prefix_twiddle="table"))
    prefix = dict(index=0, core="register-tile", codelet="native", shared_layout="linear")
    suffix = dict(index=1, core="cufftdx-block", codelet="native", shared_layout="writer-aligned")
    assert key(omitted, prefix) == key(runtime, prefix)
    assert key(omitted, prefix) != key(table, prefix)
    assert key(omitted, suffix) == key(table, suffix)
