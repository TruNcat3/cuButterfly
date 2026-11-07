import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_compile_requests import compile_request


def point(mapping, **semantics):
    return dict(operator="fft", precision="fp32", logN=12, batch=32,
                mapping_json=json.dumps(mapping), **semantics)


@pytest.mark.parametrize("policy", ["research", "auto", "on-demand"])
def test_shared_columns_are_prepared_before_gpu_execution(policy):
    mapping = dict(backend="shared-iterative", stage_partition=[6,6], tile_threads=128,
                   data_tiles_per_cta=4)
    request = compile_request(point(mapping), policy)
    assert request["data_tiles_per_cta"] == 4
    with pytest.raises(ValueError, match="on-demand"):
        compile_request(point(mapping), "precompiled")


def radix4_point(**changes):
    mapping = dict(backend="online-reorder", fft_core="register-tile", local_stages=8,
                   stage_partition=[8, 10], prefix_codelet="radix4", prefix_threads=64,
                   prefix_ept=16, suffix_threads=512, suffix_ept=8)
    mapping.update(changes)
    return dict(operator="fft", precision="fp32", logN=sum(mapping["stage_partition"]),
                batch=32, mapping_json=mapping)


@pytest.mark.parametrize("policy", ["research", "auto", "on-demand", "precompiled"])
@pytest.mark.parametrize("suffix,ept,columns", [(10, 8, 4), (10, 16, 4), (10, 16, 8),
                                               (12, 16, 2), (12, 16, 4), (12, 32, 2), (12, 32, 4)])
def test_radix4_preparation_uses_linked_inventory(policy, suffix, ept, columns):
    value = radix4_point(stage_partition=[8, suffix], suffix_ept=ept,
                         suffix_threads=(1 << suffix) // ept * columns)
    assert compile_request(value, policy) is None


@pytest.mark.parametrize("changes", [
    {"prefix_shared_layout": "xor"}, {"suffix_core": "native"},
    {"prefix_codelet_lanes": 2, "prefix_ept": 8, "prefix_threads": 128},
    {"stage_partition": [6, 10], "prefix_ept": 8, "prefix_threads": 32},
    {"stage_partition": [8, 8], "suffix_threads": 128},
    {"prefix_threads": 32},
    {"suffix_ept": 32, "suffix_threads": 128},
    {"suffix_ept": 8, "suffix_threads": 256},
    {"exchange_chunks": [0, 2]}, {"local_stage_partitions": [[4, 4], []]},
    {"stage_partition": [5, 10], "prefix_ept": 4, "prefix_threads": 32,
     "local_stage_partitions": [[2, 3], []], "prefix_shared_layout": "xor"},
    {"stage_partition": [7, 10], "prefix_ept": 4, "prefix_threads": 128,
     "prefix_codelet_lanes": 2, "local_stage_partitions": [[3, 4], []]},
])
@pytest.mark.parametrize("policy", ["research", "auto", "on-demand"])
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_radix4_preparation_projects_new_geometry_to_runtime_jit(changes, policy, precision):
    value = dict(radix4_point(**changes), precision=precision)
    request = compile_request(value, policy)
    native = dict(value, mapping_json=dict(value["mapping_json"], prefix_codelet="native"))
    assert request == dict(compile_request(native, policy), prefix_codelet="radix4")
    with pytest.raises(ValueError, match="prebuilt inventory"):
        compile_request(value, "precompiled")


@pytest.mark.parametrize("changes", [
    {"prefix_threads": 65}, {"prefix_ept": 8},
    {"shared_layout": "linear"}, {"cross_twiddle": "table"},
    {"compute_unit": "radix4"}, {"backend": "factor-streamed"},
    {"factor_io_policies": ["dynamic", "dynamic"]},
    {"stage_overlap": True, "batch_tile_count": 0},
    {"execution_group_mappings": [{"codelet": "native"}, {}]},
    {"stage_partition": [8, 5, 5]},
    {"stage_partition": [5, 10], "prefix_ept": 4, "prefix_threads": 32},
    {"exchange_chunks": [2, 0]},
    {"exchange_chunks": [0, 3]},
    {"local_stage_partitions": [[4, 4], [5, 5]]},
    {"boundaries": [{"residency": "fused"}]},
    {"prefix_codelet_lanes": 3},
    {"prefix_codelet_lanes": 2, "prefix_threads": 128},
    {"prefix_shared_layout": "xor", "stage_partition": [7, 10],
     "prefix_ept": 4, "prefix_threads": 256, "prefix_codelet_lanes": 2,
     "local_stage_partitions": [[3, 4], []]},
])
@pytest.mark.parametrize("policy", ["research", "auto", "on-demand", "precompiled"])
def test_radix4_preparation_rejects_invalid_mappings_before_inventory_fallback(changes, policy):
    with pytest.raises(ValueError):
        compile_request(radix4_point(**changes), policy)


@pytest.mark.parametrize("change", [{"precision": "fp16"}, {"operator": "fwht"},
                                    {"operator": "ntt"}])
def test_radix4_preparation_keeps_operator_precision_contract(change):
    with pytest.raises(ValueError, match="FP32/FP64 FFT"):
        compile_request(dict(radix4_point(), **change))


@pytest.mark.parametrize("policy", ["research", "auto", "on-demand"])
def test_fp64_radix4_default_geometry_requires_jit(policy):
    value = dict(radix4_point(), precision="fp64")
    request = compile_request(value, policy)
    assert request["precision"] == "fp64" and request["prefix_codelet"] == "radix4"
    with pytest.raises(ValueError, match="prebuilt inventory"):
        compile_request(value, "precompiled")


def test_factor_request_matches_runtime_omitted_defaults_and_reuses_batch_schedule():
    mapping = dict(backend="factor-streamed", fft_core="cufftdx-block",
                   stage_partition=[12], factor_partition=[6, 6], factor_ept=8,
                   factor_columns=4, data_tiles_per_cta=1, prefetch_depth=0,
                   factor_slices=1, factor_overlap=False)
    expected = dict(backend="factor-streamed", precision="fp32", logN=12,
                    stage_partition=[12], factor_partition=[6, 6], factor_ept=8,
                    factor_columns=4, data_tiles_per_cta=1, prefetch_depth=0)
    assert compile_request(point(mapping)) == expected
    other = point(dict(mapping, factor_overlap=True), direction="inverse", normalization="inverse")
    other["batch"] = 128
    assert compile_request(other) == expected
    assert compile_request(point(dict(mapping, factor_slices=2))) == dict(expected, factor_slices=2)
    assert compile_request(point(dict(mapping, fft_core="register-tile"))) == dict(expected, fft_core="register-tile")


def test_shared_butterfly_and_ntt_have_distinct_runtime_request_fields():
    mapping = dict(backend="shared-iterative", stage_partition=[6, 6],
                   tile_threads=128, shared_layout="writer-aligned", complex_multiply="four-mul")
    request = compile_request(point(mapping))
    assert request == dict(backend="shared-iterative", operator="fft", precision="fp32",
                          logN=12, stage_partition=[6, 6], threads=128, writer_aligned=True,
                          accumulation="native", complex_multiply="four-mul", batch_tile=1)
    ntt = point(dict(mapping, threads_per_block=256, dataflow_layout="linear"))
    ntt.update(operator="ntt", precision="word64", modulus=576460756061519873)
    assert compile_request(ntt) == dict(backend="shared-iterative", operator="ntt", precision="word64",
                                       logN=12, stage_partition=[6, 6], threads=256, writer_aligned=False,
                                       output_order="natural", batch_tile=1)
    assert compile_request(point(mapping), "auto") is None


def shared_physical_point(operator="fft", threads=(64, 256)):
    entries = ([dict(core="radix2", threads_per_block=t, units_per_cta=1)
                for t in threads] if operator == "ntt" else
               [dict(core="scalar", exchange="shared", threads=t, ept=0,
                     codelet="native", io_policy="dynamic") for t in threads])
    mapping = dict(backend="shared-iterative", stage_partition=[3, 3, 6],
                   tile_threads=128, threads_per_block=128,
                   execution_group_mappings=entries)
    if operator == "ntt":
        mapping["boundary_mappings"] = [dict(storage="resident-fused"), dict(storage="full-scratch")]
    else:
        mapping["boundaries"] = [dict(residency="fused"), dict(residency="global-scratch")]
    p = point(mapping)
    p.update(operator=operator, precision="word32" if operator == "ntt" else "fp32")
    return p


@pytest.mark.parametrize("operator", ["fft", "fwht", "ntt"])
@pytest.mark.parametrize("policy", ["research", "auto", "on-demand"])
def test_shared_physical_mapping_projects_threads_after_fused_boundaries(operator, policy):
    request = compile_request(shared_physical_point(operator), policy)
    assert request["stage_partition"] == [6, 6]
    assert request["group_threads"] == [64, 256]
    assert request["threads"] == 128


@pytest.mark.parametrize("operator", ["fft", "ntt"])
def test_shared_empty_physical_mapping_keeps_default_request(operator):
    p = shared_physical_point(operator, threads=())
    request = compile_request(p)
    assert "group_threads" not in request
    mapping = json.loads(p["mapping_json"])
    mapping.pop("execution_group_mappings")
    assert compile_request(dict(p, mapping_json=mapping)) == request
    assert compile_request(shared_physical_point(operator), "precompiled") is None


@pytest.mark.parametrize("operator", ["fft", "ntt"])
@pytest.mark.parametrize("policy", ["research", "auto", "on-demand", "precompiled"])
@pytest.mark.parametrize("threads", [(64,), (64, 128, 256), (0, 128), (33, 128),
                                     (1056, 128), (True, 128), (64.0, 128), ("64", 128)])
def test_shared_physical_mapping_rejects_bad_thread_counts_and_values(operator, policy, threads):
    with pytest.raises(ValueError):
        compile_request(shared_physical_point(operator, threads), policy)


@pytest.mark.parametrize("operator,changes", [
    ("fft", {"core": "register-tile"}), ("fft", {"exchange": "shuffle"}),
    ("fft", {"ept": 4}), ("fft", {"codelet": "radix4"}),
    ("fft", {"io_policy": "static-unrolled"}),
    ("ntt", {"core": "radix4-shoup"}), ("ntt", {"core": "radix2-shoup"}),
    ("ntt", {"units_per_cta": 2}),
    ("ntt", {"data_space": 2}), ("ntt", {"data_time": 2}),
    ("ntt", {"role_stages": 2}), ("ntt", {"token_interleave": 2}),
    ("ntt", {"coefficient_reuse_stages": 1}), ("ntt", {"cta_weight": 1}),
])
def test_shared_physical_mapping_rejects_unconsumed_implementation_axes(operator, changes):
    p = shared_physical_point(operator)
    mapping = json.loads(p["mapping_json"])
    mapping["execution_group_mappings"][0].update(changes)
    with pytest.raises(ValueError):
        compile_request(dict(p, mapping_json=mapping))


@pytest.mark.parametrize("entries", [None, "64x256", [{}, None]])
def test_shared_physical_mapping_rejects_malformed_descriptors(entries):
    p = shared_physical_point()
    mapping = json.loads(p["mapping_json"])
    mapping["execution_group_mappings"] = entries
    with pytest.raises(ValueError):
        compile_request(dict(p, mapping_json=mapping))


def test_ntt_preparation_preserves_target_output_semantics_not_donor_mapping():
    mapping = dict(backend="shared-iterative", stage_partition=[6, 6],
                   threads_per_block=128, output_order="natural")
    p = point(mapping)
    p.update(operator="ntt", precision="word64", output_order="bit-reversed")
    request = compile_request(p)
    assert request["output_order"] == "bit-reversed"
    assert request != compile_request(dict(p, output_order="natural"))
    p.pop("output_order")
    assert compile_request(dict(p, placement="bit-reversed")) == request
    assert compile_request(dict(p, placement="out-of-place"))["output_order"] == "natural"
    # Direction and modulus are supplied at launch; they do not change codegen.
    assert compile_request(dict(p, output_order="bit-reversed", direction="inverse",
                                modulus=576460756061519873)) == request
    with pytest.raises(ValueError, match="output"):
        compile_request(dict(p, output_order="appt-static"))


def test_register_request_requires_resolved_axes_and_ignores_runtime_strides():
    mapping = dict(backend="online-reorder", fft_core="register-tile", local_stages=6,
                   prefix_threads=128, prefix_ept=8, suffix_threads=128, suffix_ept=8,
                   stage_overlap=True, batch_tile_count=2)
    request = compile_request(point(mapping, element_stride=2, batch_stride=8192))
    assert request == {k: v for k, v in dict(mapping, precision="fp32", logN=12).items()
                       if k not in {"backend", "stage_overlap", "batch_tile_count"}}
    with pytest.raises(ValueError, match="resolved prefix_threads"):
        compile_request(point(dict(mapping, prefix_threads=0)))
    assert compile_request(point(dict(mapping, fft_core="cufftdx-block"))) is None
    assert compile_request(point(mapping), "precompiled") is None


def test_unresolved_local_axis_is_explicit_and_workload_mapping_is_not_mutated():
    p = point(dict(backend="shared-iterative", tile_threads=128, stage_partition=[]))
    saved = dict(p)
    with pytest.raises(ValueError, match="resolved local_stages"):
        compile_request(p)
    assert p == saved


def test_shared_defaults_and_resident_edges_match_physical_lowering():
    mapping = dict(backend="shared-iterative", tile_threads=128, local_stages=5,
                   stage_partition=[])
    assert compile_request(point(mapping))["stage_partition"] == [5, 5, 2]
    fused = dict(mapping, stage_partition=[3, 3, 6],
                 boundaries=[{"residency": "fused"}, {"residency": "global-scratch"}])
    assert compile_request(point(fused))["stage_partition"] == [6, 6]
    ntt = point(dict(backend="shared-iterative", stage_partition=[], flow_tile_log_n=0,
                     threads_per_block=0, dataflow_layout="hermes-xor"))
    ntt.update(operator="ntt", precision="word32")
    request = compile_request(ntt)
    assert request["stage_partition"] == [10, 2] and request["threads"] == 128


def test_factor_default_partition_and_explicit_register_partition_take_runtime_precedence():
    mapping = dict(backend="factor-streamed", fft_core="cufftdx-block", stage_partition=[],
                   factor_partition=[], factor_ept=8, factor_columns=4, data_tiles_per_cta=1)
    request = compile_request(point(mapping))
    assert request["stage_partition"] == request["factor_partition"] == [12]
    register = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                    local_stages=10, prefix_threads=128, prefix_ept=8, suffix_threads=128, suffix_ept=8)
    assert compile_request(point(register))["local_stages"] == 6


def test_codegen_strategy_axes_are_forwarded_only_when_non_default():
    mapping = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                   local_stages=6, prefix_threads=128, prefix_ept=8,
                   suffix_threads=128, suffix_ept=8,
                   prefix_codelet="cufftdx-thread", prefix_shared_layout="xor")
    request = compile_request(point(mapping))
    assert request["prefix_codelet"] == "cufftdx-thread"
    assert request["prefix_shared_layout"] == "xor"
    # Historical native/linear defaults remain omitted from the JIT request.
    default = dict(mapping, prefix_codelet="native", prefix_shared_layout="linear")
    assert "prefix_codelet" not in compile_request(point(default))
    assert "prefix_shared_layout" not in compile_request(point(default))


def test_register_cooperative_prefix_request_forwards_non_default_lanes_and_defaults_omit():
    mapping = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                   prefix_threads=256, prefix_ept=4, suffix_threads=128, suffix_ept=8,
                   prefix_codelet_lanes=2)
    request = compile_request(point(mapping))
    assert request["prefix_codelet_lanes"] == 2
    assert request["prefix_threads"] == 256 and request["prefix_ept"] == 4

    default = dict(mapping, prefix_codelet_lanes=1, prefix_threads=128, prefix_ept=8)
    omitted = dict(default)
    omitted.pop("prefix_codelet_lanes")
    assert compile_request(point(default)) == compile_request(point(omitted))
    assert "prefix_codelet_lanes" not in compile_request(point(default))


@pytest.mark.parametrize("lanes", [True, 3, 16, 2.0, "2"])
def test_register_cooperative_prefix_rejects_invalid_lane_values(lanes):
    mapping = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                   prefix_threads=256, prefix_ept=4, suffix_threads=128, suffix_ept=8,
                   prefix_codelet_lanes=lanes)
    with pytest.raises(ValueError, match="prefix_codelet_lanes"):
        compile_request(point(mapping))


def test_register_cooperative_prefix_rejects_non_integral_columns_and_irrelevant_lane_axis():
    mapping = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                   prefix_threads=130, prefix_ept=8, suffix_threads=128, suffix_ept=8)
    with pytest.raises(ValueError, match="integral"):
        compile_request(point(mapping))
    irrelevant = dict(backend="online-reorder", fft_core="cufftdx-block", stage_partition=[6, 6],
                      prefix_codelet_lanes=2)
    with pytest.raises(ValueError, match="only for online-reorder register-tile"):
        compile_request(point(irrelevant))


def test_factor_io_policy_is_compiler_identity_and_default_is_omitted():
    mapping = dict(backend="factor-streamed", fft_core="cufftdx-block",
                   stage_partition=[12], factor_partition=[6, 6], factor_ept=8,
                   factor_columns=4, data_tiles_per_cta=1, prefetch_depth=0)
    assert "factor_io_policies" not in compile_request(point(mapping))
    policy = dict(mapping, factor_io_policies=["static-unrolled", "dynamic"])
    assert compile_request(point(policy))["factor_io_policies"] == ["static-unrolled", "dynamic"]
    with pytest.raises(ValueError, match="match factor_partition"):
        compile_request(point(dict(mapping, factor_io_policies=["static-unrolled"])))


def test_unsupported_prefix_strategy_is_not_silently_ignored():
    mapping = dict(backend="online-reorder", fft_core="cufftdx-block", stage_partition=[6, 6],
                   local_stages=6, prefix_codelet="cufftdx-thread")
    with pytest.raises(ValueError, match="only for online-reorder register-tile"):
        compile_request(point(mapping))
    register = dict(backend="online-reorder", fft_core="register-tile", stage_partition=[6, 6],
                    local_stages=6, prefix_threads=128, prefix_ept=8,
                    suffix_threads=128, suffix_ept=8, prefix_codelet="cufftdx-thread")
    fp64 = point(register)
    fp64["precision"] = "fp64"
    with pytest.raises(ValueError, match="unavailable for FP64"):
        compile_request(fp64)
