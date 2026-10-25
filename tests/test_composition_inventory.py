"""CPU-only contracts for native composition request generation."""
import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from composition_inventory import build_inventory
from calibration_space import command_for


def _record(backend="shared-iterative", operator="fft", precision="fp32", batch=16,
            partition=(4, 4), core="scalar", direction="forward"):
    mapping = {"backend": backend, "fft_core": core, "stage_partition": list(partition),
               "stage_overlap": False, "batch_tile_count": 1,
               "factor_overlap": False, "factor_slices": 1,
               "shared_layout": "writer-aligned", "cross_twiddle": "recurrence",
               "direct_boundary": "direct-strided", "local_exchange": "shared",
               "complex_multiply": "four-mul", "factor_columns": 2,
               "factor_partition": list(partition), "factor_ept": 8}
    sample = {"operator": operator, "backend": backend, "precision": precision,
              "direction": direction, "accumulation": "native", "batch": str(batch),
              "logN": str(sum(partition)), "element_stride": "1",
              "batch_stride": str(1 << sum(partition)), "correct": "1",
              "placement": "out-of-place", "normalization": "none",
              "fft_core": core, "mapping_json": json.dumps(mapping),
              "execution_groups_json": json.dumps([
                  {"stage_count": n, "core": core, "threads": 128} for n in partition]),
              "kernel_ms": "0.1", "workspace_bytes": "9876"}
    return {"name": backend + "-" + operator, "correct": True,
            "status": "measured", "samples": [sample]}


def _pairs(inventory):
    pairs = {}
    for request in inventory["requests"]:
        pairs.setdefault(request["pair_id"], []).append(request)
    return list(pairs.values())


def test_shared_pairs_have_original_and_explicit_tail_and_no_fabricated_metadata():
    record = _record()
    original = copy.deepcopy(record)
    result = build_inventory([record])
    assert record == original
    assert {r["point"]["batch"] for r in result["requests"]} == {16, 15}
    assert len(_pairs(result)) == 6
    for pair in _pairs(result):
        assert {r["role"] for r in pair} == {"bulk", "overlap"}
        for r in pair:
            p = r["point"]
            m = json.loads(p["mapping_json"])
            assert p["stage_overlap"] == m["stage_overlap"] == (r["role"] == "overlap")
            assert p["batch_tile_count"] == m["batch_tile_count"]
            assert m["stage_partition"] == [4, 4]
            assert "execution_groups_json" not in p
            assert "kernel_ms" not in p and "workspace_bytes" not in p
            assert r["holdout"] is True
            assert p["batch_stride"] == 256
        if pair[0]["workload"] == "additional-tail":
            assert pair[1]["point"]["batch"] % pair[1]["point"]["batch_tile_count"]


@pytest.mark.parametrize("core", ["cufftdx-block", "register-tile"])
def test_online_two_group_fft(core):
    result = build_inventory([_record("online-reorder", core=core)])
    assert result["requests"]
    assert "online-reorder/fft/" + core in result["coverage"]["covered"]


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_factor_three_way_control_holds_mapping_and_slices_fixed(precision):
    record = _record("factor-streamed", precision=precision, partition=(6, 6, 6),
                     core="cufftdx-block", batch=2)
    result = build_inventory([record])
    assert len(_pairs(result)) == 3
    for pair in _pairs(result):
        assert [r["role"] for r in pair] == ["bulk", "sliced-serial", "overlap"]
        maps = [json.loads(r["point"]["mapping_json"]) for r in pair]
        assert maps[0]["factor_slices"] == 1
        assert maps[1]["factor_slices"] == maps[2]["factor_slices"] > 1
        assert maps[1]["factor_overlap"] is False
        assert maps[2]["factor_overlap"] is True
        for r, mapping in zip(pair, maps):
            assert "factor_slices" not in r["point"]
            assert "factor_overlap" not in r["point"]
            for key in ("stage_overlap", "factor_overlap", "factor_slices", "batch_tile_count"):
                mapping.pop(key)
        assert maps[0] == maps[1] == maps[2]


def test_inverse_uint64_ntt_keeps_exact_large_moduli_and_semantics():
    record = _record(operator="ntt", precision="uint64", direction="inverse",
                     partition=(8, 4))
    modulus = 18446744069414584321
    record["samples"][0].update(modulus=str(modulus), word64modulus=str(modulus))
    result = build_inventory([record])
    assert result["requests"]
    for r in result["requests"]:
        p = r["point"]
        assert p["modulus"] == modulus
        assert "word64modulus" not in p
        assert isinstance(p["modulus"], int)
        assert p["direction"] == "inverse" and p["precision"] == "uint64"
        assert json.loads(p["mapping_json"])["stage_partition"] == [8, 4]
    assert json.loads(json.dumps(result))["requests"][0]["point"]["modulus"] == modulus


def test_ntt_cli_roundtrip_keeps_inverse_orders_and_modulus_without_unknown_aliases():
    record = _record(operator="ntt", precision="uint64", direction="inverse")
    modulus = 18446744069414584321
    record["samples"][0].update(
        input_order="bit-reversed", output_order="bit-reversed", modulus=str(modulus),
        word64modulus=str(modulus), inverse="1", warmup="10", repeat="20")
    for request in build_inventory([record])["requests"]:
        point = request["point"]
        assert point["input_order"] == point["output_order"] == "bit-reversed"
        command = command_for("cuntt_bench", point)
        assert command[command.index("--input-order") + 1] == "bit-reversed"
        assert all(command[i + 1] == "bit-reversed" for i, arg in enumerate(command)
                   if arg == "--output-order")
        assert command.count("--inverse") == 1
        assert command[command.index("--modulus") + 1] == str(modulus)
        assert command[command.index("--word-bits") + 1] == "64"
        for flag in ("--word64modulus", "--warmup", "--repeat", "--factor-overlap"):
            assert flag not in command


def test_ntt_mapping_preserves_public_keys_without_butterfly_factor_policy():
    record = _record(operator="ntt", precision="uint64", partition=(8, 4))
    mapping = {"schema_version": 1, "kind": "ntt", "backend": "shared-iterative",
               "stage_partition": [8, 4], "local_stage_partitions": [[4, 4], [4]],
               "exchange_chunks": [0, 0], "stage_overlap": False,
               "batch_tile_count": 1, "modular_multiply": "shoup",
               "subgraph_mappings": [], "execution_group_mappings": [],
               "boundary_mappings": [], "appt_role_mapping": {"data_time_role_mask": 7}}
    record["samples"][0]["mapping_json"] = json.dumps(mapping)
    requests = build_inventory([record])["requests"]
    assert requests
    for request in requests:
        actual = json.loads(request["point"]["mapping_json"])
        assert not any(key.startswith("factor_") for key in actual)
        assert actual.keys() == mapping.keys()
        for key in mapping.keys() - {"stage_overlap", "batch_tile_count"}:
            assert actual[key] == mapping[key]
        command = command_for("cuntt_bench", request["point"])
        assert json.loads(command[command.index("--mapping-json") + 1]) == actual


def test_structured_cli_roundtrip_lowers_every_matrix():
    record = _record(operator="structured-2x2")
    record["samples"][0]["stage_matrices"] = "1:0.25:-0.5:1x2:0:0:2"
    for request in build_inventory([record])["requests"]:
        point = request["point"]
        assert point["stage_matrix"] == ["1,0.25,-0.5,1", "2,0,0,2"]
        assert "stage_matrices" not in point
        command = command_for("cubutterfly_bench", point)
        assert [command[i + 1] for i, arg in enumerate(command)
                if arg == "--stage-matrix"] == point["stage_matrix"]
        assert "--stage-matrices" not in command
        assert "--factor-overlap" not in command


def test_factor_cli_uses_mapping_policy_without_unknown_scalar_flags():
    record = _record("factor-streamed", partition=(6, 6, 6), core="cufftdx-block", batch=2)
    for request in build_inventory([record])["requests"]:
        command = command_for("cubutterfly_bench", request["point"])
        assert "--factor-slices" not in command and "--factor-overlap" not in command
        mapping = json.loads(command[command.index("--mapping-json") + 1])
        assert mapping["factor_overlap"] == (request["role"] == "overlap")


def test_element_cap_omits_whole_original_instead_of_replacing_it():
    result = build_inventory([_record()], max_elements=256 * 15)
    assert {r["point"]["batch"] for r in result["requests"]} == {15}
    omission = next(o for o in result["omissions"] if o["reason"] == "whole-workload-element-limit")
    assert omission["batch"] == 16 and omission["workload"] == "original"
    assert all(r["workload"] == "additional-tail" for r in result["requests"])
    assert result["coverage"]["status"] == "partial"


def test_dedup_budget_unsupported_and_unverified_seeds_are_audited():
    first = _record()
    second, third = copy.deepcopy(first), copy.deepcopy(first)
    second["samples"][0]["batch"] = "32"
    third["samples"][0]["batch"] = "64"
    invalid = _record("factor-streamed", partition=(6, 6), core="cufftdx-block")
    unverified = _record()
    unverified["correct"] = False
    result = build_inventory([first, copy.deepcopy(first), second, third, invalid, unverified])
    reasons = [o["reason"] for o in result["omissions"]]
    assert "duplicate-semantic-mapping" in reasons
    assert "design-seed-budget" in reasons
    assert "factor-requires-at-least-three-physical-groups" in reasons
    assert "source-not-verified" in reasons
    assert len({r["id"] for r in result["requests"]}) == len(result["requests"])
    assert {o["source"]["record_index"] for o in result["omissions"]} >= {1, 4, 5}


def test_inventory_selection_ignores_performance_and_is_repeatable():
    records = [_record(), _record("online-reorder", core="register-tile")]
    result = build_inventory(records)
    records[0]["median_kernel_ms"] = 999999
    records[0]["samples"][0]["kernel_ms"] = "0.0000001"
    assert build_inventory(records) == result


def test_ineligible_seeds_do_not_consume_design_budget():
    # All three share the same family; B=1 and B=1024 sort before B=16.
    small = _record(batch=1)
    large = _record(batch=1024)
    legal = _record(batch=16)
    result = build_inventory([small, large, legal], max_seeds_per_family=1,
                             max_elements=256 * 16)
    assert result["requests"]
    assert {r["source"]["record_index"] for r in result["requests"]} == {2}
    assert {r["point"]["batch"] for r in result["requests"]} == {16, 15}
    reasons = {(o["source"]["record_index"], o["reason"]) for o in result["omissions"]}
    assert (0, "no-requested-legal-tile-or-slices") in reasons
    assert (1, "whole-workload-element-limit") in reasons
    assert not any(o["reason"] == "design-seed-budget" for o in result["omissions"])


def test_all_unsupported_does_not_claim_complete():
    result = build_inventory([_record(partition=(8,)), _record("hierarchical")])
    assert result["requests"] == []
    assert result["coverage"]["status"] == "partial"
    assert result["coverage"]["gaps"]


def test_factor_slice_upper_bound_is_last_digit_over_columns():
    record = _record("factor-streamed", partition=(6, 6, 5), core="cufftdx-block", batch=2)
    mapping = json.loads(record["samples"][0]["mapping_json"])
    mapping["factor_columns"] = 16
    record["samples"][0]["mapping_json"] = json.dumps(mapping)
    result = build_inventory([record])
    assert {r["tile_or_slices"] for r in result["requests"]} == {2}
    assert all(r["point"]["batch"] == 2 for r in result["requests"])


def test_float_modulus_cannot_silently_enter_exact_integer_requests():
    record = _record(operator="ntt", precision="uint64")
    record["samples"][0]["modulus"] = float(18446744069414584321)
    result = build_inventory([record])
    assert not result["requests"]
    assert result["omissions"][0]["reason"] == "invalid-workload-integers"


def test_missing_precision_cannot_fall_back_to_benchmark_default():
    record = _record(operator="ntt", precision="uint64")
    del record["samples"][0]["precision"]
    result = build_inventory([record])
    assert not result["requests"]
    assert result["omissions"][0]["reason"] == "required-public-semantics-unavailable"
    assert result["omissions"][0]["fields"] == ["precision"]


@pytest.mark.parametrize("kwargs", [{"max_elements": 0}, {"tile_counts": (3,)},
                                    {"max_seeds_per_family": 0}])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        build_inventory([], **kwargs)
