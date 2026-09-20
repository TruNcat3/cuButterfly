import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/comprehensive_20260919/workloads.json"

_SPEC = importlib.util.spec_from_file_location(
    "crypto_application_matrix", ROOT / "scripts/build_crypto_application_matrix.py")
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


def _documents():
    return matrix.build_documents(ORIGINAL)


REUSED = ROOT / "results/batch_precision_20260920/workloads.json"


def test_catalog_primes_bit_widths_and_two_adicity():
    specs = matrix.modulus_specs()
    assert len(specs) == 8
    for spec in specs.values():
        modulus = int(spec["modulus"])
        assert modulus.bit_length() == int(spec["bits"])
        assert matrix._miller_rabin(modulus)
        assert matrix._two_adicity(modulus - 1) == int(spec["two_adicity"])
        assert modulus < 1 << 63
        if spec["word_bits"] == 32:
            assert modulus < 1 << 31
    synthetic = [spec for spec in specs.values()
                 if spec["application"].startswith("synthetic")]
    assert [spec["bits"] for spec in synthetic] == [30, 40, 50, 60, 62]
    assert all(spec["two_adicity"] >= 18 for spec in synthetic)
    # This is a known strong pseudoprime to several small witnesses; the
    # complete deterministic 64-bit witness set must still reject it.
    assert not matrix._miller_rabin(341550071728321)


def test_finite_population_counts_and_expected_batch_axes():
    documents = _documents()
    assert len(documents["workloads.json"]["workloads"]) == 217
    assert len(documents["composed_workloads.json"]["workloads"]) == 138
    assert len(documents["population.json"]["cells"]) == 291
    assert len(documents["precompile_workloads.json"]["workloads"]) == 50
    direct = documents["workloads.json"]["workloads"]
    assert {int(row["batch"]) for row in direct} == {1, 3, 7, 16, 256}
    assert len({matrix.cell_key(row) for row in direct}) == len(direct)
    assert 300 <= len(direct) + len(documents["composed_workloads.json"]["workloads"]) <= 400


def test_reuse_workloads_merges_exact_cells_and_records_studies():
    documents = matrix.build_documents(ORIGINAL, [REUSED])
    assert len(documents["workloads.json"]["workloads"]) == 208
    assert len(documents["population.json"]["cells"]) == 566
    assert documents["population.json"]["limits"]["reused_studies"] == [
        "batch_precision_20260920", "comprehensive_20260919"]
    original = json.loads(ORIGINAL.read_text())["workloads"]
    reused = json.loads(REUSED.read_text())["workloads"]
    by_id = {cell["id"]: cell for cell in documents["population.json"]["cells"]}
    original_ids = {matrix.cell_key(row) for row in original}
    reused_ids = {matrix.cell_key(row) for row in reused}
    assert original_ids.isdisjoint(reused_ids)
    assert all(by_id[key]["source_studies"] == ["comprehensive_20260919"]
               for key in original_ids)
    assert all(by_id[key]["source_studies"] == ["batch_precision_20260920"]
               for key in reused_ids)


def test_root_constraints_reject_illegal_small_modulus_lengths():
    specs = matrix.modulus_specs()
    dilithium = specs["dilithium23"]
    assert matrix.valid_log_n(dilithium, 8)
    assert matrix.valid_log_n(dilithium, 13)
    assert not matrix.valid_log_n(dilithium, 13, mode="negacyclic")
    for row in matrix._direct_workloads(specs):
        spec = next(item for item in specs.values() if str(item["modulus"]) == row["modulus"])
        assert matrix.valid_log_n(spec, row["logN"])
        assert row["logN"] != 13 or spec["name"] != "dilithium23"
    for row in matrix._composed_workloads(specs):
        if row["mode"] == "negacyclic":
            for modulus in row["moduli"]:
                spec = next(item for item in specs.values() if str(item["modulus"]) == modulus)
                assert matrix.valid_log_n(spec, row["logN"], mode="negacyclic")


def test_memory_cap_and_word_width_are_separate_from_modulus_bits():
    documents = _documents()
    for cell in documents["population.json"]["cells"]:
        assert int(cell["input_bytes"]) <= matrix.MAX_INPUT_BYTES
    for row in documents["composed_workloads.json"]["workloads"]:
        assert int(row["input_bytes"]) <= matrix.MAX_INPUT_BYTES
        assert int(row["word_bits"]) in (32, 64)
        assert all(isinstance(modulus, str) for modulus in row["moduli"])
        if row.get("composition") == "rns":
            assert int(row["rns_channel_count"]) == len(row["moduli"])

    high_modulus = next(row for row in documents["workloads.json"]["workloads"]
                         if row["modulus"] == "2391411402133733377")
    assert high_modulus["precision"] == "word64"
    assert int(high_modulus["modulus"]).bit_length() == 62


def test_rns_channel_count_is_not_batch_count():
    rns = [row for row in _documents()["composed_workloads.json"]["workloads"]
           if row.get("composition") == "rns"]
    assert {int(row["batch"]) for row in rns} == {1, 7, 16}
    assert {int(row["rns_channel_count"]) for row in rns} == {1, 2, 4}
    assert any(int(row["batch"]) == 7 and int(row["rns_channel_count"]) == 2 for row in rns)
    assert all(row["rns_stream_order"] == "single-stream" for row in rns)


def test_composed_math_ids_merge_rns_l1_without_dropping_membership():
    composed = _documents()["composed_workloads.json"]["workloads"]
    assert len({row["id"] for row in composed}) == len(composed)
    l1 = [row for row in composed
          if row.get("composition") == "rns" and int(row["rns_channel_count"]) == 1]
    assert l1
    merged = [row for row in l1 if "single-modulus" in row["suites"]]
    assert len(merged) == 12
    assert all({"single-modulus", "rns-prefix"} <= set(row["suites"]) for row in merged)
    assert all(row["rns_memberships"] for row in l1)


def test_composed_rows_are_plans_and_precompile_expands_cyclic_subplans():
    documents = _documents()
    composed = documents["composed_workloads.json"]["workloads"]
    required = {"contract_id", "mode", "logN", "batch", "word_bits", "moduli", "direction"}
    assert all(required <= set(row) for row in composed)
    assert all(row["id"] == row["contract_id"] for row in composed)
    assert all(row["status"] == "planned" and row["expected"] == "planned-composition-benchmark"
               for row in composed)
    assert all(row["mode"] in {"negacyclic", "coset"} for row in composed)
    assert all(row.get("coset_generator") == 7 for row in composed if row["mode"] == "coset")
    precompile = documents["precompile_workloads.json"]["workloads"]
    assert precompile and all(int(row["batch"]) == 1 for row in precompile)
    precompile_ids = {matrix.cell_key(row) for row in precompile}
    assert len(precompile_ids) == len(precompile)
    for entry in composed:
        for modulus in entry["moduli"]:
            spec = next(spec for spec in matrix.modulus_specs().values()
                        if str(spec["modulus"]) == modulus)
            subplan = matrix._ntt_workload(spec, entry["logN"], 1, entry["direction"])
            assert matrix.cell_key(subplan) in precompile_ids


def test_unsupported_categories_are_explicit_and_not_timed_rows():
    documents = _documents()
    entries = documents["unsupported.json"]["entries"]
    categories = {entry["category"] for entry in entries}
    assert {"goldilocks-full-word", "bn254-scalar-field", "bls12-381-scalar-field",
            "kyber-incomplete-ntt", "extension-field"} <= categories
    assert all(entry["status"] == "unsupported" for entry in entries)
    direct_moduli = {row["modulus"] for row in documents["workloads.json"]["workloads"]}
    assert "18446744069414584321" not in direct_moduli
    assert "3329" not in direct_moduli


def test_frozen_outputs_refuse_overwrite_and_original_is_unchanged(tmpdir):
    tmp_path = Path(str(tmpdir))
    original_before = ORIGINAL.read_bytes()
    first = matrix.generate(ORIGINAL, tmp_path)
    snapshots = {name: (tmp_path / name).read_bytes() for name in first}
    second = matrix.generate(ORIGINAL, tmp_path)
    assert {name: (tmp_path / name).read_bytes() for name in second} == snapshots
    assert ORIGINAL.read_bytes() == original_before

    path = tmp_path / "composed_workloads.json"
    path.write_bytes(path.read_bytes().replace(b"planned-composition-benchmark",
                                                b"mutated-composition-benchmark", 1))
    with pytest.raises(ValueError, match="refusing to overwrite frozen output"):
        matrix.generate(ORIGINAL, tmp_path)
