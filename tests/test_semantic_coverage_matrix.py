import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_coverage", ROOT / "scripts/build_semantic_coverage_matrix.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matrix_is_factorized_and_cpu_only():
    document = MODULE.build()
    assert document["schema"] == "cubutterfly-semantic-coverage-v1"
    assert document["run_mode"] == "cpu-dry-run"
    rows = document["rows"]
    assert rows
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["status"] == "cpu-manifest"
        assert row["placement"] in MODULE.PLACEMENTS
        assert row["direction"] in MODULE.directions()
        assert row["input_order"] == row["output_order"]
        assert len(row["shape"]) == row["rank"]


def test_required_boundary_axes_are_present():
    rows = MODULE.rows()
    assert {row["operator"] for row in rows} == set(MODULE.OPERATORS)
    assert {row["precision"] for row in rows if row["operator"] == "fft"} == set(MODULE.FLOAT_PRECISIONS)
    assert {row["precision"] for row in rows if row["operator"] == "ntt"} == set(MODULE.NTT_PRECISIONS)
    assert {row["placement"] for row in rows} == set(MODULE.PLACEMENTS)
    assert {row["direction"] for row in rows} == {"forward", "inverse"}
    assert {row["input_order"][0] for row in rows} == set(MODULE.ORDERS)
    assert {row["rank"] for row in rows} == set(MODULE.RANKS)
    assert {row["batch"] for row in rows} == set(MODULE.BATCHES)


def test_xd_is_a_mode_of_ranked_requests_not_a_second_semantics():
    rows = MODULE.rows()
    for row in rows:
        if row["rank"] > 1:
            assert row["execution_modes"] == ["composite", "xd"]
        else:
            assert row["execution_modes"] == ["composite"]


def test_invalid_legacy_bit_reversal_is_excluded_without_erasing_ranked_order():
    rows = MODULE.rows()
    ntt = [row for row in rows if row["operator"] == "ntt"]
    assert any(row["rank"] > 1 and row["input_order"] == ["bit-reversed"] * row["rank"]
               for row in ntt)
    assert any(row["rank"] > 1 and row["input_order"] == ["bit-reversed"] * row["rank"]
               for row in rows if row["operator"] == "fwht")


def test_compact_real_axis_is_natural():
    for row in MODULE.rows():
        if row["operator"] == "fft" and row["fft_type"] != "c2c":
            assert row["input_order"][-1] == "natural"
            assert row["output_order"][-1] == "natural"
