import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "operator_coverage_gap_matrix",
    ROOT / "scripts" / "build_operator_coverage_gap_matrix.py",
)
matrix = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(matrix)


def _manifest():
    return matrix.build_manifest()


def _rows(manifest, section):
    return [row for row in manifest["rows"] if row["section"] == section]


def test_section_counts_and_cpu_boundary_are_explicit():
    manifest = _manifest()

    assert manifest["schema"] == "operator-coverage-gap-v1"
    assert manifest["contract"] == {
        "path": "docs/operator_coverage_contract_20261006.md",
        "section": "7",
    }
    assert manifest["counts"]["rows"] == 113
    assert manifest["counts"]["by_section"] == matrix.EXPECTED_SECTION_COUNTS
    assert manifest["execution"] == {
        "mode": "cpu-dry-run",
        "gpu_execution": False,
        "benchmark_invocations": 0,
        "library_loads": 0,
        "device_queries": 0,
    }
    assert len({row["id"] for row in manifest["rows"]}) == 113
    assert all(row["run_mode"] == "cpu-dry-run" for row in manifest["rows"])


def test_low_precision_public_axes_are_complete():
    rows = _rows(_manifest(), "low_precision_public_api")
    assert len(rows) == 48
    keys = {
        (row["operator"], row["precision"], row["accumulation"],
         tuple(row["shape"]), row["batch"], row["direction"])
        for row in rows
    }
    expected = {
        (operator, precision, accumulation, shape, batch, direction)
        for operator in ("fft", "fwht", "structured-2x2")
        for precision in ("fp16", "bf16")
        for accumulation in ("native", "fp32")
        for shape, batch in (((8,), 3), ((4, 8), 2))
        for direction in ("forward", "inverse")
    }
    assert keys == expected
    assert all(row["expected_execution"] == "candidate" for row in rows)
    assert all(row["storage"].startswith("complex_") == (row["operator"] == "fft")
               for row in rows)
    assert {row.get("stage_matrix") for row in rows
            if row["operator"] == "structured-2x2"} == {"1,0.25,-0.5,1"}


def test_real_public_axes_keep_compact_shapes_and_direction_pairing():
    rows = _rows(_manifest(), "real_public_api")
    assert len(rows) == 16
    assert {row["precision"] for row in rows} == {"fp32", "fp64"}
    assert {tuple(row["shape"]) for row in rows} == {(5,), (6,), (4, 6), (2, 3, 5)}
    assert {row["fft_type"] for row in rows} == {"r2c", "c2r"}
    assert all(row["batch"] == 2 and row["placement"] == "out-of-place"
               and row["layout"] == "packed" for row in rows)
    assert all(row["expected_execution"] == "candidate" for row in rows)
    for row in rows:
        shape = tuple(row["shape"])
        half = shape[:-1] + (shape[-1] // 2 + 1,)
        if row["fft_type"] == "r2c":
            assert tuple(row["input_shape"]) == shape
            assert tuple(row["output_shape"]) == half
            assert row["direction"] == "forward"
        else:
            assert tuple(row["input_shape"]) == half
            assert tuple(row["output_shape"]) == shape
            assert row["direction"] == "inverse"


def test_low_precision_real_boundary_is_not_counted_as_a_candidate():
    rows = _rows(_manifest(), "unsupported_real_low_precision")
    assert len(rows) == 4
    assert {(row["precision"], row["fft_type"]) for row in rows} == {
        (precision, fft_type)
        for precision in ("fp16", "bf16")
        for fft_type in ("r2c", "c2r")
    }
    assert all(row["coverage_state"] == "unsupported"
               and row["expected_execution"] == "unsupported"
               for row in rows)
    assert all("FP16/BF16" in row["source_contract"] for row in rows)


def test_ntt_modulus_anchors_and_single_bit_reversed_row():
    rows = _rows(_manifest(), "ntt_semantic_anchors")
    assert len(rows) == 21
    for word_bits, moduli in matrix.NTT_MODULI.items():
        for log_n in (12, 16):
            for modulus in moduli:
                for direction in ("forward", "inverse"):
                    assert sum(
                        row["word_bits"] == word_bits
                        and row["logN"] == log_n
                        and row["modulus"] == modulus
                        and row["direction"] == direction
                        and row["output_order"] == "natural"
                        for row in rows
                    ) == 1
    bit_reversed = [row for row in rows if row["output_order"] == "bit-reversed"]
    assert len(bit_reversed) == 1
    assert bit_reversed[0]["word_bits"] == 32
    assert bit_reversed[0]["logN"] == 12
    assert bit_reversed[0]["input_order"] == "natural"
    assert all(row["placement"] == "out-of-place" for row in rows)


def test_rank_layout_rows_encode_stride_and_in_place_variants():
    rows = _rows(_manifest(), "rank_layout")
    assert len(rows) == 12
    assert {row["operator"] for row in rows} == {"fft", "fwht", "structured-2x2"}
    assert {tuple(row["shape"]) for row in rows} == {(4, 8), (4, 4, 4)}
    assert {row["layout"] for row in rows} == {
        "positive-strided-oop", "packed-in-place"
    }
    for row in rows:
        if row["layout"] == "positive-strided-oop":
            assert row["placement"] == "out-of-place"
            assert any(stride > 1 for stride in row["input_strides"])
            assert row["input_batch_stride"] > matrix._product(row["shape"])
        else:
            assert row["placement"] == "in-place"
            assert row["input_strides"] == list(matrix._packed_strides(tuple(row["shape"])))
            assert row["input_batch_stride"] == matrix._product(row["shape"])


def test_zeta_and_ntt_layout_boundaries_are_packed_only():
    rows = _rows(_manifest(), "layout_boundaries")
    assert len(rows) == 3
    assert {row["operator"] for row in rows} == {
        "subset-zeta", "superset-zeta", "ntt"
    }
    assert all(row["coverage_state"] == "boundary"
               and row["expected_execution"] == "packed-only"
               and row["layout"] == "packed"
               for row in rows)


def test_large_n_batch_anchors_have_requested_axes():
    rows = _rows(_manifest(), "large_n_batch")
    assert len(rows) == 9
    assert {
        (row["operator"], row["precision"], row["logN"])
        for row in rows
    } == {
        ("fft", "fp32", 23),
        ("fft", "fp64", 20),
        ("ntt", "word64", 20),
    }
    assert {row["batch"] for row in rows} == {1, 16, 256}
    assert all(row["direction"] == "forward" and row["placement"] == "out-of-place"
               for row in rows)


def test_cli_writes_reproducible_cpu_manifest(tmpdir):
    output = Path(str(tmpdir)) / "coverage-gap"
    assert matrix.main(["--output", str(output), "--dry-run"]) == 0
    first = (output / "manifest.json").read_bytes()
    assert matrix.main(["--output", str(output), "--dry-run"]) == 0
    second = (output / "manifest.json").read_bytes()
    assert first == second
    manifest = json.loads(first)
    assert manifest["counts"]["rows"] == 113


def test_cli_requires_explicit_dry_run_marker(tmpdir):
    with pytest.raises(SystemExit):
        matrix.main(["--output", str(Path(str(tmpdir)) / "coverage-gap")])
