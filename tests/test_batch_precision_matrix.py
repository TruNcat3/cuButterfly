import json
from collections import defaultdict
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/comprehensive_20260919/workloads.json"

_SPEC = importlib.util.spec_from_file_location(
    "batch_precision_matrix", ROOT / "scripts/build_batch_precision_matrix.py")
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)
cell_key = matrix.cell_key


def _suite_rows():
    return matrix.suite_workloads()


def _batch_values(rows):
    return sorted({int(row["batch"]) for row in rows})


def test_explicit_suite_sizes_and_byte_caps():
    suites = _suite_rows()
    assert len(suites["batch_scaling"]) == 91
    assert len(suites["precision_scaling"]) == 78
    assert len(suites["precision_anchors"]) == 120
    assert len(suites["inverse_anchors"]) == 24
    assert len(suites["ntt_inverse"]) == 6
    assert len(suites["ntt_same_modulus"]) == 9

    for rows in suites.values():
        assert all(matrix.input_bytes(row) <= matrix.MAX_INPUT_BYTES for row in rows)

    batch_rows = suites["batch_scaling"]
    assert max(matrix.input_bytes(row) for row in batch_rows) == matrix.MAX_INPUT_BYTES
    fp64_rows = [row for row in suites["precision_scaling"] if row["precision"] == "fp64"]
    word32_rows = [row for row in suites["precision_scaling"] if row["precision"] == "word32"]
    assert max(matrix.input_bytes(row) for row in fp64_rows) == matrix.MAX_INPUT_BYTES
    assert max(matrix.input_bytes(row) for row in word32_rows) == matrix.MAX_INPUT_BYTES


def test_batch_axes_are_contiguous_powers_of_two():
    for suite_name in ("batch_scaling", "precision_scaling"):
        grouped = defaultdict(list)
        for row in _suite_rows()[suite_name]:
            grouped[(row["operator"], row["precision"], int(row["logN"]))].append(
                int(row["batch"]))
        for values in grouped.values():
            values.sort()
            assert values == [1 << index for index in range(len(values))]


def test_population_reuses_only_complete_semantic_keys_and_keeps_original():
    before = json.loads(ORIGINAL.read_text())
    documents = matrix.build_documents(ORIGINAL)
    after = json.loads(ORIGINAL.read_text())
    assert after == before

    population = documents["population.json"]
    original_cells = population["cells"][: len(before["workloads"])]
    assert [cell["workload"] for cell in original_cells] == before["workloads"]
    assert all(cell["source"] == "original" for cell in original_cells)
    assert [cell["id"] for cell in original_cells] == [cell_key(row) for row in before["workloads"]]

    by_id = {cell["id"]: cell for cell in population["cells"]}
    suites = _suite_rows()
    for suite_name, rows in suites.items():
        for row in rows:
            cell = by_id[cell_key(row)]
            assert suite_name in cell["suites"]
            assert cell["source"] in {"original", "extension"}

    # Direction, accumulation, and NTT modulus are all part of the exact key.
    forward = next(row for row in suites["precision_anchors"]
                   if row["operator"] == "fft" and row["precision"] == "fp32"
                   and row["logN"] == 12 and row["batch"] == 1)
    inverse = next(row for row in suites["inverse_anchors"]
                   if row["operator"] == "fft" and row["precision"] == "fp32"
                   and row["batch"] == 1)
    assert cell_key(forward) != cell_key(inverse)
    native = next(row for row in suites["precision_anchors"]
                  if row["operator"] == "fft" and row["precision"] == "fp16"
                  and row["accumulation"] == "native" and row["logN"] == 8
                  and row["batch"] == 1)
    fp32_accum = dict(native, accumulation="fp32")
    assert cell_key(native) != cell_key(fp32_accum)
    ntt_a = matrix._ntt_workload("word64", 12, 1, "576460756061519873")
    ntt_b = matrix._ntt_workload("word64", 12, 1, "2013265921")
    assert cell_key(ntt_a) != cell_key(ntt_b)


def test_runnable_queue_is_extension_only_and_round_robins_precision():
    documents = matrix.build_documents(ORIGINAL)
    rows = documents["workloads.json"]["workloads"]
    by_id = {cell["id"]: cell for cell in documents["population.json"]["cells"]}
    assert rows
    assert all(by_id[cell_key(row)]["source"] == "extension" for row in rows)
    assert all("source" not in row and "suites" not in row and "input_bytes" not in row
               for row in rows)

    previous_batch = 0
    for row in rows:
        assert int(row["batch"]) >= previous_batch
        previous_batch = int(row["batch"])
    for batch in sorted({int(row["batch"]) for row in rows}):
        batch_rows = [row for row in rows if int(row["batch"]) == batch]
        precisions = [row["precision"] for row in batch_rows]
        # While multiple precision queues have work left, adjacent rounds do
        # not consume the same precision before covering its peers.
        remaining = defaultdict(int)
        for precision in precisions:
            remaining[precision] += 1
        seen_round = set()
        for precision in precisions:
            if precision in seen_round and len(seen_round) < sum(value > 0 for value in remaining.values()):
                pytest.fail(f"precision {precision} was repeated before peer coverage at batch {batch}")
            seen_round.add(precision)
            remaining[precision] -= 1
            if all(value == 0 for value in remaining.values()):
                seen_round.clear()


def test_precompile_is_batch_one_and_structurally_deduplicated():
    documents = matrix.build_documents(ORIGINAL)
    rows = documents["precompile_workloads.json"]["workloads"]
    assert rows and all(int(row["batch"]) == 1 for row in rows)
    keys = []
    for row in rows:
        keys.append((row["operator"], row["precision"], int(row["logN"]),
                     row.get("accumulation", "native"),
                     row.get("output_order") if row["operator"] == "ntt" else None))
    assert len(keys) == len(set(keys))
    assert all("suites" not in row and "source" not in row for row in rows)


def test_frozen_outputs_are_deterministic_and_refuse_changed_content(tmpdir):
    output = Path(str(tmpdir))
    first = matrix.generate(ORIGINAL, output)
    snapshots = {name: (output / name).read_bytes() for name in first}
    second = matrix.generate(ORIGINAL, output)
    assert {name: (output / name).read_bytes() for name in second} == snapshots

    path = output / "population.json"
    path.write_bytes(path.read_bytes().replace(b"batch-precision-study-v1", b"changed-study-v1", 1))
    with pytest.raises(ValueError, match="refusing to overwrite frozen output"):
        matrix.generate(ORIGINAL, output)


def test_unsupported_does_not_hide_audited_low_precision_contracts():
    entries = matrix.unsupported()
    assert not any(item["operator"] == "fft" and item["precision"] in {"fp16", "bf16"}
                   for item in entries)
    assert any(item["operator"] == "ntt" and item["precision"] == "rns" for item in entries)
    assert any(item["operator"] == "subset-zeta" and item["precision"] == "uint64"
               for item in entries)
