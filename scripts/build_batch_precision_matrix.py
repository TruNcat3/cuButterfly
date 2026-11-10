#!/usr/bin/env python3
"""Build the finite batch/precision extension workload population.

The population is a CPU-only manifest.  It does not inspect a device, invoke
the benchmark binaries, or claim that a lowering is available.  Existing
semantic cells are retained as ``source=original`` cells and are never added
to the runnable extension queue a second time.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
import sys


# ``run_research_acceptance`` is importable when this file is run directly,
# while the fallback keeps the module importable as ``scripts.<name>`` in tests.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_research_acceptance import cell_key  # noqa: E402


SCHEMA = "batch-precision-study-v1"
WORKLOAD_SCHEMA = "cubutterfly-install-search-v1"
MAX_INPUT_BYTES = 2 * 1024**3
BATCH_ELEMENT_LIMIT = 1 << 28
DEFAULT_STAGE_MATRIX = "1,0.25,-0.5,1"

SUITE_ORDER = (
    "batch_scaling",
    "precision_scaling",
    "precision_anchors",
    "inverse_anchors",
    "ntt_inverse",
    "ntt_same_modulus",
)

# This order is part of the queue contract.  The queue interleaves these
# precision groups at every batch level before advancing to a larger batch.
PRECISION_ORDER = ("fp32", "word64", "fp64", "word32", "fp16", "bf16")

FLOAT_CONTRACTS = (
    ("fp16", "native"),
    ("fp16", "fp32"),
    ("bf16", "native"),
    ("bf16", "fp32"),
    ("fp32", "native"),
    ("fp64", "native"),
)

SCALAR_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "word32": 4,
    "word64": 8,
    "uint32": 4,
    "uint64": 8,
}

LIMITATIONS = (
    "This is a finite extension queue, not an exhaustive design-space or migration certificate.",
    "External baseline availability is recorded by the acceptance runner; missing adapters remain limitations.",
    "Structured-2x2 inverse anchors remain deferred pending the final support check.",
    "Arbitrary-length transforms and other operator-specific extensions are outside this queue.",
)

NOTES = (
    "NTT cells use exact modular correctness; floating-point error metrics do not apply.",
    "BF16 native cells intentionally model per-stage low-precision rounding; emulated_native is determined from measured CSV evidence.",
    "The six floating contracts are semantic anchors. The dedicated WMMA fp16/fp32 implementation is not part of this finite queue.",
    "Inverse FFT/FWHT anchors use strict inverse normalization. Structured-2x2 inverse is deferred rather than aliased to forward semantics.",
)


def _read(path: Path):
    return json.loads(Path(path).read_text())


def _scalar_bytes(precision: str) -> int:
    try:
        return SCALAR_BYTES[str(precision)]
    except KeyError as error:
        raise ValueError(f"unknown precision for memory accounting: {precision}") from error


def element_bytes(workload) -> int:
    """Return bytes for one logical element of ``workload``.

    FFT elements are complex, while the other floating and integer operators
    use one scalar per logical element.  Accumulation does not change storage
    bytes for the fp16/bf16 contracts.
    """
    scalar = _scalar_bytes(workload["precision"])
    return scalar * 2 if workload.get("operator") == "fft" else scalar


def input_bytes(workload) -> int:
    return (1 << int(workload["logN"])) * int(workload["batch"]) * element_bytes(workload)


def _float_workload(operator, precision, log_n, batch, *, accumulation="native",
                    direction="forward", normalization="none"):
    workload = {
        "operator": operator,
        "precision": precision,
        "logN": int(log_n),
        "batch": int(batch),
        "placement": "out-of-place",
        "direction": direction,
        "normalization": normalization,
        "accumulation": accumulation,
        "element_stride": 1,
        "batch_stride": 1 << int(log_n),
    }
    if operator == "structured-2x2":
        workload["stage_matrix"] = DEFAULT_STAGE_MATRIX
    return workload


def _ntt_workload(precision, log_n, batch, modulus, *, inverse=False):
    workload = {
        "operator": "ntt",
        "precision": precision,
        "logN": int(log_n),
        "batch": int(batch),
        # NTT placement is represented by the order contract.  In particular,
        # do not add ``placement=out-of-place`` here: canonical_semantics maps
        # NTT placement from output_order and a donor placement would pollute
        # semantic identity.
        "modulus": str(modulus),
        "input_order": "natural",
        "output_order": "natural",
    }
    if inverse:
        workload["direction"] = "inverse"
    return workload


def _powers_while(predicate):
    batch = 1
    while predicate(batch):
        yield batch
        batch <<= 1


def _element_limited_batches(log_n, workload):
    return _powers_while(
        lambda batch: (1 << int(log_n)) * batch * element_bytes(workload) <= MAX_INPUT_BYTES)


def _batch_scaling():
    points = []
    for log_n in (12, 16, 20, 22, 23):
        template = _float_workload("fft", "fp32", log_n, 1)
        for batch in _powers_while((lambda b, n=log_n: (1 << n) * b <= BATCH_ELEMENT_LIMIT)):
            points.append(_float_workload("fft", "fp32", log_n, batch))
    for log_n in (12, 16, 20):
        template = _ntt_workload("word64", log_n, 1, "576460756061519873")
        del template
        for batch in _powers_while((lambda b, n=log_n: (1 << n) * b <= BATCH_ELEMENT_LIMIT)):
            points.append(_ntt_workload("word64", log_n, batch, "576460756061519873"))
    return points


def _precision_scaling():
    points = []
    for log_n in (12, 16, 20):
        template = _float_workload("fft", "fp64", log_n, 1)
        for batch in _element_limited_batches(log_n, template):
            points.append(_float_workload("fft", "fp64", log_n, batch))
    for log_n in (12, 16, 20):
        template = _ntt_workload("word32", log_n, 1, "2013265921")
        for batch in _element_limited_batches(log_n, template):
            points.append(_ntt_workload("word32", log_n, batch, "2013265921"))
    return points


def _precision_anchors():
    points = []
    for operator in ("fft", "fwht", "structured-2x2"):
        logs = (8, 12, 15) if operator == "fwht" else (8, 12)
        for log_n in logs:
            for precision, accumulation in FLOAT_CONTRACTS:
                for batch in (1, 16, 256) if log_n != 15 else (1, 256):
                    points.append(_float_workload(
                        operator, precision, log_n, batch,
                        accumulation=accumulation))
    return points


def _inverse_anchors():
    points = []
    # Structured inverse is intentionally absent until its support contract is
    # independently confirmed. FFT and FWHT use all six audited contracts.
    for operator in ("fft", "fwht"):
        for precision, accumulation in FLOAT_CONTRACTS:
            for batch in (1, 256):
                points.append(_float_workload(
                    operator, precision, 12, batch,
                    accumulation=accumulation, direction="inverse",
                    normalization="inverse"))
    return points


def _ntt_inverse():
    return [
        _ntt_workload(precision, 16, batch, modulus, inverse=True)
        for precision, modulus in (
            ("word32", "2013265921"),
            ("word64", "576460756061519873"),
        )
        for batch in (1, 16, 256)
    ]


def _ntt_same_modulus():
    return [
        _ntt_workload("word64", log_n, batch, "2013265921")
        for log_n in (12, 16, 20)
        for batch in (1, 16, 256)
    ]


def suite_workloads():
    """Return fresh, ordered semantic workloads for every finite suite."""
    result = {
        "batch_scaling": _batch_scaling(),
        "precision_scaling": _precision_scaling(),
        "precision_anchors": _precision_anchors(),
        "inverse_anchors": _inverse_anchors(),
        "ntt_inverse": _ntt_inverse(),
        "ntt_same_modulus": _ntt_same_modulus(),
    }
    for suite, workloads in result.items():
        for workload in workloads:
            if input_bytes(workload) > MAX_INPUT_BYTES:
                raise ValueError(f"{suite} exceeds input byte limit: {workload}")
    return result


def unsupported():
    """Return only intentionally out-of-scope capability categories."""
    result = []
    for operator in ("subset-zeta", "superset-zeta"):
        for precision in ("uint64", "float"):
            result.append({
                "operator": operator,
                "precision": precision,
                "reason": "zeta precision is outside this finite extension queue",
            })
    for precision, reason in (
        ("multi-limb", "arbitrary multi-limb NTT is outside the finite word32/word64 queue"),
        ("rns", "RNS NTT is outside the finite word32/word64 queue"),
        ("negacyclic", "negacyclic NTT is outside the finite cyclic queue"),
    ):
        result.append({"operator": "ntt", "precision": precision, "reason": reason})
    for operator in ("fft", "fwht", "structured-2x2"):
        result.append({
            "operator": operator,
            "precision": "fp16-fp32",
            "reason": "generic fp16/fp32 path is outside this queue; dedicated WMMA is not included",
        })
    return result


def _limitations():
    return list(LIMITATIONS)


def _notes():
    return list(NOTES)


def _limits(suites):
    return {
        "max_input_bytes": MAX_INPUT_BYTES,
        "batch_scaling_max_elements": BATCH_ELEMENT_LIMIT,
        "batch_scaling_points": len(suites["batch_scaling"]),
        "precompile_batch": 1,
        "memory_accounting": "logical contiguous input extent; FFT complex elements use two scalar values",
    }


def _check_original(document):
    if document.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("original workload matrix must use cubutterfly-install-search-v1")
    workloads = document.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("original workload matrix has no workloads")
    keys = [cell_key(workload) for workload in workloads]
    if len(set(keys)) != len(keys):
        raise ValueError("original workload matrix contains duplicate semantic cells")
    return workloads, keys


def _suite_map(suites):
    by_key = defaultdict(list)
    workload_by_key = {}
    for suite in SUITE_ORDER:
        for workload in suites[suite]:
            key = cell_key(workload)
            if key not in workload_by_key:
                workload_by_key[key] = copy.deepcopy(workload)
            if suite not in by_key[key]:
                by_key[key].append(suite)
    return by_key, workload_by_key


def _workload_sort_key(workload):
    return (
        int(workload["logN"]),
        str(workload.get("operator", "")),
        str(workload.get("accumulation", "native")),
        str(workload.get("direction", "forward")),
        str(workload.get("modulus", "")),
        cell_key(workload),
    )


def order_extension_workloads(cells):
    """Interleave precision queues within each increasing batch level."""
    by_batch = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        workload = cell["workload"] if "workload" in cell else cell
        by_batch[int(workload["batch"])][str(workload["precision"])].append(workload)

    ordered = []
    for batch in sorted(by_batch):
        queues = {
            precision: sorted(rows, key=_workload_sort_key)
            for precision, rows in by_batch[batch].items()
        }
        while any(queues.values()):
            for precision in PRECISION_ORDER:
                if queues.get(precision):
                    ordered.append(queues[precision].pop(0))
            # Keep unknown future precisions deterministic without changing the
            # established priority order above.
            for precision in sorted(set(queues) - set(PRECISION_ORDER)):
                if queues[precision]:
                    ordered.append(queues[precision].pop(0))
    return ordered


def _population_cells(original_workloads, suite_by_key, extension_by_key):
    original_keys = {cell_key(workload) for workload in original_workloads}
    cells = []
    for workload in original_workloads:
        key = cell_key(workload)
        cells.append({
            "id": key,
            "workload": copy.deepcopy(workload),
            "suites": list(suite_by_key.get(key, ())),
            "source": "original",
            "input_bytes": input_bytes(workload),
        })

    extension_rows = []
    for key, workload in extension_by_key.items():
        if key in original_keys:
            continue
        extension_rows.append({
            "id": key,
            "workload": copy.deepcopy(workload),
            "suites": list(suite_by_key[key]),
            "source": "extension",
            "input_bytes": input_bytes(workload),
        })
    ordered = order_extension_workloads(extension_rows)
    extension_by_id = {row["id"]: row for row in extension_rows}
    cells.extend(extension_by_id[cell_key(workload)] for workload in ordered)
    return cells, ordered


def _precompile_workloads(extension_workloads):
    """Deduplicate compile preparation by structural semantic dimensions.

    Batch, direction, normalization and NTT modulus do not add another timed
    cell to this file.  NTT output order remains explicit because it selects a
    different lowering; all preparation rows use batch one.
    """
    selected = {}
    for raw in extension_workloads:
        workload = copy.deepcopy(raw)
        workload["batch"] = 1
        if workload.get("operator") != "ntt":
            workload["batch_stride"] = 1 << int(workload["logN"])
        key = (
            workload.get("operator", "fft"),
            workload.get("precision", "fp32"),
            int(workload["logN"]),
            workload.get("accumulation", "native"),
            workload.get("output_order") if workload.get("operator") == "ntt" else None,
        )
        selected.setdefault(key, workload)
    return list(selected.values())


def build_documents(original_path):
    """Build population, runnable extension, and precompile documents."""
    original_path = Path(original_path).expanduser().resolve()
    original = _read(original_path)
    original_workloads, original_keys = _check_original(original)
    suites = suite_workloads()
    suite_by_key, extension_by_key = _suite_map(suites)
    if len(set(original_keys)) != len(original_keys):
        raise ValueError("original semantic keys are not unique")

    population_cells, extension_workloads = _population_cells(
        original_workloads, suite_by_key, extension_by_key)
    if any(cell["input_bytes"] > MAX_INPUT_BYTES for cell in population_cells):
        raise ValueError("population contains a cell over the input byte limit")

    limits = _limits(suites)
    population = {
        "schema": SCHEMA,
        "cells": population_cells,
        "unsupported": unsupported(),
        "limits": limits,
        "notes": _notes(),
    }
    workloads = {
        "schema": WORKLOAD_SCHEMA,
        "scope": "Finite batch/precision extension queue; extension cells only",
        "limitations": _limitations(),
        "workloads": copy.deepcopy(extension_workloads),
    }
    precompile = {
        "schema": WORKLOAD_SCHEMA,
        "scope": "Finite batch/precision extension precompile preparation; batch one only",
        "limitations": _limitations() + [
            "Precompile rows are structural preparation points and do not expand the timed workload matrix.",
        ],
        "workloads": _precompile_workloads(extension_workloads),
    }
    return {
        "population.json": population,
        "workloads.json": workloads,
        "precompile_workloads.json": precompile,
    }


def _render(document) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True,
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_frozen(output_dir, documents):
    """Write missing outputs and reject changed existing frozen outputs."""
    output_dir = Path(output_dir).expanduser().resolve()
    rendered = {name: _render(document) for name, document in documents.items()}
    for name, content in rendered.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite frozen output: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        path = output_dir / name
        if path.exists():
            continue
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)


def generate(original_path, output_dir):
    documents = build_documents(original_path)
    write_frozen(output_dir, documents)
    return documents


# A descriptive alias for callers that prefer an explicit name.
build = generate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True,
                        help="existing cubutterfly-install-search-v1 workload matrix")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for frozen population/workload manifests")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generate(args.original, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
