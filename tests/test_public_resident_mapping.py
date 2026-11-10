"""Public resident-mapping correctness and validation checks.

The executable checks are intentionally opt-in.  Collection and the mapping
helpers remain useful on a CPU-only host, while GPU execution is controlled by
``CUBUTTERFLY_TEST_BINARY`` and ``CUNTT_TEST_BINARY``.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE_MATRIX = "0.9238795325,-0.3826834324,0.3826834324,0.9238795325"
OPERATORS = ("fft", "fwht", "subset-zeta", "superset-zeta", "structured-2x2", "xor-zeta")


class PublicCase:
    def __init__(
        self,
        name: str,
        kind: str,
        args: Sequence[str],
        mapping: Mapping[str, Any],
        compile_mode: str = "research",
    ) -> None:
        self.name = name
        self.kind = kind
        self.args = tuple(args)
        self.mapping = mapping
        self.compile_mode = compile_mode

    def __repr__(self) -> str:
        return f"PublicCase({self.name!r}, {self.kind!r})"


def resident_mapping(
    kind: str,
    stage_partition: Sequence[int],
    local_stage_partitions: Sequence[Sequence[int]] | None = None,
    exchange_chunks: Sequence[int] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a replayable public mapping descriptor.

    Leaving the two new axes as ``None`` models an old descriptor.  Passing
    empty arrays models an explicit new descriptor with the old whole-tile
    behavior.  Keeping this distinction makes the old/new replay test usable
    by plan-probe and precompilation callers without importing C++ internals.
    """

    mapping: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "stage_partition": list(stage_partition),
    }
    mapping.update(fields)
    if local_stage_partitions is not None:
        mapping["local_stage_partitions"] = [list(part) for part in local_stage_partitions]
    if exchange_chunks is not None:
        mapping["exchange_chunks"] = list(exchange_chunks)
    return mapping


def mapping_json(mapping: Mapping[str, Any]) -> str:
    """Serialize a mapping exactly as a command-line helper should pass it."""

    return json.dumps(dict(mapping), sort_keys=True, separators=(",", ":"))


def _batch_stride(log_n: int, element_stride: int = 2, padding: int = 16) -> int:
    return ((1 << log_n) - 1) * element_stride + 1 + padding


def _shared_butterfly_mapping(
    stages: Sequence[int], locals_: Sequence[Sequence[int]], *, explicit_axes: bool = True
) -> dict[str, Any]:
    return resident_mapping(
        "butterfly",
        stages,
        locals_ if explicit_axes else None,
        [0] * len(locals_) if explicit_axes else None,
        backend="shared-iterative",
        compute_unit="radix2",
        fft_core="scalar",
        local_exchange="shared",
        shared_layout="linear",
        tile_threads=128,
    )


def _butterfly_args(
    operator: str,
    log_n: int,
    mapping: Mapping[str, Any],
    *,
    inverse: bool,
    precision: str | None = None,
    accumulation: str | None = None,
    auto_select: bool = False,
) -> tuple[str, ...]:
    args = [
        "--operator",
        operator,
        "--backend",
        "shared-iterative",
        "--logN",
        str(log_n),
        "--batch",
        "3",
        "--element-stride",
        "2",
        "--batch-stride",
        str(_batch_stride(log_n)),
        "--mapping-json",
        mapping_json(mapping),
    ]
    if precision:
        args.extend(("--precision", precision))
    if accumulation:
        args.extend(("--accumulation", accumulation))
    if operator == "structured-2x2":
        args.extend(("--stage-matrix", STAGE_MATRIX))
    if inverse:
        args.extend(("--inverse", "--placement", "in-place", "--normalization", "inverse"))
    else:
        args.extend(("--placement", "out-of-place", "--normalization", "none"))
    if auto_select:
        args.append("--auto-select")
    args.extend(("--warmup", "0", "--repeat", "1", "--verify", "--verify-batches", "0", "--csv"))
    return tuple(args)


def _shared_cases() -> list[PublicCase]:
    cases: list[PublicCase] = []
    # The requested [6,4]/[[2,4],[1,3]] shape exercises two resident units
    # whose local widths differ.  The logN=12 shape keeps the same first
    # group and covers the second group with the legal [1,5] decomposition.
    shapes = {
        False: (10, (6, 4), ((2, 4), (1, 3))),
        True: (12, (6, 6), ((2, 4), (1, 5))),
    }
    for inverse, (log_n, stages, locals_) in shapes.items():
        for operator in OPERATORS:
            precision = "fp32" if operator in {"fft", "fwht", "structured-2x2"} else "uint32"
            mapping = _shared_butterfly_mapping(stages, locals_)
            name = f"{operator}-n{log_n}-{'inverse' if inverse else 'forward'}"
            if not inverse and operator == "fwht":
                # Mapping application deliberately wins over auto selection,
                # proving a non-default explicit resident shape is accepted.
                name += "-explicit-auto"
            cases.append(
                PublicCase(
                    name=name,
                    kind="butterfly",
                    args=_butterfly_args(
                        operator,
                        log_n,
                        mapping,
                        inverse=inverse,
                        precision=precision,
                        auto_select=not inverse and operator == "fwht",
                    ),
                    mapping=mapping,
                    compile_mode="auto" if not inverse and operator == "fwht" else "research",
                )
            )
    for suffix, accumulation in (("accfp32", "fp32"), ("native", "native")):
        low_mapping = _shared_butterfly_mapping((6, 4), ((2, 4), (1, 3)))
        cases.append(
            PublicCase(
                name=f"fwht-n10-forward-fp16-{suffix}",
                kind="butterfly",
                args=_butterfly_args(
                    "fwht",
                    10,
                    low_mapping,
                    inverse=False,
                    precision="fp16",
                    accumulation=accumulation,
                ),
                mapping=low_mapping,
            )
        )
    return cases


def _ntt_mapping(stages: Sequence[int], locals_: Sequence[Sequence[int]]) -> dict[str, Any]:
    return resident_mapping(
        "ntt",
        stages,
        locals_,
        [0] * len(stages),
        backend="shared-iterative",
        compute_unit="radix2",
        threads_per_block=128,
        dataflow_layout="hermes-xor",
    )


def _ntt_cases() -> list[PublicCase]:
    specs = (
        ("n10-w32-natural-forward", 10, 32, "natural", False, (5, 5), ((2, 3), (2, 3))),
        ("n10-w32-bitrev-inverse", 10, 32, "bit-reversed", True, (5, 5), ((2, 3), (2, 3))),
        ("n12-w64-natural-forward", 12, 64, "natural", False, (6, 6), ((3, 3), (3, 3))),
        ("n12-w64-bitrev-inverse", 12, 64, "bit-reversed", True, (6, 6), ((3, 3), (3, 3))),
    )
    cases: list[PublicCase] = []
    for name, log_n, word_bits, output_order, inverse, stages, locals_ in specs:
        mapping = _ntt_mapping(stages, locals_)
        args = [
            "--backend",
            "shared-iterative",
            "--logN",
            str(log_n),
            "--batch",
            "3",
            "--word-bits",
            str(word_bits),
            "--modulus",
            "998244353" if word_bits == 32 else "1152921504606584833",
            "--input-order",
            "natural",
            "--output-order",
            output_order,
            "--mapping-json",
            mapping_json(mapping),
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--verify",
            "--verify-batches",
            "0",
            "--csv",
        ]
        if inverse:
            args.append("--inverse")
        cases.append(PublicCase(name=name, kind="ntt", args=tuple(args), mapping=mapping))
    return cases


def _register_mapping(
    *,
    factors: Sequence[int],
    lanes: int,
    precision: str,
    codelet: str = "native",
    prefix_layout: str = "linear",
    suffix_threads: int = 64,
    suffix_ept: int = 8,
    chunk: int = 4,
    square: bool = False,
) -> dict[str, Any]:
    local = sum(factors)
    suffix = 12 - local
    prefix_ept = (1 << factors[0]) // lanes
    prefix_threads = 32
    stages = (local, suffix)
    return resident_mapping(
        "butterfly",
        stages,
        [list(factors), []],
        [0, chunk],
        backend="online-reorder",
        fft_core="register-tile",
        compute_unit="radix2",
        local_exchange="shared",
        shared_layout="writer-aligned",
        cross_twiddle="recurrence",
        direct_boundary="direct-strided",
        local_stages=local,
        prefix_threads=prefix_threads,
        prefix_ept=prefix_ept,
        prefix_codelet_lanes=lanes,
        prefix_codelet=codelet,
        prefix_shared_layout=prefix_layout,
        suffix_threads=suffix_threads,
        suffix_ept=suffix_ept,
        reorder_columns=1,
        boundaries=[{"twiddle": "recurrence", "layout": "direct-strided", "residency": "global-scratch"}],
    )


def _register_args(
    mapping: Mapping[str, Any], *, precision: str, inverse: bool = False, pipeline: bool = False
) -> tuple[str, ...]:
    stages = mapping["stage_partition"]
    args = [
        "--operator",
        "fft",
        "--backend",
        "online-reorder",
        "--fft-core",
        "register-tile",
        "--logN",
        "12",
        "--batch",
        "3",
        "--element-stride",
        "2",
        "--batch-stride",
        str(_batch_stride(12)),
        "--precision",
        precision,
        "--placement",
        "in-place" if inverse else "out-of-place",
        "--normalization",
        "inverse" if inverse else "none",
        "--stage-partition",
        ",".join(str(value) for value in stages),
        "--mapping-json",
        mapping_json(mapping),
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--verify",
        "--verify-batches",
        "0",
        "--csv",
    ]
    if inverse:
        args.append("--inverse")
    if pipeline:
        args.extend(("--stage-overlap", "--batch-tile-count", "2"))
    return tuple(args)


def _register_cases() -> list[PublicCase]:
    specs = (
        ("rect32-native-forward", (3, 2), 2, "fp32", "native", "linear", False, False),
        ("rect32-native-inverse-pipeline", (3, 2), 2, "fp32", "native", "linear", True, True),
        ("rect32-native-fp64", (3, 2), 2, "fp64", "native", "linear", False, False),
        ("rect32-native-fp64-inverse-xor", (3, 2), 2, "fp64", "native", "xor", True, False),
        ("rect32-cufftdx-thread", (3, 2), 2, "fp32", "cufftdx-thread", "xor", False, False),
        ("rect-reverse-native", (2, 3), 1, "fp32", "native", "linear", False, False),
        ("squareprefix-chunk4", (3, 3), 1, "fp32", "native", "linear", False, True),
    )
    cases: list[PublicCase] = []
    for name, factors, lanes, precision, codelet, layout, inverse, pipeline in specs:
        suffix_threads = 32 if factors == (3, 3) else 64
        mapping = _register_mapping(
            factors=factors,
            lanes=lanes,
            precision=precision,
            codelet=codelet,
            prefix_layout=layout,
            suffix_threads=suffix_threads,
            suffix_ept=8,
            chunk=4,
            square=factors == (3, 3),
        )
        if pipeline:
            mapping = dict(mapping, stage_overlap=True, batch_tile_count=2)
        cases.append(
            PublicCase(
                name=name,
                kind="butterfly",
                args=_register_args(mapping, precision=precision, inverse=inverse, pipeline=pipeline),
                mapping=mapping,
            )
        )
    return cases


SHARED_CASES = tuple(_shared_cases())
NTT_CASES = tuple(_ntt_cases())
REGISTER_CASES = tuple(_register_cases())
_fused_shared_mapping = dict(
    _shared_butterfly_mapping((3, 3, 4), ((2, 4), (1, 3))),
    boundaries=[
        dict(twiddle="table", layout="direct-strided", residency="fused"),
        dict(twiddle="table", layout="direct-strided", residency="global-scratch"),
    ],
)
FUSED_CASE = PublicCase("fwht-physical-group-indexing", "butterfly",
    _butterfly_args("fwht", 10, _fused_shared_mapping, inverse=False, precision="fp32"),
    _fused_shared_mapping)
GPU_CASES = SHARED_CASES + NTT_CASES + REGISTER_CASES + (FUSED_CASE,)


def command_for_case(case: PublicCase, binary: str | os.PathLike[str] | None = None) -> list[str]:
    """Return a complete benchmark command for reuse by CPU/JIT drivers."""

    if binary is None:
        binary = os.environ.get("CUNTT_TEST_BINARY" if case.kind == "ntt" else "CUBUTTERFLY_TEST_BINARY", "")
    return [str(binary), *case.args]


def _binary(kind: str) -> str:
    names = ("CUNTT_TEST_BINARY", "CUBUTTERFLY_NTT_TEST_BINARY") if kind == "ntt" else ("CUBUTTERFLY_TEST_BINARY",)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    pytest.skip(f"set {' or '.join(names)} to run {kind} GPU checks")


def _csv_row(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if line.startswith("device,"))
    except StopIteration as error:
        raise AssertionError(f"benchmark emitted no CSV header:\n{stdout[-2000:]}") from error
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    if len(rows) != 1:
        raise AssertionError(f"expected one CSV result, got {len(rows)}:\n{stdout[-2000:]}")
    return rows[0]


def _run(case: PublicCase, *, expect_success: bool = True, compile_mode: str | None = None) -> dict[str, str] | None:
    binary = _binary(case.kind)
    environment = os.environ.copy()
    environment["CUBUTTERFLY_COMPILE_MODE"] = compile_mode or case.compile_mode
    completed = subprocess.run(
        command_for_case(case, binary),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if expect_success:
        if completed.returncode != 0:
            pytest.fail(f"{case.name} failed ({completed.returncode}):\n{combined[-5000:]}")
        return _csv_row(completed.stdout)
    if completed.returncode == 0:
        pytest.fail(f"{case.name} unexpectedly succeeded:\n{combined[-3000:]}")
    return None


def _assert_success_row(case: PublicCase, row: Mapping[str, str]) -> None:
    assert row.get("correct") == "1", (case.name, row)
    assert int(row.get("verified_batches", "0")) == 3, (case.name, row)
    assert row.get("batch") == "3", (case.name, row)
    if case.kind == "butterfly":
        inverse = "--inverse" in case.args
        operator = case.args[case.args.index("--operator") + 1]
        assert row.get("element_stride") == "2", (case.name, row)
        assert int(row.get("batch_stride", "0")) == _batch_stride(int(row["logN"])), (case.name, row)
        assert row.get("direction") == ("inverse" if inverse else "forward"), (case.name, row)
        assert row.get("placement") == ("in-place" if inverse else "out-of-place"), (case.name, row)
        expected_normalization = "inverse" if inverse and operator in {"fft", "fwht"} else "none"
        assert row.get("normalization") == expected_normalization, (case.name, row)
    resolved = json.loads(row["mapping_json"])
    assert resolved["stage_partition"] == case.mapping["stage_partition"]
    if "local_stage_partitions" in case.mapping:
        assert resolved["local_stage_partitions"] == case.mapping["local_stage_partitions"]
    if "exchange_chunks" in case.mapping:
        assert resolved["exchange_chunks"] == case.mapping["exchange_chunks"]


@pytest.mark.parametrize("case", GPU_CASES, ids=lambda case: case.name)
def test_public_resident_mapping_correctness(case: PublicCase) -> None:
    row = _run(case)
    assert row is not None
    _assert_success_row(case, row)


def _invalid_cases() -> tuple[PublicCase, ...]:
    base = _shared_butterfly_mapping((6, 4), ((2, 4), (1, 3)))
    bad_count = dict(base, local_stage_partitions=[[2, 4]], exchange_chunks=[0, 0])
    bad_sum = dict(base, local_stage_partitions=[[2, 5], [1, 3]], exchange_chunks=[0, 0])
    bad_chunk = dict(base, exchange_chunks=[0, 4])
    return (
        PublicCase(
            "reject-resident-group-count",
            "butterfly",
            _butterfly_args("fft", 10, bad_count, inverse=False, precision="fp32"),
            bad_count,
        ),
        PublicCase(
            "reject-resident-partition-sum",
            "butterfly",
            _butterfly_args("fwht", 10, bad_sum, inverse=False, precision="fp32"),
            bad_sum,
        ),
        PublicCase(
            "reject-shared-exchange-chunk",
            "butterfly",
            _butterfly_args("fft", 10, bad_chunk, inverse=False, precision="fp32"),
            bad_chunk,
        ),
        PublicCase(
            "reject-precompiled-resident-jit",
            "butterfly",
            _butterfly_args("fft", 10, base, inverse=False, precision="fp32"),
            base,
            compile_mode="precompiled",
        ),
    )


INVALID_CASES = _invalid_cases()


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda case: case.name)
def test_invalid_public_resident_mapping_is_rejected(case: PublicCase) -> None:
    _run(case, expect_success=False, compile_mode=case.compile_mode)


def test_old_and_explicit_zero_resident_descriptors_have_same_axes() -> None:
    old = _shared_butterfly_mapping((6, 4), ((2, 4), (1, 3)), explicit_axes=False)
    explicit = _shared_butterfly_mapping((6, 4), ((), ()), explicit_axes=True)

    def canonical_axes(mapping: Mapping[str, Any]) -> tuple[Any, Any]:
        return (mapping.get("local_stage_partitions", []), mapping.get("exchange_chunks", []))

    assert canonical_axes(old) == ([], [])
    assert canonical_axes(explicit) == ([[], []], [0, 0])
    assert mapping_json(old) != mapping_json(explicit)
    # The plan serializer intentionally normalizes both descriptors to the
    # same semantic legacy behavior, despite preserving their input spelling.
    assert all(not part for part in canonical_axes(explicit)[0])
    assert all(chunk == 0 for chunk in canonical_axes(explicit)[1])


def test_public_case_population_is_bounded_and_covers_requested_axes() -> None:
    assert 20 <= len(GPU_CASES) <= 30
    assert {case.kind for case in GPU_CASES} == {"butterfly", "ntt"}
    assert {case.name.split("-n", 1)[1].split("-", 1)[0] for case in SHARED_CASES} == {"10", "12"}
    assert any("fp64" in case.name for case in REGISTER_CASES)
    assert any("cufftdx-thread" in case.name for case in REGISTER_CASES)
    assert any("pipeline" in case.name for case in REGISTER_CASES)
    assert any("bitrev" in case.name for case in NTT_CASES)
