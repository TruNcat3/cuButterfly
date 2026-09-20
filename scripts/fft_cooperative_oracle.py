#!/usr/bin/env python3
"""CPU oracle for the cooperative register-prefix FFT.

The CUDA implementation assigns ``G`` lanes to one local FFT and keeps the
remaining ``K=R/G`` points in each lane's registers.  This module mirrors the
mathematical decomposition and the shared-tile address contract without
depending on CUDA, a compiled kernel, or a performance measurement.

Logical local input and output use the flattening ``a * R + b`` with ``a`` as
the first FFT coordinate.  The public transform accepts either that flat
shape or an equivalent ``(R, R, Columns)`` shape and returns the same shape.
Each column is an independent local transform.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Iterable, Sequence

import numpy as np


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _log2(value: int, name: str = "value") -> int:
    if not _is_power_of_two(value):
        raise ValueError(f"{name} must be a positive power of two")
    return value.bit_length() - 1


def _validate_shape(r: int, g: int, columns: int) -> tuple[int, int, int]:
    if isinstance(r, bool) or not isinstance(r, int):
        raise ValueError("R must be an integer")
    if isinstance(g, bool) or not isinstance(g, int):
        raise ValueError("G must be an integer")
    if isinstance(columns, bool) or not isinstance(columns, int):
        raise ValueError("columns must be an integer")
    _log2(r, "R")
    _log2(g, "G")
    if g > r:
        raise ValueError("G must not exceed R")
    if columns <= 0:
        raise ValueError("columns must be positive")
    # This is the same cooperative shuffle envelope as the CUDA code.  The
    # one-lane path has no cross-lane shuffle and is intentionally unrestricted.
    if g > 1 and g * columns > 32:
        raise ValueError("G * columns must be at most one warp (32)")
    return r, g, columns


def _normalise_layout(layout: str) -> str:
    aliases = {"xor-swizzle": "xor", "xor_swizzle": "xor"}
    layout = aliases.get(layout, layout)
    if layout not in {"linear", "xor"}:
        raise ValueError("layout must be 'linear' or 'xor'")
    return layout


def _normalise_complex_bytes(complex_bytes: int) -> int:
    if isinstance(complex_bytes, bool) or not isinstance(complex_bytes, int):
        raise ValueError("complex_bytes must be an integer")
    if complex_bytes <= 0:
        raise ValueError("complex_bytes must be positive")
    return complex_bytes


def _as_flat_columns(values: np.ndarray | Sequence[complex], r: int, columns: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return ``(R*R, Columns)`` data and remember the caller's shape."""

    array = np.asarray(values)
    expected = r * r * columns
    if array.size != expected:
        raise ValueError(
            f"input has {array.size} values, expected {expected} for R={r}, columns={columns}"
        )
    original_shape = tuple(array.shape)
    if array.ndim == 3 and array.shape == (r, r, columns):
        flat = array.reshape(r * r, columns)
    elif array.ndim == 2 and array.shape == (r * r, columns):
        flat = array
    elif columns == 1 and array.ndim == 2 and array.shape == (r, r):
        flat = array.reshape(r * r, 1)
    elif array.ndim == 1:
        flat = array.reshape(r * r, columns)
    else:
        raise ValueError(
            "input shape must be (R*R,), (R*R, columns), or (R, R, columns)"
        )
    # np.fft uses complex128 for ordinary Python/float inputs.  Keeping the
    # oracle in complex128 makes numerical failures reflect the decomposition,
    # not an accidental storage dtype conversion.
    return np.asarray(flat, dtype=np.complex128), original_shape


def _restore_shape(flat: np.ndarray, shape: tuple[int, ...], r: int, columns: int) -> np.ndarray:
    if shape == (r, r, columns):
        return flat.reshape(shape)
    if shape == (r, r) and columns == 1:
        return flat.reshape(shape)
    return flat.reshape(shape)


def _fft_unscaled(values: np.ndarray, axis: int, inverse: bool) -> np.ndarray:
    """FFT with the CUDA convention: inverse is unnormalised."""

    size = values.shape[axis]
    if inverse:
        return np.fft.ifft(values, axis=axis) * size
    return np.fft.fft(values, axis=axis)


def _cooperative_1d(values: np.ndarray, r: int, g: int, inverse: bool) -> np.ndarray:
    """Apply the K-point/lane-G-point decomposition to one length-R vector.

    ``values`` is indexed by the physical lane first and the register slot
    second, with shape ``(G, K)``.  The returned array has the same indexing;
    entry ``[q, j]`` is frequency ``j + K*q``.
    """

    k = r // g
    lane = np.arange(g, dtype=np.int64)[:, None]
    slot = np.arange(k, dtype=np.int64)[None, :]
    sign = 1.0 if inverse else -1.0

    work = _fft_unscaled(values, axis=1, inverse=inverse)
    work = work * np.exp(sign * 2j * np.pi * lane * slot / r)
    return _fft_unscaled(work, axis=0, inverse=inverse)


def cooperative_fft(
    values: np.ndarray | Sequence[complex],
    r: int,
    g: int,
    *,
    inverse: bool = False,
    normalize: bool = False,
    columns: int = 1,
    layout: str = "linear",
) -> np.ndarray:
    """Run the two-dimensional cooperative local FFT on CPU.

    The result is in natural one-dimensional frequency order
    ``row_frequency + R * column_frequency``.  ``layout`` only changes the
    physical shared address used between the two passes; it cannot change the
    mathematical result.  ``normalize`` applies one final ``1/(R*R)`` scale,
    including for the inverse direction, matching the optional final CUDA
    normalization policy.
    """

    r, g, columns = _validate_shape(r, g, columns)
    layout = _normalise_layout(layout)
    flat, original_shape = _as_flat_columns(values, r, columns)
    del layout  # The logical tile is independent of its physical permutation.
    k = r // g
    local_n = r * r
    sign = 1.0 if inverse else -1.0
    tile = np.empty((r, r, columns), dtype=np.complex128)

    # First FFT axis: x[(G*j+lane), row], followed by the R^2 cross twiddle.
    for row in range(r):
        lane_values = np.empty((g, k, columns), dtype=np.complex128)
        for lane_index in range(g):
            for j in range(k):
                source = (g * j + lane_index) * r + row
                lane_values[lane_index, j, :] = flat[source, :]
        transformed = np.empty_like(lane_values)
        for col in range(columns):
            transformed[:, :, col] = _cooperative_1d(
                lane_values[:, :, col], r, g, inverse
            )
        for q in range(g):
            for j in range(k):
                frequency = j + k * q
                phase = np.exp(sign * 2j * np.pi * frequency * row / local_n)
                tile[frequency, row, :] = transformed[q, j, :] * phase

    # Second FFT axis: read tile[row, G*j+lane] and flatten frequencies as
    # row_frequency + R * column_frequency.
    output = np.empty_like(flat)
    for row_frequency in range(r):
        lane_values = np.empty((g, k, columns), dtype=np.complex128)
        for lane_index in range(g):
            for j in range(k):
                source = g * j + lane_index
                lane_values[lane_index, j, :] = tile[row_frequency, source, :]
        transformed = np.empty_like(lane_values)
        for col in range(columns):
            transformed[:, :, col] = _cooperative_1d(
                lane_values[:, :, col], r, g, inverse
            )
        for q in range(g):
            for j in range(k):
                column_frequency = j + k * q
                output[row_frequency + r * column_frequency, :] = transformed[q, j, :]

    if normalize:
        output /= local_n
    return _restore_shape(output, original_shape, r, columns)


def dense_fft(
    values: np.ndarray | Sequence[complex],
    r: int,
    *,
    inverse: bool = False,
    normalize: bool = False,
    columns: int = 1,
) -> np.ndarray:
    """Reference transform used by tests and the command-line check."""

    r, _, columns = _validate_shape(r, 1, columns)
    flat, original_shape = _as_flat_columns(values, r, columns)
    if inverse:
        result = np.fft.ifft(flat, axis=0)
        if not normalize:
            result *= r * r
    else:
        result = np.fft.fft(flat, axis=0)
        if normalize:
            result /= r * r
    return _restore_shape(result, original_shape, r, columns)


def _rotate_right_low(value: int, bits: int, rotation: int) -> int:
    if bits <= 0:
        return value
    mask = (1 << bits) - 1
    rotation %= bits
    if rotation == 0:
        return value
    low = value & mask
    return (value & ~mask) | ((low >> rotation) | ((low << (bits - rotation)) & mask))


def shared_row(
    a: int,
    b: int,
    r: int,
    g: int,
    columns: int,
    *,
    layout: str = "linear",
    complex_bytes: int = 8,
) -> int:
    """Return the physical row offset ``f(a,b)`` used by shared staging.

    The XOR branch mirrors ``shared_row`` in ``src/fft_register_tile.cuh``:
    ``L`` is the low address-bit budget, ``H=min(log2(G), L)``, and the lane
    digit is ``(a >> log2(K)) << (L-H)``.  All rotations are low-``L``-bit
    right rotations; the upper bits of ``b`` remain untouched by a rotation.
    """

    r, g, columns = _validate_shape(r, g, columns)
    layout = _normalise_layout(layout)
    complex_bytes = _normalise_complex_bytes(complex_bytes)
    if not 0 <= a < r or not 0 <= b < r:
        raise ValueError("a and b must be in [0, R)")
    if layout == "linear":
        return b
    if g == 1:
        return b ^ a

    # The CUDA implementation uses integer division followed by floor log2;
    # this retains the defined L=0 behavior when one vector exceeds 128 bytes.
    rows_per_transaction = 128 // (complex_bytes * columns)
    low_bits = min(_log2(r, "R"), rows_per_transaction.bit_length() - 1 if rows_per_transaction else 0)
    lane_bits = _log2(g, "G")
    rotation = min(lane_bits, low_bits)
    mask = (1 << low_bits) - 1 if low_bits else 0
    k = r // g
    lane_digit = (a >> _log2(k, "K")) << (low_bits - rotation)
    return (
        _rotate_right_low(b, low_bits, rotation)
        ^ a
        ^ (_rotate_right_low(lane_digit, low_bits, rotation) & mask)
    )


def physical_thread(row: int, lane: int, col: int, g: int, columns: int) -> int:
    """CUDA ``threadIdx.x`` for the cooperative row/lane/column tuple."""

    if row < 0 or lane < 0 or col < 0 or lane >= g or col >= columns:
        raise ValueError("physical thread coordinates are out of range")
    return (row * g + lane) * columns + col


def _reverse_bits(value: int, width: int) -> int:
    result = 0
    for bit in range(width):
        result = (result << 1) | ((value >> bit) & 1)
    return result


def shuffle_source_thread(row: int, lane: int, col: int, r: int, g: int, columns: int) -> int:
    """Model the CUDA shuffle source as a block-global thread index.

    CUDA's ``__shfl_sync`` source argument is a lane within the caller's warp;
    adding the caller's warp base makes the returned value convenient for the
    row/column checks below.
    """

    _validate_shape(r, g, columns)
    if g == 1:
        return physical_thread(row, lane, col, g, columns)
    lane_bits = _log2(g, "G")
    thread = physical_thread(row, lane, col, g, columns)
    source_lane = (thread & 31) ^ ((lane ^ _reverse_bits(lane, lane_bits)) * columns)
    return (thread & ~31) + source_lane


@dataclass(frozen=True)
class SharedAddress:
    a: int
    b: int
    col: int
    address: int


@dataclass(frozen=True)
class SharedPair:
    """One logical tile element and its writer/read-side thread coordinates."""

    a: int
    b: int
    col: int
    address: int
    writer_row: int
    writer_lane: int
    writer_j: int
    reader_row: int
    reader_lane: int
    reader_j: int


def shared_address(
    a: int,
    b: int,
    col: int,
    r: int,
    g: int,
    columns: int,
    *,
    layout: str = "linear",
    complex_bytes: int = 8,
) -> int:
    """Physical complex-element address ``(a*R+f(a,b))*Columns+col``."""

    r, g, columns = _validate_shape(r, g, columns)
    if not 0 <= col < columns:
        raise ValueError("col must be in [0, columns)")
    return (a * r + shared_row(a, b, r, g, columns, layout=layout, complex_bytes=complex_bytes)) * columns + col


def enumerate_shared_addresses(
    r: int,
    g: int,
    columns: int,
    *,
    layout: str = "linear",
    complex_bytes: int = 8,
) -> tuple[SharedAddress, ...]:
    """Enumerate every logical tile element and its physical shared address."""

    _validate_shape(r, g, columns)
    return tuple(
        SharedAddress(a, b, col, shared_address(a, b, col, r, g, columns, layout=layout, complex_bytes=complex_bytes))
        for a in range(r)
        for b in range(r)
        for col in range(columns)
    )


def enumerate_write_read_pairs(
    r: int,
    g: int,
    columns: int,
    *,
    layout: str = "linear",
    complex_bytes: int = 8,
) -> tuple[SharedPair, ...]:
    """Enumerate first-pass writers and second-pass readers for every element."""

    _validate_shape(r, g, columns)
    k = r // g
    records: list[SharedPair] = []
    for a in range(r):
        writer_lane, writer_j = divmod(a, k)
        for b in range(r):
            reader_j, reader_lane = divmod(b, g)
            for col in range(columns):
                records.append(
                    SharedPair(
                        a=a,
                        b=b,
                        col=col,
                        address=shared_address(a, b, col, r, g, columns, layout=layout, complex_bytes=complex_bytes),
                        writer_row=b,
                        writer_lane=writer_lane,
                        writer_j=writer_j,
                        reader_row=a,
                        reader_lane=reader_lane,
                        reader_j=reader_j,
                    )
                )
    return tuple(records)


def verify_shared_layout(
    r: int,
    g: int,
    columns: int,
    *,
    layout: str = "linear",
    complex_bytes: int = 8,
) -> dict[str, object]:
    """Check bijection, range, and matching writer/read addresses."""

    records = enumerate_write_read_pairs(
        r, g, columns, layout=layout, complex_bytes=complex_bytes
    )
    addresses = [record.address for record in records]
    expected = r * r * columns
    unique = len(set(addresses)) == expected
    in_range = set(addresses) == set(range(expected))
    matching = True
    k = r // g
    for record in records:
        # Reconstruct the logical tile coordinate from each side's physical
        # loop variables before comparing their addresses.
        writer_a = record.writer_j + k * record.writer_lane
        writer_b = record.writer_row
        reader_a = record.reader_row
        reader_b = record.reader_j * g + record.reader_lane
        writer_address = shared_address(
            writer_a,
            writer_b,
            record.col,
            r,
            g,
            columns,
            layout=layout,
            complex_bytes=complex_bytes,
        )
        reader_address = shared_address(
            reader_a,
            reader_b,
            record.col,
            r,
            g,
            columns,
            layout=layout,
            complex_bytes=complex_bytes,
        )
        matching &= (
            (writer_a, writer_b) == (record.a, record.b)
            and (reader_a, reader_b) == (record.a, record.b)
            and writer_address == reader_address == record.address
        )
    return {
        "ok": unique and in_range and matching,
        "layout": _normalise_layout(layout),
        "R": r,
        "G": g,
        "K": r // g,
        "columns": columns,
        "elements": expected,
        "unique_addresses": len(set(addresses)),
        "in_range": in_range,
        "writer_read_match": matching,
    }


def verify_shuffle_groups(r: int, g: int, columns: int) -> dict[str, object]:
    """Check shuffle sources stay within their row group, including partial warps."""

    _validate_shape(r, g, columns)
    active_threads = r * g * columns
    records = []
    same_row = True
    same_col = True
    active = True
    for row in range(r):
        for lane in range(g):
            for col in range(columns):
                source = shuffle_source_thread(row, lane, col, r, g, columns)
                target_row, target_lane_col = divmod(source, g * columns)
                target_lane, target_col = divmod(target_lane_col, columns)
                records.append((row, lane, col, source, target_row, target_lane, target_col))
                same_row &= target_row == row
                same_col &= target_col == col
                active &= 0 <= source < active_threads
    return {
        "ok": same_row and same_col and active,
        "R": r,
        "G": g,
        "columns": columns,
        "active_threads": active_threads,
        "partial_warp": active_threads % 32 != 0,
        "same_row": same_row,
        "same_column": same_col,
        "sources_active": active,
        "records": records,
    }


def shared_bank_diagnostic(
    accesses: Iterable[tuple[int, int, int]],
    *,
    complex_bytes: int,
    base_byte_offset: int = 0,
) -> dict[str, object]:
    """Return a conditional shared-bank/128-byte transaction model.

    This is intentionally only an address-level diagnostic.  It assumes
    32 four-byte banks and counts touched 128-byte sectors; it does not model
    compiler vectorization, instruction issue, replay, or actual bank service.
    NCU on the compiled kernel is required before making a hardware claim.
    """

    complex_bytes = _normalise_complex_bytes(complex_bytes)
    addresses = tuple(int(address) for address in accesses)
    byte_ranges = [
        range(base_byte_offset + address * complex_bytes,
              base_byte_offset + (address + 1) * complex_bytes, 4)
        for address in addresses
    ]
    words = [word // 4 for byte_range in byte_ranges for word in byte_range]
    bank_counts = [0] * 32
    for word in words:
        bank_counts[word % 32] += 1
    sectors = {word * 4 // 128 for word in words}
    return {
        "addresses": len(addresses),
        "complex_bytes": complex_bytes,
        "base_byte_offset": base_byte_offset,
        "aligned_128b_base": base_byte_offset % 128 == 0,
        "sectors_128b": len(sectors),
        "bank_counts": bank_counts,
        "max_bank_uses": max(bank_counts, default=0),
        "diagnostic_only": True,
        "requires_ncu_confirmation": True,
    }


def run_numpy_check(
    r: int,
    g: int,
    *,
    columns: int = 1,
    layout: str = "linear",
    seed: int = 817,
) -> dict[str, object]:
    """Generate a deterministic dense check report for CLI/automation use."""

    _validate_shape(r, g, columns)
    rng = np.random.RandomState(seed)
    values = rng.normal(size=(r * r, columns)) + 1j * rng.normal(size=(r * r, columns))
    actual = cooperative_fft(values, r, g, columns=columns, layout=layout)
    expected = dense_fft(values, r, columns=columns)
    delta = np.asarray(actual) - np.asarray(expected)
    reference_norm = float(np.linalg.norm(expected))
    return {
        "R": r,
        "G": g,
        "K": r // g,
        "columns": columns,
        "layout": _normalise_layout(layout),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2_error": float(np.linalg.norm(delta) / max(reference_norm, 1e-30)),
        "ok": bool(np.allclose(actual, expected, rtol=2e-11, atol=2e-11)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", type=int, default=8, help="local square root (power of two)")
    parser.add_argument("--g", type=int, default=2, help="cooperative lanes (power of two)")
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--layout", choices=("linear", "xor"), default="linear")
    parser.add_argument("--complex-bytes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--addresses", action="store_true", help="include the shared-address report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report: dict[str, object] = {"numpy": run_numpy_check(args.r, args.g, columns=args.columns, layout=args.layout, seed=args.seed)}
    if args.addresses:
        report["shared"] = verify_shared_layout(
            args.r,
            args.g,
            args.columns,
            layout=args.layout,
            complex_bytes=args.complex_bytes,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(bool(item.get("ok")) for item in report.values() if isinstance(item, dict) and "ok" in item) else 1


if __name__ == "__main__":
    raise SystemExit(main())
