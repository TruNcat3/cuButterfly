"""CPU ownership/dependency model for shared_resident_kernel.

This model deliberately checks the address contract rather than CUDA timing.  A
resident leaf may fuse several consecutive radix-2 stages, but it must not mix
the lower or higher bits that identify a different dependency-closed leaf.
"""

from __future__ import annotations

from collections import Counter

import pytest


def iterative_layout_index(logical: int, stages: int, consumer_stage: int) -> int:
    low_mask = (1 << consumer_stage) - 1
    return (
        (logical & low_mask)
        | ((logical >> (consumer_stage + 1)) << consumer_stage)
        | (((logical >> consumer_stage) & 1) << (stages - 1))
    )


def iterative_global_index(tile: int, local: int, first: int, stages: int) -> int:
    return (
        ((tile >> first) << (first + stages))
        | (tile & ((1 << first) - 1))
        | (local << first)
    )


def bit_reverse(value: int, width: int) -> int:
    result = 0
    for bit in range(width):
        result = (result << 1) | ((value >> bit) & 1)
    return result


def apply_pair(stage: int, offset: int, left: int, right: int) -> tuple[int, int]:
    # Distinguish stage and offset so an incorrect coefficient address cannot
    # accidentally pass an ownership-only check.
    return (
        left + 3 * right + 17 * stage + offset,
        5 * left - right - 11 * stage + 2 * offset,
    )


def finalize(value: int, n: int) -> int:
    return value + 97 * n


def _logical_groups(size: int, stage_offset: int, leaf_stages: int):
    groups = size >> leaf_stages
    low_mask = (1 << stage_offset) - 1
    for group in range(groups):
        low = group & low_mask
        high = group >> stage_offset
        base = (high << (stage_offset + leaf_stages)) | low
        yield base, [base | (lane << stage_offset) for lane in range(1 << leaf_stages)]


def _input_value(log_n: int, batch: int, global_index: int) -> int:
    return 100000 * batch + 13 * global_index + 7


def _reference_tile(
    *,
    log_n: int,
    tile: int,
    first: int,
    stages: int,
    bit_reverse_input: bool,
    bit_reversed_output: bool,
    batch: int = 0,
):
    size = 1 << stages
    values = {}
    for logical in range(size):
        global_index = iterative_global_index(tile, logical, first, stages)
        if bit_reverse_input and first == 0:
            global_index = bit_reverse(global_index, log_n)
        values[logical] = _input_value(log_n, batch, global_index)

    for local_stage in range(stages):
        half = 1 << local_stage
        for pair in range(size // 2):
            left = ((pair >> local_stage) << (local_stage + 1)) | (pair & (half - 1))
            right = left + half
            global_left = iterative_global_index(tile, left, first, stages)
            stage = first + local_stage
            values[left], values[right] = apply_pair(
                stage, global_left & ((1 << stage) - 1), values[left], values[right]
            )
    result = {}
    terminal = first + stages == log_n
    for i in range(size):
        reverse = bit_reversed_output and terminal
        logical = bit_reverse(i, stages) if reverse else i
        value = values[logical]
        if terminal:
            value = finalize(value, 1 << log_n)
        global_index = iterative_global_index(tile, logical, first, stages)
        if reverse:
            global_index = bit_reverse(global_index, log_n)
        result[global_index] = value
    return result


def resident_tile(
    *,
    log_n: int,
    tile: int,
    first: int,
    stages: int,
    parts: tuple[int, ...],
    writer_aligned: bool,
    bit_reverse_input: bool = False,
    bit_reversed_output: bool = False,
    batch: int = 0,
):
    size = 1 << stages
    assert sum(parts) == stages
    assert all(1 <= part <= 5 for part in parts)

    current = [None] * size
    next_values = [None] * size

    for logical in range(size):
        global_index = iterative_global_index(tile, logical, first, stages)
        if bit_reverse_input and first == 0:
            global_index = bit_reverse(global_index, log_n)
        physical = iterative_layout_index(logical, stages, 0) if writer_aligned else logical
        current[physical] = _input_value(log_n, batch, global_index)
    assert all(value is not None for value in current)

    stage_offset = 0
    for leaf_stages in parts:
        read_counts = Counter()
        write_counts = Counter()
        leaf_mask = ((1 << leaf_stages) - 1) << stage_offset
        for _, logical_group in _logical_groups(size, stage_offset, leaf_stages):
            assert len({logical & ~leaf_mask for logical in logical_group}) == 1
            registers = []
            for logical in logical_group:
                physical = (
                    iterative_layout_index(logical, stages, stage_offset)
                    if writer_aligned
                    else logical
                )
                read_counts[physical] += 1
                value = current[physical]
                assert value is not None
                registers.append(value)

            for local_stage in range(leaf_stages):
                half = 1 << local_stage
                stage = first + stage_offset + local_stage
                for pair in range((1 << leaf_stages) // 2):
                    left = ((pair >> local_stage) << (local_stage + 1)) | (pair & (half - 1))
                    right = left + half
                    assert logical_group[left] ^ logical_group[right] == 1 << (stage_offset + local_stage)
                    global_left = iterative_global_index(
                        tile, logical_group[left], first, stages
                    )
                    registers[left], registers[right] = apply_pair(
                        stage,
                        global_left & ((1 << stage) - 1),
                        registers[left],
                        registers[right],
                    )

            consumer_stage = (
                stage_offset + leaf_stages
                if stage_offset + leaf_stages < stages
                else stages - 1
            )
            for logical, value in zip(logical_group, registers):
                physical = (
                    iterative_layout_index(logical, stages, consumer_stage)
                    if writer_aligned
                    else logical
                )
                assert next_values[physical] is None
                next_values[physical] = value
                write_counts[physical] += 1

        assert all(count == 1 for count in read_counts.values())
        assert all(count == 1 for count in write_counts.values())
        assert all(value is not None for value in next_values)
        current, next_values = next_values, [None] * size
        stage_offset += leaf_stages

    result = {}
    terminal = first + stages == log_n
    for i in range(size):
        reverse = bit_reversed_output and terminal
        logical = bit_reverse(i, stages) if reverse else i
        source = iterative_layout_index(logical, stages, stages - 1) if writer_aligned else logical
        value = current[source]
        assert value is not None
        if terminal:
            value = finalize(value, 1 << log_n)
        global_index = iterative_global_index(tile, logical, first, stages)
        if reverse:
            global_index = bit_reverse(global_index, log_n)
        result[global_index] = value
    return result


@pytest.mark.parametrize("writer_aligned", [False, True])
@pytest.mark.parametrize("parts", [(1, 2, 3), (2, 2, 2), (5, 1), (3, 3)])
def test_resident_leaves_match_stage_by_stage_reference(writer_aligned, parts):
    log_n = 6
    expected = _reference_tile(
        log_n=log_n, tile=0, first=0, stages=log_n,
        bit_reverse_input=True, bit_reversed_output=True,
    )
    actual = resident_tile(
        log_n=log_n, tile=0, first=0, stages=log_n, parts=parts,
        writer_aligned=writer_aligned, bit_reverse_input=True,
        bit_reversed_output=True,
    )
    assert actual == expected


@pytest.mark.parametrize("writer_aligned", [False, True])
def test_interior_tile_preserves_global_ownership(writer_aligned):
    log_n, first, stages, tile = 8, 2, 4, 3
    expected = _reference_tile(
        log_n=log_n, tile=tile, first=first, stages=stages,
        bit_reverse_input=False, bit_reversed_output=False,
    )
    actual = resident_tile(
        log_n=log_n, tile=tile, first=first, stages=stages,
        parts=(1, 3), writer_aligned=writer_aligned,
    )
    assert actual == expected
    assert set(actual) == {
        iterative_global_index(tile, local, first, stages) for local in range(1 << stages)
    }


def test_strided_batched_addresses_do_not_touch_padding():
    log_n = stages = 5
    stride = 2
    batch_stride = (1 << log_n) * stride + 7
    valid = []
    for batch in range(3):
        values = resident_tile(
            log_n=log_n, tile=0, first=0, stages=stages,
            parts=(2, 1, 2), writer_aligned=True,
            bit_reverse_input=False, bit_reversed_output=False, batch=batch,
        )
        addresses = {batch * batch_stride + index * stride for index in values}
        valid.extend(addresses)
    assert len(valid) == len(set(valid)) == 3 * (1 << log_n)
    for batch in range(3):
        interval = range(batch * batch_stride, (batch + 1) * batch_stride)
        expected_padding = {
            batch * batch_stride + index
            for index in range(batch_stride)
            if index >= (1 << log_n) * stride or index % stride
        }
        assert set(interval) - set(valid) == expected_padding
