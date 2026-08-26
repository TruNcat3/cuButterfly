#!/usr/bin/env python3
import random
import unittest


MODULUS = 12289  # 3 * 2^12 + 1


def primitive_root(modulus):
    for candidate in range(2, modulus):
        if pow(candidate, (modulus - 1) // 2, modulus) != 1 and \
           pow(candidate, (modulus - 1) // 3, modulus) != 1:
            return candidate
    raise AssertionError("primitive root not found")


def bit_reverse(value, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def reference(values, log_n, inverse):
    n = 1 << log_n
    root = pow(primitive_root(MODULUS), (MODULUS - 1) // n, MODULUS)
    if inverse:
        root = pow(root, MODULUS - 2, MODULUS)
    state = [values[bit_reverse(index, log_n)] for index in range(n)]
    for stage in range(log_n):
        half = 1 << stage
        step = pow(root, n // (2 * half), MODULUS)
        for base in range(0, n, 2 * half):
            omega = 1
            for offset in range(half):
                left = state[base + offset]
                right = state[base + offset + half] * omega % MODULUS
                state[base + offset] = (left + right) % MODULUS
                state[base + offset + half] = (left - right) % MODULUS
                omega = omega * step % MODULUS
    if inverse:
        scale = pow(n, MODULUS - 2, MODULUS)
        state = [value * scale % MODULUS for value in state]
    return state


def hybrid_schedule(values, log_n, flow_tile_log_n, stage_space, inverse):
    n = 1 << log_n
    root = pow(primitive_root(MODULUS), (MODULUS - 1) // n, MODULUS)
    if inverse:
        root = pow(root, MODULUS - 2, MODULUS)
    state = [values[bit_reverse(index, log_n)] for index in range(n)]
    for round_base in range(0, log_n, flow_tile_log_n):
        round_stages = min(flow_tile_log_n, log_n - round_base)
        for fold in range(0, round_stages, stage_space):
            active_stages = min(stage_space, round_stages - fold)
            stage_base = round_base + fold
            token_points = 1 << active_stages
            stage_stride = 1 << stage_base
            for token in range(n // token_points):
                subgraph_block, subgraph_offset = divmod(token, stage_stride)
                subgraph_base = subgraph_block << (stage_base + active_stages)
                logical = [subgraph_base + subgraph_offset + (local << stage_base)
                           for local in range(token_points)]
                data = [state[index] for index in logical]
                for local_stage in range(active_stages):
                    half = 1 << local_stage
                    global_stage = stage_base + local_stage
                    step = pow(root, n // (2 << global_stage), MODULUS)
                    for butterfly in range(token_points // 2):
                        group, offset = divmod(butterfly, half)
                        left_index = group * (2 * half) + offset
                        right_index = left_index + half
                        twiddle_offset = subgraph_offset + (offset << stage_base)
                        right = data[right_index] * pow(step, twiddle_offset, MODULUS) % MODULUS
                        left = data[left_index]
                        data[left_index] = (left + right) % MODULUS
                        data[right_index] = (left - right) % MODULUS
                for index, value in zip(logical, data):
                    state[index] = value
    if inverse:
        scale = pow(n, MODULUS - 2, MODULUS)
        state = [value * scale % MODULUS for value in state]
    return state


class HybridDataflowScheduleTest(unittest.TestCase):
    def test_token_indexing_and_twiddles(self):
        random.seed(0xC0FFEE)
        for log_n in (6, 7, 8, 10, 12):
            values = [random.randrange(MODULUS) for _ in range(1 << log_n)]
            for flow_tile_log_n in (5, 6, 7, 8):
                if flow_tile_log_n > log_n:
                    continue
                for stage_space in range(2, 9):
                    if stage_space > flow_tile_log_n:
                        continue
                    for inverse in (False, True):
                        self.assertEqual(reference(values, log_n, inverse),
                                         hybrid_schedule(values, log_n, flow_tile_log_n,
                                                         stage_space, inverse))

    def test_xor_layout_is_a_permutation(self):
        for log_n in (6, 8, 12):
            for flow_tile_log_n in (5, 6, 7, 8):
                if flow_tile_log_n > log_n:
                    continue
                for data_space in (8, 16, 32):
                    mask = 2 * data_space - 1
                    physical = [((logical & ~mask) |
                                 ((logical ^ (logical >> flow_tile_log_n)) & mask))
                                for logical in range(1 << log_n)]
                    self.assertEqual(list(range(1 << log_n)), sorted(physical))


if __name__ == "__main__":
    unittest.main()
