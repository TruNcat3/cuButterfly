#!/usr/bin/env python3
"""CPU witness for the paper's subgraph/layout construction, not GPU support."""
import argparse
import itertools
import json
from pathlib import Path


def bands(order, cuts):
    result, start = [], 0
    for end in range(1, len(order) + 1):
        if end == len(order) or cuts & (1 << (end - 1)):
            result.append(order[start:end])
            start = end
    return result


def components(log_n, bits):
    mask = sum(1 << bit for bit in bits)
    return [[base | sum(((v >> j) & 1) << bit for j, bit in enumerate(bits))
             for v in range(1 << len(bits))]
            for base in range(1 << log_n) if not base & mask]


def layout(log_n, bits):
    positions = tuple(bits) + tuple(bit for bit in range(log_n) if bit not in bits)
    return [sum(((i >> bit) & 1) << j for j, bit in enumerate(positions))
            for i in range(1 << log_n)]


def execute(values, stage_bands):
    """Stage-local arithmetic with physical successor-layout handoffs."""
    log_n = len(values).bit_length() - 1
    storage, current = list(values), list(range(len(values)))
    for index, bits in enumerate(stage_bands):
        successor = (layout(log_n, stage_bands[index + 1])
                     if index + 1 < len(stage_bands) else list(range(len(values))))
        output = [None] * len(values)
        for component in components(log_n, bits):
            local = {i: storage[current[i]] for i in component}
            for bit in bits:
                for i in component:
                    if not i & (1 << bit):
                        partner = i ^ (1 << bit)
                        a, b = local[i], local[partner]
                        local[i], local[partner] = a + b, a - b
            for i, value in local.items():
                assert output[successor[i]] is None
                output[successor[i]] = value
        assert all(value is not None for value in output)
        storage, current = output, successor
    return storage


def dense_hadamard(values):
    # Independent matrix definition; does not call the staged transform.
    return [sum(value * (-1 if bin(i & k).count("1") % 2 else 1)
            for i, value in enumerate(values)) for k in range(len(values))]


def check_boundary_relations():
    """Compare symbolic bit relations with explicitly intersected value sets."""
    mask_pairs, intersections, packets_checked = 0, 0, 0
    for log_n in range(1, 6):
        n, full = 1 << log_n, (1 << log_n) - 1
        groups_by_mask = {
            mask: [set(group) for group in components(
                log_n, tuple(bit for bit in range(log_n) if mask & (1 << bit)))]
            for mask in range(n)
        }
        for p_mask, producers in groups_by_mask.items():
            for q_mask, consumers in groups_by_mask.items():
                mask_pairs += 1
                common = p_mask & q_mask
                union = p_mask | q_mask
                outside = full ^ union
                fanin = 1 << bin(q_mask & (full ^ p_mask)).count("1")
                fanout = 1 << bin(p_mask & (full ^ q_mask)).count("1")
                incoming = [set() for _ in consumers]
                outgoing = [set() for _ in producers]
                for p, producer in enumerate(producers):
                    for c, consumer in enumerate(consumers):
                        overlap = producer & consumer
                        a, b = min(producer), min(consumer)
                        expected = not ((a ^ b) & outside)
                        assert bool(overlap) == expected
                        assert len(overlap) == (1 << bin(common).count("1") if expected else 0)
                        intersections += 1
                        if overlap:
                            incoming[c].add(p)
                            outgoing[p].add(c)
                assert all(len(predecessors) == fanin for predecessors in incoming)
                assert all(len(successors) == fanout for successors in outgoing)
                # Explicit graph traversal, independent of the symbolic packet construction.
                remaining = {(0, p) for p in range(len(producers))}
                remaining |= {(1, c) for c in range(len(consumers))}
                actual_packets = []
                while remaining:
                    seen, queue = set(), [next(iter(remaining))]
                    while queue:
                        node = queue.pop()
                        if node in seen:
                            continue
                        seen.add(node)
                        side, index = node
                        if side == 0:
                            queue.extend((1, c) for c in outgoing[index])
                        else:
                            queue.extend((0, p) for p in incoming[index])
                    remaining -= seen
                    values = set().union(*(producers[index] if side == 0 else consumers[index]
                                           for side, index in seen))
                    actual_packets.append(frozenset(values))
                expected_packets = {frozenset(values) for values in groups_by_mask[union]}
                assert set(actual_packets) == expected_packets
                assert len(actual_packets) == 1 << (log_n - bin(union).count("1"))
                packets_checked += len(actual_packets)
    return dict(mask_pairs=mask_pairs, explicit_intersections=intersections,
                connected_packets=packets_checked, log_n_range=[1, 5])


def run():
    log_n, n = 4, 16
    vectors = [[int(i == j) for i in range(n)] for j in range(n)]
    vectors.append([3 * i - 11 for i in range(n)])
    references = [dense_hadamard(vector) for vector in vectors]
    cases, transforms, checked_bands = 0, 0, 0
    for order in itertools.permutations(range(log_n)):
        for cuts in range(1 << (log_n - 1)):
            stage_bands = bands(order, cuts)
            for bits in stage_bands:
                groups, addresses = components(log_n, bits), layout(log_n, bits)
                assert sorted(i for group in groups for i in group) == list(range(n))
                assert sorted(addresses) == list(range(n))
                for group in groups:
                    assert all(i ^ (1 << bit) in group for i in group for bit in bits)
                    physical = sorted(addresses[i] for i in group)
                    assert physical == list(range(physical[0], physical[0] + len(group)))
                checked_bands += 1
            for vector, reference in zip(vectors, references):
                assert execute(vector, stage_bands) == reference
                transforms += 1
            cases += 1
    rows, columns = components(4, (0, 1)), components(4, (2, 3))
    assert layout(4, (2, 3)) == [4 * (i % 4) + i // 4 for i in range(16)]
    predecessors = [{r for r, row in enumerate(rows) if set(row) & set(column)}
                    for column in columns]
    assert all(required == {0, 1, 2, 3} for required in predecessors)
    assert not any(required <= {0} for required in predecessors)
    return {
        "status": "passed", "scope": "Exact CPU construction at N=16 and boundary relations at N=2..32; no GPU support/performance claim",
        "stage_orders": 24, "ordered_compositions_per_order": 8,
        "mapping_cases": cases, "band_closure_and_layout_checks": checked_bands,
        "dense_reference_comparisons": transforms,
        "example_logical_value_bytes": {"1+1+1+1": 512, "2+2": 256, "4": 128},
        "negative_control": "one completed producer row cannot release a complete consumer column",
        "boundary_relations": check_boundary_relations(),
        "not_verified": ["arbitrary FFT coefficient adaptation", "CUDA lowering coverage",
                         "bank/global transaction efficiency", "GPU performance or optimality"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
