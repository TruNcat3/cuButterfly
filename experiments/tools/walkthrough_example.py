"""Exact analytical witness shared by the abstract workflow and layout figure.

This is a constructed 16-point Hadamard schedule, not a measured GPU plan.
Four-value local storage is per worker after auxiliary reservations; a separate
16-word, four-bank boundary holds one transform. No cross-band overlap is used.
"""


def placement(r, c):
    """Successor-major offset followed by a bank permutation."""
    return 4 * c + (r ^ c)


def checked_example():
    rows = [[4 * r + c for c in range(4)] for r in range(4)]
    columns = [[4 * r + c for r in range(4)] for c in range(4)]
    assert sorted(placement(r, c) for r in range(4) for c in range(4)) == list(range(16))
    for row in rows:
        assert all((i ^ (1 << q)) in row for i in row for q in (0, 1))
    for column in columns:
        assert all((i ^ (1 << q)) in column for i in column for q in (2, 3))
    assert all(len(set(row) & set(column)) == 1 for row in rows for column in columns)
    for r in range(4):
        assert sorted(placement(r, c) % 4 for c in range(4)) == list(range(4))
    for c in range(4):
        assert sorted(placement(r, c) % 4 for r in range(4)) == list(range(4))
        assert {placement(r, c) // 4 for r in range(4)} == {c}
    assert [placement(r, 1) for r in range(4)] == [5, 4, 7, 6]
    # Two workers reuse four-value local slots for two regions in each band.
    schedule = [[('P', 0), ('P', 2), ('C', 0), ('C', 2)],
                [('P', 1), ('P', 3), ('C', 1), ('C', 3)]]
    completed = set()
    for step in range(4):
        for worker in schedule:
            kind, region = worker[step]
            if kind == 'C':
                assert completed.issuperset({('P', r) for r in range(4)})
            completed.add((kind, region))
    assert len(completed) == 8
    # The swap changes traversal, not the mathematical separable transform.
    def pair_stage(values, q):
        result = list(values)
        for i in range(16):
            j = i ^ (1 << q)
            if i < j:
                result[i], result[j] = values[i] + values[j], values[i] - values[j]
        return result
    inputs = [i - 7 for i in range(16)]
    prefix = pair_stage(pair_stage(inputs, 0), 1)
    boundary = [None] * 16
    for r in range(4):
        for c in range(4):
            boundary[placement(r, c)] = prefix[4 * r + c]
    restored = [boundary[placement(i // 4, i % 4)] for i in range(16)]
    actual = pair_stage(pair_stage(restored, 2), 3)
    reference = [sum(x * (-1 if bin(i & j).count('1') % 2 else 1)
                     for i, x in enumerate(inputs)) for j in range(16)]
    assert actual == reference
    return rows, columns, schedule


if __name__ == '__main__':
    checked_example()
    print('16-point witness: closure, intersections, placement, readiness, and output verified')
