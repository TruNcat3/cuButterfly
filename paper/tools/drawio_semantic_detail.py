"""Complete N=8 labeled dependency transport, as editable draw.io geometry.

Importing this module never writes a figure.  The caller owns generation.
The right graph relabels each layer of the left graph; time still runs left
 to right, and the local arithmetic labels retain their ordered operands.
"""
from drawio_common import Diagram, BLUE, GREEN, GRAY, LIGHT, PALE


def _reverse3(i):
    return ((i & 1) << 2) | (i & 2) | ((i >> 2) & 1)


def _checked_transport():
    """Check full directed edges, ordered pairs, and a two-stage closed region."""
    rev = [_reverse3(i) for i in range(8)]
    assert rev == [0, 4, 2, 6, 1, 5, 3, 7]
    assert sorted(rev) == list(range(8))
    assert all(rev[rev[i]] == i for i in range(8))
    bits = (0, 1, 2)
    mapped_bits = (2, 1, 0)
    edges = lambda q: {(i, j) for i in range(8)
                       for j in (i, i ^ (1 << q))}
    for q, mapped_q in zip(bits, mapped_bits):
        assert {(rev[i], rev[j]) for i, j in edges(q)} == edges(mapped_q)
        # Operand order and pair-specific label identity are transported too.
        pairs = [(i, i ^ (1 << q)) for i in range(8) if not i & (1 << q)]
        transported = [(rev[a], rev[b]) for a, b in pairs]
        assert all(b == a ^ (1 << mapped_q) for a, b in transported)
        assert len({frozenset(p) for p in transported}) == 4
    region = set(range(4))
    mapped_region = {rev[i] for i in region}
    assert mapped_region == {0, 2, 4, 6}
    for active_bits, values in ((bits[:2], region),
                               (mapped_bits[:2], mapped_region)):
        for q in active_bits:
            assert {i ^ (1 << q) for i in values} == values
            assert all((i in values) == (j in values) for i, j in edges(q))
    # The third stage crosses the selected prefix boundary on both sides.
    assert all((i ^ 4) not in region for i in region)
    assert all((i ^ 1) not in mapped_region for i in mapped_region)
    return rev, region, mapped_region


def semantic_transport():
    rev, region, mapped_region = _checked_transport()
    d = Diagram('semantic_transport', 480, 328)
    d.title('Transport the complete labeled dependency graph')
    d.text(9, 36, 213, 21, '(a) Bits 0 → 1 → 2', bold=True, color=BLUE)
    d.text(258, 36, 213, 21, '(b) Bits 2 → 1 → 0', bold=True, color=GREEN)
    d.text(12, 59, 456, 21, 'Every layer: i → rev₃(i); stage order retained', color=GRAY)

    ys = [110 + 17 * i for i in range(8)]
    for xs, bits, selected, color, pale in (
            ([39, 91, 143, 195], (0, 1, 2), region, BLUE, PALE[0]),
            ([285, 337, 389, 441], (2, 1, 0), mapped_region, GREEN, PALE[2])):
        # Shading covers exactly the first two stages of the chosen closed
        # region.  On the right the same region is noncontiguous in index order.
        for i in selected:
            d.box(xs[0] - 5, ys[i] - 7, xs[2] - xs[0] + 10, 14,
                  fill=pale, stroke='none', rounded=False)
        for k, q in enumerate(bits):
            d.text(xs[k] + 3, 84, 46, 20, f'L{k}', bold=True, color=color)
            for i in range(8):
                for j in (i, i ^ (1 << q)):
                    inside = k < 2 and i in selected
                    d.line([(xs[k], ys[i]), (xs[k + 1], ys[j])],
                           color=color if inside else LIGHT,
                           width=1.25 if inside else .8)
        for k, x in enumerate(xs):
            for i, y in enumerate(ys):
                d.dot(x, y, radius=2.1,
                      color=color if k <= 2 and i in selected else GRAY)
        for i, y in enumerate(ys):
            d.text(xs[0] - 23, y - 8, 17, 16, str(i), color=GRAY)
            d.text(xs[-1] + 7, y - 8, 17, 16, str(i), color=GRAY)
    d.line([(224, 166), (258, 166)], color=GRAY, width=1.6, arrow=True)
    d.text(218, 138, 49, 21, 'rev₃', color=GRAY)
    d.text(12, 237, 456, 22,
           'Closed prefix: {0,1,2,3} → {0,4,2,6}', bold=True)
    d.line([(14, 266), (466, 266)], color=LIGHT)
    d.text(12, 272, 456, 21,
           'Lk carries ordered operands, coefficient w, and direction.', size=14)
    d.text(12, 296, 456, 22,
           'Example CT label: (a,b) → (a + wb, a − wb)', size=14)
    return d.save()
