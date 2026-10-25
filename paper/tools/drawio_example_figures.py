#!/usr/bin/env python3
"""Native editable analytical examples; no images or performance samples.

Run with /home/wt/yes/envs/cubutterfly/bin/python. All maps are exact toy
examples from the method and execution sections, not hardware guarantees.
"""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, GRAY, LIGHT, PALE


ROW_COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def hierarchy():
    d = Diagram('hierarchy', 480, 360)
    d.title('Row producers to column consumers')
    d.text(20, 38, 170, 23, 'P = {0, 1}: vary c', bold=True)
    d.text(290, 38, 170, 23, 'Q = {2, 3}: vary r', bold=True)
    step, y0 = 34, 72
    addresses = []
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            address = 4 * c + r
            addresses.append(address)
            assert divmod(address, 4) == (c, r)
            selected = c == 1
            for x, y in ((35 + c * step, y0 + r * step),
                         (309 + r * step, y0 + c * step)):
                d.box(x, y, step, step, str(i), fill=PALE[r],
                      stroke=ROW_COLORS[r] if selected else '#ffffff',
                      width=2.5 if selected else 1, rounded=False,
                      bold=selected)
    assert sorted(addresses) == list(range(16))
    d.text(35, 58, 136, 14, 'logical i = 4r + c')
    d.text(292, 58, 170, 14, 'address = 4c + r')
    d.line([(184, 133), (291, 133)], color=GREEN, width=2, arrow=True)
    d.text(183, 99, 109, 25, 'writeback', color=GREEN)
    d.text(180, 145, 115, 44, 'c = 1\nfan-in: 4 rows')
    d.text(12, 216, 456, 22, 'Highlighted consumer: {1, 5, 9, 13}, all four colors')
    d.line([(16, 246), (464, 246)], color=LIGHT)
    d.text(12, 251, 456, 23, 'Illustrative refinement: L = 24, k_max = 8', bold=True)
    for j in range(3):
        x = 24 + j * 153
        d.box(x, 284, 126, 34, f'8-stage group {j}',
              fill=PALE[j], stroke=ROW_COLORS[j])
        if j < 2:
            d.line([(x + 130, 301), (x + 149, 301)], arrow=True)
    assert sum([8, 8, 8]) == 24
    d.text(10, 322, 460, 30,
           'Bounded local state; exchanges between groups.\nParent domain need not reside on chip.')
    return d.save()


def bank_layout():
    d = Diagram('bank_layout', 480, 342)
    d.title('Four-bank layout witness')
    step, y0 = 35, 86
    chosen = [4 * r + 1 for r in range(4)]
    bank_lists = []
    for x0, swizzle, title in ((49, False, '(a) Linear'),
                                (293, True, '(b) Row XOR')):
        d.text(x0 - 20, 37, 180, 23, title, bold=True)
        d.text(x0 - 29, 62, 28, 22, 'r')
        for bank in range(4):
            d.text(x0 + bank * step, 62, step, 22, f'b{bank}')
        addresses, occupied = [], []
        for r in range(4):
            d.text(x0 - 29, y0 + r * step, 28, step, str(r), color=GRAY)
            for c in range(4):
                i = 4 * r + c
                bank = c ^ r if swizzle else c
                address = 4 * r + bank
                addresses.append(address)
                selected = c == 1
                if selected:
                    occupied.append(bank)
                d.box(x0 + bank * step, y0 + r * step, step, step,
                      str(i), fill=PALE[0] if selected else '#f3f4f5',
                      stroke=BLUE if selected else '#ffffff',
                      width=2 if selected else 1, bold=selected,
                      rounded=False)
        assert sorted(addresses) == list(range(16))
        assert len(set(occupied)) == (4 if swizzle else 1)
        bank_lists.append(occupied)
        d.text(x0 - 20, 226, 180, 22,
               '4 reads, 4 banks' if swizzle else '4 reads, 1 bank', bold=True)
    assert chosen == [1, 5, 9, 13]
    assert bank_lists == [[1, 1, 1, 1], [1, 0, 3, 2]]
    d.line([(204, 155), (267, 155)], color=GREEN, width=2, arrow=True)
    d.text(10, 253, 460, 22, 'Address: 4r + c  ->  4r + (c XOR r)')
    d.text(10, 277, 460, 22, 'Lanes 0, 1, 2, 3 read indices 1, 5, 9, 13 (c = 1)')
    d.text(10, 302, 460, 30,
           'Four banks; one word/value; stated lane/index pairing.\nVector values and partial warps need transaction analysis.')
    return d.save()


def reverse3(i):
    return ((i & 1) << 2) | (i & 2) | ((i >> 2) & 1)


def semantic_transport():
    d = Diagram('semantic_transport', 480, 380)
    d.title('Three-bit index reversal')
    permutation = [reverse3(i) for i in range(8)]
    assert permutation == [0, 4, 2, 6, 1, 5, 3, 7]
    assert sorted(permutation) == list(range(8))
    assert all(reverse3(reverse3(i)) == i for i in range(8))
    assert all(reverse3(i ^ (1 << q)) == (reverse3(i) ^ (1 << (2 - q)))
               for i in range(8) for q in range(3))
    d.text(31, 37, 145, 22, 'q = 0: adjacent', color=BLUE, bold=True)
    d.text(303, 37, 145, 22, 'q = 2: stride 4', color=GREEN, bold=True)
    for first in range(0, 8, 2):
        assert permutation[first] ^ permutation[first + 1] == 4
        for x, color, fill in ((35, BLUE, PALE[0]), (309, GREEN, PALE[2])):
            d.box(x, 62 + first * 20, 136, 40,
                  fill=fill, stroke=color, rounded=False)
    for i, j in enumerate(permutation):
        y = 62 + i * 20
        d.text(39, y, 128, 20, f'{i} : {i:03b}')
        d.text(313, y, 128, 20, f'{j} : {j:03b}')
        d.line([(180, y + 10), (300, y + 10)], arrow=True)
    d.text(12, 227, 456, 22, 'Stage-bit sequence: (0, 1, 2) -> (2, 1, 0)')
    d.text(12, 250, 456, 20, 'Same index domain; axis labels for shape (4, 2)')
    d.text(12, 271, 456, 20, 'i = 2r + c:  q0 = c0,  q1 = r0,  q2 = r1')
    assert all(2 * r + c == ((r >> 1) << 2) + ((r & 1) << 1) + c
               for r in range(4) for c in range(2))
    for x, w, title, formula, color, fill in (
        (12, 204, 'CT local core', '(a + wb, a - wb)', BLUE, PALE[0]),
        (228, 240, 'Inverse local GS', '((u+v)/2, (u-v)/(2w))', ORANGE, PALE[1]),
    ):
        d.box(x, 297, w, 44, fill=fill, stroke=color, rounded=False)
        d.text(x + 4, 299, w - 8, 20, title, bold=True)
        d.text(x + 4, 320, w - 8, 19, formula)
    d.text(10, 344, 460, 18, 'Local inverse requires invertible w AND 2')
    d.text(10, 364, 460, 16, 'Graph relabeling; arithmetic labels remain distinct', color=GRAY)
    return d.save()


if __name__ == '__main__':
    for generate in (hierarchy, bank_layout, semantic_transport):
        print(generate())
