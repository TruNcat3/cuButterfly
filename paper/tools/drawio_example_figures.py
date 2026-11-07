#!/usr/bin/env python3
"""Native editable analytical examples; no images or performance samples.

Run with /home/wt/yes/envs/cubutterfly/bin/python. All maps are exact toy
examples from the method and execution sections, not hardware guarantees.
"""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, GRAY, LIGHT, PALE


ROW_COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def hierarchy():
    d = Diagram('hierarchy', 480, 436)
    d.title('(a) Row producers to column consumers')
    d.text(20, 30, 170, 20, 'P = {0, 1}: vary c', size=16, bold=True)
    d.text(290, 30, 170, 20, 'Q = {2, 3}: vary r', size=16, bold=True)
    step, y0 = 30, 68
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
                d.box(x, y, step, step, str(i), fill=PALE[r], size=16,
                      stroke=ROW_COLORS[r] if selected else '#ffffff',
                      width=2.5 if selected else 1, rounded=False,
                      bold=selected)
    assert sorted(addresses) == list(range(16))
    d.text(27, 50, 150, 18, 'logical i = 4r + c')
    d.text(289, 50, 170, 18, 'address = 4c + r')
    d.line([(173, 123), (291, 123)], color=GREEN, width=2, arrow=True)
    d.text(179, 94, 109, 25, 'writeback', size=16, color=GREEN)
    d.text(175, 136, 118, 44, 'c = 1\nfan-in: 4 rows', size=16)
    chosen = [4 * r + 1 for r in range(4)]
    assert chosen == [1, 5, 9, 13]
    assert len({i // 4 for i in chosen}) == 4
    d.text(12, 192, 456, 22, 'Consumer {1, 5, 9, 13}: all four producers', size=16)
    d.line([(16, 217), (464, 217)], color=LIGHT)
    d.text(12, 220, 456, 20, 'Capacity: L = 24; each child band ≤ k_max = 8', bold=True)
    d.box(22, 245, 436, 39, fill='#ffffff', stroke=GRAY, dashed=True)
    for k in range(3):
        d.box(32 + k * 142, 250, 130, 28, f'band {k}: 8 bits',
              fill=PALE[k], stroke=ROW_COLORS[k])
        if k < 2:
            d.line([(162 + k * 142, 264), (174 + k * 142, 264)], arrow=True)
    assert sum([8, 8, 8]) == 24
    assert max([8, 8, 8]) <= 8
    d.text(12, 286, 456, 18, 'Parent domain spans children; exchange at their boundaries.')
    d.line([(16, 308), (464, 308)], color=LIGHT)
    d.text(12, 311, 456, 21, '(b) Separate four-bank example: reads {1, 5, 9, 13}', bold=True)
    bank_lists = []
    for swizzle in (False, True):
        addresses, occupied = [], []
        for r in range(4):
            for c in range(4):
                bank = c ^ r if swizzle else c
                address = 4 * r + bank
                addresses.append(address)
                assert address % 4 == bank
                if c == 1:
                    occupied.append(bank)
        assert sorted(addresses) == list(range(16))
        assert len(set(occupied)) == (4 if swizzle else 1)
        bank_lists.append(occupied)
    assert bank_lists == [[1, 1, 1, 1], [1, 0, 3, 2]]
    for x0, banks, label in [(15, bank_lists[0], '4r + c'),
                              (260, bank_lists[1], '4r + (c XOR r)')]:
        d.text(x0, 334, 205, 18, label, bold=True)
        for bank in range(4):
            x = x0 + bank * 51
            d.box(x, 356, 47, 56, fill='#f3f4f5', stroke=LIGHT, rounded=False)
            d.text(x, 357, 47, 16, f'b{bank}', color=GRAY)
            values = [str(chosen[r]) for r in range(4) if banks[r] == bank]
            if values:
                d.box(x + 1, 376, 45, 32, ' '.join(values), fill=PALE[1] if len(values)>1 else PALE[0],
                      stroke=ORANGE if len(values)>1 else BLUE, size=13, rounded=False)
        d.text(x0, 414, 205, 18, '4 reads contend' if len(set(banks)) == 1 else '4 banks serve 4 reads', bold=True)
    d.line([(224, 386), (254, 386)], color=GREEN, arrow=True, width=2)
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
    d = Diagram('semantic_transport', 480, 274)
    d.title('Bit reversal preserves the pairing relation')
    permutation = [reverse3(i) for i in range(8)]
    assert permutation == [0, 4, 2, 6, 1, 5, 3, 7]
    assert sorted(permutation) == list(range(8))
    assert all(reverse3(reverse3(i)) == i for i in range(8))
    assert all(reverse3(i ^ (1 << q)) == (reverse3(i) ^ (1 << (2 - q)))
               for i in range(8) for q in range(3))

    d.text(8, 35, 197, 20, 'q = 0: adjacent pairs', color=BLUE, bold=True)
    d.text(275, 35, 197, 20, 'q = 2: stride-four pairs', color=GREEN, bold=True)
    d.text(198, 35, 83, 20, 'rev₃', color=GRAY, bold=True)
    # Two complete input/output columns expose both directed dependencies.
    # Mapped columns are indexed by physical label, so their long edges show stride four.
    for x0, bit, color in [(34, 0, BLUE), (321, 2, GREEN)]:
        for i in range(8):
            y = 68 + i * 16
            d.text(x0 - 27, y - 8, 22, 16, str(i), size=14)
            d.text(x0 + 101, y - 8, 22, 16, str(i), size=14)
            d.line([(x0,y),(x0+95,y)], color=LIGHT, arrow=True, width=.8)
            d.line([(x0,y),(x0+95,68+(i^(1<<bit))*16)], color=color, arrow=True, width=1)
        for x in (x0, x0+95):
            for i in range(8):
                d.dot(x, 68+i*16, radius=2, color=color)
    d.line([(164,108),(296,108)], color=GRAY, arrow=True, width=2)
    d.text(166, 64, 128, 38, '(0, 1) → (0, 4)\n(2, 3) → (2, 6)', size=14)
    d.text(165, 120, 130, 40, '(4, 5) → (1, 5)\n(6, 7) → (3, 7)', size=14)
    d.text(163, 164, 134, 28, 'input → output', size=14, color=GRAY)
    d.line([(16,195),(464,195)], color=LIGHT)
    d.text(12, 199, 456, 20, 'Carry the ordered pair, coefficient w, and direction:', bold=True)
    d.text(12, 222, 456, 20, 'CT: (a, b) → (a + wb, a − wb), before and after rev₃', size=14)
    d.text(12, 246, 456, 20, 'Whole sequence: (0, 1, 2) → (2, 1, 0); labels travel with stages.', size=14)
    return d.save()


if __name__ == '__main__':
    for generate in (hierarchy, bank_layout, semantic_transport):
        print(generate())
