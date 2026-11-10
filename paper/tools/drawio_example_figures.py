#!/usr/bin/env python3
"""Native editable analytical examples; no images or performance samples.

Run with /home/wt/yes/envs/cubutterfly/bin/python. All maps are exact toy
examples from the method and execution sections, not hardware guarantees.
"""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, GRAY, LIGHT, PALE


ROW_COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def hierarchy():
    from drawio_boundary_detail import boundary_detail
    return boundary_detail()


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


def semantic_transport():
    from drawio_semantic_detail import semantic_transport as detailed_transport
    return detailed_transport()


if __name__ == '__main__':
    for generate in (hierarchy, bank_layout, semantic_transport):
        print(generate())
