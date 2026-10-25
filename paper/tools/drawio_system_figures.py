"""Editable overview and generation/selection diagrams for the paper."""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE, butterfly


def overview():
    d = Diagram('overview', 1000, 370)
    for x in (320, 680):
        d.line([(x, 15), (x, 355)], color=LIGHT, width=1)
    d.text(10, 12, 300, 25, '(a) Labeled dependencies', size=16, bold=True)
    d.text(330, 12, 340, 25, '(b) Bounded bands and fan-in', size=16, bold=True)
    d.text(690, 12, 300, 25, '(c) Recursive hardware placement', size=16, bold=True)

    # Arithmetic contracts remain attached to the common pairing envelope.
    arithmetic = [('FFT', 'a +/- twiddle * b', BLUE),
                  ('NTT', 'a +/- twiddle * b (mod q)', ORANGE),
                  ('FWHT', 'a + b, a - b', GREEN),
                  ('zeta', 'a, a + b', PURPLE),
                  ('structured', 'M [a, b]', INK)]
    for i, (name, operation, color) in enumerate(arithmetic):
        y = 47 + 23 * i
        d.dot(25, y + 10, radius=3, color=color)
        d.text(38, y, 83, 20, name, color=color, bold=True, align='left')
        d.text(123, y, 184, 20, operation, align='left')
    d.line([(160, 164), (160, 185)], color=GRAY, arrow=True)
    xs = [64, 131, 198, 265]
    ys = [211 + 16 * i for i in range(8)]
    butterfly(d, xs, ys, [0, 1, 2], color=BLUE, node_radius=2.3)
    for x, label in zip(xs, ['input', 'bit 0', 'bit 1', 'bit 2']):
        d.text(x - 27, 187, 54, 20, label, color=GRAY)
    d.text(17, 330, 289, 25, 'N = 8; common pairing envelope', color=GRAY)

    # Bits 0 and 1 close over the two components obtained by fixing bit 2.
    d.text(334, 47, 151, 35, 'band {0, 1}', color=BLUE, bold=True)
    d.text(567, 47, 104, 35, 'band {2}', color=GREEN, bold=True)
    for y, color, fill, label in [(91, BLUE, PALE[0], 'bit 2 = 0'),
                                  (208, ORANGE, PALE[1], 'bit 2 = 1')]:
        d.box(341, y, 146, 104, fill=fill, stroke=color, rounded=False)
        d.text(349, y + 4, 130, 21, label, color=color, bold=True)
        butterfly(d, [359, 415, 471], [y + 36 + 17 * i for i in range(4)],
                  [0, 1], color=color, node_radius=2.3)
    d.text(339, 315, 150, 22, '2 closed subgraphs', color=BLUE)
    d.text(339, 338, 150, 20, '4 values each', color=GRAY)
    for i in range(4):
        y = 95 + 54 * i
        d.box(571, y, 94, 44, fill=PALE[2], stroke=GREEN, rounded=False)
        d.text(572, y + 2, 92, 18, f'pair ({i}, {i + 4})', color=GREEN)
        # Both components supply one value to each bit-2 consumer.
        d.line([(471, 127 + 17 * i), (571, y + 26)], color=BLUE, width=1)
        d.line([(471, 244 + 17 * i), (571, y + 37)], color=ORANGE, width=1)
        butterfly(d, [587, 650], [y + 26, y + 37], [0], color=GREEN,
                  node_radius=2)
    d.text(497, 83, 65, 23, 'handoff', color=GRAY)
    d.text(566, 315, 106, 22, '4 consumers', color=GREEN)
    d.text(560, 338, 113, 20, '2 values each', color=GRAY)

    # Boxes are execution groups, not a nesting claim about address spaces.
    d.box(699, 52, 282, 245, stroke=PURPLE, fill='#ffffff', rounded=False)
    d.text(710, 59, 260, 23, 'Device work: global handoffs', color=PURPLE, bold=True)
    d.box(715, 94, 250, 151, stroke=ORANGE, fill=PALE[1], rounded=False)
    d.text(723, 101, 234, 23, 'Block group: shared exchange', color=ORANGE, bold=True)
    for y, name in [(137, 'Thread group A'), (188, 'Thread group B')]:
        d.box(729, y, 222, 40, fill='#ffffff', stroke=BLUE, rounded=False)
        d.text(737, y + 3, 130, 18, name, color=BLUE, align='left')
        d.text(737, y + 21, 130, 16, 'register residency', color=GRAY, align='left')
        butterfly(d, [887, 933], [y + 11, y + 29], [0], color=BLUE)
    d.line([(840, 177), (840, 188)], color=ORANGE, arrow=True)
    d.text(713, 259, 252, 26, 'Repeat local work + explicit exchange', color=PURPLE)
    for i, x in enumerate([710, 800, 890]):
        d.box(x, 312, 71, 28, f'tile {i}', fill=PALE[2], stroke=GREEN, rounded=False)
    d.text(701, 342, 279, 26, 'Independent tiles reuse resources', color=GRAY)
    return d.save()


def system_architecture():
    d = Diagram('system_architecture', 1000, 390)
    d.text(20, 9, 960, 25, 'Automatic realization: finite search, confirmed execution plan',
           size=16, bold=True)
    inputs = [(35, 180, 'Semantic contract', 'arithmetic, precision, order', BLUE, PALE[0]),
              (245, 180, 'Workload', 'N, batch, shape, placement', GREEN, PALE[2]),
              (455, 180, 'Target + core catalog', 'capacity, compatible cores', ORANGE, PALE[1]),
              (730, 240, 'Measured services', 'core, tile, layout, resources', PURPLE, PALE[3])]
    for x, w, title, detail, color, fill in inputs:
        d.box(x, 48, w, 55, fill=fill, stroke=color)
        d.text(x + 5, 53, w - 10, 22, title, bold=True, color=color)
        d.text(x + 5, 77, w - 10, 20, detail)
    for x in [125, 335, 545]:
        d.line([(x, 103), (x, 117)], color=GRAY)
    d.line([(125, 117), (545, 117)], color=GRAY)
    d.line([(142, 117), (142, 151)], color=GRAY, arrow=True)
    d.line([(850, 103), (850, 126), (650, 126), (650, 151)],
           color=PURPLE, arrow=True)

    d.text(260, 124, 355, 20, 'Preparation: build, validate and select',
           color=GRAY, align='left')
    stages = [(35, 215, 'Legal candidates', 'subgraphs + handoffs\ncores, layouts, schedules', GREEN, PALE[2]),
              (290, 210, 'Compile + validate', 'resource / dependency checks\nnumerical contract', ORANGE, PALE[1]),
              (540, 220, 'Prioritize + measure', 'service model, finite budget\ncomplete-plan confirmation', PURPLE, PALE[3]),
              (810, 160, 'Selected plan / cache', 'fastest confirmed\nmatching contract', BLUE, PALE[0])]
    for x, w, title, detail, color, fill in stages:
        d.box(x, 151, w, 65, fill=fill, stroke=color)
        d.text(x + 5, 155, w - 10, 21, title, color=color, bold=True)
        d.text(x + 6, 178, w - 12, 34, detail)
    for x1, x2 in [(250, 290), (500, 540), (760, 810)]:
        d.line([(x1, 184), (x2, 184)], color=INK, arrow=True)

    d.box(290, 230, 210, 39, 'Unsupported / invalid\nrecord explicit gap',
          fill='#ffffff', stroke=ORANGE, rounded=False)
    d.line([(395, 216), (395, 230)], color=ORANGE, arrow=True)
    # Only measurement feedback is dashed. The gap branch terminates explicitly.
    d.line([(650, 216), (650, 273), (142, 273), (142, 216)],
           color=PURPLE, dashed=True, arrow=True)
    d.text(155, 230, 128, 38, 'Measurements\nrefine candidates', color=PURPLE)

    d.line([(20, 288), (980, 288)], color=LIGHT, width=1)
    d.text(27, 293, 750, 24, 'Warm device path: adaptation, execution, handoffs and synchronization',
           size=16, bold=True, color=ORANGE, align='left')
    runtime = [(35, 95, 'Input', BLUE), (170, 120, 'Adapt', BLUE),
               (330, 165, 'Resident groups', GREEN),
               (535, 190, 'Handoffs + sync', PURPLE), (765, 130, 'Output', ORANGE)]
    for x, w, title, color in runtime:
        d.box(x, 328, w, 39, title, stroke=color, fill='#ffffff', bold=True)
    for x1, x2 in [(130, 170), (290, 330), (495, 535), (725, 765)]:
        d.line([(x1, 347), (x2, 347)], color=INK, arrow=True)
    # Dispatch the whole plan at adaptation; a cache hit does not bypass it.
    d.line([(970, 184), (985, 184), (985, 321), (230, 321), (230, 328)],
           color=BLUE, arrow=True)
    d.text(821, 243, 150, 38, 'Matching cache hit:\ndispatch complete plan', color=BLUE)
    d.line([(35, 375), (895, 375)], color=ORANGE, width=1.5)
    d.line([(35, 370), (35, 380)], color=ORANGE, width=1.5)
    d.line([(895, 370), (895, 380)], color=ORANGE, width=1.5)
    return d.save()


if __name__ == '__main__':
    for make in (overview, system_architecture):
        print(make())
