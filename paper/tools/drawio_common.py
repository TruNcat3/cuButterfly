"""Native editable mxGraph primitives for the paper's analytical figures.

Coordinates are diagram pixels. A 1000 px full-width or 480 px column canvas
maps a 14 px label to about 7.2 pt at the manuscript's inclusion width.
No SVG/image embeds: each vertex, label, and connection is a draw.io cell.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

BLUE, GREEN, ORANGE, PURPLE = '#246a9e', '#30856d', '#b86b14', '#785690'
INK, GRAY, LIGHT = '#243341', '#65717c', '#d5dde3'
PALE = ['#e5f0f7', '#fbefdb', '#e6f2ed', '#efe8f5']
OUT = Path(__file__).resolve().parents[1] / 'figures' / 'drawio'


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.file = ET.Element('mxfile', host='app.diagrams.net', type='device',
                               version='32.4.1', agent='cuButterfly figure sources')
        page = ET.SubElement(self.file, 'diagram', id=name, name=name)
        graph = ET.SubElement(page, 'mxGraphModel', dx=str(width), dy=str(height),
                              grid='1', gridSize='10', guides='1', tooltips='1',
                              connect='1', arrows='1', fold='1', page='1',
                              pageScale='1', pageWidth=str(width), pageHeight=str(height),
                              math='0', shadow='0', background='#ffffff')
        self.root = ET.SubElement(graph, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.next_id = 2
        # An editable page-sized background fixes all export aspect ratios.
        self.box(0, 0, width, height, fill='#ffffff', stroke='none', rounded=False)

    def _cell(self, value='', style='', **attrs):
        cell = ET.SubElement(self.root, 'mxCell', id=str(self.next_id),
                             value=value, style=style, parent='1', **attrs)
        self.next_id += 1
        return cell

    def box(self, x, y, w, h, label='', fill='#ffffff', stroke=BLUE,
            size=14, bold=False, rounded=True, dashed=False, align='center',
            width=1.3, shape='rectangle'):
        style = (f'shape={shape};rounded={int(rounded)};arcSize=10;whiteSpace=wrap;'
                 f'html=0;fillColor={fill};strokeColor={stroke};strokeWidth={width};'
                 f'fontFamily=Helvetica;fontSize={size};fontColor={INK};'
                 f'fontStyle={int(bold)};align={align};verticalAlign=middle;'
                 f'spacing=4;dashed={int(dashed)};')
        cell = self._cell(label, style, vertex='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w),
                      height=str(h), **{'as': 'geometry'})
        return cell.get('id')

    def text(self, x, y, w, h, label, size=14, color=INK, bold=False, align='center', html=False):
        style = (f'text;html={int(html)};whiteSpace=wrap;overflow=hidden;fillColor=none;'
                 f'strokeColor=none;fontFamily=Helvetica;fontSize={size};'
                 f'fontColor={color};fontStyle={int(bold)};align={align};'
                 'verticalAlign=middle;spacing=0;')
        cell = self._cell(label, style, vertex='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w),
                      height=str(h), **{'as': 'geometry'})
        return cell.get('id')

    def line(self, points, color=GRAY, width=1.3, dashed=False, arrow=False,
             label='', source=None, target=None):
        style = (f'edgeStyle=none;rounded=0;html=0;strokeColor={color};'
                 f'strokeWidth={width};dashed={int(dashed)};'
                 f'endArrow={"block" if arrow else "none"};endFill=1;endSize=6;'
                 f'fontFamily=Helvetica;fontSize=14;fontColor={color};')
        attrs = {'edge': '1'}
        if source: attrs['source'] = source
        if target: attrs['target'] = target
        cell = self._cell(label, style, **attrs)
        geo = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        for pt, tag in ((points[0], 'sourcePoint'), (points[-1], 'targetPoint')):
            ET.SubElement(geo, 'mxPoint', x=str(pt[0]), y=str(pt[1]), **{'as': tag})
        if len(points) > 2:
            a = ET.SubElement(geo, 'Array', **{'as': 'points'})
            for pt in points[1:-1]: ET.SubElement(a, 'mxPoint', x=str(pt[0]), y=str(pt[1]))
        return cell.get('id')

    def dot(self, x, y, radius=2.5, color=INK):
        return self.box(x-radius, y-radius, radius*2, radius*2,
                        fill=color, stroke=color, rounded=False, width=.5, shape='ellipse')

    def title(self, label):
        self.text(12, 8, self.width-24, 25, label, size=16, bold=True)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / (self.name + '.drawio')
        ET.indent(self.file, space='  ')
        ET.ElementTree(self.file).write(path, encoding='utf-8', xml_declaration=True)
        return path


def butterfly(d, xs, ys, bits, color=BLUE, node_radius=2.2):
    """True full pairing envelope; ys indexed by logical value."""
    assert len(xs) == len(bits) + 1
    for k, bit in enumerate(bits):
        for i, y in enumerate(ys):
            d.line([(xs[k], y), (xs[k+1], y)], color=LIGHT, width=.8)
            d.line([(xs[k], y), (xs[k+1], ys[i ^ (1 << bit)])], color=color, width=.9)
    for x in xs:
        for y in ys: d.dot(x, y, radius=node_radius)
