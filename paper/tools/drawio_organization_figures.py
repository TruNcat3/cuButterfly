"""Build native draw.io organization, roofline and readiness witnesses.

These are analytical constructions. No experimental records are read.
"""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE


def organization_tradeoffs():
    d = Diagram('organization_tradeoffs', 1000, 340)
    for panel, (name, depths, color, fill) in enumerate([
        ('(a) Stage-wise', [1, 1, 1, 1], BLUE, PALE[0]),
        ('(b) Two bands', [2, 2], GREEN, PALE[2]),
        ('(c) Whole transform', [4], PURPLE, PALE[3]),
    ]):
        origin = 10 + panel * 333
        if panel: d.line([(origin-10, 16), (origin-10, 298)], color=LIGHT)
        d.text(origin, 10, 313, 25, name, size=16, bold=True)
        d.text(origin, 37, 313, 22, 'Bands ' + ' + '.join(map(str, depths)), color=GRAY)
        start, x, unit = 0, origin+15, 72
        for depth in depths:
            bits = range(start, start+depth)
            mask = sum(1 << q for q in bits)
            groups = [[i for i in range(16) if i & ~mask == base]
                      for base in range(16) if base & mask == 0]
            ys = {i: 82 + r*10 for r,i in enumerate(sum(groups, []))}
            x0, x1 = x+7, x+depth*unit-7
            xs = [x0+(x1-x0)*k/depth for k in range(depth+1)]
            for group in groups:
                lo, hi = min(ys[i] for i in group), max(ys[i] for i in group)
                d.box(x0-4, lo-4, x1-x0+8, hi-lo+8, fill=fill,
                      stroke=color, rounded=False, width=.9)
                assert all(i ^ (1 << q) in group for i in group for q in bits)
            for k, bit in enumerate(bits):
                d.text((xs[k]+xs[k+1])/2-16, 58, 32, 20, 'q'+str(bit), color=GRAY)
                for i in range(16):
                    d.line([(xs[k],ys[i]),(xs[k+1],ys[i])], color=LIGHT, width=.6)
                    d.line([(xs[k],ys[i]),(xs[k+1],ys[i^(1<<bit)])], color=color, width=.8)
            for xx in xs:
                for yy in ys.values(): d.dot(xx, yy, radius=1.8)
            d.line([(x,76),(x,238)], color=GRAY, width=2)
            d.line([(x+depth*unit,76),(x+depth*unit,238)], color=GRAY, width=2)
            x += depth*unit
            start += depth
        assert start == 4
        component = 2**depths[0]
        d.text(origin, 248, 313, 22, f'{component} values / component', bold=True, color=color)
        noun = 'component' if component == 16 else 'components'
        d.text(origin, 272, 313, 22, f'{16//component} independent {noun} / band')
        d.text(origin, 296, 313, 22, f'32 pairs  |  {32*len(depths)} value transfers')
    d.text(10, 320, 980, 18, 'Same N = 16 FWHT  |  colored boxes: closed components  |  gray cuts: materialization  |  analytical counts', color=GRAY)
    return d.save()


def roofline():
    d = Diagram('roofline', 480, 350)
    d.title('(a) Resource ceiling at boundary h')
    d.line([(62,154),(452,154)], color=INK, arrow=True)
    d.line([(62,154),(62,44)], color=INK, arrow=True)
    d.text(3,77,55,45,'Pair\nrate P',size=14)
    d.line([(63,153),(233,81),(435,81)], color=BLUE, width=2.6)
    d.line([(233,81),(300,53)], color=GRAY, dashed=True)
    d.text(86,76,113,22,'BW<sub>h</sub> × I<sub>h</sub>',color=BLUE,html=True)
    d.text(270,52,155,23,'Core ceiling P<sub>core</sub>',color=BLUE,html=True)
    d.text(65,164,385,24,'Boundary intensity I<sub>h</sub> (pair work / byte)',html=True)
    d.text(55,192,385,24,'P ≤ min(P<sub>core</sub>, BW<sub>h</sub> × I<sub>h</sub>)',bold=True,html=True)
    d.text(8,224,464,24,'(b) What is needed to approach the ceiling?',size=16,bold=True)
    for x,label,color,fill in [
        (12,'Ready\nsubgraphs',BLUE,PALE[0]),
        (171,'Resident\nresources',GREEN,PALE[2]),
        (330,'Valid reads /\nsynchronization',PURPLE,PALE[3])]:
        d.box(x,257,138,44,label,stroke=color,fill=fill)
        d.line([(x+69,302),(240,318)],color=color,arrow=True)
    d.text(8,324,464,20,'Low readiness or occupancy leaves the roof unused.',color=GRAY)
    return d.save()


def pipeline():
    d = Diagram('pipeline',480,366)
    d.title('Independent 16-point transforms A, B, C')
    d.text(10,37,460,22,'Each column waits for all four rows of its transform.',color=GRAY)
    x0, unit, rows, cols = 70, 24, 76, 155
    d.text(7,rows,58,26,'Rows',align='left',bold=True)
    d.text(7,cols,58,26,'Cols',align='left',bold=True)
    for j,(tag,color,fill) in enumerate(zip('ABC',[BLUE,GREEN,PURPLE],[PALE[0],PALE[2],PALE[3]])):
        for r in range(4):
            d.box(x0+(j*4+r)*unit,rows,unit,26,tag+str(r),fill=fill,stroke=color,rounded=False,width=1)
            d.box(x0+(j*4+r+4)*unit,cols,unit,26,tag+str(r),fill=fill,stroke=color,rounded=False,width=1)
        release=x0+(j+1)*4*unit
        d.line([(release,103),(release,151)],color=color,arrow=True,dashed=True)
        d.text(release+5,116,84,23,f'{tag}: all 4',color=color)
    for name,y in [('Slot 0',218),('Slot 1',260)]:
        d.text(7,y,61,27,name,align='left')
    for left,right,y,label,color,fill in [
        (0,8,218,'A live',BLUE,PALE[0]),
        (8,16,218,'C live',PURPLE,PALE[3]),
        (4,12,260,'B live',GREEN,PALE[2])]:
        d.box(x0+left*unit,y,(right-left)*unit,27,label,fill=fill,stroke=color,rounded=False)
        d.box(x0+right*unit-4,y+9,8,8,fill=color,stroke=color,rounded=False,shape='rhombus')
    # C's row production reuses A's slot only after A's final consumer.
    d.line([(x0+8*unit,181),(x0+8*unit,214)],color=ORANGE,arrow=True,width=1.8)
    d.text(10,290,460,23,"Slot 0 is reused after A's final column read.",color=GRAY)
    d.line([(x0,317),(454,317)],color=INK,arrow=True)
    for t in (0,4,8,12,16):d.text(x0+t*unit-14,319,28,20,str(t),color=GRAY)
    d.text(10,337,460,22,'Illustrative units; overlap requires available resources.',size=14,color=GRAY)
    return d.save()


if __name__ == '__main__':
    for fn in (organization_tradeoffs,roofline,pipeline): print(fn())
