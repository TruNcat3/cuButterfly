"""Build native draw.io organization, roofline and readiness witnesses.

The organization and pipeline pages are analytical constructions. The Roofline
page reads only the frozen derived case table
``generated/roofline_cases_20261029.csv``; it never reads a live campaign.
The symbolic roof remains schematic and is not fitted to those observations.
"""
import csv
from pathlib import Path
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE

ROOT = Path(__file__).resolve().parents[1]


def organization_tradeoffs():
    d=Diagram('organization_tradeoffs',480,414)
    d.title('One dependency graph, multiple legal organizations')
    xs=[42,133,224,315,406]; ys=[57+i*7 for i in range(16)]
    for j,lab in enumerate(['input','q0','q1','q2','q3']):
        d.text(xs[j]-22,33,44,19,lab,size=13,color=GRAY)
    # Four dependency-closed prefixes: each contains four logical values.
    for g in range(4):
        d.box(32,ys[4*g]-4,198,29,fill=PALE[g],stroke=[BLUE,ORANGE,GREEN,PURPLE][g],rounded=False,width=.6)
    for q in range(4):
        for i in range(16):
            d.line([(xs[q],ys[i]),(xs[q+1],ys[i])],color=LIGHT,width=.65)
            d.line([(xs[q],ys[i]),(xs[q+1],ys[i^(1<<q)])],color=BLUE if q<2 else GREEN,width=.7)
    for x in xs:
        for y in ys:d.dot(x,y,radius=1.5)
    d.line([(244,51),(244,168)],color=ORANGE,dashed=True)
    d.text(249,170,206,18,'partners change across the boundary',size=12,color=GREEN)
    d.text(8,170,232,18,'four closed four-value regions',size=12,color=BLUE)
    d.line([(8,194),(472,194)],color=LIGHT)
    d.text(8,198,120,18,'Organization',size=13,bold=True,align='left')
    d.text(137,198,325,18,'Stage bands; vertical bars are materialized cuts',size=12,color=GRAY)
    for row,(name,depths,color,fill) in enumerate([
        ('Stagewise',[1,1,1,1],BLUE,PALE[0]),
        ('Two bands',[2,2],GREEN,PALE[2]),
        ('Whole',[4],PURPLE,PALE[3])]):
        y=225+row*52
        d.text(8,y,115,23,name,size=14,bold=True,align='left')
        start=0
        for depth in depths:
            x=139+start*81
            d.box(x,y,depth*81-8,22,fill=fill,stroke=color,rounded=False)
            for q in range(depth):d.text(x+q*81,y,73,22,'q'+str(start+q),size=13,color=color)
            d.line([(x-4,y-2),(x-4,y+24)],color=INK,width=1.4)
            start+=depth
        d.line([(459,y-2),(459,y+24)],color=INK,width=1.4)
        assert start==4
        local=2**depths[0];regions=16//local;transfers=32*len(depths)
        assert local*regions==16 and 16*sum(depths)//2==32
        region_text='region' if regions==1 else 'regions'
        d.text(8,y+25,464,19,f'{local} local values | {regions} {region_text}/band | {transfers} value transfers',size=13,align='left')
    d.text(8,384,464,21,'32 pair updates in every case; transfers include input/output.',size=13,bold=True)
    return d.save()


def roofline():
    d = Diagram('roofline', 480, 292)
    d.title('Resource roof, organization gap, and execution service')
    # The roof is symbolic. Measured points use the same pair-rate and
    # ideal-intensity definitions as the surrounding text; the y scale is
    # fixed by the selected derived observations, not by a hardware peak.
    axis_left, axis_right, axis_top, axis_bottom = 62, 462, 43, 170
    d.line([(axis_left,axis_bottom),(axis_right,axis_bottom)], color=INK, arrow=True)
    d.line([(axis_left,axis_bottom),(axis_left,axis_top)], color=INK, arrow=True)
    d.text(0,35,61,32,'Gpairs/s',size=13)
    d.line([(62,170),(245,48),(452,48)], color=BLUE, width=2.2)
    d.text(260,24,202,22,'Symbolic resource roof',size=14,color=BLUE)
    d.text(75,46,135,23,'BW<sub>h</sub> x I<sub>h</sub>',size=15,color=BLUE,html=True)
    d.line([(245,48),(245,170)], color=LIGHT, width=1,dashed=True)
    d.text(244,62,106,19,'execution gap',size=13,color=GRAY,align='left')

    cases_path = ROOT / 'generated' / 'roofline_cases_20261029.csv'
    with cases_path.open(newline='') as handle:
        cases = list(csv.DictReader(handle))
    required = {'F0','F2','F4','H1','H16','N2','N4'}
    assert {row['case'] for row in cases} == required
    assert all(int(row['materialized_groups']) == 2 for row in cases)
    xmax, ymax = .4, 160.0
    def px(value): return axis_left + float(value) / xmax * (axis_right-axis_left)
    def py(value): return axis_bottom - float(value) / ymax * (axis_bottom-axis_top)
    styles = {'FFT': (BLUE, 'ellipse'), 'FWHT': (GREEN, 'triangle'), 'NTT': (PURPLE, 'rhombus')}
    label_offsets = {
        'F0': (8, 4), 'F2': (8, -24), 'F4': (-43, -8),
        'H1': (8, -11), 'H16': (8, -24),
        'N2': (8, -11), 'N4': (8, 7),
    }
    for row in cases:
        x, y = px(row['ideal_intensity_pairs_per_byte']), py(row['pair_rate_gpairs_per_s'])
        color, shape = styles[row['operator']]
        if shape == 'ellipse':
            d.dot(x, y, radius=4, color=color)
        else:
            d.box(x-4, y-4, 8, 8, fill=color, stroke=color, rounded=False,
                  width=.7, shape=shape)
        dx, dy = label_offsets[row['case']]
        if row['operator']=='FFT':
            d.line([(x,y),(x+dx+(34 if dx<0 else 0),y+dy+8)],color=color,width=.7)
        d.text(min(x+dx, 425), max(min(y+dy, axis_bottom-16), axis_top+5),
               38, 16, row['case'], size=11, color=color, align='left')
    for rate in (0,40,80,120):
        y=py(rate)
        d.line([(axis_left-4,y),(axis_left,y)],color=GRAY,width=.8)
        d.text(20,y-8,33,16,str(rate),size=11,color=GRAY)
    for value, label in ((0,'0'),(.16,'.16'),(.31,'.31'),(.4,'.4')):
        x = px(value)
        d.line([(x,axis_bottom),(x,axis_bottom+4)], color=GRAY, width=.8)
        d.text(x-16, axis_bottom+5, 32, 16, label, size=11, color=GRAY)
    d.text(64,187,397,20,'Ideal boundary intensity I<sub>h</sub> (pairs / byte)',size=14,html=True)
    d.text(8,211,464,18,'Derived case medians: F=FFT, H=FWHT, N=NTT.',size=12,color=GRAY,align='left')
    d.text(8,229,464,18,'Points are measured; roof is schematic, not a peak.',size=12,color=GRAY,align='left')
    # Categorical causal legend. Box sizes never encode measured resource use.
    d.line([(8,252),(472,252)],color=LIGHT,width=.8)
    for i,(label,col,fill) in enumerate([
        ('Transactions',ORANGE,PALE[1]),('Synchronization',PURPLE,PALE[3]),('Live state / work',GREEN,PALE[2])]):
        d.box(8+i*156,260,150,24,label,fill=fill,stroke=col,rounded=False,size=13)
    return d.save()


def pipeline():
    d = Diagram('pipeline',480,414)
    d.title('Tile-subgraph readiness, handoff, and buffer reuse')
    d.text(10,34,460,19,'A consumer starts after its predecessor packet is ready.',color=GRAY)
    x0, unit, rows, cols = 70, 24, 59, 113
    d.text(7,rows,58,26,'Rows',align='left',bold=True)
    d.text(7,cols,58,26,'Cols',align='left',bold=True)
    for j,(tag,color,fill) in enumerate(zip('ABC',[BLUE,GREEN,PURPLE],[PALE[0],PALE[2],PALE[3]])):
        for r in range(4):
            d.box(x0+(j*4+r)*unit,rows,unit,26,tag+str(r),fill=fill,stroke=color,rounded=False,width=1)
            d.box(x0+(j*4+r+4)*unit,cols,unit,26,tag+str(r),fill=fill,stroke=color,rounded=False,width=1)
        release=x0+(j+1)*4*unit
        d.line([(release,85),(release,109)],color=color,arrow=True,dashed=True)
        d.text(release+5,86,84,23,f'{tag}: all 4',color=color)
    for name,y in [('Slot 0',158),('Slot 1',189)]:
        d.text(7,y,61,27,name,align='left')
    for left,right,y,label,color,fill in [
        (0,8,158,'A live',BLUE,PALE[0]),
        (8,16,158,'C live',PURPLE,PALE[3]),
        (4,12,189,'B live',GREEN,PALE[2])]:
        d.box(x0+left*unit,y,(right-left)*unit,27,label,fill=fill,stroke=color,rounded=False)
        d.box(x0+right*unit-4,y+9,8,8,fill=color,stroke=color,rounded=False,shape='rhombus')
    # C's row production reuses A's slot only after A's final consumer.
    d.line([(x0+8*unit,139),(x0+8*unit,154)],color=ORANGE,arrow=True,width=1.8)
    d.line([(x0,228),(454,228)],color=INK,arrow=True)
    for t in (0,4,8,12,16):d.text(x0+t*unit-14,229,28,18,str(t),color=GRAY)
    d.text(10,251,460,18,'Illustrative time; diamonds release buffer slots.',size=14,color=GRAY)
    d.line([(10,274),(470,274)],color=LIGHT)
    d.text(10,279,460,21,'Within one band: independent tiles, two buffer slots',size=14,bold=True)
    origin,step=116,45
    stages=[('Load',BLUE,PALE[0]),('Compute',GREEN,PALE[2]),('Store',ORANGE,PALE[1])]
    starts=[0,1,3,4]
    for r,(label,color,fill) in enumerate(stages):
        y=309+r*26
        d.text(10,y,100,22,label,size=13,align='left')
        for tile,start in enumerate(starts):
            x=origin+(start+r)*step
            d.box(x,y,step-3,22,'t'+str(tile),fill=fill,stroke=color,rounded=False,size=13)
    # Each buffer is held through the store; t2 reuses t0's slot at 3.
    assert starts[2]>=starts[0]+3 and starts[3]>=starts[1]+3
    d.text(10,391,460,18,'Illustrative service lanes; overlap requires independent resources.',size=12,color=GRAY)
    return d.save()


if __name__ == '__main__':
    for fn in (organization_tradeoffs,roofline,pipeline): print(fn())
