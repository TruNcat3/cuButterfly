"""Native diagrams: a unified method and a concrete generated-plan witness."""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE, butterfly


def overview():
    d = Diagram('overview', 1000, 310)
    # Distinct inputs share a construction, not a universal arithmetic kernel.
    for x,w,title,detail,color,fill in [
        (12,294,'Operator + numeric contract','FFT / NTT / FWHT / zeta / structured',BLUE,PALE[0]),
        (328,320,'Workload domain','shape, batch, strides, placement',GREEN,PALE[2]),
        (670,318,'Hardware capabilities','local capacity, service rates, compatible cores',ORANGE,PALE[1])]:
        d.box(x,8,w,45,fill=fill,stroke=color,rounded=False)
        d.text(x+5,10,w-10,20,title,size=15,bold=True,color=color)
        d.text(x+5,31,w-10,18,detail,size=14)
    d.line([(159,61),(829,61)],color=GRAY)
    for x in [159,488,829]:
        d.line([(x,53),(x,61)],color=GRAY)
        d.line([(x,61),(x,70)],color=GRAY,arrow=True)
    d.text(12,72,296,23,'1  Derive regions and handoffs',size=15,bold=True)
    d.text(339,72,315,23,'2  Characterize legal organizations',size=15,bold=True)
    d.text(687,72,301,23,'3  Bind cores and execute tiles',size=15,bold=True)
    # Real N8 graph, two closed four-value prefix subgraphs, then q2 handoff.
    ys=[116+15*i+(12 if i>=4 else 0) for i in range(8)]
    for y,c,f,lab in [(104,BLUE,PALE[0],'A'),(176,ORANGE,PALE[1],'B')]:
        d.box(24,y,152,65,fill=f,stroke=c,rounded=False)
        d.text(27,y+2,18,16,lab,size=13,bold=True,color=c)
    butterfly(d,[45,105,165,280],ys,[0,1,2],node_radius=2)
    d.line([(191,101),(191,242)],color=GREEN,dashed=True)
    d.text(200,105,77,18,'handoff',size=13,color=GREEN)
    d.text(12,246,294,19,'closed regions -> predecessor sets',size=14,color=BLUE)
    d.line([(305,177),(326,177)],color=INK,arrow=True)
    # Two orthogonal dimensions, each may use space or time. No made-up bars.
    d.text(350,104,90,20,'Dimension',size=13,color=GRAY)
    d.text(451,104,88,20,'Space',size=14,bold=True,color=GREEN)
    d.text(552,104,88,20,'Time',size=14,bold=True,color=PURPLE)
    for y,name in [(132,'Stages'),(177,'Data tiles')]:
        d.text(343,y+8,88,22,name,size=14,bold=True)
        for xx,color,fill in [(450,GREEN,PALE[2]),(552,PURPLE,PALE[3])]:
            for j in range(3):
                d.box(xx+j*27,y,22,24,str(j),fill=fill,stroke=color,rounded=False,size=13)
            if xx==552: d.line([(xx,y+32),(xx+78,y+32)],color=color,arrow=True)
    d.box(340,224,310,42,'state / traffic / ready work\nchecked against capacity and service',fill='#ffffff',stroke=ORANGE,rounded=False,size=14)
    d.line([(655,177),(680,177)],color=INK,arrow=True)
    # Nested scope has bounded tiles; outer data is not all resident.
    d.box(694,107,281,128,fill='#ffffff',stroke=GRAY,rounded=False)
    d.text(700,110,269,20,'External handoff buffers',size=14,color=GRAY)
    for row in range(2):
        y=140+42*row
        d.box(706,y,257,35,fill=PALE[2],stroke=GREEN,rounded=False)
        d.text(711,y+6,72,22,'worker '+str(row),size=13)
        for j in range(3):
            d.box(790+j*52,y+6,43,23,'tile '+str(2*j+row),fill=PALE[row],stroke=[BLUE,ORANGE][row],rounded=False,size=13)
    # Expand one tile into its local pair graph rather than add a prose box.
    butterfly(d,[714,746,778],[243,251,259,267],[0,1],node_radius=1.2)
    d.line([(788,254),(810,254)],color=GREEN,arrow=True)
    d.text(819,239,154,31,'local core +\noutput placement',size=13,color=BLUE)
    d.box(12,279,976,23,'Same construction across storage scopes; one band and expansion factor 1 remain legal endpoints.',fill='#f3f5f7',stroke='none',rounded=False,size=14)
    return d.save()


def system_architecture():
    # One column: the overview now owns the common architecture. This figure
    # makes a selected plan concrete without repeating its input boxes.
    d=Diagram('system_architecture',480,292)
    d.title('A selected plan connects construction to execution')
    d.text(12,36,210,20,'Preparation (outside timing)',size=14,bold=True,color=BLUE)
    d.text(265,36,203,20,'Reusable execution',size=14,bold=True,color=GREEN)
    for y,title,detail,color,fill in [
        (65,'Static profile','capacity + local service',ORANGE,PALE[1]),
        (120,'Construct and check','bands + core + layout',BLUE,PALE[0]),
        (175,'Confirm complete plan','correctness + timing',PURPLE,PALE[3])]:
        d.box(17,y,198,42,fill=fill,stroke=color,rounded=False)
        d.text(20,y+2,192,19,title,size=14,bold=True,color=color)
        d.text(20,y+22,192,17,detail,size=13)
        if y<175: d.line([(115,y+42),(115,y+55)],color=GRAY,arrow=True)
    d.line([(216,194),(248,194),(248,85),(270,85)],color=BLUE,arrow=True)
    for j,(label,col,fill) in enumerate([('input',BLUE,PALE[0]),('subgraph A | B',GREEN,PALE[2]),('boundary layout',PURPLE,PALE[3]),('successor tiles',GREEN,PALE[2]),('output',ORANGE,PALE[1])]):
        y=67+j*32
        d.box(275,y,185,25,label,fill=fill,stroke=col,rounded=False,size=14)
        if j<4: d.line([(367,y+25),(367,y+32)],color=GRAY,arrow=True)
    d.line([(12,231),(468,231)],color=LIGHT)
    d.text(12,236,456,18,'Example binding: bands {0,1} -> {2}; four-value local core',size=13,bold=True)
    for i in range(4):
        x=90+i*75
        d.box(x,259,29,23,str(i),fill=PALE[0],stroke=BLUE,rounded=False,size=13)
        d.box(x+29,259,29,23,str(i+4),fill=PALE[1],stroke=ORANGE,rounded=False,size=13)
    d.text(8,260,76,20,'next pairs',size=13,color=GREEN)
    return d.save()

if __name__=='__main__':
    for make in (overview,system_architecture): print(make())
