"""Native diagrams: a unified method and a concrete generated-plan witness."""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE, butterfly


def overview():
    from walkthrough_example import checked_example
    rows, columns, schedule = checked_example()
    d = Diagram('overview', 1000, 392)
    for x,w,title,detail,color,fill in [
        (10,305,'Operator contract','arithmetic, precision, direction, order',BLUE,PALE[0]),
        (333,309,'Workload domain','shape, batch, strides, placement',GREEN,PALE[2]),
        (660,330,'Hardware capabilities','capacity, core support, service rates',ORANGE,PALE[1])]:
        d.box(x,6,w,42,fill=fill,stroke=color,rounded=False)
        d.text(x+3,8,w-6,19,title,size=15,bold=True,color=color)
        d.text(x+3,28,w-6,17,detail,size=14)
    # Shared inputs feed every stage of the construction.
    d.line([(104,56),(896,56)],color=GRAY,width=.9)
    for x in (162,487,825):d.line([(x,48),(x,56)],color=GRAY,width=.9)
    for k in range(5):d.line([(104+198*k,56),(104+198*k,65)],color=GRAY,width=.9,arrow=True)
    # The abstract row and case row use the same five steps and symbols.
    titles=['1  Close regions','2  Derive handoffs','3  Match resources',
            '4  Place values','5  Execute subgraphs']
    details=['stage bits → local domains','intersections → predecessors',
             'space/time → resource demand','ownership → bank + offset',
             'ready → compute → publish']
    for k in range(5):
        x=10+198*k
        d.box(x,65,188,55,fill='#f3f5f7',stroke=LIGHT,rounded=False)
        d.text(x+2,69,184,21,titles[k],size=15,bold=True)
        d.text(x+2,94,184,20,details[k],size=13)
        if k<4: d.line([(x+188,91),(x+198,91)],color=GRAY,arrow=True)
        d.line([(x+94,120),(x+94,142)],color=GRAY,arrow=True)
    for k in range(4):d.line([(204+198*k,150),(204+198*k,340)],color=LIGHT)
    # Step 1: full N=16 graph, with closed prefix subgraphs and a suffix column.
    d.text(12,145,183,20,'Hadamard, N = 16',size=14,bold=True)
    ys=[185+7*i+4*(i//4) for i in range(16)]
    xs=[30,68,106,144,182]
    for r in range(4):
        d.box(24,ys[4*r]-4,88,29,fill=PALE[r],stroke=[BLUE,ORANGE,GREEN,PURPLE][r],rounded=False,width=.7)
        d.text(5,ys[4*r]+2,18,18,'P'+str(r),size=12,color=[BLUE,ORANGE,GREEN,PURPLE][r])
    butterfly(d,xs,ys,[0,1,2,3],color=LIGHT,node_radius=1.2)
    for k in range(4):
        d.text(xs[k]+3,166,32,16,'q'+str(k),size=12,color=GRAY)
        for i,y in enumerate(ys):
            if k<2 or i%4==1:
                color=[BLUE,ORANGE,GREEN,PURPLE][i//4] if k<2 else ORANGE
                for j in (i,i^(1<<k)):
                    d.line([(xs[k],y),(xs[k+1],ys[j])],color=color,width=.9)
    d.line([(116,179),(116,310)],color=GREEN,dashed=True,width=.8)
    d.text(13,312,180,29,'Pairing bits {0,1} → {2,3}',size=13)
    # Step 2: intersections are real singleton values, not arbitrary arrows.
    x0=210
    d.text(x0,145,185,20,'Every column needs 4 rows',size=14,bold=True)
    for r in range(4):
        for c in range(4):
            d.line([(x0+42,184+33*r),(x0+135,184+33*c)],
                   color=ORANGE if c==1 else LIGHT,width=1.5 if c==1 else .8)
    for i in range(4):
        for x,label,color,fill in [(x0+3,'P'+str(i),BLUE,PALE[0]),(x0+135,'C'+str(i),GREEN,PALE[2])]:
            d.box(x,173+33*i,39,23,label,fill=fill,stroke=color,rounded=False,size=13)
    d.text(x0,310,185,30,'C1 receives {1,5,9,13}',size=14,color=ORANGE)
    # Step 3: live-state demand, a local bound, and nested storage ownership.
    x0=408
    d.text(x0,145,185,20,'4 values / local worker',size=14,bold=True)
    bar=x0+77
    for j,(stages,values,color) in enumerate([(1,2,GRAY),(2,4,GREEN),(4,16,ORANGE)]):
        y=176+24*j
        d.text(x0,y,73,20,f'{stages} stage(s)',size=13,align='left')
        d.box(bar,y+3,5*values,14,fill=PALE[2] if values<=4 else PALE[1],stroke=color,rounded=False,width=.7)
        d.text(x0+164,y-1,20,22,str(values),size=12,color=color)
    d.line([(bar+20,170),(bar+20,247)],color=INK,dashed=True,width=1)
    d.text(x0+2,251,178,18,'Two workers reuse local slots',size=12,color=GREEN)
    for worker in range(2):
        x=x0+4+92*worker
        d.box(x,274,85,39,fill=PALE[2],stroke=GREEN,rounded=False)
        d.text(x,275,85,16,'W'+str(worker),size=12,color=GREEN)
        for j in range(4):d.box(x+7+18*j,294,15,13,fill=PALE[0],stroke=BLUE,rounded=False,width=.7)
    d.line([(x0+46,313),(x0+46,320),(x0+139,320),(x0+139,313)],color=GRAY)
    d.text(x0,322,185,19,'Separate 16-word store',size=12,color=GRAY)
    # Step 4: one composed successor-major and XOR placement.
    x0=606
    d.text(x0,145,185,20,'offset = c; bank = r XOR c',size=13,bold=True)
    for c in range(4):
        for bank in range(4):
            r=bank ^ c; value=4*r+c
            d.box(x0+28+32*bank,183+27*c,30,25,str(value),fill=PALE[r],
                  stroke=ORANGE if c==1 else '#ffffff',width=1.6 if c==1 else 1,
                  rounded=False,size=14)
    for bank in range(4):d.text(x0+28+32*bank,165,30,17,'b'+str(bank),size=12,color=GRAY)
    d.text(x0,295,185,22,'C1 → addr. {5,4,7,6}',size=13,color=ORANGE)
    d.text(x0,320,185,20,'4 banks per access group',size=12)
    # Step 5: topological schedule; slot widths are order, not durations.
    x0=804
    d.text(x0,145,185,20,'Reuse two local workers',size=14,bold=True)
    for worker in range(2):
        d.text(x0,183+42*worker,26,28,'W'+str(worker),size=12,color=GRAY)
        for t,(kind,region) in enumerate(schedule[worker]):
            d.box(x0+29+38*t,183+42*worker,34,28,kind+str(region),
                  fill=PALE[0] if kind=='P' else PALE[2],stroke=BLUE if kind=='P' else GREEN,
                  rounded=False,size=13)
    d.line([(x0+103,174),(x0+103,262)],color=ORANGE,dashed=True)
    d.text(x0,261,185,20,'All P publish before C starts',size=12,color=ORANGE)
    # Zoom into a tile: retained intermediate nodes connect two pair stages.
    d.text(x0,291,35,26,'tile',size=12,color=GRAY)
    butterfly(d,[x0+40,x0+88,x0+136],[289,298,307,316],[0,1],color=GREEN,node_radius=1.5)
    d.line([(x0+139,303),(x0+178,303)],color=GREEN,arrow=True)
    d.text(x0,322,185,19,'retain locally → publish boundary',size=12)
    d.box(10,351,980,32,'Repeat the construction within bounded storage scopes. Arithmetic labels travel with each region.',
          fill='#f3f5f7',stroke='none',rounded=False,size=15)
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
    for j,(label,col,fill) in enumerate([('input',BLUE,PALE[0]),('producer tiles P0–P3',GREEN,PALE[2]),('boundary layout',PURPLE,PALE[3]),('consumer tiles C0–C3',GREEN,PALE[2]),('output',ORANGE,PALE[1])]):
        y=67+j*32
        d.box(275,y,185,25,label,fill=fill,stroke=col,rounded=False,size=14)
        if j<4: d.line([(367,y+25),(367,y+32)],color=GRAY,arrow=True)
    d.line([(12,231),(468,231)],color=LIGHT)
    d.text(12,236,456,18,'Same N=16 binding: {0,1} → {2,3}; four-value core',size=13,bold=True)
    d.text(8,260,135,20,'C1 input roles',size=13,color=GREEN)
    for i,value in enumerate([1,5,9,13]):
        x=150+i*72
        d.box(x,258,55,25,str(value),fill=PALE[i],stroke=[BLUE,ORANGE,GREEN,PURPLE][i],rounded=False,size=14)
    return d.save()

if __name__=='__main__':
    for make in (overview,system_architecture): print(make())
