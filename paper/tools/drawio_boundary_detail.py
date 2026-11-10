"""A complete producer/store/boundary/load/consumer path, without fake timings."""
from drawio_common import Diagram, BLUE, GREEN, ORANGE, PURPLE, INK, GRAY, LIGHT, PALE, butterfly
from walkthrough_example import checked_example, placement


def boundary_detail():
    rows, columns, _ = checked_example()
    d = Diagram('hierarchy', 1000, 414)
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    d.text(10,5,980,23,'(a) Local outputs become the next core\'s inputs through one agreed boundary representation.',size=16,bold=True)
    headers=[(10,155,'Producer subgraphs'),(174,170,'Mapped stores'),
             (365,177,'Shared boundary'),(557,134,'Consumer lanes'),(702,164,'Local core'),(883,107,'Final output')]
    for x,w,label in headers:d.text(x,35,w,22,label,size=14,bold=True)
    # Four genuine two-stage producer subgraphs. Colors carry producer identity.
    for r in range(4):
        ys=[74+42*r+9*c for c in range(4)]
        d.box(32,ys[0]-5,118,37,fill=PALE[r],stroke=colors[r],rounded=False,width=.8)
        d.text(4,ys[0]+4,26,22,'P'+str(r),size=14,bold=True,color=colors[r])
        butterfly(d,[44,93,141],ys,[0,1],color=colors[r],node_radius=1.7)
        for c in range(4):
            i=4*r+c; y=ys[c]
            d.line([(141,y),(182,74+10.2*i)],color=colors[r],width=.9)
            if c==1:d.text(151,y-8,25,18,str(i),size=14,color=colors[r])
    d.text(12,238,150,20,'bits {0,1}; i = 4r + c',size=13)
    # This is exactly a(i), with all sixteen wires; it is not a reorder launch.
    for r in range(4):
        for c in range(4):
            i=4*r+c; a=placement(r,c)
            d.line([(182,74+10.2*i),(325,74+10.2*a)],color=colors[r],width=1.15 if c==1 else .8)
            d.dot(182,74+10.2*i,1.8,colors[r]);d.dot(325,74+10.2*a,1.8,colors[r])
    d.line([(333,110),(341,110),(341,150),(333,150)],color=ORANGE,width=1.2)
    d.text(331,153,30,17,'4..7',size=12,color=ORANGE)
    d.text(171,238,180,20,'a = 4c + (r XOR c)',size=13,bold=True)
    # Offset c chooses a row of the boundary; bank order need not equal lane order.
    gx,gy,step=397,91,31
    d.text(367,61,177,19,'offset c; bank r XOR c',size=13)
    for bank in range(4):d.text(gx+step*bank,75,29,16,'b'+str(bank),size=13,color=GRAY)
    for c in range(4):
        d.text(366,gy+step*c,28,28,'c'+str(c),size=13,color=GRAY)
        for bank in range(4):
            r=bank^c;i=4*r+c
            d.box(gx+step*bank,gy+step*c,29,29,str(i),fill=PALE[r],
                  stroke=ORANGE if c==1 else '#ffffff',width=2 if c==1 else .8,rounded=False,size=15)
    d.line([(350,194),(365,194)],color=GRAY,arrow=True)
    d.text(369,220,173,33,'16 words; publish after\nall four producers finish',size=13)
    # C1 is highlighted, with requested lane order and bank selection explicit.
    d.line([(521,136),(548,136)],color=ORANGE,width=2)
    for r,i in enumerate(columns[1]):
        y=87+37*r
        d.line([(548,136),(561,y+13)],color=ORANGE,width=1)
        d.text(563,y,45,26,'r'+str(r),size=14)
        d.box(611,y,35,26,str(i),fill=PALE[r],stroke=ORANGE,rounded=False,size=15)
        d.text(649,y,39,26,'b'+str(r^1),size=13,color=GRAY)
        d.line([(689,y+13),(719,y+13)],color=colors[r],arrow=True)
    d.text(555,238,142,20,'C1: {1,5,9,13}',size=13,color=ORANGE)
    ys=[100+37*r for r in range(4)]
    butterfly(d,[719,776,833],ys,[0,1],color=GREEN,node_radius=2.3)
    d.text(718,61,59,20,'bit 2',size=13,color=GREEN)
    d.text(779,61,59,20,'bit 3',size=13,color=GREEN)
    d.text(708,238,153,20,'local r bits → global {2,3}',size=12)
    for r,i in enumerate(columns[1]):
        y=ys[r]
        d.line([(833,y),(890,y)],color=GREEN,arrow=True)
        d.box(893,y-13,90,26,'out['+str(i)+']',fill='#ffffff',stroke=GRAY,rounded=False,size=14)
    d.text(880,238,111,20,'requested order',size=13)
    d.line([(10,269),(990,269)],color=LIGHT)
    # An independent external-scope illustration, using explicitly toy segments.
    d.text(10,277,483,21,'(b) External layout aligns a consumer\'s addresses.',size=14,bold=True)
    for y,label,addresses in [(307,'4r + c',[1,5,9,13]),(341,'4c + r',[4,5,6,7])]:
        d.text(14,y,77,22,label,size=14)
        for segment in range(4):
            x=99+92*segment
            d.box(x,y-2,86,27,fill='#ffffff',stroke=GRAY,rounded=False,dashed=True,width=.7)
            for j in range(4):
                a=4*segment+j
                d.box(x+2+21*j,y,20,22,str(a),fill=PALE[2] if a in addresses else '#f3f5f7',
                      stroke=GREEN if a in addresses else '#ffffff',rounded=False,size=12)
        assert len({a//4 for a in addresses})==(4 if label=='4r + c' else 1)
    d.text(10,373,483,22,'4-word segments: C1 reads 4 → 1; P0 stores 1 → 4.',size=13)
    # Shared-bank counterexample explains why successor alignment alone is not enough.
    d.line([(503,281),(503,395)],color=LIGHT)
    d.text(514,277,476,21,"(c) Bank remapping distributes P0's stores.",size=14,bold=True)
    for x0,swizzle,label in [(518,False,'4c + r'),(764,True,'4c + (r XOR c)')]:
        d.text(x0,304,224,20,label,size=14,color=GREEN if swizzle else GRAY)
        for b in range(4):
            d.text(x0+b*53,328,48,17,'b'+str(b),size=12,color=GRAY)
            vals=[c for c in range(4) if (c if swizzle else 0)==b]
            d.box(x0+b*53,348,48,25,','.join(map(str,vals)) if vals else '–',
                  fill=PALE[0] if vals else '#f3f5f7',stroke=BLUE if vals else LIGHT,
                  rounded=False,size=12)
    d.line([(742,359),(759,359)],color=GREEN,arrow=True)
    d.text(514,376,476,20,'P0 (r=0); entries are c. Concurrent groups may contend.',size=12,color=GRAY)
    d.text(10,397,980,15,'Analytical layouts at separate storage scopes. Wire/box labels denote value roles or addresses, not measured throughput.',size=12,color=GRAY)
    return d.save()
