#!/usr/bin/env python3
"""Measure actual cross-stage kernel overlap from a Nsight Systems SQLite export.

Use a cuFFTDx two-stage run; stage 1 contains online_first kernel names and
stage 2 contains online_direct_second / online_second. Transpose kernels are
excluded, so this measures overlap of the FFT computation itself.
"""
import argparse
import json
import re
import sqlite3


def overlap(first, second):
    """Intersection duration of two internally disjoint sorted interval sets."""
    i = j = total = 0
    while i < len(first) and j < len(second):
        a, b = first[i], second[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path, plot=None):
    with sqlite3.connect(path) as db:
        kernels = db.execute("""SELECT k.start, k.end, k.streamId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
            ORDER BY k.start""").fetchall()
    first = [row for row in kernels if "online_first" in row[3] or "online_direct_first" in row[3]]
    second = [row for row in kernels if "online_direct_second" in row[3] or "online_second" in row[3]]
    if not first or not second:
        raise ValueError("both FFT stages must be present in the CUDA trace")
    first_streams, second_streams = {row[2] for row in first}, {row[2] for row in second}
    concurrent_ns = overlap(first, second)
    stage_ns = sum(row[1] - row[0] for row in first + second)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        origin = min(first[0][0], second[0][0])
        horizon = max(row[1] for row in first[:8] + second[:8])
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for index, (label, rows, color) in enumerate((("Stage 1 FFT", first, "#287bb5"),
                                                      ("Stage 2 FFT", second, "#df8735"))):
            bars = [((row[0] - origin) / 1000, (row[1] - row[0]) / 1000)
                    for row in rows if row[0] < horizon]
            ax.broken_barh(bars, (index - 0.3, 0.6), facecolors=color)
        ax.set_yticks([0, 1], ["Stage 1 FFT", "Stage 2 FFT"])
        ax.set_xlim(0, (horizon - origin) / 1000)
        ax.set_xlabel("GPU time from first FFT kernel (microseconds)")
        ax.set_title("Measured CUDA kernel overlap (early pipeline window)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot, dpi=180)
        plt.close(fig)
    return {"stage1_kernels": len(first), "stage2_kernels": len(second),
            "stage1_streams": sorted(first_streams), "stage2_streams": sorted(second_streams),
            "distinct_stage_streams": first_streams.isdisjoint(second_streams),
            "compute_overlap_us": concurrent_ns / 1000,
            "sum_compute_us": stage_ns / 1000,
            "overlap_fraction_of_compute_union": concurrent_ns / (stage_ns - concurrent_ns),
            "actual_overlap": concurrent_ns > 0 and first_streams.isdisjoint(second_streams)}


def factor_parameters(name):
    marker="factor_streamed::kernel<"
    if marker not in name: return None
    arguments=[]; depth=0; token=""
    for char in name.split(marker,1)[1]:
        if char==">" and depth==0:
            arguments.append(token.strip()); break
        if char=="," and depth==0:
            arguments.append(token.strip()); token=""; continue
        if char=="<": depth+=1
        elif char==">": depth-=1
        token+=char
    if len(arguments)<9: raise ValueError("unrecognized factor kernel template")
    return tuple(int(re.search(r"\d+",value).group()) for value in arguments[1:4])


def analyze_factors(path,slices,plot=None):
    """Validate same-transform ready edges, including the final all-slice join.

    Each invocation has slices launches per non-final factor and one final
    launch. Group identities come from template stage coordinates, independent
    of stream numbering. Profiling times are diagnostic, not benchmark ratios.
    """
    if slices<2: raise ValueError("partial factor trace requires slices>=2")
    with sqlite3.connect(path) as db:
        rows=db.execute("""SELECT k.start,k.end,k.streamId,s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
            ORDER BY k.start""").fetchall()
    grouped={}; totals=set()
    for row in rows:
        shape=factor_parameters(row[3])
        if shape is None: continue
        total,log,done=shape; totals.add(total)
        grouped.setdefault((done,log),[]).append(row)
    keys=sorted(grouped)
    if len(totals)!=1 or len(keys)<2: raise ValueError("one multi-factor mapping is required")
    if keys[0][0]!=0 or any(a[0]+a[1]!=b[0] for a,b in zip(keys,keys[1:])) or sum(k[1] for k in keys)!=next(iter(totals)):
        raise ValueError("trace mixes incomplete or different factor partitions")
    count=len(grouped[keys[-1]])
    if any(len(grouped[key])!=count*slices for key in keys[:-1]):
        raise ValueError("incomplete factor invocations or incorrect slice count")
    violations=[]; early=[]; edges=[]
    for edge in range(len(keys)-1):
        producer,consumer=grouped[keys[edge]],grouped[keys[edge+1]]
        intersection=overlap(producer,consumer)
        edges.append(dict(producer_group=edge,consumer_group=edge+1,
            compute_overlap_us=intersection/1000,final_join=edge+2==len(keys)))
    for invocation in range(count):
        for edge in range(len(keys)-1):
            producer=grouped[keys[edge]][invocation*slices:(invocation+1)*slices]
            consumer=([grouped[keys[edge+1]][invocation]] if edge+2==len(keys)
                      else grouped[keys[edge+1]][invocation*slices:(invocation+1)*slices])
            for slot,row in enumerate(consumer):
                dependency_end=max(p[1] for p in producer) if len(consumer)==1 else producer[slot][1]
                if row[0]<dependency_end: violations.append(dict(invocation=invocation,edge=edge,slice=slot))
                if len(consumer)>1 and row[0]<max(p[1] for p in producer):
                    early.append(dict(invocation=invocation,edge=edge,slice=slot))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,3.2))
        origin=grouped[keys[0]][0][0]
        for group,key in enumerate(keys):
            spans=grouped[key][:(1 if group+1==len(keys) else slices)]
            ax.broken_barh([((r[0]-origin)/1000,(r[1]-r[0])/1000) for r in spans],(group-.3,.6))
        ax.set_yticks(range(len(keys)),[f"Factor {i}: 2^{k[1]}" for i,k in enumerate(keys)])
        ax.set_xlabel("GPU time from first factor (microseconds)"); ax.grid(axis="x",alpha=.2)
        fig.tight_layout(); fig.savefig(plot,dpi=180); plt.close(fig)
    return dict(factor_partition=[key[1] for key in keys],slices=slices,invocations=count,
        group_streams=[sorted({r[2] for r in grouped[key]}) for key in keys],edges=edges,
        dependency_violations=violations,early_consumer_launches=len(early),
        dependency_order_verified=not violations,
        actual_overlap=not violations and bool(early) and any(e["compute_overlap_us"]>0 for e in edges))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--plot", help="save a timeline figure of the first eight tiles")
    parser.add_argument("--factor-slices",type=int,help="analyze factor-streamed partial readiness")
    args = parser.parse_args()
    result = analyze_factors(args.sqlite,args.factor_slices,args.plot) if args.factor_slices else analyze(args.sqlite,args.plot)
    print(json.dumps(result, indent=2))
    if args.require_overlap and not result["actual_overlap"]:
        raise SystemExit("No actual cross-stage overlap measured")
