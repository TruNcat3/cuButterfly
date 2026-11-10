#!/usr/bin/env python3
"""Resumable factor/data-fold/prefetch ablation through the public benchmark.

Explicit mappings only: this is not an automatic-selector or full-matrix claim.
JIT/plan setup and full CPU verification stay outside CUDA-event kernel timing.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess

from run_comprehensive_suite import require_exclusive_gpu


DEFAULT_FACTOR_SLICES = (2, 4, 8)


def points(log_n, *, core="cufftdx-block", factors=None, columns=8, ept=16):
    # A controlled, balanced factorization; search owns the full candidate set.
    count=(log_n+7)//8
    factors=factors or [log_n//count+(i<log_n%count) for i in range(count)]
    if sum(factors)!=log_n: raise ValueError("factor partition must cover logN")
    for tiles,depth in ((1,0),(4,0),(4,1),(4,2),(16,0),(16,1),(16,2)):
        yield f"tiles{tiles}-depth{depth}",dict(
            backend="factor-streamed",fft_core=core,compute_unit="auto",
            shared_layout="writer-aligned",cross_twiddle="recurrence",
            stage_partition=[log_n],factor_partition=factors,factor_columns=columns,factor_ept=ept,
            data_tiles_per_cta=tiles,prefetch_depth=depth)


def schedule_points(log_n, *, core="cufftdx-block", factors=None, columns=8, ept=16,
                    data_tiles_per_cta=4, prefetch_depth=1,
                    factor_slices=DEFAULT_FACTOR_SLICES):
    """Yield whole and partial-range schedule controls for one factorization."""
    count=(log_n+7)//8
    factors=list(factors) if factors else [log_n//count+(i<log_n%count) for i in range(count)]
    if sum(factors)!=log_n: raise ValueError("factor partition must cover logN")
    if data_tiles_per_cta < 1 or prefetch_depth < 0:
        raise ValueError("data tiles must be positive and prefetch depth non-negative")
    if columns < 1:
        raise ValueError("factor columns must be positive")
    requested=tuple(factor_slices)
    if not requested:
        raise ValueError("factor_slices must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("factor_slices must not contain duplicates")
    max_slices=(1 << factors[-1]) // columns
    if max_slices < 1:
        raise ValueError("last factor cannot supply one complete column tile")
    if any(value < 1 or value & (value-1) or value > max_slices for value in requested):
        raise ValueError("factor_slices must be powers of two within the last factor")
    base=dict(
        backend="factor-streamed",fft_core=core,compute_unit="auto",
        shared_layout="writer-aligned",cross_twiddle="recurrence",
        stage_partition=[log_n],factor_partition=factors,factor_columns=columns,factor_ept=ept,
        data_tiles_per_cta=data_tiles_per_cta,prefetch_depth=prefetch_depth)
    yield "whole", dict(base, factor_slices=1, factor_overlap=False)
    for slices in requested:
        yield f"slice{slices}-serial", dict(base, factor_slices=slices, factor_overlap=False)
        yield f"slice{slices}-overlap", dict(base, factor_slices=slices, factor_overlap=True)


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--logN",type=int,nargs="+",default=[24])
    parser.add_argument("--precision",choices=["fp32","fp64"],default="fp64")
    parser.add_argument("--batch",type=int,default=1)
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--core",choices=["cufftdx-block","register-tile"],default="cufftdx-block")
    parser.add_argument("--factor-partition",type=int,nargs="+")
    parser.add_argument("--factor-columns",type=int,default=8)
    parser.add_argument("--factor-ept",type=int,default=16)
    parser.add_argument("--schedule-ablation",action="store_true",
                        help="compare whole, serial partial-range and overlap schedules")
    parser.add_argument("--data-tiles-per-cta",type=int,default=4)
    parser.add_argument("--prefetch-depth",type=int,default=1)
    parser.add_argument("--factor-slices",type=int,nargs="+",default=list(DEFAULT_FACTOR_SLICES))
    parser.add_argument("--verify-batches",type=int,default=0,
                        help="verification batch limit; zero checks the full batch")
    return parser


def main(argv=None):
    parser=make_parser()
    args=parser.parse_args(argv)
    if args.trials < 1 or args.batch < 1 or min(args.logN) < 1:
        parser.error("logN, batch and trials must be positive")
    if args.verify_batches < 0:
        parser.error("verify-batches must be non-negative")
    if args.schedule_ablation and (args.data_tiles_per_cta < 1 or args.prefetch_depth < 0):
        parser.error("schedule ablation requires positive data tiles and non-negative prefetch depth")
    if args.schedule_ablation and not args.factor_slices:
        parser.error("schedule ablation requires at least one factor slice count")
    identity=dict(protocol_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.bench.read_bytes()).hexdigest(),
        compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE"),gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        logN=args.logN,precision=args.precision,batch=args.batch,trials=args.trials,warmup=20,repeat=50,
        core=args.core,factor_partition=args.factor_partition,factor_columns=args.factor_columns,factor_ept=args.factor_ept,
        schedule_ablation=args.schedule_ablation,data_tiles_per_cta=args.data_tiles_per_cta,
        prefetch_depth=args.prefetch_depth,factor_slices=args.factor_slices,
        verify_batches=args.verify_batches)
    saved=json.loads(args.output.read_text()) if args.output.exists() else dict(identity=identity,rows=[])
    if saved["identity"]!=identity: raise RuntimeError("ablation resume identity changed")
    finished={(r["logN"],r["name"],r["trial"]) for r in saved["rows"]}
    for n in args.logN:
        if args.schedule_ablation:
            variants=list(schedule_points(n,core=args.core,factors=args.factor_partition,
                                          columns=args.factor_columns,ept=args.factor_ept,
                                          data_tiles_per_cta=args.data_tiles_per_cta,
                                          prefetch_depth=args.prefetch_depth,
                                          factor_slices=args.factor_slices))
        else:
            variants=list(points(n,core=args.core,factors=args.factor_partition,
                                 columns=args.factor_columns,ept=args.factor_ept))
        variants.append(("cufft",None))
        for trial in range(args.trials):
            for name,mapping in variants if trial%2==0 else variants[::-1]:
                if (n,name,trial) in finished: continue
                command=[str(args.bench),"--operator","fft","--precision",args.precision,
                    "--logN",str(n),"--batch",str(args.batch),"--placement","out-of-place",
                    "--normalization","none","--warmup","20","--repeat","50","--csv"]
                command+=(["--mapping-json",json.dumps(mapping)] if mapping else ["--backend","cufft"])
                if trial==0:
                    command.extend(["--verify","--verify-batches",str(args.verify_batches)])
                require_exclusive_gpu()
                result=subprocess.run(command,text=True,capture_output=True)
                require_exclusive_gpu()
                if result.returncode: raise RuntimeError(f"{n}/{name}: {result.stderr}")
                sample=next(csv.DictReader(io.StringIO(result.stdout)))
                if trial==0:
                    first_trial_passed=sample.get("correct") == "1"
                    if not first_trial_passed:
                        print(f"{n}/{name}: numerical verification failed; timing is not accepted",flush=True)
                else:
                    first_trial_passed=any(
                        row["logN"]==n and row["name"]==name and row["trial"]==0 and
                        row.get("correctness_verified",False) for row in saved["rows"])
                saved["rows"].append(dict(logN=n,name=name,trial=trial,sample=sample,
                    correctness_verified=trial==0 and first_trial_passed,
                    performance_accepted=first_trial_passed))
                args.output.write_text(json.dumps(saved,indent=2)+"\n")
                print(n,name,trial,sample["kernel_ms"],flush=True)
        medians={name:statistics.median(float(r["sample"]["kernel_ms"]) for r in saved["rows"]
                    if r["logN"]==n and r["name"]==name) for name,_ in variants}
        print(json.dumps(dict(logN=n,median_ms=medians,
            throughput_vs_cufft={name:medians["cufft"]/value for name,value in medians.items()})),flush=True)


if __name__=="__main__": main()
