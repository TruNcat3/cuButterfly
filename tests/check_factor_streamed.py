#!/usr/bin/env python3
"""Focused public-benchmark contracts for the factor-streamed FFT lowering."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_comprehensive_suite import require_exclusive_gpu


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core",choices=["cufftdx-block","register-tile"],default="cufftdx-block")
    parser.add_argument("--factor-slices",type=int,default=1)
    parser.add_argument("--factor-overlap",action="store_true")
    args=parser.parse_args()
    identity=dict(protocol_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(args.bench.read_bytes()).hexdigest(),core=args.core,
                  factor_slices=args.factor_slices,factor_overlap=args.factor_overlap,
                  compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE"),gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    saved=json.loads(args.output.read_text()) if args.output.exists() else dict(identity=identity,cases=[])
    if saved["identity"]!=identity: raise RuntimeError("correctness resume identity changed")
    finished={r["id"] for r in saved["cases"]}
    shapes=((10,[10],1,32),(12,[6,6],4,8),(18,[6,6,6],8,8),(16,[4,4,4,4],8,4)) if args.core=="register-tile" else (
        (8,[8],1,8),(12,[6,6],4,8),(18,[6,6,6],8,16),(19,[4,5,6,4],8,4))
    for precision in ("fp32","fp64"):
        for n,factors,columns,ept in shapes:
            if args.factor_slices>1 and len(factors)==1: continue
            for depth in (0,1,2):
                mapping=dict(backend="factor-streamed",fft_core=args.core,compute_unit="auto",
                             shared_layout="writer-aligned",cross_twiddle="recurrence",stage_partition=[n],
                             factor_partition=factors,factor_columns=columns,factor_ept=ept,
                             data_tiles_per_cta=5,prefetch_depth=depth)
                mapping.update(factor_slices=args.factor_slices,factor_overlap=args.factor_overlap)
                for inverse in (False,True):
                    name=f"{precision}-{n}-{depth}-{int(inverse)}"
                    if name in finished: continue
                    cmd=[str(args.bench),"--operator","fft","--precision",precision,"--logN",str(n),
                         "--batch","3","--element-stride","2","--batch-stride",str((1<<n)*2+16),
                         "--placement","in-place" if inverse else "out-of-place","--normalization","inverse",
                         "--mapping-json",json.dumps(mapping),"--warmup","0","--repeat","1","--verify","--csv"]
                    if inverse: cmd.append("--inverse")
                    require_exclusive_gpu()
                    result=subprocess.run(cmd,text=True,capture_output=True)
                    require_exclusive_gpu()
                    if result.returncode: raise RuntimeError(f"{name}: {result.stderr}")
                    row=next(csv.DictReader(io.StringIO(result.stdout)))
                    if row["correct"]!="1": raise RuntimeError(f"{name}: incorrect output")
                    groups=json.loads(row["execution_groups_json"])
                    assert len(groups)==len(factors) and int(row["decomposition_count"])==1
                    assert all(g["compiler_local_resources_known"] for g in groups)
                    assert all(g["data_tiles_per_cta"]==5 and g["prefetch_depth"]==depth for g in groups)
                    canonical=json.loads(row["mapping_json"])
                    assert canonical["factor_partition"]==factors and canonical["stage_partition"]==[n]
                    assert canonical["factor_slices"]==args.factor_slices and canonical["factor_overlap"]==args.factor_overlap
                    assert [g["launch_count"] for g in groups]==[args.factor_slices]*(len(factors)-1)+[1]
                    assert [g["partial_dependency_ready"] for g in groups]==[
                        args.factor_overlap and 0<i<len(factors)-1 for i in range(len(factors))]
                    saved["cases"].append(dict(id=name,sample=row,correct=True,performance_accepted=False))
                    args.output.write_text(json.dumps(saved,indent=2)+"\n")
                    print(name,"correct",flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
