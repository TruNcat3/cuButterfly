#!/usr/bin/env python3
"""Paired regression anchors. Requires an idle GPU before and after every run."""
import argparse
import csv
import io
import json
import math
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
from run_comprehensive_suite import require_exclusive_gpu


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--old", type=pathlib.Path, required=True)
    parser.add_argument("--new", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args=parser.parse_args()
    rows=[]
    for log_n in (18,20):
        for batch in (1,4,16):
            for trial in range(args.trials):
                order=["previous","current","cufft"]
                if trial%2: order.reverse()
                for name in order:
                    require_exclusive_gpu()
                    command=[str(args.old if name=="previous" else args.new),"--operator","fft","--precision","fp32",
                             "--logN",str(log_n),"--batch",str(batch),"--placement","out-of-place","--normalization","none",
                             "--warmup","100","--repeat","200","--csv"]
                    command += ["--backend","cufft"] if name=="cufft" else ["--auto-select"]
                    result=subprocess.run(command,text=True,capture_output=True,check=True)
                    require_exclusive_gpu()
                    sample=next(csv.DictReader(io.StringIO(result.stdout)))
                    rows.append(dict(name=name,trial=trial,logN=log_n,batch=batch,sample=sample))
                    args.output.write_text(json.dumps(rows,indent=2)+"\n")
            medians={name:statistics.median(float(r["sample"]["kernel_ms"]) for r in rows
                       if r["logN"]==log_n and r["batch"]==batch and r["name"]==name)
                     for name in ("previous","current","cufft")}
            print(dict(logN=log_n,batch=batch,**medians,
                       throughput_vs_cufft=medians["cufft"]/medians["current"],
                       throughput_vs_previous=medians["previous"]/medians["current"]),flush=True)
    ratios=[]
    for log_n in (18,20):
        for batch in (1,4,16):
            latency=lambda name:statistics.median(float(r["sample"]["kernel_ms"]) for r in rows
                       if r["logN"]==log_n and r["batch"]==batch and r["name"]==name)
            ratios.append(latency("cufft")/latency("current"))
    print("FP32 six-cell geometric mean:",math.exp(statistics.mean(map(math.log,ratios))))


if __name__=="__main__": main()
