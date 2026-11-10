#!/usr/bin/env python3
"""Numerical validation of shared-iterative groups; timing is not an acceptance metric."""
import argparse
import csv
import io
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=pathlib.Path, required=True)
    parser.add_argument("--ntt-bench", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--stage-overlap", action="store_true")
    args = parser.parse_args()
    rows = []
    for operator, precisions in (("fft", ("fp32", "fp64", "fp16", "bf16")),
                                 ("fwht", ("fp32", "fp64", "fp16", "bf16")),
                                 ("structured-2x2", ("fp32", "fp64", "fp16", "bf16")),
                                 ("subset-zeta", ("uint32",)), ("superset-zeta", ("uint32",)), ("xor-zeta", ("uint32",))):
        for precision in precisions:
            for inverse, placement in ((False, "out-of-place"), (True, "in-place")):
                command = [str(args.bench), "--operator", operator, "--precision", precision,
                           "--backend", "shared-iterative", "--fft-core", "scalar", "--compute-unit", "radix2",
                           "--stage-partition", "3,4,5", "--shared-layout", "writer-aligned", "--tile-threads", "128",
                           "--logN", "12", "--batch", "7" if args.stage_overlap else "3", "--element-stride", "2", "--batch-stride", "8192",
                           "--placement", placement, "--verify", "--warmup", "1", "--repeat", "2", "--csv"]
                if inverse:
                    command.append("--inverse")
                if args.stage_overlap:
                    command.extend(["--stage-overlap", "--batch-tile-count", "2"])
                if precision in ("fp16", "bf16"):
                    command.extend(["--accumulation", "fp32"])
                if operator == "structured-2x2":
                    command.extend(["--stage-matrix", "1,0.25,-0.5,1"])
                result = subprocess.run(command, text=True, capture_output=True)
                samples = list(csv.DictReader(io.StringIO(result.stdout))) if result.returncode == 0 else []
                correct = len(samples) == 1 and samples[0].get("correct") == "1"
                rows.append(dict(command=command, returncode=result.returncode, correct=correct,
                                 samples=samples, error=result.stderr))
                print(f"{operator} {precision} {placement} inverse={inverse}: {'PASS' if correct else 'FAIL'}", flush=True)
    if args.ntt_bench:
        for bits, modulus in ((32,2013265921),(64,1152921504606584833)):
            for partition in ("3,4,5","12",",".join(["1"]*12)):
                if args.stage_overlap and partition == "12": partition = "6,6"
                for inverse in (False,True):
                    command=[str(args.ntt_bench),"--backend","shared-iterative","--compute-unit","radix2",
                        "--word-bits",str(bits),"--modulus",str(modulus),"--stage-partition",partition,
                        "--threads-per-block","128","--logN","12","--batch","7" if args.stage_overlap else "3","--verify","--csv","--warmup","1","--repeat","2"]
                    if args.stage_overlap: command.extend(["--stage-overlap","--batch-tile-count","2"])
                    if inverse: command.append("--inverse")
                    result=subprocess.run(command,text=True,capture_output=True)
                    samples=list(csv.DictReader(io.StringIO(result.stdout))) if result.returncode==0 else []
                    correct=len(samples)==1 and samples[0].get("correct")=="1"
                    rows.append(dict(command=command,returncode=result.returncode,correct=correct,samples=samples,error=result.stderr))
                    print(f"ntt word{bits} {partition} inverse={inverse}: {'PASS' if correct else 'FAIL'}",flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(r["correct"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
