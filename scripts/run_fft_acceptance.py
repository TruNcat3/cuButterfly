#!/usr/bin/env python3
"""Resumable, hardware-specific FFT acceptance; plan/JIT time is never timed.

--prepare-only exports the exact memory-clamped calibration workloads.
Calibrate those workloads before running acceptance. An uncalibrated selector
is retained as diagnostic evidence and is explicitly counted in the report.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import statistics
import subprocess

from run_comprehensive_suite import require_exclusive_gpu, visible_gpu
from calibration_space import command_for


def configured_workloads(path):
    """Reuse the exact FFT semantics from a migration manifest's workload file."""
    specification = json.loads(path.read_text())
    if specification.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported calibration workload schema")
    allowed = {"operator", "precision", "logN", "batch", "placement", "normalization",
               "direction", "accumulation", "element_stride", "batch_stride"}
    workloads, seen = [], set()
    for row in specification["workloads"]:
        if row.get("operator", "fft") != "fft":
            continue
        if set(row) - allowed:
            raise ValueError("FFT workload contains non-semantic or unsupported fields")
        cell = {"operator": "fft", "placement": "out-of-place", "normalization": "none", **row}
        if (cell.get("precision") not in ("fp32", "fp64") or not 3 <= int(cell["logN"]) <= 24
                or int(cell["batch"]) < 1):
            raise ValueError("unsupported FFT acceptance precision, size or batch")
        key = json.dumps(cell, sort_keys=True)
        if key not in seen:
            seen.add(key)
            workloads.append(dict(cell, requested_batches=[cell["batch"]]))
    if not workloads:
        raise ValueError("no FFT workloads found in calibration configuration")
    return workloads


def workload_matrix(precisions, sizes, batches, memory_bytes):
    cells = {}
    for precision in precisions:
        for log_n in sizes:
            width = 16 if precision == "fp64" else 8
            # Four full device arrays plus a 10% reserve; both libraries use
            # the same effective batch. Clamp to a power of two and deduplicate.
            maximum = max(1, int(memory_bytes * .9) // (4 * width * (1 << log_n)))
            for requested in batches:
                batch = min(requested, 1 << (maximum.bit_length()-1))
                key = (precision, log_n, batch)
                cell = cells.setdefault(key, dict(operator="fft", precision=precision, logN=log_n,
                    batch=batch, placement="out-of-place", normalization="none", requested_batches=[]))
                cell["requested_batches"].append(requested)
    return list(cells.values())


def summarize(document):
    rows = []
    for cell in document["cells"]:
        measurements = cell.get("measurements", [])
        by_name = {name: [r["sample"] for r in measurements if r["name"] == name]
                   for name in ("cuButterfly", "cuFFT")}
        if any(len(samples) != document["protocol"]["trials"] for samples in by_name.values()):
            continue
        latency = {name: statistics.median(float(s["kernel_ms"]) for s in samples)
                   for name, samples in by_name.items()}
        current = by_name["cuButterfly"][0]
        rows.append(dict(workload=cell["workload"], **latency,
            throughput_vs_cufft=latency["cuFFT"]/latency["cuButterfly"],
            selection_confidence=current.get("selection_confidence", "unknown"),
            backend=current["backend"], core=current.get("fft_core"), mapping_json=current["mapping_json"]))
    full_matrix=(not document["protocol"].get("explicit_workloads", False) and
                 set(document["protocol"].get("log_n",[]))==set(range(3,25)) and
                 set(document["protocol"].get("batches",[]))=={1,4,16,64})
    result = dict(cells=rows, precisions={}, full_size_batch_matrix=full_matrix)
    for precision in document["protocol"]["precisions"]:
        selected = [r for r in rows if r["workload"]["precision"] == precision]
        expected = sum(c["workload"]["precision"] == precision for c in document["cells"])
        geomean = math.exp(statistics.mean(math.log(r["throughput_vs_cufft"]) for r in selected)) if selected else None
        unmeasured = sum(r["selection_confidence"] == "unmeasured-feasible" for r in selected)
        scope_met=bool(len(selected)==expected and selected and not unmeasured and geomean>=.8)
        result["precisions"][precision] = dict(completed=len(selected), expected=expected,
            geometric_mean=geomean, uncalibrated_fallback_cells=unmeasured,
            selected_scope_target_met=scope_met, target_met=bool(scope_met and full_matrix))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--workloads", type=pathlib.Path,
                        help="compare the exact FFT cells of a calibration workload file, without changing batch or placement")
    parser.add_argument("--precision", nargs="+", choices=("fp32", "fp64"), default=["fp32", "fp64"])
    parser.add_argument("--log-n", nargs="+", type=int, default=list(range(3,25)))
    parser.add_argument("--batch", nargs="+", type=int, default=[1,4,16,64])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--verify-batches", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch)<1 or min(args.log_n)<3 or max(args.log_n)>24 or min(args.trials,args.repeat)<1 or args.warmup<0 or args.verify_batches<0:
        parser.error("invalid FFT matrix or timing protocol")
    binary = (args.build_dir/"cubutterfly_bench").resolve()
    info = subprocess.run(["nvidia-smi", "-i", visible_gpu(), "--query-gpu=name,compute_cap,memory.total,uuid,driver_version",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    name, sm, memory, uuid, driver = next(csv.reader(io.StringIO(info),skipinitialspace=True))
    identity=subprocess.run([str(binary),"--device-identity"],check=True,text=True,capture_output=True)
    cuda_device=next(csv.DictReader(io.StringIO(identity.stdout)))
    if cuda_device["device"]!=name or cuda_device["compute_capability"]!=sm:
        raise RuntimeError("benchmark and visibility query disagree about the GPU")
    device = dict(name=name, compute_capability=sm, memory_bytes=int(cuda_device["global_memory_bytes"]), uuid=uuid, driver=driver)
    workloads = configured_workloads(args.workloads) if args.workloads else workload_matrix(
        args.precision,args.log_n,args.batch,device["memory_bytes"])
    workload_sha256 = hashlib.sha256(args.workloads.read_bytes()).hexdigest() if args.workloads else None
    args.output_dir.mkdir(parents=True,exist_ok=True)
    workload_path = args.output_dir/"workloads.json"
    if not args.workloads or args.workloads.resolve() != workload_path.resolve():
        workload_path.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1", scope=(
            "exact configured FFT calibration cells" if args.workloads else "FFT full matrix for exact target GPU capacity"),
            workloads=[{k:v for k,v in w.items() if k!="requested_batches"} for w in workloads]),indent=2)+"\n")
    protocol = dict(precisions=sorted({w["precision"] for w in workloads}),
        log_n=sorted({w["logN"] for w in workloads}),
        batches=sorted({w["batch"] for w in workloads}) if args.workloads else args.batch, trials=args.trials,
        warmup=args.warmup, repeat=args.repeat, verify_batches=args.verify_batches,
        compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE","auto"),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        semantics=("exact workload-file semantics; CUDA-event kernel time only" if args.workloads else
                   "forward, out-of-place, contiguous, normalization none; CUDA-event kernel time only"))
    if args.workloads:
        protocol.update(explicit_workloads=True, workloads_sha256=workload_sha256)
    journal = args.output_dir/"measurements.json"
    document = dict(schema="cubutterfly-fft-acceptance-v1",device=device,protocol=protocol,
                    cells=[dict(workload=w,measurements=[]) for w in workloads])
    (args.output_dir/"prepared_matrix.json").write_text(json.dumps(
        dict(device=device,protocol=protocol,workloads=workloads),indent=2)+"\n")
    if args.resume and journal.exists():
        old = json.loads(journal.read_text())
        if old["device"]!=device or old["protocol"]!=protocol:
            raise ValueError("cannot mix another binary, hardware identity, compile policy or timing protocol")
        document = old
    def save():
        temporary=journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(document,indent=2)+"\n"); temporary.replace(journal)
        summary=summarize(document)
        (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        return summary
    if args.prepare_only:
        print(f"Prepared {len(workloads)} distinct cells for {name}: {workload_path}")
        return
    for index, cell in enumerate(document["cells"]):
        workload=cell["workload"]
        complete={(r["name"],r["trial"]) for r in cell["measurements"]}
        for trial in range(args.trials):
            for implementation in (["cuButterfly","cuFFT"] if trial%2==0 else ["cuFFT","cuButterfly"]):
                if (implementation,trial) in complete: continue
                prior=[r["sample"] for r in cell["measurements"] if r["name"]==implementation]
                command=command_for(binary, {k:v for k,v in workload.items() if k!="requested_batches"}) + [
                    "--warmup",str(args.warmup),"--repeat",str(args.repeat),"--csv"]
                command += ["--mapping-json",prior[0]["mapping_json"]] if prior else (
                    ["--auto-select"] if implementation=="cuButterfly" else ["--backend","cufft"])
                if not prior: command += ["--verify","--verify-batches",str(args.verify_batches)]
                require_exclusive_gpu()
                process=subprocess.run(command,text=True,capture_output=True)
                require_exclusive_gpu()
                if process.returncode:
                    cell["error"]=dict(implementation=implementation,command=command,stderr=process.stderr)
                    save()
                    raise RuntimeError(f"{implementation} {workload}: {process.stderr}")
                sample=next(csv.DictReader(io.StringIO(process.stdout)))
                if (not prior and sample["correct"]!="1") or not math.isfinite(float(sample["kernel_ms"])) or float(sample["kernel_ms"])<=0:
                    raise RuntimeError("invalid correctness or timing evidence")
                if prior and (prior[0]["mapping_json"]!=sample["mapping_json"] or prior[0]["runtime_fingerprint"]!=sample["runtime_fingerprint"]):
                    raise RuntimeError("mapping or code changed during paired trials")
                cell.pop("error",None)
                cell["measurements"].append(dict(name=implementation,trial=trial,sample=sample))
                save()
        summary=save()
        print(f"{index+1}/{len(workloads)} {workload['precision']} logN={workload['logN']} batch={workload['batch']}: "
              f"{summary['cells'][-1]['throughput_vs_cufft']:.3f} x cuFFT",flush=True)
    print(json.dumps(save()["precisions"],indent=2),flush=True)


if __name__ == "__main__": main()
