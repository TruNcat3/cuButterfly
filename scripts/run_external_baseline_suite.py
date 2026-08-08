#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import random
import subprocess
import sys
import tempfile


FIELDS = (
    "case_id", "group", "trial", "implementation", "reference", "runner", "device",
    "operator", "precision", "direction", "normalization", "placement", "output_order",
    "logN", "N", "batch", "modulus", "modulus_bits", "warmup", "repeat", "backend",
    "kernel_ms", "throughput_s", "correct",
)


def read_one_csv(text):
    lines = [line for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "kernel_ms" not in line or "," not in line:
            continue
        rows = list(csv.DictReader(lines[index:]))
        if rows:
            return rows[0]
    raise ValueError(f"no benchmark CSV record found in output: {text[-500:]}")


def expand(document):
    cases = []
    for workload in document["workloads"]:
        for batch in workload["batches"]:
            group = f"{workload['id']}-batch{batch}"
            for implementation in workload["implementations"]:
                cases.append({
                    **workload,
                    **implementation,
                    "id": f"{workload['id']}_{implementation['id']}_b{batch}",
                    "group": group,
                    "batch": int(batch),
                })
    return cases


def execute(root, case, protocol, fht_python, gpuntt_binary):
    common = ["--logN", str(case["logN"]), "--batch", str(case["batch"]),
              "--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"])]
    if case["runner"] == "butterfly":
        command = [str(root / "build/cubutterfly_bench"), "--operator", case["operator"],
                   "--precision", case["precision"], "--normalization", case["normalization"],
                   "--placement", case["placement"],
                   *common, *case["args"], "--verify", "--csv"]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
        row = read_one_csv(result.stdout)
        correct = row.get("correct") == "1"
        throughput = row.get("transforms_s", "")
    elif case["runner"] == "ntt":
        command = [str(root / "build/cuntt_bench"), *common, "--modulus", case["modulus"],
                   *case["args"], "--verify", "--csv"]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
        row = read_one_csv(result.stdout)
        correct = row.get("correct") == "1"
        throughput = row.get("kernel_ntt_s", "")
    elif case["runner"] == "dao-fht":
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "dao.csv"
            command = [str(fht_python), str(root / "scripts/benchmark_external_fht.py"),
                       "--logNs", str(case["logN"]), "--dtypes", case["precision"],
                       "--batch", str(case["batch"]), "--warmup", str(protocol["warmup"]),
                       "--repeat", str(protocol["repeat"]), "--trials", "1",
                       "--normalization", case["normalization"], "--sm70-patched",
                       "--output", str(output)]
            subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
            with output.open() as source:
                row = next(csv.DictReader(source))
        correct = row.get("correct") == "1"
        throughput = row.get("transforms_s", "")
    elif case["runner"] == "gpuntt":
        if gpuntt_binary is None:
            raise ValueError("GPU-NTT cases require --gpuntt-binary")
        command = [str(gpuntt_binary), str(case["logN"]), str(case["batch"]),
                   str(protocol["warmup"]), str(protocol["repeat"]), "1",
                   "1" if case.get("naturalize") else "0"]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
        row = read_one_csv(result.stdout)
        if row.get("modulus") != case["modulus"]:
            raise RuntimeError(f"GPU-NTT modulus mismatch for {case['id']}: {row.get('modulus')}")
        expected = "natural_mismatches=0" if case.get("naturalize") else "bit_reversed_mismatches=0"
        correct = expected in result.stderr
        throughput = row.get("kernel_ntt_s", "")
    else:
        raise ValueError(f"unsupported runner {case['runner']}")
    if not correct:
        raise RuntimeError(f"correctness failed for {case['id']}")
    return {
        "device": row.get("device", "Tesla V100-SXM2-16GB"),
        "backend": row.get("backend", row.get("implementation", case["runner"])),
        "kernel_ms": row["kernel_ms"],
        "throughput_s": throughput,
        "correct": 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Run matching-protocol library and cuButterfly/cuNTT comparisons.")
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_external_baseline_suite.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/v100_external_baselines_raw.csv"))
    parser.add_argument("--fht-python", type=pathlib.Path, default=pathlib.Path(sys.executable))
    parser.add_argument("--gpuntt-binary", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    document = json.loads(manifest.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported external baseline schema")
    cases = expand(document)
    if args.only:
        cases = [case for case in cases if any(token in case["id"] or token in case["group"] for token in args.only)]
    existing = []
    if args.resume and output.exists():
        with output.open() as source:
            existing = list(csv.DictReader(source))
    complete = {(row["case_id"], int(row["trial"])) for row in existing}
    pending = [(case, trial) for case in cases for trial in range(1, int(document["protocol"]["trials"]) + 1)
               if (case["id"], trial) not in complete]
    random.Random(int(document["protocol"]["seed"])).shuffle(pending)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(existing)
    for index, (case, trial) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {case['id']} trial={trial}", flush=True)
        measured = execute(root, case, document["protocol"], args.fht_python, args.gpuntt_binary)
        record = {
            "case_id": case["id"], "group": case["group"], "trial": trial,
            "implementation": case["name"], "reference": int(case.get("reference", False)),
            "runner": case["runner"], **measured,
            "operator": case["operator"], "precision": case["precision"],
            "direction": case["direction"], "normalization": case["normalization"],
            "placement": case["placement"], "output_order": case.get("output_order", "natural"),
            "logN": case["logN"], "N": 1 << int(case["logN"]), "batch": case["batch"],
            "modulus": case.get("modulus", ""), "modulus_bits": case.get("modulus_bits", ""),
            "warmup": document["protocol"]["warmup"], "repeat": document["protocol"]["repeat"],
        }
        records.append(record)
        with output.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in FIELDS} for row in records)


if __name__ == "__main__":
    main()
