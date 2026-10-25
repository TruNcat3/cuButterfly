#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile

from run_comprehensive_suite import require_exclusive_gpu

try:
    from benchmark.runner_registry import resolve as resolve_runner
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from benchmark.runner_registry import resolve as resolve_runner


FIELDS = (
    "case_id", "group", "trial", "implementation", "reference", "runner", "device",
    "compute_capability", "hardware", "hardware_uuid", "hardware_memory_mb",
    "selection_target", "selected_implementation", "selection_confidence", "selection_reason", "reason",
    "runtime_fingerprint", "mapping_json", "execution_groups_json", "verified_batches",
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


def query_runtime_hardware():
    """Read the selected GPU identity without relying on a benchmark adapter."""
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    result = subprocess.run(
        ["nvidia-smi", "-i", selector, "--query-gpu=name,compute_cap,memory.total,uuid",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi hardware query failed")
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one target GPU from nvidia-smi, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi hardware row: {rows[0]}")
    name, compute_capability, memory_mb, uuid = fields
    return {
        "name": name,
        "compute_capability": compute_capability,
        "memory_mb": memory_mb,
        "uuid": uuid,
        "hardware": f"{name} / sm_{compute_capability.replace('.', '')} / {memory_mb} MiB / {uuid}",
    }


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


def execute(root, case, protocol, fht_python, gpuntt_binary, butterfly_binary=None, ntt_binary=None,
            runtime_hardware=None):
    common = ["--logN", str(case["logN"]), "--batch", str(case["batch"]),
              "--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"])]
    if case["runner"] == "butterfly":
        command = [str(butterfly_binary or root / "build/cubutterfly_bench"), "--operator", case["operator"],
                   "--precision", case["precision"], "--normalization", case["normalization"],
                   "--placement", case["placement"],
                   *common, *case["args"], "--verify", "--csv"]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
        row = read_one_csv(result.stdout)
        correct = row.get("correct") == "1"
        throughput = row.get("transforms_s", "")
    elif case["runner"] == "ntt":
        command = [str(ntt_binary or root / "build/cuntt_bench"), *common, "--modulus", case["modulus"],
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
    runtime_hardware = runtime_hardware or {}
    row_device = row.get("device", "").strip() if row.get("device") else ""
    device = row_device or runtime_hardware.get("name", "")
    if not device:
        raise RuntimeError(f"runtime hardware identity unavailable for {case['id']}")
    selected = row.get("selected_implementation", "").strip() if row.get("selected_implementation") else ""
    confidence = row.get("selection_confidence", "").strip() if row.get("selection_confidence") else ""
    reason = row.get("selection_reason", "").strip() if row.get("selection_reason") else ""
    if not reason and row.get("reason"):
        reason = row["reason"].strip()
    if not selected:
        selected = row.get("implementation", "").strip() if row.get("implementation") else ""
    if not selected:
        selected = case["name"]
    if not confidence:
        confidence = "external-fixed" if case["runner"] in ("dao-fht", "gpuntt") else "unreported"
    if not reason and case["runner"] in ("dao-fht", "gpuntt"):
        reason = "external baseline implementation is fixed by the adapter"
    compute_capability = row.get("compute_capability", "") or runtime_hardware.get("compute_capability", "")
    hardware_uuid = row.get("hardware_uuid", "") or runtime_hardware.get("uuid", "")
    hardware_memory_mb = row.get("hardware_memory_mb", "") or runtime_hardware.get("memory_mb", "")
    hardware = row.get("hardware", "") or runtime_hardware.get("hardware", "")
    if not hardware and row_device:
        hardware = row_device + (f" / sm_{compute_capability}" if compute_capability else "")
    return {
        "device": device,
        "compute_capability": compute_capability,
        "hardware": hardware,
        "hardware_uuid": hardware_uuid,
        "hardware_memory_mb": hardware_memory_mb,
        "selection_target": row.get("selection_target", ""),
        "selected_implementation": selected,
        "selection_confidence": confidence,
        "selection_reason": reason,
        "reason": reason,
        # Preserve the benchmark's selection identity for downstream joins;
        # an external adapter must not silently discard runtime provenance.
        "runtime_fingerprint": row.get("runtime_fingerprint", ""),
        "mapping_json": row.get("mapping_json", ""),
        "execution_groups_json": row.get("execution_groups_json", ""),
        "verified_batches": row.get("verified_batches", ""),
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
    parser.add_argument("--butterfly-binary", type=pathlib.Path)
    parser.add_argument("--ntt-binary", type=pathlib.Path)
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    document = json.loads(manifest.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported external baseline schema")
    # CLI paths remain authoritative for compatibility.  A campaign may also
    # provide the shared runner registry, allowing the same manifest to launch
    # the pinned GPU-NTT executable without a second path-only configuration.
    if "runners" in document or "binaries" in document:
        if args.gpuntt_binary is None:
            try:
                args.gpuntt_binary = resolve_runner(document, "gpuntt", root)["path"]
            except ValueError:
                pass
        if args.butterfly_binary is None:
            try:
                args.butterfly_binary = resolve_runner(document, "butterfly", root)["path"]
            except ValueError:
                pass
        if args.ntt_binary is None:
            try:
                args.ntt_binary = resolve_runner(document, "ntt", root)["path"]
            except ValueError:
                pass
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
    runtime_hardware = {}
    if pending:
        try:
            runtime_hardware = query_runtime_hardware()
        except (OSError, RuntimeError) as error:
            # Butterfly/Dao rows may carry their own identity; GPU-NTT does
            # not, and will fail honestly in execute() without this evidence.
            print(f"warning: runtime hardware query unavailable: {error}", file=sys.stderr)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = list(existing)
    for index, (case, trial) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {case['id']} trial={trial}", flush=True)
        if args.require_exclusive_gpu:
            require_exclusive_gpu(0)
        measured = execute(root, case, document["protocol"], args.fht_python, args.gpuntt_binary,
                           args.butterfly_binary, args.ntt_binary, runtime_hardware)
        if args.require_exclusive_gpu:
            require_exclusive_gpu(0)
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
