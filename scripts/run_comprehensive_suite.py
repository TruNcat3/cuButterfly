#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import pathlib
import random
import shlex
import subprocess
import sys


MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
METADATA_FIELDS = (
    "suite_case_id", "suite_group", "suite_tier", "suite_runner", "implementation",
    "reference", "trial", "performance_batch", "preflight_correct", "preflight_max_error",
)
CASE_DESCRIPTOR_FIELDS = (
    "accumulation", "storage_bits", "compute_bits", "accumulator_bits", "emulated_native",
    "coefficient_policy", "arithmetic_pipeline", "mapping_id", "ctas_per_transform",
    "estimated_registers_per_thread", "estimated_shared_bytes_per_cta",
    "resident_ctas_per_sm", "resident_warps_per_sm", "occupancy_upper_bound",
    "total_ctas", "grid_waves", "grid_concurrent_fraction", "limiting_resource",
    "working_set_bytes", "working_set_over_l2", "resource_cliff",
)


def is_prime(value):
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    for base in MILLER_RABIN_BASES:
        if base % value == 0:
            continue
        result = pow(base, odd_part, value)
        if result in (1, value - 1):
            continue
        for _ in range(1, power):
            result = result * result % value
            if result == value - 1:
                break
        else:
            return False
    return True


def ntt_prime(bits, log_n):
    step = 1 << log_n
    multiplier = ((1 << bits) - 2) // step
    while multiplier > 0:
        candidate = multiplier * step + 1
        if candidate.bit_length() != bits:
            break
        if is_prime(candidate):
            return candidate
        multiplier -= 1
    raise ValueError(f"could not find a {bits}-bit prime for logN={log_n}")


def argument_value(arguments, option, default=""):
    if option not in arguments:
        return default
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise ValueError(f"missing value after {option}")
    return str(arguments[index + 1])


def stage_matrix_value(arguments):
    values = []
    for index, argument in enumerate(arguments):
        if argument != "--stage-matrix":
            continue
        if index + 1 >= len(arguments):
            raise ValueError("missing value after --stage-matrix")
        entries = str(arguments[index + 1]).split(",")
        if len(entries) != 4:
            raise ValueError("--stage-matrix requires four coefficients")
        values.append(":".join(format(float(entry), ".17g") for entry in entries))
    return "x".join(values)


def expected_semantics(case, batch=None):
    arguments = case["args"]
    runner = case["runner"]
    log_n = int(case["logN"])
    element_stride = int(case.get("element_stride", 1))
    if runner == "fft-library":
        placement = "in-place"
        normalization = "none"
    elif runner == "butterfly":
        placement = argument_value(arguments, "--placement", "out-of-place")
        normalization = argument_value(arguments, "--normalization", "inverse")
        if case["operator"] in ("structured-2x2", "subset-zeta", "superset-zeta", "xor-zeta"):
            normalization = "none"
    else:
        placement = ""
        normalization = ""
    semantics = {
        "operator": case["operator"],
        "precision": case["precision"],
        "accumulation": case.get("accumulation", argument_value(arguments, "--accumulation", "native")),
        "direction": "inverse" if "--inverse" in arguments else case.get("direction", "forward"),
        "normalization": normalization,
        "placement": placement,
        "stage_matrices": stage_matrix_value(arguments),
        "logN": str(log_n),
        "N": str(1 << log_n),
        "element_stride": str(element_stride),
        "output_order": argument_value(arguments, "--output-order", ""),
    }
    if batch is not None:
        semantics["batch"] = str(batch)
        transform_extent = ((1 << log_n) - 1) * element_stride + 1
        semantics["batch_stride"] = str(transform_extent + int(case.get("batch_padding", 0)))
    return semantics


def load_manifest(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported comprehensive-suite schema")
    for mode in ("quick", "full"):
        protocol = document.get("protocols", {}).get(mode, {})
        required = {"target_points", "warmup", "repeat", "trials", "verify_batch"}
        if required - set(protocol):
            raise ValueError(f"protocol {mode} is incomplete")
    case_ids = [case["id"] for case in document.get("cases", [])]
    if len(case_ids) != len(set(case_ids)) or not case_ids:
        raise ValueError("suite case ids must be nonempty and unique")
    valid_runners = set(document.get("binaries", {}))
    for case in document["cases"]:
        missing = {"id", "tier", "group", "runner", "implementation", "operator", "precision", "logN", "args"} - set(case)
        if missing:
            raise ValueError(f"case {case.get('id', '<unknown>')} lacks {sorted(missing)}")
        if case["tier"] not in ("quick", "full") or case["runner"] not in valid_runners:
            raise ValueError(f"case {case['id']} has an invalid tier or runner")
        if not 1 <= int(case["logN"]) <= 30:
            raise ValueError(f"case {case['id']} has invalid logN")
    groups = {}
    for case in document["cases"]:
        groups.setdefault(case["group"], []).append(case)
    comparable_fields = ("operator", "precision", "accumulation", "direction", "normalization", "placement", "stage_matrices", "logN", "element_stride", "output_order")
    for group, cases in groups.items():
        contracts = {
            tuple(expected_semantics(case)[field] for field in comparable_fields)
            + (str(case.get("batch", "")),)
            for case in cases
        }
        if len(contracts) != 1:
            raise ValueError(f"comparison group {group} mixes semantic contracts")
    return document


def selected_cases(document, mode, filters, exact=False):
    tiers = {"quick"} if mode == "quick" else {"quick", "full"}
    cases = [case for case in document["cases"] if case["tier"] in tiers]
    if filters:
        if exact:
            cases = [case for case in cases if case["id"] in filters or case["group"] in filters]
        else:
            cases = [case for case in cases if any(token in case["id"] or token in case["group"] for token in filters)]
    if not cases:
        raise ValueError("case selection is empty")
    return cases


def case_batch(case, protocol):
    return int(case.get("batch", max(1, int(protocol["target_points"]) // (1 << int(case["logN"])))))


def build_command(root, document, case, protocol, batch, verify):
    binary = root / document["binaries"][case["runner"]]
    if not binary.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {binary}")
    command = [str(binary), *[str(value) for value in case["args"]], "--logN", str(case["logN"]), "--batch", str(batch)]
    if case["runner"] == "butterfly":
        element_stride = int(case.get("element_stride", 1))
        transform_extent = ((1 << int(case["logN"])) - 1) * element_stride + 1
        batch_stride = transform_extent + int(case.get("batch_padding", 0))
        command += ["--element-stride", str(element_stride), "--batch-stride", str(batch_stride)]
    if case["runner"] == "ntt":
        command += ["--modulus", str(ntt_prime(int(case["modulus_bits"]), int(case["logN"])))]
    if verify:
        command += ["--warmup", "0", "--repeat", "1", "--verify", "--csv"]
    else:
        command += ["--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"]), "--csv"]
    return command


def parse_csv_record(output):
    lines = [line for line in output.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if "kernel_ms" not in line or "," not in line:
            continue
        rows = list(csv.DictReader(io.StringIO(f"{line}\n{lines[index + 1]}\n")))
        if len(rows) == 1 and rows[0].get("kernel_ms"):
            return rows[0]
    raise RuntimeError(f"benchmark did not produce one CSV record:\n{output}")


def execute(command):
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return parse_csv_record(completed.stdout)


def validate_record(case, batch, record):
    expected = expected_semantics(case, batch)
    aliases = {"performance_batch": "batch"}
    for expected_field, expected_value in expected.items():
        record_field = aliases.get(expected_field, expected_field)
        observed = record.get(record_field, "")
        if observed and observed != expected_value:
            raise RuntimeError(
                f"semantic contract mismatch for {case['id']}: {record_field}={observed}, expected {expected_value}")
    return expected


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(METADATA_FIELDS)
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_existing(path):
    if not path.is_file():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description="Run the reproducible single-GPU comprehensive comparison suite.")
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_comprehensive_suite.json"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--only", nargs="+", help="Run case ids/groups containing any supplied token.")
    parser.add_argument("--exact", action="store_true", help="Require --only tokens to equal a case id or group.")
    parser.add_argument("--output", "-o", type=pathlib.Path, default=pathlib.Path("results/comprehensive_v100_quick_raw.csv"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    document = load_manifest(manifest_path)
    protocol = document["protocols"][args.mode]
    cases = selected_cases(document, args.mode, args.only, args.exact)
    existing = read_existing(output_path) if args.resume else []
    complete = {(row["suite_case_id"], int(row["trial"])) for row in existing if row.get("trial", "").isdigit()}
    records = list(existing)

    pending = [(case, trial) for trial in range(1, int(protocol["trials"]) + 1) for case in cases
               if (case["id"], trial) not in complete]
    random.Random(0xC0B17 + (0 if args.mode == "quick" else 1)).shuffle(pending)
    if args.dry_run:
        for case, trial in pending:
            command = build_command(root, document, case, protocol, case_batch(case, protocol), False)
            print(f"trial={trial} case={case['id']} {shlex.join(command)}")
        print(f"cases={len(cases)} pending_runs={len(pending)}")
        return

    preflights = {}
    preflight_cache = {}
    for case, _ in pending:
        if case["id"] in preflights:
            continue
        if args.skip_verify:
            preflights[case["id"]] = {"correct": "-1", "max_error": ""}
            continue
        verify_batch = min(case_batch(case, protocol), int(protocol["verify_batch"]))
        command = build_command(root, document, case, protocol, verify_batch, True)
        cache_key = tuple(command)
        if cache_key in preflight_cache:
            preflights[case["id"]] = preflight_cache[cache_key]
            continue
        print(f"verify case={case['id']} batch={verify_batch}", flush=True)
        record = execute(command)
        validate_record(case, verify_batch, record)
        if record.get("correct") != "1":
            raise RuntimeError(f"correctness preflight failed for {case['id']}: {record}")
        preflight_cache[cache_key] = record
        preflights[case["id"]] = record

    for run_index, (case, trial) in enumerate(pending, 1):
        batch = case_batch(case, protocol)
        command = build_command(root, document, case, protocol, batch, False)
        print(f"run={run_index}/{len(pending)} trial={trial} case={case['id']} batch={batch}", flush=True)
        measured = execute(command)
        semantics = validate_record(case, batch, measured)
        preflight = preflights[case["id"]]
        records.append({
            "suite_case_id": case["id"],
            "suite_group": case["group"],
            "suite_tier": case["tier"],
            "suite_runner": case["runner"],
            "implementation": case["implementation"],
            "reference": int(bool(case.get("reference"))),
            "trial": trial,
            "performance_batch": batch,
            "preflight_correct": preflight.get("correct", ""),
            "preflight_max_error": preflight.get("max_error", preflight.get("max_roundtrip_error", "")),
            **{field: case[field] for field in CASE_DESCRIPTOR_FIELDS if field in case},
            **measured,
            **semantics,
        })
        write_csv(output_path, records)
    print(f"wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
