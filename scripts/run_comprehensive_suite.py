#!/usr/bin/env python3
import argparse
import csv
import io
import json
import math
import os
import pathlib
import random
import shlex
import subprocess
import sys

try:
    from benchmark.runner_registry import resolve as resolve_runner
    from benchmark.runner_registry import validate_references as validate_runner_references
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from benchmark.runner_registry import resolve as resolve_runner
    from benchmark.runner_registry import validate_references as validate_runner_references


MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
METADATA_FIELDS = (
    "suite_case_id", "suite_group", "suite_tier", "suite_runner", "implementation",
    "reference", "trial", "performance_batch", "verified_batches", "preflight_correct",
    "preflight_max_error",
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
        if "verify_timed_batches" in protocol:
            value = protocol["verify_timed_batches"]
            if type(value) is not int or value < 0:
                raise ValueError(f"protocol {mode} verify_timed_batches must be a nonnegative integer")
    case_ids = [case["id"] for case in document.get("cases", [])]
    if len(case_ids) != len(set(case_ids)) or not case_ids:
        raise ValueError("suite case ids must be nonempty and unique")
    try:
        valid_runners = set(validate_runner_references(document))
    except ValueError as error:
        raise ValueError(f"invalid runner registry: {error}") from error
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


def timed_verification_batch_limit(case, protocol, batch):
    if case["runner"] not in ("butterfly", "ntt"):
        return 0
    requested = int(protocol.get("verify_timed_batches", 0))
    return min(int(batch), requested) if requested > 0 else 0


def build_command(root, document, case, protocol, batch, verify, timed_verify_batches=0):
    binary = resolve_runner(document, case["runner"], root)["path"]
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
        if case["runner"] in ("butterfly", "ntt") and int(timed_verify_batches) > 0:
            command += ["--verify", "--verify-batches", str(int(timed_verify_batches))]
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


class CommandFailure(RuntimeError):
    """A failed benchmark process with its raw stdout/stderr attached."""

    def __init__(self, command, returncode, stdout, stderr):
        command_text = " ".join(shlex.quote(str(value)) for value in command)
        super().__init__(
            f"command failed with exit code {returncode}: {command_text}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}")
        self.command = [str(value) for value in command]
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr


def execute(command):
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise CommandFailure(command, completed.returncode, completed.stdout, completed.stderr)
    return parse_csv_record(completed.stdout)


def visible_gpu():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]


def require_exclusive_gpu(idle_context_memory_mb=64):
    gpu = visible_gpu()
    utilization = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    processes = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not processes:
        return
    active = []
    gpu_busy = int(float(utilization or "0")) > 0
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(",")]
        try:
            memory_mb = float(fields[-1].split()[0])
        except (IndexError, ValueError):
            memory_mb = float("inf")
        if gpu_busy or memory_mb > idle_context_memory_mb:
            active.append(line)
    if active:
        raise RuntimeError(f"target GPU {gpu} is not exclusive; active compute processes: {'; '.join(active)}")


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


def failure_sidecar(path):
    return path.with_name(path.name + ".failures.json")


def read_failures(path):
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    failures = document.get("failures", [])
    return list(failures) if isinstance(failures, list) else []


def write_failures(path, failures):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "status": "failed" if failures else "clear",
        "failure_count": len(failures),
        "failures": failures,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure_key(entry):
    return (entry.get("phase", ""), entry.get("case_id", ""), str(entry.get("trial", "")))


def failure_entry(phase, case, trial, command, error, record=None):
    entry = {
        "phase": phase,
        "case_id": case["id"],
        "group": case.get("group", ""),
        "trial": trial,
        "command": [str(value) for value in command],
        "error": str(error),
    }
    if isinstance(error, CommandFailure):
        entry.update({
            "returncode": error.returncode,
            "stdout": error.stdout,
            "stderr": error.stderr,
        })
    if record is not None:
        entry["record"] = record
    return entry


def remember_failure(history, sidecar, entry):
    key = _failure_key(entry)
    history[:] = [item for item in history if _failure_key(item) != key]
    history.append(entry)
    write_failures(sidecar, history)


def clear_failures(history, sidecar, *, case_id, phase=None, trial=None):
    def keep(item):
        if item.get("case_id") != case_id:
            return True
        if phase is not None and item.get("phase") != phase:
            return True
        if trial is not None and str(item.get("trial", "")) != str(trial):
            return True
        return False

    history[:] = [item for item in history if keep(item)]
    if history or sidecar.exists():
        write_failures(sidecar, history)


def main():
    parser = argparse.ArgumentParser(description="Run the reproducible single-GPU comprehensive comparison suite.")
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_comprehensive_suite.json"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--only", nargs="+", help="Run case ids/groups containing any supplied token.")
    parser.add_argument("--exact", action="store_true", help="Require --only tokens to equal a case id or group.")
    parser.add_argument("--output", "-o", type=pathlib.Path, default=pathlib.Path("results/comprehensive_v100_quick_raw.csv"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--idle-context-memory-mb", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.idle_context_memory_mb < 0:
        raise ValueError("idle context memory cannot be negative")

    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    document = load_manifest(manifest_path)
    protocol = document["protocols"][args.mode]
    cases = selected_cases(document, args.mode, args.only, args.exact)
    existing = read_existing(output_path) if args.resume else []
    complete = {(row["suite_case_id"], int(row["trial"])) for row in existing if row.get("trial", "").isdigit()}
    records = list(existing)
    sidecar_path = failure_sidecar(output_path)
    failure_history = read_failures(sidecar_path) if args.resume else []
    current_failures = []

    pending = [(case, trial) for trial in range(1, int(protocol["trials"]) + 1) for case in cases
               if (case["id"], trial) not in complete]
    random.Random(0xC0B17 + (0 if args.mode == "quick" else 1)).shuffle(pending)
    if args.dry_run:
        for case, trial in pending:
            batch = case_batch(case, protocol)
            timed_verify_batches = timed_verification_batch_limit(case, protocol, batch)
            command = build_command(root, document, case, protocol, batch, False, timed_verify_batches)
            command_text = " ".join(shlex.quote(str(value)) for value in command)
            print(f"trial={trial} case={case['id']} {command_text}")
        print(f"cases={len(cases)} pending_runs={len(pending)}")
        return

    preflights = {}
    preflight_cache = {}
    preflight_failures = {}
    preflight_failed_ids = set()
    for case, _ in pending:
        if case["id"] in preflights:
            continue
        if args.skip_verify:
            preflights[case["id"]] = {"correct": "-1", "max_error": ""}
            continue
        verify_batch = min(case_batch(case, protocol), int(protocol["verify_batch"]))
        command = []
        record = None
        try:
            command = build_command(root, document, case, protocol, verify_batch, True)
            cache_key = tuple(command)
            if cache_key in preflight_cache:
                preflights[case["id"]] = preflight_cache[cache_key]
                clear_failures(failure_history, sidecar_path, case_id=case["id"], phase="preflight")
                continue
            if cache_key in preflight_failures:
                raise RuntimeError("shared preflight command already failed for an independent case")
            print(f"verify case={case['id']} batch={verify_batch}", flush=True)
            if args.require_exclusive_gpu:
                require_exclusive_gpu(args.idle_context_memory_mb)
            record = execute(command)
            if args.require_exclusive_gpu:
                require_exclusive_gpu(args.idle_context_memory_mb)
            validate_record(case, verify_batch, record)
            if record.get("correct") != "1":
                raise RuntimeError(f"correctness preflight failed for {case['id']}: {record}")
            preflight_cache[cache_key] = record
            preflights[case["id"]] = record
            clear_failures(failure_history, sidecar_path, case_id=case["id"], phase="preflight")
        except Exception as error:
            entry = failure_entry("preflight", case, None, command, error, locals().get("record"))
            preflight_failed_ids.add(case["id"])
            if command:
                preflight_failures[tuple(command)] = entry
            remember_failure(failure_history, sidecar_path, entry)
            current_failures.append(entry)
            print(f"preflight failed case={case['id']}; continuing", file=sys.stderr, flush=True)

    for run_index, (case, trial) in enumerate(pending, 1):
        if case["id"] in preflight_failed_ids:
            print(f"skip case={case['id']} trial={trial}: preflight failed", file=sys.stderr, flush=True)
            continue
        batch = case_batch(case, protocol)
        timed_verify_batches = timed_verification_batch_limit(case, protocol, batch)
        command = []
        measured = None
        try:
            command = build_command(root, document, case, protocol, batch, False, timed_verify_batches)
            print(f"run={run_index}/{len(pending)} trial={trial} case={case['id']} batch={batch}", flush=True)
            if args.require_exclusive_gpu:
                require_exclusive_gpu(args.idle_context_memory_mb)
            measured = execute(command)
            if args.require_exclusive_gpu:
                require_exclusive_gpu(args.idle_context_memory_mb)
            semantics = validate_record(case, batch, measured)
            verified_batches = measured.get("verified_batches", "")
            if timed_verify_batches:
                if measured.get("correct") != "1":
                    raise RuntimeError(f"timed correctness check failed for {case['id']}: {measured}")
                if verified_batches != str(timed_verify_batches):
                    raise RuntimeError(
                        f"timed verification count mismatch for {case['id']}: "
                        f"verified_batches={verified_batches}, expected {timed_verify_batches}")
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
                "verified_batches": verified_batches,
                "preflight_correct": preflight.get("correct", ""),
                "preflight_max_error": preflight.get("max_error", preflight.get("max_roundtrip_error", "")),
                **{field: case[field] for field in CASE_DESCRIPTOR_FIELDS if field in case},
                **measured,
                **semantics,
            })
            clear_failures(failure_history, sidecar_path, case_id=case["id"], phase="measure", trial=trial)
            write_csv(output_path, records)
        except Exception as error:
            entry = failure_entry("measure", case, trial, command, error, measured)
            remember_failure(failure_history, sidecar_path, entry)
            current_failures.append(entry)
            print(f"measurement failed case={case['id']} trial={trial}; continuing", file=sys.stderr, flush=True)
    print(f"wrote {len(records)} samples to {output_path}")
    if current_failures:
        first = current_failures[0]
        raise RuntimeError(
            f"{len(current_failures)} comprehensive-suite case(s) failed; "
            f"first failure {first['case_id']} ({first['phase']}): {first['error']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
