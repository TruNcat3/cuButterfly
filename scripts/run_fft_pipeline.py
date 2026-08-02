#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import random
import subprocess


OUTPUT_FIELDS = (
    "candidate_id", "group", "trial", "implementation", "reference", "static_score",
    "mapping_id", "decomposition_count", "stages_per_decomposition", "segment_threads", "segment_ept",
    "boundary_twiddles", "boundary_layouts", "boundary_residencies", "execution_group_count",
    "execution_group_stages", "group_threads", "group_ept", "processing_unit",
    "prefix_log_n", "suffix_log_n", "prefix_threads",
    "suffix_threads", "prefix_ept", "suffix_ept", "cross_twiddle", "direct_boundary", "device",
    "compute_capability", "operator", "precision", "direction", "normalization", "placement",
    "backend", "fft_core", "logN", "N", "batch", "warmup", "repeat", "kernel_ms",
    "points_s", "max_error", "correct",
)


def parse_record(output):
    lines = [line for line in output.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "kernel_ms" in line and "operator" in line:
            rows = list(csv.DictReader(lines[index:]))
            if rows:
                return rows[0]
    raise ValueError(f"no benchmark CSV row found: {output[-500:]}")


def expand(document):
    cases = []
    shapes = {}
    for candidate in document["runnable_candidates"]:
        group = f"{candidate['mapping_id']}-batch{candidate['batch']}"
        cases.append({**candidate, "group": group, "implementation": candidate["id"], "reference": False})
        shapes[(candidate["mapping_id"], candidate["batch"])] = candidate
    for (mapping_id, batch), seed in shapes.items():
        cases.append({
            **seed, "id": f"{mapping_id}_b{batch}_cufft", "group": f"{mapping_id}-batch{batch}",
            "implementation": "cuFFT", "reference": True, "processing_unit": "vendor-baseline",
            "prefix_log_n": 0, "suffix_log_n": 0, "prefix_threads": 0, "suffix_threads": 0,
            "prefix_ept": 0, "suffix_ept": 0, "cross_twiddle": "none", "static_score": 0.0,
            "direct_boundary": "none", "decomposition_count": 0, "stages_per_decomposition": [],
            "segment_mappings": [], "boundaries": [], "execution_group_mappings": [],
            "args": ["--operator", "fft", "--precision", seed["precision"], "--backend", "cufft",
                     "--placement", seed["placement"], "--normalization", seed["normalization"]],
        })
    return cases


def execute(binary, case, protocol):
    command = [str(binary), *case["args"], "--logN", str(case["logN"]), "--batch", str(case["batch"]),
               "--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"]), "--verify", "--csv"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    row = parse_record(result.stdout)
    if row.get("correct") != "1":
        raise RuntimeError(f"correctness failed for {case['id']}: {row}")
    return row


def filter_existing(rows, cases):
    valid_ids = {case["id"] for case in cases}
    return [row for row in rows if row["candidate_id"] in valid_ids]


def case_metadata(case, trial):
    return {
        "candidate_id": case["id"], "group": case["group"], "trial": trial,
        "implementation": case["implementation"], "reference": int(case["reference"]),
        "static_score": case["static_score"], "mapping_id": case["mapping_id"],
        "decomposition_count": case["decomposition_count"],
        "stages_per_decomposition": "x".join(map(str, case["stages_per_decomposition"])),
        "segment_threads": "x".join(str(item["threads"]) for item in case.get("segment_mappings", [])),
        "segment_ept": "x".join(str(item["ept"]) for item in case.get("segment_mappings", [])),
        "boundary_twiddles": "x".join(item["cross_twiddle"] for item in case.get("boundaries", [])),
        "boundary_layouts": "x".join(item["layout"] for item in case.get("boundaries", [])),
        "boundary_residencies": "x".join(item.get("residency", "global-scratch")
                                           for item in case.get("boundaries", [])),
        "execution_group_count": case.get("execution_group_count", case["decomposition_count"]),
        "execution_group_stages": "x".join(map(str, case.get("execution_group_stages", []))),
        "group_threads": "x".join(str(item["threads"])
                                    for item in case.get("execution_group_mappings", [])),
        "group_ept": "x".join(str(item["ept"])
                                for item in case.get("execution_group_mappings", [])),
        "processing_unit": case["processing_unit"], "prefix_log_n": case["prefix_log_n"],
        "suffix_log_n": case["suffix_log_n"], "prefix_threads": case["prefix_threads"],
        "suffix_threads": case["suffix_threads"], "prefix_ept": case["prefix_ept"],
        "suffix_ept": case["suffix_ept"], "cross_twiddle": case["cross_twiddle"],
        "direct_boundary": case["direct_boundary"],
    }


def write_records(path, records):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Benchmark generated FFT mapping candidates and cuFFT.")
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_fft_pipeline_candidates.json"))
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/cubutterfly_bench"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/v100_fft_pipeline_raw.csv"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    cases = expand(document)
    if args.only:
        cases = [case for case in cases if any(token in case["id"] or token in case["group"] for token in args.only)]
    existing = []
    if args.resume and args.output.exists():
        with args.output.open() as source:
            existing = list(csv.DictReader(source))
        existing = filter_existing(existing, cases)
        cases_by_id = {case["id"]: case for case in cases}
        for row in existing:
            row.update(case_metadata(cases_by_id[row["candidate_id"]], int(row["trial"])))
    complete = {(row["candidate_id"], int(row["trial"])) for row in existing}
    pending = [(case, trial) for case in cases for trial in range(1, int(document["protocol"]["trials"]) + 1)
               if (case["id"], trial) not in complete]
    random.Random(int(document["protocol"]["seed"])).shuffle(pending)
    records = list(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        write_records(args.output, records)
    for index, (case, trial) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {case['id']} trial={trial}", flush=True)
        measured = execute(args.binary, case, document["protocol"])
        record = {field: measured.get(field, "") for field in OUTPUT_FIELDS}
        # Generated metadata owns these fields; benchmark output owns the timing fields.
        record.update(case_metadata(case, trial))
        records.append(record)
        write_records(args.output, records)


if __name__ == "__main__":
    main()
