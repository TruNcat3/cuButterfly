#!/usr/bin/env python3
"""Fresh, resumable GPU-NTT comparisons for the existing crypto matrix.

Historical records supply mapping proposals only. Both implementations are
measured again on the selected UUID, with the same complete semantic contract.
No calibration or search is performed by this comparison runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_crypto_composition as composition
import run_research_target as target

SCHEMA = "cubutterfly-crypto-external-v1"


def contract(row):
    result = dict(logN=int(row["logN"]), batch=int(row["batch"]),
                  word_bits=int(row["word_bits"]) if "word_bits" in row else int(row["precision"][4:]),
                  moduli=[str(int(p)) for p in row.get("moduli", [row.get("modulus")])],
                  mode=row.get("mode", "cyclic"), direction=row.get("direction", "forward"))
    if result["mode"] == "coset":
        result["coset_generator"] = int(row.get("coset_generator", 7))
    result["id"] = target.key(result)[:20]
    return result


def unavailable_reason(workload):
    if not 1 <= workload["logN"] <= 28:
        return "GPU-NTT merge dispatch supports logN 1..28"
    if not 1 <= workload["batch"] <= 65535:
        return "GPU-NTT merge batch exceeds CUDA grid.z limit"
    if workload["word_bits"] not in (32, 64):
        return "GPU-NTT adapter supports single-word 32/64-bit storage"
    max_bits = 30 if workload["word_bits"] == 32 else 62
    if any(not 2 < int(p) < 1 << max_bits for p in workload["moduli"]):
        return f"GPU-NTT Barrett word{workload['word_bits']} supports at most {max_bits}-bit moduli"
    return None


def features(w):
    """Select coverage before looking at any performance measurements."""
    bits = max(int(p).bit_length() for p in w["moduli"])
    return {("mode-word-direction", w["mode"], w["word_bits"], w["direction"]),
            ("bits-direction", bits, w["direction"]), ("logN", w["logN"]),
            ("batch", w["batch"]), ("channels", len(w["moduli"]))}


def input_bytes(w):
    return (1 << w["logN"]) * w["batch"] * len(w["moduli"]) * (w["word_bits"] // 8)


def pilot_ids(workloads):
    remaining = [w for w in workloads if unavailable_reason(w) is None]
    uncovered = set().union(*(features(w) for w in remaining)) if remaining else set()
    result = []
    while uncovered:
        chosen = min(remaining, key=lambda w: (-len(features(w) & uncovered), input_bytes(w), w["id"]))
        result.append(chosen["id"])
        uncovered -= features(chosen)
        remaining.remove(chosen)
    return result


def mapping_proposal(w, sources):
    mappings, provenance = [], []
    for modulus in w["moduli"]:
        key = composition.cyclic_cell_key(w, modulus)
        selected = None
        evidence = None
        for source in sources:
            cell = source["cells"].get(key)
            if cell is None or cell.get("status") != "complete":
                continue
            # A hash match alone is not sufficient for importing a proposal.
            from hardware_registry import canonical_semantics
            if canonical_semantics(cell["workload"]) != canonical_semantics(composition._cyclic_workload(w, modulus)):
                continue
            sample = composition._selected_sample(cell)
            if sample is None or str(sample.get("correct")).lower() not in {"true", "1"}:
                continue
            selected = composition._mapping_value(sample)
            if selected is not None:
                evidence = dict(kind="historical-mapping-proposal", cell_id=key,
                                source=source["path"], source_sha256=source["sha256"],
                                source_uuid=source["uuid"],
                                reuse="mapping only; all target timings and correctness measured afresh")
                break
        mappings.append(selected)
        provenance.append(evidence or dict(kind="public-default", cell_id=key))
    return (mappings if any(m is not None for m in mappings) else None), provenance


def prepare(args):
    workloads = {}
    source_files = {}
    validation = target.read(args.baseline_validation) if args.baseline_validation else None
    if validation:
        source_files[str(args.baseline_validation.resolve())] = target.sha(args.baseline_validation)
        for item in validation["modulus_failures"].values():
            evidence = args.baseline_validation.parent / item["evidence"]
            source_files[str(evidence.resolve())] = target.sha(evidence)
    for name in ("workloads.json", "composed_workloads.json"):
        path = args.campaign / name
        source_files[str(path.resolve())] = target.sha(path)
        for row in target.read(path)["workloads"]:
            w = contract(row)
            workloads[w["id"]] = w
    sources = []
    for path in args.mapping_source:
        data = target.read(path)
        sources.append(dict(path=str(path.resolve()), sha256=target.sha(path),
                            uuid=composition._source_uuid(data),
                            cells={c["id"]: c for c in data["cells"]}))
    rows = []
    for w in sorted(workloads.values(), key=lambda w: (input_bytes(w), w["id"])):
        mapping, provenance = mapping_proposal(w, sources)
        blocked = [dict(modulus=p, **validation["modulus_failures"][p]) for p in w["moduli"]
                   if validation and p in validation["modulus_failures"]]
        rows.append(dict(workload=w, mapping=mapping, mapping_provenance=provenance,
                         unavailable=unavailable_reason(w), baseline_validation_failures=blocked))
    document = dict(schema=SCHEMA, source_files=source_files, pilot_ids=pilot_ids(list(workloads.values())),
                    baseline_validation=validation,
                    cells=rows, scope="existing cyclic/negacyclic/coset single-word and sequential RNS contracts",
                    limitations=["Not a complete cryptographic application or CRT/HE benchmark.",
                                 "Historical selected mappings are proposals, not target calibration or timing evidence.",
                                 "Unsupported GPU-NTT cells are excluded with reasons, never counted as speedups."])
    target.freeze(args.output_dir / "matrix.json", document)
    return document


def external_identity(args, device):
    source = args.gpuntt_source.resolve()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("GPU-NTT source must be a clean checkout; use a separate build directory")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    inputs = {str(path.resolve()): target.sha(path) for path in
              (Path(__file__), ROOT / "scripts/run_crypto_composition.py", ROOT / "scripts/run_research_target.py")}
    for directory in (ROOT / "benchmarks/gpuntt_crypto", ROOT / "benchmarks/crypto_ntt"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".cu", ".cuh", ".hpp", ".txt"):
                inputs[str(path.resolve())] = target.sha(path)
    return dict(schema=SCHEMA, device=device, matrix_sha256=target.sha(args.output_dir / "matrix.json"),
                binaries={"cubutterfly": target.sha(args.binary), "gpuntt": target.sha(args.gpuntt_binary)},
                gpuntt=dict(url="https://github.com/Alisah-Ozcan/GPU-NTT", revision=revision),
                source_inputs=inputs,
                protocol=dict(trials=args.trials, warmup=args.warmup, repeat=args.repeat,
                              patterns=["random", "boundary"], seed=20260929),
                compile_mode="research", comparison="natural input and output, exact modular arithmetic",
                timing="CUDA events, includes order/twist/normalization; excludes setup, compilation and CPU oracle")


def freeze_binary(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if target.sha(source) != target.sha(destination):
        raise ValueError(f"binary changed on resume: {source}")
    return destination.resolve()


def validate_pair(ours, baseline, workload):
    for sample in (ours, baseline):
        channels = sample.get("channels", [])
        if len(channels) != len(workload["moduli"]):
            raise ValueError("missing per-channel semantic root evidence")
    for left, right, modulus in zip(ours["channels"], baseline["channels"], workload["moduli"]):
        for key in ("modulus", "root", "psi"):
            if key not in left or key not in right or int(left[key]) != int(right[key]):
                raise ValueError(f"paired channel {key} differs")
        if int(left["modulus"]) != int(modulus):
            raise ValueError("paired channel order differs")


def summarize(document):
    complete = [c for c in document["cells"] if c["status"] == "complete"]
    groups = defaultdict(list)
    for cell in complete:
        w = cell["workload"]
        ratio = cell["speedup"]
        for label in ("all-complete", w["mode"], w["direction"], f"word{w['word_bits']}",
                      "RNS" if len(w["moduli"]) > 1 else "single-modulus"):
            groups[label].append(ratio)
    statistics_by_group = {}
    for label, values in groups.items():
        statistics_by_group[label] = dict(count=len(values),
            geometric_mean=math.exp(statistics.fmean(math.log(v) for v in values)),
            median=statistics.median(values), minimum=min(values), maximum=max(values),
            wins=sum(v > 1 for v in values))
    return dict(schema=SCHEMA, counts=dict(Counter(c["status"] for c in document["cells"])),
                groups=statistics_by_group, device=document["identity"]["device"],
                protocol=document["identity"].get("protocol"),
                baseline="GPU-NTT Merge; sequential RNS composition where applicable",
                ratio="GPU-NTT time / cuButterfly time; larger than one favors cuButterfly",
                population="Completed pairs only; partial coverage is not the full-matrix result",
                failures=[dict(id=c["workload"]["id"], reason=c.get("error")) for c in document["cells"]
                          if c["status"] == "failed"],
                unavailable_reasons=dict(Counter(c.get("unavailable") for c in document["cells"]
                                                if c["status"] == "unavailable")),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S %z"))


def save(args, document):
    target.write(args.output_dir / "journal.json", document)
    target.write(args.output_dir / "summary.json", summarize(document))


def execute(args):
    matrix = target.read(args.output_dir / "matrix.json")
    device = target.target_device(args.gpu)
    identity = external_identity(args, device)
    if matrix.get("baseline_validation") and matrix["baseline_validation"]["revision"] != identity["gpuntt"]["revision"]:
        raise ValueError("baseline validation belongs to a different upstream revision")
    target.freeze(args.output_dir / "identity.json", identity)
    binaries = {"cubutterfly": freeze_binary(args.binary, args.output_dir / "bin/crypto_ntt_bench"),
                "gpuntt": freeze_binary(args.gpuntt_binary, args.output_dir / "bin/gpuntt_crypto_bench")}
    journal_path = args.output_dir / "journal.json"
    if journal_path.exists():
        document = target.read(journal_path)
        if document["identity"] != identity:
            raise ValueError("measurement identity changed; use a new output directory")
    else:
        document = dict(schema=SCHEMA, identity=identity, cells=[dict(row,
                        status="unavailable" if row["unavailable"] else
                               "blocked-baseline-correctness" if row.get("baseline_validation_failures") else "pending",
                        measurements=[])
                        for row in matrix["cells"]])
    protocol = identity["protocol"]
    pilot = matrix["pilot_ids"]
    ordered = sorted(document["cells"], key=lambda c: (c["workload"]["id"] not in pilot,
                     input_bytes(c["workload"]), c["workload"]["id"]))
    os.environ.update(CUDA_VISIBLE_DEVICES=device["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      CUBUTTERFLY_COMPILE_MODE="research")
    def interrupted_by_signal(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted_by_signal)
    save(args, document)
    with target.target_lock(device["uuid"]):
        target.exclusive(device["uuid"])
        for cell in ordered:
            w = cell["workload"]
            if args.scope == "pilot" and w["id"] not in pilot:
                continue
            if cell["status"] in ("complete", "unavailable", "blocked-baseline-correctness") or (cell["status"] == "failed" and not args.retry_failed):
                continue
            cell["status"] = "running"
            save(args, document)
            try:
                for pattern in protocol["patterns"]:
                    for trial in range(protocol["trials"]):
                        order = ("cubutterfly", "gpuntt") if trial % 2 == 0 else ("gpuntt", "cubutterfly")
                        pair = {}
                        for name in order:
                            mapping = cell["mapping"] if name == "cubutterfly" else None
                            previous = [m for m in cell["measurements"] if
                                        (m["library"], m["pattern"], m["trial"]) == (name, pattern, trial)]
                            if len(previous) > 1:
                                raise ValueError("duplicate measurement key")
                            if previous:
                                sample = previous[0]["sample"]
                            else:
                                command = composition._command(binaries[name], w, pattern, trial, protocol, mapping)
                                log = args.output_dir / "logs" / f"{w['id']}-{name}-{pattern}-{trial}.log"
                                result = target.guarded(command, device["uuid"], log)
                                sample = json.loads(result.stdout)
                            composition.validate_sample(sample, w, device["uuid"], pattern, trial, protocol, mapping)
                            if name == "gpuntt" and (sample.get("baseline") != "GPU-NTT" or
                                    identity["gpuntt"]["revision"] not in sample.get("source", "")):
                                raise ValueError("external executable does not match the pinned GPU-NTT source")
                            pair[name] = sample
                            if not previous:
                                cell["measurements"].append(dict(library=name, pattern=pattern, trial=trial, sample=sample))
                                save(args, document)
                        validate_pair(pair["cubutterfly"], pair["gpuntt"], w)
                # Boundary values test reduction edge cases; headline timings use
                # the random-input trials so duplicated patterns are not extra cells.
                times = {name: statistics.median(m["sample"]["kernel_ms"] for m in cell["measurements"]
                                                if m["library"] == name and m["pattern"] == "random")
                         for name in binaries}
                cell.update(status="complete", median_ms=times, speedup=times["gpuntt"] / times["cubutterfly"])
                cell.pop("error", None)
                print(json.dumps(dict(id=w["id"], logN=w["logN"], batch=w["batch"], mode=w["mode"],
                                      word_bits=w["word_bits"], direction=w["direction"],
                                      channels=len(w["moduli"]), speedup=cell["speedup"])), flush=True)
            except BaseException as error:
                message = str(error)
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit)) or any(
                    term in message for term in ("GPU busy", "foreign GPU clients", "exclusivity monitor failed"))
                cell.update(status="interrupted" if interrupted else "failed", error=message)
                save(args, document)
                if interrupted:
                    raise
                print(json.dumps(dict(id=w["id"], status="failed", error=message)), flush=True)
            save(args, document)
    print(json.dumps(summarize(document), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "status"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, default=ROOT / "results/crypto_application_20260920")
    parser.add_argument("--mapping-source", type=Path, action="append", default=[])
    parser.add_argument("--baseline-validation", type=Path,
                        help="Optional documented primitive validation failures from the pinned upstream revision")
    parser.add_argument("--gpu")
    parser.add_argument("--binary", type=Path, default=ROOT / "results/crypto_application_20260920/build/crypto_ntt_bench")
    parser.add_argument("--gpuntt-binary", type=Path)
    parser.add_argument("--gpuntt-source", type=Path)
    parser.add_argument("--scope", choices=("pilot", "all"), default="all")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.trials < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("invalid timing protocol")
    if args.mode == "prepare":
        document = prepare(args)
        print(json.dumps(dict(total=len(document["cells"]), pilot=len(document["pilot_ids"]),
                              unavailable=sum(bool(c["unavailable"]) for c in document["cells"]),
                              blocked=sum(bool(c["baseline_validation_failures"]) and not c["unavailable"]
                                          for c in document["cells"]))))
    elif args.mode == "status":
        print(json.dumps(summarize(target.read(args.output_dir / "journal.json")), indent=2))
    else:
        if not all((args.gpu, args.gpuntt_binary, args.gpuntt_source)):
            parser.error("run requires --gpu, --gpuntt-binary and --gpuntt-source")
        execute(args)


if __name__ == "__main__":
    main()
