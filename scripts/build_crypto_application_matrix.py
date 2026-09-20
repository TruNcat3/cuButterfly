#!/usr/bin/env python3
"""Build a finite cryptographic NTT application matrix.

This generator is CPU-only.  It separates three things which are often
accidentally conflated in crypto benchmarks:

* ``word_bits`` is the machine storage/arithmetic container;
* ``modulus`` and its bit length identify one finite field modulus;
* an RNS entry is a list of independent moduli, not one wider machine word.

The cyclic part intentionally uses the existing install-search workload
schema.  Negacyclic, coset, and RNS entries are contracts for a future
composition runner and are never presented as measured or currently
implemented support.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from run_research_acceptance import cell_key  # noqa: E402


SCHEMA = "crypto-application-study-v1"
WORKLOAD_SCHEMA = "cubutterfly-install-search-v1"
MAX_INPUT_BYTES = 2 * 1024**3

# These are finite workload choices, not claims that every transform length
# used by a protocol is covered.  Dilithium is kept at its application-shaped
# N=256 point; its p-1 still permits cyclic lengths through logN=13.
BATCHES = (1, 3, 7, 16, 256)
DIRECT_LOGS = {
    "dilithium23": (8,),
    "synthetic30": (12, 16, 18),
    "babybear31": (12, 16, 20),
    "koalabear31": (12, 16, 20),
    "synthetic40": (12, 16, 20),
    "synthetic50": (12, 16, 20),
    "synthetic60": (12, 16, 20),
    "synthetic62": (12, 16, 20),
}


def _miller_rabin(n: int) -> bool:
    """Deterministic Miller-Rabin for the unsigned 64-bit range."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # This published seven-base set is deterministic for n < 2^64.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        base %= n
        if base == 0:
            continue
        value = pow(base, d, n)
        if value in (1, n - 1):
            continue
        for _ in range(s - 1):
            value = value * value % n
            if value == n - 1:
                break
        else:
            return False
    return True


def _two_adicity(value: int) -> int:
    count = 0
    while value % 2 == 0:
        value //= 2
        count += 1
    return count


def _find_prime(bit_width: int, two_adicity: int) -> int:
    """Find the first p=k*2^a+1 at the requested bit width.

    The search order and Miller-Rabin basis are fixed, so the generated
    synthetic moduli are reproducible without an external prime package.
    ``k`` starts odd because an even k would make p composite for a>0.
    """
    if bit_width >= 64 or two_adicity < 1:
        raise ValueError("synthetic modulus must fit below 2^64")
    low_k = ((1 << (bit_width - 1)) + (1 << two_adicity) - 1) >> two_adicity
    high_k = ((1 << bit_width) - 2) >> two_adicity
    for k in range(max(1, low_k) | 1, high_k + 1, 2):
        candidate = (k << two_adicity) + 1
        if _miller_rabin(candidate):
            return candidate
    raise RuntimeError("no deterministic prime found for bit_width={} two_adicity={}"
                       .format(bit_width, two_adicity))


def _synthetic_specs():
    # A >= 30 leaves room for logN=20 negacyclic contracts while keeping the
    # resulting p below the runtime's word64 p<2^63 validation boundary.
    choices = ((40, 30), (50, 40), (60, 50), (62, 52))
    return {
        f"synthetic{bits}": {
            "name": f"synthetic{bits}",
            "modulus": _find_prime(bits, adicity),
            "bits": bits,
            "word_bits": 64,
            "two_adicity": adicity,
            "application": "synthetic-primitive-root-ntt",
            "source": "deterministic Miller-Rabin p=k*2^a+1 search",
        }
        for bits, adicity in choices
    }


def modulus_specs():
    """Return the named real/application and deterministic synthetic plans."""
    specs = {
        "dilithium23": {
            "name": "dilithium23",
            "modulus": 8380417,
            "bits": 23,
            "word_bits": 32,
            "two_adicity": 13,
            "application": "Dilithium",
            "source": "https://github.com/pq-crystals/dilithium/blob/master/ref/params.h",
        },
        "synthetic30": {
            "name": "synthetic30",
            "modulus": 1073479681,
            "bits": 30,
            "word_bits": 32,
            "two_adicity": 18,
            "application": "synthetic-word32-ntt",
            "source": "deterministic Miller-Rabin validation of the fixed modulus",
        },
        "babybear31": {
            "name": "babybear31",
            "modulus": 2013265921,
            "bits": 31,
            "word_bits": 32,
            "two_adicity": 27,
            "application": "BabyBear",
            "source": "https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/baby_bear.rs",
        },
        "koalabear31": {
            "name": "koalabear31",
            "modulus": 2130706433,
            "bits": 31,
            "word_bits": 32,
            "two_adicity": 24,
            "application": "KoalaBear",
            "source": "https://github.com/Plonky3/Plonky3/blob/main/koala-bear/src/koala_bear.rs",
        },
    }
    specs.update(_synthetic_specs())
    for spec in specs.values():
        _validate_spec(spec)
    return specs


def _validate_spec(spec):
    p = int(spec["modulus"])
    bits = int(spec["bits"])
    word_bits = int(spec["word_bits"])
    if p.bit_length() != bits:
        raise ValueError(f"modulus bit width mismatch: {spec['name']}")
    if not _miller_rabin(p):
        raise ValueError(f"modulus is not prime: {spec['name']}")
    if _two_adicity(p - 1) != int(spec["two_adicity"]):
        raise ValueError(f"modulus 2-adicity mismatch: {spec['name']}")
    if p >= 1 << 63:
        raise ValueError(f"cyclic runtime modulus exceeds p<2^63: {spec['name']}")
    if word_bits == 32 and p >= 1 << 31:
        raise ValueError(f"word32 modulus exceeds p<2^31: {spec['name']}")
    if word_bits not in (32, 64):
        raise ValueError(f"unsupported machine word width: {word_bits}")


def valid_log_n(spec, log_n, *, mode="cyclic"):
    """Check primitive root availability for a power-of-two transform."""
    log_n = int(log_n)
    required = log_n + (1 if mode == "negacyclic" else 0)
    return 0 <= log_n and required <= int(spec["two_adicity"])


def _input_bytes(log_n, batch, word_bits, channels=1):
    return (1 << int(log_n)) * int(batch) * (int(word_bits) // 8) * int(channels)


def _workload_word_bits(workload):
    if "word_bits" in workload:
        return int(workload["word_bits"])
    precision = str(workload.get("precision", ""))
    if precision in ("word32", "uint32"):
        return 32
    if precision in ("word64", "uint64"):
        return 64
    raise ValueError(f"workload has no machine word width: {workload}")


def input_bytes(workload):
    """Logical bytes for a cyclic workload."""
    return _input_bytes(workload["logN"], workload["batch"], _workload_word_bits(workload))


def _ntt_workload(spec, log_n, batch, direction):
    return {
        "operator": "ntt",
        "precision": f"word{spec['word_bits']}",
        "logN": int(log_n),
        "batch": int(batch),
        "modulus": str(spec["modulus"]),
        "input_order": "natural",
        "output_order": "natural",
        "direction": direction,
    }


def _direct_workloads(specs):
    rows = []
    for name in specs:
        spec = specs[name]
        for log_n in DIRECT_LOGS[name]:
            if not valid_log_n(spec, log_n):
                raise ValueError(f"cyclic root unavailable for {name} logN={log_n}")
            for batch in BATCHES:
                for direction in ("forward", "inverse"):
                    row = _ntt_workload(spec, log_n, batch, direction)
                    if input_bytes(row) > MAX_INPUT_BYTES:
                        raise ValueError(f"cyclic workload exceeds memory cap: {row}")
                    rows.append(row)
    rows.sort(key=_workload_sort_key)
    return rows


def _composition_log(spec):
    # Every modulus gets one contract at a size that satisfies its actual
    # two-adicity.  The 23-bit Dilithium plan stays at N=256; wider plans use
    # N=65536 to expose the larger root-capacity without a large cross-product.
    return 8 if int(spec["bits"]) == 23 else 16


def _composition_entry(mode, spec, log_n, batch, direction, *, coset=None):
    moduli = [str(spec["modulus"])]
    payload = {
        "mode": mode,
        "logN": int(log_n),
        "batch": int(batch),
        "word_bits": int(spec["word_bits"]),
        "moduli": moduli,
        "direction": direction,
        "contract_id": "",
        "expected": "planned-composition-benchmark",
        "status": "planned",
        "composition": "single-modulus",
        "suites": ["single-modulus"],
        "rns_channel_count": 1,
        "rns_stream_order": "single-stream",
        "rns_memberships": [],
        "application": spec["application"],
        "modulus_bits": [int(spec["bits"])],
        "source": [spec["source"]],
    }
    if coset is not None:
        if not _valid_coset_generator(spec, log_n, coset):
            raise ValueError(f"coset generator is an N-th root for {spec['name']} logN={log_n}")
        payload["coset_generator"] = int(coset)
    payload["input_bytes"] = _input_bytes(log_n, batch, spec["word_bits"])
    return payload


def _valid_coset_generator(spec, log_n, generator):
    modulus = int(spec["modulus"])
    return (int(generator) % modulus != 0 and
            pow(int(generator), 1 << int(log_n), modulus) != 1)


def _rns_entry(name, specs, log_n, batch, direction):
    word_bits = int(specs[0]["word_bits"])
    if any(int(spec["word_bits"]) != word_bits for spec in specs):
        raise ValueError(f"RNS plan mixes machine word widths: {name}")
    if any(not valid_log_n(spec, log_n, mode="negacyclic") for spec in specs):
        raise ValueError(f"RNS plan lacks a 2N negacyclic root: {name} logN={log_n}")
    moduli = [str(spec["modulus"]) for spec in specs]
    return {
        # RNS is a channel/list dimension of a negacyclic contract, not a
        # fourth mathematical transform mode.
        "mode": "negacyclic",
        "logN": int(log_n),
        "batch": int(batch),
        "word_bits": word_bits,
        "moduli": moduli,
        "direction": direction,
        "contract_id": name,
        "expected": "planned-composition-benchmark",
        "status": "planned",
        "composition": "rns",
        "suites": ["rns-prefix"],
        "rns_channel_count": len(moduli),
        "rns_stream_order": "single-stream",
        "rns_memberships": [name],
        "modulus_bits": [int(spec["bits"]) for spec in specs],
        "source": [spec["source"] for spec in specs],
        "input_bytes": _input_bytes(log_n, batch, word_bits, len(moduli)),
    }


def _composed_workloads(specs):
    rows = []
    # These contracts cover one valid negacyclic point per modulus.  The
    # batch pair is deliberately smaller than the public timing matrix because
    # these rows are not executable by the current public runner.
    for spec in specs.values():
        log_n = _composition_log(spec)
        if not valid_log_n(spec, log_n, mode="negacyclic"):
            raise ValueError(f"negacyclic root unavailable for {spec['name']}")
        for mode in ("negacyclic", "coset"):
            for batch in (1, 7, 16):
                for direction in ("forward", "inverse"):
                    rows.append(_composition_entry(mode, spec, log_n, batch, direction,
                                                   coset=7 if mode == "coset" else None))

    # Fixed base sequences expose L=1/2/4 independently of batch B.  The
    # large sequence is also the mixed 40/50/60/62-bit FHE-style probe.
    rns_plans = (
        ("rns-word32-prefix", ("dilithium23", "synthetic30", "babybear31", "koalabear31"), 8),
        ("rns-word64-prefix", ("synthetic40", "synthetic50", "synthetic60", "synthetic62"), 16),
        ("rns-word64-mixed-large", ("synthetic40", "synthetic50", "synthetic60", "synthetic62"), 20),
    )
    for name, members, log_n in rns_plans:
        member_specs = [specs[key] for key in members]
        for length in (1, 2, 4):
            for batch in (1, 7, 16):
                prefix_specs = member_specs[:length]
                for direction in ("forward", "inverse"):
                    rows.append(_rns_entry(name + f"-L{length}", prefix_specs, log_n, batch, direction))

    for row in rows:
        row["contract_id"] = _composition_id(row)
        row["id"] = row["contract_id"]
        if row["input_bytes"] > MAX_INPUT_BYTES:
            raise ValueError(f"composition workload exceeds memory cap: {row}")
    return _merge_composed_rows(rows)


def _merge_composed_rows(rows):
    """Merge mathematical duplicates while preserving composition memberships.

    An RNS prefix of length one is the same single-modulus transform as its
    corresponding standalone row.  It remains visible through ``suites`` and
    ``rns_memberships`` so L=1 curves are not silently discarded.
    """
    merged = {}
    for row in rows:
        key = row["id"]
        current = merged.get(key)
        if current is None:
            current = copy.deepcopy(row)
            current["suites"] = sorted(set(current.get("suites", ())))
            current["rns_memberships"] = sorted(set(current.get("rns_memberships", ())))
            merged[key] = current
            continue
        current["suites"] = sorted(set(current.get("suites", ())) |
                                   set(row.get("suites", ())))
        current["rns_memberships"] = sorted(set(current.get("rns_memberships", ())) |
                                              set(row.get("rns_memberships", ())))
        current["rns_channel_count"] = max(int(current.get("rns_channel_count", 1)),
                                             int(row.get("rns_channel_count", 1)))
        if row.get("composition") == "rns":
            current["composition"] = "rns"
        # These are normalized for every composition row, so a merge never
        # changes scheduling semantics or creates a second timing contract.
        current["rns_stream_order"] = "single-stream"
        current["contract_id"] = key
        current["id"] = key
    result = list(merged.values())
    result.sort(key=lambda row: row["contract_id"])
    return result


def _composition_id(row):
    identity = {key: row[key] for key in (
        "mode", "logN", "batch", "word_bits", "moduli", "direction",
        "coset_generator") if key in row}
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:20]


def _workload_sort_key(workload):
    return (int(workload["batch"]), int(workload["logN"]),
            _workload_word_bits(workload), str(workload.get("modulus", "")),
            str(workload.get("direction", "forward")), cell_key(workload))


def _suite_for(spec, direction):
    return ["cyclic", spec["name"], direction]


def _original_check(original):
    if original.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("original workload matrix must use cubutterfly-install-search-v1")
    rows = original.get("workloads")
    if not isinstance(rows, list) or not rows:
        raise ValueError("original workload matrix has no workloads")
    ids = [cell_key(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("original workload matrix contains duplicate semantic cells")
    return rows, set(ids)


def _study_label(path):
    path = Path(path)
    return path.parent.name if path.name == "workloads.json" else path.stem


def _collect_existing(original_path, reuse_workloads=()):
    """Merge existing studies by exact semantic cell, retaining provenance."""
    if isinstance(reuse_workloads, (str, Path)):
        reuse_workloads = (reuse_workloads,)
    paths = [Path(original_path), *[Path(path) for path in reuse_workloads]]
    merged = []
    ids = set()
    source_studies = defaultdict(list)
    for path in paths:
        rows, _ = _original_check(json.loads(path.expanduser().resolve().read_text()))
        label = _study_label(path)
        for row in rows:
            key = cell_key(row)
            if label not in source_studies[key]:
                source_studies[key].append(label)
            if key not in ids:
                ids.add(key)
                merged.append(row)
    return merged, ids, dict(source_studies)


def _population(original_rows, direct_rows, specs, source_studies):
    original_ids = {cell_key(row) for row in original_rows}
    direct_by_id = {cell_key(row): row for row in direct_rows}
    if len(direct_by_id) != len(direct_rows):
        raise ValueError("crypto cyclic matrix contains duplicate semantic cells")
    cells = []
    for row in original_rows:
        if row.get("operator") == "ntt":
            original_bytes = input_bytes(row)
        else:
            precision_bytes = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}.get(
                str(row.get("precision", "fp32")), 4)
            element_bytes = precision_bytes * (2 if row.get("operator") == "fft" else 1)
            original_bytes = (1 << int(row["logN"])) * int(row["batch"]) * element_bytes
        cells.append({"id": cell_key(row), "workload": copy.deepcopy(row),
                      "suites": [], "source": "original",
                      "input_bytes": original_bytes,
                      "source_studies": list(source_studies.get(cell_key(row), ()))})
    for row in direct_rows:
        key = cell_key(row)
        if key in original_ids:
            continue
        spec = next(spec for spec in specs.values() if str(spec["modulus"]) == row["modulus"])
        cells.append({"id": key, "workload": copy.deepcopy(row),
                      "suites": _suite_for(spec, row["direction"]),
                      "source": "extension", "input_bytes": input_bytes(row),
                      "modulus_bits": int(spec["bits"]),
                      "word_bits": int(spec["word_bits"]),
                      "two_adicity": int(spec["two_adicity"]),
                      "application": spec["application"]})
    return cells


def _precompile(direct_rows, composed_rows, specs):
    rows = {}
    for row in direct_rows:
        rows.setdefault(cell_key({**row, "batch": 1}),
                        {**row, "batch": 1})
    for entry in composed_rows:
        for modulus in entry["moduli"]:
            spec = next(spec for spec in specs.values() if str(spec["modulus"]) == modulus)
            row = _ntt_workload(spec, entry["logN"], 1, entry["direction"])
            rows.setdefault(cell_key(row), row)
    result = list(rows.values())
    result.sort(key=_workload_sort_key)
    return result


def unsupported():
    """Capabilities recorded explicitly without putting them in timed rows."""
    return [
        {
            "category": "goldilocks-full-word",
            "modulus": "18446744069414584321",
            "modulus_bits": 64,
            "status": "unsupported",
            "reason": "current cuButterfly NTT validation requires p < 2^63",
            "source": "https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/goldilocks.rs",
        },
        {
            "category": "bn254-scalar-field",
            "modulus_bits": 254,
            "status": "unsupported",
            "reason": "requires a multi-limb field element and arithmetic core",
            "source": "https://github.com/arkworks-rs/curves/blob/master/bn254/src/fields/fr.rs",
        },
        {
            "category": "bls12-381-scalar-field",
            "modulus_bits": 255,
            "status": "unsupported",
            "reason": "requires a multi-limb field element and arithmetic core",
            "source": "https://github.com/arkworks-rs/curves/blob/master/bls12_381/src/fields/fr.rs",
        },
        {
            "category": "kyber-incomplete-ntt",
            "modulus": "3329",
            "modulus_bits": 12,
            "status": "unsupported",
            "reason": "Kyber uses an incomplete protocol-specific NTT; it is not aliased to ordinary cyclic/negacyclic NTT",
            "source": "https://github.com/pq-crystals/kyber/blob/master/ref/params.h",
        },
        {
            "category": "extension-field",
            "modulus_bits": "base-field-dependent",
            "status": "unsupported",
            "reason": "extension-field coordinates require an explicit multi-coordinate arithmetic contract",
            "source": "https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/baby_bear.rs",
        },
    ]


def _limitations():
    return [
        "Finite application-oriented matrix; it is not an exhaustive protocol corpus.",
        "Only cyclic entries are compatible with the current public acceptance runner.",
        "Negacyclic, coset, and RNS rows are planned composition contracts and are not measured support.",
        "Synthetic 40/50/60/62-bit moduli are deterministic primitive-root probes, not named protocol fields.",
        "Missing multi-limb, extension-field, Goldilocks, and Kyber-incomplete lowerings remain explicit unsupported scope.",
    ]


def build_documents(original_path, reuse_workloads=()):
    original_rows, original_ids, source_studies = _collect_existing(
        original_path, reuse_workloads)
    specs = modulus_specs()
    direct = _direct_workloads(specs)
    direct_new = [row for row in direct if cell_key(row) not in original_ids]
    composed = _composed_workloads(specs)
    population = {
        "schema": SCHEMA,
        "scope": "Finite cryptographic NTT application matrix",
        "cells": _population(original_rows, direct, specs, source_studies),
        "unsupported": unsupported(),
        "limits": {
            "max_input_bytes": MAX_INPUT_BYTES,
            "batches": list(BATCHES),
            "cyclic_count": len(direct_new),
            "composed_count": len(composed),
            "reused_studies": sorted(set(label for values in source_studies.values()
                                          for label in values)),
        },
        "moduli": list(specs.values()),
        "limitations": _limitations(),
    }
    workloads = {
        "schema": WORKLOAD_SCHEMA,
        "scope": "New cyclic cryptographic NTT cells only; exact cell_key compatible",
        "limitations": _limitations(),
        "workloads": copy.deepcopy(direct_new),
    }
    composed_document = {
        "schema": "cubutterfly-composed-workload-v1",
        "scope": "Planned negacyclic, coset, and single-stream RNS composition contracts",
        "limitations": _limitations(),
        "workloads": composed,
    }
    precompile = {
        "schema": WORKLOAD_SCHEMA,
        "scope": "Batch-one cyclic subplans for cyclic and composed application contracts",
        "limitations": _limitations(),
        "workloads": _precompile(direct, composed, specs),
    }
    unsupported_document = {
        "schema": "cubutterfly-unsupported-capabilities-v1",
        "scope": "Explicitly unsupported or deferred cryptographic application categories",
        "entries": unsupported(),
    }
    return {
        "population.json": population,
        "workloads.json": workloads,
        "composed_workloads.json": composed_document,
        "precompile_workloads.json": precompile,
        "unsupported.json": unsupported_document,
    }


def _render(document):
    return (json.dumps(document, indent=2, sort_keys=True,
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_frozen(output_dir, documents):
    output_dir = Path(output_dir).expanduser().resolve()
    rendered = {name: _render(document) for name, document in documents.items()}
    for name, content in rendered.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite frozen output: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        path = output_dir / name
        if path.exists():
            continue
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)


def generate(original_path, output_dir, reuse_workloads=()):
    documents = build_documents(original_path, reuse_workloads)
    write_frozen(output_dir, documents)
    return documents


build = generate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reuse-workloads", type=Path, action="append", default=[],
                        help="existing workload matrix to merge for exact-key deduplication; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generate(args.original, args.output_dir, args.reuse_workloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
