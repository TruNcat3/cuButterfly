"""Atomic, incremental mapping registry shared by installation and runtime.

Workload semantics and full resolved mappings are stored separately. Calibration
history remains attached to its hardware; importing another device never erases it.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = "cubutterfly-registry-v1"
SEMANTICS = ("operator", "precision", "placement", "logN", "batch", "modulus",
             "accumulation", "direction", "normalization", "element_stride",
             "batch_stride", "stage_matrices", "input_order", "output_order")


def canonical_semantics(sample):
    result = {key: str(sample[key]) for key in SEMANTICS if key in sample}
    result.setdefault("operator", "ntt" if "word_bits" in sample else "fft")
    if result["operator"] == "ntt":
        result.setdefault("precision", "word" + str(sample.get("word_bits", 64)))
    if "inverse" in sample:
        result.setdefault("direction", "inverse" if str(sample["inverse"]).lower() in ("1", "true") else "forward")
    for key, default in (("direction", "forward"), ("accumulation", "native"),
                         ("normalization", "none"), ("element_stride", "1"),
                         ("stage_matrices", ""), ("modulus", "0")):
        result.setdefault(key, default)
    if "logN" in result:
        if int(result.get("batch_stride", 0)) == 0:
            result["batch_stride"] = str(((1 << int(result["logN"])) - 1) * int(result["element_stride"]) + 1)
    if result["direction"] == "forward" or result["operator"] not in ("fft", "fwht"):
        result["normalization"] = "none"
    if result["operator"] == "ntt":
        result["input_order"] = result.get("input_order", "natural")
        result["output_order"] = result.get("output_order", result.get("placement", "natural"))
        result["placement"] = result["output_order"]
    return result


def identity(profile):
    return {"device": str(profile["device"]), "compute_capability": str(profile["compute_capability"]),
            "global_memory_bytes": int(profile["global_memory_bytes"])}


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def default_path(prefix=None):
    override = os.environ.get("CUBUTTERFLY_REGISTRY")
    if override:
        return Path(override)
    if prefix:
        return Path(prefix) / "share/cuButterfly/hardware/registry.json"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cubutterfly/registry.json"


@contextlib.contextmanager
def locked_registry(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        document = json.loads(path.read_text()) if path.exists() else {"schema": SCHEMA, "targets": []}
        if document.get("schema") != SCHEMA or not isinstance(document.get("targets"), list):
            raise ValueError("unsupported hardware registry schema")
        yield document
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(document, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def promote(path, profile, calibration, *, verified=True):
    """Merge correct measured candidates; never promote a whole-library baseline."""
    hardware = identity(profile)
    candidates = calibration.get("candidates", []) if isinstance(calibration, dict) else calibration
    added = 0
    with locked_registry(path) as document:
        target = next((entry for entry in document["targets"] if entry["hardware"] == hardware), None)
        if target is None:
            target = {"hardware": hardware, "records": []}
            document["targets"].append(target)
        records = {record["record_id"]: record for record in target["records"]}
        imported = {}
        for candidate in candidates:
            if candidate.get("status") != "measured" or not candidate.get("correct") or not candidate.get("samples"):
                continue
            sample = candidate["samples"][0]
            if sample.get("backend") == "cufft":
                continue
            latency = float(candidate["median_kernel_ms"])
            if not math.isfinite(latency) or latency <= 0:
                continue
            key = canonical_semantics(sample)
            # New benchmarks export the complete mapping. Runtime timings and
            # descriptive fields must not change the identity of a design point.
            mapping = (json.loads(sample["mapping_json"]) if sample.get("mapping_json") else
                       {k: str(v) for k, v in sample.items() if k not in SEMANTICS and
                        k not in {"kernel_ms", "correct", "max_error", "h2d_ms", "d2h_ms",
                                  "transforms_s", "points_s", "Gbutterfly_s", "warmup", "repeat"}})
            if mapping.get("backend") == "cufft":
                continue
            fingerprint = sample.get("runtime_fingerprint", "")
            complete = mapping.get("schema_version") == 1 and bool(fingerprint)
            record_id = stable_id({"key": key, "mapping": mapping, "runtime_fingerprint": fingerprint})
            if record_id in imported and imported[record_id] <= latency:
                continue
            imported[record_id] = latency
            record = {"record_id": record_id, "key": key, "mapping": mapping,
                      "runtime_fingerprint": fingerprint,
                      "status": "measured" if verified and complete else "needs-revalidation",
                      "candidate_id": candidate["name"], "kernel_ms": latency,
                      "evidence": {k: candidate[k] for k in ("command", "trials", "samples", "binary_sha256") if k in candidate}}
            # Replacement is a fresh measurement of the same mapping; do not
            # take the minimum of historical noisy timings.
            records[record_id] = record
            added += 1
        target["records"] = [records[key] for key in sorted(records)]
    return added


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=default_path())
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--needs-revalidation", action="store_true")
    args = parser.parse_args()
    if bool(args.profile) != bool(args.calibration):
        parser.error("profile and calibration must be supplied together")
    if args.profile:
        print(promote(args.registry, json.loads(args.profile.read_text()),
                      json.loads(args.calibration.read_text()), verified=not args.needs_revalidation))
    else:
        print(args.registry.read_text() if args.registry.exists() else json.dumps({"schema": SCHEMA, "targets": []}))


if __name__ == "__main__":
    main()
