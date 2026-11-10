#!/usr/bin/env python3
"""Extract actual RNS prepare observations, then prepare their cache offline.

extract is CPU-only. Counts describe preparation observations, never kernel or
application execution counts. prepare reports preparation wall time, not NTT
performance. Native plan construction validates request legality; exact output
validation is a separate step and is not established by cache preparation.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


OBSERVATION_SCHEMA = "cubutterfly-rns-request-v1"
MANIFEST_SCHEMA = "cubutterfly-rns-request-manifest-v1"
RESULT_SCHEMA = "cubutterfly-rns-preparation-results-v1"
COUNT_KIND = "prepare_observations_not_execution_counts"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} requires exactly these fields: {', '.join(expected)}")


def integer(value, name, bits=64, minimum=0):
    if type(value) is not int or not minimum <= value < 1 << bits:
        raise ValueError(f"{name} must be a uint{bits} integer >= {minimum}")


def choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"invalid {name}: {value!r}")


def validate_config(config):
    fields(config, ("schema_version", "log_n", "inverse", "ring", "input_order",
                    "output_order", "domains", "polynomials"), "config")
    integer(config["schema_version"], "schema_version", 32)
    if config["schema_version"] != 1:
        raise ValueError("unsupported config schema_version")
    integer(config["log_n"], "log_n", 32)
    if type(config["inverse"]) is not bool:
        raise ValueError("inverse must be bool")
    choice(config["ring"], ("cyclic", "negacyclic"), "ring")
    for key in ("input_order", "output_order"):
        choice(config[key], ("natural", "bit_reversed"), key)
    for key in ("domains", "polynomials"):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty array")
    for domain in config["domains"]:
        fields(domain, ("modulus", "root"), "domain")
        for key in domain:
            integer(domain[key], key)
    for polynomial in config["polynomials"]:
        fields(polynomial, ("domain", "input_offset", "output_offset"), "polynomial")
        integer(polynomial["domain"], "domain index", 32)
        if polynomial["domain"] >= len(config["domains"]):
            raise ValueError("polynomial domain index is outside domains")
        for key in ("input_offset", "output_offset"):
            integer(polynomial[key], key)


def validate_mapping(mapping):
    unsigned = ("n1_log", "rows_per_block", "threads_per_block",
                "second_rows_per_block", "second_threads_per_block")
    enums = {"compute_unit": ("radix2", "radix4", "radix8"),
             "cross_twiddle": ("first_pass", "second_pass"),
             "core_family": ("hybrid2d", "fused_ctgs", "gpu_ntt_merge"),
             "arithmetic": ("shoup", "barrett")}
    expected = ("schema_version", *unsigned, *enums, "physical_boundary_traversal")
    fields(mapping, (*expected, "exchange") if isinstance(mapping, dict) and "exchange" in mapping else expected,
           "mapping")
    integer(mapping["schema_version"], "mapping schema_version", 32)
    if mapping["schema_version"] != 1:
        raise ValueError("unsupported mapping schema_version")
    for key in unsigned:
        integer(mapping[key], key, 32)
    for key, values in enums.items():
        choice(mapping[key], values, key)
    # Missing exchange in schema-v1 observations means block. Preserve the
    # original record so extraction does not rewrite historical observations.
    choice(mapping.get("exchange", "block"), ("block", "dependency_scoped"), "exchange")
    if type(mapping["physical_boundary_traversal"]) is not bool:
        raise ValueError("physical_boundary_traversal must be bool")


def request_id(config, placement):
    payload = canonical({"config": config, "placement": placement}).encode("utf-8")
    return "rns-" + hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract(inputs, output):
    output = Path(output).resolve()
    requests = {}
    sources = []
    for source in inputs:
        source = Path(source).resolve()
        sources.append(str(source))
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    observation = loads(line)
                    fields(observation, ("schema", "origin", "placement", "config", "mapping", "cache_hit"),
                           "observation")
                    if observation["schema"] != OBSERVATION_SCHEMA:
                        raise ValueError("unsupported observation schema")
                    choice(observation["origin"], ("scalar", "stage"), "origin")
                    choice(observation["placement"], ("in_place", "separate"), "placement")
                    if type(observation["cache_hit"]) is not bool:
                        raise ValueError("cache_hit must be bool")
                    validate_config(observation["config"])
                    validate_mapping(observation["mapping"])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{source}:{line_number}: {error}") from error
                config, placement = observation["config"], observation["placement"]
                identity = request_id(config, placement)
                if identity not in requests:
                    requests[identity] = {
                        "id": identity, "config": config, "placement": placement,
                        "request_file": f"requests/{identity}.json", "observation_count": 0,
                        "origins": {}, "cache_hit_observation_count": 0,
                        "observed_mappings": {},
                    }
                entry = requests[identity]
                entry["observation_count"] += 1
                origin = observation["origin"]
                entry["origins"][origin] = entry["origins"].get(origin, 0) + 1
                entry["cache_hit_observation_count"] += int(observation["cache_hit"])
                key = canonical(observation["mapping"])
                mappings = entry["observed_mappings"]
                if key not in mappings:
                    mappings[key] = {"mapping": observation["mapping"], "observation_count": 0}
                mappings[key]["observation_count"] += 1
    if not requests:
        raise ValueError("input contains no RNS prepare observations")
    entries = []
    for identity in sorted(requests):
        entry = requests[identity]
        write_json(output / entry["request_file"], entry.pop("config"))
        mappings = entry["observed_mappings"]
        entry["observed_mappings"] = [mappings[key] for key in sorted(mappings)]
        entries.append(entry)
    manifest = {"schema": MANIFEST_SCHEMA, "count_kind": COUNT_KIND, "sources": sources,
                "observation_count": sum(entry["observation_count"] for entry in entries),
                "unique_request_count": len(entries), "requests": entries}
    write_json(output / "manifest.json", manifest)
    return manifest


def _load_requests(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported request manifest schema")
    if manifest.get("count_kind") != COUNT_KIND:
        raise ValueError("manifest must identify prepare observation counts")
    if not isinstance(manifest.get("requests"), list) or not manifest["requests"]:
        raise ValueError("manifest requests must be a nonempty array")
    result = []
    seen = set()
    for entry in manifest["requests"]:
        if not isinstance(entry, dict):
            raise ValueError("manifest request must be an object")
        identity, placement = entry.get("id"), entry.get("placement")
        choice(placement, ("in_place", "separate"), "placement")
        if not isinstance(identity, str) or identity in seen:
            raise ValueError("request id must be a unique string")
        expected = f"requests/{identity}.json"
        if entry.get("request_file") != expected:
            raise ValueError("request_file must match request id")
        path = (manifest_path.parent / expected).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("request_file escapes manifest directory")
        config = loads(path.read_text(encoding="utf-8"))
        validate_config(config)
        if identity != request_id(config, placement):
            raise ValueError("request config or placement does not match its id")
        seen.add(identity)
        result.append((identity, placement, config))
    return result


def _rns_module():
    try:
        return importlib.import_module("cubutterfly.rns")
    except ModuleNotFoundError as error:
        if error.name not in ("cubutterfly", "cubutterfly.rns"):
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        return importlib.import_module("cubutterfly.rns")


def prepare(manifest, library, cache_directory, output, *, policy="prefer",
            repetitions=5, request_ids=None, limit=None, rns_module=None):
    choice(policy, ("prefer", "require", "refresh"), "policy")
    integer(repetitions, "repetitions", 32, minimum=1)
    if limit is not None:
        integer(limit, "limit", 64, minimum=1)
    available = _load_requests(manifest)
    wanted = set(request_ids) if request_ids else {entry[0] for entry in available}
    unknown = wanted - {entry[0] for entry in available}
    if unknown:
        raise ValueError(f"unknown request ids: {', '.join(sorted(unknown))}")
    selected = [entry for entry in available if entry[0] in wanted]
    if limit is not None:
        selected = selected[:limit]
    rns = rns_module if rns_module is not None else _rns_module()
    native_library = rns.Library(library)
    result = {"schema": RESULT_SCHEMA, "manifest": str(Path(manifest).resolve()),
              "library": os.fspath(library), "cache_directory": str(Path(cache_directory).resolve()),
              "policy": policy, "repetitions": repetitions,
              "timing_kind": "plan_construction_and_preparation_wall_seconds_not_ntt_performance",
              "available_request_count": len(available),
              "selected_request_ids": [entry[0] for entry in selected],
              "requested_request_ids": sorted(wanted), "limit": limit, "results": []}
    for identity, placement, config in selected:
        row = {"id": identity, "placement": placement}
        started = time.perf_counter()
        try:
            with rns.Plan(native_library, config) as plan:
                plan.prepare(placement=getattr(rns.Placement, placement.upper()),
                             cache_directory=cache_directory,
                             policy=getattr(rns.CachePolicy, policy.upper()), repetitions=repetitions)
                elapsed = time.perf_counter() - started
                info = plan.info()
                row.update(status="ok", preparation_wall_seconds=elapsed,
                           cache_hit=info["cache_hit"], cache_key=info["cache_key"],
                           source=info["source"], mapping=info["mapping"])
        except Exception as error:
            row.update(status="error", error=str(error),
                       preparation_wall_seconds=time.perf_counter() - started)
        result["results"].append(row)
        write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="extract complete requests without CUDA")
    extraction.add_argument("--input", nargs="+", required=True, type=Path)
    extraction.add_argument("--output", required=True, type=Path, help="output directory")
    preparation = commands.add_parser("prepare", help="prepare all or explicitly selected requests")
    preparation.add_argument("--manifest", required=True, type=Path)
    preparation.add_argument("--library", required=True, type=Path)
    preparation.add_argument("--cache-directory", required=True, type=Path)
    preparation.add_argument("--output", required=True, type=Path, help="result JSON file")
    preparation.add_argument("--policy", choices=("prefer", "require", "refresh"), default="prefer")
    preparation.add_argument("--repetitions", type=int, default=5)
    preparation.add_argument("--request-id", action="append", dest="request_ids")
    preparation.add_argument("--limit", type=int)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    try:
        if command == "extract":
            arguments["inputs"] = arguments.pop("input")
            result = extract(**arguments)
            print(f"{result['unique_request_count']} requests from {result['observation_count']} prepare observations")
            return 0
        result = prepare(**arguments)
        failures = sum(row["status"] != "ok" for row in result["results"])
        print(f"{len(result['results'])} requests prepared; {failures} failures")
        return int(bool(failures))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
