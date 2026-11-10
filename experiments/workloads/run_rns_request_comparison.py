#!/usr/bin/env python3
"""Compare explicitly selected exported RNS requests with the pinned GPU-NTT.

This runs an already built rns_request_compare binary. Results are real-request
operator diagnostics, not complete CKKS application measurements.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_KIND = "real_request_operator_diagnostic"
DISPLACED_CACHE_SCOPE = "artificial_l2_displacement_not_application_cache_replay"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requests = _module("rns_request_manifest", ROOT / "scripts/prepare_rns_requests.py")
evidence = _module("rns_request_evidence", ROOT / "experiments/workloads/run_heongpu_ckks_e2e.py")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--request-id", action="append", dest="request_ids", required=True,
                        help="Exact manifest ID; repeat this option to select multiple requests")
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--output-directory", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--cache-policy", choices=("prefer", "require", "refresh"), default="require")
    result.add_argument("--tune-repetitions", type=int, default=5)
    result.add_argument("--repetitions", type=int, default=20)
    result.add_argument("--rounds", type=int, default=10)
    result.add_argument("--thermal-guard", action="store_true")
    result.add_argument("--cache-regime", choices=("warm", "l2-displaced"), default="warm",
                        help="Artificial L2 displacement component control; requires --repetitions 1")
    result.add_argument("--native-dispatch", choices=("modulus-ordered", "contiguous"),
                        default="modulus-ordered",
                        help="Native component baseline; does not establish original application dispatch")
    result.add_argument("--exchange-override", choices=("block", "dependency_scoped"),
                        help="Component control: change only exchange after loading/preparing the mapping")
    result.add_argument("--gpu-uuid", help="Set CUDA_VISIBLE_DEVICES; otherwise inherit its full GPU UUID")
    result.add_argument("--heongpu-source", type=Path,
                        default=evidence.ASSETS / "sources/HEonGPU")
    result.add_argument("--cached-build", type=Path,
                        default=evidence.ASSETS / "heongpu-build-e13")
    result.add_argument("--cuntt-library", type=Path,
                        default=ROOT / "build-contract-20261017/libcuntt.so")
    return result


def selected_requests(manifest: Path, request_ids: list[str]):
    if not request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs must be explicit, nonempty and unique")
    available = {entry[0]: entry for entry in requests._load_requests(manifest)}
    unknown = set(request_ids) - available.keys()
    if unknown:
        raise ValueError(f"unknown request ids: {', '.join(sorted(unknown))}")
    selected = [available[identity] for identity in request_ids]
    return selected


def _evidence_paths(output: Path):
    return (output, output.with_suffix(".log"), output.with_suffix(".metadata.json"),
            Path(str(output) + ".selection.csv"), Path(str(output) + ".samples.csv"))


def native_dispatch_description(config, dispatch, placement="in_place"):
    if dispatch not in ("modulus-ordered", "contiguous"):
        raise ValueError("native-dispatch must be modulus-ordered or contiguous")
    count, domains = len(config["polynomials"]), len(config["domains"])
    if dispatch == "contiguous":
        if not 12 <= config["log_n"] <= 20:
            raise ValueError("contiguous native dispatch requires logN 12..20")
        n = 1 << config["log_n"]
        if not 0 < domains <= count <= min(65535, ((1 << 31) - 1) // n):
            raise ValueError("contiguous native dispatch has unsupported polynomial/domain counts")
        rows = set()
        for polynomial in config["polynomials"]:
            offset = polynomial["input_offset"]
            row = offset // n
            if (offset != polynomial["output_offset"] or offset % n or row >= count or row in rows):
                raise ValueError("contiguous native dispatch requires dense equal input/output physical rows")
            rows.add(row)
            if polynomial["domain"] != row % domains:
                raise ValueError("contiguous native dispatch requires physical row domain = row % domain_count")
    if placement not in ("in_place", "separate"):
        raise ValueError("placement must be in_place or separate")
    api = ("GPU_NTT_Modulus_Ordered" if dispatch == "modulus-ordered" else
           "GPU_INTT" if config["inverse"] else "GPU_NTT")
    if placement == "in_place":
        api += "_Inplace"
    return (f"{api};PerPolynomial;X_N_plus;batch_size={count};"
            f"mod_count={count if dispatch == 'modulus-ordered' else domains}")


def validate_placement_evidence(output, placement):
    """Separate timings require exact output and preserved-input evidence."""
    if not output.is_file():
        if placement == "separate":
            raise ValueError("separate output is missing CSV evidence")
        return
    expected = "separate" if placement == "separate" else "in-place"
    backends = set()
    with output.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("placement", "in-place") != expected:
                raise ValueError("benchmark placement does not match manifest placement")
            if placement == "separate":
                if (row.get("exact") != "1" or row.get("input_preserved") != "1" or
                        row.get("validation_trials") != "3" or
                        row.get("repeated_input_semantics") != "same_preserved_input"):
                    raise ValueError("separate output lacks exact/preserved-input validation evidence")
                backends.add(row.get("backend"))
    if placement == "separate" and backends != {"cubutterfly", "native-gpuntt"}:
        raise ValueError("separate output needs validation evidence for both backends")


def validate_native_evidence(output, dispatch, description):
    if not output.is_file():
        return
    native_rows = 0
    with output.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("native_dispatch", "modulus-ordered") != dispatch:
                raise ValueError("benchmark native dispatch does not match requested dispatch")
            if dispatch == "contiguous" and row.get("native_dispatch_description") != description:
                raise ValueError("benchmark native dispatch description does not match request")
            if row.get("backend") == "native-gpuntt" and dispatch == "contiguous":
                native_rows += 1
                if not row.get("mapping", "").startswith(description + ";grid="):
                    raise ValueError("native mapping does not describe requested contiguous API")
    if dispatch == "contiguous" and not native_rows:
        raise ValueError("contiguous output has no native dispatch evidence")


def mapping_evidence(output: Path, exchange_override: str | None):
    if not output.is_file():
        return None, None
    actual = prepared = None
    with output.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("backend") != "cubutterfly":
                continue
            if exchange_override is not None and not row.get("prepared_mapping"):
                raise ValueError("component control output is missing original prepared mapping")
            current = requests.loads(row["mapping"])
            original = requests.loads(row.get("prepared_mapping") or row["mapping"])
            requests.validate_mapping(current)
            requests.validate_mapping(original)
            if exchange_override is not None:
                if row.get("selection_kind") != "component_control" or row.get("source") != "component_control":
                    raise ValueError("exchange override result must be labeled component_control")
                if current.get("exchange", "block") != exchange_override:
                    raise ValueError("actual mapping does not match exchange override")
                if ({k: v for k, v in current.items() if k != "exchange"} !=
                        {k: v for k, v in original.items() if k != "exchange"}):
                    raise ValueError("component control changed mapping fields other than exchange")
            if actual is not None and (actual != current or prepared != original):
                raise ValueError("benchmark rows disagree about actual or prepared mapping")
            actual, prepared = current, original
    if exchange_override is not None and actual is None:
        raise ValueError("component control output has no actual/prepared mapping evidence")
    return actual, prepared


def cache_evidence(output: Path, regime: str):
    if not output.is_file():
        return None
    observed = None
    backends = set()
    with output.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("cache_regime", "warm") != regime:
                raise ValueError("benchmark cache regime does not match requested cache-regime")
            if regime == "warm":
                if row.get("cache_displacement_bytes", "0") != "0":
                    raise ValueError("warm cache regime must not displace L2")
                if row.get("l2_cache_bytes"):
                    try:
                        observed = {"l2_cache_bytes": max(0, int(row["l2_cache_bytes"])),
                                    "cache_displacement_bytes": 0}
                    except ValueError as error:
                        raise ValueError("warm output has invalid L2 byte evidence") from error
                continue
            try:
                l2_bytes = int(row["l2_cache_bytes"])
                displaced_bytes = int(row["cache_displacement_bytes"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("L2 displacement output is missing valid byte evidence") from error
            if not (0 < l2_bytes <= (1 << 29) and
                    2 * l2_bytes <= displaced_bytes <= (1 << 30)):
                raise ValueError("L2 displacement must cover at least twice the supported device L2")
            if (row.get("cache_scope") != DISPLACED_CACHE_SCOPE or row.get("repetitions") != "1"
                    or row.get("measurement_kind") != "component_control"):
                raise ValueError("L2 displacement output must describe a single-transform artificial component_control")
            if row.get("backend") == "cubutterfly" and (
                    row.get("selection_kind") != "component_control" or
                    row.get("source") != "component_control" or not row.get("prepared_mapping")):
                raise ValueError("L2 displacement output must retain prepared mapping and component_control selection")
            current = {"l2_cache_bytes": l2_bytes, "cache_displacement_bytes": displaced_bytes}
            if observed is not None and current != observed:
                raise ValueError("benchmark rows disagree about L2 displacement byte evidence")
            observed = current
            backends.add(row.get("backend"))
    if regime == "l2-displaced" and backends != {"cubutterfly", "native-gpuntt"}:
        raise ValueError("L2 displacement output needs evidence for both backends")
    return observed or {"l2_cache_bytes": None, "cache_displacement_bytes": 0}


def execute(args: argparse.Namespace) -> int:
    if args.exchange_override not in (None, "block", "dependency_scoped"):
        raise ValueError("exchange-override must be block or dependency_scoped")
    if not 1 <= args.repetitions <= 1000 or not 1 <= args.tune_repetitions <= 1000 or args.rounds < 1:
        raise ValueError("repetitions and tune-repetitions must be 1..1000; rounds must be positive")
    if args.cache_regime not in ("warm", "l2-displaced"):
        raise ValueError("cache-regime must be warm or l2-displaced")
    if args.cache_regime == "l2-displaced" and args.repetitions != 1:
        raise ValueError("l2-displaced requires --repetitions 1")
    manifest = args.manifest.resolve()
    selected = selected_requests(manifest, args.request_ids)
    descriptions = {identity: native_dispatch_description(config, args.native_dispatch, placement)
                    for identity, placement, config in selected}
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"benchmark binary is missing: {binary}")
    output_directory = args.output_directory.resolve()
    suffix = f".exchange-{args.exchange_override}" if args.exchange_override is not None else ""
    if args.cache_regime != "warm":
        suffix += f".cache-{args.cache_regime}"
    outputs = [output_directory / f"{identity}{suffix}.csv" for identity, _, _ in selected]
    # Check the whole explicit selection before launching its first request.
    for output in outputs:
        if any(path.exists() for path in _evidence_paths(output)):
            raise FileExistsError(f"refusing to overwrite benchmark output or evidence: {output}")
    env = os.environ.copy()
    env.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
    if args.gpu_uuid is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    cuntt = args.cuntt_library.resolve()
    cached = args.cached_build.resolve()
    source = args.heongpu_source.resolve()
    env["LD_LIBRARY_PATH"] = str(cuntt.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUBUTTERFLY_RNS_CACHE"] = str(args.cache_dir.resolve())
    env["CUBUTTERFLY_RNS_CACHE_POLICY"] = args.cache_policy
    output_directory.mkdir(parents=True, exist_ok=True)
    for (identity, placement, _), output in zip(selected, outputs):
        # _load_requests verifies request_file equals this standard relative
        # path, stays within the manifest directory and matches the config ID.
        config_path = manifest.parent / "requests" / f"{identity}.json"
        identities = {name: evidence.file_identity(path) for name, path in (
            ("manifest", manifest), ("config", config_path),
            ("benchmark_source", ROOT / "benchmarks/rns_request_compare.cu"),
            ("experiment_runner", Path(__file__).resolve()),
            ("manifest_extractor", ROOT / "scripts/prepare_rns_requests.py"))}
        command = [str(binary), "--request", str(config_path),
                   "--cache-dir", str(args.cache_dir.resolve()), "--cache-policy", args.cache_policy,
                   "--tune-repetitions", str(args.tune_repetitions),
                   "--repetitions", str(args.repetitions), "--rounds", str(args.rounds),
                   "--output", str(output)]
        if placement == "separate":
            command += ["--placement", "separate"]
        if args.thermal_guard:
            command.append("--thermal-guard")
        if args.exchange_override is not None:
            command += ["--exchange-override", args.exchange_override]
        # Omit the legacy default option so older benchmark binaries still work.
        if args.native_dispatch != "modulus-ordered":
            command += ["--native-dispatch", args.native_dispatch]
        if args.cache_regime != "warm":
            command += ["--cache-regime", args.cache_regime]
        metadata_path = output.with_suffix(".metadata.json")
        evidence_collision = False
        result = None
        try:
            # The shared runner records HEonGPU archives as provenance. They are
            # not asserted to be linked into this standalone operator diagnostic.
            archive = cached / "src/libheongpu.a"
            result = evidence.run_with_evidence(command, env, output, source, cuntt,
                                                archive, archive, True)
        except FileExistsError:
            evidence_collision = True
            raise
        finally:
            if not evidence_collision and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if result == 0:
                    try:
                        validate_placement_evidence(output, placement)
                    except ValueError as error:
                        metadata.update(status="failed_evidence_validation", performance_eligible=False,
                                        evidence_error=str(error), request_id=identity,
                                        request_placement=placement)
                        requests.write_json(metadata_path, metadata)
                        raise
                actual_mapping, prepared_mapping = mapping_evidence(output, args.exchange_override)
                validate_native_evidence(output, args.native_dispatch, descriptions[identity])
                cache_bytes = cache_evidence(output, args.cache_regime)
                if result == 0 and args.cache_regime == "l2-displaced" and cache_bytes is None:
                    raise ValueError("successful L2 displacement run is missing CSV evidence")
                if ((args.native_dispatch == "contiguous" or args.cache_regime == "l2-displaced")
                        and args.exchange_override is None
                        and actual_mapping != prepared_mapping):
                    raise ValueError("component control changed the prepared cuButterfly mapping")
                metadata.update(measurement_kind=("component_control" if args.exchange_override is not None or args.native_dispatch == "contiguous" or args.cache_regime == "l2-displaced" else MEASUREMENT_KIND),
                                selection_kind=("component_control" if args.exchange_override is not None or args.cache_regime == "l2-displaced" else "automatic_library"),
                                exchange_override=args.exchange_override,
                                allowed_mapping_changes=(["exchange"] if args.exchange_override is not None else []),
                                actual_mapping=actual_mapping, prepared_mapping=prepared_mapping,
                                native_dispatch=args.native_dispatch,
                                native_dispatch_description=descriptions[identity],
                                native_dispatch_scope="semantic_equivalent_not_original_application_dispatch",
                                cache_regime=args.cache_regime,
                                cache_scope=(DISPLACED_CACHE_SCOPE if args.cache_regime == "l2-displaced"
                                             else "warm_repeated_graph"),
                                cache_displacement_bytes=(cache_bytes["cache_displacement_bytes"] if cache_bytes else None),
                                device_l2_cache_bytes=(cache_bytes["l2_cache_bytes"] if cache_bytes else None),
                                headline_eligible=False,
                                performance_scope="operator_only_not_application_end_to_end",
                                request_id=identity, request_placement=placement,
                                requested_request_ids=args.request_ids,
                                cache_policy=args.cache_policy,
                                tune_repetitions=args.tune_repetitions,
                                request_provenance=identities,
                                archive_provenance_note="Shared-runner HEonGPU archive hashes are provenance, not diagnostic linkage claims.")
                # Keep the shared runner's completion/occupancy/error status.
                requests.write_json(metadata_path, metadata)
        if result:
            return result
    return 0


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
