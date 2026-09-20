#!/usr/bin/env python3
"""Portable execution and resume driver for composed cryptographic NTTs.

This runner is intentionally independent of any frozen research result
directory.  It measures the public ``crypto_ntt_bench`` composition binary,
keeps GPU composition results as raw evidence, and only replays a mapping
from an acceptance journal when the journal is for the same GPU UUID, build,
and timing protocol.  It does not search or report an external-library win.

The caller owns the outer scheduling lock.  The runner still checks the
requested UUID before every launch and once per second while a child process
is running.  If another process appears, only the runner's child is
terminated and the attempt is recorded as an interruption rather than being
silently treated as a missing cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hardware_registry import canonical_semantics, stable_id

SCHEMA = "cubutterfly-portable-crypto-composition-v1"
MANIFEST_SCHEMA = "cubutterfly-portable-crypto-composition-manifest-v1"
RESULT_SCHEMA = "cubutterfly-portable-crypto-composition-results-v1"
SUMMARY_SCHEMA = "cubutterfly-portable-crypto-composition-summary-v1"
DEFAULT_COMPILE_MODE = "research"
DEFAULT_TRIALS = 3
DEFAULT_WARMUP = 100
DEFAULT_REPEAT = 100
DEFAULT_SEED = 20260920
DEFAULT_PATTERNS = ("random", "boundary")
MONITOR_SECONDS = 1.0
EXIT_OK = 0
EXIT_ERROR = 2
EXIT_RETRY = 75


class CompositionError(RuntimeError):
    """A measured attempt was invalid or could not be completed."""


class ExclusiveViolation(CompositionError):
    """Another process used the requested GPU during a measurement."""


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _walk_files(root: Path, suffixes: set[str]) -> list[Path]:
    """Walk one known runtime tree while pruning unrelated large trees."""
    root = Path(root)
    if not root.is_dir():
        return []
    ignored = {".git", ".pytest_cache", "__pycache__", "build", "results", "external"}
    files: list[Path] = []
    # os.walk lets us prune before descending; rglob would still traverse
    # the multi-gigabyte result and third-party trees before filtering them.
    for directory, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in ignored)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in suffixes:
                files.append(path)
    return files


def _template_files(root: Path) -> list[Path]:
    """Return only inputs consumed by runtime/JIT composition lowerings.

    Documentation, campaign metadata, generated build trees and third-party
    source are deliberately outside this identity.  MathDx is the one
    external dependency hashed because ``compile_module.py`` feeds its headers
    directly to NVCC for cufftdx-based modules.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    files: list[Path] = []
    files.extend(_walk_files(root / "include", {".h", ".hh", ".hpp"}))
    files.extend(_walk_files(root / "src", {".h", ".hh", ".hpp", ".cuh"}))
    for relative in ("scripts/compile_module.py", "scripts/research_compile_requests.py",
                     "scripts/resident_mapping.py"):
        path = root / relative
        if path.is_file():
            files.append(path)
    mathdx = root / "external/mathdx/nvidia/mathdx/include"
    files.extend(_walk_files(mathdx, {".hpp"}))
    return sorted(set(files), key=lambda p: _safe_rel(p, root))


def template_fingerprint(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    records = [{"path": _safe_rel(path, root), "sha256": sha256(path)} for path in _template_files(root)]
    return {"file_count": len(records), "files": records, "sha256": json_sha256(records)}


def _ldd_paths(binary: Path) -> list[Path]:
    try:
        completed = subprocess.run(["ldd", str(binary)], text=True, capture_output=True, check=False)
    except OSError:
        return []
    paths: list[Path] = []
    for line in completed.stdout.splitlines():
        match = re.search(r"=>\s+(/[^\s(]+)", line)
        candidate = match.group(1) if match else (line.strip().split(" ", 1)[0] if line.startswith("/") else "")
        if candidate:
            path = Path(candidate)
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def _build_files(build_dir: Path) -> list[Path]:
    build_dir = Path(build_dir)
    if not build_dir.is_dir():
        raise ValueError(f"build directory does not exist: {build_dir}")
    files: list[Path] = []
    root_configs = {"CMakeCache.txt", "Makefile", "build.ninja", "cmake_install.cmake"}
    library_suffixes = (".so", ".so.", ".a", ".dylib", ".dll")
    ignored = {"CMakeFiles", ".cache", "__pycache__", ".git", "results", "external"}
    for directory, directories, names in os.walk(build_dir):
        directories[:] = sorted(name for name in directories if name not in ignored)
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(build_dir)
            if len(relative.parts) == 1 and name in root_configs:
                files.append(path)
                continue
            # Only executable/library outputs are identity inputs.  Logs,
            # manifests, compiler dependency files and generated metadata are
            # intentionally excluded from resume identity.
            if (name.startswith(("crypto_", "cuntt_", "cubutterfly_"))
                    and os.access(path, os.X_OK)) or name.endswith(library_suffixes):
                files.append(path)
    return sorted(set(files), key=lambda p: _safe_rel(p, build_dir))


def build_fingerprint(build_dir: Path) -> dict[str, Any]:
    build_dir = Path(build_dir).resolve()
    records = [{"path": _safe_rel(path, build_dir), "sha256": sha256(path)} for path in _build_files(build_dir)]
    return {"root": build_dir.name, "files": records, "sha256": json_sha256(records)}


def binary_fingerprint(binary: Path, build_dir: Path) -> dict[str, Any]:
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise ValueError(f"composition binary does not exist: {binary}")
    libraries = []
    for path in _ldd_paths(binary):
        libraries.append({"name": path.name, "sha256": sha256(path)})
    return {
        "name": binary.name,
        "sha256": sha256(binary),
        "build": build_fingerprint(build_dir),
        "dynamic_libraries": sorted(libraries, key=lambda row: (row["name"], row["sha256"])),
    }


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "trials": int(args.trials),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "patterns": list(args.patterns),
        "seed": int(args.seed),
    }


def _mapping_source_fingerprint(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [{"name": Path(path).name, "sha256": sha256(Path(path))} for path in paths]


def _workloads_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = read_json(path)
    rows = document.get("workloads") if isinstance(document, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("workloads JSON must contain a non-empty workloads list")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    semantics: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every composition workload must be an object")
        workload = dict(row)
        identifier = str(workload.get("id") or workload.get("contract_id") or "")
        if not identifier:
            identifier = composition_key(workload)
            workload["id"] = identifier
        if identifier in ids:
            raise ValueError(f"duplicate composition workload id: {identifier}")
        semantic = composition_key(workload)
        if semantic in semantics:
            raise ValueError(f"duplicate composition semantic contract: {identifier}")
        ids.add(identifier)
        semantics.add(semantic)
        validate_workload(workload)
        result.append(workload)
    return document, result


def composition_semantics(workload: Mapping[str, Any]) -> dict[str, Any]:
    # Application labels and suite membership are provenance, not execution
    # semantics.  Normalizing defaults here prevents an RNS L=1 row from being
    # counted again merely because one generator wrote explicit defaults.
    result = {
        "logN": int(workload["logN"]), "batch": int(workload["batch"]),
        "word_bits": int(workload["word_bits"]), "mode": str(workload["mode"]),
        "direction": str(workload.get("direction", "forward")),
        "moduli": [str(value) for value in workload["moduli"]],
        "normalization": str(workload.get("normalization", "inverse"
                                              if workload.get("direction", "forward") == "inverse" else "none")),
        "layout": str(workload.get("layout", "channel-batch-coefficient")),
        "rns_stream_order": str(workload.get("rns_stream_order", "single-stream")),
    }
    if result["mode"] == "coset":
        result["coset_generator"] = int(workload.get("coset_generator", 7))
    return result


def composition_key(workload: Mapping[str, Any]) -> str:
    return stable_id(composition_semantics(workload))[:20]


def validate_workload(workload: Mapping[str, Any]) -> None:
    required = ("logN", "batch", "word_bits", "mode", "moduli")
    missing = [field for field in required if field not in workload]
    if missing:
        raise ValueError("composition workload missing: " + ", ".join(missing))
    if int(workload["logN"]) < 1 or int(workload["logN"]) > 30:
        raise ValueError("composition logN is outside the portable safety range")
    if int(workload["batch"]) < 1:
        raise ValueError("composition batch must be positive")
    if int(workload["word_bits"]) not in (32, 64):
        raise ValueError("composition word_bits must be 32 or 64")
    if workload["mode"] not in {"cyclic", "negacyclic", "coset"}:
        raise ValueError("unsupported composition mode")
    moduli = workload["moduli"]
    if not isinstance(moduli, list) or not moduli or any(int(value) <= 2 for value in moduli):
        raise ValueError("composition moduli must be a non-empty list of integers")
    if workload.get("direction", "forward") not in {"forward", "inverse"}:
        raise ValueError("unsupported composition direction")
    expected_normalization = "inverse" if workload.get("direction", "forward") == "inverse" else "none"
    if workload.get("normalization", expected_normalization) != expected_normalization:
        raise ValueError("composition normalization does not match direction")
    if workload.get("layout", "channel-batch-coefficient") != "channel-batch-coefficient":
        raise ValueError("unsupported composition layout")
    if workload.get("rns_stream_order", "single-stream") != "single-stream":
        raise ValueError("unsupported composition RNS schedule")
    if workload["mode"] == "coset" and int(workload.get("coset_generator", 7)) <= 0:
        raise ValueError("coset generator must be positive")


def _cyclic_workload(workload: Mapping[str, Any], modulus: Any) -> dict[str, Any]:
    direction = str(workload.get("direction", "forward"))
    return {
        "operator": "ntt",
        "precision": f"word{int(workload['word_bits'])}",
        "logN": int(workload["logN"]),
        "batch": int(workload["batch"]),
        "placement": "out-of-place",
        "direction": direction,
        "normalization": "inverse" if direction == "inverse" else "none",
        "modulus": int(modulus),
        "input_order": "natural",
        "output_order": "natural",
    }


def cyclic_cell_key(workload: Mapping[str, Any], modulus: Any) -> str:
    return stable_id(canonical_semantics(_cyclic_workload(workload, modulus)))[:20]


def _selected_sample(cell: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selected = cell.get("selected")
    if isinstance(selected, Mapping):
        samples = selected.get("samples")
        if isinstance(samples, list) and samples and isinstance(samples[0], Mapping):
            return samples[0]
        sample = selected.get("sample")
        if isinstance(sample, Mapping):
            return sample
        if "mapping_json" in selected:
            return selected
    return None


def _mapping_value(sample: Mapping[str, Any]) -> Any:
    raw = sample.get("mapping_json")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("mapping source contains invalid mapping_json") from error
    return raw


def _source_uuid(document: Mapping[str, Any]) -> str | None:
    identity = document.get("identity")
    if not isinstance(identity, Mapping):
        return None
    device = identity.get("device")
    if isinstance(device, Mapping):
        value = device.get("uuid") or device.get("gpu_uuid")
        return str(value) if value is not None else None
    value = identity.get("gpu_uuid") or identity.get("target_uuid") or identity.get("targetUUID")
    return str(value) if value is not None else None


def _source_protocol(document: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = document.get("identity")
    if isinstance(identity, Mapping) and isinstance(identity.get("protocol"), Mapping):
        return identity["protocol"]
    protocol = document.get("protocol")
    return protocol if isinstance(protocol, Mapping) else {}


def _source_build_matches(document: Mapping[str, Any], fingerprint: Mapping[str, Any], build_dir: Path) -> bool:
    identity = document.get("identity")
    if not isinstance(identity, Mapping):
        return False
    declared_build = identity.get("build_dir_sha256") or document.get("build_dir_sha256")
    if declared_build is not None:
        return str(declared_build) == str(fingerprint["build"]["sha256"])
    declared_binaries = identity.get("binaries")
    if not isinstance(declared_binaries, Mapping):
        return False
    compared = 0
    for name, expected in declared_binaries.items():
        candidate = Path(build_dir) / str(name)
        if not candidate.is_file():
            return False
        compared += 1
        if sha256(candidate) != str(expected):
            return False
    return compared > 0


def mapping_portfolios(workload: Mapping[str, Any], gpu_uuid: str, protocol: Mapping[str, Any],
                       build_dir: Path, sources: Iterable[Path],
                       compile_mode: str = DEFAULT_COMPILE_MODE) -> list[dict[str, Any]]:
    sources = [Path(source) for source in sources]
    local_build = build_fingerprint(build_dir)
    # Reconstruct channel order from provenance.  A source may cover only a
    # subset, so search every source independently rather than consuming the
    # first source's channel list.
    mappings: list[Any] = []
    channel_provenance: list[dict[str, Any]] = []
    for modulus in workload["moduli"]:
        mapping = None
        evidence = None
        key = cyclic_cell_key(workload, modulus)
        for source_path in sources:
            try:
                document = read_json(source_path)
                source_hash = sha256(source_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if _source_uuid(document) != gpu_uuid:
                continue
            identity = document.get("identity")
            if isinstance(identity, Mapping) and identity.get("compile_mode") not in (None, compile_mode):
                continue
            source_protocol = _source_protocol(document)
            if any(source_protocol.get(field) != protocol.get(field) for field in ("trials", "warmup", "repeat")):
                continue
            if not _source_build_matches(document, {"build": local_build}, build_dir):
                continue
            cell = next((row for row in document.get("cells", [])
                         if isinstance(row, Mapping) and str(row.get("id")) == key
                         and row.get("status") == "complete"), None)
            if cell is None or canonical_semantics(cell.get("workload", {})) != canonical_semantics(_cyclic_workload(workload, modulus)):
                continue
            selected = _selected_sample(cell)
            if selected is None or str(selected.get("correct")) not in {"1", "true", "True"}:
                continue
            if selected.get("gpu_uuid") is not None and str(selected.get("gpu_uuid")) != gpu_uuid:
                continue
            mapping = _mapping_value(selected)
            if mapping is not None:
                evidence = {"kind": "exact-contract-cyclic-search", "cell_id": key,
                            "source": source_path.name, "source_sha256": source_hash}
                break
        mappings.append(mapping)
        channel_provenance.append(evidence or {"kind": "public-default", "reason": "no compatible exact-contract mapping",
                                                "cell_id": key})
    result = [dict(name="public-default", mappings=None,
                   provenance="public shared-iterative proposal; not claimed optimal")]
    if any(mapping is not None for mapping in mappings):
        result.append(dict(name="searched-cyclic-cores", mappings=mappings,
                           provenance=channel_provenance,
                           scope="exact cyclic mappings replayed inside composed semantics; no joint RNS search"))
    return result


def nvidia_clients(gpu_uuid: str) -> list[dict[str, Any]]:
    command = ["nvidia-smi", "-i", gpu_uuid,
               "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise CompositionError("nvidia-smi GPU query failed: " + (completed.stderr.strip() or "unknown error"))
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        try:
            pid = int(fields[0])
        except (ValueError, IndexError):
            continue
        rows.append({"pid": pid, "process_name": fields[1] if len(fields) > 1 else "",
                     "used_gpu_memory": fields[2] if len(fields) > 2 else ""})
    return rows


def foreign_clients(gpu_uuid: str, own_pid: int | None = None) -> list[dict[str, Any]]:
    return [row for row in nvidia_clients(gpu_uuid) if own_pid is None or row["pid"] != own_pid]


def _command(binary: Path, workload: Mapping[str, Any], pattern: str, trial: int,
             protocol: Mapping[str, Any], mapping: Any) -> list[str]:
    command = [str(binary), "--log-n", str(workload["logN"]), "--batch", str(workload["batch"]),
               "--word-bits", str(workload["word_bits"]), "--moduli", ",".join(map(str, workload["moduli"])),
               "--mode", str(workload["mode"]), "--direction", str(workload.get("direction", "forward")),
               "--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"]),
               "--seed", str(int(protocol["seed"]) + trial), "--input-pattern", pattern, "--json"]
    if workload["mode"] == "coset":
        command += ["--coset-generator", str(workload.get("coset_generator", 7))]
    if mapping is not None:
        command += ["--mapping-json", json.dumps(mapping, separators=(",", ":"))]
    return command


def _same_json(left: Any, right: Any) -> bool:
    return left == right


def validate_sample(sample: Mapping[str, Any], workload: Mapping[str, Any], gpu_uuid: str,
                    pattern: str, trial: int, protocol: Mapping[str, Any], mapping: Any,
                    compile_mode: str = DEFAULT_COMPILE_MODE) -> None:
    expected = {
        "logN": int(workload["logN"]), "batch": int(workload["batch"]),
        "word_bits": int(workload["word_bits"]), "mode": workload["mode"],
        "direction": workload.get("direction", "forward"), "gpu_uuid": gpu_uuid,
        "compile_mode": compile_mode, "correct": True,
        "verified_batches": int(workload["batch"]), "verified_channels": len(workload["moduli"]),
        "warmup": int(protocol["warmup"]), "repeat": int(protocol["repeat"]),
        "input_pattern": pattern, "seed": int(protocol["seed"]) + trial,
        "normalization": "inverse" if workload.get("direction", "forward") == "inverse" else "none",
        "layout": "channel-batch-coefficient", "rns_stream_order": "single-stream",
    }
    for field, value in expected.items():
        if field == "correct":
            actual = sample.get(field) is True or str(sample.get(field)).lower() in {"1", "true"}
        else:
            actual = sample.get(field)
            if field in {"logN", "batch", "word_bits", "verified_batches", "verified_channels", "warmup", "repeat", "seed"}:
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    actual = None
        if actual != value:
            raise CompositionError(f"composition sample {field} differs: {actual!r} != {value!r}")
    if [str(value) for value in sample.get("moduli", [])] != [str(value) for value in workload["moduli"]]:
        raise CompositionError("composition sample modulus channels differ")
    if workload["mode"] == "coset" and int(sample.get("coset_generator", 0)) != int(workload.get("coset_generator", 7)):
        raise CompositionError("composition sample coset differs")
    if not _same_json(sample.get("mapping_request"), mapping):
        raise CompositionError("composition sample mapping request differs")
    try:
        kernel_ms = float(sample.get("kernel_ms"))
    except (TypeError, ValueError):
        kernel_ms = 0.0
    if not math.isfinite(kernel_ms) or not kernel_ms > 0:
        raise CompositionError("composition sample timing is invalid")


def _parse_output(stdout: str) -> Mapping[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise CompositionError("composition benchmark did not emit one JSON result") from error
    if not isinstance(value, Mapping):
        raise CompositionError("composition benchmark result is not an object")
    return value


def measure(command: list[str], env: Mapping[str, str], gpu_uuid: str, log_path: Path) -> Mapping[str, Any]:
    foreign = foreign_clients(gpu_uuid)
    if foreign:
        raise ExclusiveViolation("target GPU is not exclusive before launch: " + repr(foreign))
    process = subprocess.Popen(command, cwd=ROOT, env=dict(env), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ""
    stderr = ""
    completed = False
    failure: BaseException | None = None
    try:
        while process.poll() is None:
            try:
                stdout, stderr = process.communicate(timeout=MONITOR_SECONDS)
                completed = True
                break
            except subprocess.TimeoutExpired as error:
                stdout = error.output or stdout or ""
                stderr = error.stderr or stderr or ""
                foreign = foreign_clients(gpu_uuid, process.pid)
                if foreign:
                    failure = ExclusiveViolation("target GPU became non-exclusive: " + repr(foreign))
                    process.terminate()
                    break
        if not completed:
            stdout, stderr = process.communicate()
        if failure is None:
            foreign = foreign_clients(gpu_uuid, process.pid)
            if foreign:
                failure = ExclusiveViolation("target GPU was not exclusive after measurement: " + repr(foreign))
        if failure is None and process.returncode:
            failure = CompositionError(f"composition benchmark failed ({process.returncode})")
        record = {"command": command, "returncode": process.returncode,
                  "stdout": stdout, "stderr": stderr}
        if failure is not None:
            record["status"] = "interrupted" if isinstance(failure, ExclusiveViolation) else "failed"
            record["error"] = str(failure)
            write_json(log_path, record)
            raise failure
        write_json(log_path, record)
        return _parse_output(stdout)
    except BaseException as error:
        if process.poll() is None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        if not log_path.exists():
            write_json(log_path, {"command": command, "returncode": process.returncode,
                                  "status": "failed", "error": str(error),
                                  "stdout": stdout, "stderr": stderr})
        raise


def preflight_workloads(workloads: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_word: dict[int, list[str]] = {}
    for workload in workloads:
        word_bits = int(workload["word_bits"])
        by_word.setdefault(word_bits, [])
        for modulus in workload["moduli"]:
            value = str(modulus)
            if value not in by_word[word_bits]:
                by_word[word_bits].append(value)
    result = []
    for word_bits in sorted(by_word):
        moduli = by_word[word_bits][:2]
        for mode in ("cyclic", "negacyclic", "coset"):
            for direction in ("forward", "inverse"):
                row = {"id": f"preflight-{word_bits}-{mode}-{direction}", "logN": 8,
                       "batch": 3, "word_bits": word_bits, "mode": mode,
                       "direction": direction, "moduli": moduli}
                if mode == "coset":
                    row["coset_generator"] = 7
                result.append(row)
    return result


def _manifest(args: argparse.Namespace, workload_doc: Mapping[str, Any], workloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    sources = [Path(path).resolve() for path in args.mapping_sources]
    binary = Path(args.binary).resolve()
    build_dir = Path(args.build_dir).resolve()
    protocol = _protocol(args)
    return {
        "schema": MANIFEST_SCHEMA,
        "version": SCHEMA,
        "gpu_uuid": args.gpu_uuid,
        "compile_mode": args.compile_mode,
        "protocol": protocol,
        "workloads": {"name": Path(args.workloads).name, "sha256": sha256(Path(args.workloads)),
                       "semantic_sha256": json_sha256([composition_semantics(w) for w in workloads]),
                       "count": len(workloads)},
        "binary": binary_fingerprint(binary, build_dir),
        "template_root": Path(args.template_root).name,
        "templates": template_fingerprint(Path(args.template_root)),
        "mapping_sources": _mapping_source_fingerprint(sources),
        "scope": "composed modular single-word plans; sequential RNS channels; no joint search or external baseline",
        "workload_document_schema": workload_doc.get("schema"),
    }


def _validate_manifest(current: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    if current != frozen:
        raise ValueError("portable composition identity/protocol changed; start a new output directory")


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"gpu_uuid": manifest["gpu_uuid"], "manifest_sha256": json_sha256(manifest),
            "compile_mode": manifest["compile_mode"]}


def _new_results(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": RESULT_SCHEMA, "identity": _identity(manifest), "status": "partial",
            "cells": [], "failures": [], "external_baseline": "unavailable; composition characterization only",
            "host_metadata": {"hostname": socket.gethostname(), "platform": platform.platform(),
                              "python": platform.python_version()}}


def _result_path(output_dir: Path) -> Path:
    return Path(output_dir) / "results.json"


def _record_failure(document: dict[str, Any], workload: Mapping[str, Any], portfolio: Mapping[str, Any],
                    pattern: str, trial: int, error: BaseException) -> None:
    document.setdefault("failures", []).append({
        "workload_id": workload["id"], "portfolio": portfolio["name"], "pattern": pattern,
        "trial": trial, "status": "interrupted" if isinstance(error, ExclusiveViolation) else "failed",
        "error": str(error),
    })


def _summary(document: Mapping[str, Any], workloads: list[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    workload_map = {str(w["id"]): w for w in workloads}
    expected_per_cell = int(protocol["trials"]) * len(protocol["patterns"])
    complete = 0
    for cell in document.get("cells", []):
        workload = workload_map.get(str(cell.get("id")))
        if workload is None or cell.get("status") != "complete":
            continue
        complete += 1
        for portfolio in cell.get("portfolios", []):
            name = portfolio["name"]
            for pattern in protocol["patterns"]:
                samples = [row["sample"] for row in cell.get("measurements", [])
                           if row.get("portfolio") == name and row.get("pattern") == pattern]
                if len(samples) != int(protocol["trials"]):
                    continue
                ms = sorted(float(sample["kernel_ms"]) for sample in samples)[len(samples) // 2]
                batch = int(workload["batch"])
                channels = len(workload["moduli"])
                rows.append({"id": workload["id"], "pattern": pattern, "portfolio": name,
                             "kernel_ms": ms, "rns_batch_s": batch * 1000.0 / ms,
                             "residue_transforms_s": batch * channels * 1000.0 / ms,
                             "modulus_bits": [int(value).bit_length() for value in workload["moduli"]],
                             "workload": workload})
    total = len(workloads)
    return {"schema": SUMMARY_SCHEMA, "status": "complete" if complete == total else "partial",
            "declared_cells": total, "complete_cells": complete, "failed_attempts": len(document.get("failures", [])),
            "protocol": protocol, "rows": rows,
            "claims": ["No external-library comparison or joint RNS optimum is claimed.",
                       "rns_batch_s counts complete composed residue batches; residue_transforms_s counts channels too."]}


def run(args: argparse.Namespace) -> int:
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload_doc, workloads = _workloads_document(Path(args.workloads))
    current_manifest = _manifest(args, workload_doc, workloads)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("output directory already has a manifest; pass --resume or choose a new directory")
        frozen_manifest = read_json(manifest_path)
        _validate_manifest(current_manifest, frozen_manifest)
        manifest = frozen_manifest
    else:
        if args.resume:
            raise ValueError("--resume requires an existing manifest")
        manifest = current_manifest
        write_json(manifest_path, manifest)
    protocol = manifest["protocol"]
    result_path = _result_path(args.output_dir)
    document = read_json(result_path) if result_path.exists() else _new_results(manifest)
    if document.get("identity") != _identity(manifest):
        raise ValueError("composition result identity changed")
    if args.preflight:
        preflight_path = args.output_dir / "preflight.json"
        preflight_rows = preflight_workloads(workloads)
        preflight_doc = read_json(preflight_path) if preflight_path.exists() else {
            "schema": "cubutterfly-portable-crypto-preflight-v1", "identity": _identity(manifest),
            "status": "partial", "samples": [], "workloads": preflight_rows,
        }
        if preflight_doc.get("identity") != _identity(manifest):
            raise ValueError("preflight identity changed")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
               "CUBUTTERFLY_COMPILE_MODE": args.compile_mode,
               "CUBUTTERFLY_TEMPLATE_ROOT": str(Path(args.template_root).resolve())}
        for workload in preflight_rows:
            for pattern in protocol["patterns"]:
                prior = [row for row in preflight_doc["samples"] if row["id"] == workload["id"] and row["pattern"] == pattern]
                if prior:
                    validate_sample(prior[0]["sample"], workload, args.gpu_uuid, pattern, 0, protocol, None,
                                    manifest["compile_mode"])
                    continue
                command = _command(Path(args.binary), workload, pattern, 0, protocol, None)
                try:
                    sample = measure(command, env, args.gpu_uuid,
                                     args.output_dir / "preflight_logs" / f"{workload['id']}-{pattern}.json")
                    validate_sample(sample, workload, args.gpu_uuid, pattern, 0, protocol, None,
                                    manifest["compile_mode"])
                except BaseException as error:
                    preflight_doc.setdefault("failures", []).append({"id": workload["id"], "pattern": pattern,
                                                                       "error": str(error)})
                    write_json(preflight_path, preflight_doc)
                    raise
                preflight_doc["samples"].append({"id": workload["id"], "pattern": pattern, "sample": sample})
                write_json(preflight_path, preflight_doc)
        preflight_doc["status"] = "complete"
        write_json(preflight_path, preflight_doc)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
           "CUBUTTERFLY_COMPILE_MODE": args.compile_mode,
           "CUBUTTERFLY_TEMPLATE_ROOT": str(Path(args.template_root).resolve())}
    for workload in workloads:
        cell = next((row for row in document["cells"] if row.get("id") == workload["id"]), None)
        expected_portfolios = mapping_portfolios(workload, args.gpu_uuid, protocol, Path(args.build_dir),
                                                 args.mapping_sources, manifest["compile_mode"])
        if cell is None:
            cell = {"id": workload["id"], "workload": workload, "status": "pending",
                    "portfolios": expected_portfolios, "measurements": []}
            document["cells"].append(cell)
        if cell.get("workload") != workload or cell.get("portfolios") != expected_portfolios:
            raise ValueError("composition workload or mapping portfolio changed on resume")
        for pattern in protocol["patterns"]:
            for trial in range(int(protocol["trials"])):
                # Alternate portfolio order per trial to keep the public
                # proposal and replayed cyclic cores from inheriting a fixed
                # launch-order bias.
                ordered_portfolios = cell["portfolios"][::1 if trial % 2 == 0 else -1]
                for portfolio in ordered_portfolios:
                    previous = [row for row in cell.get("measurements", [])
                                if row.get("pattern") == pattern and row.get("trial") == trial
                                and row.get("portfolio") == portfolio["name"]]
                    if previous:
                        if len(previous) != 1:
                            raise ValueError("duplicate composition trial")
                        validate_sample(previous[0]["sample"], workload, args.gpu_uuid, pattern, trial,
                                        protocol, portfolio.get("mappings"), manifest["compile_mode"])
                        continue
                    command = _command(Path(args.binary), workload, pattern, trial, protocol,
                                       portfolio.get("mappings"))
                    try:
                        sample = measure(command, env, args.gpu_uuid,
                                         args.output_dir / "logs" / f"{workload['id']}-{portfolio['name']}-{pattern}-{trial}.json")
                        validate_sample(sample, workload, args.gpu_uuid, pattern, trial, protocol,
                                        portfolio.get("mappings"), manifest["compile_mode"])
                    except BaseException as error:
                        _record_failure(document, workload, portfolio, pattern, trial, error)
                        cell["status"] = "interrupted" if isinstance(error, ExclusiveViolation) else "failed"
                        write_json(result_path, document)
                        write_json(args.output_dir / "summary.json", _summary(document, workloads, protocol))
                        raise
                    cell["measurements"].append({"pattern": pattern, "trial": trial,
                                                 "portfolio": portfolio["name"], "sample": sample})
                    cell["status"] = "partial"
                    write_json(result_path, document)
        cell["status"] = "complete"
        write_json(result_path, document)
    document["status"] = "complete"
    write_json(result_path, document)
    write_json(args.output_dir / "summary.json", _summary(document, workloads, protocol))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--workloads", type=Path, required=True,
                        help="JSON file containing a composed workloads list")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--build-dir", type=Path, required=True)
    result.add_argument("--compile-mode", default=DEFAULT_COMPILE_MODE)
    result.add_argument("--template-root", type=Path, default=ROOT)
    result.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    result.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    result.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    result.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS),
                        help="comma-separated input patterns")
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--mapping-sources", type=Path, action="append", default=[],
                        help="acceptance.json source; repeat for multiple sources")
    result.add_argument("--preflight", action="store_true",
                        help="run the 24-case-per-two-word-width GPU correctness gate first")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.patterns = tuple(pattern.strip() for pattern in args.patterns.split(",") if pattern.strip())
    if not args.patterns or any(pattern not in {"random", "boundary"} for pattern in args.patterns):
        parser().error("--patterns must contain random and/or boundary")
    if args.trials < 1 or args.warmup < 0 or args.repeat < 1:
        parser().error("trials/repeat must be positive and warmup must be non-negative")
    try:
        return run(args)
    except ExclusiveViolation as error:
        # The outer campaign controller can retry this bounded attempt after
        # the GPU becomes exclusive.  The raw failure remains checkpointed.
        print(f"run_crypto_composition: {error}", file=sys.stderr)
        return EXIT_RETRY
    except Exception as error:
        print(f"run_crypto_composition: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
