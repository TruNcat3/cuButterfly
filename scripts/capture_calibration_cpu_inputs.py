#!/usr/bin/env python3
"""Capture immutable calibration inputs before running the offline CPU preparer.

Capture is not CPU-only: listing real design points invokes the current build
and may query the selected GPU. It performs no timed measurements. Once this
manifest is published, its consumer needs only these frozen files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import time
from types import SimpleNamespace

from calibrate_local_hardware import measurement_identity
from calibration_seeds import candidates_for_workload
from calibration_space import runtime_candidates


SCHEMA = "cubutterfly-calibration-cpu-input-v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_json(path, value):
    _atomic(path, (_json(value) + "\n").encode())


def _ref(root, path):
    return {"path": str(path.relative_to(root)), "sha256": _sha(path.read_bytes())}


def _read_ref(root, reference):
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("capture reference escapes output directory")
    data = path.read_bytes()
    if _sha(data) != reference["sha256"]:
        raise ValueError("capture file checksum mismatch: " + str(path))
    return data


def _runner(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _environment():
    names = ("CUBUTTERFLY_COMPILE_MODE", "CUBUTTERFLY_PARTITION_BUDGET",
             "CUBUTTERFLY_PLATFORM_TUNING", "CUDA_VISIBLE_DEVICES",
             "CUBUTTERFLY_TEMPLATE_ROOT", "CUBUTTERFLY_MATHDX_ROOT",
             "CUBUTTERFLY_NVCC", "CUBUTTERFLY_JIT_CACHE", "CUBUTTERFLY_CALIBRATION_BROKER")
    return {name: os.environ.get(name) for name in names}


def capture(build_dir, calibration_dir, workloads, output, context_json,
            expected_measurement_json=None, *, runner=_runner):
    """Resume an interrupted capture without rereading active measurements."""
    build_dir, calibration_dir = pathlib.Path(build_dir), pathlib.Path(calibration_dir)
    output = pathlib.Path(output).resolve()
    root = output.parent
    state_path = output.with_name(output.name + ".capture-state.json")
    context = json.loads(pathlib.Path(context_json).read_bytes())
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    specification = json.loads(pathlib.Path(workloads).read_bytes())
    if specification.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported search workload schema")
    requested = [{"operator": "fft", **workload} for workload in specification["workloads"]]
    if len({_json(workload) for workload in requested}) != len(requested):
        raise ValueError("duplicate capture workload")
    expected = (json.loads(pathlib.Path(expected_measurement_json).read_bytes())
                if expected_measurement_json else context.get("measurement", context.get("measurement_identity")))
    if expected is not None and not isinstance(expected, dict):
        raise ValueError("expected measurement identity must be an object")
    invocation = dict(build_dir=str(build_dir.resolve()), context=context, workloads=requested,
                      environment=_environment(), source_sha256={name: _sha(
                          pathlib.Path(__file__).with_name(name).read_bytes()) for name in (
                              "capture_calibration_cpu_inputs.py", "calibration_space.py",
                              "calibration_seeds.py", "benchmark_semantics.py", "resident_mapping.py")})
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if state["invocation"] != invocation:
            raise ValueError("capture inputs/build/environment changed; use a new output")
        for reference in state["snapshots"].values():
            _read_ref(root, reference)
    else:
        # Read each active document once; preserve its exact bytes as evidence.
        names = {"profile": "calibration_device.json", "training": "search_measurements.json",
                 "descriptors": "plan_descriptors.json", "seed_snapshot": "mapping_seed_snapshot.json",
                 "search_coverage": "search_coverage.json"}
        frozen = {name: (calibration_dir / filename).read_bytes() for name, filename in names.items()}
        documents = {name: json.loads(data) for name, data in frozen.items()}
        recorded = documents["search_coverage"].get("measurement_identity")
        if not isinstance(recorded, dict):
            raise ValueError("active search coverage has no measurement identity")
        current = measurement_identity(SimpleNamespace(build_dir=build_dir,
                                                       verify_batches=recorded.get("verify_batches", 0)))
        if recorded != current or (expected is not None and expected != current):
            raise ValueError("active/expected measurement identity does not match current build")
        descriptor_measurement = documents["descriptors"].get("identity", {}).get("measurement")
        if descriptor_measurement is not None and descriptor_measurement != current:
            raise ValueError("descriptor measurement identity does not match current build")
        snapshots = {}
        for name, data in frozen.items():
            target = root / "inputs" / (pathlib.Path(names[name]).stem + "-" + _sha(data) + ".json")
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("frozen snapshot checksum mismatch: " + str(target))
            else:
                _atomic(target, data)
            snapshots[name] = _ref(root, target)
        state = dict(schema=SCHEMA, invocation=invocation, measurement=current,
                     snapshots=snapshots, inventories={})
        _write_json(state_path, state)

    current = measurement_identity(SimpleNamespace(build_dir=build_dir,
                                                   verify_batches=state["measurement"].get("verify_batches", 0)))
    if current != state["measurement"] or (expected is not None and expected != current):
        raise ValueError("capture measurement identity no longer matches current build")
    coverage = json.loads(_read_ref(root, state["snapshots"]["search_coverage"]))
    seeds = json.loads(_read_ref(root, state["snapshots"]["seed_snapshot"]))
    complete = {_json({"operator": "fft", **cell["workload"]})
                for cell in coverage.get("workloads", []) if cell.get("complete")}
    cells = []
    for workload in requested:
        key = _sha(_json(workload).encode())
        done = _json(workload) in complete
        identity = dict(measurement=current, workload=workload, environment=invocation["environment"],
                        source_sha256=invocation["source_sha256"],
                        policy="runtime-candidates-ordered-v1", complete=done)
        previous = state["inventories"].get(key)
        if previous:
            if previous["identity"] != identity:
                raise ValueError("capture inventory identity changed")
            _read_ref(root, previous["inventory"])
        else:
            binary = build_dir / ("cuntt_bench" if workload["operator"] == "ntt" else "cubutterfly_bench")
            points = [] if done else runtime_candidates(binary, workload, runner)
            if measurement_identity(SimpleNamespace(build_dir=build_dir,
                                    verify_batches=current.get("verify_batches", 0))) != current:
                raise ValueError("build changed while capturing candidates")
            target = root / "inventories" / (key + "-" + _sha(_json(identity).encode()) + ".jsonl.gz")
            payload = "".join(_json(point) + "\n" for point in points).encode()
            _atomic(target, gzip.compress(payload, mtime=0))
            previous = {"identity": identity, "inventory": _ref(root, target), "count": len(points)}
            state["inventories"][key] = previous
            _write_json(state_path, state)
        cells.append(dict(workload=workload, inventory=previous["inventory"], complete=done,
                          seeds=[item["point"] for item in candidates_for_workload(seeds, workload)]))
    manifest = dict(schema=SCHEMA, context=context,
                    **{name: state["snapshots"][name] for name in ("profile", "training", "descriptors")},
                    cells=cells, measurement_identity=current,
                    projection_scope=coverage.get("search_protocol", {}).get("search_stage_projection", "full"),
                    capture="GPU-capable inventory capture; no timed measurements",
                    snapshots=state["snapshots"])
    _write_json(output, manifest)
    return manifest


def rebind_score_context(input_manifest, context_json, output_manifest):
    """Publish a new scoring context while preserving verified capture evidence.

    The output stays beside the source so all frozen relative paths retain
    their meaning. Capture state and the source manifest are never changed.
    No candidate enumeration, hardware query or measurement is performed.
    """
    source, target = pathlib.Path(input_manifest).resolve(), pathlib.Path(output_manifest).resolve()
    if source == target or source.parent != target.parent:
        raise ValueError("rebound manifest must use a different filename beside the original")
    data = source.read_bytes()
    manifest = json.loads(data)
    context = json.loads(pathlib.Path(context_json).read_bytes())
    previous = manifest.get("context")
    if (manifest.get("schema") != SCHEMA or not isinstance(previous, dict)
            or not isinstance(context, dict) or not isinstance(previous.get("measurement"), dict)):
        raise ValueError("unsupported capture manifest or scoring context")
    for key in ("measurement", "partition_budget", "projection_scope"):
        if previous.get(key) != context.get(key):
            raise ValueError("scoring context cannot change capture identity: " + key)
    if (manifest.get("measurement_identity") is not None and
            manifest["measurement_identity"] != context["measurement"]):
        raise ValueError("manifest measurement identity does not match scoring context")
    references = [manifest[name] for name in ("profile", "training", "descriptors")]
    references.extend(manifest.get("snapshots", {}).values())
    references.extend(cell["inventory"] for cell in manifest["cells"] if cell.get("inventory"))
    for reference in references:
        _read_ref(source.parent, reference)
    result = dict(manifest, context=context, context_updates=[*manifest.get("context_updates", []), {
        "updated_unix": time.time(), "source_manifest": source.name,
        "source_sha256": _sha(data), "previous_context": previous, "current_context": context,
        "policy": "CPU scoring context only; verified capture evidence unchanged"}])
    _write_json(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path)
    parser.add_argument("--calibration-dir", type=pathlib.Path)
    parser.add_argument("--workloads", type=pathlib.Path)
    parser.add_argument("--rebind-score-context", type=pathlib.Path, metavar="INPUT_MANIFEST",
                        help="CPU-only: publish a new scoring context beside a verified capture")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="published manifest JSON path")
    parser.add_argument("--context-json", type=pathlib.Path, required=True)
    parser.add_argument("--expected-measurement-json", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.rebind_score_context:
        if args.build_dir or args.calibration_dir or args.workloads or args.expected_measurement_json:
            parser.error("context rebinding accepts only --context-json and --output")
        rebind_score_context(args.rebind_score_context, args.context_json, args.output)
    else:
        if args.build_dir is None or args.calibration_dir is None or args.workloads is None:
            parser.error("capture requires --build-dir, --calibration-dir and --workloads")
        values = vars(args)
        values.pop("rebind_score_context")
        capture(**values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
