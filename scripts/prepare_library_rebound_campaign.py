#!/usr/bin/env python3
"""Bind an existing semantic population to a newly verified library build.

The source campaign remains immutable.  This command creates a fresh manifest
whose semantic cases are unchanged, but whose library fingerprint and
calibration identity point at the new build.  It is intended for a controlled
continuation after a library rebuild, never for adopting rows measured by the
previous binary.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from semantic_campaign_manifest import manifest_hash, validate_manifest_hash
import run_semantic_campaign as runner
import semantic_compile_broker as broker
import jit_calibration_identity as jit


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evidence(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": runner.sha256_file(path)}


def rebound_manifest(source: Path, library: Path, revision: str,
                     c_api_source: Path | None = None) -> dict:
    parent = json.loads(source.read_text())
    if not validate_manifest_hash(parent):
        raise ValueError(f"source manifest checksum mismatch: {source}")
    old_library = parent["fingerprints"]["library"]
    value = copy.deepcopy(parent)
    source_changes = []
    current = Path(__file__).resolve().parents[1] / "src/c_api.cpp"
    matches = [item for item in value["fingerprints"]["sources"]
               if Path(item["path"]).name in {"c_api.cpp", "historical_c_api.cpp"}]
    for item in matches:
        if Path(item["path"]).name == "historical_c_api.cpp":
            if evidence(Path(item["path"]))["sha256"] != item["sha256"]:
                raise ValueError("historical C API source fingerprint mismatch")
            if c_api_source is None and runner.sha256_file(current) != item["sha256"]:
                raise ValueError("historical C API differs from current source; explicit source rebound required")
    if c_api_source is not None:
        if c_api_source.resolve(strict=True) != current.resolve(strict=True):
            raise ValueError("C API rebound source must be current repository src/c_api.cpp")
        if len(matches) != 1:
            raise ValueError("C API rebound requires one historical C API fingerprint")
        item = matches[0]
        new = dict(evidence(current), bytes=current.stat().st_size)
        source_changes.append({"historical_source": copy.deepcopy(item), "current_source": new})
        item.clear()
        item.update(new)
    # Library replacement is intentional; all remaining source/config inputs must match.
    for item in value["fingerprints"]["sources"]:
        if runner.sha256_file(Path(item["path"]).resolve(strict=True)) != item["sha256"]:
            raise ValueError("manifest source fingerprint mismatch: " + item["path"])
    config = value["fingerprints"].get("config", {})
    if config.get("path") and evidence(Path(config["path"]))["sha256"] != config["sha256"]:
        raise ValueError("manifest config fingerprint mismatch")
    value["fingerprints"]["library"] = {
        "path": str(library.resolve(strict=True)),
        "bytes": library.stat().st_size,
        "sha256": runner.sha256_file(library),
    }
    value["campaign_revision"] = revision
    provenance = {
        "parent": evidence(source),
        "parent_library": old_library,
        "rebound_library": value["fingerprints"]["library"],
        "scope": "same semantic population; zero historical rows adopted after library rebuild",
        "source_rebindings": source_changes,
    }
    if "continuation_provenance" in parent:
        provenance["parent_continuation_provenance"] = copy.deepcopy(parent["continuation_provenance"])
    value["continuation_provenance"] = provenance
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = manifest_hash(value)
    return value


def validate_calibration(plan: dict, calibration: dict) -> dict:
    if (calibration.get("schema") != "cubutterfly-local-calibration-v1"
            or calibration.get("status") != "ready-with-measured-mappings"):
        raise ValueError("unsupported or incomplete calibration manifest")
    if calibration.get("selector_replay", {}).get("status") != "passed":
        raise ValueError("calibration is not finalized with passed selector replay")
    missing = {"hardware_rates_source", "global_memory_bytes", "registry_path", "cost_model"} - calibration.keys()
    if missing:
        raise ValueError("calibration missing required fields: " + ", ".join(sorted(missing)))
    profile_path = Path(calibration["hardware_rates_source"]).resolve(strict=True)
    profile = json.loads(profile_path.read_text())
    gpu_uuid = profile.get("provenance", {}).get("cuda_visible_devices")
    if not gpu_uuid or not runner.same_gpu_uuid(plan["gpu_uuid"], gpu_uuid):
        raise ValueError("calibration GPU UUID differs from source plan")
    identity = {"library_sha256": runner.sha256_file(plan["library"]),
                "compile_mode": "research", "gpu_name": profile["device"],
                "gpu_total_memory": profile.get("global_memory_bytes", calibration["global_memory_bytes"]),
                "compute_capability": str(profile["compute_capability"])}
    runner.validate_calibration(calibration, identity)
    for field in ("registry_path", "cost_model"):
        Path(calibration[field]).resolve(strict=True)
    document = broker._document(Path(plan["compile_broker"]).resolve(strict=True))
    runtime = jit.validate_timing(calibration, document)
    return {"gpu_uuid": gpu_uuid, "hardware_profile": evidence(profile_path),
            "jit_runtime": runtime,
            "memory_identity_validation": ("hardware-profile" if "global_memory_bytes" in profile
                                           else "recorded-calibration-only; hardware profile omits memory capacity")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-socket", type=Path,
                        help="short existing-socket path; Unix socket paths have a strict length limit")
    parser.add_argument("--rebind-c-api-source", type=Path,
                        help="explicitly replace historical c_api.cpp fingerprint with current repository source")
    parser.add_argument("--revision", default="0.9.3.5-library-rebound-20261011")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output must be fresh: {output}")
    source_plan = args.source_plan.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    library = args.library.resolve(strict=True)
    calibration = args.calibration.resolve(strict=True)
    plan = json.loads(source_plan.read_text())
    selected = [phase for phase in plan["phases"]
                if Path(phase["manifest"]).resolve(strict=True) == source_manifest]
    if len(selected) != 1:
        raise ValueError("source manifest must belong to exactly one source-plan phase")
    omitted = [copy.deepcopy(phase) for phase in plan["phases"] if phase is not selected[0]]
    rebound_path = output / "manifest.json"
    manifest = rebound_manifest(source_manifest, library, args.revision, args.rebind_c_api_source)
    plan["calibration_manifest"] = str(calibration)
    plan["library"] = str(library)
    calibration_validation = validate_calibration(plan, json.loads(calibration.read_text()))
    phase = copy.deepcopy(selected[0])
    phase["manifest"] = str(rebound_path)
    plan["phases"] = [phase]
    plan["rebound_scope"] = {"selected_phase": copy.deepcopy(selected[0]),
                             "omitted_phases": omitted, "full_source_plan": not omitted}
    socket_path = (args.compile_socket or (output / "compile.sock")).resolve()
    if len(os.fsencode(socket_path)) > 107:
        raise ValueError("compile socket path exceeds AF_UNIX 107-byte limit")
    plan["compile_socket"] = str(socket_path)
    plan["historical_completed_adopted"] = False
    plan["new_completed_cases"] = 0
    plan["revision"] = args.revision
    plan_path = output / source_plan.name
    if plan_path.name in {"manifest.json", "provenance.json"}:
        plan_path = output / "plan.json"
    save(rebound_path, manifest)
    save(plan_path, plan)
    save(output / "provenance.json", {
        "schema": "cubutterfly-library-rebound-campaign-v1",
        "revision": args.revision,
        "source_plan": evidence(source_plan),
        "source_manifest": evidence(source_manifest),
        "library": evidence(library),
        "calibration": evidence(calibration),
        "new_manifest": evidence(rebound_path),
        "historical_rows_adopted": 0,
        "selected_phase": copy.deepcopy(selected[0]),
        "omitted_phases": omitted,
        "full_source_plan": not omitted,
        "calibration_validation": calibration_validation,
        "source_rebindings": manifest["continuation_provenance"]["source_rebindings"],
    })
    print(json.dumps({"plan": str(plan_path),
                      "manifest": str(rebound_path),
                      "library_sha256": runner.sha256_file(library),
                      "cases": len(manifest["cases"]),
                      "selected_phase": phase["name"],
                      "omitted_phases": [item["name"] for item in omitted],
                      "full_source_plan": not omitted},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
