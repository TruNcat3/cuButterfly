#!/usr/bin/env python3
"""CPU-only validation, selection and explicit export of hardware artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "cubutterfly-hardware-model-v1"
FILENAME = "hardware_model.json"
INTEGER_CAPS = ("memory_bytes", "sm_count", "max_threads_per_block",
                "max_threads_per_sm", "max_blocks_per_sm", "registers_per_sm",
                "registers_per_block", "shared_bytes_per_sm", "shared_bytes_per_block")
FIELDS = {"device", "compute_capability", "cooperative_launch", *INTEGER_CAPS}


def identity(document):
    hardware = document["hardware"]
    return dict(device=hardware["device"], compute_capability=hardware["compute_capability"],
                global_memory_bytes=hardware["memory_bytes"])


def validate(document, expected_identity=None):
    """Return the validated document; never fill unknown resource capacities."""
    if not isinstance(document, dict) or set(document) != {"schema", "hardware", "provenance"}:
        raise ValueError("hardware model requires exactly schema, hardware and provenance")
    if document["schema"] != SCHEMA:
        raise ValueError("unsupported hardware model schema")
    hardware = document["hardware"]
    if not isinstance(hardware, dict) or set(hardware) != FIELDS:
        raise ValueError("hardware requires full exact capability fields: " + ", ".join(sorted(FIELDS)))
    if not isinstance(hardware["device"], str) or not hardware["device"].strip():
        raise ValueError("hardware.device must be a nonempty string")
    if not isinstance(hardware["compute_capability"], str) or not re.fullmatch(r"[1-9][0-9]*\.[0-9]+", hardware["compute_capability"]):
        raise ValueError("compute_capability must use major.minor format")
    for key in INTEGER_CAPS:
        maximum = (1 << (64 if key == "memory_bytes" else 32)) - 1
        if type(hardware[key]) is not int or not 0 < hardware[key] <= maximum:
            raise ValueError(f"hardware.{key} must be a positive uint{64 if key == 'memory_bytes' else 32} integer")
    if type(hardware["cooperative_launch"]) is not bool:
        raise ValueError("hardware.cooperative_launch must be boolean")
    for block, sm in (("max_threads_per_block", "max_threads_per_sm"),
                      ("registers_per_block", "registers_per_sm"),
                      ("shared_bytes_per_block", "shared_bytes_per_sm")):
        if hardware[block] > hardware[sm]:
            raise ValueError(f"hardware.{block} exceeds {sm}")
    if not isinstance(document["provenance"], dict):
        raise ValueError("provenance must be an object")
    if expected_identity is not None:
        actual = identity(document)
        for key in actual:
            if key not in expected_identity or actual[key] != expected_identity[key]:
                raise ValueError(f"hardware identity mismatch: {key}")
    return document


def load(hardware_path, expected_identity=None):
    return validate(json.loads(Path(hardware_path).read_text()), expected_identity)


def _tag(expected_identity):
    device = re.sub(r"[^a-zA-Z0-9.-]+", "-", str(expected_identity["device"])).strip("-") or "unknown"
    capability = str(expected_identity["compute_capability"]).replace(".", "")
    return f"{device}-sm{capability}-{int(expected_identity['global_memory_bytes'])}B"


def compatible_sources(expected_identity, *, explicit=None, local_dir=None, bundled_root=None):
    """Inspect finite explicit/local/bundled candidates in precedence order."""
    candidates = []
    if explicit is not None:
        candidates.append(("explicit", Path(explicit)))
    if local_dir is not None:
        candidates.append(("local", Path(local_dir) / FILENAME))
    if bundled_root is not None:
        candidates.append(("bundled", Path(bundled_root) / _tag(expected_identity) / FILENAME))
    result = []
    for source, path in candidates:
        record = dict(source=source, path=str(path), compatible=False)
        try:
            load(path, expected_identity)
            record["compatible"] = True
        except (OSError, ValueError, TypeError) as error:
            record["reason"] = str(error)
        result.append(record)
    return result


def cost_model_validation_reason(model):
    """Check producer validation only; request compatibility is a separate gate."""
    from stage_cost_model import VERSION
    if model.get("schema") != "cubutterfly-stage-cost-model-v1" or model.get("version") != VERSION:
        return "cost model schema/version is unsupported or stale"
    if (model.get("usable_for_unmeasured_ranking") is not True
            or model.get("status") not in ("calibrated-local", "calibrated-local-serial")):
        return "cost model is not validated for unmeasured ranking"
    validation = model.get("validation", {})
    if validation.get("scope") not in (
            "leave_one_complete_workload_out", "independent-stage-holdout-plus-serial-ranking"):
        return "cost model lacks the producer's complete-workload validation scope"
    parameters = model.get("parameters")
    if not isinstance(parameters, dict) or any(
            not isinstance(values, dict) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 for value in values.values())
            for values in parameters.values()):
        return "cost model coefficients are invalid"
    profile = model.get("profile")
    if not isinstance(profile, dict):
        return "cost model profile is missing or invalid"
    service = model.get("stage_service") or profile.get("stage_service")
    if service:
        from calibrate_local_hardware import _stage_service_usable_for_ranking
        if not _stage_service_usable_for_ranking({"stage_service": service}):
            return "cost model stage-service scope is unvalidated"
        if (model.get("whole_plan_validation", {}).get("status") != "passed"
                and validation.get("global_validation", {}).get("passed") is not True):
            return "cost model complete-plan service composition is unvalidated"
    return ""


def _performance_layers(directory):
    result = {}
    for name, filename in (("hardware_profile", "hardware_profile.json"),
                           ("cost_model", "cost_model.json"),
                           ("stage_service", "stage_service_profile.json")):
        path = directory / filename
        layer = dict(present=path.is_file(), path=str(path), validated=False)
        if layer["present"]:
            try:
                value = json.loads(path.read_text())
                for key in ("schema", "version", "status", "calibration_status",
                            "usable_for_unmeasured_ranking", "validated", "inventory_complete"):
                    if key in value:
                        layer[key] = value[key]
                if name == "cost_model":
                    reason = cost_model_validation_reason(value)
                    layer.update(validated=not reason, validation_reason=reason,
                                 compatibility="requires-request-check")
                elif name == "stage_service":
                    layer["validated"] = (value.get("validated") is True
                                          and value.get("calibration_status") == "complete")
                else:
                    # Measured rates are a separate layer, not model validation.
                    layer["validated"] = False
            except (OSError, ValueError, TypeError, AttributeError) as error:
                layer.update(validated=False, error=str(error))
        result[name] = layer
    return result


def inspect(hardware_path, expected_identity=None):
    document = load(hardware_path, expected_identity)
    return dict(path=str(hardware_path), identity=identity(document), hardware_complete=True,
                provenance=document["provenance"], layers=_performance_layers(Path(hardware_path).parent))


def inspect_legacy(profile_path, expected_identity):
    """Inspect a matching legacy calibration without upgrading its contract."""
    path = Path(profile_path)
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != "cubutterfly-hardware-profile-v1":
        raise ValueError("unsupported legacy calibration profile schema")
    actual = {key: document.get(key) for key in
              ("device", "compute_capability", "global_memory_bytes")}
    if type(actual["global_memory_bytes"]) is not int or actual["global_memory_bytes"] <= 0:
        raise ValueError("legacy calibration requires exact memory identity")
    if any(actual[key] != expected_identity.get(key) for key in actual):
        raise ValueError("legacy hardware identity mismatch")
    hardware = document.get("hardware", {})
    if not isinstance(hardware, dict):
        raise ValueError("legacy hardware capabilities must be an object")
    missing = sorted(FIELDS - set(hardware))
    return dict(path=str(path), identity=actual, hardware_complete=False,
                hardware_status="legacy-not-validated-static-caps", missing_capabilities=missing,
                provenance=document.get("provenance", {}),
                layers=_performance_layers(path.parent))


def select(expected_identity, *, explicit=None, local_dir=None, bundled_root=None):
    sources = compatible_sources(expected_identity, explicit=explicit, local_dir=local_dir,
                                 bundled_root=bundled_root)
    authoritative = [sources[0]] if explicit is not None else sources
    broken = None
    for item in authoritative:
        if item["compatible"]:
            break
        path = Path(item["path"])
        if explicit is not None or path.exists() or path.is_symlink():
            broken = item
            break
    if broken is not None:
        return dict(selected=None, sources=sources, action="error",
                    error="invalid hardware artifact: " + broken["reason"],
                    recalibration_requested=False)
    # An explicit request must not silently select a different artifact.
    chosen = next((item for item in sources if item["compatible"]), None)
    if explicit is not None and not sources[0]["compatible"]:
        chosen = None
    selected = inspect(chosen["path"], expected_identity) if chosen else None
    action = "reuse" if chosen else "hardware-artifact-required"
    legacy = []
    if explicit is None and local_dir is not None:
        legacy.append(("local-legacy", Path(local_dir) / "calibration_device.json"))
    for source, path in legacy:
        record = dict(source=source, path=str(path), compatible=False)
        try:
            partial = inspect_legacy(path, expected_identity)
            record["compatible"] = True
            if selected is None:
                selected = partial
                action = "reuse-partial"
        except (OSError, ValueError, TypeError, AttributeError) as error:
            record["reason"] = str(error)
        sources.append(record)
    return dict(selected=selected,
                sources=sources, action=action,
                recalibration_requested=False)


def export_from_calibration(profile_dir, output_path, supplemental_hardware=None, *, refresh=False):
    """Export complete caps; supplements require explicit verified source files.

    supplemental_hardware is {hardware: {...}, provenance: {verified: true,
    sources: [existing-file, ...]}}. These evidence files are hashed, not queried.
    """
    output = Path(output_path)
    if output.exists() and not refresh:
        return load(output)
    directory = Path(profile_dir)
    sources = []
    historical = {}
    for name in ("hardware_profile.json", "calibration_device.json"):
        path = directory / name
        if path.is_file():
            historical.update(json.loads(path.read_text()))
            sources.append(path)
    if not historical:
        raise ValueError("no existing calibration profile found")
    hardware = dict(historical.get("hardware", {}))
    hardware.update(device=historical.get("device"), compute_capability=historical.get("compute_capability"))
    if "global_memory_bytes" in historical:
        hardware["memory_bytes"] = historical["global_memory_bytes"]
    supplement_provenance = None
    if supplemental_hardware is not None:
        supplement_provenance = supplemental_hardware.get("provenance", {})
        evidence = supplement_provenance.get("sources", [])
        if (supplement_provenance.get("verified") is not True or not evidence
                or not isinstance(evidence, list)):
            raise ValueError("supplemental hardware requires verified source files")
        sources.extend(Path(path) for path in evidence)
        for key, value in supplemental_hardware.get("hardware", {}).items():
            if key in hardware and hardware[key] != value:
                raise ValueError(f"supplemental hardware contradicts existing {key}")
            hardware[key] = value
    layers = _performance_layers(directory)
    sources.extend(Path(layer["path"]) for layer in layers.values() if layer["present"])
    document = dict(schema=SCHEMA, hardware=hardware, provenance=dict(
        calibration_provenance=historical.get("provenance", {}),
        supplemental_provenance=supplement_provenance,
        performance_layers=layers,
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources}))
    validate(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects default reuse from a concurrent exporter.
    with output.open("w" if refresh else "x") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("inspect")
    view.add_argument("hardware_path", type=Path)
    view.add_argument("--expected-identity", type=Path)
    export = commands.add_parser("export")
    export.add_argument("profile_dir", type=Path)
    export.add_argument("output_path", type=Path)
    export.add_argument("--supplemental-hardware", type=Path)
    export.add_argument("--refresh", action="store_true")
    choose = commands.add_parser("select")
    choose.add_argument("--expected-identity", type=Path, required=True)
    choose.add_argument("--explicit", type=Path)
    choose.add_argument("--local-dir", type=Path)
    choose.add_argument("--bundled-root", type=Path)
    args = parser.parse_args(argv)
    try:
        expected = json.loads(args.expected_identity.read_text()) if getattr(args, "expected_identity", None) else None
        if args.command == "inspect":
            result = inspect(args.hardware_path, expected)
        elif args.command == "select":
            result = select(expected, explicit=args.explicit, local_dir=args.local_dir, bundled_root=args.bundled_root)
        else:
            supplemental = json.loads(args.supplemental_hardware.read_text()) if args.supplemental_hardware else None
            result = export_from_calibration(args.profile_dir, args.output_path, supplemental, refresh=args.refresh)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps(dict(error=str(error))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
