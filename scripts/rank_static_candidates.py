#!/usr/bin/env python3
"""Batch CPU ranking through the existing physical stage cost model."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

import stage_cost_model as costs
import static_hardware_profile as hardware_profiles


REQUEST_SCHEMA = "cubutterfly-static-ranking-request-v1"
SCHEMA = "cubutterfly-static-ranking-v1"
RATE_FIELDS = ("global_feedback_bytes_per_second", "equivalent_butterflies_per_second",
               "interstage_shared_bytes_per_second", "cta_barriers_per_second")


def _read(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _identity(profile, hardware, source):
    for key, hardware_key in (("device", "device"), ("compute_capability", "compute_capability"),
                              ("global_memory_bytes", "memory_bytes")):
        if profile.get(key) != hardware[hardware_key]:
            raise ValueError(f"{source}: hardware identity mismatch: {key}")


def _model_reason(model, request):
    reason = hardware_profiles.cost_model_validation_reason(model)
    if reason:
        return reason
    identity = model.get("identity", {})
    fingerprint = model.get("runtime_fingerprint", identity.get("runtime_fingerprint"))
    if fingerprint != request["runtime_fingerprint"]:
        return "cost model runtime fingerprint is missing or incompatible"
    protocol = model.get("protocol", identity.get("protocol"))
    if not isinstance(protocol, dict) or not protocol:
        return "cost model timing protocol is missing"
    expected = request.get("timing_protocol")
    if expected is None:
        return "request timing protocol is unspecified; model protocol compatibility is unproven"
    expected_timing = expected.get("timing") if isinstance(expected, dict) else expected
    if expected_timing != "direct-launch" or protocol.get("timing") != expected_timing:
        return "cost model timing protocol is incompatible"
    try:
        _identity(model["profile"], request["hardware"], "cost model profile")
    except (KeyError, ValueError) as error:
        return str(error)
    return ""


def _positive_rates(profile):
    caps = profile.get("capabilities", {})
    return all(isinstance(caps.get(key), (int, float)) and not isinstance(caps[key], bool)
               and math.isfinite(caps[key]) and caps[key] > 0 for key in RATE_FIELDS)


def _sample(value):
    # Normalization only changes top-level fields. PreparedPredictor owns the
    # independent snapshot used for scoring; nested descriptors stay read-only.
    sample = dict(value)
    mapping = costs.decoded(sample.get("mapping_json"), {})
    if not isinstance(mapping, dict):
        raise ValueError("mapping_json must encode an object")
    for key, item in mapping.items():
        sample.setdefault(key, item)
    sample["mapping_json"] = mapping
    groups = costs.decoded(sample.get("execution_groups_json"), [])
    if not isinstance(groups, list) or not groups or not all(isinstance(g, dict) for g in groups):
        raise ValueError("physical execution groups must be a nonempty array of objects")
    sample["execution_groups_json"] = groups
    sample.setdefault("descriptor_source", "runtime-static-projection")
    return sample


def _single_chunk_service_view(sample):
    """Use the shared runtime's serial fast path without changing its mapping.

    With one scheduler chunk, overlap kernels launch sequentially and each CTA
    consumes one batch item. Their services are the same as bulk batch_tile=1.
    This private cost view does not rewrite the user's executable mapping.
    """
    if (sample.get("backend") != "shared-iterative"
            or not costs.number(sample, "stage_overlap") or costs.number(sample, "factor_overlap")):
        return sample, False
    batch = int(costs.number(sample, "batch", 1))
    tile = int(costs.number(sample, "batch_tile_count", 1))
    if batch < 1 or tile < batch:
        return sample, False
    serial = dict(sample)
    serial.update(stage_overlap=False, batch_tile_count=1)
    mapping = dict(costs.decoded(serial.get("mapping_json"), {}))
    mapping.update(stage_overlap=False, batch_tile_count=1)
    serial["mapping_json"] = mapping
    serial.pop("stage_service_projection", None)
    return serial, True


def _rank(request):
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported static-ranking request schema")
    hardware = request["hardware"]
    hardware_profiles.validate(dict(schema=hardware_profiles.SCHEMA,
        hardware={key: hardware[key] for key in hardware_profiles.FIELDS}, provenance={}))
    if not isinstance(request.get("runtime_fingerprint"), str) or not request["runtime_fingerprint"]:
        raise ValueError("runtime_fingerprint must be a nonempty string")
    candidates = request["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidates must be an array")
    indices = [item["index"] for item in candidates]
    if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be distinct nonnegative integers")
    result = dict(schema=SCHEMA, mode="unavailable", profile_source="", reason="no static performance profile",
                  ranked=[], rejected=[], unranked=[])
    profile, model = None, None
    directory = request.get("profile_dir")
    if directory is not None:
        directory = Path(directory)
        for filename in ("calibration_device.json", "hardware_profile.json"):
            source = directory / filename
            if source.exists():
                profile = _read(source)
                _identity(profile, hardware, str(source))
                result["profile_source"] = str(source)
                result["reason"] = "no compatible validated cost model"
                break
        model_file = directory / "cost_model.json"
        if not model_file.exists() and profile is not None:
            result["reason"] = f"static cost model file is absent: {model_file}"
        if model_file.exists():
            candidate_model = _read(model_file)
            reason = _model_reason(candidate_model, request)
            if reason:
                result["reason"] = reason
            else:
                model = copy.deepcopy(candidate_model)
                model["profile"]["hardware"] = dict(hardware)
                model["profile"]["sm_count"] = hardware["sm_count"]
                result.update(mode="validated-static-model", profile_source=str(model_file),
                              reason="compatible validated stage cost model; candidate coverage checked individually")
    prior = None
    if profile is not None:
        profile = copy.deepcopy(profile)
        profile["hardware"] = dict(hardware)
        profile["sm_count"] = hardware["sm_count"]
        profile.pop("stage_service", None)
        if _positive_rates(profile):
            prior = costs.fit([], profile)
            if model is None:
                result["mode"] = "hardware-rate-prior"
                result["reason"] += "; using measured hardware-rate priors, not validated whole-plan predictions"
        elif model is None:
            result["reason"] += "; measured hardware capability rates are missing or invalid"
    predictor = costs.PreparedPredictor(model) if model is not None else None
    score = predictor.bind(model) if predictor else None
    prior_predictor = costs.PreparedPredictor(prior) if prior is not None else None
    prior_score = prior_predictor.bind(prior) if prior_predictor else None
    for candidate in candidates:
        index = candidate["index"]
        try:
            value = candidate["sample"]
            descriptor_status = value.get("physical_descriptor_status", "projected")
            if descriptor_status == "unavailable":
                result["unranked"].append(dict(index=index, reason=str(
                    value.get("physical_descriptor_reason") or "physical descriptor unavailable")))
                continue
            if descriptor_status not in ("projected", "resolved"):
                raise ValueError(f"unknown physical_descriptor_status: {descriptor_status!r}")
            sample = _sample(value)
            for position, group in enumerate(sample["execution_groups_json"]):
                costs._resource_residency(group, {"hardware": hardware}, position)
            sample, serial_single_chunk = _single_chunk_service_view(sample)
            details, confidence, reason = None, "hardware-rate-prior", result["reason"]
            overlap = bool(costs.number(sample, "stage_overlap") or costs.number(sample, "factor_overlap"))
            if predictor is not None and (not overlap or
                    model.get("composition_validation", {}).get("status") == "passed"):
                details = score(predictor.prepare(sample), True)
                if (details.get("missing_primitives") or details.get("extrapolated_primitives")
                        or details.get("unmeasured_groups")):
                    details = None
                    reason = "candidate exceeds validated model primitive coverage; using hardware-rate priors"
                else:
                    confidence = "validated-static-model"
                    reason = "validated model estimate; not an exact measured workload"
            elif predictor is not None:
                reason = "overlap composition is unvalidated; using hardware-rate priors"
            if details is None and prior_predictor is not None:
                details = prior_score(prior_predictor.prepare(sample), True)
            if details is None:
                result["unranked"].append(dict(index=index, reason=reason))
                continue
            if details.get("status") == "infeasible":
                result["rejected"].append(dict(index=index, reason=details["reason"]))
                continue
            estimate = details["kernel_ms"]
            if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate <= 0:
                raise ValueError("stage model produced a nonpositive or nonfinite estimate")
            row = dict(index=index, predicted_kernel_ms=estimate, confidence=confidence, reason=reason)
            if serial_single_chunk:
                row["service_view"] = "equivalent-serial-single-chunk"
                row["stage_overlap_effective"] = False
            result["ranked"].append(row)
        except costs.ResourceInfeasibleError as error:
            result["rejected"].append(dict(index=index, reason=str(error)))
        except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
            result["unranked"].append(dict(index=index, reason=str(error)))
    result["ranked"].sort(key=lambda item: item["predicted_kernel_ms"])
    confidence_modes = {item["confidence"] for item in result["ranked"]}
    if len(confidence_modes) > 1:
        result["mode"] = "mixed-static-model-and-hardware-rate-prior"
    elif confidence_modes:
        result["mode"] = next(iter(confidence_modes))
    else:
        result["mode"] = "unavailable"
    return result


def rank(request):
    # Reuse the source/architecture-keyed CPU accelerator without compiling.
    # An explicit user disable remains authoritative.
    import stage_schedule_native
    previous = os.environ.get("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY")
    os.environ["CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY"] = "1"
    try:
        result = _rank(request)
        result["schedule_backend"] = stage_schedule_native.backend()
        return result
    finally:
        if previous is None:
            os.environ.pop("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", None)
        else:
            os.environ["CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rank(_read(args.request))
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"static ranking error: {error}", file=sys.stderr)
        raise SystemExit(1)
