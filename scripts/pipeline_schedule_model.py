"""Measured lower bounds for the BatchPipeline event/submission schedule.

This model deliberately describes only the minimum scheduling floor exposed by
the one-CTA pipeline probe.  It is not an operator service model and it does
not claim that a complete multi-kernel concurrency calibration has been done.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from statistics import median
from typing import Any, Callable


SCHEMA = "cubutterfly-pipeline-schedule-v1"
VERSION = "pipeline-schedule-piecewise-v1"
POLICY_VERSION = "batch-pipeline-minimum-floor-v1"

DEFAULT_GROUPS = (2, 3, 4)
DEFAULT_TILES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
TRAINING_TILES = (1, 2, 4, 8, 16, 32)
VALIDATION_TILES = (3, 6, 12, 24)
# These aliases make the train/holdout split explicit to callers that build
# their own calibration reports.
TRAIN_TILES = TRAINING_TILES
HOLDOUT_TILES = VALIDATION_TILES

DEFAULT_PROTOCOL = {"warmup": 10, "repeat": 100, "trials": 3}
POLICY = {
    "version": POLICY_VERSION,
    "groups": list(DEFAULT_GROUPS),
    "tiles": list(DEFAULT_TILES),
    "training_tiles": list(TRAINING_TILES),
    "validation_tiles": list(VALIDATION_TILES),
    "batch": "tiles",
    "tile_batch": 1,
    "kernel": "one-CTA one-thread distinct uint64 store",
    "dispatch": "BatchPipeline::enqueue",
}


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (positive and number <= 0):
        relation = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {relation}")
    return number


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "correct", "measured"}
    return bool(value)


def _json_copy(value: Any) -> Any:
    """Copy JSON-like metadata while rejecting non-serializable cache input."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("pipeline schedule metadata must be JSON serializable") from error


def _measurement_values(row: dict[str, Any]) -> list[float]:
    values = row.get("trial_kernel_ms")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("pipeline schedule rows require non-empty trial_kernel_ms")
    result = []
    for index, value in enumerate(values):
        result.append(_finite_number(value, name=f"trial_kernel_ms[{index}]", positive=True))
    return result


def _segment_points(points: dict[int, float]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for lower_tiles, upper_tiles in zip(TRAINING_TILES, TRAINING_TILES[1:]):
        if lower_tiles not in points or upper_tiles not in points:
            continue
        lower_ms = points[lower_tiles]
        upper_ms = points[upper_tiles]
        slope = (upper_ms - lower_ms) / float(upper_tiles - lower_tiles)
        segments.append({
            "tiles": [lower_tiles, upper_tiles],
            "lower_tiles": lower_tiles,
            "upper_tiles": upper_tiles,
            "slope_ms_per_tile": slope,
            "intercept_ms": lower_ms - slope * lower_tiles,
        })
    return segments


def _predict_training(points: dict[int, float], tiles: int) -> tuple[float | None, str]:
    """Interpolate only between measured training endpoints; never extrapolate."""
    if tiles in points:
        return points[tiles], "measured"
    if not points or tiles < TRAINING_TILES[0] or tiles > TRAINING_TILES[-1]:
        return None, "missing"
    for lower_tiles, upper_tiles in zip(TRAINING_TILES, TRAINING_TILES[1:]):
        if lower_tiles < tiles < upper_tiles:
            if lower_tiles not in points or upper_tiles not in points:
                return None, "missing"
            fraction = (tiles - lower_tiles) / float(upper_tiles - lower_tiles)
            return points[lower_tiles] + fraction * (points[upper_tiles] - points[lower_tiles]), "interpolated"
    # The only remaining possibility is a gap caused by malformed/non-integer
    # input.  Keep it observable as missing instead of silently extrapolating.
    return None, "missing"


def _holdout_check(tiles: int, measured: float | None, points: dict[int, float]) -> dict[str, Any]:
    if measured is None:
        return {"tiles": tiles, "measured_ms": None, "predicted_ms": None, "relative_error": None,
                "status": "missing", "independent": True}
    predicted, state = _predict_training(points, tiles)
    if predicted is None:
        return {"tiles": tiles, "measured_ms": measured, "predicted_ms": None, "relative_error": None,
                "status": state, "independent": True}
    relative_error = abs(predicted - measured) / measured
    return {"tiles": tiles, "measured_ms": measured, "predicted_ms": predicted,
            "relative_error": relative_error, "status": "covered", "independent": True}


def _validation_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [item for item in checks if item.get("status") == "covered"]
    errors = [float(item["relative_error"]) for item in covered if item.get("relative_error") is not None]
    missing = [item for item in checks if item.get("status") == "missing"]
    outside = [item for item in checks if item.get("status") == "outside"]
    if not checks:
        status = "insufficient-holdout"
    elif len(covered) == len(checks):
        # Coverage is useful provenance, but no accuracy threshold is defined
        # for this scheduling floor.  Do not call it an error-validation pass.
        status = "covered"
    else:
        status = "incomplete"
    result: dict[str, Any] = {
        "status": status,
        "holdout_count": len(checks),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "outside_count": len(outside),
        "relative_errors": errors,
        "mean_relative_error": (sum(errors) / len(errors)) if errors else None,
        "max_relative_error": max(errors) if errors else None,
    }
    # Keep the names used by the other calibration summaries available without
    # changing the independent train/holdout semantics here.
    result["heldout_count"] = result["holdout_count"]
    result["heldout_covered_count"] = result["covered_count"]
    result["heldout_uncovered_count"] = result["missing_count"] + result["outside_count"]
    result["heldout_mean_relative_error"] = result["mean_relative_error"]
    result["heldout_max_relative_error"] = result["max_relative_error"]
    return result


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    """Fit per-group piecewise-linear curves from the designated train points.

    Rows at 3, 6, 12 and 24 tiles are retained only as independent validation;
    they never enter an interpolation segment.  Missing train or holdout rows
    remain explicit in the returned report.
    """
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("unsupported pipeline schedule microbenchmark schema")
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise ValueError("pipeline schedule profile rows must be an array")

    training: dict[int, dict[int, float]] = {group: {} for group in DEFAULT_GROUPS}
    holdouts: dict[int, dict[int, float]] = {group: {} for group in DEFAULT_GROUPS}
    seen: set[tuple[int, int]] = set()
    unsupported_groups: set[int] = set()
    outside_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pipeline schedule rows must be objects")
        groups = _positive_int(row.get("groups"), name="groups")
        tiles = _positive_int(row.get("tiles"), name="tiles")
        key = (groups, tiles)
        if key in seen:
            raise ValueError("duplicate pipeline schedule group/tile row")
        seen.add(key)
        if "correct" not in row or not _truthy(row["correct"]):
            raise ValueError("pipeline schedule measurements must be correct")
        values = _measurement_values(row)
        measured = float(median(values))
        if groups not in DEFAULT_GROUPS:
            unsupported_groups.add(groups)
            outside_rows.append({"groups": groups, "tiles": tiles, "status": "unsupported-group"})
        elif tiles in TRAINING_TILES:
            training[groups][tiles] = measured
        elif tiles in VALIDATION_TILES:
            holdouts[groups][tiles] = measured
        else:
            outside_rows.append({"groups": groups, "tiles": tiles, "status": "outside-policy"})

    if not any(training[group] for group in DEFAULT_GROUPS):
        raise ValueError("no supported pipeline schedule training measurements")

    curves: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    for groups in DEFAULT_GROUPS:
        points = training[groups]
        checks = [_holdout_check(tiles, holdouts[groups].get(tiles), points) for tiles in VALIDATION_TILES]
        for check in checks:
            check_with_group = dict(check, groups=groups)
            all_checks.append(check_with_group)
            if check["status"] == "missing":
                missing.append({"groups": groups, "tiles": check["tiles"], "kind": "validation"})
            elif check["status"] == "outside":
                outside_rows.append({"groups": groups, "tiles": check["tiles"], "status": "outside-training-range"})
        missing_training = [tiles for tiles in TRAINING_TILES if tiles not in points]
        for tiles in missing_training:
            missing.append({"groups": groups, "tiles": tiles, "kind": "training"})
        training_rows = [{"tiles": tiles, "median_kernel_ms": points[tiles], "kernel_ms": points[tiles]}
                         for tiles in sorted(points)]
        validation = _validation_summary(checks)
        curves.append({
            "groups": groups,
            "training": training_rows,
            "training_points": training_rows,
            "segments": _segment_points(points),
            "measured_tile_range": [min(points), max(points)] if points else None,
            "missing_training_tiles": missing_training,
            "holdout": checks,
            "validation": validation,
        })

    validation = _validation_summary(all_checks)
    validation["groups"] = list(DEFAULT_GROUPS)
    validation["training_tiles"] = list(TRAINING_TILES)
    validation["validation_tiles"] = list(VALIDATION_TILES)
    return {
        "version": VERSION,
        "schema": SCHEMA,
        "hardware": _json_copy(raw.get("hardware", {})),
        "protocol": _json_copy(raw.get("protocol", {})),
        "policy": _json_copy(POLICY),
        "training_tiles": list(TRAINING_TILES),
        "validation_tiles": list(VALIDATION_TILES),
        "curves": curves,
        "validation": validation,
        "missing": missing,
        "outside": outside_rows,
        "unsupported_groups": sorted(unsupported_groups),
        "source": "measured BatchPipeline event/submission scheduling floor",
        "limitations": [
            "One-CTA stores isolate a minimum event/submission scheduling floor.",
            "Validation tiles are independent holdouts and do not define interpolation segments.",
            "This is not a full operator concurrency calibration or service model.",
            "Queries outside measured training endpoints are never extrapolated.",
        ],
    }


def _summary_from_profile(profile: Any) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    for key in ("pipeline_scheduling", "pipeline_schedule", "pipeline_schedule_model"):
        value = profile.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(profile.get("summary"), dict) and isinstance(profile["summary"].get("curves"), list):
        return profile["summary"]
    if isinstance(profile.get("curves"), list):
        return profile
    return None


def _curve_for(summary: dict[str, Any], groups: int) -> dict[str, Any] | None:
    curves = summary.get("curves", [])
    if isinstance(curves, dict):
        curve = curves.get(str(groups), curves.get(groups))
        return curve if isinstance(curve, dict) else None
    if not isinstance(curves, list):
        return None
    for curve in curves:
        if not isinstance(curve, dict):
            continue
        try:
            if _positive_int(curve.get("groups"), name="curve.groups") == groups:
                return curve
        except ValueError:
            continue
    return None


def _curve_training_points(curve: dict[str, Any]) -> dict[int, float]:
    rows = curve.get("training", curve.get("training_points", []))
    if isinstance(rows, dict):
        rows = [{"tiles": key, "median_kernel_ms": value} for key, value in rows.items()]
    if not isinstance(rows, list):
        return {}
    points: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            tiles = _positive_int(row.get("tiles"), name="training.tiles")
            value = row.get("median_kernel_ms", row.get("kernel_ms", row.get("measured_ms")))
            value = _finite_number(value, name="training.kernel_ms", positive=True)
        except ValueError:
            continue
        if tiles in TRAINING_TILES:
            points[tiles] = value
    return points


def _validation_for_query(curve: dict[str, Any], tiles: int) -> dict[str, Any] | None:
    checks = curve.get("holdout", curve.get("validation_points", []))
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, dict):
            continue
        try:
            if _positive_int(check.get("tiles"), name="holdout.tiles") == tiles:
                return check
        except ValueError:
            continue
    return None


def _uncovered(groups: Any, tiles: Any, status: str, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "measured_minimum_ms": None,
        "covered": False,
        "validation": {"status": status, "query_status": status, "groups": groups, "tiles": tiles},
    }
    if reason is not None:
        result["validation"]["reason"] = reason
    return result


def costs(profile: dict[str, Any], groups: Any, tiles: Any) -> dict[str, Any]:
    """Return an in-range measured/interpolated minimum for one pipeline shape.

    Unknown groups, missing points and out-of-range tiles are normal coverage
    outcomes.  They return ``covered=False`` so a caller can retain its own
    conservative fallback without catching model exceptions.
    """
    try:
        group_count = _positive_int(groups, name="groups")
        tile_count = _positive_int(tiles, name="tiles")
    except ValueError:
        return _uncovered(groups, tiles, "invalid")
    if group_count not in DEFAULT_GROUPS:
        return _uncovered(group_count, tile_count, "missing", "unsupported-group")
    if tile_count < DEFAULT_TILES[0] or tile_count > DEFAULT_TILES[-1]:
        return _uncovered(group_count, tile_count, "outside", "outside-policy-range")

    summary = _summary_from_profile(profile)
    if summary is None:
        return _uncovered(group_count, tile_count, "missing", "no-pipeline-schedule-profile")
    curve = _curve_for(summary, group_count)
    if curve is None:
        return _uncovered(group_count, tile_count, "missing", "missing-group-curve")
    points = _curve_training_points(curve)
    estimate, state = _predict_training(points, tile_count)
    if estimate is None:
        return _uncovered(group_count, tile_count, state, "missing-training-endpoint")

    check = _validation_for_query(curve, tile_count)
    validation: dict[str, Any] = {
        "status": state,
        "query_status": state,
        "groups": group_count,
        "tiles": tile_count,
        "independent_holdout": tile_count in VALIDATION_TILES,
        "curve_status": curve.get("validation", {}).get("status") if isinstance(curve.get("validation"), dict) else None,
    }
    if check is not None:
        validation["holdout"] = check
        validation["holdout_relative_error"] = check.get("relative_error")
    return {"measured_minimum_ms": estimate, "covered": True, "validation": validation}


def _hardware_identity(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("hardware profile must be an object")
    nested = profile.get("hardware") if isinstance(profile.get("hardware"), dict) else {}
    source = dict(nested)
    if "device" not in source and "device_name" in source:
        source["device"] = source["device_name"]
    if "global_memory_bytes" not in source and "memory_bytes" in source:
        source["global_memory_bytes"] = source["memory_bytes"]
    source.update({key: profile[key] for key in ("device", "compute_capability", "global_memory_bytes", "sm_count")
                   if key in profile})
    required = ("device", "compute_capability", "global_memory_bytes")
    missing = [key for key in required if source.get(key) in (None, "")]
    if missing:
        raise ValueError("hardware profile is missing " + ", ".join(missing))
    try:
        memory = _positive_int(source["global_memory_bytes"], name="global_memory_bytes")
    except ValueError as error:
        raise ValueError("global_memory_bytes must be a positive integer") from error
    result: dict[str, Any] = {
        "device": str(source["device"]),
        "compute_capability": str(source["compute_capability"]),
        "global_memory_bytes": memory,
    }
    if source.get("sm_count") not in (None, ""):
        result["sm_count"] = _positive_int(source["sm_count"], name="sm_count")
    return result


def _identity(binary: pathlib.Path, profile: dict[str, Any]) -> dict[str, Any]:
    hardware = _hardware_identity(profile)
    identity: dict[str, Any] = {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        **hardware,
        "hardware": hardware,
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "version": VERSION,
        "policy": _json_copy(POLICY),
        "protocol": _json_copy(DEFAULT_PROTOCOL),
    }
    return identity


def _same_scalar(left: Any, right: Any) -> bool:
    if isinstance(right, int) and not isinstance(right, bool):
        try:
            return _positive_int(left, name="identity") == right
        except ValueError:
            return False
    return str(left) == str(right)


def _validate_raw_identity(raw: dict[str, Any], identity: dict[str, Any]) -> None:
    hardware = raw.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("pipeline schedule probe hardware identity is missing")
    for field in ("device", "compute_capability", "global_memory_bytes"):
        if field not in hardware or not _same_scalar(hardware[field], identity[field]):
            raise ValueError(f"pipeline schedule probe identity mismatch: {field}")
    if "sm_count" in identity and ("sm_count" not in hardware or
                                    not _same_scalar(hardware["sm_count"], identity["sm_count"])):
        raise ValueError("pipeline schedule probe identity mismatch: sm_count")

    protocol = raw.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("pipeline schedule probe protocol is missing")
    for field, expected in DEFAULT_PROTOCOL.items():
        if field not in protocol or not _same_scalar(protocol[field], expected):
            raise ValueError(f"pipeline schedule probe protocol mismatch: {field}")
    accepted_launch_modes = {"BatchPipeline::enqueue", "batch-pipeline", "batch_pipeline", "BatchPipeline"}
    for mode_field in ("launch_mode", "mode"):
        if mode_field in protocol and str(protocol[mode_field]) not in accepted_launch_modes:
            raise ValueError(f"pipeline schedule probe policy mismatch: {mode_field}")
    if "tile_batch" in protocol and not _same_scalar(protocol["tile_batch"], 1):
        raise ValueError("pipeline schedule probe policy mismatch: tile_batch")
    for field, expected in (("groups", DEFAULT_GROUPS), ("tiles", DEFAULT_TILES)):
        if field not in protocol:
            continue
        try:
            actual = tuple(_positive_int(value, name=f"protocol.{field}") for value in protocol[field])
        except (TypeError, ValueError):
            raise ValueError(f"pipeline schedule probe policy mismatch: {field}")
        if actual != tuple(expected):
            raise ValueError(f"pipeline schedule probe policy mismatch: {field}")


def _read_stdout(result: Any) -> str:
    value = getattr(result, "stdout", result)
    if isinstance(value, bytes):
        return value.decode()
    if not isinstance(value, str):
        raise ValueError("pipeline schedule probe runner returned no JSON stdout")
    return value


def calibrate(binary: pathlib.Path | str, path: pathlib.Path | str, profile: dict[str, Any],
              run: Callable[..., Any], exclusive: Callable[[], Any]) -> dict[str, Any]:
    """Run or reuse the fixed-protocol pipeline schedule calibration.

    Cache reuse requires exact binary, hardware, policy and protocol identity.
    The old artifact is only replaced after the complete probe output has been
    validated and summarized.
    """
    binary_path = pathlib.Path(binary)
    cache_path = pathlib.Path(path)
    identity = _identity(binary_path, profile)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, ValueError, TypeError):
            cached = None
        if isinstance(cached, dict) and cached.get("identity") == identity and isinstance(cached.get("summary"), dict):
            # Refit from cached raw rows so a model-only validation policy
            # change never leaves an old summary silently in use.
            cached_raw = cached.get("raw")
            if isinstance(cached_raw, dict):
                try:
                    _validate_raw_identity(cached_raw, identity)
                    refreshed = summarize(cached_raw)
                except (TypeError, ValueError):
                    refreshed = None
                if refreshed is not None:
                    if refreshed != cached["summary"]:
                        refreshed_artifact = dict(cached, summary=refreshed)
                        temporary = cache_path.with_name(cache_path.name + ".tmp")
                        try:
                            temporary.write_text(json.dumps(refreshed_artifact, indent=2, allow_nan=False) + "\n")
                            temporary.replace(cache_path)
                        except Exception:
                            try:
                                temporary.unlink()
                            except FileNotFoundError:
                                pass
                    return refreshed
            else:
                return cached["summary"]

    command = [str(binary_path), "--warmup", str(DEFAULT_PROTOCOL["warmup"]),
               "--repeat", str(DEFAULT_PROTOCOL["repeat"]), "--trials", str(DEFAULT_PROTOCOL["trials"])]
    exclusive()
    try:
        result = run(command)
    finally:
        # Release the measurement reservation even when the runner or parser
        # fails; no failed result is eligible to replace a previous artifact.
        exclusive()
    raw = json.loads(_read_stdout(result))
    _validate_raw_identity(raw, identity)
    summary = summarize(raw)
    artifact = {
        "schema": "cubutterfly-pipeline-schedule-calibration-v1",
        "identity": identity,
        "raw": raw,
        "summary": summary,
        "measurement_exclusive_gpu": True,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        temporary.replace(cache_path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return summary
