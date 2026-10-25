#!/usr/bin/env python3
"""CPU-only expansion of an explicitly finite public Descriptor population."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import cubutterfly as api  # noqa: E402


CONFIG_SCHEMA = "cubutterfly-semantic-campaign-config-v1"
MANIFEST_SCHEMA = "cubutterfly-semantic-campaign-v1"
DEFAULT_MATRIX = [1.0, 0.25, -0.5, 1.0]
DIMENSIONS = ("shapes", "batches", "directions", "normalizations", "placements",
              "order_patterns", "layouts", "regimes")
CHOICES = {
    "directions": {"forward", "inverse"},
    "normalizations": {"none", "inverse"},
    "placements": {"out-of-place", "in-place"},
    "order_patterns": {"natural", "bit-reversed", "alternating", "reverse-alternating"},
    "layouts": {"packed", "independent-padded"},
}
TRANSFORM_FIELDS = {"op", "storage", "compute", "fft_type", "length_mode",
                    "fft_output_scale", "modulus", "word_bits", "stage_matrices"}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def semantic_id(semantics):
    return hashlib.sha256(canonical_bytes(semantics)).hexdigest()


def manifest_hash(manifest):
    return semantic_id({key: value for key, value in manifest.items()
                        if key != "manifest_sha256"})


def validate_manifest_hash(manifest):
    """Return whether the entire manifest matches its stored content digest."""
    return (isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA
            and manifest.get("manifest_sha256") == manifest_hash(manifest))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _fields(value, required, optional, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location}: object required")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{location}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")


def _integer(value, location, minimum=0, maximum=(1 << 64) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{location}: integer in [{minimum}, {maximum}] required")


def _name(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: nonempty string required")


def validate_config(config):
    """Reject schema/type errors before any library or GPU action."""
    _fields(config, {"schema", "populations"}, set(), "config")
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"config.schema must be {CONFIG_SCHEMA}")
    if not isinstance(config["populations"], list) or not config["populations"]:
        raise ValueError("config.populations: nonempty list required")
    population_names = set()
    for pi, population in enumerate(config["populations"]):
        loc = f"populations[{pi}]"
        _fields(population, {"name", *DIMENSIONS}, set(), loc)
        _name(population["name"], loc + ".name")
        if population["name"] in population_names:
            raise ValueError(f"{loc}: duplicate population name")
        population_names.add(population["name"])
        for dimension in DIMENSIONS:
            values = population[dimension]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{loc}.{dimension}: nonempty list required")
            if dimension in CHOICES:
                for value in values:
                    if not isinstance(value, str) or value not in CHOICES[dimension]:
                        raise ValueError(f"{loc}.{dimension}: invalid value {value!r}")
        for shape in population["shapes"]:
            if not isinstance(shape, list) or not shape:
                raise ValueError(f"{loc}.shapes: nonempty extent lists required")
            for extent in shape:
                _integer(extent, loc + ".shapes.extent", minimum=1)
        for batch in population["batches"]:
            _integer(batch, loc + ".batches", minimum=1)
        regime_names = set()
        for ri, regime in enumerate(population["regimes"]):
            rloc = f"{loc}.regimes[{ri}]"
            _fields(regime, {"name", "transform"}, set(), rloc)
            _name(regime["name"], rloc + ".name")
            if regime["name"] in regime_names:
                raise ValueError(f"{rloc}: duplicate regime name")
            regime_names.add(regime["name"])
            transform = regime["transform"]
            _fields(transform, {"op", "storage", "compute"}, TRANSFORM_FIELDS, rloc + ".transform")
            limits = {"op": 5, "storage": 9, "compute": 4, "fft_type": 2, "length_mode": 1,
                      "modulus": (1 << 64) - 1, "word_bits": (1 << 32) - 1}
            for field, maximum in limits.items():
                if field in transform:
                    _integer(transform[field], rloc + "." + field, maximum=maximum)
            if transform["op"] == int(api.Operator.NTT) and not {"modulus", "word_bits"} <= transform.keys():
                raise ValueError(f"{rloc}: NTT requires explicit modulus and word_bits")
            if "fft_output_scale" in transform:
                value = transform["fft_output_scale"]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{rloc}.fft_output_scale: finite number required")
            if transform["op"] == int(api.Operator.STRUCTURED_2X2) and "stage_matrices" not in transform:
                raise ValueError(f"{rloc}: structured operator requires explicit stage_matrices")
            if "stage_matrices" in transform:
                matrices = transform["stage_matrices"]
                if isinstance(matrices, str) and matrices == "repeat-default":
                    if transform["op"] != int(api.Operator.STRUCTURED_2X2):
                        raise ValueError(f"{rloc}: repeat-default requires structured operator")
                    continue
                if not isinstance(matrices, list):
                    raise ValueError(f"{rloc}.stage_matrices: per-axis matrix lists or repeat-default required")
                for axis in matrices:
                    if not isinstance(axis, list):
                        raise ValueError(f"{rloc}.stage_matrices: per-axis list required")
                    for matrix in axis:
                        if not isinstance(matrix, list) or len(matrix) != 4:
                            raise ValueError(f"{rloc}.stage_matrices: four coefficients per matrix required")
                        if any(type(x) not in (int, float) or not math.isfinite(x) for x in matrix):
                            raise ValueError(f"{rloc}.stage_matrices: finite coefficients required")


def side_extents(shape, inverse, transform):
    """Mirror the public side-shape rule; never change the requested shape."""
    input_shape, output_shape = list(shape), list(shape)
    if transform.get("length_mode", int(api.LengthMode.STANDARD)) == int(api.LengthMode.ZERO_EXTENDED_EMBEDDING):
        extended = [1 << (extent - 1).bit_length() for extent in shape]
        if inverse:
            input_shape = extended
        else:
            output_shape = extended
    if transform["op"] == int(api.Operator.FFT):
        fft_type = transform.get("fft_type", int(api.FftType.C2C))
        if fft_type == int(api.FftType.C2R):
            input_shape[-1] = input_shape[-1] // 2 + 1
        elif fft_type == int(api.FftType.R2C):
            output_shape[-1] = output_shape[-1] // 2 + 1
    return input_shape, output_shape


def layout_strides(extents, layout, output=False):
    strides = [0] * len(extents)
    span = 1
    padding = 3 if output else 2
    for axis in reversed(range(len(extents))):
        strides[axis] = span if layout == "packed" else span + padding
        span += (extents[axis] - 1) * strides[axis]
    return strides, span if layout == "packed" else span + (17 if output else 11)


def logical_orders(pattern, rank):
    if pattern == "natural":
        return [0] * rank
    if pattern == "bit-reversed":
        return [1] * rank
    if pattern == "reverse-alternating":
        return [1 - axis % 2 for axis in range(rank)]
    # Axis zero is natural, axis one bit-reversed, then repeat.
    return [axis % 2 for axis in range(rank)]


def _requests(config):
    for pi, population in enumerate(config["populations"]):
        axes = [population[key] for key in DIMENSIONS[:-1]]
        # Input and output pattern choices are independent dimensions.
        axes.insert(6, population["order_patterns"])
        axes.append(population["regimes"])
        for indices in itertools.product(*(range(len(axis)) for axis in axes)):
            shape, batch, direction, normalization, placement, ip, op, layout, regime = (
                axis[index] for axis, index in zip(axes, indices))
            yield {
                "population": population["name"], "regime": regime["name"],
                "shape": shape, "batch": batch, "direction": direction,
                "normalization": normalization, "placement": placement,
                "input_order_pattern": ip, "output_order_pattern": op, "layout": layout,
                "config_indices": [pi, *indices],
            }, regime["transform"]


def _transform(request, regime):
    fields = copy.deepcopy(regime)
    shape = request["shape"]
    if fields.get("stage_matrices") == "repeat-default":
        fields["stage_matrices"] = [[list(DEFAULT_MATRIX)] for _ in shape]
        request["matrix_source"] = {"kind": "repeat-default", "coefficients": list(DEFAULT_MATRIX),
                                    "matrices_per_axis": 1}
    inverse = request["direction"] == "inverse"
    input_shape, output_shape = side_extents(shape, inverse, fields)
    input_strides, input_batch_stride = layout_strides(input_shape, request["layout"])
    output_strides, output_batch_stride = layout_strides(output_shape, request["layout"], output=True)
    request["input_extents"], request["output_extents"] = input_shape, output_shape
    return api.Transform(**fields, extents=list(shape), batch=request["batch"], inverse=inverse,
                         normalize_inverse=request["normalization"] == "inverse",
                         in_place=request["placement"] == "in-place",
                         input_orders=logical_orders(request["input_order_pattern"], len(shape)),
                         output_orders=logical_orders(request["output_order_pattern"], len(shape)),
                         input_strides=input_strides, output_strides=output_strides,
                         input_batch_stride=input_batch_stride, output_batch_stride=output_batch_stride)


def generate_manifest(config, library_path, *, config_path=None, library=None):
    validate_config(config)
    library_path = Path(library_path).resolve(strict=True)
    library = api.Library(library_path) if library is None else library
    cases, invalid = {}, []
    metrics = {p["name"]: {"requested_variants": 0, "valid_variants": 0, "invalid_variants": 0,
                           "gap_variants": 0} for p in config["populations"]}
    for request, regime in _requests(config):
        counts = metrics[request["population"]]
        counts["requested_variants"] += 1
        try:
            transform = _transform(request, regime)
            with api.Descriptor(library, transform) as description:
                description.validate()
                envelope = json.loads(description.json())
                capabilities = json.loads(description.capabilities_json())
        except (api.Error, ValueError, TypeError, OverflowError) as error:
            invalid.append({"request": request, "reason": str(error),
                            "status_code": getattr(error, "status", None)})
            counts["invalid_variants"] += 1
            continue
        if capabilities.get("semantic_contract") != envelope["semantics"]:
            raise ValueError("public capability semantic contract differs from descriptor")
        if capabilities.get("status") not in {"eligible", "missing-lowering"}:
            raise ValueError(f"unrecognized capability status: {capabilities.get('status')!r}")
        counts["valid_variants"] += 1
        counts["gap_variants"] += capabilities["status"] == "missing-lowering"
        identity = semantic_id(envelope["semantics"])
        if identity not in cases:
            cases[identity] = {"id": identity, "populations": [], "descriptor": envelope,
                               "capabilities": capabilities, "requested_variants": []}
        case = cases[identity]
        if case["descriptor"] != envelope or case["capabilities"] != capabilities:
            raise ValueError("canonical identity collision or inconsistent public response")
        if request["population"] not in case["populations"]:
            case["populations"].append(request["population"])
        case["requested_variants"].append(request)
    for name, counts in metrics.items():
        counts["unique_cases"] = sum(name in case["populations"] for case in cases.values())
        counts["unique_gap_cases"] = sum(name in case["populations"] and
                                         case["capabilities"]["status"] == "missing-lowering"
                                         for case in cases.values())
    source_paths = [Path(__file__), ROOT / "python/cubutterfly/__init__.py",
                    ROOT / "src/semantic_contract.cpp", ROOT / "src/semantic_contract.hpp",
                    ROOT / "src/c_api.cpp", ROOT / "include/cubutterfly/cubutterfly.h"]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scope": "Finite provided sets only; no exhaustive public-contract or GPU execution claim.",
        "config": copy.deepcopy(config),
        "fingerprints": {"library": fingerprint(library_path),
                         "config": fingerprint(config_path) if config_path else {"sha256": semantic_id(config)},
                         "sources": [fingerprint(path) for path in source_paths]},
        "layout_recipe": {"packed": "row-major, span batch stride",
                          "independent-padded": "reverse axes: stride=span+2 input / span+3 output; "
                          "span+= (extent-1)*stride; batch stride=span+11 input / span+17 output",
                          "alternating_order": "axis index modulo two, starting natural",
                          "reverse_alternating_order": "one minus axis index modulo two"},
        "cases": list(cases.values()), "invalid_requests": invalid,
        "metrics": {"populations": metrics,
                    "requested_variants": sum(v["requested_variants"] for v in metrics.values()),
                    "valid_variants": sum(v["valid_variants"] for v in metrics.values()),
                    "gap_variants": sum(v["gap_variants"] for v in metrics.values()),
                    "invalid_variants": len(invalid), "unique_cases": len(cases),
                    "unique_gap_cases": sum(c["capabilities"]["status"] == "missing-lowering"
                                            for c in cases.values())},
    }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    return manifest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid number {value}")))
    validate_config(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be fresh; existing files cannot be replaced")
    try:
        config = load_config(args.config)
        manifest = generate_manifest(config, args.library, config_path=args.config)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, api.Error) as error:
        parser.error(str(error))
    print(json.dumps(manifest["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
