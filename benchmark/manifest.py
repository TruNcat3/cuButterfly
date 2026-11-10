"""Small, auditable manifest format for cuButterfly benchmark campaigns.

The module deliberately only expands a semantic matrix.  It does not start a
runner, select a GPU, or reinterpret an unsupported case.  Runner-specific
scripts can consume the emitted case manifest and keep their existing command
line contracts.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmark.compute_types import FFT_TYPES, fft_semantics


SCHEMA = "cubutterfly-benchmark-manifest-v1"
SCOPES = {"operator", "e2e", "profile"}
DIMENSIONS = (
    "sizes",
    "batches",
    "precisions",
    "directions",
    "layouts",
    "operators",
    "scenes",
    "compute_types",
)
_SINGULAR = {
    "sizes": "size",
    "batches": "batch",
    "precisions": "precision",
    "directions": "direction",
    "layouts": "layout",
    "operators": "operator",
    "scenes": "scene",
    "backends": "backend",
    "providers": "provider",
    "compute_types": "compute_type",
}
_CASE_KEYS = {
    "id", "scope", "modes", "sizes", "batches", "precisions", "directions",
    "layouts", "operators", "scenes", "backends", "providers", "backend",
    "provider", "gpu", "cpu", "execution", "metadata", "args", "runner",
    "size", "batch", "precision", "direction", "layout", "operator", "scene",
    "targets", "compute_types", "compute_type", "storage_precision", "compute_precision", "accumulation",
}
_ROOT_KEYS = {"schema", "id", "description", "axes", "modes", "cases", "output", "execution", "metadata"}


class BenchmarkManifestError(ValueError):
    """Raised for a malformed manifest or an empty selection."""


def _list(value: Any, *, field: str, allow_empty: bool = False) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    if not value and not allow_empty:
        raise BenchmarkManifestError(f"{field} must not be empty")
    return list(value)


def _selection(value: Any, *, field: str) -> list[Any] | None:
    if value is None:
        return None
    values = _list(value, field=field)
    if any(item is None or item == "" for item in values):
        raise BenchmarkManifestError(f"{field} contains an empty value")
    return values


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest without executing any benchmark."""
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise BenchmarkManifestError("manifest root must be an object")
    if document.get("schema") != SCHEMA:
        raise BenchmarkManifestError(f"schema must be {SCHEMA!r}")
    unknown = set(document) - _ROOT_KEYS
    if unknown:
        raise BenchmarkManifestError(f"unknown manifest field(s): {', '.join(sorted(unknown))}")
    if not isinstance(document.get("id"), str) or not document["id"]:
        raise BenchmarkManifestError("id must be a non-empty string")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkManifestError("cases must be a non-empty list")
    axes = document.get("axes", {})
    if not isinstance(axes, dict):
        raise BenchmarkManifestError("axes must be an object")
    unknown_axes = set(axes) - set(DIMENSIONS)
    if unknown_axes:
        raise BenchmarkManifestError(f"unknown axis field(s): {', '.join(sorted(unknown_axes))}")
    for field, values in axes.items():
        _list(values, field=f"axes.{field}")
    root_execution = document.get("execution", {})
    if not isinstance(root_execution, dict) or set(root_execution) - {"gpu", "cpu"}:
        raise BenchmarkManifestError("execution must contain only gpu/cpu")
    if any(not isinstance(value, bool) for value in root_execution.values()):
        raise BenchmarkManifestError("execution values must be boolean")
    modes = document.get("modes", {})
    if not isinstance(modes, dict) or not modes:
        raise BenchmarkManifestError("modes must be a non-empty object")
    if set(modes) - {"quick", "full"}:
        raise BenchmarkManifestError("modes only supports quick and full")
    for mode, spec in modes.items():
        if not isinstance(spec, dict):
            raise BenchmarkManifestError(f"modes.{mode} must be an object")
        if "case_ids" in spec:
            _list(spec["case_ids"], field=f"modes.{mode}.case_ids")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkManifestError(f"cases[{index}] must be an object")
        unknown_case = set(case) - _CASE_KEYS
        if unknown_case:
            raise BenchmarkManifestError(f"cases[{index}] unknown field(s): {', '.join(sorted(unknown_case))}")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise BenchmarkManifestError(f"cases[{index}].id must be a non-empty string")
        if case_id in ids:
            raise BenchmarkManifestError(f"case id must be unique: {case_id}")
        ids.add(case_id)
        if case.get("scope") not in SCOPES:
            raise BenchmarkManifestError(f"cases[{index}].scope must be one of {sorted(SCOPES)}")
        if "targets" in case and ("backends" in case or "providers" in case or "backend" in case or "provider" in case):
            raise BenchmarkManifestError(f"cases[{index}] cannot mix targets with backend/provider fields")
        modes_for_case = case.get("modes", ["quick", "full"])
        if not isinstance(modes_for_case, list) or not modes_for_case or set(modes_for_case) - {"quick", "full"}:
            raise BenchmarkManifestError(f"cases[{index}].modes must contain quick/full")
        for field in DIMENSIONS + ("backends", "providers"):
            if field in case:
                _list(case[field], field=f"cases[{index}].{field}")
        for field in ("backend", "provider"):
            if field in case and not isinstance(case[field], str):
                raise BenchmarkManifestError(f"cases[{index}].{field} must be a string")
        if "targets" in case:
            targets = case["targets"]
            if not isinstance(targets, list) or not targets:
                raise BenchmarkManifestError(f"cases[{index}].targets must be a non-empty list")
            for target_index, target in enumerate(targets):
                if not isinstance(target, dict):
                    raise BenchmarkManifestError(f"cases[{index}].targets[{target_index}] must be an object")
                unknown_target = set(target) - {"backend", "provider", "execution"}
                if unknown_target:
                    raise BenchmarkManifestError(
                        f"cases[{index}].targets[{target_index}] unknown field(s): "
                        f"{', '.join(sorted(unknown_target))}")
                for field in ("backend", "provider"):
                    if not isinstance(target.get(field), str) or not target[field]:
                        raise BenchmarkManifestError(
                            f"cases[{index}].targets[{target_index}].{field} must be a non-empty string")
                target_execution = target.get("execution", {})
                if not isinstance(target_execution, dict) or set(target_execution) - {"gpu", "cpu"}:
                    raise BenchmarkManifestError(
                        f"cases[{index}].targets[{target_index}].execution must contain only gpu/cpu")
                if any(not isinstance(value, bool) for value in target_execution.values()):
                    raise BenchmarkManifestError(
                        f"cases[{index}].targets[{target_index}].execution values must be boolean")
            target_ids = [(target["backend"], target["provider"]) for target in targets]
            if len(target_ids) != len(set(target_ids)):
                raise BenchmarkManifestError(f"cases[{index}].targets must not contain duplicate backend/provider pairs")
        execution = case.get("execution", {})
        if not isinstance(execution, dict) or set(execution) - {"gpu", "cpu"}:
            raise BenchmarkManifestError(f"cases[{index}].execution must contain only gpu/cpu")
        for key in execution:
            if not isinstance(execution[key], bool):
                raise BenchmarkManifestError(f"cases[{index}].execution.{key} must be boolean")
    for mode, spec in modes.items():
        ids_for_mode = set(spec.get("case_ids", ids))
        unknown_ids = ids_for_mode - ids
        if unknown_ids:
            raise BenchmarkManifestError(f"modes.{mode}.case_ids references unknown case(s): {sorted(unknown_ids)}")
    output = document.get("output", {})
    if not isinstance(output, dict) or set(output) - {"root", "format"}:
        raise BenchmarkManifestError("output supports root and format")
    if "root" in output and (not isinstance(output["root"], str) or not output["root"]):
        raise BenchmarkManifestError("output.root must be a non-empty string")
    return document


def _values(case: Mapping[str, Any], axes: Mapping[str, Any], dimension: str) -> list[Any]:
    plural = case.get(dimension)
    if plural is not None:
        return _list(plural, field=f"case.{dimension}")
    singular = _SINGULAR.get(dimension, dimension[:-1])
    if singular in case:
        return [case[singular]]
    if dimension in axes:
        return _list(axes[dimension], field=f"axes.{dimension}")
    return [None]


def _case_enabled(case: Mapping[str, Any], mode: str, mode_spec: Mapping[str, Any]) -> bool:
    if mode not in case.get("modes", ["quick", "full"]):
        return False
    selected_ids = mode_spec.get("case_ids")
    return selected_ids is None or case["id"] in selected_ids


def expand_cases(document: Mapping[str, Any], mode: str = "quick", filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expand one mode and semantic filters into an auditable case list."""
    if mode not in {"quick", "full"}:
        raise BenchmarkManifestError("mode must be quick or full")
    modes = document.get("modes") or {}
    mode_spec = modes.get(mode)
    if mode_spec is None:
        raise BenchmarkManifestError(f"mode {mode!r} is not defined")
    filters = dict(filters or {})
    unknown_filters = set(filters) - set(DIMENSIONS) - {"scope", "backend", "provider", "gpu", "cpu"}
    if unknown_filters:
        raise BenchmarkManifestError(f"unknown filter(s): {', '.join(sorted(unknown_filters))}")
    normalized: dict[str, list[Any]] = {}
    for key, value in filters.items():
        if key in {"gpu", "cpu"}:
            if not isinstance(value, bool):
                raise BenchmarkManifestError(f"filter {key} must be boolean")
            normalized[key] = [value]
        else:
            normalized[key] = _selection(value, field=f"filter.{key}") or []
    if "compute_types" in normalized and set(normalized["compute_types"]) - FFT_TYPES:
        raise BenchmarkManifestError("compute_types must contain fft-h/fft-l")
    expanded: list[dict[str, Any]] = []
    for case in document["cases"]:
        if not _case_enabled(case, mode, mode_spec):
            continue
        axes = document.get("axes", {})
        allowed_types = _values(case, axes, "compute_types")
        allowed_types = None if allowed_types == [None] else set(allowed_types)
        if allowed_types is not None and allowed_types - FFT_TYPES:
            raise BenchmarkManifestError(f"unknown compute_type(s): {sorted(allowed_types - FFT_TYPES)}")
        dimensions = {dimension: _values(case, axes, dimension)
                      for dimension in DIMENSIONS if dimension != "compute_types"}
        execution = dict(document.get("execution", {}))
        execution.update(case.get("execution", {}))
        if "gpu" in case:
            execution["gpu"] = case["gpu"]
        if "cpu" in case:
            execution["cpu"] = case["cpu"]
        execution.setdefault("gpu", True)
        execution.setdefault("cpu", False)
        if "targets" in case:
            targets = []
            for target in case["targets"]:
                target_execution = dict(execution)
                target_execution.update(target.get("execution", {}))
                targets.append({
                    "backend": target["backend"],
                    "provider": target["provider"],
                    "execution": target_execution,
                })
        else:
            backends = _values(case, {}, "backends") if "backends" in case else ([case["backend"]] if "backend" in case else [None])
            providers = _values(case, {}, "providers") if "providers" in case else ([case["provider"]] if "provider" in case else [None])
            targets = [
                {"backend": backend, "provider": provider, "execution": dict(execution)}
                for backend in backends for provider in providers
            ]
        dimensions["targets"] = targets
        fields = list(dimensions)
        for values in itertools.product(*(dimensions[field] for field in fields)):
            item = dict(zip(fields, values))
            target = item.pop("targets")
            item.update(case_id=case["id"], scope=case["scope"], mode=mode,
                        execution=target["execution"], backend=target["backend"], provider=target["provider"],
                        runner=case.get("runner"), args=list(case.get("args", [])), metadata=dict(case.get("metadata", {})))
            for dimension in dimensions:
                if dimension == "targets":
                    continue
                item[_SINGULAR[dimension]] = item.pop(dimension)
            item["compute_type"] = None
            requested_operator = item["operator"]
            requested_type = next(iter(allowed_types)) if allowed_types is not None and len(allowed_types) == 1 else None
            if requested_operator in FFT_TYPES:
                if allowed_types is not None and requested_operator not in allowed_types:
                    raise BenchmarkManifestError("FFT operator alias conflicts with compute_type")
                requested_type = requested_operator
                item["operator"] = "fft"
            if item["operator"] == "fft":
                try:
                    item.update(fft_semantics(item["precision"], **{
                        key: case.get(key) for key in ("storage_precision", "compute_precision", "accumulation")
                    }))
                except ValueError as exc:
                    raise BenchmarkManifestError(f"case {case['id']}: {exc}") from exc
                if requested_type is not None and requested_type != item["compute_type"]:
                    raise BenchmarkManifestError(
                        f"case {case['id']}: {requested_type} conflicts with precision/compute semantics ({item['compute_type']})")
                if allowed_types is not None and item["compute_type"] not in allowed_types:
                    continue
            elif allowed_types is not None:
                raise BenchmarkManifestError("compute_type only applies to FFT")
            def selected_value(key: str) -> Any:
                if key in {"gpu", "cpu"}:
                    return item["execution"][key]
                return item.get(_SINGULAR[key] if key in DIMENSIONS else key)

            def matches_filter(key: str) -> bool:
                if key == "operators":
                    return any(value == item["operator"] or value == item["compute_type"]
                               for value in normalized[key])
                return _matches(selected_value(key), normalized[key])

            if all(matches_filter(key) for key in normalized):
                expanded.append(item)
    if not expanded:
        raise BenchmarkManifestError("selection produced no cases")
    return expanded


def _matches(value: Any, accepted: Iterable[Any]) -> bool:
    return value in accepted


def write_case_manifest(document: Mapping[str, Any], cases: list[Mapping[str, Any]], output: str | Path) -> None:
    if not cases:
        raise BenchmarkManifestError("cannot write an empty case manifest")
    payload = {
        "schema": "cubutterfly-benchmark-case-manifest-v1",
        "source": document.get("id"),
        "mode": cases[0]["mode"] if cases else None,
        "case_count": len(cases),
        "output": document.get("output", {}),
        "cases": cases,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and expand a cuButterfly benchmark manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--operator", dest="operators", action="append")
    parser.add_argument("--compute-type", dest="compute_types", action="append", choices=sorted(FFT_TYPES))
    parser.add_argument("--scene", dest="scenes", action="append")
    parser.add_argument("--precision", dest="precisions", action="append")
    parser.add_argument("--direction", dest="directions", action="append")
    parser.add_argument("--layout", dest="layouts", action="append")
    parser.add_argument("--size", dest="sizes", action="append", type=int)
    parser.add_argument("--batch", dest="batches", action="append", type=int)
    parser.add_argument("--scope", choices=sorted(SCOPES))
    parser.add_argument("--backend")
    parser.add_argument("--provider")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="select cases whose resolved target requests GPU execution")
    parser.add_argument("--cpu", action="store_true", default=None,
                        help="select cases whose resolved target requests CPU execution")
    parser.add_argument("--output", type=Path, help="case manifest path (defaults below output.root)")
    args = parser.parse_args(argv)
    document = load_manifest(args.manifest)
    filters = {
        key: value for key, value in vars(args).items()
        if value is not None and (key in DIMENSIONS or key in {"scope", "backend", "provider", "gpu", "cpu"})
    }
    cases = expand_cases(document, args.mode, filters)
    output = args.output
    if output is None:
        root = Path(document.get("output", {}).get("root", "results/benchmark"))
        output = root / f"{document['id']}-{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_case_manifest(document, cases, output)
    print(f"wrote {len(cases)} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
