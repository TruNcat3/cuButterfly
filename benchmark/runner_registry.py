"""Shared executable registry for benchmark campaign runners.

The registry is deliberately small.  It resolves an executable and records
the runner kind; semantic case expansion and runner-specific command building
remain in the corresponding adapter.  Legacy ``binaries`` mappings are
accepted while campaigns migrate to the richer ``runners`` form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class RunnerRegistryError(ValueError):
    """Raised when a campaign runner registry is malformed."""


def normalize(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = document.get("runners")
    if raw is None:
        raw = document.get("binaries", {})
    if not isinstance(raw, Mapping):
        raise RunnerRegistryError("runner registry must be an object")
    result: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise RunnerRegistryError("runner names must be non-empty strings")
        if isinstance(value, str):
            spec = {"kind": name, "executable": value}
        elif isinstance(value, Mapping):
            spec = dict(value)
        else:
            raise RunnerRegistryError(f"runners.{name} must be a string or object")
        executable = spec.get("executable")
        if not isinstance(executable, str) or not executable:
            raise RunnerRegistryError(f"runners.{name}.executable must be a non-empty string")
        kind = spec.get("kind", name)
        if not isinstance(kind, str) or not kind:
            raise RunnerRegistryError(f"runners.{name}.kind must be a non-empty string")
        spec["kind"] = kind
        spec["executable"] = executable
        result[name] = spec
    return result


def referenced(document: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for case in document.get("cases", []):
        if isinstance(case, Mapping) and isinstance(case.get("runner"), str):
            names.add(case["runner"])
        if isinstance(case, Mapping):
            for baseline in case.get("baselines", []):
                if isinstance(baseline, str):
                    names.add(baseline)
    for workload in document.get("workloads", []):
        if not isinstance(workload, Mapping):
            continue
        for implementation in workload.get("implementations", []):
            if isinstance(implementation, Mapping) and isinstance(implementation.get("runner"), str):
                names.add(implementation["runner"])
    return names


def validate_references(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    registry = normalize(document)
    missing = sorted(referenced(document) - set(registry))
    if missing:
        raise RunnerRegistryError("runner(s) missing from registry: " + ", ".join(missing))
    return registry


def resolve(document: Mapping[str, Any], name: str, root: str | Path) -> dict[str, Any]:
    registry = normalize(document)
    try:
        spec = dict(registry[name])
    except KeyError as exc:
        raise RunnerRegistryError(f"runner {name!r} is not registered") from exc
    path = Path(spec["executable"])
    if not path.is_absolute():
        path = Path(root) / path
    spec["name"] = name
    spec["path"] = path.resolve()
    return spec
