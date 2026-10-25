#!/usr/bin/env python3
"""Validate and print the unified external comparison catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASSES = {"direct-baseline", "processing-unit", "method-context", "end-to-end-context"}
STATUSES = {"measured", "measured-local-core", "source-only", "prepared"}


def validate(document: dict) -> list[dict]:
    if document.get("schema_version") != 1:
        raise ValueError("comparison catalog schema_version must be 1")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("comparison catalog entries must be non-empty")
    seen: set[str] = set()
    for entry in entries:
        required = {"id", "name", "comparison_class", "operators", "status", "source", "scope"}
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"entry {entry.get('id', '<unknown>')} missing {sorted(missing)}")
        entry_id = entry["id"]
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen:
            raise ValueError(f"duplicate or invalid comparison id: {entry_id!r}")
        seen.add(entry_id)
        if entry["comparison_class"] not in CLASSES:
            raise ValueError(f"entry {entry_id} has unknown comparison class")
        if entry["status"] not in STATUSES:
            raise ValueError(f"entry {entry_id} has unknown status")
        if not isinstance(entry["operators"], list) or not entry["operators"]:
            raise ValueError(f"entry {entry_id} must list at least one operator")
        if not isinstance(entry["source"], str) or not entry["source"].startswith("http"):
            raise ValueError(f"entry {entry_id} must have an external source URL")
        if entry["status"] in {"measured", "measured-local-core"} and not entry.get("runner"):
            raise ValueError(f"measured entry {entry_id} must name a runner")
        if entry["comparison_class"] == "direct-baseline" and entry["status"] == "source-only":
            raise ValueError(f"direct baseline {entry_id} cannot be source-only")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, nargs="?",
                        default=Path("config/external_comparison_catalog.json"))
    args = parser.parse_args()
    document = json.loads(args.catalog.read_text())
    entries = validate(document)
    for entry in entries:
        runner = entry.get("runner") or "—"
        print(f"{entry['id']:20} {entry['comparison_class']:20} {entry['status']:20} {runner}")


if __name__ == "__main__":
    main()
