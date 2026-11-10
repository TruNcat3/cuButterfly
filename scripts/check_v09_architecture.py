#!/usr/bin/env python3
"""Validate the v0.9 layer manifest and its repository-owned paths."""

import argparse
import json
import pathlib


REQUIRED_LAYERS = {
    "templates",
    "platform_profiles",
    "parameters",
    "search_and_calibration",
    "runtime_selection",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    if document.get("schema_version") != 1 or document.get("release") != "0.9.0":
        raise SystemExit("manifest must use schema_version=1 and release=0.9.0")
    layers = document.get("layers", {})
    missing = REQUIRED_LAYERS - set(layers)
    if missing:
        raise SystemExit(f"manifest missing layers: {sorted(missing)}")
    for name, layer in layers.items():
        paths = layer.get("paths", [])
        if not paths:
            raise SystemExit(f"layer {name} has no paths")
        for relative in paths:
            if not (args.root / relative).exists():
                raise SystemExit(f"layer {name} references missing path: {relative}")
    profiles = document.get("compatibility_profiles", {})
    if "v100-sm70" not in profiles or "target-local" not in profiles:
        raise SystemExit("manifest must declare v100-sm70 and target-local profiles")


if __name__ == "__main__":
    main()
