#!/usr/bin/env python3
"""Reject reuse of a local hardware model on a different GPU."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a cuButterfly profile against the current GPU.")
    parser.add_argument("profile", type=pathlib.Path)
    parser.add_argument("--skip-device-check", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile.get("status") != "calibrated-local":
        raise SystemExit("profile is not a locally calibrated cuButterfly profile")
    if args.skip_device_check:
        print("profile status: calibrated-local (device check skipped)")
        return 0
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise SystemExit("nvidia-smi is required; use --skip-device-check only for inspection")
    query = [nvidia_smi]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        # CUDA remaps the first visible physical device to logical device 0.
        # Query the same physical index (or UUID) instead of always checking GPU 0.
        query.extend(["-i", visible.split(",", 1)[0].strip()])
    query.extend(["--query-gpu=name", "--format=csv,noheader,nounits"])
    result = subprocess.run(
        query,
        check=True,
        capture_output=True,
        text=True,
    )
    current = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    expected = str(profile.get("device", "")).strip()
    if current != expected:
        raise SystemExit(f"hardware profile mismatch: profile={expected!r}, current={current!r}")
    print(f"profile valid: {expected} / compute capability {profile.get('compute_capability')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
