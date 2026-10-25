#!/usr/bin/env python3
"""Freeze the existing A100 comprehensive cases against the current build.

The historical manifest points at dated build directories.  This helper keeps
its case population and semantics, but rewrites the cuButterfly runners to a
fresh current-source build and records the binary/source identities.  It never
starts CUDA work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
try:
    from benchmark.runner_registry import normalize as normalize_runners
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    from benchmark.runner_registry import normalize as normalize_runners

DEFAULT_BASE = ROOT / "config" / "a100_comprehensive_suite.json"
DEFAULT_OUTPUT = ROOT / "results" / "comprehensive_current_20261006" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(base_path: Path, butterfly: Path, ntt: Path, fft_library: Path) -> dict[str, Any]:
    document = json.loads(base_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not document.get("cases"):
        raise ValueError("base manifest is not a valid comprehensive-suite manifest")
    for path in (butterfly, ntt, fft_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    current = dict(document)
    # The historical A100 suite predates the current calibration table.  Its
    # hand-written cuFFTDx points are useful provenance, but several are no
    # longer legal points in the current executable (for example the old
    # logN=8 block mapping).  Keep the semantic workload contract and let the
    # current selector choose a legal mapping.  The old arguments are retained
    # below so the rewrite is auditable and reproducible.
    rewrites = []
    for case in current["cases"]:
        if (case.get("runner") == "butterfly"
                and case.get("operator") == "fft"
                and str(case.get("implementation", "")).startswith("cuButterfly-")
                and case.get("precision") != "fp16-fp32"):
            old_args = list(case["args"])
            semantic_options = []
            for option in ("--operator", "--precision", "--placement", "--normalization"):
                if option in old_args:
                    index = old_args.index(option)
                    if index + 1 < len(old_args):
                        semantic_options.extend((option, old_args[index + 1]))
            if "--inverse" in old_args:
                semantic_options.append("--inverse")
            case["args"] = ["--auto-select", *semantic_options]
            case["mapping_policy"] = "auto-current-calibration"
            rewrites.append({"case": case["id"], "old_args": old_args,
                             "new_args": list(case["args"]),
                             "reason": "historical fixed point is not guaranteed in current calibration table"})
    current["current_rewrites"] = rewrites
    binaries = {
        "butterfly": str(butterfly),
        "ntt": str(ntt),
        "fft-library": str(fft_library),
    }
    runners = normalize_runners(document)
    for name, executable in binaries.items():
        runners.setdefault(name, {"kind": name})["executable"] = executable
    current["runners"] = runners
    current["binaries"] = {name: spec["executable"] for name, spec in runners.items()}
    current["source_freeze"] = {
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256(base_path),
        "source_root": str(ROOT),
        "library_sha256": sha256(butterfly.parent / "libcuntt.so"),
        "butterfly_sha256": sha256(butterfly),
        "ntt_sha256": sha256(ntt),
        "fft_library_sha256": sha256(fft_library),
        "compile_mode": "research",
        "target_arch": "sm80",
    }
    current["status"] = "prepared-current-source"
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--butterfly", type=Path,
                        default=ROOT / "build-e13-shared" / "cubutterfly_bench")
    parser.add_argument("--ntt", type=Path,
                        default=ROOT / "build-e13-shared" / "cuntt_bench")
    parser.add_argument("--fft-library", type=Path,
                        default=ROOT / "build-a100-vkfft" / "vkfft_bench")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = prepare(args.base.resolve(), args.butterfly.resolve(), args.ntt.resolve(),
                       args.fft_library.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "cases": len(document["cases"]),
                      "binaries": document["binaries"],
                      "library_sha256": document["source_freeze"]["library_sha256"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
