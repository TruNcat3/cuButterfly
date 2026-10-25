#!/usr/bin/env python3
"""Declare the finite 0.9.2 retest populations without starting GPU work.

All values below are explicit research choices, not inferred support claims.
The manifest generator asks the public library to classify every combination.
"""
import argparse
import json
from pathlib import Path

from semantic_campaign_manifest import generate_manifest, validate_config


def regimes():
    result = []
    for op, label, offset in ((0, "fft-c2c", 4), (2, "fwht", 0), (5, "structured", 0)):
        for storage, compute, precision in ((0, 0, "fp16-native"), (0, 1, "fp16-fp32"),
                                            (1, 0, "bf16-native"), (1, 1, "bf16-fp32"),
                                            (2, 1, "fp32"), (3, 2, "fp64")):
            transform = dict(op=op, storage=storage + offset, compute=compute)
            if op == 5:
                transform["stage_matrices"] = "repeat-default"
            result.append(dict(name=f"{label}-{precision}", transform=transform))
    for storage, width, modulus in ((8, 32, 998244353), (9, 64, 576460756061519873)):
        result.append(dict(name=f"ntt-word{width}", transform=dict(
            op=1, storage=storage, compute=width // 32 + 2, modulus=modulus, word_bits=width)))
    for op, label in ((3, "subset-zeta"), (4, "superset-zeta")):
        for storage, compute, precision in ((8, 3, "uint32"), (9, 4, "uint64")):
            result.append(dict(name=f"{label}-{precision}", transform=dict(op=op, storage=storage, compute=compute)))
    for fft_type, label in ((1, "r2c"), (2, "c2r")):
        for storage, compute, precision in ((0, 0, "fp16"), (1, 0, "bf16"), (2, 1, "fp32"), (3, 2, "fp64")):
            result.append(dict(name=f"fft-{label}-{precision}", transform=dict(
                op=0, storage=storage, compute=compute, fft_type=fft_type)))
    result.append(dict(name="fft-fp32-compute-fp64", transform=dict(op=0, storage=6, compute=2)))
    return result


def population(name, shapes, batches, numeric, *, orders, layouts, placements, normalizations):
    return dict(name=name, shapes=shapes, batches=batches, regimes=numeric,
                directions=["forward", "inverse"], normalizations=normalizations,
                placements=placements, order_patterns=orders, layouts=layouts)


def configuration():
    numeric = regimes()
    # Each named population is a complete Cartesian product of its listed
    # values. Larger scaling points do not replace the smaller boundary cases.
    return {"schema": "cubutterfly-semantic-campaign-config-v1", "populations": [
        population("semantic-crossing", [[8], [4, 8], [2, 4, 8], [2, 2, 2, 4]], [1, 3], numeric,
                   orders=["natural", "bit-reversed", "alternating", "reverse-alternating"],
                   layouts=["packed", "independent-padded"],
                   placements=["out-of-place", "in-place"], normalizations=["none", "inverse"]),
        population("rank-boundary", [[2] * 8, [2] * 9], [1], numeric,
                   orders=["natural", "reverse-alternating"], layouts=["packed"],
                   placements=["out-of-place"], normalizations=["inverse"]),
        population("size-batch-scaling", [[256], [4096], [65536], [1048576], [16777216],
                                          [64, 64], [256, 256], [1024, 1024],
                                          [16, 16, 16], [64, 64, 64], [128, 128, 128]],
                   [1, 16, 256, 4096], numeric,
                   orders=["natural", "bit-reversed"], layouts=["packed"],
                   placements=["out-of-place", "in-place"], normalizations=["inverse"]),
        population("arbitrary-and-embedding", [[7], [3, 5], [3, 5, 7]], [1, 3],
                   [dict(name=regime["name"] + suffix,
                         transform=dict(regime["transform"], length_mode=mode))
                    for regime in numeric if regime["name"] in
                    ("fft-c2c-fp32", "fft-c2c-fp64", "fwht-fp32", "structured-fp32")
                    for mode, suffix in ((0, "-standard"), (1, "-embedding"))],
                   orders=["natural", "bit-reversed", "reverse-alternating"],
                   layouts=["packed", "independent-padded"],
                   placements=["out-of-place", "in-place"], normalizations=["inverse"]),
    ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path, manifest_path = args.output / "config.json", args.output / "manifest.json"
    if config_path.exists() or manifest_path.exists():
        parser.error("use a fresh configuration/manifest output")
    config = configuration()
    validate_config(config)
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    manifest = generate_manifest(config, args.library, config_path=config_path)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps(manifest["metrics"], indent=2))


if __name__ == "__main__":
    main()
