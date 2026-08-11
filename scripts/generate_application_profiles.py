#!/usr/bin/env python3
import argparse
import json
import pathlib


OPERATORS = {"fft": 0, "ntt": 1, "fwht": 2}
STORAGE = {"fp32": 2, "complex-fp32": 6, "uint32": 8, "uint64": 9}
DIRECTION = {"forward": 0, "inverse": 1}
PLACEMENT = {"out-of-place": 0, "in-place": 1}


def generate(spec_path: pathlib.Path, header_path: pathlib.Path) -> None:
    spec = json.loads(spec_path.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("application profile schema_version must be 1")
    target = spec["target"]
    profiles = spec["profiles"]
    lines = [
        "// Generated from config/v100_application_profiles.json. Do not edit.",
        "#pragma once",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "namespace cuntt::detail {",
        "struct GeneratedApplicationProfile { int op; int storage; int direction; int placement; std::uint64_t modulus; std::uint32_t log_n; const std::size_t* batches; std::size_t batch_count; const char* implementation; };",
        f"inline constexpr int kApplicationProfileComputeMajor = {int(target['compute_major'])};",
        f"inline constexpr int kApplicationProfileComputeMinor = {int(target['compute_minor'])};",
        f"inline constexpr int kApplicationProfileMultiprocessors = {int(target['multiprocessors'])};",
        f"inline constexpr const char* kApplicationProfileTarget = \"{target['name']}\";",
    ]
    for index, profile in enumerate(profiles):
        batches = profile["batches"]
        if not batches or any(int(value) <= 0 for value in batches):
            raise ValueError(f"profile {index} has invalid batches")
        values = ", ".join(f"{int(value)}ULL" for value in batches)
        lines.append(f"inline constexpr std::array<std::size_t, {len(batches)}> kApplicationBatches{index} = {{{{{values}}}}};")
    lines.append(f"inline constexpr std::array<GeneratedApplicationProfile, {len(profiles)}> kApplicationProfiles = {{{{")
    for index, profile in enumerate(profiles):
        lines.append(
            "    GeneratedApplicationProfile{" +
            f"{OPERATORS[profile['operator']]}, {STORAGE[profile['storage']]}, " +
            f"{DIRECTION[profile['direction']]}, {PLACEMENT[profile['placement']]}, " +
            f"{int(profile.get('modulus', 0))}ULL, {int(profile['logN'])}, kApplicationBatches{index}.data(), kApplicationBatches{index}.size(), " +
            f"\"{profile['implementation']}\"" + "},"
        )
    lines.extend(["}};", "}  // namespace cuntt::detail", ""])
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate built-in application workload profiles.")
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--header", type=pathlib.Path, required=True)
    args = parser.parse_args()
    generate(args.spec, args.header)


if __name__ == "__main__":
    main()
