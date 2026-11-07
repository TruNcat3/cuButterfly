#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
expected_revision="d03c5eaeadaa780d153496afcb3b6a9b79a13a63"
vendor_source="${GPUNTT_SOURCE_DIR:-$repo_root/external/GPU-NTT-$expected_revision}"
build_root="${GPUNTT_BUILD_DIR:-$repo_root/build-gpuntt-semantic}"
output="${GPUNTT_OUTPUT:-$build_root/libgpuntt_semantic_bridge_guarded_20261102.so}"
root_helper="${CUNTT_LIBRARY:-$repo_root/build-e13-cufftdx/libcuntt.so}"
nvcc="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
python="${PYTHON:-python3}"
vendor_archive="$build_root/vendor/libgpuntt_crypto_vendor.a"
manifest="$output.manifest.json"
# Preserve the historical SM80 default; target selection never changes arithmetic.
cuda_arch="${CUDA_ARCH-80}"
if [[ ! "$cuda_arch" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' "Invalid CUDA_ARCH '$cuda_arch': expected a positive numeric SM (for example 80 or 120)" >&2
    exit 2
fi

if [[ -e "$output" || -L "$output" || -e "$manifest" || -L "$manifest" ]]; then
    printf '%s\n' "Output must be fresh: $output" >&2
    exit 1
fi
[[ "$(git -C "$vendor_source" rev-parse HEAD)" == "$expected_revision" ]]
[[ -z "$(git -C "$vendor_source" status --porcelain)" ]]
[[ -f "$root_helper" ]]
mkdir -p "$build_root" "$(dirname "$output")"
vendor_source="$(realpath "$vendor_source")"
root_helper="$(realpath "$root_helper")"
build_root="$(realpath "$build_root")"
output="$(realpath -m "$output")"
manifest="$output.manifest.json"
vendor_archive="$build_root/vendor/libgpuntt_crypto_vendor.a"
cmake -S "$repo_root/benchmarks/gpuntt_crypto" -B "$build_root/vendor" \
    -DCMAKE_BUILD_TYPE=Release -DGPUNTT_SOURCE_DIR="$vendor_source" \
    -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DCUBUTTERFLY_BUILD_DIR="${CUBUTTERFLY_BUILD_DIR:-$repo_root/build-e13-cufftdx}" \
    -DCUNTT_LIBRARY="$root_helper"
cmake --build "$build_root/vendor" --target gpuntt_crypto_vendor --parallel "${BUILD_JOBS:-4}"
build_dir="$(mktemp -d "$build_root/bridge.XXXXXX")"
command=("$nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC
    "--generate-code=arch=compute_${cuda_arch},code=sm_${cuda_arch}"
    "--generate-code=arch=compute_${cuda_arch},code=compute_${cuda_arch}"
    -I "$vendor_source/src/include" -I "$repo_root/include"
    "$repo_root/benchmarks/gpuntt_crypto/semantic_bridge.cu"
    "$vendor_archive" "$root_helper"
    -Xlinker -rpath -Xlinker "$(dirname "$root_helper")"
    -o "$build_dir/libgpuntt_semantic_bridge.so")
printf '%q ' "${command[@]}" > "$build_dir/build-command.txt"
printf '\n' >> "$build_dir/build-command.txt"
sha256sum "$vendor_archive" "$root_helper" \
    "$repo_root/benchmarks/gpuntt_crypto/semantic_bridge.cu" \
    "$vendor_source/src/lib/ntt_merge/ntt.cu" \
    "$vendor_source/src/include/gpuntt/ntt_merge/ntt.cuh" > "$build_dir/input-sha256.txt"
"${command[@]}"
sha256sum "$build_dir/libgpuntt_semantic_bridge.so" > "$build_dir/output-sha256.txt"
"$python" - "$repo_root" "$vendor_source" "$vendor_archive" "$root_helper" \
    "$build_dir" "$output" "$expected_revision" "$cuda_arch" <<'PY'
import json
from pathlib import Path
import runpy
import sys

repo, vendor, archive, helper, build, output, revision, cuda_arch = sys.argv[1:]
fingerprint = runpy.run_path(str(Path(repo) / "scripts/semantic_gpuntt_baseline.py"))["_fingerprint"]
binary = fingerprint(Path(build) / "libgpuntt_semantic_bridge.so")
binary["path"] = output
sources = [Path(repo) / name for name in (
    "benchmarks/gpuntt_crypto/semantic_bridge.cu", "benchmarks/gpuntt_crypto/build_semantic_bridge.sh",
    "benchmarks/gpuntt_crypto/CMakeLists.txt", "scripts/semantic_gpuntt_baseline.py", "src/reference.cpp",
    "include/cuntt/ntt.hpp")]
sources += sorted(path for path in (Path(vendor) / "src").rglob("*") if path.is_file())
record = dict(schema="cubutterfly-gpuntt-build-v1", binary=binary,
              vendor_archive=fingerprint(archive), root_helper_library=fingerprint(helper),
              sources=[fingerprint(path) for path in sources], vendor_revision=revision,
              architecture=f"sm{cuda_arch} + compute{cuda_arch} PTX", vendor_source=vendor,
              build_directory=build, build_command=(Path(build) / "build-command.txt").read_text())
with (Path(build) / "manifest.json").open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY
# Stage on the output filesystem, then publish complete files without replacing.
publish_binary="$(mktemp "$(dirname "$output")/.gpuntt-binary.XXXXXX")"
publish_manifest="$(mktemp "$(dirname "$output")/.gpuntt-manifest.XXXXXX")"
trap 'rm -f "$publish_binary" "$publish_manifest"' EXIT
cp "$build_dir/libgpuntt_semantic_bridge.so" "$publish_binary"
cp "$build_dir/manifest.json" "$publish_manifest"
ln "$publish_manifest" "$manifest"
ln "$publish_binary" "$output"
printf '%s\n' "Built fresh bridge: $output" "Reproducibility record: $build_dir"
