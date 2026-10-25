#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vendor_source="/tmp/cubutterfly-gpuntt-crypto-20260929"
vendor_archive="/tmp/cubutterfly-gpuntt-crypto-build-20260929/libgpuntt_crypto_vendor.a"
output="$repo_root/build-e13-cufftdx/libgpuntt_semantic_bridge.so"
nvcc="/usr/local/cuda-12.4/bin/nvcc"
expected_revision="d03c5eaeadaa780d153496afcb3b6a9b79a13a63"

if [[ -e "$output" || -L "$output" ]]; then
    printf '%s\n' "Output must be fresh: $output" >&2
    exit 1
fi
[[ "$(git -C "$vendor_source" rev-parse HEAD)" == "$expected_revision" ]]
[[ -z "$(git -C "$vendor_source" status --porcelain)" ]]
[[ -f "$vendor_archive" && -f "$repo_root/build-e13-cufftdx/libcuntt.so" ]]
build_dir="$(mktemp -d /tmp/cubutterfly-gpuntt-semantic-bridge.XXXXXX)"
command=("$nvcc" -std=c++17 -O3 -shared -Xcompiler=-fPIC
    --generate-code=arch=compute_80,code=sm_80
    --generate-code=arch=compute_80,code=compute_80
    -I "$vendor_source/src/include" -I "$repo_root/include"
    "$repo_root/benchmarks/gpuntt_crypto/semantic_bridge.cu"
    "$vendor_archive" -L "$repo_root/build-e13-cufftdx" -l:libcuntt.so
    -Xlinker -rpath -Xlinker "$repo_root/build-e13-cufftdx"
    -o "$build_dir/libgpuntt_semantic_bridge.so")
printf '%q ' "${command[@]}" > "$build_dir/build-command.txt"
printf '\n' >> "$build_dir/build-command.txt"
sha256sum "$vendor_archive" "$repo_root/build-e13-cufftdx/libcuntt.so" \
    "$repo_root/benchmarks/gpuntt_crypto/semantic_bridge.cu" \
    "$vendor_source/src/lib/ntt_merge/ntt.cu" \
    "$vendor_source/src/include/gpuntt/ntt_merge/ntt.cuh" > "$build_dir/input-sha256.txt"
"${command[@]}"
sha256sum "$build_dir/libgpuntt_semantic_bridge.so" > "$build_dir/output-sha256.txt"
# Publish from the target filesystem: /tmp may be mounted separately.
# Linking is atomic and refuses an existing target.
publish_file="$(mktemp "$repo_root/build-e13-cufftdx/.gpuntt-semantic-bridge.XXXXXX")"
trap 'rm -f "$publish_file"' EXIT
cp "$build_dir/libgpuntt_semantic_bridge.so" "$publish_file"
ln "$publish_file" "$output"
printf '%s\n' "Built fresh bridge: $output" "Reproducibility record: $build_dir"
