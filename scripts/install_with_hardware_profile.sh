#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build"}
PREFIX=${PREFIX:-"$HOME/.local"}
PROFILE_DIR=${PROFILE_DIR:-"$PREFIX/share/cuButterfly/hardware/local"}
TRIALS=${TRIALS:-5}
SKIP_TESTS=0

usage() {
    cat <<'EOF'
Usage: scripts/install_with_hardware_profile.sh [options]

Build/install a configured cuButterfly tree, then calibrate the current GPU.
The build directory must already be configured with the desired
CMAKE_CUDA_ARCHITECTURES.

Options:
  --build-dir DIR       CMake build directory (default: ./build)
  --prefix DIR          install prefix (default: ~/.local)
  --profile-dir DIR     hardware profile output directory
  --trials N            capability probe trials (default: 5)
  --skip-tests          skip ctest before installation
  -h, --help            show this message
EOF
}

while (($#)); do
    case "$1" in
        --build-dir) BUILD_DIR=$2; shift 2 ;;
        --prefix) PREFIX=$2; shift 2 ;;
        --profile-dir) PROFILE_DIR=$2; shift 2 ;;
        --trials) TRIALS=$2; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "build directory is not configured: $BUILD_DIR" >&2
    echo "run cmake -S . -B '$BUILD_DIR' -DCMAKE_CUDA_ARCHITECTURES=<sm> first" >&2
    exit 1
fi

cmake --build "$BUILD_DIR" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
if (( ! SKIP_TESTS )); then
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

"$ROOT/scripts/initialize_hardware_profile.py" \
    --microbench "$BUILD_DIR/cuntt_hardware_microbench" \
    --output-dir "$PROFILE_DIR" \
    --trials "$TRIALS" \
    --force

echo "installed cuButterfly under $PREFIX"
echo "hardware model stored under $PROFILE_DIR"
