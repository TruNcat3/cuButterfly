#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DESTINATION=${MATHDX_ROOT:-${DESTINATION:-"$ROOT/external/mathdx"}}
MATHDX_VERSION=${MATHDX_VERSION:-24.4.0}
PYTHON=${PYTHON:-python3}

"$PYTHON" -m pip install --upgrade --target "$DESTINATION" "nvidia-mathdx==$MATHDX_VERSION"
test -f "$DESTINATION/nvidia/mathdx/include/cufftdx.hpp"
printf 'MathDx %s installed at %s\n' "$MATHDX_VERSION" "$DESTINATION"
