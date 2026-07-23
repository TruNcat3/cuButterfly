#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
SOURCE_DIR=${SOURCE_DIR:-external/fast-hadamard-transform}
COMMIT=${COMMIT:-e7706faf8d1c3b9f241e36860640ad1dac644ede}
REPOSITORY=https://github.com/Dao-AILab/fast-hadamard-transform.git
PATCH=patches/dao_fast_hadamard_sm70.patch

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
"$PYTHON" -m pip install 'setuptools<70' 'numpy<2'
"$PYTHON" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone "$REPOSITORY" "$SOURCE_DIR"
fi

actual_remote=$(git -C "$SOURCE_DIR" remote get-url origin)
if [[ "$actual_remote" != "$REPOSITORY" ]]; then
    echo "unexpected external repository: $actual_remote" >&2
    exit 1
fi
git -C "$SOURCE_DIR" fetch origin "$COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$COMMIT"

if git -C "$SOURCE_DIR" apply --check "$(realpath "$PATCH")" 2>/dev/null; then
    git -C "$SOURCE_DIR" apply "$(realpath "$PATCH")"
elif ! rg -q 'arch=compute_70,code=sm_70' "$SOURCE_DIR/setup.py"; then
    echo "sm_70 patch cannot be applied cleanly" >&2
    exit 1
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8} \
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
MAX_JOBS=${MAX_JOBS:-4} \
    "$PYTHON" -m pip install --no-build-isolation --no-cache-dir -v "$SOURCE_DIR"

"$PYTHON" -c 'import torch, fast_hadamard_transform; print("installed on", torch.cuda.get_device_name(), "sm", torch.cuda.get_device_capability())'
