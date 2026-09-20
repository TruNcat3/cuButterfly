#!/usr/bin/env bash
set -euo pipefail

# Prepare a portable, target-local build for a research machine.  This entry
# point deliberately stops before GPU calibration: the experiment runner owns
# device selection, exclusivity, and timing on the target host.

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
REPO_DIR=${SCRIPT_ROOT}
BUILD_DIR=
PREFIX=
ENV_PREFIX=
CUDA_ARCH=70
JOBS=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
MATHDX_ROOT=
JSON_ROOT=
RUNTIME_SELECTOR_SUMMARY=
RUNTIME_SELECTOR_SOURCE=legacy-build-input
ARCHITECTURE_MANIFEST_SOURCE=
PORTABLE_ARCHITECTURE_MANIFEST=
NVCC=
PYTHON_BIN=${PYTHON:-}
WITH_VKFFT=0
CREATE_ENV=0
INSTALL=0
DRY_RUN=0
PYTHON_EXPLICIT=0
[[ -n "${PYTHON:-}" ]] && PYTHON_EXPLICIT=1

usage() {
    cat <<'EOF'
Usage: scripts/prepare_research_target.sh [options]

Prepare an isolated target build for V100 research.  The default target is
sm70.  The script configures and builds the top-level project, configures the
isolated crypto NTT benchmark, and runs `ctest -N` only; it does not calibrate,
run GPU tests, or install the package unless explicitly requested.

Options:
  --repo-dir DIR       repository root (default: directory containing script)
  --build-dir DIR      target-local build directory
                       (default: <repo>/build-v100-sm70)
  --prefix DIR         install prefix used only with --install
  --env-prefix DIR     conda prefix used with --create-env, or its Python
  --create-env         explicitly create --env-prefix with conda
  --install            run cmake --install into --prefix (or target-local
                       build/install when no prefix is supplied)
  --cuda-arch ARCH     explicit CUDA architecture (default: 70)
  --cuda-arch70        alias for --cuda-arch 70
  --jobs N             parallel build jobs (default: CMAKE_BUILD_PARALLEL_LEVEL or 2)
  --with-vkfft         enable the existing local VkFFT checkout; never downloads it
  --mathdx-root DIR    existing MathDx root containing include/cufftdx.hpp
  --json-root DIR      existing nlohmann_json source tree for offline CMake
                       configure (contains include/nlohmann/json.hpp)
  --nvcc PATH          CUDA compiler path
  --python PATH        Python interpreter (default: current CONDA_PREFIX/bin/python
                       when available, otherwise python3)
  --dry-run            print commands and checks without executing them
  -h, --help           show this message

MathDx is required for this research build and is never silently disabled.
If it is absent, install it separately with scripts/install_cufftdx.sh or pass
--mathdx-root.  V100 has no native BF16 arithmetic path; this script records
BF16 as emulated storage/rounding, never as native BF16 throughput. The
legacy runtime-selector CSV is a required build input and is never treated as
target calibration data.
EOF
}

die() {
    printf 'prepare_research_target: %s\n' "$*" >&2
    exit 2
}

need_arg() {
    (($# >= 2)) || die "missing argument for $1"
}

while (($#)); do
    case "$1" in
        --repo-dir)
            need_arg "$@"; REPO_DIR=$2; shift 2 ;;
        --build-dir)
            need_arg "$@"; BUILD_DIR=$2; shift 2 ;;
        --prefix)
            need_arg "$@"; PREFIX=$2; shift 2 ;;
        --env-prefix)
            need_arg "$@"; ENV_PREFIX=$2; shift 2 ;;
        --create-env)
            CREATE_ENV=1; shift ;;
        --install)
            INSTALL=1; shift ;;
        --cuda-arch|--cuda-architectures)
            need_arg "$@"; CUDA_ARCH=$2; shift 2 ;;
        --cuda-arch70)
            CUDA_ARCH=70; shift ;;
        --jobs)
            need_arg "$@"; JOBS=$2; shift 2 ;;
        --with-vkfft)
            WITH_VKFFT=1; shift ;;
        --mathdx-root)
            need_arg "$@"; MATHDX_ROOT=$2; shift 2 ;;
        --json-root)
            need_arg "$@"; JSON_ROOT=$2; shift 2 ;;
        --nvcc)
            need_arg "$@"; NVCC=$2; shift 2 ;;
        --python)
            need_arg "$@"; PYTHON_BIN=$2; PYTHON_EXPLICIT=1; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "unknown option: $1" ;;
    esac
done

[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] || die "invalid CUDA architecture '$CUDA_ARCH' (expected digits, e.g. 70)"
(( CUDA_ARCH >= 50 && CUDA_ARCH <= 100 )) || die "unsupported CUDA architecture '$CUDA_ARCH'"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "invalid jobs value '$JOBS'"

REPO_DIR=$(cd "$REPO_DIR" 2>/dev/null && pwd -P) || die "repository directory does not exist: $REPO_DIR"
if (( CREATE_ENV )); then
    [[ -n "$ENV_PREFIX" ]] || die "--create-env requires --env-prefix DIR"
    [[ "$ENV_PREFIX" != "/" ]] || die "refusing to use / as --env-prefix"
fi
[[ -n "$BUILD_DIR" ]] || BUILD_DIR="$REPO_DIR/build-v100-sm${CUDA_ARCH}"
if (( DRY_RUN )); then
    BUILD_DIR=$(realpath -m "$BUILD_DIR")
else
    BUILD_DIR=$(mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" && pwd -P)
fi
[[ -n "$MATHDX_ROOT" ]] || MATHDX_ROOT="$REPO_DIR/external/mathdx/nvidia/mathdx"
if (( DRY_RUN )); then
    MATHDX_ROOT=$(realpath -m "$MATHDX_ROOT")
else
    MATHDX_ROOT=$(cd "$MATHDX_ROOT" 2>/dev/null && pwd -P) || die "MathDx root does not exist: $MATHDX_ROOT; install it separately with $REPO_DIR/scripts/install_cufftdx.sh or pass --mathdx-root"
fi
if [[ -n "$JSON_ROOT" ]]; then
    if (( DRY_RUN )); then
        JSON_ROOT=$(realpath -m "$JSON_ROOT")
    else
        JSON_ROOT=$(cd "$JSON_ROOT" 2>/dev/null && pwd -P) || die "nlohmann_json source root does not exist: $JSON_ROOT"
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
if [[ -n "$ENV_PREFIX" && "$PYTHON_EXPLICIT" -eq 0 ]]; then
    if (( CREATE_ENV )); then
        PYTHON_BIN="$ENV_PREFIX/bin/python"
    elif [[ -x "$ENV_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$ENV_PREFIX/bin/python"
    fi
fi

if [[ -z "$NVCC" ]]; then
    NVCC=$(command -v nvcc 2>/dev/null || true)
fi
if [[ -z "$NVCC" ]]; then
    NVCC=nvcc
fi

VKFFT_ROOT="$REPO_DIR/external/VkFFT"
if (( WITH_VKFFT )); then
    if [[ ! -f "$VKFFT_ROOT/vkFFT/vkFFT.h" && ! -f "$VKFFT_ROOT/vkFFT.h" ]]; then
        if (( ! DRY_RUN )); then
            die "VkFFT headers not found under $VKFFT_ROOT; prepare a checkout separately or omit --with-vkfft"
        fi
        printf 'warning: VkFFT headers not found under %q (dry-run only)\n' "$VKFFT_ROOT" >&2
    fi
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cached_arch=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)
    if [[ -n "$cached_arch" && "$cached_arch" != "$CUDA_ARCH" ]]; then
        die "build directory $BUILD_DIR is configured for CUDA architecture '$cached_arch'; use a fresh --build-dir for sm$CUDA_ARCH"
    fi
fi

if (( CREATE_ENV )); then
    CONDA=${CONDA_EXE:-conda}
    if (( DRY_RUN )); then
        printf '+ %q create -p %q python=3.11 pip -y\n' "$CONDA" "$ENV_PREFIX"
    else
        command -v "$CONDA" >/dev/null 2>&1 || die "conda was not found; install it or omit --create-env"
        "$CONDA" create -p "$ENV_PREFIX" python=3.11 pip -y
        PYTHON_BIN="$ENV_PREFIX/bin/python"
    fi
fi

run() {
    if (( DRY_RUN )); then
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run_with_empty_cuda_visibility() {
    if (( DRY_RUN )); then
        printf '+ CUDA_VISIBLE_DEVICES= %q ' "$1"
        shift
        printf '%q ' "$@"
        printf '\n'
    else
        CUDA_VISIBLE_DEVICES= "$@"
    fi
}

if (( ! DRY_RUN )); then
    command -v cmake >/dev/null 2>&1 || die "cmake was not found"
    if [[ "$PYTHON_BIN" == */* ]]; then
        [[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
    else
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python was not found: $PYTHON_BIN"
    fi
    if [[ "$NVCC" == */* ]]; then
        [[ -x "$NVCC" ]] || die "nvcc is not executable: $NVCC"
    else
        command -v "$NVCC" >/dev/null 2>&1 || die "nvcc was not found: $NVCC"
    fi
    [[ -f "$MATHDX_ROOT/include/cufftdx.hpp" ]] || die "MathDx headers not found at $MATHDX_ROOT/include/cufftdx.hpp; install separately with $REPO_DIR/scripts/install_cufftdx.sh or pass --mathdx-root"
elif [[ ! -f "$MATHDX_ROOT/include/cufftdx.hpp" ]]; then
    printf 'warning: MathDx headers are absent at %q; install separately with %q or pass --mathdx-root\n' \
        "$MATHDX_ROOT/include/cufftdx.hpp" "$REPO_DIR/scripts/install_cufftdx.sh" >&2
fi
if [[ -n "$JSON_ROOT" ]]; then
    if (( ! DRY_RUN )); then
        [[ -f "$JSON_ROOT/include/nlohmann/json.hpp" ]] || die "nlohmann_json header not found at $JSON_ROOT/include/nlohmann/json.hpp"
    elif [[ ! -f "$JSON_ROOT/include/nlohmann/json.hpp" ]]; then
        printf 'warning: nlohmann_json header is absent at %q (dry-run only)\n' "$JSON_ROOT/include/nlohmann/json.hpp" >&2
    fi
fi

RUNTIME_SELECTOR_SUMMARY="$REPO_DIR/results/v100_scaling_full_summary.csv"
[[ -f "$RUNTIME_SELECTOR_SUMMARY" ]] || die "missing required legacy runtime-selector CSV: $RUNTIME_SELECTOR_SUMMARY; include the tracked file in the source package"

ARCHITECTURE_MANIFEST_SOURCE="$REPO_DIR/config/v09_architecture_manifest.json"
PORTABLE_ARCHITECTURE_MANIFEST="$BUILD_DIR/portable_architecture_manifest.json"
[[ -f "$ARCHITECTURE_MANIFEST_SOURCE" ]] || die "missing required architecture manifest: $ARCHITECTURE_MANIFEST_SOURCE"
if (( DRY_RUN )); then
    printf 'would write portable architecture manifest: %q\n' "$PORTABLE_ARCHITECTURE_MANIFEST"
else
    "$PYTHON_BIN" - "$ARCHITECTURE_MANIFEST_SOURCE" "$PORTABLE_ARCHITECTURE_MANIFEST" <<'PY'
import json
import pathlib
import sys

source_path, portable_path = map(pathlib.Path, sys.argv[1:])
document = json.loads(source_path.read_text())
generated = {}
for layer_name, layer in document.get("layers", {}).items():
    paths = list(layer.get("paths", []))
    source_paths = [path for path in paths if not path.startswith("results/")]
    generated_paths = [path for path in paths if path.startswith("results/")]
    layer["paths"] = source_paths
    if generated_paths:
        generated[layer_name] = generated_paths
document["generated_artifact_paths"] = generated
document["portable_source_manifest"] = {
    "generated_artifacts_are_informational": True,
    "source_manifest": str(source_path.resolve()),
}
portable_path.parent.mkdir(parents=True, exist_ok=True)
portable_path.write_text(json.dumps(document, indent=2) + "\n")
PY
fi

if [[ -n "$PREFIX" ]]; then
    if (( DRY_RUN )); then
        PREFIX=$(realpath -m "$PREFIX")
    elif (( INSTALL )); then
        PREFIX=$(mkdir -p "$PREFIX" && cd "$PREFIX" && pwd -P)
    else
        PREFIX=$(realpath -m "$PREFIX")
    fi
else
    PREFIX="$BUILD_DIR/install"
fi

CMAKE_ARGS=(
    cmake -S "$REPO_DIR" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    -DCMAKE_CUDA_COMPILER="$NVCC"
    -DPython3_EXECUTABLE="$PYTHON_BIN"
    -DCUBUTTERFLY_ENABLE_CUFFTDX=ON
    -DCUBUTTERFLY_MATHDX_ROOT="$MATHDX_ROOT"
    -DCUBUTTERFLY_RUNTIME_SELECTOR_SUMMARY="$RUNTIME_SELECTOR_SUMMARY"
    -DCUBUTTERFLY_ARCHITECTURE_MANIFEST="$PORTABLE_ARCHITECTURE_MANIFEST"
)
if [[ -n "$JSON_ROOT" ]]; then
    CMAKE_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="$JSON_ROOT")
fi
if (( WITH_VKFFT )); then
    CMAKE_ARGS+=(-DCUBUTTERFLY_ENABLE_VKFFT=ON -DCUBUTTERFLY_VKFFT_ROOT="$VKFFT_ROOT")
fi

printf 'target: V100-compatible sm%s\n' "$CUDA_ARCH"
printf 'repository: %s\n' "$REPO_DIR"
printf 'build: %s\n' "$BUILD_DIR"
printf 'python: %s\n' "$PYTHON_BIN"
printf 'nvcc: %s\n' "$NVCC"
printf 'MathDx: %s\n' "$MATHDX_ROOT"
printf 'runtime selector: %s (%s)\n' "$RUNTIME_SELECTOR_SUMMARY" "$RUNTIME_SELECTOR_SOURCE"
printf 'architecture manifest: %s (portable build copy)\n' "$PORTABLE_ARCHITECTURE_MANIFEST"
printf 'BF16: emulated storage/rounding only; SM70 has no native BF16 arithmetic path\n'
printf 'GPU execution: disabled during preparation (CUDA_VISIBLE_DEVICES=empty)\n'

run_with_empty_cuda_visibility "${CMAKE_ARGS[@]}"
run_with_empty_cuda_visibility cmake --build "$BUILD_DIR" --parallel "$JOBS"

CRYPTO_BUILD_DIR="$BUILD_DIR/crypto-ntt-sm${CUDA_ARCH}"
CRYPTO_ARGS=(
    cmake -S "$REPO_DIR/benchmarks/crypto_ntt" -B "$CRYPTO_BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    -DCMAKE_CUDA_COMPILER="$NVCC"
    -DCUBUTTERFLY_SOURCE_DIR="$REPO_DIR"
    -DCUBUTTERFLY_BUILD_DIR="$BUILD_DIR"
)
if [[ -n "$JSON_ROOT" ]]; then
    CRYPTO_ARGS+=(-DCUBUTTERFLY_NLOHMANN_INCLUDE="$JSON_ROOT/include")
fi
run_with_empty_cuda_visibility "${CRYPTO_ARGS[@]}"
run_with_empty_cuda_visibility cmake --build "$CRYPTO_BUILD_DIR" --target crypto_ntt_bench --parallel "$JOBS"

# Listing tests is intentionally the only CTest action in this preparation
# step.  GPU correctness and calibration belong to the target-host runner.
run_with_empty_cuda_visibility ctest --test-dir "$BUILD_DIR" -N

MANIFEST="$BUILD_DIR/research_target_manifest.json"
if (( DRY_RUN )); then
    printf 'would write: %s\n' "$MANIFEST"
else
    "$PYTHON_BIN" - "$MANIFEST" "$REPO_DIR" "$BUILD_DIR" "$CUDA_ARCH" "$MATHDX_ROOT" "$WITH_VKFFT" "$JSON_ROOT" "$RUNTIME_SELECTOR_SUMMARY" "$RUNTIME_SELECTOR_SOURCE" "$ARCHITECTURE_MANIFEST_SOURCE" "$PORTABLE_ARCHITECTURE_MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest_path, repo, build, arch, mathdx, with_vkfft, json_root, selector, selector_source, architecture_source, architecture_manifest = sys.argv[1:]
pathlib.Path(manifest_path).write_text(json.dumps({
    "schema": "cubutterfly-research-target-v1",
    "target": "V100-compatible",
    "cuda_architecture": int(arch),
    "repository": str(pathlib.Path(repo).resolve()),
    "build_directory": str(pathlib.Path(build).resolve()),
    "mathdx_root": str(pathlib.Path(mathdx).resolve()),
    "nlohmann_json_root": (str(pathlib.Path(json_root).resolve()) if json_root else None),
    "runtime_selector": {
        "path": str(pathlib.Path(selector).resolve()),
        "source": selector_source,
        "calibration_required": True,
        "note": "Legacy V100 selector input is retained only for build generation; target hardware calibration must replace local selections before performance measurements.",
    },
    "architecture_manifest": {
        "source": str(pathlib.Path(architecture_source).resolve()),
        "portable": str(pathlib.Path(architecture_manifest).resolve()),
        "generated_artifacts_informational": True,
    },
    "cufftdx_enabled": True,
    "vkfft_enabled": bool(int(with_vkfft)),
    "bf16": {
        "native_arithmetic": False,
        "measurement_label": "emulated_native",
        "note": "SM70 has no native BF16 arithmetic path; BF16 storage and per-operation rounding must not be reported as native BF16 throughput.",
    },
    "preparation": {
        "gpu_execution": False,
        "cuda_visible_devices": "",
        "ctest_mode": "list-only",
    },
}, indent=2, sort_keys=True) + "\n")
PY
fi

if (( INSTALL )); then
    run_with_empty_cuda_visibility cmake --install "$BUILD_DIR" --prefix "$PREFIX"
    printf 'installed cuButterfly under %s\n' "$PREFIX"
else
    printf 'installation skipped; pass --install explicitly (prefix: %s)\n' "$PREFIX"
fi

printf 'target preparation complete: %s\n' "$MANIFEST"
