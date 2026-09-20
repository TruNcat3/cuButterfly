#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build"}
PREFIX=${PREFIX:-"$HOME/.local"}
PROFILE_DIR=${PROFILE_DIR:-}
PROFILE_ROOT=${PROFILE_ROOT:-}
TRIALS=${TRIALS:-5}
OPERATOR_TRIALS=${OPERATOR_TRIALS:-3}
OPERATOR_WARMUP=${OPERATOR_WARMUP:-100}
OPERATOR_REPEAT=${OPERATOR_REPEAT:-100}
SEARCH_BUDGET=${SEARCH_BUDGET:-64}
SEARCH_FINALISTS=${SEARCH_FINALISTS:-3}
SEARCH_SECONDS=${SEARCH_SECONDS:-}
COMPILE_SECONDS=${COMPILE_SECONDS:-}
SEED_BUDGET=${SEED_BUDGET:-8}
COST_MODEL=${COST_MODEL:-staged}
STAGE_CALIBRATION=${STAGE_CALIBRATION:-full}
SEARCH_STRATEGY=${SEARCH_STRATEGY:-model}
SEARCH_WORKLOADS=${SEARCH_WORKLOADS:-"$ROOT/config/install_search_workloads.json"}
VERIFY_BATCHES=${VERIFY_BATCHES:-0}
MAPPING_SEEDS_ARGS=()
STAGE_IMPORT_ARGS=()
if [[ -n "${MAPPING_SEEDS:-}" ]]; then
    MAPPING_SEEDS_ARGS+=(--mapping-seeds "$MAPPING_SEEDS")
fi
RESUME_SEARCH=0
SKIP_TESTS=0
SKIP_OPERATOR_CALIBRATION=0
CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-}
ENABLE_CUFFTDX=${ENABLE_CUFFTDX:-}
FFT_CODEGEN_MODE=${FFT_CODEGEN_MODE:-}
CUDA_ARCHITECTURES_SET=0
FFT_CODEGEN_MODE_SET=0
[[ -n "$CUDA_ARCHITECTURES" ]] && CUDA_ARCHITECTURES_SET=1
[[ -n "$FFT_CODEGEN_MODE" ]] && FFT_CODEGEN_MODE_SET=1

usage() {
    cat <<'EOF'
Usage: scripts/install_with_hardware_profile.sh [options]

Configure (when needed), build/install cuButterfly, then calibrate the current
GPU. A new build detects the CUDA architecture from the first visible device.

Options:
  --build-dir DIR       CMake build directory (default: ./build)
  --prefix DIR          install prefix (default: ~/.local)
  --profile-dir DIR     hardware profile output directory
  --profile-root DIR    parent for an automatically named GPU profile
  --trials N            capability probe trials (default: 5)
  --operator-trials N   trials per operator smoke point (default: 3)
  --operator-warmup N   warmup launches per operator point (default: 100)
  --operator-repeat N   timed launches per operator point (default: 100)
  --search-budget N     screened mappings per workload (64; 0 = exhaustive)
  --search-finalists N  candidates confirmed per workload (default: 3)
  --cost-model M        staged (default) or legacy whole-kernel regression
  --stage-calibration M full (default), bounded, or skip independent stage sampling
  --stage-import-checkpoint FILE
                        reuse compatible stage measurements; may be repeated
  --search-strategy S   model (default) or evolutionary neighborhood search
  --search-seconds N    wall budget (full: 0/unrestricted; bounded: 300 seconds)
  --compile-seconds N   additional specialization compilation budget (600; 0 = research)
  --seed-budget N       historical mapping seeds to revalidate per workload (8; 0 = disabled)
  --mapping-seeds FILE  mapping seed file; may be repeated (mapping-only)
  --search-workloads FILE
                        workload cells to search; independent of kernel mappings
  --verify-batches N    verify this many transforms in each timed batch (0 = all)
  --resume-search      reuse matching search records for an unchanged binary/device
  --skip-tests          skip ctest after calibration and final rebuild
  --skip-operator-calibration
                        only install and generate the hardware capability profile
  --cuda-architectures LIST
                        CUDA architecture(s) when configuring a new build
  --enable-cufftdx      install MathDx and enable cuFFTDx when configuring
  --fft-codegen-mode M  precompiled (default) or on-demand research mode
  -h, --help            show this message
EOF
}

while (($#)); do
    case "$1" in
        --build-dir) BUILD_DIR=$2; shift 2 ;;
        --prefix) PREFIX=$2; shift 2 ;;
        --profile-dir) PROFILE_DIR=$2; shift 2 ;;
        --profile-root) PROFILE_ROOT=$2; shift 2 ;;
        --trials) TRIALS=$2; shift 2 ;;
        --operator-trials) OPERATOR_TRIALS=$2; shift 2 ;;
        --operator-warmup) OPERATOR_WARMUP=$2; shift 2 ;;
        --operator-repeat) OPERATOR_REPEAT=$2; shift 2 ;;
        --search-budget) SEARCH_BUDGET=$2; shift 2 ;;
        --search-finalists) SEARCH_FINALISTS=$2; shift 2 ;;
        --cost-model) COST_MODEL=$2; shift 2 ;;
        --stage-calibration) STAGE_CALIBRATION=$2; shift 2 ;;
        --stage-import-checkpoint) STAGE_IMPORT_ARGS+=(--stage-import-checkpoint "$2"); shift 2 ;;
        --search-strategy) SEARCH_STRATEGY=$2; shift 2 ;;
        --search-seconds) SEARCH_SECONDS=$2; shift 2 ;;
        --compile-seconds) COMPILE_SECONDS=$2; shift 2 ;;
        --seed-budget) SEED_BUDGET=$2; shift 2 ;;
        --mapping-seeds) MAPPING_SEEDS_ARGS+=(--mapping-seeds "$2"); shift 2 ;;
        --search-workloads) SEARCH_WORKLOADS=$2; shift 2 ;;
        --verify-batches) VERIFY_BATCHES=$2; shift 2 ;;
        --resume-search) RESUME_SEARCH=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-operator-calibration) SKIP_OPERATOR_CALIBRATION=1; shift ;;
        --cuda-architectures) CUDA_ARCHITECTURES=$2; CUDA_ARCHITECTURES_SET=1; shift 2 ;;
        --enable-cufftdx) ENABLE_CUFFTDX=ON; shift ;;
        --fft-codegen-mode) FFT_CODEGEN_MODE=$2; FFT_CODEGEN_MODE_SET=1; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$SEARCH_SECONDS" ]]; then
    if [[ "$STAGE_CALIBRATION" == full ]]; then SEARCH_SECONDS=0; else SEARCH_SECONDS=300; fi
fi
if [[ -z "$COMPILE_SECONDS" ]]; then
    if [[ "$STAGE_CALIBRATION" == full ]]; then COMPILE_SECONDS=0; else COMPILE_SECONDS=600; fi
fi

detect_architecture() {
    local query=(nvidia-smi)
    local visible=${CUDA_VISIBLE_DEVICES:-}
    if [[ -n "$visible" && "$visible" != "-1" && "$visible" != "NoDevFiles" ]]; then
        query+=(-i "${visible%%,*}")
    fi
    query+=(--query-gpu=compute_cap --format=csv,noheader,nounits)
    local capability
    capability=$("${query[@]}" | head -n 1 | tr -d '[:space:].')
    if [[ ! "$capability" =~ ^[0-9]+$ ]]; then
        echo "unable to detect CUDA architecture; pass --cuda-architectures <sm>" >&2
        exit 1
    fi
    printf '%s\n' "$capability"
}

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    [[ -n "$CUDA_ARCHITECTURES" ]] || CUDA_ARCHITECTURES=$(detect_architecture)
    [[ -n "$FFT_CODEGEN_MODE" ]] || FFT_CODEGEN_MODE=precompiled
    CMAKE_ARGS=(
        -S "$ROOT"
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
        -DPython3_EXECUTABLE="${PYTHON:-python3}"
        -DCUBUTTERFLY_FFT_CODEGEN_MODE="$FFT_CODEGEN_MODE"
    )
    if [[ -n "$ENABLE_CUFFTDX" ]]; then
        if [[ "$ENABLE_CUFFTDX" == ON ]]; then
            PYTHON="${PYTHON:-python3}" DESTINATION="$ROOT/external/mathdx" "$ROOT/scripts/install_cufftdx.sh"
        fi
        CMAKE_ARGS+=("-DCUBUTTERFLY_ENABLE_CUFFTDX=$ENABLE_CUFFTDX")
    fi
    cmake "${CMAKE_ARGS[@]}"
else
    # An existing build owns its cached architecture and codegen policy. Only
    # reconfigure when the caller explicitly overrides either one, or enables
    # cuFFTDx; an omitted option must not silently reset the cache.
    RECONFIGURE_ARGS=(-S "$ROOT" -B "$BUILD_DIR")
    if (( CUDA_ARCHITECTURES_SET )); then
        RECONFIGURE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
    fi
    if (( FFT_CODEGEN_MODE_SET )); then
        RECONFIGURE_ARGS+=("-DCUBUTTERFLY_FFT_CODEGEN_MODE=$FFT_CODEGEN_MODE")
    fi
    if [[ "$ENABLE_CUFFTDX" == ON ]]; then
        PYTHON="${PYTHON:-python3}" DESTINATION="$ROOT/external/mathdx" "$ROOT/scripts/install_cufftdx.sh"
        RECONFIGURE_ARGS+=("-DCUBUTTERFLY_ENABLE_CUFFTDX=ON")
    fi
    if (( ${#RECONFIGURE_ARGS[@]} > 4 )); then
        cmake "${RECONFIGURE_ARGS[@]}"
    fi
fi

cmake --build "$BUILD_DIR" -j"${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

if [[ -z "$PROFILE_DIR" ]]; then
    if [[ -z "$PROFILE_ROOT" ]]; then
        PROFILE_ROOT="$PREFIX/share/cuButterfly/hardware"
    fi
    # Keep directory identity resolution in the unified entry point's module;
    # directory mtimes are not a reliable selector when profiles coexist.
    PROFILE_DIR=$("${PYTHON:-python3}" - "$ROOT" "$BUILD_DIR" "$PROFILE_ROOT" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "scripts"))
import calibrate_hardware

identity = calibrate_hardware.detect_identity(pathlib.Path(sys.argv[2]).resolve(), required=True)
profile_root = pathlib.Path(sys.argv[3]).expanduser().resolve()
print(profile_root / calibrate_hardware._tag(identity))
PY
    )
fi

CALIBRATION_ARGS=(
    --mode execute
    --compile-seconds "$COMPILE_SECONDS"
    --search-seconds "$SEARCH_SECONDS"
    --search-budget "$SEARCH_BUDGET"
    --search-finalists "$SEARCH_FINALISTS"
    --seed-budget "$SEED_BUDGET"
    --cost-model "$COST_MODEL"
    --stage-calibration "$STAGE_CALIBRATION"
    --search-strategy "$SEARCH_STRATEGY"
    --search-workloads "$SEARCH_WORKLOADS"
    --build-dir "$BUILD_DIR"
    --trials "$TRIALS"
    --operator-trials "$OPERATOR_TRIALS"
    --operator-warmup "$OPERATOR_WARMUP"
    --operator-repeat "$OPERATOR_REPEAT"
    --verify-batches "$VERIFY_BATCHES"
)
if (( ${#MAPPING_SEEDS_ARGS[@]} )); then
    CALIBRATION_ARGS+=("${MAPPING_SEEDS_ARGS[@]}")
fi
if (( ${#STAGE_IMPORT_ARGS[@]} )); then
    CALIBRATION_ARGS+=("${STAGE_IMPORT_ARGS[@]}")
fi
if [[ -n "$PROFILE_DIR" ]]; then
    CALIBRATION_ARGS+=(--profile-dir "$PROFILE_DIR")
fi
if (( ! RESUME_SEARCH )); then
    CALIBRATION_ARGS+=(--force)
fi
if (( SKIP_OPERATOR_CALIBRATION )); then
    CALIBRATION_ARGS+=(--skip-operator-calibration)
fi
if (( RESUME_SEARCH )); then
    CALIBRATION_ARGS+=(--resume-search)
fi
"${PYTHON:-python3}" "$ROOT/scripts/calibrate_hardware.py" "${CALIBRATION_ARGS[@]}"
if (( ! SKIP_TESTS )); then
    # The legacy correctness tests intentionally exercise the V100 selector.
    # On another architecture, run the rest of the suite while the generated
    # local selector is validated by the package consumer and benchmarks.
    if grep -Eq '^CMAKE_CUDA_ARCHITECTURES:.*=([^;]*;)*70(;|$)' "$BUILD_DIR/CMakeCache.txt"; then
        ctest --test-dir "$BUILD_DIR" --output-on-failure
    else
        ctest --test-dir "$BUILD_DIR" --output-on-failure -E '^(cuntt_correctness|cubutterfly_correctness|fft_auto_dispatch_correctness|comprehensive_suite|v08_comprehensive_analysis|v08_selector_holdout_analysis)$'
    fi
fi
cmake --install "$BUILD_DIR" --prefix "$PREFIX"

echo "installed cuButterfly under $PREFIX"
echo "local calibration stored under $PROFILE_DIR"
echo "operator recommendations: $PROFILE_DIR/operator_recommendations.json"
