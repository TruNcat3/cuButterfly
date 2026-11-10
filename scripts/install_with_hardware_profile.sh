#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build"}
PREFIX=${PREFIX:-"$HOME/.local"}
PROFILE_DIR=${PROFILE_DIR:-}
PROFILE_ROOT=${PROFILE_ROOT:-${CUBUTTERFLY_PROFILE_ROOT:-}}
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
REFRESH_PROFILE=0
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

Configure (when needed), build/install cuButterfly, then reuse an exact matching
local or bundled static profile. Missing profiles are initialized. Profile
updates require --refresh-profile. A new build detects the CUDA architecture.

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
  --refresh-profile    explicitly measure and replace the current profile (--force)
  --resume-search      explicitly resume measurements for an unchanged binary/device;
                        bypass static reuse, mutually exclusive with --refresh-profile
  --skip-tests          skip ctest after calibration and final rebuild
  --skip-operator-calibration
                        when measuring, generate capabilities without operator calibration
  --cuda-architectures LIST
                        CUDA architecture(s) when configuring a new build
  --enable-cufftdx      install MathDx and enable cuFFTDx when configuring
  --fft-codegen-mode M  precompiled (default) or on-demand research mode
  -h, --help            show this message

Environment:
  CUBUTTERFLY_HARDWARE_MODEL  exact hardware artifact, ahead of local/bundled files
  CUBUTTERFLY_PROFILE_ROOT    fallback profile root when PROFILE_ROOT/CLI are absent
Explicit measurement modes isolate inherited static profile environment values.
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
        --refresh-profile) REFRESH_PROFILE=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-operator-calibration) SKIP_OPERATOR_CALIBRATION=1; shift ;;
        --cuda-architectures) CUDA_ARCHITECTURES=$2; CUDA_ARCHITECTURES_SET=1; shift 2 ;;
        --enable-cufftdx) ENABLE_CUFFTDX=ON; shift ;;
        --fft-codegen-mode) FFT_CODEGEN_MODE=$2; FFT_CODEGEN_MODE_SET=1; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if (( REFRESH_PROFILE && RESUME_SEARCH )); then
    echo "--refresh-profile and --resume-search are mutually exclusive" >&2
    exit 2
fi

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
        -DCMAKE_INSTALL_PREFIX="$PREFIX"
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
    CACHED_PREFIX=$(sed -n 's/^CMAKE_INSTALL_PREFIX:PATH=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)
    if [[ "$CACHED_PREFIX" != "$PREFIX" ]]; then
        RECONFIGURE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
    fi
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

[[ -n "$PROFILE_ROOT" ]] || PROFILE_ROOT="$PREFIX/share/cuButterfly/hardware"
PROFILE_DECISION=$("${PYTHON:-python3}" - "$ROOT" "$BUILD_DIR" "$PROFILE_ROOT" "$PROFILE_DIR" "$REFRESH_PROFILE" "$RESUME_SEARCH" <<'PY'
import json
import os
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "scripts"))
import calibrate_hardware
import static_hardware_profile

identity = calibrate_hardware.detect_identity(pathlib.Path(sys.argv[2]).resolve(), required=True)
profile_root = pathlib.Path(sys.argv[3]).expanduser().resolve()
directory = pathlib.Path(sys.argv[4]).expanduser().resolve() if sys.argv[4] else profile_root / calibrate_hardware._tag(identity)
refresh, resume = sys.argv[5] == "1", sys.argv[6] == "1"
explicit = os.environ.get("CUBUTTERFLY_HARDWARE_MODEL") or None
selection = static_hardware_profile.select(identity, explicit=explicit, local_dir=directory, bundled_root=root / "config/hardware")
if selection.get("action") == "error" and not (refresh or resume):
    raise SystemExit(selection["error"] + "; profile refresh requires --refresh-profile")
chosen = selection.get("selected")
action = "refresh" if refresh else "resume" if resume else "reuse" if chosen else "initialize"
runtime_hardware_model = None
if action == "initialize" and any((directory / name).exists() for name in ("hardware_model.json", "calibration_device.json")):
    raise SystemExit("existing profile is incompatible or unreadable; choose a matching --profile-dir or explicitly use --refresh-profile")
if action == "reuse":
    selected_file = pathlib.Path(chosen["path"]).resolve()
    if explicit:
        runtime_hardware_model = str(selected_file)
    source = selected_file.parent
    if source.resolve() != directory.resolve() or (explicit and selected_file != directory / "hardware_model.json"):
        # Publish only recognized profile artifacts, preserving existing files.
        names = ("hardware_model.json", "calibration_device.json", "hardware_profile.json",
                 "cost_model.json", "stage_service_profile.json", "operator_recommendations.json",
                 "mapping_registry.json", "pipeline_scheduling_profile.json")
        for name in ("hardware_model.json", "calibration_device.json"):
            if (directory / name).exists():
                matching = any(item.get("compatible") and pathlib.Path(item["path"]).resolve() == (directory / name).resolve()
                               for item in selection["sources"])
                if not matching and name == "calibration_device.json":
                    try:
                        static_hardware_profile.inspect_legacy(directory / name, identity)
                        matching = True
                    except (OSError, ValueError, TypeError, AttributeError):
                        pass
                if not matching:
                    raise SystemExit("selected profile cannot replace an existing incompatible local profile; use --refresh-profile explicitly")
        directory.mkdir(parents=True, exist_ok=True)
        if explicit and not (directory / "hardware_model.json").exists():
            shutil.copyfile(selected_file, directory / "hardware_model.json")
        for name in names:
            if explicit and name == "hardware_model.json":
                continue
            if (source / name).is_file() and not (directory / name).exists():
                shutil.copyfile(source / name, directory / name)
        selection["published"] = static_hardware_profile.select(identity, local_dir=directory)["selected"]
        if not explicit:
            chosen = selection["published"]
    print("reusing profile artifacts from " + str(source), file=sys.stderr)
    if runtime_hardware_model:
        print("runtime hardware artifact: " + runtime_hardware_model + "; existing local artifacts preserved", file=sys.stderr)
    if not chosen.get("hardware_complete"):
        print("hardware capabilities: partial; missing exact resource fields remain unknown", file=sys.stderr)
    else:
        print("hardware capabilities: complete static artifact", file=sys.stderr)
    for name, layer in chosen.get("layers", {}).items():
        status = "validated" if layer.get("validated") else "present, unvalidated" if layer.get("present") else "missing"
        print(name + ": " + status + "; status=" + str(layer.get("status", layer.get("calibration_status", "unknown"))), file=sys.stderr)
    print("profile updates are manual: rerun this installer with --refresh-profile (or --resume-search to resume measurements)", file=sys.stderr)
directory.mkdir(parents=True, exist_ok=True)
decision = dict(profile_dir=str(directory), action=action, identity=identity, selection=selection,
                runtime_hardware_model=runtime_hardware_model)
(directory / "installation_profile_selection.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision))
PY
)
PROFILE_DIR=$("${PYTHON:-python3}" -c 'import json,sys; print(json.loads(sys.argv[1])["profile_dir"])' "$PROFILE_DECISION")
PROFILE_ACTION=$("${PYTHON:-python3}" -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$PROFILE_DECISION")
RUNTIME_HARDWARE_MODEL=$("${PYTHON:-python3}" -c 'import json,sys; print(json.loads(sys.argv[1]).get("runtime_hardware_model") or "")' "$PROFILE_DECISION")

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
if (( REFRESH_PROFILE )); then
    CALIBRATION_ARGS+=(--force)
fi
if (( SKIP_OPERATOR_CALIBRATION )); then
    CALIBRATION_ARGS+=(--skip-operator-calibration)
fi
if (( RESUME_SEARCH )); then
    CALIBRATION_ARGS+=(--resume-search)
fi
if [[ "$PROFILE_ACTION" != reuse ]]; then
    echo "profile measurements use live hardware; inherited static profile environment is isolated"
    unset CUBUTTERFLY_HARDWARE_MODEL CUBUTTERFLY_PROFILE_ROOT
    "${PYTHON:-python3}" "$ROOT/scripts/calibrate_hardware.py" "${CALIBRATION_ARGS[@]}"
fi
[[ -n "$RUNTIME_HARDWARE_MODEL" ]] || RUNTIME_HARDWARE_MODEL="$PROFILE_DIR/hardware_model.json"
if [[ -f "$RUNTIME_HARDWARE_MODEL" ]]; then
    export CUBUTTERFLY_HARDWARE_MODEL="$RUNTIME_HARDWARE_MODEL"
    export CUBUTTERFLY_PROFILE_ROOT
    CUBUTTERFLY_PROFILE_ROOT=$(dirname "$PROFILE_DIR")
    {
        printf 'export CUBUTTERFLY_HARDWARE_MODEL=%q\n' "$CUBUTTERFLY_HARDWARE_MODEL"
        printf 'export CUBUTTERFLY_PROFILE_ROOT=%q\n' "$CUBUTTERFLY_PROFILE_ROOT"
    } > "$PROFILE_DIR/profile_environment.sh"
fi
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

if [[ -d "$PREFIX/conda-meta" ]]; then
    install -d "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
    install -m 644 "$ROOT/scripts/conda_activate_profile.sh" \
        "$PREFIX/etc/conda/activate.d/cubutterfly_profile.sh"
    install -m 644 "$ROOT/scripts/conda_deactivate_profile.sh" \
        "$PREFIX/etc/conda/deactivate.d/cubutterfly_profile.sh"
fi

echo "installed cuButterfly under $PREFIX"
echo "profile artifacts: $PROFILE_DIR (action: $PROFILE_ACTION)"
if [[ -f "$PROFILE_DIR/profile_environment.sh" ]]; then
    printf 'runtime profile environment: source %q\n' "$PROFILE_DIR/profile_environment.sh"
fi
if [[ -f "$PROFILE_DIR/operator_recommendations.json" ]]; then
    echo "operator recommendations: $PROFILE_DIR/operator_recommendations.json"
else
    echo "operator recommendations: missing; profile reuse does not certify performance calibration"
fi
