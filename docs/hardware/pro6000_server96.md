# RTX PRO 6000 Blackwell Server Edition 96GB

This runbook prepares a separate target installation and runs the frozen main
operator comparison: quick has 12 requests; full has 66. It does not certify
migration of the whole paper. No PRO6000 GPU results are claimed here.

NVIDIA lists the Server Edition at compute capability 12.0 (SM120).
[GPU list](https://developer.nvidia.com/cuda-gpus) and
[cuFFTDx release notes](https://docs.nvidia.com/cuda/cufftdx/release_notes.html)
document the target and cuFFTDx 1.5 / MathDx 25.06 support. Use CUDA 12.8 and
MathDx package 25.6.0 for this setup. The compiler and a small cuFFTDx compile
probe must accept the target; a package version alone is insufficient.

## Install in a separate target environment

Run from a transferred source checkout on local SSD or NAS. Keep the A100
checkout, CUDA toolkit, conda environment, builds, profiles and results intact.
Do not source an A100 `profile_environment.sh` or copy its calibration/cache.
Install CUDA 12.8 separately at `/usr/local/cuda-12.8`; do not repoint the A100
compiler or the system CUDA symlink. A compatible driver is required when the
GPU phase starts.

```bash
conda create -p "$HOME/envs/cubutterfly-pro6000" python=3.11 pip -y
conda activate "$HOME/envs/cubutterfly-pro6000"
python -m pip install cmake ninja packaging wheel 'setuptools<70' 'numpy<2'
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
MATHDX_VERSION=25.6.0 MATHDX_ROOT="$PWD/external/mathdx-25.6.0" \
  PYTHON="$CONDA_PREFIX/bin/python" bash scripts/install_cufftdx.sh
```

Use the same target prefix for Python and the library installation. The platform build requests a shared library, on-demand FFT
code generation, and installs the existing conda activation/deactivation hooks.
Reactivate the environment after installation to load those hooks. They set the
default profile lookup location; they do not publish later research calibration
from the results directory into that location.

## CPU build and comparison dependencies

These steps hide CUDA devices. They compile CUDA source but do not execute GPU
kernels. Download dependencies into this target checkout, not a live A100 tree.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=
export BUILD="$PWD/build-pro6000-server96-sm120"
export OUTPUT="$PWD/results/pro6000-server96"

bash scripts/install_vkfft.sh
python scripts/research_platform.py plan --prefix "$CONDA_PREFIX" --with-vkfft
python scripts/research_platform.py doctor --prefix "$CONDA_PREFIX"
python scripts/research_platform.py build --prefix "$CONDA_PREFIX" --with-vkfft --jobs 4
```

`doctor` checks interpreter/package metadata, the compiler target and Torch
compiled-in architecture flags without querying a GPU. Torch must contain
`sm_120`; installing the system toolkit does not add that target to an older
Torch wheel. The existing Torch 2.5.1 environment lacks this target and blocks
this check; use the separate target environment above. Baseline gaps are separate from core prerequisites. `build`
checks actual MathDx compilation support; an unsupported core fails explicitly.
`--json-root DIR` can supply a local nlohmann-json checkout for offline builds.
The main CMake build produces `BUILD/libvkfft_xd_bridge.so` when VkFFT is enabled.

Build GPU-NTT against the target shared library. Its clean, pinned official
source and fresh output are checked by the builder; reruns need a new output
filename if the previous binary/receipt already exists.

```bash
git clone https://github.com/Alisah-Ozcan/GPU-NTT.git \
  external/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63
git -C external/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63 \
  checkout --detach d03c5eaeadaa780d153496afcb3b6a9b79a13a63
CUDA_ARCH=120 NVCC="$CUDA_HOME/bin/nvcc" PYTHON="$CONDA_PREFIX/bin/python" \
  CUBUTTERFLY_BUILD_DIR="$BUILD" CUNTT_LIBRARY="$BUILD/libcuntt.so" \
  GPUNTT_BUILD_DIR="$PWD/build-gpuntt-pro6000-sm120" \
  GPUNTT_OUTPUT="$BUILD/libgpuntt_semantic_bridge.so" \
  bash benchmarks/gpuntt_crypto/build_semantic_bridge.sh
```

Build Dao FHT directly from its pinned source with devices hidden. That revision's
`setup.py` emits SM120 code for CUDA 12.8. The old
`install_external_fht.sh` ends with `torch.cuda` device queries and is not a
CPU-only installer. Its SM70 patch is unnecessary for this target.

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
  external/fast-hadamard-transform-pro6000
git -C external/fast-hadamard-transform-pro6000 \
  checkout --detach e7706faf8d1c3b9f241e36860640ad1dac644ede
CUDA_VISIBLE_DEVICES= CUDA_HOME="$CUDA_HOME" \
  FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install --no-build-isolation --no-cache-dir \
  ./external/fast-hadamard-transform-pro6000
```

Prepare both populations on CPU. This creates target-bound manifests, CPU
references and a finite set of precompiled proposals. Adaptive search can still
compile additional candidates outside timing. These artifacts are not GPU
correctness evidence.

```bash
python scripts/research_platform.py prepare --mode quick --prefix "$CONDA_PREFIX" --jobs 4
python scripts/research_platform.py prepare --mode full --prefix "$CONDA_PREFIX" --jobs 4
```

## GPU calibration, comparison and report

Select the physical GPU by UUID on the target host. `run` checks UUID, model,
SM120 and the configured 96GB memory range, then uses the existing guarded
refresh driver to calibrate before comparison. Exclusivity, correctness checks,
checkpoint compatibility and timing boundaries remain the runner's responsibility.
Quick/full use separate run directories and target-local calibration; full does
not promote quick timings into evidence for missing requests.

```bash
export GPU_UUID=GPU-REPLACE-WITH-TARGET-UUID
python scripts/research_platform.py run --mode quick --prefix "$CONDA_PREFIX" \
  --gpu "$GPU_UUID" --vkfft-library "$BUILD/libvkfft_xd_bridge.so" \
  --gpuntt-library "$BUILD/libgpuntt_semantic_bridge.so" --jobs 4
python scripts/research_platform.py run --mode full --prefix "$CONDA_PREFIX" \
  --gpu "$GPU_UUID" --vkfft-library "$BUILD/libvkfft_xd_bridge.so" \
  --gpuntt-library "$BUILD/libgpuntt_semantic_bridge.so" --jobs 4
python scripts/research_platform.py report --mode full --prefix "$CONDA_PREFIX"
```

Use the same build, output, compiler, MathDx and prefix throughout. Defaults match
the variables above; for custom locations, pass `--build-dir`, `--output`,
`--nvcc` and `--mathdx` to every platform command. Compatible interrupted runs
resume from their status. Changed bindings require a fresh output directory.
`run` generates a report automatically; `report` regenerates it on CPU under
`OUTPUT/<mode>/report/summary.json`. Inspect missing baselines, remaining requests, invalid
rows and floating accuracy review before drawing performance conclusions.
`--allow-baseline-gaps` is an explicit diagnostic option, not complete comparison.

## Use the recorded calibration explicitly

`run` stores confirmed research calibration under
`OUTPUT/<mode>/run/calibration/finalized`; it does not install that calibration
under `CONDA_PREFIX/share/cuButterfly/hardware`. A completed run and the installed
conda hooks therefore do not establish a calibrated default installation.
The report also does not validate profile publication.

After a successful full run, the following CPU-only shell binding selects its
capacity artifact and measured registry for a subsequent application process:

```bash
cat > "$OUTPUT/full/runtime_environment.sh" <<EOF
export CUBUTTERFLY_HARDWARE_MODEL="$OUTPUT/full/run/calibration/hardware/hardware_model.json"
export CUBUTTERFLY_REGISTRY="$OUTPUT/full/run/calibration/finalized/registry.json"
export CUBUTTERFLY_COMPILE_MODE=research
EOF
source "$OUTPUT/full/runtime_environment.sh"
```

Use this only with the matching target, source/build and runtime protocol.
This binding does not publish the fitted model and service files into the
installed profile tree, and does not certify a complete static-model fallback.
Keep the calibration manifest, original directory and referenced sidecars together.
The platform runner manages its own explicit bindings and ignores inherited
profile selections, so sourcing this file does not replace its calibration gates.

The full66 scope excludes cyclic90, public FP16/FFT-L nine requests, controlled
mechanism/ablation studies and application tracks. Migrate those separately with
their own contracts and reports. BF16 is deferred; this runbook makes no native
BF16 performance claim. See the [population contract](../../config/paper_portability/README.md).

## Preparation validation, November 7

The migration entry passes 23 CPU contract tests. The real cuFFTDx compile probe
also succeeds with the existing CUDA 12.4/MathDx toolchain at SM80; this checks
the probe itself, not SM120 compatibility. CPU descriptor expansion confirms
12/66 requests with no invalid or capability-gap rows in the current API, and
quick is an unchanged subset of full. Target manifests must still be regenerated
against the target library.

The existing A100 environment is correctly rejected for SM120: nvcc 12.4 does
not advertise that target and Torch 2.5.1 has no native SM120 kernels. The new
CUDA/MathDx/Torch stack, target build and all PRO6000 runtime measurements remain
to be checked on the destination. No target performance is inferred from these
CPU checks. Changing CUDA or provider versions is recorded in each run; it also
limits attribution of a cross-platform difference solely to GPU architecture.
