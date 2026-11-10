# GPU-NTT crypto adapter

`gpuntt_crypto_bench` is a bounded comparison adapter for the clean official
GPU-NTT checkout pinned at `d03c5eaeadaa780d153496afcb3b6a9b79a13a63`. It
builds only GPU-NTT `common`, `nttparameters`, and `ntt_merge` sources. The
adapter uses cuNTT's deterministic primitive-root finder and the existing
crypto benchmark's input/oracle contract.

The benchmark accepts natural input and returns natural output. GPU-NTT Merge's
native forward output is bit-reversed, so the required timed CUDA permutation
is included. Inverse inputs are naturalized to GPU-NTT's bit-reversed input
contract inside the timed region. Negacyclic mode uses native `X_N_plus` when
the selected 2N root squares exactly to cuNTT's N root. Coset mode uses timed
CUDA twists. RNS channels are submitted sequentially on one stream; this is a
single-word composition comparison and does not claim native fused RNS.

GPU-NTT's arithmetic is used without modification. Data32 requires
`2 < p < 2^30`, Data64 requires `2 < p < 2^61`; every modulus must be prime
and support the requested transform order. The pinned Data64 Barrett path is
not exact for 62-bit moduli. Merge dispatch supports
`1 <= logN <= 28` and `1 <= batch <= 65535`. Invalid or unsupported cases
return nonzero (and emit `{"correct":false,...}` with `--json`).

## Build

```bash
cmake -S benchmarks/gpuntt_crypto \
  -B build-gpuntt-semantic/vendor \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPUNTT_SOURCE_DIR="$PWD/external/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpuntt-semantic/vendor --parallel 4
```

The executable is
`build-gpuntt-semantic/vendor/gpuntt_crypto_bench`.

## Prepared Semantic Bridge

Keep the official checkout and build artifacts on persistent storage:

```bash
git clone https://github.com/Alisah-Ozcan/GPU-NTT.git \
  external/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63
git -C external/GPU-NTT-d03c5eaeadaa780d153496afcb3b6a9b79a13a63 \
  checkout --detach d03c5eaeadaa780d153496afcb3b6a9b79a13a63
PYTHON=python \
  bash benchmarks/gpuntt_crypto/build_semantic_bridge.sh
```

The script builds only the existing CMake `gpuntt_crypto_vendor` target and
the bridge. Compilation and inventory need the CUDA toolkit but no GPU or
CUDA context. Defaults are `external/GPU-NTT-<revision>` for source,
`build-gpuntt-semantic/vendor` for the archive, and
`build-gpuntt-semantic/libgpuntt_semantic_bridge_guarded_20261102.so` for the bridge.
Override `GPUNTT_SOURCE_DIR`, `GPUNTT_BUILD_DIR`, `GPUNTT_OUTPUT`,
`CUNTT_LIBRARY`, `CUBUTTERFLY_BUILD_DIR`, `NVCC`, `PYTHON`, or `BUILD_JOBS`
as needed. Source must be a clean checkout at the pinned revision. The default
root helper is `build-e13-cufftdx/libcuntt.so`.

An existing output or adjacent `.manifest.json` is rejected before building.
Each fresh output has its own build command, hashes, and adjacent manifest.
Inventory verifies that manifest against the selected binary, vendor archive,
root helper, and recorded source files before loading the bridge. Preserve all
recorded paths; moving, replacing, or removing evidence requires a fresh build
at a fresh output path. A library without this sidecar is a
`missing-external-binding`, as are missing or changed build inputs. Semantic
domain rejections remain `unsupported-semantic`, even without a binding.

```bash
GPUNTT_TEST_LIBRARY="$PWD/build-gpuntt-semantic/libgpuntt_semantic_bridge_guarded_20261102.so" \
  python -m pytest -q \
  tests/test_semantic_gpuntt_baseline.py
```

This CPU-only check validates prime/root support without executing GPU work.
Prepared execution still times the full core including order conversion and
inverse normalization; setup and allocations stay outside the timed call.

Example:

```bash
gpuntt_crypto_bench --log-n 12 --batch 1 --word-bits 64 \
  --moduli 576460756061519873 --mode cyclic --direction forward \
  --warmup 5 --repeat 20 --seed 1 --input-pattern random --json
```

JSON includes `baseline`, `source`, `method`, exact per-channel `root`/`psi`
evidence, `correct`, `verified_batches`, `verified_channels`, and the shared
`kernel_ms`/`normalization`/`layout`/`rns_stream_order` contract. `setup_ms`
is informational only. CUDA event timing excludes compilation, setup, and CPU
validation, while including the required order conversion, twists, and inverse
normalization.
