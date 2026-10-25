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
`2 < p < 2^30`, Data64 requires `2 < p < 2^62`; every modulus must be prime
and support the requested transform order. Merge dispatch supports
`1 <= logN <= 28` and `1 <= batch <= 65535`. Invalid or unsupported cases
return nonzero (and emit `{"correct":false,...}` with `--json`).

## Build

```bash
cmake -S benchmarks/gpuntt_crypto \
  -B /tmp/cubutterfly-gpuntt-crypto-build-20260929 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPUNTT_SOURCE_DIR=/tmp/cubutterfly-gpuntt-crypto-20260929 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build /tmp/cubutterfly-gpuntt-crypto-build-20260929 --parallel 4
```

The executable is
`/tmp/cubutterfly-gpuntt-crypto-build-20260929/gpuntt_crypto_bench`.

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
