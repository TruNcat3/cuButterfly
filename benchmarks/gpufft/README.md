# GPU-FFT prepared comparator

Official source: https://github.com/Alisah-Ozcan/GPU-FFT
Pinned commit: `7f05858f7af7c0bbf55a77472426d9ac56ec6b23`; Apache-2.0.

The adapter links the unmodified upstream `src/lib/fft.cu`. Host API dispatches
its existing low-ring/staged CT/GS kernels; no method autotuning is introduced.
Supported core: packed contiguous batch-major rank-one cyclic C2C, native FP32
or FP64, powers of two from 2 through 2^28, both directions, inverse optionally
normalized. Upstream integer bound is batch*N < 2^31; N > 512 additionally
requires batch <= 65535. Allocation/resource viability remains a runtime check.
Other ranks, real transforms, embeddings and lower precision are unsupported.

Upstream `test/test_fft.cu` functions `forward_value` / `inverse_value` document
positive-exponent natural-to-bit-reversed forward and negative-exponent
bit-reversed-to-natural inverse. Standard negative-exponent forward uses
upstream INVERSE twiddle tables with FORWARD dispatch; standard inverse uses
FORWARD tables with INVERSE dispatch. Inverse scaling is passed as mod_inverse.
No FFT arithmetic is rewritten. Timed forward: copy input to prepared scratch,
GPU_FFT, bit-reverse scratch into output. Timed inverse: bit-reverse input into
scratch, GPU_FFT, copy scratch to output. Each operation uses the caller stream.
Source and destination may alias; scratch protects the original input.

`support(s)` describes this packed core, as does semantic_fft_baseline's grouped
core contract. The complete semantic runner must apply requested physical
layouts and `fft_output_scale` at its timed boundary. It must reject unsupported
core semantics before planning. Do not expose this packed core as accepting
arbitrary physical pointer layouts without that boundary.

CPU-only preparation (does not load CUDA or launch a GPU):
run from the repository root with Python and `nvcc` available on `PATH`.
Prepare the official pinned source first:

```sh
mkdir -p external
git clone --no-checkout https://github.com/Alisah-Ozcan/GPU-FFT.git external/GPU-FFT
git -C external/GPU-FFT checkout --detach 7f05858f7af7c0bbf55a77472426d9ac56ec6b23
python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from gpufft_baseline import build
build('build/gpufft', upstream='external/GPU-FFT', nvcc='nvcc', arch='80')
PY
export CUBUTTERFLY_GPUFFT_LIBRARY="$PWD/build/gpufft/libgpufft_baseline.so"
```

For an existing official checkout, pass its path as `upstream` after checking out
the pinned commit. `build` is a Python API, not a command-line subcommand; it
accepts explicit output, upstream source, nvcc path and target architecture.

Measurement integration: construct `Plan(library, canonical_semantics)` outside
capture/timing; retain its lifetime. Call
`plan.execute(source.data_ptr(), output.data_ptr(), stream=stream.cuda_stream)`
inside the timed complete semantic path. Run independent numpy FFT correctness
for both directions, precisions, multiple batches and nondefault streams before
collecting performance. Synchronize before `plan.close()`. Concurrent calls on
one plan are unsupported. Graph capture has not been validated.

CPU compilation and mathematical sign/order tests are preparation evidence only.
No GPU accuracy or timing result is claimed by these files.
