# FFT-L CPU Preparation

The bounded configuration is
[`config/fft_l_quick_semantic.json`](../config/fft_l_quick_semantic.json).
It requests 32 C2C cases: FP16/BF16 input storage, native or explicit FP32
arithmetic, forward or normalized inverse, shapes `[4096]` and `[64,64]`,
batch 1/4, packed natural order, out of place. Public descriptor validation and
capabilities are checked against the selected library. This preparation does
not establish GPU executability, fair baseline coverage, or accuracy acceptance.

```bash
python -m benchmark.prepare_fft_l_quick \
  --library build-e13-cufftdx/libcuntt.so \
  --output /tmp/fft-l-quick-new
```

The output directory must be fresh. `raw_manifest.json` uses the existing
semantic campaign schema; `preparation.json` records capabilities, exact vendor
support checks, provenance, input cohorts, and source-artifact checksums. NPZ
files retain the complete quantized source, encoded low-precision storage, and
full NumPy complex128 diagnostic reference. The reference is higher precision;
it is neither a same-precision numerical oracle nor a low-precision CPU
performance baseline. No FP32 GPU request substitutes for a low-precision one.

`semantic_cpu_pipeline.prepare_case` and `run_semantic_campaign.reference`
provide the reused CPU path. Future execution must retain complete actual,
reference, absolute/relative error arrays and distributions using
`--float-verdict error-report`; no error threshold is accepted by this setup.
Memory-boundary checks and nonfinite failures remain failures independently
of pointwise error review.

| Implementation | Audited Local Path | FFT-L Boundary |
| --- | --- | --- |
| cuButterfly | Public descriptor plus semantic runner | FP16/BF16 native and FP32 arithmetic; bounded cases must still be executed and reviewed |
| cuFFT | `run_semantic_campaign.baseline_contract`, `diagnose_xd_fft.NativeCufft` | Exact FP16/BF16 adapter missing; current bindings use C2C/Z2Z FP32/FP64 |
| VkFFT | Same contract gate, `vkfft_xd.NativeVkfft` | Exact FP16/BF16 adapter missing; current bridge accepts complex64/complex128 |
| FFTW CPU | `cpu_fftw_baseline` | Existing FP32/FP64 contract does not supply a matching FFT-L baseline |
| tcFFT | `scripts/tcfft_baseline.py`, `build/tcfft_sm80/libtcfft_baseline.so` | CPU-compiled SM80 adapter only; GPU and Graph validation pending; no supported length intersects this quick population |

The vendor rank-above-three axis adapter also accepts only FP32/FP64 storage.
Upstream provider half/BF16 capabilities do not fill these local binding gaps.
The prepared tcFFT adapter pins GPL-3.0 upstream commit
`3e87a7083577741815e21b2a3c2bed22520786ce` and CUDA 12.4 compilation. It accepts
only rank-1 forward unnormalized FP16 storage/multiply/accumulation requests
passing its explicit initialized-length, batch-divisibility and grid checks.
Its compiled shared object is preparation evidence, not an executed baseline.
FP32 arithmetic with low storage still narrows intermediate results to low
storage in the current local path. It remains FFT-L and must not be described
as FP32 intermediate storage.

The existing `semantic_input_profiles` module prepares 16 separately identified
FP16 scaled controls from the complete input L1 bound and fixed quarter-range
headroom. At these small sizes the chosen factor can be 1; record it without
claiming a range improvement. The reference is recomputed from the final
quantized input. Raw requests remain intact. BF16 scaled-input support is a
recorded gap. Ordinary semantic execution regenerates raw inputs and cannot
consume these scaled artifacts as if they were raw. The existing
`run_fft_range_control.py --prepare-only` path instead requires a compatible
measured source row, verified identity, and exact mapping for scaled replay.

Historical range evidence is summarized in
`numeric_reporting_repair_20261008.md`
and `numeric_range_controls_20261009_v2.md`.
Those records retain FP16 failures and separate scaled controls; they do not
establish current vendor-relative performance or an accepted accuracy bound.
No historical raw records or frozen scope are changed by this preparation.
