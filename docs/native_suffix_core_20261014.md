# Explicit native suffix core, 2026-10-14

Status after `0.9.3.11-provider.3`: this is a historical implementation and
measurement record. The native suffix remains available by explicit mapping but
is no longer automatically enumerated. The missing vendor EPT32 implementation
did not make the FFT request unsupported; mature cuFFTDx configurations already
execute it. Further work prioritizes those cores and their supported parameters.
See [the correction and current search policy](provider_reuse_20261014.md).

Revision `0.9.3.11-provider.2` adds an executable native suffix to the existing
online register FFT organization. The previous provider check correctly rejected
two missing cuFFTDx FP64/EPT32 implementations; this revision supplies a separate
core with those physical geometries. It does not modify the vendor database or
claim to implement the vendor's missing algorithms.

## Public contract and implementation

An online FFT mapping with `fft_core="register-tile"` accepts
`suffix_core="native"` or `suffix_core="cufftdx-block"`. Omission retains the
historical cuFFTDx suffix. The native choice is available through public mapping
JSON, candidate enumeration, compilation preparation, JIT, plan replay and stage
calibration. A provider rejection does not silently switch the requested core.
Unrelated operator/backends reject a non-default suffix choice.

The native core keeps the requested precision, EPT, columns and block dimensions
`{local_N/EPT, columns}`. It composes the existing register FFT codelet with a
diagonal phase and a cooperative row FFT. Warp exchanges handle intra-warp
stages; shared memory handles cross-warp stages and output ownership changes.
A bijective XOR write offset distributes shared accesses across banks. The
existing grouped suffix performs whole/chunked writeback and inverse scaling.
Both directions produce natural cyclic ownership without an extra global
conversion kernel. EPT1, one-row and sub-warp row groups have explicit handling.

This implementation reserves `local_N * columns * sizeof(complex)` shared bytes.
The writeback chunk does **not** reduce that core reservation. Native candidates
are filtered against the actual platform capacity, and compiled resources are
checked before execution. For FP64 local N4096 and two columns the core needs
128 KiB. Four columns require 256 KiB and are therefore not a valid instance of
this particular implementation on the tested A100. Other cores remain separate
candidates. A future streamed internal exchange is a distinct implementation
improvement, not an already supported capability.

Physical segment/group records identify the native suffix as `register-tile`.
Conflicting explicit group records are rejected. Module/cache identities include
the core selection and native source. Stage service keys distinguish the changed
suffix while allowing reuse of the unchanged prefix's service curve. Candidate
projections preserve the full native shared reservation, including chunked stores.
Installation copies the new template with the existing JIT templates.

## Bounded validation

All preparation runs on CPU before acquiring a GPU: twelve standalone modules
(ten native and two cuFFTDx controls), inputs and same-precision SciPy references.
The GPU run is separate and guarded against foreign compute processes. Evidence
is under `results/provider_contract_20261014/native_suffix_checks`.

- 209 affected CPU tests passed, including real C++ mapping roundtrip/candidates,
  replay conflicts, capability checks, module identity and stage service keys.
- Twelve compile-only native core instances passed NVCC on SM80.
- The final twelve standalone modules completed 432 numerical/structural checks on
  A100 40GB UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`. They cover FP32/FP64,
  forward, normalized/unnormalized inverse, stride 1/2, padded batch distance,
  whole/group launches, EPT/row/column boundaries and exchange chunks.
- Every output was finite; all padding sentinels and exact zero/origin-impulse
  checks passed. Maximum relative L2 error against the same-precision CPU
  reference was 2.16e-7 for FP32 and 4.50e-16 for FP64. The matched cuFFT maxima
  were 2.70e-7 and 4.92e-16. These are observed errors, not new acceptance thresholds.
- Four public Python/C API replay cases execute both missing FP64 geometries in
  both directions and retain EPT32, threads and the native physical group core.
  The prepared module cache is unchanged. Explicitly requesting the missing
  cuFFTDx core still returns `NOT_SUPPORTED` rather than changing the core.
- Four actual stage-probe descriptions resolve both physical groups as independently
  executable and report full native shared storage. These resource descriptions
  do not constitute calibrated latency curves or registry publication.
- The isolated shared library, benchmark and stage probe built successfully.

Reproduce preparation and the guarded GPU execution separately:

```bash
python tests/check_native_suffix.py --prepare results/native_suffix_check
python tests/check_native_suffix_public.py --prepared results/native_suffix_check --prepare-only
CUDA_VISIBLE_DEVICES=<exclusive-gpu-uuid> \
  python tests/check_native_suffix.py --run results/native_suffix_check
CUDA_VISIBLE_DEVICES=<exclusive-gpu-uuid> CUBUTTERFLY_COMPILE_MODE=research \
  CUBUTTERFLY_JIT_CACHE="$PWD/results/native_suffix_check/public_modules" \
  python tests/check_native_suffix_public.py --prepared results/native_suffix_check \
  --library build-provider/libcuntt.so
```

Preparation uses the two preserved original mapping fixtures in
`tests/data/native_suffix_provider_gaps.json`. The GPU test's `guard.json` records the actual exclusive
run. A public API replay check is provided in `tests/check_native_suffix_public.py`.

## Performance boundary

The following are complete two-group FFT CUDA Graph timings for FP64 N65536,
B64, two suffix columns on the same A100 40GB. Compilation, planning, CPU reference
generation and validation are excluded. Five timing trials each replay twenty
graphs. The timing input is identical complex ones for every core. These explicit
mapping controls are not the automatic library comparison or a fixed-core
organization ablation; the native and imported rows change core, EPT and threads.

| Local suffix N | Native EPT32, us | cuFFTDx EPT16, us | cuFFT full FFT, us |
| ---: | ---: | ---: | ---: |
| 1024 | 582.40 | 405.76 | 235.62 |
| 4096 | 952.63 | 440.58 | 235.21 |

The native core supplies an additional exact physical mapping but does not win
these timings. This was not a missing FFT semantic capability. The following
revision removes it from automatic enumeration and retains it for explicit
experiments. Compiled FP64 suffixes report 255 registers per thread
and 432--736 local bytes per thread. Removing a duplicate source-level value
array did not change these compiler allocations. These static resources identify
remaining core pressure but do not measure dynamic spill traffic. The earlier
80GB trial and source are preserved under `before_register_alias`; timings from
the two cards are not combined or used to claim an optimization speedup.

The original experiment continues in its frozen source
tree and has not adopted this revision or these measurements as calibration.
The isolated build disables unrelated xD cuFFTDx templates to bound validation
build time; this experiment does not validate xD performance or constitute an
installation into the active shared conda environment.
