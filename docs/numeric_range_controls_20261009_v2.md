# Real FFT input-range controls and failure classification

Revision `0.9.3.4-real-range-controls.2`, October 9, 2026. This is an experiment
tooling revision. The running v0.9.3.4 library, calibration, raw inputs and
frozen campaign sources are unchanged.

## Audited failures

The completed recovery population contains 120 distinct correctness failures
(19 historical and 101 newly measured). The
[checked audit](../results/real_fft_range_repair_20261009_v2/raw_failure_audit.json)
retains their original verdicts, row/report checksums and lossless error arrays.

| Failure class | Requests | Evidence and independent control |
|---|---:|---|
| C2C FFT / FWHT FP16 | 96 | Finite reference outputs exceed FP16 range. Existing separate controls already prepared and queued. |
| R2C FFT FP16 | 12 | Finite real inputs, but reference spectral components exceed FP16 range. Real-input controls added. |
| C2R FFT FP16 | 12 | Spectrum construction produces finite values that overflow when quantized to FP16, before GPU execution. Prequantization controls added. |

For C2R, the frozen generator constructs `rfftn(sin(index * .17))` and then
quantizes the packed spectrum. At N=2^20 its real/imaginary component peaks are
approximately 379,902/163,146, above the FP16 limit 65,504. The resulting input
contains Inf and its inverse reference is nonfinite. The old range classifier
incorrectly grouped a nonfinite reference with nonfinite output whose reference
is in range. The new audit distinguishes input-nonfinite, reference-nonfinite,
reference-out-of-range and finite-reference execution failures. Historical
labels are preserved in their original reports, with the corrected explanation
stored separately. No failed raw request becomes a pass.

These failures differ from the earlier finite-input inverse-normalization bug,
which was repaired in [v0.9.3.4](numeric_plan_repair_20261009_v0934.md).

## Input policy and implementation

[semantic_real_input_profiles.py](../scripts/semantic_real_input_profiles.py)
adds a separately named FP16 R2C/C2R profile. It chooses a nonpositive power of
two from the complete input L1 bound, targeting at most one quarter of FP16's
finite range. C2R reconstructs the original spectrum **before** storage
quantization. Its packed last-axis interior frequencies have weight two, while
DC and Nyquist have weight one, including xD inputs. The quantized input is
checked against the bound; this decision never consults measured outputs.

All implementations and the CPU reference receive the final quantized input.
The profile records input hashes, scale, original nonfinite components, bounds
and nonzero components that become zero during quantization. Precision, compute
policy, dimensions, batch, normalization, order and layout remain unchanged.
The bound concerns exact intermediate magnitudes, not a floating-error guarantee.
This is a labelled experiment control, not automatic scaling of a library user's
input or an attempt to make an unrepresentable raw output acceptable.

[prepare_async_real_scaled_controls.py](../scripts/prepare_async_real_scaled_controls.py)
validates the original failure/report/library/calibration identities and writes
CPU-prepared input, oracle and layout records before GPU execution. Unsupported
semantics and requests above the complete-reference bound are rejected. The
existing C2C/FWHT controls are reused rather than duplicated. Frozen active
scripts remain intact, so ongoing timing retains its original identities.

## Completed GPU diagnostic

All 24 added controls completed on A100 PCIe 40GB
(`GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`). This is fresh selection on that
device, explicitly separate from original A100 80GB mapping replay. The
[complete report](../results/real_fft_range_repair_20261009_v2/gpu0_diagnostic/results.json)
has no nonfinite inputs, references or outputs, and no reported input-preservation,
padding, order or layout failures. Each table row includes four physical/order
variants with batch one and native FP16 computation. C2R uses normalized inverse.

| Transform | Logical shape | Scale | Relative L2 error | Newly zero input components |
|---|---|---:|---:|---:|
| R2C | 2^20 | 2^-6 | 0.1127% | 3 |
| R2C | 1024 × 1024 | 2^-6 | 0.1180% | 3 |
| R2C | 64 × 64 × 64 | 2^-4 | 0.1043% | 0 |
| C2R | 2^20 | 2^-10 | 0.0860% | 40 |
| C2R | 1024 × 1024 | 2^-11 | 0.1001% | 33 |
| C2R | 64 × 64 × 64 | 2^-9 | 0.0905% | 5 |

All 24 remain `accuracy-review-required`. L2 alone is not an acceptance rule:
near-zero reference components produce large relative errors, up to about
1,469 in this diagnostic. Complete absolute/relative distributions and arrays
are retained. Current cuFFT/VkFFT adapters lack matching FP16 real-FFT bindings;
they remain explicit baseline gaps, without an FP32 substitute or speedup claim.

The new CPU verification covers profile bounds, packed Hermitian weights,
real large-input overflow, quantization loss, error classification, evidence
integrity, cache identity and queue dependency guards (53 tests passed).

## Pending original-device verification

The 24 original A100 80GB mappings and their prepared controls are ready under
`results/real_fft_range_repair_20261009_v2/gpu1_prepared/`. The
[follow-up status](../results/real_fft_range_repair_20261009_v2/gpu1_followup/status.json)
waits only for the identified owned GPU1 queue to finish timing and its existing
96 controls, then replays these 24 mappings using the same per-device calibration.
The six-hour dependency deadline bounds this wait. Dependency failure, identity
change or a foreign GPU process stops the follow-up; it does not restart a
foreign-workload waiting budget. No current GPU experiment is interrupted.

The 80GB replay, the previously prepared 96 controls, complete performance matrix
and floating-error acceptance are still pending. The 40GB diagnostic resolves
the new real-FFT control coverage gap and supports the input-range explanation;
it does not replace any of those pending measurements.
