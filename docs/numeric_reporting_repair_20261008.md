# FP16 range diagnosis and reporting repair

Harness revision `0.9.2.1-numeric-reporting` fixes a complex-output decoding
defect. The numerical failure remains a recorded failure. The library stays
at 0.9.2 and no kernel, input, precision, normalization or tolerance changes.

## Evidence and scope

The case is FFT C2C, FP16 storage with native computation, N=1,048,576, B=1,
forward, natural input/output order, out of place and scale=1. Its semantic ID is
`eabef06a45452e6ecfb6352979c25a2fb45702b56f0e118f62a5c5fa5adb293e`.
The frozen library SHA256 is
`7267481779274368b6e34171ea92cf27bdfadbb05c3f63db02dd566d6667ba56`.
The target is A100 80GB PCIe, UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`.

The original evidence is under
`results/semantic_campaign_20261007_v092/scaling_conformance/`.
Additional evidence is under that campaign's `fp16_range_repair/`:

- `core_audit.md` traces the concrete arithmetic and storage path.
- `core_dit_diagnostic.json` simulates the original input in CPU half arithmetic.
- `gpu_unscaled_decode_fix/` preserves the unchanged-input GPU reproduction.
- `core_dit_after_decode.json` compares that reproduction with the CPU sequence.
- `run_semantic_campaign.before.py` preserves the previous runner source.

The reference has 21 complex outputs with at least one component above 65,504,
the maximum finite FP16 magnitude. Its largest real component is about 379,923.
Thus finite FP16 output is impossible for every exact output of this request.
This representability test is distinct from the round-to-nearest conversion
overflow boundary and from an accuracy threshold.

The actual kernel produces 44 nonfinite complex outputs, including 23 whose
reference components are in range. Native half intermediate arithmetic explains
this difference: the CPU sequence reproduces all 44 nonfinite positions.
Its first overflow occurs in butterfly addition/subtraction at zero-based
stage 17. Nonfinite counts after stages 17, 18 and 19 are 28, 32 and 44.
Inf propagation can contaminate later outputs that would otherwise be in range.
The simulator does not establish bitwise equality of finite values or validate
other mappings. No indexing or synchronization defect is needed to explain
this particular nonfinite mask.

## Confirmed reporting defect

The decoder used `real + 1j * imag`. An infinite imaginary component can make
that arithmetic introduce a real NaN. The repair assigns real and imaginary
components independently. It applies to FP16 and BF16 complex decoding and
preserves infinities, NaNs and signed zero.

| Result for the same input and kernel | Old decoder | Fixed decoder |
| --- | ---: | ---: |
| Nonfinite complex outputs | 44 | 44 |
| Real-component NaNs | 34 | 28 |
| Real-component infinities | 10 | 16 |
| Imaginary-component NaNs | 28 | 28 |
| Imaginary-component infinities | 6 | 6 |

The corrected component counts also match the CPU simulation. The new range
report separates reference range violations, component NaNs and signed
infinities, and actual nonfinite values at reference positions within range.
It does not make this case pass or infer errors for unsampled configurations.

The current FP32-computation alternative still narrows intermediate values to
half. Selecting it alone is not evidence of an FP32 intermediate-storage path.
A wider or explicitly range-managed implementation would need its own contract
and validation; silently changing this native FP16 case is not a repair.

## Validation and continuation

Affected checks passed: 25 runner tests, 3 half-sequence diagnostic tests,
11 continuation tests and 4 supervisor tests. The GPU reproduction uses the
same descriptor, generated input, mapping, library and calibration binding.
It remains `correctness-failure`; no performance result is inferred from it.

The continuation controller preserves original evidence and executes remaining
requests in fresh, separately identified shards. Failed cases remain explicit
quarantines. Runtime or infrastructure errors stop rather than trigger retries.
The separate comparison phase excludes quarantined requests with a recorded
exclusion list. Floats still require user review of errors before thresholds.

The user approved separately labelled, power-of-two-scaled input controls, with
the same final quantized input for every implementation. The coefficient is
chosen before execution from the maximum per-transform input L1 bound and
fixed range headroom. Both that coefficient and quantization losses are
recorded. The reference is recomputed from the final quantized input, not merely
rescaled from an earlier reference. These controls remain separate from the
original pressure test and cannot replace a failed row. A prior CPU-only 1/16
amplitude control remains diagnostic evidence; it is not the new bound-based
input policy.

## Approved scaled-input GPU control

`scripts/semantic_input_profiles.py` separates the input profile from operator
semantics. `scripts/run_fft_range_control.py` verifies the original row and
measurement identity, preserves the descriptor, and replays its exact mapping.
It does not search or time kernels. Seven input-profile and ten control-runner
CPU tests passed, including source integrity, CPU-only preparation, refusing
output overwrite, and failure propagation. The supported control scope is
explicit: FP16 C2C, power-of-two exact lengths, with absolute output scale at
most one. Other domains require their own justified range bound.

The GPU control completed at 14:50 +08 under
`results/semantic_campaign_20261007_v092/fp16_range_repair/gpu_scaled_l1_control/`.
It used the same frozen kernel, mapping, GPU and precision as the failed row.
The original input L1 bound was 1,394,614.5292. The fixed target is one quarter
of the maximum finite FP16 magnitude, 16,376, giving exponent -7 and factor
1/128. The final quantized input bound is 10,895.4260. Nine initially nonzero
components became zero during quantization; no complete complex element newly
became zero. `source.npz` retains the exact executed input, storage bytes and
reference recomputed from that input.

| Diagnostic | Unscaled pressure test | Separate scaled control |
| --- | ---: | ---: |
| Reference outputs exceeding FP16 range | 21 | 0 |
| Nonfinite actual complex outputs | 44 | 0 |
| Relative L2 error | Nonfinite | 0.00133062 |
| Absolute error median / p99 | See retained nonfinite artifact | 0.00453787 / 0.03253057 |
| Pointwise relative error median / p99 | See retained nonfinite artifact | 0.293740 / 3.815607 |

Input preservation and padding checks passed. Finite output is not an accuracy
pass: under the old diagnostic tolerances 570,442 of 1,048,576 outputs differ
from the reference beyond that historical bound. Those tolerances are not the
user's acceptance criterion. The full distribution remains
`accuracy-review-required`, with no tolerance relaxation. The global norm error
must not be substituted for the pointwise tail. Current FP16 cuFFT/VkFFT
bindings are missing, so this control provides no vendor-relative accuracy or
speedup result and uses no higher-precision substitute.

The scheduler yielded between GPU shards for approximately 11 seconds and
resumed in a `finally` handler. Its handoff record is
`results/semantic_campaign_20261008_v0921/numeric_control_handoff.json`.
The original unscaled campaign and failed records are still intact; the new
input-profile ID is distinct from the original semantic case ID.
